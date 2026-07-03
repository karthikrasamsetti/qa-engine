"""Structured audit log for the QA Engine.

Every TrajectoryEvent emitted via emitter.set_audit() is appended as one JSONL
line to:
    backend/data/audit/<run_id>.jsonl

Fields per entry:
  run_id       — workflow run identifier
  node         — agent name (e.g. "INVEST Reviewer")
  phase        — phase number (1..5)
  event_type   — TrajectoryEvent type (thought|action|tool_call|...)
  tool         — inferred tool category (e.g. "llm:reasoning", "sandbox:run")
  args_hash    — SHA-256 hex of JSON-serialised data (no raw values, avoids logging secrets)
  ts           — ISO-8601 UTC timestamp
  outcome      — "success" | "error" | "skipped" | "violation"
  latency_ms   — duration in milliseconds (0 when not measured)
  message      — first 200 chars of the human-readable message

Least-privilege registry
────────────────────────
NODE_TOOL_REGISTRY maps each agent name (lowercase) to the set of tool
categories it is permitted to invoke. Violations are logged as WARNING and
the audit outcome is set to "violation" — the run continues normally
(soft enforcement via observability, not hard blocking).

Graceful degradation: any I/O failure is caught and logged; the run is never
interrupted.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_AUDIT_DIR = Path(__file__).parent.parent.parent / "data" / "audit"

# ---------------------------------------------------------------------------
# Least-privilege registry
# ---------------------------------------------------------------------------
# Maps agent name (lowercase as emitted by each node) → allowed tool categories.
#
# Categories:
#   llm:reasoning     — calls to the reasoning LLM tier (claude-sonnet, gpt-4o…)
#   llm:fast          — calls to the fast LLM tier (claude-haiku, gpt-4o-mini…)
#   browser:find      — BrowserSession.find_locators() DOM scan
#   sandbox:run       — run_script() Docker execution
#   jira:fetch        — MCP Jira ticket fetch
#   vector_store:read — retrieve_similar()
#   vector_store:write — persist_run()
#   hitl:interrupt    — LangGraph interrupt() for human-in-the-loop pauses

NODE_TOOL_REGISTRY: dict[str, frozenset[str]] = {
    "context agent":            frozenset({"jira:fetch"}),
    "policy enforcer":          frozenset(),
    "invest reviewer":          frozenset({"llm:reasoning", "hitl:interrupt"}),
    "requirements analyst":     frozenset({"llm:reasoning", "vector_store:read"}),
    "ui mapper":                frozenset({"llm:fast", "browser:find"}),
    "scaffolder":               frozenset({"llm:reasoning"}),
    "test engineer (critic)":   frozenset({"llm:reasoning"}),
    "execution agent":          frozenset({"sandbox:run"}),
    "self-heal":                frozenset({"llm:fast", "browser:find", "hitl:interrupt"}),
    "synthesis agent":          frozenset({"llm:reasoning", "vector_store:write"}),
    # The LLM client emits cost events under the "System" agent.
    "system":                   frozenset({"llm:reasoning", "llm:fast"}),
}


# ---------------------------------------------------------------------------
# Tool inference
# ---------------------------------------------------------------------------

def _infer_tool(event: Any) -> str:
    """Guess the tool category from event type, agent, message, and data.

    Returns an empty string when no specific tool can be identified.
    """
    agent = event.agent.lower()
    etype = event.type
    msg = event.message.lower()
    data = event.data

    # LLM cost events: emitted by "System" agent as tool_result
    if agent == "system" and etype == "tool_result":
        model = data.get("model", "")
        if any(x in model for x in ("sonnet", "gpt-4o", "o3", "opus", "o4")):
            return "llm:reasoning"
        if any(x in model for x in ("haiku", "gpt-4o-mini", "gpt-3")):
            return "llm:fast"
        # Unknown model — fall back by tier guess
        return "llm:reasoning"

    # Docker sandbox
    if etype in ("tool_call", "tool_result") and (
        "sandbox" in msg or "docker" in msg or "exit_code" in data
        or agent == "execution agent"
    ):
        return "sandbox:run"

    # Jira fetch
    if etype in ("tool_call", "tool_result") and (
        "jira" in msg or "jira_id" in data or agent == "context agent"
    ):
        return "jira:fetch"

    # Browser / DOM scan
    if etype in ("tool_call", "tool_result") and (
        "scanning dom" in msg or "locator" in msg or "find_locators" in msg
        or (agent in ("ui mapper", "self-heal") and etype == "tool_call")
    ):
        return "browser:find"

    # Vector store
    if "cache_hit" in data and data.get("cache_hit"):
        return "vector_store:read"
    if "persist" in msg and "vector" in msg:
        return "vector_store:write"

    # HITL interrupt
    if etype == "hitl_request":
        return "hitl:interrupt"

    return ""


def _args_hash(data: dict[str, Any]) -> str:
    """SHA-256 (first 16 hex chars) of the sorted JSON-serialised data dict."""
    try:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except Exception:
        return "hash-error"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLog:
    """JSONL writer for run audit trails.

    One file per run: ``<audit_dir>/<run_id>.jsonl``.
    Every call to write_event() appends exactly one line.
    All errors are caught and logged — never propagate.
    """

    def __init__(self, audit_dir: Path | None = None) -> None:
        self._dir = audit_dir or _DEFAULT_AUDIT_DIR

    def _path_for(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.jsonl"

    def write_event(
        self,
        event: Any,
        outcome: str = "success",
        latency_ms: float = 0.0,
    ) -> None:
        """Append one JSONL audit entry derived from a TrajectoryEvent."""
        try:
            agent_lower = event.agent.lower()
            tool = _infer_tool(event)

            # Least-privilege check — log violation, never block.
            if tool:
                allowed = NODE_TOOL_REGISTRY.get(agent_lower, frozenset())
                if allowed and tool not in allowed:
                    logger.warning(
                        "AUDIT: least-privilege violation — agent %r invoked tool %r "
                        "(allowed: %s) in run %s",
                        event.agent, tool,
                        ", ".join(sorted(allowed)),
                        event.run_id,
                    )
                    outcome = "violation"

            entry: dict[str, Any] = {
                "run_id":     event.run_id,
                "ts":         datetime.fromtimestamp(event.ts, tz=timezone.utc).isoformat(),
                "node":       event.agent,
                "phase":      event.phase,
                "event_type": event.type,
                "tool":       tool,
                "args_hash":  _args_hash(event.data),
                "outcome":    outcome,
                "latency_ms": round(latency_ms, 2),
                "message":    event.message[:200],
            }

            self._dir.mkdir(parents=True, exist_ok=True)
            with self._path_for(event.run_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as exc:
            logger.error("AuditLog.write_event failed — %s", exc)

    def read_run(self, run_id: str) -> list[dict]:
        """Read all audit entries for *run_id*. Returns [] when file missing."""
        try:
            path = self._path_for(run_id)
            if not path.exists():
                return []
            entries: list[dict] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
        except Exception as exc:
            logger.error("AuditLog.read_run failed for %s — %s", run_id, exc)
            return []


# Module-level singleton — wire into main.py with emitter.set_audit(audit_log.write_event)
audit_log = AuditLog()
