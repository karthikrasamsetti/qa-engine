"""Stage H tests: audit log + least-privilege registry + eval judge.

Four scenarios:
  1. test_audit_log_writes_jsonl
     — write an event, read back JSONL; all required fields present
  2. test_audit_captures_invest_events_via_hook
     — emitter.set_audit() captures invest_node trajectory events
  3. test_node_tool_registry_covers_all_graph_nodes
     — NODE_TOOL_REGISTRY has an entry for every active graph node
  4. test_judge_run_evalset_produces_scoreboard
     — run_evalset() returns a scoreboard with total/correct/accuracy/stories
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.streaming.events import emitter, TrajectoryEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INVEST_PASS_JSON = json.dumps({
    "passed": True,
    "scores": {k: 8 for k in ["independent", "negotiable", "valuable", "estimable", "small", "testable"]},
    "gaps": [],
    "overall_assessment": "Good story — all INVEST criteria met.",
})

_INVEST_FAIL_JSON = json.dumps({
    "passed": False,
    "scores": {k: 2 for k in ["independent", "negotiable", "valuable", "estimable", "small", "testable"]},
    "gaps": ["Too vague", "No acceptance criteria", "Not testable"],
    "overall_assessment": "Vague story — fails INVEST on all criteria.",
})


def _llm_mock_factory(*texts: str):
    """Return an async LLM stub that serves texts in sequence; repeats the last."""
    from app.llm.client import LLMResponse
    seq = list(texts)
    state = {"i": 0}

    async def _mock(messages, model_tier, **kw):
        i = state["i"]
        text = seq[i] if i < len(seq) else seq[-1]
        state["i"] = i + 1
        return LLMResponse(text=text, input_tokens=5, output_tokens=5, model="mock", cost_usd=0.0)

    return _mock


# ---------------------------------------------------------------------------
# 1. AuditLog writes & reads JSONL correctly
# ---------------------------------------------------------------------------

def test_audit_log_writes_jsonl(tmp_path):
    """write_event() appends a JSONL line; read_run() returns it with all fields."""
    from app.observability.audit import AuditLog

    audit = AuditLog(audit_dir=tmp_path)
    event = TrajectoryEvent(
        run_id="test-audit-001",
        agent="INVEST Reviewer",
        phase=1,
        type="decision",
        message="INVEST verdict: PASS — independent=8, negotiable=8",
        data={"verdict": {"passed": True}, "run_total_usd": 0.003},
    )
    audit.write_event(event, outcome="success", latency_ms=123.4)

    entries = audit.read_run("test-audit-001")
    assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
    e = entries[0]

    # Required fields
    assert e["run_id"] == "test-audit-001"
    assert e["node"] == "INVEST Reviewer"
    assert e["phase"] == 1
    assert e["event_type"] == "decision"
    assert e["outcome"] == "success"
    assert e["latency_ms"] == 123.4
    assert "ts" in e, "entry must have an ISO timestamp"
    assert "args_hash" in e, "entry must have SHA hash of data (no raw secrets)"
    assert "tool" in e

    # args_hash must NOT contain the raw verdict data
    assert "passed" not in e["args_hash"], "args_hash should be a hash, not raw data"


# ---------------------------------------------------------------------------
# 2. Emitter hook captures invest_node events into the audit file
# ---------------------------------------------------------------------------

async def test_audit_captures_invest_events_via_hook(tmp_path, monkeypatch):
    """set_audit() hook causes invest_node trajectory events to appear in JSONL."""
    from app.llm import client as llm_mod
    from app.observability.audit import AuditLog
    from app.graph.nodes.invest import invest_node

    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_mock_factory(_INVEST_PASS_JSON))

    audit = AuditLog(audit_dir=tmp_path)
    run_id = "test-audit-invest-hook"
    state: dict = {
        "run_id": run_id,
        "raw_input": "As a user I want to log in with email and password so I can access the dashboard",
        "sanitized_story": "As a user I want to log in",
    }

    emitter.set_audit(audit.write_event)
    try:
        await invest_node(state)
    finally:
        emitter.set_audit(None)
        await emitter.close(run_id)

    entries = audit.read_run(run_id)
    assert entries, "Audit log must contain entries after invest_node runs"

    nodes = [e["node"] for e in entries]
    assert "INVEST Reviewer" in nodes, (
        f"Expected 'INVEST Reviewer' in audit nodes, got: {nodes}"
    )
    event_types = [e["event_type"] for e in entries]
    assert "decision" in event_types, (
        f"Expected at least one 'decision' event, got: {event_types}"
    )
    # All entries must have mandatory structural fields
    for e in entries:
        assert "run_id" in e
        assert "ts" in e
        assert "args_hash" in e
        assert "outcome" in e


# ---------------------------------------------------------------------------
# 3. NODE_TOOL_REGISTRY covers every active graph node
# ---------------------------------------------------------------------------

def test_node_tool_registry_covers_all_graph_nodes():
    """Every node active in the compiled graph must appear in NODE_TOOL_REGISTRY."""
    from app.observability.audit import NODE_TOOL_REGISTRY

    # Agent names as they appear in emitter.emit() calls (lowercase, for registry lookup)
    required = {
        "context agent",
        "policy enforcer",
        "invest reviewer",
        "requirements analyst",
        "ui mapper",
        "scaffolder",
        "test engineer (critic)",
        "execution agent",
        "self-heal",
        "synthesis agent",
    }
    registry_keys = set(NODE_TOOL_REGISTRY.keys())
    missing = required - registry_keys
    assert not missing, (
        f"These node names are missing from NODE_TOOL_REGISTRY: {sorted(missing)}\n"
        f"Registry has: {sorted(registry_keys)}"
    )


# ---------------------------------------------------------------------------
# 4. Judge run_evalset() produces a complete scoreboard
# ---------------------------------------------------------------------------

async def test_judge_run_evalset_produces_scoreboard(tmp_path, monkeypatch):
    """run_evalset() must return a scoreboard with total/correct/accuracy/stories."""
    from app.llm import client as llm_mod

    # Build a minimal evalset in tmp_path so the test is self-contained.
    evalset_path = tmp_path / "evalset.jsonl"
    test_stories = [
        {
            "id": "t1",
            "story": (
                "As a registered user I want to log in with my email and password "
                "so that I can access the dashboard."
            ),
            "expected_invest_pass": True,
            "expected_hitl": False,
            "rationale": "Specific, testable login story — INVEST pass.",
        },
        {
            "id": "t2",
            "story": "Improve the user experience of the application.",
            "expected_invest_pass": False,
            "expected_hitl": True,
            "rationale": "Vague, non-testable request — INVEST fail, HITL triggered.",
        },
    ]
    with evalset_path.open("w", encoding="utf-8") as f:
        for s in test_stories:
            f.write(json.dumps(s) + "\n")

    # Serve INVEST pass for t1, INVEST fail for t2.
    monkeypatch.setattr(
        llm_mod.llm_client, "complete",
        _llm_mock_factory(_INVEST_PASS_JSON, _INVEST_FAIL_JSON),
    )

    from evals.judge import run_evalset
    scoreboard = await run_evalset(evalset_path=evalset_path, use_judge=False)

    # Top-level fields
    assert "total" in scoreboard, f"scoreboard missing 'total': {scoreboard}"
    assert "correct" in scoreboard
    assert "accuracy" in scoreboard
    assert "stories" in scoreboard

    assert scoreboard["total"] == 2
    assert 0.0 <= scoreboard["accuracy"] <= 1.0

    # Per-story fields
    assert len(scoreboard["stories"]) == 2
    for story_result in scoreboard["stories"]:
        assert "id" in story_result
        assert "actual_invest_pass" in story_result
        assert "actual_hitl" in story_result
        assert "correct" in story_result
        assert "latency_ms" in story_result

    # t1 should be correct (INVEST pass, no HITL)
    t1 = next(s for s in scoreboard["stories"] if s["id"] == "t1")
    assert t1["actual_invest_pass"] is True
    assert t1["actual_hitl"] is False
    assert t1["correct"] is True

    # t2 should also be correct (INVEST fail → HITL triggered)
    t2 = next(s for s in scoreboard["stories"] if s["id"] == "t2")
    assert t2["actual_invest_pass"] is False
    assert t2["actual_hitl"] is True
    assert t2["correct"] is True

    # Both correct → accuracy = 1.0
    assert scoreboard["correct"] == 2
    assert scoreboard["accuracy"] == pytest.approx(1.0)
