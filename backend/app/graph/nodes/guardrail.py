"""Policy Enforcer: sanitize all inputs against prompt-injection attacks.

All Jira-fetched text is treated as untrusted and delimiter-wrapped before any
LLM ever sees it as instructions.  Detected patterns are stripped (replaced with
[REDACTED]) and listed in the emitted decision event.
"""
from __future__ import annotations

import logging
import re

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

# (label, compiled pattern) — checked against every piece of text.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("role-override",
     re.compile(r'\b(ignore|disregard)\s+(previous|prior|above)\s+'
                r'(instructions?|prompts?|context)', re.I)),
    ("system-tag",
     re.compile(r'<\s*(system|assistant|human)\s*>', re.I)),
    ("delimiter-escape",
     re.compile(r'(```|\[INST\]|\[/INST\]|###\s*System|###\s*Human)', re.I)),
    ("persona-hijack",
     re.compile(r'(you are now|act as|pretend (you are|to be)|'
                r'your new (role|persona|instructions?))', re.I)),
    ("jailbreak-marker",
     re.compile(r'\bDAN\b|do anything now|jailbreak', re.I)),
]


def _scrub(text: str) -> tuple[str, list[str]]:
    """Strip injection patterns; return (scrubbed_text, list_of_flags)."""
    flags: list[str] = []
    out = text
    for name, pat in _INJECTION_PATTERNS:
        if pat.search(out):
            flags.append(name)
            out = pat.sub("[REDACTED]", out)
    return out, flags


def _wrap(label: str, text: str) -> str:
    return f"<{label}>\n{text}\n</{label}>"


async def guardrail_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Policy Enforcer"
    raw = state["raw_input"]
    jira_ctx = state.get("jira_context") or {}

    await emitter.emit(run_id, agent, 1, "thought",
                       "Scanning inputs for prompt-injection patterns…")

    all_flags: list[str] = []

    cleaned_raw, raw_flags = _scrub(raw)
    all_flags.extend(raw_flags)
    parts: list[str] = [_wrap("story", cleaned_raw)]

    if jira_ctx:
        story_text = jira_ctx.get("story", "")
        cleaned_story, story_flags = _scrub(story_text)
        all_flags.extend(f"jira:{f}" for f in story_flags)
        parts.append(_wrap("jira_context", cleaned_story))

        ac_list = jira_ctx.get("acceptance_criteria") or []
        if ac_list:
            cleaned_ac, ac_flags = _scrub("\n".join(ac_list))
            all_flags.extend(f"jira_ac:{f}" for f in ac_flags)
            parts.append(_wrap("acceptance_criteria", cleaned_ac))

    sanitized = "\n\n".join(parts)

    if all_flags:
        msg = f"Injection pattern(s) detected and redacted: {', '.join(all_flags)}."
        logger.warning("Guardrail flagged %s in run %s", all_flags, run_id)
    else:
        msg = "No injection patterns detected — input is clean."

    await emitter.emit(run_id, agent, 1, "decision", msg,
                       data={"flags": all_flags})

    logger.info("Guardrail: sanitized_story length=%d", len(sanitized))
    return {"sanitized_story": sanitized}
