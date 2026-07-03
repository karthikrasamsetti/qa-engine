"""Critic: review the generated script and gate the reflection loop.

Emits one decision event per critique item so the loop is fully visible in the
trajectory. Emits a best-effort warning when the reflection cap is hit without
approval so the frontend can flag the script accordingly.
"""
from __future__ import annotations

import json
import logging
import re

from app.config import get_settings
from app.graph.state import QAState
from app.llm.client import llm_client
from app.llm.prompts.critic import CRITIC_SYSTEM, build_critic_user
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _syntax_check(script: str) -> tuple[bool, str]:
    """Return (ok, error_message).  Uses compile() to catch all Python syntax errors."""
    try:
        compile(script, "<generated>", "exec")
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno}): {exc.text!r}"


def _parse_verdict(text: str) -> tuple[bool, list[dict]]:
    """Return (approved, feedback).  Defaults to approved=False on any parse failure."""
    m = _JSON_RE.search(text)
    if not m:
        logger.warning("Critic: no JSON found in LLM response")
        return False, [
            {"issue": "Critic response contained no JSON",
             "severity": "high", "suggestion": "Retry"},
        ]
    try:
        d = json.loads(m.group())
        approved = bool(d.get("approved", False))
        feedback = [
            {
                "issue":      str(item.get("issue", "")),
                "severity":   str(item.get("severity", "medium")),
                "suggestion": str(item.get("suggestion", "")),
            }
            for item in d.get("feedback", [])
            if isinstance(item, dict)
        ]
        return approved, feedback
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Critic: JSON parse error — %s", exc)
        return False, [
            {"issue": "Critic JSON parse error", "severity": "high", "suggestion": str(exc)},
        ]


async def critic_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Test Engineer (Critic)"

    script: str = state.get("script") or ""
    test_plan: list[dict] = state.get("test_plan") or []
    locators: dict = state.get("locators") or {}
    reflection_count: int = state.get("reflection_count", 0)

    await emitter.emit(
        run_id, agent, 3, "thought",
        f"Reviewing script (pass {reflection_count}) for assertions, "
        "logic, locator usage, and syntax…",
    )

    if not script:
        logger.warning("Critic: no script to review at pass %d", reflection_count)
        await emitter.emit(run_id, agent, 3, "decision", "No script produced; cannot approve.")
        return {
            "critic_approved": False,
            "critic_feedback": [
                {"issue": "No script produced", "severity": "high",
                 "suggestion": "Check scaffolder output."},
            ],
        }

    # Hard syntax gate: reject before calling the LLM if the script doesn't parse.
    syntax_ok, syntax_err = _syntax_check(script)
    if not syntax_ok:
        logger.warning(
            "Critic: syntax gate rejected script at pass %d — %s", reflection_count, syntax_err
        )
        feedback_item = {
            "issue": f"Python syntax error: {syntax_err}",
            "severity": "high",
            "suggestion": "Fix the syntax error before re-submitting. "
                          "Ensure no JS-style regex literals (/.+/) or other invalid Python.",
        }
        await emitter.emit(
            run_id, agent, 3, "decision",
            f"[HIGH] Syntax error detected — script rejected without LLM review: {syntax_err}",
            data=feedback_item,
        )
        return {"critic_approved": False, "critic_feedback": [feedback_item]}

    resp = await llm_client.complete(
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user",   "content": build_critic_user(script, test_plan, locators)},
        ],
        model_tier="reasoning",
        max_tokens=1024,
        run_id=run_id,
    )

    approved, feedback = _parse_verdict(resp.text)

    # Safety override: never approve when any HIGH severity issue is present,
    # regardless of what the LLM returned in the "approved" field.
    if approved and any(item.get("severity") == "high" for item in feedback):
        logger.warning(
            "Critic: LLM returned approved=true with HIGH issues at pass %d — "
            "overriding to unapproved. run=%s",
            reflection_count, run_id,
        )
        approved = False

    # Emit each critique item as its own decision event so the loop is visible.
    for item in feedback:
        await emitter.emit(
            run_id, agent, 3, "decision",
            f"[{item['severity'].upper()}] {item['issue']} — {item['suggestion']}",
            data=item,
        )

    settings = get_settings()
    at_cap = reflection_count >= settings.max_reflection_loops

    if approved:
        await emitter.emit(
            run_id, agent, 3, "decision",
            "Script approved — proceeding to execution.",
            data={"critic_approved": True, "reflection_count": reflection_count},
        )
    elif at_cap:
        await emitter.emit(
            run_id, agent, 3, "decision",
            f"Reflection cap reached ({reflection_count}/{settings.max_reflection_loops}) — "
            "script is best-effort/unapproved; proceeding to execution.",
            data={
                "critic_approved": False,
                "reflection_count": reflection_count,
                "best_effort": True,
            },
        )
    else:
        await emitter.emit(
            run_id, agent, 3, "decision",
            f"Script not approved ({len(feedback)} issue(s)); "
            "sending back to scaffolder for revision.",
            data={"critic_approved": False, "reflection_count": reflection_count},
        )

    logger.info(
        "Critic: pass %d, approved=%s, issues=%d, run=%s",
        reflection_count, approved, len(feedback), run_id,
    )
    return {"critic_approved": approved, "critic_feedback": feedback}
