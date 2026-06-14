"""INVEST Reviewer: score the story; interrupt for human review when it fails.

On first run with a vague story:
  1. Calls Sonnet with the INVEST prompt.
  2. If passed=False: emits an hitl_request event, then calls interrupt(hitl_req).
  3. LangGraph saves the checkpoint; ainvoke() returns with __interrupt__ set.

On resume (LangGraph re-executes this node from the top):
  1. Calls Sonnet again (same result — still failed).
  2. interrupt() returns immediately with the human's clarification string.
  3. Accepts the clarification and proceeds with passed=True.
"""
from __future__ import annotations

import json
import logging
import re

from langgraph.types import interrupt

from app.graph.state import QAState
from app.llm.client import llm_client
from app.llm.prompts.invest import INVEST_SYSTEM, build_invest_user
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _parse_verdict(text: str) -> dict:
    """Extract JSON from the LLM response, tolerating markdown code fences."""
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object in LLM response: {text[:300]!r}")
    return json.loads(m.group())


async def invest_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "INVEST Reviewer"
    story = state.get("sanitized_story") or state.get("raw_input", "")

    await emitter.emit(run_id, agent, 1, "thought",
                       "Evaluating story against INVEST principles…")

    resp = await llm_client.complete(
        messages=[
            {"role": "system", "content": INVEST_SYSTEM},
            {"role": "user",   "content": build_invest_user(story)},
        ],
        model_tier="reasoning",
        max_tokens=1024,
        run_id=run_id,
    )

    try:
        verdict = _parse_verdict(resp.text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("INVEST: failed to parse LLM response — %s", exc)
        verdict = {
            "passed": False,
            "scores": {},
            "gaps": [f"LLM response parse error: {exc}"],
            "overall_assessment": "Parse error — treating as failed.",
        }

    scores: dict = verdict.get("scores", {})
    gaps: list = verdict.get("gaps", [])
    passed: bool = bool(verdict.get("passed", False))

    score_summary = (
        ", ".join(f"{k}={v}" for k, v in scores.items()) if scores else "no scores"
    )
    await emitter.emit(
        run_id, agent, 1, "decision",
        f"INVEST verdict: {'PASS' if passed else 'FAIL'} — {score_summary}",
        data={"verdict": verdict},
    )

    hitl_req: dict | None = None
    clarification: str | None = None

    if not passed:
        gap_summary = "; ".join(gaps) if gaps else "requirements are unclear"
        question = (
            "The story does not meet INVEST criteria. "
            f"Gaps: {gap_summary}. "
            "Please clarify with specific acceptance criteria, scope, and expected UI behaviour."
        )
        hitl_req = {
            "reason": "Story failed INVEST review",
            "question": question,
            "context": {"scores": scores, "gaps": gaps},
        }

        await emitter.emit(
            run_id, agent, 1, "hitl_request",
            f"Pausing — human clarification needed: {question}",
            data=hitl_req,
        )

        # LangGraph saves state here; on resume interrupt() returns immediately.
        clarification = interrupt(hitl_req)

        logger.info("INVEST: clarification received for run %s", run_id)
        await emitter.emit(
            run_id, agent, 1, "thought",
            f"Clarification received — proceeding to analyst. "
            f"({str(clarification)[:80]}…)",
        )
        # Accept clarification; re-scoring deferred to Stage H.
        verdict = {
            "passed": True,
            "scores": scores,
            "gaps": [],
            "overall_assessment": f"Clarification accepted: {clarification}",
            "clarification": clarification,
        }

    update: dict = {"invest_verdict": verdict}
    if hitl_req is not None:
        update["hitl_request"] = hitl_req
        update["hitl_response"] = clarification
    return update
