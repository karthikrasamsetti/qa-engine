"""Synthesis Agent — Phase 5.

Analyses the execution result and produces a structured stakeholder report:
  verdict:         pass | bug | flaky
  outcome_label:   short human-readable label
  summary:         1-2 sentence executive summary
  tested:          what feature was exercised
  assessment:      for failures — real bug vs test-infrastructure problem
  recommendation:  what should happen next

On a successful run (execution_result.passed == True) the story, test plan, and
verified locators are persisted to the Chroma vector store so the Analyst can
surface similar prior work in future runs.
"""
from __future__ import annotations

import json
import logging
import re

from app.graph.state import QAState
from app.llm.client import llm_client
from app.llm.prompts.synthesis import SYNTHESIS_SYSTEM, build_synthesis_user
from app.streaming.events import emitter
from app.tools.sandbox import is_assertion_failure

logger = logging.getLogger(__name__)

_JSON_OBJ_RE = re.compile(r'\{.*\}', re.DOTALL)

_VERDICT_LABELS: dict[str, str] = {
    "pass":  "All tests passed",
    "bug":   "Application Bug Detected",
    "flaky": "Test Infrastructure Issue",
}


def _pre_classify(execution_result: dict) -> str:
    """Fast heuristic classification before the LLM call."""
    if execution_result.get("passed"):
        return "pass"
    error_text = (
        str(execution_result.get("error") or "")
        + " "
        + str(execution_result.get("logs") or "")
    )
    return "bug" if is_assertion_failure(error_text) else "flaky"


def _parse_report(
    text: str,
    pre_verdict: str,
    run_id: str,
    steps: int,
    heal_attempts: int,
) -> dict:
    """Extract the structured report from the LLM response.

    Falls back to a pre_verdict-based report when the response is not
    valid JSON or is missing the mandatory 'verdict' key.
    """
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            data = json.loads(m.group())
            if data.get("verdict") in ("pass", "bug", "flaky"):
                data.setdefault("run_id", run_id)
                data.setdefault("steps_covered", steps)
                data.setdefault("heal_attempts", heal_attempts)
                return data
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    logger.warning(
        "Synthesis: could not parse LLM report JSON — using fallback for run %s", run_id
    )
    return {
        "verdict":        pre_verdict,
        "outcome_label":  _VERDICT_LABELS.get(pre_verdict, pre_verdict),
        "summary":        f"Test run completed with verdict: {pre_verdict}.",
        "tested":         f"{steps} step(s) were exercised.",
        "assessment":     "Automated classification applied (LLM report unavailable).",
        "recommendation": "Review the execution logs for details.",
        "run_id":         run_id,
        "steps_covered":  steps,
        "heal_attempts":  heal_attempts,
    }


async def synthesis_node(state: QAState) -> dict:
    run_id       = state["run_id"]
    agent        = "Synthesis Agent"

    execution_result: dict      = state.get("execution_result") or {}
    test_plan: list[dict]       = state.get("test_plan") or []
    story: str                  = state.get("sanitized_story") or state.get("raw_input") or ""
    target_url: str             = state.get("target_url") or ""
    heal_attempts: int          = state.get("heal_attempts", 0)
    locators: dict              = state.get("locators") or {}

    pre_verdict = _pre_classify(execution_result)
    logger.info("Synthesis: pre-verdict=%s for run %s", pre_verdict, run_id)

    await emitter.emit(
        run_id, agent, 5, "thought",
        f"Pre-classified as {pre_verdict!r}. Generating stakeholder report…",
    )

    resp = await llm_client.complete(
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": build_synthesis_user(
                story=story,
                test_plan=test_plan,
                execution_result=execution_result,
                heal_attempts=heal_attempts,
                pre_verdict=pre_verdict,
            )},
        ],
        model_tier="reasoning",
        max_tokens=1024,
        run_id=run_id,
    )

    report = _parse_report(resp.text, pre_verdict, run_id, len(test_plan), heal_attempts)

    # Persist to vector store on genuinely successful runs only.
    if execution_result.get("passed"):
        try:
            from app.memory.vector_store import persist_run
            persist_run(
                run_id=run_id,
                story=story,
                test_plan=test_plan,
                locators=locators,
                verdict=report["verdict"],
                target_url=target_url,
            )
        except Exception as exc:
            # Non-fatal — memory is best-effort.
            logger.error("Synthesis: vector store persistence failed — %s", exc)

    await emitter.emit(
        run_id, agent, 5, "complete",
        f"Report complete — verdict: {report['verdict']} | {report.get('outcome_label', '')}",
        data={"report": report},
    )

    return {"status": "done", "report": report}
