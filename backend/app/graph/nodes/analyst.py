"""Requirements Analyst: decompose the validated story into a structured test plan."""
from __future__ import annotations

import json
import logging
import re

from app.graph.state import QAState
from app.llm.client import llm_client
from app.llm.prompts.analyst import ANALYST_SYSTEM, build_analyst_user
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

_JSON_ARR_RE = re.compile(r'\[.*\]', re.DOTALL)


def _parse_test_plan(text: str) -> list[dict]:
    """Extract JSON array from LLM response and normalise each step.

    Raises ValueError when no array is found or when items are not dicts
    (e.g. the regex matched an inline string-array inside a verdict JSON).
    """
    m = _JSON_ARR_RE.search(text)
    if not m:
        raise ValueError(f"No JSON array found in LLM response: {text[:200]!r}")
    raw = json.loads(m.group())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array, got {type(raw).__name__}")
    plan: list[dict] = []
    for i, s in enumerate(raw, start=1):
        if not isinstance(s, dict):
            raise ValueError(
                f"Array item {i} is not a dict (got {type(s).__name__!r}); "
                "response likely isn't a test plan"
            )
        plan.append({
            "step_id":  s.get("step_id",  f"step-{i:03d}"),
            "intent":   s.get("intent",   ""),
            "action":   s.get("action",   ""),
            "expected": s.get("expected", ""),
        })
    return plan


async def analyst_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Requirements Analyst"
    story = state.get("sanitized_story") or state.get("raw_input", "")

    await emitter.emit(
        run_id, agent, 2, "thought",
        "Decomposing story into a step-by-step test plan…",
    )

    # Light retrieval: surface similar prior runs before generating fresh content.
    # Non-fatal — any failure is logged and the node continues normally.
    try:
        from app.memory.vector_store import retrieve_similar
        similar = retrieve_similar(story, n_results=1)
        if similar:
            hit = similar[0]
            await emitter.emit(
                run_id, agent, 2, "decision",
                f"Similar prior run found (run_id={hit['run_id']!r}, "
                f"verdict={hit['verdict']!r}, distance={hit['distance']:.3f}). "
                "Generating fresh plan — prior results available for context.",
                data={"cache_hit": True, "prior_run": hit},
            )
            logger.info(
                "Analyst: cache hit for run %s — prior=%s verdict=%s",
                run_id, hit["run_id"], hit["verdict"],
            )
    except Exception as exc:
        logger.warning("Analyst: vector store check failed — %s", exc)

    resp = await llm_client.complete(
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM},
            {"role": "user",   "content": build_analyst_user(story)},
        ],
        model_tier="reasoning",
        max_tokens=2048,
        run_id=run_id,
    )

    try:
        test_plan = _parse_test_plan(resp.text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Analyst: failed to parse test plan — %s", exc)
        test_plan = []

    step_count = len(test_plan)
    await emitter.emit(
        run_id, agent, 2, "decision",
        f"Test plan generated: {step_count} step(s).",
        data={"step_count": step_count, "steps": [s["step_id"] for s in test_plan]},
    )
    logger.info("Analyst: generated %d steps for run %s", step_count, run_id)
    return {"test_plan": test_plan}
