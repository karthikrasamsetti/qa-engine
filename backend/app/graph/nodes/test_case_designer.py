"""Test Case Designer: decompose a validated story into test scenarios.

Runs after INVEST passes and before the Requirements Analyst.

Steps:
  1. Call the reasoning LLM to produce a JSON array of 3-7 test scenarios.
  2. Parse scenarios; fall back to a single generic scenario on parse failure.
  3. Emit one decision event per scenario, then a summary event with the full list.
  4. If review_scenarios=True in state, pause via interrupt() for human edit/approval.
  5. Return {"scenarios": <approved list>}.
"""
from __future__ import annotations

import json
import logging
import re

from langgraph.types import interrupt

from app.graph.state import QAState
from app.llm.client import llm_client
from app.llm.prompts.test_case_designer import DESIGNER_SYSTEM, build_designer_user
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

_JSON_ARR_RE = re.compile(r'\[.*\]', re.DOTALL)

_REQUIRED_FIELDS = frozenset({
    "scenario_id", "title", "type", "preconditions", "description", "expected_outcome"
})

_FALLBACK_SCENARIO: list[dict] = [
    {
        "scenario_id": "sc-001",
        "title": "Happy path scenario",
        "type": "happy_path",
        "preconditions": [],
        "description": "Execute the primary user flow as described in the story.",
        "expected_outcome": "The primary user goal is achieved successfully.",
    }
]


def _parse_scenarios(text: str) -> list[dict]:
    """Extract a JSON array of scenarios from *text*.

    Returns a normalised list on success; falls back to ``_FALLBACK_SCENARIO``
    when no parseable array is found or none of the items have the required fields.
    """
    m = _JSON_ARR_RE.search(text)
    if not m:
        logger.warning("Test Case Designer: no JSON array found in LLM response — using fallback")
        return list(_FALLBACK_SCENARIO)

    try:
        raw = json.loads(m.group())
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Test Case Designer: JSON parse error (%s) — using fallback", exc)
        return list(_FALLBACK_SCENARIO)

    if not isinstance(raw, list):
        logger.warning("Test Case Designer: LLM returned non-list JSON — using fallback")
        return list(_FALLBACK_SCENARIO)

    normalised: list[dict] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        if not (_REQUIRED_FIELDS & set(item.keys())):
            continue
        normalised.append({
            "scenario_id": item.get("scenario_id", f"sc-{i:03d}"),
            "title": str(item.get("title", f"Scenario {i}")),
            "type": item.get("type", "happy_path"),
            "preconditions": item.get("preconditions", []),
            "description": str(item.get("description", "")),
            "expected_outcome": str(item.get("expected_outcome", "")),
        })

    if not normalised:
        logger.warning("Test Case Designer: no valid scenario items found — using fallback")
        return list(_FALLBACK_SCENARIO)

    return normalised


async def test_case_designer_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Test Case Designer"
    story = state.get("sanitized_story") or state.get("raw_input", "")
    invest_verdict: dict = state.get("invest_verdict", {})
    review_scenarios: bool = bool(state.get("review_scenarios", False))

    await emitter.emit(run_id, agent, 1, "thought",
                       "Decomposing validated story into test scenarios…")

    resp = await llm_client.complete(
        messages=[
            {"role": "system", "content": DESIGNER_SYSTEM},
            {"role": "user",   "content": build_designer_user(story, invest_verdict)},
        ],
        model_tier="reasoning",
        max_tokens=2048,
        run_id=run_id,
    )

    scenarios = _parse_scenarios(resp.text)

    # One decision event per scenario
    for sc in scenarios:
        await emitter.emit(
            run_id, agent, 1, "decision",
            f"[{sc['type'].upper()}] {sc['title']}",
            data=sc,
        )

    # Summary event carrying the full approved list
    await emitter.emit(
        run_id, agent, 1, "decision",
        f"{len(scenarios)} scenario(s) generated for story.",
        data={"scenarios": scenarios},
    )

    # Optional human review gate
    if review_scenarios:
        hitl_req = {
            "reason": "Scenario review requested",
            "question": (
                "Review and edit the generated test scenarios. "
                "Return the approved list as a JSON array using the same schema."
            ),
            "context": {"scenarios": scenarios},
        }

        await emitter.emit(
            run_id, agent, 1, "hitl_request",
            f"Pausing for human review of {len(scenarios)} scenario(s).",
            data=hitl_req,
        )

        # LangGraph saves checkpoint here; on resume interrupt() returns immediately.
        human_response = interrupt(hitl_req)

        # Try to parse human-edited scenarios; fall back to LLM-generated ones.
        edited = _parse_scenarios(
            human_response if isinstance(human_response, str) else json.dumps(human_response)
        )
        if edited and edited != _FALLBACK_SCENARIO:
            scenarios = edited
            await emitter.emit(
                run_id, agent, 1, "thought",
                f"Human-approved scenario list: {len(scenarios)} scenario(s).",
            )

    return {"scenarios": scenarios}
