"""Stage I-a tests: Test Case Designer node.

Five scenarios:
  1. test_story_produces_multiple_distinct_scenarios
     — LLM returns 3 typed scenarios; all required fields present
  2. test_designer_emits_per_scenario_and_summary_events
     — one decision event per scenario + summary event with 'scenarios' data key
  3. test_review_gate_pauses_when_flag_set
     — review_scenarios=True → interrupt() called, hitl_request event emitted
  4. test_designer_auto_proceeds_when_flag_not_set
     — review_scenarios=False (default) → interrupt() NOT called
  5. test_resume_with_edited_scenarios_updates_list
     — human returns edited JSON via interrupt → result["scenarios"] reflects edits
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from app.streaming.events import emitter, TrajectoryEvent

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_STORY = (
    "As a registered user I want to log in with my email and password "
    "so that I can access my personal dashboard."
)

_THREE_SCENARIOS = json.dumps([
    {
        "scenario_id": "sc-001",
        "title": "Successful login with valid credentials",
        "type": "happy_path",
        "preconditions": ["User is registered", "User is on the login page"],
        "description": "User enters valid email and password then clicks Sign In.",
        "expected_outcome": "User is redirected to the personal dashboard.",
    },
    {
        "scenario_id": "sc-002",
        "title": "Login rejected with wrong password",
        "type": "negative",
        "preconditions": ["User is registered", "User is on the login page"],
        "description": "User enters a valid email but an incorrect password.",
        "expected_outcome": "An inline error message is displayed; user stays on login page.",
    },
    {
        "scenario_id": "sc-003",
        "title": "Login with unregistered email address",
        "type": "edge_case",
        "preconditions": ["User is on the login page"],
        "description": "User enters an email address that has no account.",
        "expected_outcome": "A user-friendly error message is shown.",
    },
])

_EDITED_SCENARIOS = json.dumps([
    {
        "scenario_id": "sc-001",
        "title": "Happy path — human edited",
        "type": "happy_path",
        "preconditions": ["User exists"],
        "description": "Human-edited happy path.",
        "expected_outcome": "Dashboard loads.",
    }
])


def _make_llm_mock(text: str):
    from app.llm.client import LLMResponse

    async def _mock(messages, model_tier, **kw):
        return LLMResponse(
            text=text, input_tokens=10, output_tokens=50,
            model="mock", cost_usd=0.0,
        )

    return _mock


def _base_state(run_id: str, review: bool = False) -> dict:
    return {
        "run_id":            run_id,
        "raw_input":         _STORY,
        "sanitized_story":   _STORY,
        "invest_verdict":    {"passed": True, "scores": {}, "gaps": []},
        "review_scenarios":  review,
        "status":            "running",
    }


async def _drain(run_id: str) -> list[TrajectoryEvent]:
    events: list[TrajectoryEvent] = []
    while True:
        try:
            ev = emitter._queues[run_id].get_nowait()
        except (KeyError, asyncio.QueueEmpty):
            break
        if ev is None:
            break
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# 1. Story produces multiple distinct typed scenarios
# ---------------------------------------------------------------------------

async def test_story_produces_multiple_distinct_scenarios(monkeypatch):
    """LLM returns 3 scenarios; result contains them all with correct fields."""
    from app.llm import client as llm_mod
    from app.graph.nodes.test_case_designer import test_case_designer_node

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_THREE_SCENARIOS))

    run_id = "si-multi"
    result = await test_case_designer_node(_base_state(run_id))
    await emitter.close(run_id)

    scenarios = result.get("scenarios", [])
    assert len(scenarios) == 3, f"Expected 3 scenarios, got {len(scenarios)}: {scenarios}"

    required_fields = {"scenario_id", "title", "type", "preconditions", "description", "expected_outcome"}
    for sc in scenarios:
        missing = required_fields - set(sc.keys())
        assert not missing, f"Scenario {sc.get('scenario_id')} missing fields: {missing}"

    types = {sc["type"] for sc in scenarios}
    assert "happy_path" in types, "Must include at least one happy_path scenario"
    assert len(types) > 1, f"Must include multiple types (got only: {types})"


# ---------------------------------------------------------------------------
# 2. One decision event per scenario + summary event with 'scenarios' data key
# ---------------------------------------------------------------------------

async def test_designer_emits_per_scenario_and_summary_events(monkeypatch):
    """Each scenario gets its own decision event, plus a summary event with data['scenarios']."""
    from app.llm import client as llm_mod
    from app.graph.nodes.test_case_designer import test_case_designer_node

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_THREE_SCENARIOS))

    run_id = "si-events"
    await test_case_designer_node(_base_state(run_id))
    events = await _drain(run_id)

    designer_events = [e for e in events if e.agent == "Test Case Designer"]
    assert designer_events, "Test Case Designer must emit events"

    decision_events = [e for e in designer_events if e.type == "decision"]
    assert len(decision_events) >= 3, (
        f"Expected at least one decision event per scenario (3), "
        f"got {len(decision_events)}: {[e.message for e in decision_events]}"
    )

    # At least one decision event carries the full scenarios list in data
    summary_events = [e for e in decision_events if "scenarios" in e.data]
    assert summary_events, (
        "At least one decision event must carry data['scenarios'] with the full list"
    )
    assert isinstance(summary_events[-1].data["scenarios"], list)
    assert len(summary_events[-1].data["scenarios"]) == 3


# ---------------------------------------------------------------------------
# 3. Review gate pauses (interrupt called) when review_scenarios=True
# ---------------------------------------------------------------------------

async def test_review_gate_pauses_when_flag_set(monkeypatch):
    """When review_scenarios=True, interrupt() is called and hitl_request event is emitted."""
    from app.llm import client as llm_mod
    from app.graph.nodes.test_case_designer import test_case_designer_node

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_THREE_SCENARIOS))

    run_id = "si-pause"
    with patch("app.graph.nodes.test_case_designer.interrupt", return_value="ok") as mock_int:
        await test_case_designer_node(_base_state(run_id, review=True))

    events = await _drain(run_id)
    await emitter.close(run_id)

    assert mock_int.called, "interrupt() must be called when review_scenarios=True"

    hitl_events = [e for e in events if e.type == "hitl_request"]
    assert hitl_events, (
        "hitl_request event must be emitted before pausing for review"
    )


# ---------------------------------------------------------------------------
# 4. Auto-proceeds (interrupt NOT called) when flag is absent/False
# ---------------------------------------------------------------------------

async def test_designer_auto_proceeds_when_flag_not_set(monkeypatch):
    """When review_scenarios is False or absent, interrupt() must NOT be called."""
    from app.llm import client as llm_mod
    from app.graph.nodes.test_case_designer import test_case_designer_node

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_THREE_SCENARIOS))

    run_id = "si-auto"
    with patch("app.graph.nodes.test_case_designer.interrupt", return_value="ok") as mock_int:
        await test_case_designer_node(_base_state(run_id, review=False))

    await emitter.close(run_id)

    assert not mock_int.called, "interrupt() must NOT be called when review_scenarios=False"


# ---------------------------------------------------------------------------
# 5. Resume with human-edited JSON updates the scenario list
# ---------------------------------------------------------------------------

async def test_resume_with_edited_scenarios_updates_list(monkeypatch):
    """When interrupt() returns an edited JSON list, result['scenarios'] reflects the edits."""
    from app.llm import client as llm_mod
    from app.graph.nodes.test_case_designer import test_case_designer_node

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_THREE_SCENARIOS))

    run_id = "si-edit"
    # interrupt() returns a human-edited single-scenario JSON
    with patch("app.graph.nodes.test_case_designer.interrupt", return_value=_EDITED_SCENARIOS):
        result = await test_case_designer_node(_base_state(run_id, review=True))

    await emitter.close(run_id)

    scenarios = result.get("scenarios", [])
    assert len(scenarios) == 1, (
        f"Human edited down to 1 scenario but got {len(scenarios)}: {scenarios}"
    )
    assert scenarios[0]["title"] == "Happy path — human edited", (
        f"Expected edited title, got: {scenarios[0].get('title')}"
    )
