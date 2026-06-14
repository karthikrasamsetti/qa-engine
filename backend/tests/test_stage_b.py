"""Stage B tests: context + guardrail + invest (with HITL interrupt/resume).

Three scenarios:
  1. Vague story  -> INVEST fails -> interrupt() triggered -> __interrupt__ in result
  2. Good story   -> INVEST passes -> flows straight through to synthesis stub
  3. Resume       -> interrupted run resumes with clarification -> reaches analyst stub
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.types import Command

from app.graph.builder import graph
from app.streaming.events import EventEmitter, emitter

# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

_FAIL_VERDICT = (
    '{"passed": false, "scores": {"independent":3,"negotiable":5,"valuable":4,'
    '"estimable":2,"small":6,"testable":2}, "gaps": ["No acceptance criteria",'
    '"Action is too vague to test"], '
    '"overall_assessment": "Story lacks testable acceptance criteria."}'
)

_PASS_VERDICT = (
    '{"passed": true, "scores": {"independent":8,"negotiable":8,"valuable":9,'
    '"estimable":7,"small":8,"testable":9}, "gaps": [], '
    '"overall_assessment": "Story meets all INVEST criteria."}'
)


def _fail_llm(monkeypatch) -> None:
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        return LLMResponse(
            text=_FAIL_VERDICT,
            input_tokens=100, output_tokens=50,
            model="mock-model", cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _mock)


def _pass_llm(monkeypatch) -> None:
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        return LLMResponse(
            text=_PASS_VERDICT,
            input_tokens=100, output_tokens=50,
            model="mock-model", cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _mock)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(run_id: str, ev: EventEmitter = emitter):
    events = []
    async for e in ev.subscribe(run_id):
        events.append(e)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_vague_story_triggers_hitl(monkeypatch):
    """A vague story must trigger interrupt() and emit an hitl_request event."""
    _fail_llm(monkeypatch)

    run_id = "sb-vague-1"
    config = {"configurable": {"thread_id": run_id}}

    collected = []

    async def collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(collect())

    result = await graph.ainvoke(
        {"run_id": run_id, "raw_input": "Do stuff", "target_url": "",
         "status": "running"},
        config=config,
    )

    # Close stream after interrupt — consumer stops waiting
    await emitter.close(run_id)
    await consumer

    assert "__interrupt__" in result, "graph should have interrupted for HITL"

    event_types = [e.type for e in collected]
    assert "hitl_request" in event_types, "hitl_request event must be emitted"

    invest_events = [e for e in collected if e.agent == "INVEST Reviewer"]
    assert invest_events, "INVEST Reviewer must emit events"


async def test_good_story_passes_through(monkeypatch):
    """A good story passes INVEST without interruption and reaches synthesis."""
    _pass_llm(monkeypatch)

    run_id = "sb-good-1"
    config = {"configurable": {"thread_id": run_id}}

    collected = []

    async def collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(collect())

    result = await graph.ainvoke(
        {
            "run_id": run_id,
            "raw_input": (
                "As a registered user I want to log in with my email and password "
                "so that I can access my personal dashboard."
            ),
            "target_url": "",
            "status": "running",
        },
        config=config,
    )

    await emitter.close(run_id)
    await consumer

    assert "__interrupt__" not in result, "good story should not interrupt"
    assert result.get("status") == "done"

    agents = {e.agent for e in collected}
    for expected in ("Context Agent", "Policy Enforcer", "INVEST Reviewer",
                     "Requirements Analyst"):
        assert expected in agents, f"missing events from {expected}"

    event_types = [e.type for e in collected]
    assert "hitl_request" not in event_types, "no HITL event expected for passing story"
    assert "complete" in event_types


async def test_resume_after_hitl_continues_to_analyst(monkeypatch):
    """Resuming an interrupted run must reach the Requirements Analyst stub."""
    _fail_llm(monkeypatch)

    run_id = "sb-resume-1"
    config = {"configurable": {"thread_id": run_id}}

    collected = []

    async def collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(collect())

    # First invocation — triggers interrupt
    result1 = await graph.ainvoke(
        {"run_id": run_id, "raw_input": "Vague story", "target_url": "",
         "status": "running"},
        config=config,
    )
    assert "__interrupt__" in result1, "first run should be interrupted"

    # Resume with human clarification
    result2 = await graph.ainvoke(
        Command(resume=(
            "As a registered user I want to log in with my email and password "
            "so I can access my account dashboard. "
            "Acceptance criteria: valid credentials reach the dashboard; "
            "invalid credentials show an error message."
        )),
        config=config,
    )

    # Resume run reaches synthesis which does not close the emitter, so close manually
    await emitter.close(run_id)
    await consumer

    assert "__interrupt__" not in result2, "resumed run should complete without another interrupt"
    assert result2.get("status") == "done"

    agents = {e.agent for e in collected}
    assert "Requirements Analyst" in agents, "analyst stub must be reached after resume"
    assert "Synthesis Agent" in agents, "synthesis stub must be reached"

    # hitl_request is emitted twice: once on first run, once when invest_node
    # re-executes on resume (before interrupt() returns immediately).
    hitl_events = [e for e in collected if e.type == "hitl_request"]
    assert len(hitl_events) >= 1, "at least one hitl_request event expected"
