"""Stage A smoke test: run the stub graph and assert the trajectory streamed."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.builder import graph
from app.streaming.events import emitter


@pytest.mark.asyncio
async def test_stub_graph_streams_full_trajectory():
    run_id = "test-run-1"

    collected = []

    async def collect():
        async for event in emitter.subscribe(run_id):
            collected.append(event)

    consumer = asyncio.create_task(collect())

    config = {"configurable": {"thread_id": run_id}}
    final = await graph.ainvoke(
        {"run_id": run_id, "raw_input": "As a user I want to log in",
         "target_url": "http://example.com", "status": "running"},
        config=config,
    )
    await emitter.close(run_id)
    await consumer

    # Happy path through all five phases ends 'done'.
    assert final["status"] == "done"

    agents = {e.agent for e in collected}
    # At least the linear happy-path agents should have emitted.
    for expected in [
        "Context Agent", "Policy Enforcer", "INVEST Reviewer",
        "Requirements Analyst", "UI Mapper", "Scaffolder",
        "Test Engineer (Critic)", "Execution Agent", "Synthesis Agent",
    ]:
        assert expected in agents, f"missing events from {expected}"

    # Stream terminates with a 'complete' event.
    assert collected[-1].type == "complete"
