"""Trajectory streaming primitives.

Defines the single event shape the SSE stream emits and the per-run async queue
the graph nodes push into. The frontend renders `message` and styles by `type`.

FREEZE the `TrajectoryEvent` shape alongside QAState — the frontend depends on it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EventType = Literal[
    "thought",        # internal reasoning, rendered dim
    "action",         # a concrete step the agent is taking
    "tool_call",      # invoking a tool (Jira, browser, sandbox)
    "tool_result",    # result returned from a tool
    "decision",       # routing / verdict (INVEST pass, critic verdict, etc.)
    "error",          # something failed
    "hitl_request",   # paused, waiting on a human (frontend shows answer box)
    "complete",       # a node / the run finished
]


class TrajectoryEvent(BaseModel):
    """One line of the agent's live trajectory."""

    run_id: str
    ts: float = Field(default_factory=time.time)
    agent: str                       # e.g. "INVEST Reviewer"
    phase: int                       # 1..5
    type: EventType
    message: str                     # human-readable line for the terminal UI
    data: dict[str, Any] = Field(default_factory=dict)


class EventEmitter:
    """Per-run fan-out of trajectory events over asyncio queues.

    A graph run pushes events via `emit`; the SSE endpoint drains the matching
    queue via `subscribe`. A sentinel `None` closes the stream.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Optional[TrajectoryEvent]]] = {}

    def _queue_for(self, run_id: str) -> asyncio.Queue[Optional[TrajectoryEvent]]:
        if run_id not in self._queues:
            self._queues[run_id] = asyncio.Queue()
        return self._queues[run_id]

    async def emit(
        self,
        run_id: str,
        agent: str,
        phase: int,
        type: EventType,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Push one event onto the run's queue."""
        event = TrajectoryEvent(
            run_id=run_id,
            agent=agent,
            phase=phase,
            type=type,
            message=message,
            data=data or {},
        )
        await self._queue_for(run_id).put(event)

    async def close(self, run_id: str) -> None:
        """Signal end-of-stream for a run."""
        await self._queue_for(run_id).put(None)

    async def subscribe(self, run_id: str):
        """Async generator yielding events until the stream is closed."""
        queue = self._queue_for(run_id)
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
        self._queues.pop(run_id, None)


# Process-wide singleton. Swap for Redis pub/sub if you scale to multiple workers.
emitter = EventEmitter()
