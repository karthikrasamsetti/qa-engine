"""Critic stub — real reflection loop in Stage E."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)


async def critic_node(state: QAState) -> dict:
    run_id = state["run_id"]
    logger.info("[Critic] reviewing script (stub)")
    await emitter.emit(run_id, "Test Engineer (Critic)", 3, "thought",
                       "Reviewing script for assertions, logic, syntax… (stub)")
    return {"critic_approved": True}
