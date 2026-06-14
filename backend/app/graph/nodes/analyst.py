"""Requirements Analyst stub — real logic in Stage D."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)


async def analyst_node(state: QAState) -> dict:
    run_id = state["run_id"]
    logger.info("[Requirements Analyst] decomposing story into test plan (stub)")
    await emitter.emit(run_id, "Requirements Analyst", 2, "thought",
                       "Decomposing story into a step-by-step test plan… (stub)")
    return {}
