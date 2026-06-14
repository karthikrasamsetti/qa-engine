"""Scaffolder stub — real Playwright generation in Stage E."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)


async def scaffolder_node(state: QAState) -> dict:
    run_id = state["run_id"]
    logger.info("[Scaffolder] generating Playwright script (stub)")
    await emitter.emit(run_id, "Scaffolder", 3, "thought",
                       "Generating initial Playwright script… (stub)")
    return {}
