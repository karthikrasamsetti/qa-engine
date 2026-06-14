"""Execution Agent stub — real Docker sandbox in Stage F."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)


async def execution_node(state: QAState) -> dict:
    run_id = state["run_id"]
    logger.info("[Execution Agent] running script in sandbox (stub)")
    await emitter.emit(run_id, "Execution Agent", 4, "thought",
                       "Running script in Docker sandbox… (stub)")
    return {"execution_result": {"passed": True, "logs": "", "error": None}}
