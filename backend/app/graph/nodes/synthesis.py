"""Synthesis Agent stub — real reporting in Stage G."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)


async def synthesis_node(state: QAState) -> dict:
    run_id = state["run_id"]
    logger.info("[Synthesis Agent] generating stakeholder report (stub)")
    await emitter.emit(run_id, "Synthesis Agent", 5, "thought",
                       "Analysing logs and generating stakeholder report… (stub)")
    await emitter.emit(run_id, "Synthesis Agent", 5, "complete", "Run complete.")
    return {"status": "done", "report": {"summary": "stub run complete"}}
