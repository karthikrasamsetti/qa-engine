"""UI Mapper stub — real browser-use logic in Stage D."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

logger = logging.getLogger(__name__)


async def ui_mapper_node(state: QAState) -> dict:
    run_id = state["run_id"]
    logger.info("[UI Mapper] scanning DOM for locators (stub)")
    await emitter.emit(run_id, "UI Mapper", 2, "thought",
                       "Scanning DOM for stable CSS/XPath locators… (stub)")
    return {}
