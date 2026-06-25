"""UI Mapper: resolve each test-plan step to CSS/XPath locators via Playwright."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter
from app.tools.browser import BrowserSession

logger = logging.getLogger(__name__)


async def ui_mapper_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "UI Mapper"
    test_plan: list[dict] = state.get("test_plan") or []
    target_url: str = state.get("target_url") or ""

    await emitter.emit(
        run_id, agent, 2, "thought",
        f"Mapping {len(test_plan)} test step(s) to DOM locators…",
    )

    if not target_url:
        logger.warning("UI Mapper: target_url not set; skipping DOM scan")
        await emitter.emit(run_id, agent, 2, "decision",
                           "No target_url — locator mapping skipped.")
        return {"locators": {}}

    if not test_plan:
        logger.warning("UI Mapper: test_plan is empty; skipping DOM scan")
        await emitter.emit(run_id, agent, 2, "decision",
                           "Test plan is empty — locator mapping skipped.")
        return {"locators": {}}

    locators: dict = {}

    async with BrowserSession() as session:
        for step in test_plan:
            step_id: str = step["step_id"]
            intent: str = step["intent"]

            await emitter.emit(
                run_id, agent, 2, "tool_call",
                f"[{step_id}] Scanning DOM for: {intent}",
                data={"step_id": step_id, "url": target_url, "intent": intent},
            )

            locator = await session.find_locators(target_url, intent, run_id=run_id)
            locators[step_id] = locator

            confidence = locator.get("confidence", 0.0)
            css = locator.get("css", "")
            await emitter.emit(
                run_id, agent, 2, "tool_result",
                f"[{step_id}] css={css!r}  confidence={confidence:.2f}",
                data={"step_id": step_id, "locator": locator},
            )
            logger.info(
                "UI Mapper: %s → css=%r  confidence=%.2f", step_id, css, confidence,
            )

    await emitter.emit(
        run_id, agent, 2, "decision",
        f"Locator mapping complete: {len(locators)} step(s) mapped.",
        data={"total_steps": len(locators)},
    )
    return {"locators": locators}
