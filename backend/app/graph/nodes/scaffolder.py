"""Scaffolder: generate (and revise) a Playwright pytest test script.

First pass  — build from test_plan + locators from scratch.
Later passes — apply critic_feedback to the existing script.
"""
from __future__ import annotations

import logging
import re

from app.graph.state import QAState
from app.llm.client import llm_client
from app.llm.prompts.scaffolder import SCAFFOLDER_SYSTEM, build_scaffolder_user
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

# Strip optional ```python ... ``` or ``` ... ``` fences the model may add.
_FENCE_RE = re.compile(r"^```(?:python)?\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


async def scaffolder_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Scaffolder"

    test_plan: list[dict] = state.get("test_plan") or []
    locators: dict = state.get("locators") or {}
    critic_feedback: list[dict] | None = state.get("critic_feedback") or None
    existing_script: str = state.get("script") or ""
    reflection_count: int = state.get("reflection_count", 0) + 1

    is_revision = bool(critic_feedback and existing_script)

    await emitter.emit(
        run_id, agent, 3, "thought",
        (
            f"Revising script (pass {reflection_count}) addressing "
            f"{len(critic_feedback)} critique(s)…"
            if is_revision else
            "Generating initial Playwright test script from test plan and locators…"
        ),
    )

    resp = await llm_client.complete(
        messages=[
            {"role": "system", "content": SCAFFOLDER_SYSTEM},
            {"role": "user",   "content": build_scaffolder_user(
                test_plan,
                locators,
                critic_feedback=critic_feedback if is_revision else None,
                existing_script=existing_script if is_revision else "",
            )},
        ],
        model_tier="reasoning",
        max_tokens=4096,
        run_id=run_id,
    )

    script = _strip_fences(resp.text)
    if not script:
        logger.warning("Scaffolder: LLM returned empty script on pass %d", reflection_count)
        script = "# Scaffolder produced no output — check LLM response."

    await emitter.emit(
        run_id, agent, 3, "decision",
        f"Script {'revised' if is_revision else 'generated'} "
        f"(pass {reflection_count}, {len(script)} chars).",
        data={"reflection_count": reflection_count, "script": script},
    )
    logger.info("Scaffolder: pass %d, %d chars for run %s", reflection_count, len(script), run_id)

    return {"script": script, "reflection_count": reflection_count}
