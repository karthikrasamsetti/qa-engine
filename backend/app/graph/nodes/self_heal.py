"""Self-Heal Agent: re-scan DOM at the point of failure and patch the script.

Only heals locator-type failures (element not found, ambiguous selector).
Real assertion failures (wrong values, wrong URLs) are genuine bugs — the agent
reports them and exits the loop immediately without patching the script.

At the heal cap the agent emits a hitl_request event and calls interrupt() so a
human can review before the run is handed to synthesis.
"""
from __future__ import annotations

import logging
import re

from langgraph.types import interrupt

from app.config import get_settings
from app.graph.state import QAState
from app.streaming.events import emitter
from app.tools.browser import BrowserSession
from app.tools.sandbox import is_assertion_failure, is_locator_failure

logger = logging.getLogger(__name__)

# Extract a CSS selector from Playwright error lines like:
#   waiting for locator('#submit')
#   locator('#email').click()
_CSS_IN_LOG_RE = re.compile(r"""(?:waiting for )?locator\(['"]([^'"]+)['"]\)""")


def _find_failing_css(error_text: str) -> str:
    """Return the first CSS selector mentioned in the failure log, or ''."""
    m = _CSS_IN_LOG_RE.search(error_text)
    return m.group(1) if m else ""


def _find_failing_step(error_text: str, locators: dict) -> str | None:
    """Map a CSS selector in *error_text* back to its step_id via *locators*."""
    failing_css = _find_failing_css(error_text)
    if not failing_css:
        return None
    for step_id, loc in locators.items():
        if loc.get("css") == failing_css or loc.get("xpath") == failing_css:
            return step_id
    return None


def _patch_locator(script: str, old_selector: str, new_selector: str) -> str:
    """Replace all page.locator('<old>') references with page.locator('<new>')."""
    for q in ('"', "'"):
        script = script.replace(
            f"page.locator({q}{old_selector}{q})",
            f"page.locator({q}{new_selector}{q})",
        )
    return script


async def self_heal_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Self-Heal"
    settings = get_settings()

    execution_result: dict = state.get("execution_result") or {}
    error_text: str = f"{execution_result.get('error', '')} {execution_result.get('logs', '')}"
    locators: dict = dict(state.get("locators") or {})
    script: str = state.get("script") or ""
    target_url: str = state.get("target_url") or ""
    heal_attempts: int = state.get("heal_attempts", 0) + 1

    await emitter.emit(
        run_id, agent, 4, "thought",
        f"Analysing failure (heal attempt {heal_attempts}/{settings.max_heal_attempts})…",
    )

    # ------------------------------------------------------------------
    # Distinguish locator fault / assertion failure / environment error
    # ------------------------------------------------------------------
    if not is_locator_failure(error_text):
        if is_assertion_failure(error_text):
            await emitter.emit(
                run_id, agent, 4, "decision",
                "Failure is a genuine assertion error — self-heal does not apply. "
                "Reporting as a real bug.",
                data={"genuine_failure": True, "heal_attempts": heal_attempts},
            )
            logger.info("Self-Heal: assertion failure — skipping heal for run %s", run_id)
        else:
            # Neither a Playwright locator error nor a test assertion: treat as an
            # environment/infrastructure error (sandbox timeout, Docker unreachable, …).
            await emitter.emit(
                run_id, agent, 4, "error",
                "Failure appears to be an environment or infrastructure error "
                "(sandbox timeout, Docker unavailable, network unreachable) — "
                "self-heal cannot fix this. Surfacing as an execution error.",
                data={"environment_failure": True, "heal_attempts": heal_attempts},
            )
            logger.warning("Self-Heal: environment failure — skipping heal for run %s", run_id)
        # Force cap so route_after_execution routes to synthesis immediately.
        return {"heal_attempts": settings.max_heal_attempts}

    # ------------------------------------------------------------------
    # Identify which step's locator caused the failure
    # ------------------------------------------------------------------
    failing_step = _find_failing_step(error_text, locators)
    failing_css = _find_failing_css(error_text)

    if not failing_step:
        await emitter.emit(
            run_id, agent, 4, "decision",
            f"Could not identify a failing step from the error log "
            f"(selector={failing_css!r}). Incrementing counter without patching.",
            data={"heal_attempts": heal_attempts},
        )
        logger.warning("Self-Heal: could not map selector %r to a step", failing_css)
    else:
        intent = next(
            (s.get("intent", "") for s in (state.get("test_plan") or []) if s["step_id"] == failing_step),
            failing_step,
        )
        await emitter.emit(
            run_id, agent, 4, "action",
            f"Re-scanning DOM for step {failing_step}: {intent!r} (was {failing_css!r})…",
            data={"step_id": failing_step, "old_css": failing_css, "target_url": target_url},
        )

        # ------------------------------------------------------------------
        # Re-scan the live DOM — BrowserSession navigates to target_url.
        # Post-login elements (dashboards, menus) visible here because the
        # scan targets the URL at which the failure occurred rather than the
        # initial page, giving the mapper access to elements that weren't
        # present during static Phase 2 mapping.
        # ------------------------------------------------------------------
        try:
            async with BrowserSession() as session:
                new_loc = await session.find_locators(target_url, intent, run_id=run_id)
        except Exception as exc:
            logger.error("Self-Heal: BrowserSession error — %s", exc)
            new_loc = {"css": "", "xpath": "", "confidence": 0.0}

        new_css: str = new_loc.get("css", "")
        new_confidence: float = new_loc.get("confidence", 0.0)

        await emitter.emit(
            run_id, agent, 4, "tool_result",
            f"Re-scan for {failing_step}: css={new_css!r}  confidence={new_confidence:.2f}",
            data={"step_id": failing_step, "new_locator": new_loc},
        )

        if new_css and new_confidence > 0.5 and new_css != failing_css:
            script = _patch_locator(script, failing_css, new_css)
            locators[failing_step] = new_loc
            await emitter.emit(
                run_id, agent, 4, "decision",
                f"Patched {failing_step}: {failing_css!r} → {new_css!r}. "
                "Routing back to execution.",
                data={"old": failing_css, "new": new_css, "heal_attempts": heal_attempts},
            )
            logger.info(
                "Self-Heal: patched %s: %r → %r (run=%s)", failing_step, failing_css, new_css, run_id
            )
        else:
            await emitter.emit(
                run_id, agent, 4, "decision",
                f"No better locator found for {failing_step} "
                f"(css={new_css!r}, confidence={new_confidence:.2f}). "
                "Proceeding without patch.",
                data={"heal_attempts": heal_attempts},
            )

    # ------------------------------------------------------------------
    # Cap check — escalate to HITL when heal_attempts reaches the limit
    # ------------------------------------------------------------------
    if heal_attempts >= settings.max_heal_attempts:
        question = (
            f"The script has failed {heal_attempts} time(s) with locator errors "
            f"and could not be fully healed automatically. "
            f"Please review the script and locators, update the target URL, or mark "
            f"this test as a known failure."
        )
        hitl_req = {
            "reason": "Heal cap reached without resolution",
            "question": question,
            "context": {
                "heal_attempts": heal_attempts,
                "failing_step": failing_step,
                "error": execution_result.get("error", "")[:500],
            },
        }
        await emitter.emit(
            run_id, agent, 4, "hitl_request",
            f"Heal cap reached — pausing for human review: {question}",
            data=hitl_req,
        )
        logger.info("Self-Heal: escalating to HITL at cap=%d for run %s", heal_attempts, run_id)
        # LangGraph saves state here; on resume interrupt() returns immediately.
        _human_response = interrupt(hitl_req)
        logger.info("Self-Heal: resumed after HITL for run %s", run_id)

    return {
        "script": script,
        "locators": locators,
        "heal_attempts": heal_attempts,
    }
