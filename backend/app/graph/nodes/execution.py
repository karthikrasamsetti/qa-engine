"""Execution Agent: run the generated Playwright script in a Docker sandbox."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter
from app.tools.sandbox import run_script

logger = logging.getLogger(__name__)


async def execution_node(state: QAState) -> dict:
    run_id = state["run_id"]
    agent = "Execution Agent"
    script: str = state.get("script") or ""
    target_url: str = state.get("target_url") or ""
    heal_attempts: int = state.get("heal_attempts", 0)

    await emitter.emit(
        run_id, agent, 4, "thought",
        f"Running script in Docker sandbox"
        f"{f' (heal attempt {heal_attempts})' if heal_attempts else ''}…",
    )

    if not script:
        logger.warning("Execution Agent: no script in state for run %s", run_id)
        await emitter.emit(run_id, agent, 4, "error", "No script to execute.")
        return {
            "execution_result": {
                "passed": False,
                "logs": "",
                "error": "No script was generated.",
                "screenshots": [],
                "exit_code": -1,
            }
        }

    await emitter.emit(
        run_id, agent, 4, "tool_call",
        f"Sandbox: executing script ({len(script)} chars) against {target_url!r}",
        data={"target_url": target_url, "script_length": len(script)},
    )

    result = await run_script(script, target_url=target_url)

    passed: bool = result["passed"]
    exit_code: int = result["exit_code"]
    error: str | None = result.get("error")

    if passed:
        await emitter.emit(
            run_id, agent, 4, "tool_result",
            f"Script passed (exit_code={exit_code}).",
            data=result,
        )
        logger.info("Execution Agent: PASS for run %s", run_id)
    else:
        await emitter.emit(
            run_id, agent, 4, "error",
            f"Script failed (exit_code={exit_code}): {str(error)[:200]}",
            data=result,
        )
        logger.warning("Execution Agent: FAIL for run %s — %s", run_id, str(error)[:120])

    return {"execution_result": result}
