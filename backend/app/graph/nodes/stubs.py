"""Node stubs for the QA workflow graph.

Stage A: each node emits a single 'thought' event so the trajectory stream is
visible end-to-end, then returns state unchanged. Later stages replace these
stubs with real agent logic (and split them into one-file-per-node under
`nodes/`).

Every node is `async def node(state) -> dict` returning a partial state update,
which LangGraph merges into QAState.
"""

from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter

log = logging.getLogger(__name__)


async def _stub(state: QAState, agent: str, phase: int, msg: str) -> dict:
    """Shared stub body: log + emit one thought event, no state change."""
    run_id = state["run_id"]
    log.info("[%s] %s", agent, msg)
    await emitter.emit(run_id, agent, phase, "thought", msg)
    return {}


# --- Phase 1 ---
async def context_node(state: QAState) -> dict:
    return await _stub(state, "Context Agent", 1,
                       "Fetching Jira context (story, epics, history)…")


async def guardrail_node(state: QAState) -> dict:
    return await _stub(state, "Policy Enforcer", 1,
                       "Sanitizing input against prompt injection…")


async def invest_node(state: QAState) -> dict:
    # Stage A stub: pretend the story passes so the skeleton flows to the end.
    # Stage B replaces this with a real Sonnet call that may pause for HITL.
    await _stub(state, "INVEST Reviewer", 1,
                "Evaluating story against INVEST principles…")
    return {"invest_verdict": {"passed": True, "scores": {}, "gaps": []}}


# --- Phase 2 ---
async def analyst_node(state: QAState) -> dict:
    return await _stub(state, "Requirements Analyst", 2,
                       "Decomposing story into a step-by-step test plan…")


async def ui_mapper_node(state: QAState) -> dict:
    return await _stub(state, "UI Mapper", 2,
                       "Scanning DOM for stable CSS/XPath locators…")


# --- Phase 3 ---
async def scaffolder_node(state: QAState) -> dict:
    return await _stub(state, "Scaffolder", 3,
                       "Generating initial Playwright script…")


async def critic_node(state: QAState) -> dict:
    # Stage A stub: approve immediately so the reflection loop exits.
    await _stub(state, "Test Engineer (Critic)", 3,
                "Reviewing script for assertions, logic, syntax…")
    return {"critic_approved": True}


# --- Phase 4 ---
async def execution_node(state: QAState) -> dict:
    # Stage A stub: pretend the test passed so we skip self-heal.
    await _stub(state, "Execution Agent", 4,
                "Running script in Docker sandbox…")
    return {"execution_result": {"passed": True, "logs": "", "error": None}}


async def self_heal_node(state: QAState) -> dict:
    return await _stub(state, "Self-Heal", 4,
                       "Re-scanning DOM and patching broken locator…")


# --- Phase 5 ---
async def synthesis_node(state: QAState) -> dict:
    await _stub(state, "Synthesis Agent", 5,
                "Analyzing logs and generating stakeholder report…")
    await emitter.emit(state["run_id"], "Synthesis Agent", 5, "complete",
                       "Run complete.")
    return {"status": "done", "report": {"summary": "stub run complete"}}
