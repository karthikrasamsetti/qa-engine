"""Graph assembly.

Wires the node stubs into a LangGraph StateGraph with the conditional edges that
define the real workflow:
  - INVEST -> HITL pause when the story is vague
  - Scaffolder <-> Critic reflection loop until approved (or cap)
  - Execution -> Self-Heal on failure (capped) -> Execution again

In Stage A the stubs always take the happy path, but the edges are wired the way
the finished system needs them so later stages only swap node bodies.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.nodes import stubs
from app.graph.state import QAState


# --- conditional edge routers ---

def route_after_invest(state: QAState) -> str:
    """Vague story -> pause for a human; otherwise continue to Phase 2."""
    verdict = state.get("invest_verdict", {})
    if not verdict.get("passed", False):
        return "hitl_pause"
    return "analyst"


def route_after_critic(state: QAState) -> str:
    """Loop scaffolder<->critic until approved or the reflection cap is hit."""
    settings = get_settings()
    if state.get("critic_approved"):
        return "execution"
    if state.get("reflection_count", 0) >= settings.max_reflection_loops:
        return "execution"  # best-effort exit with a warning (emitted by node)
    return "scaffolder"


def route_after_execution(state: QAState) -> str:
    """Failed test -> self-heal (capped); passing test -> synthesis."""
    settings = get_settings()
    result = state.get("execution_result", {})
    if result.get("passed"):
        return "synthesis"
    if state.get("heal_attempts", 0) >= settings.max_heal_attempts:
        return "synthesis"  # escalation handled inside the node before this
    return "self_heal"


def _hitl_pause(state: QAState) -> dict:
    """Terminal-for-now node representing the paused state.

    With a checkpointer + interrupt, the graph halts here; `/resume` re-enters
    the graph with `hitl_response` set. Stage B implements the real interrupt.
    """
    return {"status": "paused_hitl"}


def build_graph():
    """Construct and compile the QA workflow graph."""
    g = StateGraph(QAState)

    # nodes
    g.add_node("context", stubs.context_node)
    g.add_node("guardrail", stubs.guardrail_node)
    g.add_node("invest", stubs.invest_node)
    g.add_node("hitl_pause", _hitl_pause)
    g.add_node("analyst", stubs.analyst_node)
    g.add_node("ui_mapper", stubs.ui_mapper_node)
    g.add_node("scaffolder", stubs.scaffolder_node)
    g.add_node("critic", stubs.critic_node)
    g.add_node("execution", stubs.execution_node)
    g.add_node("self_heal", stubs.self_heal_node)
    g.add_node("synthesis", stubs.synthesis_node)

    # Phase 1 linear chain
    g.add_edge(START, "context")
    g.add_edge("context", "guardrail")
    g.add_edge("guardrail", "invest")

    # INVEST -> HITL or continue
    g.add_conditional_edges("invest", route_after_invest,
                            {"hitl_pause": "hitl_pause", "analyst": "analyst"})
    g.add_edge("hitl_pause", END)  # Stage B: interrupt + resume into "analyst"

    # Phase 2
    g.add_edge("analyst", "ui_mapper")
    g.add_edge("ui_mapper", "scaffolder")

    # Phase 3 reflection loop
    g.add_edge("scaffolder", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"scaffolder": "scaffolder", "execution": "execution"})

    # Phase 4 execution + self-heal loop
    g.add_conditional_edges("execution", route_after_execution,
                            {"self_heal": "self_heal", "synthesis": "synthesis"})
    g.add_edge("self_heal", "execution")

    # Phase 5
    g.add_edge("synthesis", END)

    return g.compile(checkpointer=MemorySaver())


# Compiled once at import; share across requests.
graph = build_graph()
