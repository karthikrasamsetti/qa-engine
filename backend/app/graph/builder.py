"""Graph assembly.

Wires the per-node implementations into a LangGraph StateGraph with the
conditional edges that define the real workflow:
  - INVEST -> HITL pause when the story is vague (interrupt inside invest_node)
  - Scaffolder <-> Critic reflection loop until approved (or cap)
  - Execution -> Self-Heal on failure (capped) -> Execution again

Stage B: context / guardrail / invest have real logic; the rest are stubs.
The hitl_pause node is a safety fallback — with interrupt() inside invest_node
the graph never routes there, but it connects forward to analyst so if it is
ever reached the run continues rather than terminating.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.nodes.analyst import analyst_node
from app.graph.nodes.context import context_node
from app.graph.nodes.critic import critic_node
from app.graph.nodes.execution import execution_node
from app.graph.nodes.guardrail import guardrail_node
from app.graph.nodes.invest import invest_node
from app.graph.nodes.scaffolder import scaffolder_node
from app.graph.nodes.self_heal import self_heal_node
from app.graph.nodes.synthesis import synthesis_node
from app.graph.nodes.ui_mapper import ui_mapper_node
from app.graph.state import QAState


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def route_after_invest(state: QAState) -> str:
    """invest_node uses interrupt() for the HITL pause, so this router only
    runs after the node returns — which always has passed=True in Stage B.
    The hitl_pause branch is a dead-code safety net."""
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
        return "execution"
    return "scaffolder"


def route_after_execution(state: QAState) -> str:
    """Failed test -> self-heal (capped); passing test -> synthesis."""
    settings = get_settings()
    result = state.get("execution_result", {})
    if result.get("passed"):
        return "synthesis"
    if state.get("heal_attempts", 0) >= settings.max_heal_attempts:
        return "synthesis"
    return "self_heal"


def _hitl_pause(state: QAState) -> dict:
    """Safety-net node: if invest_node somehow returns passed=False without
    calling interrupt(), we still continue forward instead of terminating."""
    return {"status": "paused_hitl"}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the QA workflow graph."""
    g = StateGraph(QAState)

    # --- nodes ---
    g.add_node("context",   context_node)
    g.add_node("guardrail", guardrail_node)
    g.add_node("invest",    invest_node)
    g.add_node("hitl_pause", _hitl_pause)
    g.add_node("analyst",   analyst_node)
    g.add_node("ui_mapper", ui_mapper_node)
    g.add_node("scaffolder", scaffolder_node)
    g.add_node("critic",    critic_node)
    g.add_node("execution", execution_node)
    g.add_node("self_heal", self_heal_node)
    g.add_node("synthesis", synthesis_node)

    # Phase 1 linear chain
    g.add_edge(START, "context")
    g.add_edge("context", "guardrail")
    g.add_edge("guardrail", "invest")

    # INVEST -> HITL (safety net) or analyst
    g.add_conditional_edges(
        "invest", route_after_invest,
        {"hitl_pause": "hitl_pause", "analyst": "analyst"},
    )
    # Stage B: fallback path continues forward (Stage A had hitl_pause -> END)
    g.add_edge("hitl_pause", "analyst")

    # Phase 2
    g.add_edge("analyst",   "ui_mapper")
    g.add_edge("ui_mapper", "scaffolder")

    # Phase 3 reflection loop
    g.add_edge("scaffolder", "critic")
    g.add_conditional_edges(
        "critic", route_after_critic,
        {"scaffolder": "scaffolder", "execution": "execution"},
    )

    # Phase 4 execution + self-heal loop
    g.add_conditional_edges(
        "execution", route_after_execution,
        {"self_heal": "self_heal", "synthesis": "synthesis"},
    )
    g.add_edge("self_heal", "execution")

    # Phase 5
    g.add_edge("synthesis", END)

    return g.compile(checkpointer=MemorySaver())


# Compiled once at import; shared across requests.
graph = build_graph()
