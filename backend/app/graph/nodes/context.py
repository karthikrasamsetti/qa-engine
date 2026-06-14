"""Context Agent: resolve Jira ID → story dict, or pass raw text through."""
from __future__ import annotations

import logging

from app.graph.state import QAState
from app.streaming.events import emitter
from app.tools.mcp_jira import get_jira_client, looks_like_jira_id

logger = logging.getLogger(__name__)


async def context_node(state: QAState) -> dict:
    run_id = state["run_id"]
    raw = state["raw_input"]
    agent = "Context Agent"

    await emitter.emit(run_id, agent, 1, "thought",
                       "Analysing input — Jira ID or raw story text?")

    if looks_like_jira_id(raw):
        jira_id = raw.strip()
        await emitter.emit(run_id, agent, 1, "tool_call",
                           f"Fetching Jira ticket {jira_id}…")
        client = get_jira_client()
        context = await client.fetch_story(jira_id)
        logger.info("Context Agent: fetched %s", jira_id)
        preview = context.get("story", "")[:120]
        await emitter.emit(
            run_id, agent, 1, "tool_result",
            f"Fetched {jira_id}: {preview}…",
            data={"jira_id": jira_id},
        )
        return {"jira_context": context}

    await emitter.emit(run_id, agent, 1, "action",
                       "Raw story text detected — passing through.")
    return {"jira_context": {}}
