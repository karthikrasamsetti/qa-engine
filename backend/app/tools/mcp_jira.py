"""MCP Jira client: interface + fake fixture for offline / test use.

Stage B: FakeJiraClient is the default when JIRA_MCP_URL is not configured.
MCPJiraClient stub accepts the URL but falls back to the fixture until the
real JSON-RPC wiring is added in Stage D.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from app.config import get_settings

logger = logging.getLogger(__name__)

# Matches PROJECT-123, DEMO-1, ABC123-999, etc.
_JIRA_ID_RE = re.compile(r'^[A-Z][A-Z0-9]*-\d+$')

_FIXTURES: dict[str, dict] = {
    "DEMO-1": {
        "story": (
            "As a registered user, I want to log in with my email and password "
            "so that I can access my personal dashboard."
        ),
        "epics": ["User Authentication"],
        "related_tickets": ["DEMO-2", "DEMO-3"],
        "acceptance_criteria": [
            "Given valid credentials, user is redirected to the dashboard.",
            "Given invalid credentials, an inline error message is displayed.",
            "Session expires after 30 minutes of inactivity.",
        ],
    }
}


def looks_like_jira_id(text: str) -> bool:
    return bool(_JIRA_ID_RE.match(text.strip()))


class JiraClient(ABC):
    @abstractmethod
    async def fetch_story(self, jira_id: str) -> dict:
        ...


class FakeJiraClient(JiraClient):
    async def fetch_story(self, jira_id: str) -> dict:
        logger.info("FakeJiraClient: returning fixture for %s", jira_id)
        return _FIXTURES.get(
            jira_id.strip(),
            {
                "story": f"[Fixture] Story for {jira_id}: As a user I want to perform some action.",
                "epics": ["General Epic"],
                "related_tickets": [],
                "acceptance_criteria": [],
            },
        )


class MCPJiraClient(JiraClient):
    """Stub MCP client — real JSON-RPC wiring deferred to Stage D."""

    def __init__(self, mcp_url: str) -> None:
        self._url = mcp_url

    async def fetch_story(self, jira_id: str) -> dict:
        logger.warning("MCPJiraClient: MCP not yet wired, using fixture for %s", jira_id)
        return await FakeJiraClient().fetch_story(jira_id)


def get_jira_client() -> JiraClient:
    settings = get_settings()
    if settings.jira_mcp_url:
        return MCPJiraClient(settings.jira_mcp_url)
    return FakeJiraClient()
