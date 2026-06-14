"""Shared test fixtures.

auto_mock_llm: when no real ANTHROPIC_API_KEY is configured (placeholder or
empty), automatically patch the LLM client to return a passing INVEST verdict so
the Stage A smoke test and other tests that don't need LLM control still pass.
Tests that need specific LLM behaviour override the mock via their own
monkeypatch.setattr call (same monkeypatch instance, last write wins).
"""
from __future__ import annotations

import pytest


def _has_real_api_key() -> bool:
    from app.config import get_settings
    key = get_settings().anthropic_api_key
    return bool(key) and not key.startswith("sk-ant-...") and len(key) > 30


_PASSING_VERDICT = (
    '{"passed": true, "scores": {"independent":8,"negotiable":8,"valuable":9,'
    '"estimable":7,"small":8,"testable":9}, "gaps": [], '
    '"overall_assessment": "Auto-mock: story meets INVEST criteria."}'
)


@pytest.fixture(autouse=True)
def auto_mock_llm(monkeypatch):
    """Patch llm_client.complete with a passing verdict when key is a placeholder."""
    if _has_real_api_key():
        return  # real key present — don't interfere

    from app.llm import client as llm_mod

    async def _stub(tier, system, user, max_tokens=2048):
        return _PASSING_VERDICT

    monkeypatch.setattr(llm_mod.llm_client, "complete", _stub)
