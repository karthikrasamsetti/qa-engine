"""Shared test fixtures.

auto_mock_llm: always patches llm_client.complete with a stub that returns a
passing INVEST verdict as an LLMResponse, so the Stage A smoke test and any
test that doesn't need LLM control work deterministically without hitting the
network.

Tests that need specific LLM behaviour call monkeypatch.setattr again inside
their own body — the last write on the same monkeypatch instance wins.
"""
from __future__ import annotations

import pytest


_PASSING_VERDICT = (
    '{"passed": true, "scores": {"independent":8,"negotiable":8,"valuable":9,'
    '"estimable":7,"small":8,"testable":9}, "gaps": [], '
    '"overall_assessment": "Auto-mock: story meets INVEST criteria."}'
)


@pytest.fixture(autouse=True)
def auto_mock_llm(monkeypatch):
    """Always-on stub: returns a passing verdict as LLMResponse."""
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _stub(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        return LLMResponse(
            text=_PASSING_VERDICT,
            input_tokens=100,
            output_tokens=50,
            model="mock-model",
            cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _stub)
