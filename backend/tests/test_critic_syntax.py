"""Tests for critic.py syntax-check hard gate.

Three scenarios:
  1. test_critic_rejects_js_regex_literal
     — Script with /.+/ JS regex never reaches the LLM; auto-rejected HIGH severity.
  2. test_critic_forces_unapproved_when_llm_approves_high_issues
     — Valid Python, LLM returns approved=true despite high-severity issues;
       critic must override to critic_approved=False.
  3. test_critic_clean_script_passes_through_normally
     — Valid Python, LLM approves with no issues → critic_approved=True.
"""
from __future__ import annotations

import pytest

from app.graph.nodes.critic import critic_node
from app.streaming.events import emitter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TEST_PLAN = [
    {
        "step_id": "step-001",
        "intent": "navigate to the login page URL",
        "action": "Open the target URL in the browser",
        "expected": "Login form is visible",
    },
]

_LOCATORS = {
    "step-001": {"css": "#email", "xpath": "//input[@id='email']", "confidence": 0.95},
}

# Script with a JS-style regex literal — SyntaxError in Python
_SCRIPT_WITH_JS_REGEX = """\
from playwright.sync_api import Page, expect

def test_login(page: Page) -> None:
    page.goto("https://example.com/login")
    pattern = /.+/
    expect(page.locator("#email")).to_be_visible()
"""

# Syntactically valid Playwright script
_CLEAN_SCRIPT = """\
from playwright.sync_api import Page, expect

def test_login(page: Page) -> None:
    page.goto("https://example.com/login")
    expect(page.locator("#email")).to_be_visible()
"""


def _make_llm_mock(text: str):
    from app.llm.client import LLMResponse

    async def _mock(messages, model_tier, **kw):
        return LLMResponse(
            text=text, input_tokens=10, output_tokens=10,
            model="mock", cost_usd=0.0,
        )

    return _mock


# ---------------------------------------------------------------------------
# 1. JS-style regex literal → auto-reject, no LLM call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critic_rejects_js_regex_literal(monkeypatch):
    """Script with /.+/ must be auto-rejected before any LLM call."""
    from app.llm import client as llm_mod

    llm_was_called = False

    async def _should_not_be_called(messages, model_tier, **kw):
        nonlocal llm_was_called
        llm_was_called = True
        from app.llm.client import LLMResponse
        return LLMResponse(
            text='{"approved": true, "feedback": []}',
            input_tokens=0, output_tokens=0, model="mock", cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _should_not_be_called)

    run_id = "critic-syntax-js-regex"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "script": _SCRIPT_WITH_JS_REGEX,
        "reflection_count": 1,
        "status": "running",
    }

    result = await critic_node(state)
    await emitter.close(run_id)

    assert result["critic_approved"] is False, (
        "JS-style regex literal must cause auto-rejection"
    )
    assert not llm_was_called, (
        "LLM must NOT be called when the script fails the syntax gate"
    )
    feedback = result["critic_feedback"]
    assert feedback, "At least one feedback item must be returned"
    high_items = [f for f in feedback if f.get("severity") == "high"]
    assert high_items, f"Expected HIGH severity feedback item, got: {feedback}"


# ---------------------------------------------------------------------------
# 2. Valid Python but LLM approves despite HIGH issues → forced unapproved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critic_forces_unapproved_when_llm_approves_high_issues(monkeypatch):
    """LLM returning approved=true with a HIGH severity issue must be overridden to False."""
    from app.llm import client as llm_mod

    # LLM inconsistently approves while flagging a HIGH issue
    llm_response = (
        '{"approved": true, "feedback": ['
        '{"issue": "Missing post-login assertion", "severity": "high",'
        ' "suggestion": "Add expect() on dashboard element after redirect"}'
        ']}'
    )
    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(llm_response))

    run_id = "critic-syntax-high-override"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "script": _CLEAN_SCRIPT,
        "reflection_count": 1,
        "status": "running",
    }

    result = await critic_node(state)
    await emitter.close(run_id)

    assert result["critic_approved"] is False, (
        "critic_approved must be forced to False when any HIGH severity issue is present, "
        "even if LLM returned approved=true"
    )
    feedback = result["critic_feedback"]
    assert any(f.get("severity") == "high" for f in feedback), (
        f"HIGH severity feedback must be preserved: {feedback}"
    )


# ---------------------------------------------------------------------------
# 3. Clean script + LLM approves → normal approval path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critic_clean_script_passes_through_normally(monkeypatch):
    """Valid Python + LLM approves with no issues → critic_approved=True unchanged."""
    from app.llm import client as llm_mod

    monkeypatch.setattr(
        llm_mod.llm_client, "complete",
        _make_llm_mock('{"approved": true, "feedback": []}'),
    )

    run_id = "critic-syntax-clean"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "script": _CLEAN_SCRIPT,
        "reflection_count": 1,
        "status": "running",
    }

    result = await critic_node(state)
    await emitter.close(run_id)

    assert result["critic_approved"] is True
    assert result["critic_feedback"] == []
