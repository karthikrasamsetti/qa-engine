"""Stage E tests: scaffolder node + critic node (reflection loop).

Five scenarios:
  1. test_scaffolder_generates_initial_script  — first pass, no feedback → script
  2. test_scaffolder_revises_with_feedback     — second pass, feedback present → revised
  3. test_critic_approves_clean_script         — clean script → critic_approved True
  4. test_reflection_loop_converges            — critic rejects pass 1, approves pass 2
  5. test_reflection_cap_proceeds_unapproved   — critic always rejects, cap exits gracefully
"""
from __future__ import annotations

import pytest

from app.graph.nodes.critic import critic_node
from app.graph.nodes.scaffolder import scaffolder_node
from app.streaming.events import emitter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TEST_PLAN = [
    {
        "step_id": "step-001",
        "intent":  "navigate to the login page URL",
        "action":  "Open the target URL in the browser",
        "expected": "Login form is visible",
    },
    {
        "step_id": "step-002",
        "intent":  "enter email address in the login form email field",
        "action":  "Click the email input and type a valid email",
        "expected": "Email field shows the typed email",
    },
    {
        "step_id": "step-003",
        "intent":  "click the submit button to sign in",
        "action":  "Click the Sign in button",
        "expected": "User is redirected to the dashboard",
    },
]

_LOCATORS = {
    "step-001": {"css": "",        "xpath": "",                          "confidence": 0.0},
    "step-002": {"css": "#email",  "xpath": "//input[@id='email']",      "confidence": 0.95},
    "step-003": {"css": "#submit", "xpath": "//button[@id='submit']",    "confidence": 0.90},
}

_MOCK_SCRIPT = """\
from playwright.sync_api import Page, expect

def test_login_flow(page: Page) -> None:
    page.goto("https://example.com/login")
    # LOW-CONFIDENCE LOCATOR (confidence=0.00) — may need self-healing
    expect(page.locator("h1")).to_be_visible()
    page.locator("#email").fill("user@example.com")
    expect(page.locator("#email")).to_have_value("user@example.com")
    page.locator("#submit").click()
    expect(page).to_have_url("https://example.com/dashboard")
"""

_MOCK_SCRIPT_REVISED = """\
from playwright.sync_api import Page, expect

def test_login_flow(page: Page) -> None:
    page.goto("https://example.com/login")
    # LOW-CONFIDENCE LOCATOR (confidence=0.00) — may need self-healing
    expect(page.locator("h1")).to_be_visible()
    page.locator("#email").fill("user@example.com")
    expect(page.locator("#email")).to_have_value("user@example.com")
    page.locator("#submit").click()
    expect(page).to_have_url("https://example.com/dashboard")
    expect(page.locator(".dashboard-header")).to_be_visible()
"""

_CRITIC_REJECT = (
    '{"approved": false, "feedback": ['
    '{"issue": "Missing post-login assertion", "severity": "high",'
    ' "suggestion": "Add expect() on dashboard element after redirect"}'
    ']}'
)
_CRITIC_APPROVE = '{"approved": true, "feedback": []}'


def _make_llm_mock(text: str):
    """Return an async mock that always returns *text*."""
    from app.llm.client import LLMResponse

    async def _mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        return LLMResponse(
            text=text, input_tokens=100, output_tokens=200,
            model="mock-reasoning", cost_usd=0.0,
        )

    return _mock


def _make_sequential_mock(*texts: str):
    """Return an async mock that serves *texts* in order; repeats the last one."""
    from app.llm.client import LLMResponse
    seq = list(texts)
    state = {"idx": 0}

    async def _mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        i = state["idx"]
        text = seq[i] if i < len(seq) else seq[-1]
        state["idx"] = i + 1
        return LLMResponse(
            text=text, input_tokens=100, output_tokens=200,
            model="mock-reasoning", cost_usd=0.0,
        )

    return _mock


def _make_scaffolder_or_critic_mock(script_text: str, critic_text: str):
    """Route by user-message content: scaffolder user contains 'TEST PLAN:'."""
    from app.llm.client import LLMResponse

    async def _mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        text = script_text if "TEST PLAN:" in user else critic_text
        return LLMResponse(
            text=text, input_tokens=100, output_tokens=200,
            model="mock-reasoning", cost_usd=0.0,
        )

    return _mock


# ---------------------------------------------------------------------------
# 1. Scaffolder generates initial script
# ---------------------------------------------------------------------------

async def test_scaffolder_generates_initial_script(monkeypatch):
    """First scaffolder pass should produce a non-empty script and set reflection_count=1."""
    from app.llm import client as llm_mod

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_MOCK_SCRIPT))

    run_id = "se-scaffolder-init"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "status": "running",
    }

    result = await scaffolder_node(state)

    assert result["script"], "script must be non-empty"
    assert result["reflection_count"] == 1
    # Fence-stripping: LLM output without fences is returned verbatim (stripped).
    assert "def test_" in result["script"]
    await emitter.close(run_id)


# ---------------------------------------------------------------------------
# 2. Scaffolder revises when feedback is present
# ---------------------------------------------------------------------------

async def test_scaffolder_revises_with_feedback(monkeypatch):
    """Second scaffolder pass (with critic_feedback + existing script) increments count."""
    from app.llm import client as llm_mod

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_MOCK_SCRIPT_REVISED))

    run_id = "se-scaffolder-rev"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "script": _MOCK_SCRIPT,
        "reflection_count": 1,
        "critic_feedback": [
            {"issue": "Missing post-login assertion", "severity": "high",
             "suggestion": "Add expect() on dashboard element"},
        ],
        "status": "running",
    }

    result = await scaffolder_node(state)

    assert result["script"], "revised script must be non-empty"
    assert result["reflection_count"] == 2
    assert "def test_" in result["script"]
    await emitter.close(run_id)


# ---------------------------------------------------------------------------
# 3. Critic approves a clean script
# ---------------------------------------------------------------------------

async def test_critic_approves_clean_script(monkeypatch):
    """Critic should set critic_approved=True and return empty feedback when LLM approves."""
    from app.llm import client as llm_mod

    monkeypatch.setattr(llm_mod.llm_client, "complete", _make_llm_mock(_CRITIC_APPROVE))

    run_id = "se-critic-approve"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "script": _MOCK_SCRIPT,
        "reflection_count": 1,
        "status": "running",
    }

    result = await critic_node(state)

    assert result["critic_approved"] is True
    assert result["critic_feedback"] == []
    await emitter.close(run_id)


# ---------------------------------------------------------------------------
# 4. Full reflection loop: reject on pass 1, approve on pass 2
# ---------------------------------------------------------------------------

async def test_reflection_loop_converges(monkeypatch):
    """scaffolder→critic runs twice; critic rejects pass 1, approves pass 2.

    Calls are routed by user-message prefix so one monkeypatch covers the loop.
    """
    from app.llm import client as llm_mod

    # Round 1: scaffolder returns script, critic rejects.
    # Round 2: scaffolder returns revised script, critic approves.
    monkeypatch.setattr(
        llm_mod.llm_client, "complete",
        _make_sequential_mock(
            _MOCK_SCRIPT,          # scaffolder pass 1
            _CRITIC_REJECT,        # critic pass 1
            _MOCK_SCRIPT_REVISED,  # scaffolder pass 2
            _CRITIC_APPROVE,       # critic pass 2
        ),
    )

    run_id = "se-loop-converge"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "status": "running",
    }

    # --- Pass 1 ---
    result = await scaffolder_node(state)
    state.update(result)
    assert state["reflection_count"] == 1
    assert state["script"]

    result = await critic_node(state)
    state.update(result)
    assert state["critic_approved"] is False
    assert state["critic_feedback"]

    # Router would route back to scaffolder (1 < 3).
    from app.graph.builder import route_after_critic
    assert route_after_critic(state) == "scaffolder"

    # --- Pass 2 ---
    result = await scaffolder_node(state)
    state.update(result)
    assert state["reflection_count"] == 2

    result = await critic_node(state)
    state.update(result)
    assert state["critic_approved"] is True
    assert state["script"]  # non-empty after approval

    # Router should now proceed to execution.
    assert route_after_critic(state) == "execution"

    await emitter.close(run_id)


# ---------------------------------------------------------------------------
# 5. Reflection cap: critic never approves, loop exits gracefully
# ---------------------------------------------------------------------------

async def test_reflection_cap_proceeds_unapproved(monkeypatch):
    """When critic never approves, route_after_critic returns 'execution' at the cap.

    Uses default max_reflection_loops=3, simulating three full rounds.
    """
    from app.llm import client as llm_mod

    monkeypatch.setattr(
        llm_mod.llm_client, "complete",
        _make_scaffolder_or_critic_mock(
            script_text=_MOCK_SCRIPT,
            critic_text=_CRITIC_REJECT,
        ),
    )

    run_id = "se-cap"
    state: dict = {
        "run_id": run_id,
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "status": "running",
    }

    # Simulate 3 full rounds (default max_reflection_loops=3).
    for _ in range(3):
        result = await scaffolder_node(state)
        state.update(result)
        result = await critic_node(state)
        state.update(result)

    # Script must be non-empty and critic_approved set (False here).
    assert state["script"], "script must be non-empty even when unapproved"
    assert "critic_approved" in state
    assert state["critic_approved"] is False
    assert state["reflection_count"] == 3

    # Router must exit the loop: 3 >= max_reflection_loops(3).
    from app.graph.builder import route_after_critic
    assert route_after_critic(state) == "execution"

    await emitter.close(run_id)


# ---------------------------------------------------------------------------
# 6. target_url is passed into the LLM prompt
# ---------------------------------------------------------------------------

async def test_scaffolder_prompt_includes_target_url(monkeypatch):
    """target_url from state must appear verbatim in the user message sent to the LLM.

    Without this the model invents placeholder URLs (example.com, /dashboard).
    """
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    captured_messages: list = []

    async def _capture(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        captured_messages.extend(messages)
        return LLMResponse(
            text=_MOCK_SCRIPT, input_tokens=100, output_tokens=200,
            model="mock", cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _capture)

    target = "http://myapp.local:3000/login"
    state: dict = {
        "run_id": "se-url-in-prompt",
        "test_plan": _TEST_PLAN,
        "locators": _LOCATORS,
        "target_url": target,
        "status": "running",
    }

    await scaffolder_node(state)
    await emitter.close("se-url-in-prompt")

    user_msgs = [m for m in captured_messages if m["role"] == "user"]
    assert user_msgs, "scaffolder must send at least one user message to the LLM"
    user_text = user_msgs[0]["content"]
    assert target in user_text, (
        f"target_url {target!r} must appear verbatim in the LLM user prompt; "
        f"got first 300 chars: {user_text[:300]!r}"
    )


# ---------------------------------------------------------------------------
# 7. System prompt forbids invented URLs
# ---------------------------------------------------------------------------

def test_scaffolder_system_prompt_forbids_invented_urls():
    """SCAFFOLDER_SYSTEM must explicitly instruct the model to use the real URL,
    not invent placeholder addresses like example.com or /dashboard."""
    from app.llm.prompts.scaffolder import SCAFFOLDER_SYSTEM

    lowered = SCAFFOLDER_SYSTEM.lower()
    assert "target_url" in lowered or "target url" in lowered, (
        "System prompt must reference 'target_url' so the model knows what URL to use"
    )
    assert any(phrase in lowered for phrase in ("do not invent", "not invent", "do not guess", "do not assert")), (
        "System prompt must explicitly forbid inventing URLs for unspecified pages"
    )
