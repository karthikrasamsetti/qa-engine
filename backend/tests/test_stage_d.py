"""Stage D tests: analyst node + UI mapper node (Playwright + LLM factory).

Two scenarios:
  1. test_analyst_generates_plan   — mocked LLM returns JSON array; assert test_plan shape
  2. test_ui_mapper_finds_locators — real headless Chromium + smart locator mock;
                                     assert all four login-form elements have confidence > 0.5
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.graph.nodes.analyst import analyst_node
from app.graph.nodes.ui_mapper import ui_mapper_node
from app.streaming.events import emitter

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------

_ANALYST_PLAN = """[
  {"step_id": "step-001", "intent": "navigate to the login page URL",
   "action": "Open the target URL in the browser", "expected": "Login form is visible"},
  {"step_id": "step-002", "intent": "enter email address in login form email field",
   "action": "Click the email input and type a valid email", "expected": "Email field shows the typed email"},
  {"step_id": "step-003", "intent": "enter password in the password field",
   "action": "Click the password input and type the password", "expected": "Password field shows masked input"},
  {"step_id": "step-004", "intent": "click submit button to sign in",
   "action": "Click the Sign in button", "expected": "User is redirected to dashboard"}
]"""

# ---------------------------------------------------------------------------
# Local HTTP server fixture
# ---------------------------------------------------------------------------

@contextmanager
def _http_server(directory: str):
    """Spin up a local HTTP server serving *directory* on a random port."""
    class _Silent(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

    handler = functools.partial(_Silent, directory=directory)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_analyst_generates_plan(monkeypatch):
    """Analyst node should parse LLM's JSON array into a structured test_plan."""
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        return LLMResponse(
            text=_ANALYST_PLAN,
            input_tokens=300, output_tokens=150,
            model="mock-reasoning", cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _mock)

    run_id = "sd-analyst-1"
    collected: list = []

    async def collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(collect())

    result = await analyst_node({
        "run_id": run_id,
        "sanitized_story": "<story>As a user I want to log in</story>",
        "raw_input": "As a user I want to log in",
        "status": "running",
    })

    await emitter.close(run_id)
    await consumer

    test_plan: list = result.get("test_plan", [])
    assert len(test_plan) == 4, f"Expected 4 steps, got {len(test_plan)}"

    for step in test_plan:
        assert "step_id"  in step, "step_id missing"
        assert "intent"   in step, "intent missing"
        assert "action"   in step, "action missing"
        assert "expected" in step, "expected missing"

    assert test_plan[0]["step_id"] == "step-001"

    agents = {e.agent for e in collected}
    assert "Requirements Analyst" in agents

    decisions = [e for e in collected if e.type == "decision"]
    assert decisions, "analyst must emit at least one decision event"
    assert decisions[-1].data.get("step_count") == 4


async def test_ui_mapper_finds_locators(monkeypatch):
    """UI Mapper with real headless Chromium against login.html fixture.

    The LLM is mocked with a smart stub that returns the correct CSS selector
    based on intent keywords. Playwright verifies each selector exists on the
    page, keeping confidence >= 0.9.
    """
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _locator_mock(
        messages, model_tier, *,
        response_format=None, tools=None, max_tokens=2048, run_id=None,
    ):
        user = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        ).lower()
        if "email" in user:
            text = '{"css": "#email", "xpath": "//input[@id=\'email\']", "confidence": 0.95}'
        elif "password" in user:
            text = '{"css": "#password", "xpath": "//input[@id=\'password\']", "confidence": 0.95}'
        elif "submit" in user or "sign in" in user:
            text = '{"css": "#submit", "xpath": "//button[@id=\'submit\']", "confidence": 0.95}'
        elif "forgot" in user:
            text = '{"css": "#forgot-password", "xpath": "//a[@id=\'forgot-password\']", "confidence": 0.90}'
        else:
            text = '{"css": "", "xpath": "", "confidence": 0.0}'
        return LLMResponse(
            text=text, input_tokens=50, output_tokens=30,
            model="mock-fast", cost_usd=0.0,
        )

    monkeypatch.setattr(llm_mod.llm_client, "complete", _locator_mock)

    test_plan = [
        {
            "step_id": "step-001",
            "intent":  "enter email address in login form email field",
            "action":  "click email field and type email",
            "expected": "email field shows the typed email",
        },
        {
            "step_id": "step-002",
            "intent":  "enter password in the password field",
            "action":  "click password field and type password",
            "expected": "password field shows masked input",
        },
        {
            "step_id": "step-003",
            "intent":  "click submit button to sign in",
            "action":  "click the Sign in submit button",
            "expected": "user is redirected to dashboard",
        },
        {
            "step_id": "step-004",
            "intent":  "click forgot password link below login form",
            "action":  "click the Forgot password? anchor link",
            "expected": "password reset form is displayed",
        },
    ]

    run_id = "sd-uimapper-1"
    collected: list = []

    async def collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    with _http_server(str(FIXTURES_DIR)) as server_url:
        consumer = asyncio.create_task(collect())

        result = await ui_mapper_node({
            "run_id":     run_id,
            "test_plan":  test_plan,
            "target_url": f"{server_url}/login.html",
            "status":     "running",
        })

        await emitter.close(run_id)
        await consumer

    locators: dict = result.get("locators", {})

    assert len(locators) == 4, f"Expected 4 locators, got {len(locators)}: {locators}"

    for step_id, loc in locators.items():
        conf = loc.get("confidence", 0.0)
        css  = loc.get("css", "")
        assert conf > 0.5, (
            f"Step {step_id}: confidence {conf:.2f} is too low "
            f"(css={css!r}). Selector may not have resolved on the page."
        )

    # Check event stream: one tool_call and one tool_result per step
    tool_calls   = [e for e in collected if e.type == "tool_call"]
    tool_results = [e for e in collected if e.type == "tool_result"]
    assert len(tool_calls)   == 4, f"Expected 4 tool_call events, got {len(tool_calls)}"
    assert len(tool_results) >= 4, f"Expected >=4 tool_result events, got {len(tool_results)}"

    # Verify step IDs are present in the locator dict
    for step in test_plan:
        assert step["step_id"] in locators, f"{step['step_id']} missing from locators"
