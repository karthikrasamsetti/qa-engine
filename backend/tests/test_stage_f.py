"""Stage F tests: execution node + self_heal node + sandbox helpers.

Five scenarios:
  1. test_execution_passes                — sandbox passes → execution_result.passed
  2. test_execution_emits_error_on_fail   — sandbox fails  → error event emitted
  3. test_self_heal_patches_stale_locator — locator failure → re-scan → patch → execution passes
  4. test_assertion_failure_not_healed    — AssertionError  → heal skipped, cap forced
  5. test_heal_cap_escalates_to_hitl      — 3rd heal attempt → hitl_request + GraphInterrupt
"""
from __future__ import annotations

import asyncio

import pytest

from app.graph.builder import route_after_execution
from app.graph.nodes.execution import execution_node
from app.graph.nodes.self_heal import self_heal_node
from app.streaming.events import emitter
from app.tools.sandbox import is_locator_failure, is_assertion_failure, substitute_url, _rewrite_for_container

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SCRIPT = """\
from playwright.sync_api import Page, expect

def test_login(page: Page) -> None:
    page.goto("https://example.com/login")
    page.locator("#email").fill("user@example.com")
    page.locator("#submit").click()
    expect(page).to_have_url("https://example.com/dashboard")
"""

_LOCATORS = {
    "step-001": {"css": "",        "xpath": "",                     "confidence": 0.0},
    "step-002": {"css": "#email",  "xpath": "//input[@id='email']", "confidence": 0.95},
    "step-003": {"css": "#submit", "xpath": "//button[@id='submit']","confidence": 0.90},
}

_TEST_PLAN = [
    {"step_id": "step-001", "intent": "navigate to the login page URL",          "action": "Open URL", "expected": "Login form visible"},
    {"step_id": "step-002", "intent": "enter email address in email field",       "action": "Fill email", "expected": "Email shown"},
    {"step_id": "step-003", "intent": "click the submit button to sign in",       "action": "Click submit", "expected": "Dashboard loaded"},
]

_PASS_RESULT = {"passed": True,  "logs": "1 passed",          "error": None,           "screenshots": [], "exit_code": 0}
_LOCATOR_FAIL = {
    "passed": False, "exit_code": 1, "screenshots": [],
    "logs": "FAILED test_login.py::test_login\nTimeoutError: Timeout 30000ms exceeded.\n  waiting for locator('#submit')",
    "error": "TimeoutError: Timeout 30000ms exceeded.\n  waiting for locator('#submit')",
}
_ASSERT_FAIL = {
    "passed": False, "exit_code": 1, "screenshots": [],
    "logs": "FAILED test_login.py::test_login\nAssertionError: Locator expected to have URL",
    "error": "AssertionError: Locator expected to have URL 'https://example.com/dashboard'",
}

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_sandbox(result: dict):
    """Return an async function that replaces run_script."""
    async def _run(script, target_url="", timeout=120):
        return dict(result)
    return _run


class _MockBrowserSession:
    """Async context manager that returns *locator* from find_locators."""
    def __init__(self, locator: dict):
        self._loc = locator

    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass

    async def find_locators(self, url, intent, run_id=None):
        return dict(self._loc)


def _base_state(run_id: str, **extra) -> dict:
    return {
        "run_id": run_id,
        "status": "running",
        "script": _SCRIPT,
        "test_plan": _TEST_PLAN,
        "locators": dict(_LOCATORS),
        "target_url": "http://example.com",
        **extra,
    }


# ---------------------------------------------------------------------------
# sandbox.py unit tests
# ---------------------------------------------------------------------------

def test_substitute_url_replaces_example_com():
    script = 'page.goto("https://example.com/login")'
    result = substitute_url(script, "http://myapp.local/login")
    assert "myapp.local" in result
    assert "example.com" not in result


def test_substitute_url_replaces_caps_placeholder():
    script = "page.goto('LOGIN_PAGE_URL')"
    result = substitute_url(script, "http://myapp.local")
    assert "myapp.local" in result


def test_substitute_url_ignores_real_url():
    # A proper URL that is NOT a placeholder should not be touched.
    script = 'page.goto("http://myapp.local/login")'
    result = substitute_url(script, "http://other.com")
    # The regex only replaces placeholder patterns, not real URLs.
    assert "myapp.local" in result


def test_is_locator_failure_timeout():
    assert is_locator_failure("TimeoutError: Timeout 30000ms exceeded. waiting for locator('#submit')")


def test_is_locator_failure_strict_mode():
    assert is_locator_failure("strict mode violation: locator('#btn') resolved to 3 elements")


def test_is_locator_failure_assertion_excluded():
    assert not is_locator_failure("AssertionError: Locator expected to have URL")


def test_is_locator_failure_expect_excluded():
    assert not is_locator_failure("Locator expected to be visible")


# ---------------------------------------------------------------------------
# sandbox.py: localhost → host.docker.internal rewrite
# ---------------------------------------------------------------------------

def test_rewrite_for_container_localhost_with_port():
    assert _rewrite_for_container("http://localhost:3000/login") == "http://host.docker.internal:3000/login"


def test_rewrite_for_container_127_with_port():
    assert _rewrite_for_container("http://127.0.0.1:3000/") == "http://host.docker.internal:3000/"


def test_rewrite_for_container_localhost_no_port():
    assert _rewrite_for_container("http://localhost/app") == "http://host.docker.internal/app"


def test_rewrite_for_container_leaves_real_host_unchanged():
    url = "http://staging.myapp.com/login"
    assert _rewrite_for_container(url) == url


def test_rewrite_for_container_empty_string():
    assert _rewrite_for_container("") == ""


def test_substitute_url_rewrites_localhost_for_container_execution():
    """substitute_url called with a container-rewritten URL must produce host.docker.internal in the script."""
    script = 'page.goto("https://example.com/login")'
    container_url = _rewrite_for_container("http://localhost:3000/login")
    result = substitute_url(script, container_url)
    assert "host.docker.internal" in result


# ---------------------------------------------------------------------------
# sandbox.py: is_assertion_failure helper
# ---------------------------------------------------------------------------

def test_is_assertion_failure_detects_assertion_error():
    assert is_assertion_failure("AssertionError: expected True but got False")


def test_is_assertion_failure_detects_locator_expected():
    assert is_assertion_failure("Locator expected to have URL 'http://example.com/dashboard'")


def test_is_assertion_failure_detects_expected_to_have():
    assert is_assertion_failure("Expected page to have title 'Dashboard'")


def test_is_assertion_failure_false_for_timeout():
    assert not is_assertion_failure("TimeoutError: Timeout 30000ms exceeded.")


def test_is_assertion_failure_false_for_sandbox_timeout():
    # The string returned by run_script() on asyncio.TimeoutError.
    assert not is_assertion_failure("Sandbox timed out after 120s.")


def test_is_assertion_failure_false_for_docker_unavailable():
    assert not is_assertion_failure("Docker is not available on this host.")


# ---------------------------------------------------------------------------
# self_heal: sandbox-level timeout must not be labelled 'assertion failure'
# ---------------------------------------------------------------------------

_ENV_TIMEOUT_RESULT = {
    "passed": False, "exit_code": -1, "screenshots": [],
    "logs": "",
    "error": "Sandbox timed out after 120s.",
}


async def test_self_heal_sandbox_timeout_emits_environment_error(monkeypatch):
    """A sandbox-level timeout (not a Playwright TimeoutError) must be surfaced as an
    environment error — self-heal must not label it a 'genuine assertion error' and
    must not open a BrowserSession."""
    called = {"browser": False}

    class _ShouldNotBeUsed:
        async def __aenter__(self): called["browser"] = True; return self
        async def __aexit__(self, *_): pass
        async def find_locators(self, *a, **kw):
            return {"css": "#anything", "xpath": "", "confidence": 0.9}

    monkeypatch.setattr("app.graph.nodes.self_heal.BrowserSession", _ShouldNotBeUsed)

    run_id = "sf-env-timeout"
    state = _base_state(run_id, execution_result=dict(_ENV_TIMEOUT_RESULT), heal_attempts=0)

    collected: list = []

    async def _collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(_collect())
    result = await self_heal_node(state)
    await emitter.close(run_id)
    await consumer

    # Must not have opened a BrowserSession.
    assert not called["browser"], "BrowserSession must not be called for a sandbox timeout"

    # heal_attempts forced to cap (unrecoverable environment error).
    from app.config import get_settings
    assert result["heal_attempts"] >= get_settings().max_heal_attempts

    # The emitted event must NOT say "assertion error" — it's an environment error.
    decision_events = [e for e in collected if e.type in ("decision", "error")]
    assert decision_events, "self_heal must emit a decision/error event for env timeout"
    combined_text = " ".join(e.message for e in decision_events).lower()
    assert "assertion" not in combined_text, (
        f"Event message must not label sandbox timeout as assertion failure; got: {combined_text!r}"
    )


# ---------------------------------------------------------------------------
# 1. Execution passes
# ---------------------------------------------------------------------------

async def test_execution_passes(monkeypatch):
    """execution_node must write execution_result with passed=True."""
    monkeypatch.setattr("app.graph.nodes.execution.run_script", _mock_sandbox(_PASS_RESULT))

    run_id = "sf-exec-pass"
    result = await execution_node(_base_state(run_id))
    await emitter.close(run_id)

    assert result["execution_result"]["passed"] is True
    assert result["execution_result"]["exit_code"] == 0


# ---------------------------------------------------------------------------
# 2. Execution emits error event on failure
# ---------------------------------------------------------------------------

async def test_execution_emits_error_on_fail(monkeypatch):
    """execution_node must emit an error event and set passed=False on sandbox failure."""
    monkeypatch.setattr("app.graph.nodes.execution.run_script", _mock_sandbox(_LOCATOR_FAIL))

    run_id = "sf-exec-fail"
    collected: list = []

    async def _collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(_collect())
    result = await execution_node(_base_state(run_id))
    await emitter.close(run_id)
    await consumer

    assert result["execution_result"]["passed"] is False
    error_events = [e for e in collected if e.type == "error"]
    assert error_events, "execution_node must emit an error event on failure"


# ---------------------------------------------------------------------------
# 3. Stale locator → self_heal patches → execution passes
# ---------------------------------------------------------------------------

async def test_self_heal_patches_stale_locator(monkeypatch):
    """Full locator-heal cycle: failure → re-scan → patch script → execution passes.

    Call order:
      execution_node (fail, locator error on #submit)
      → route_after_execution → "self_heal"
      → self_heal_node (re-scans, finds #new-submit, patches)
      → execution_node (passes)
      → route_after_execution → "synthesis"
    """
    # Sandbox: fail on first call (stale locator), pass on second (patched).
    call_count = {"n": 0}

    async def _mock_run(script, target_url="", timeout=120):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return dict(_LOCATOR_FAIL)
        # Second call: verify the script was patched before returning pass.
        assert "#new-submit" in script, "Script must be patched before second execution"
        return dict(_PASS_RESULT)

    monkeypatch.setattr("app.graph.nodes.execution.run_script", _mock_run)
    monkeypatch.setattr(
        "app.graph.nodes.self_heal.BrowserSession",
        lambda: _MockBrowserSession({"css": "#new-submit", "xpath": "//button[@id='new-submit']", "confidence": 0.95}),
    )

    run_id = "sf-heal-patch"
    state = _base_state(run_id)

    # --- execution pass 1: fails ---
    exec_result = await execution_node(state)
    state.update(exec_result)
    assert not state["execution_result"]["passed"]
    assert route_after_execution(state) == "self_heal"

    # --- self_heal: patches script ---
    heal_result = await self_heal_node(state)
    state.update(heal_result)
    assert "#new-submit" in state["script"], "self_heal must patch the script"
    assert state["locators"]["step-003"]["css"] == "#new-submit"
    assert state["heal_attempts"] == 1

    # --- execution pass 2: passes with patched script ---
    exec_result = await execution_node(state)
    state.update(exec_result)
    assert state["execution_result"]["passed"]
    assert route_after_execution(state) == "synthesis"

    await emitter.close(run_id)


# ---------------------------------------------------------------------------
# 4. Genuine assertion failure — self_heal must not patch the script
# ---------------------------------------------------------------------------

async def test_assertion_failure_not_healed(monkeypatch):
    """A real assertion failure must not be healed; self_heal forces the cap."""
    # BrowserSession should never be called for an assertion failure.
    called = {"browser": False}

    class _ShouldNotBeUsed:
        async def __aenter__(self): called["browser"] = True; return self
        async def __aexit__(self, *_): pass
        async def find_locators(self, *a, **kw):
            return {"css": "#anything", "xpath": "", "confidence": 0.9}

    monkeypatch.setattr("app.graph.nodes.self_heal.BrowserSession", _ShouldNotBeUsed)

    run_id = "sf-assert-fail"
    state = _base_state(run_id, execution_result=dict(_ASSERT_FAIL), heal_attempts=0)
    original_script = state["script"]

    result = await self_heal_node(state)
    state.update(result)
    await emitter.close(run_id)

    # Script must be unchanged.
    assert state["script"] == original_script, "assertion failure must not patch the script"
    # heal_attempts forced to cap so the loop exits next round.
    from app.config import get_settings
    assert state["heal_attempts"] >= get_settings().max_heal_attempts
    # BrowserSession must NOT have been opened.
    assert not called["browser"]


# ---------------------------------------------------------------------------
# 5. Heal cap → HITL escalation
# ---------------------------------------------------------------------------

async def test_heal_cap_escalates_to_hitl(monkeypatch):
    """At the heal cap self_heal must emit hitl_request and call interrupt().

    interrupt() needs LangGraph's runnable context to raise GraphInterrupt, so we
    monkeypatch it in self_heal to capture the call.  What we really want to assert
    is (a) interrupt() was called with the right payload and (b) a hitl_request event
    was emitted — the LangGraph context plumbing is an integration concern.
    """
    from app.config import get_settings

    settings = get_settings()
    at_cap_attempts = settings.max_heal_attempts - 1   # after +1 inside node → equals cap

    monkeypatch.setattr(
        "app.graph.nodes.self_heal.BrowserSession",
        lambda: _MockBrowserSession({"css": "", "xpath": "", "confidence": 0.0}),
    )

    interrupt_calls: list = []

    def _mock_interrupt(value):
        interrupt_calls.append(value)
        # Do NOT raise — we just capture. In the real graph LangGraph intercepts
        # the raise; here we verify the call happened and the event was emitted.

    monkeypatch.setattr("app.graph.nodes.self_heal.interrupt", _mock_interrupt)

    run_id = "sf-hitl-cap"
    state = _base_state(
        run_id,
        execution_result=dict(_LOCATOR_FAIL),
        heal_attempts=at_cap_attempts,
    )

    collected: list = []

    async def _collect():
        async for e in emitter.subscribe(run_id):
            collected.append(e)

    consumer = asyncio.create_task(_collect())
    result = await self_heal_node(state)
    await emitter.close(run_id)
    await consumer

    # interrupt() must have been called with a HITL payload.
    assert interrupt_calls, "self_heal must call interrupt() at the cap"
    assert "question" in interrupt_calls[0], "interrupt payload must include 'question'"

    # A hitl_request event must have been emitted before interrupt().
    hitl_events = [e for e in collected if e.type == "hitl_request"]
    assert hitl_events, "self_heal must emit a hitl_request event at the cap"
    assert "cap" in hitl_events[0].message.lower() or "heal" in hitl_events[0].message.lower()

    # heal_attempts must equal the cap in the returned state.
    assert result["heal_attempts"] == settings.max_heal_attempts
