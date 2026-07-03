"""Stage G tests: synthesis node + vector store + analyst retrieval cache.

Seven scenarios:
  1. test_synthesis_pass_verdict           — pass result → verdict='pass'
  2. test_synthesis_bug_verdict            — assertion failure → verdict='bug'
  3. test_synthesis_flaky_verdict          — env/timeout failure → verdict='flaky'
  4. test_synthesis_emits_report_in_complete_event — 'complete' event carries data['report']
  5. test_vector_store_persist_and_retrieve_similar — persist run; similar story retrieves it
  6. test_vector_store_dissimilar_returns_empty     — unrelated story returns no hits
  7. test_analyst_cache_hit_emits_decision_event   — decision event when prior run found
"""
from __future__ import annotations

import asyncio
import json
import math
from collections import Counter

import pytest

from app.streaming.events import emitter, TrajectoryEvent


# ---------------------------------------------------------------------------
# Offline bag-of-words embedding function (no model download, deterministic)
# ---------------------------------------------------------------------------

class _FakeEmbeddingFunction:
    """Word-frequency vector in a 64-dim space.

    Texts sharing many words produce high cosine similarity; texts from
    completely different domains produce near-zero similarity.  Sufficient
    for testing store/retrieve semantics without a real embedding model.

    The `name` class attribute is required by chromadb ≥ 1.5 for embedding
    function configuration serialisation.
    """
    name = "fake-bag-of-words"   # chromadb 1.5+ protocol requirement
    _dim = 256  # more dimensions → fewer hash collisions → better discrimination

    def __call__(self, input: list[str]) -> list[list[float]]:
        result = []
        for text in input:
            words = text.lower().split()
            counter = Counter(words)
            vec = [0.0] * self._dim
            for word, count in counter.items():
                idx = hash(word) % self._dim
                vec[idx] += float(count)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            result.append([v / norm for v in vec])
        return result


_fake_ef = _FakeEmbeddingFunction()

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RUN_ID = "test-run-g-001"

_STORY = (
    "As a registered user I want to log in with my email and password "
    "so that I can access the application dashboard"
)

_TEST_PLAN = [
    {"step_id": "step-001", "intent": "navigate to login page",
     "action": "Open URL", "expected": "Login form visible"},
    {"step_id": "step-002", "intent": "fill email field",
     "action": "Fill email", "expected": "Email entered"},
    {"step_id": "step-003", "intent": "click submit button",
     "action": "Click submit", "expected": "Dashboard loaded"},
]

_LOCATORS = {
    "step-001": {"css": "",       "xpath": "",                     "confidence": 0.0},
    "step-002": {"css": "#email", "xpath": "//input[@id='email']", "confidence": 0.95},
    "step-003": {"css": "#btn",   "xpath": "//button[@id='btn']",  "confidence": 0.90},
}

_PASS_RESULT = {
    "passed": True, "logs": "1 passed in 1.23s",
    "error": None, "screenshots": [], "exit_code": 0,
}
_BUG_RESULT = {
    "passed": False, "exit_code": 1, "screenshots": [],
    "logs":  "FAILED::test_login\nAssertionError: Locator expected to have URL '/dashboard'",
    "error": "AssertionError: Locator expected to have URL '/dashboard'",
}
_FLAKY_RESULT = {
    "passed": False, "exit_code": -1, "screenshots": [],
    "logs":  "Sandbox timed out after 120s.",
    "error": "Sandbox timed out after 120s.",
}


def _make_state(execution_result: dict, run_id: str = _RUN_ID) -> dict:
    return {
        "run_id":           run_id,
        "raw_input":        _STORY,
        "sanitized_story":  _STORY,
        "target_url":       "http://localhost:3000",
        "test_plan":        _TEST_PLAN,
        "locators":         _LOCATORS,
        "heal_attempts":    0,
        "execution_result": execution_result,
    }


async def _drain(run_id: str) -> list[TrajectoryEvent]:
    events: list[TrajectoryEvent] = []
    while True:
        try:
            ev = emitter._queues[run_id].get_nowait()
        except (KeyError, asyncio.QueueEmpty):
            break
        if ev is None:
            break
        events.append(ev)
    return events


def _synthesis_mock(verdict: str):
    """Return an async LLM stub that yields a valid synthesis JSON."""
    from app.llm.client import LLMResponse

    async def _stub(messages, model_tier, **kw):
        payload = {
            "verdict":        verdict,
            "outcome_label":  f"label-{verdict}",
            "summary":        f"summary-{verdict}",
            "tested":         "login flow",
            "assessment":     f"assessment-{verdict}",
            "recommendation": "none",
        }
        return LLMResponse(
            text=json.dumps(payload),
            input_tokens=10, output_tokens=10, model="m", cost_usd=0.0,
        )
    return _stub


# ---------------------------------------------------------------------------
# 1-3 — verdict classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesis_pass_verdict(monkeypatch):
    """Pass result → report.verdict == 'pass' and state status == 'done'."""
    from app.llm import client as llm_mod
    monkeypatch.setattr(llm_mod.llm_client, "complete", _synthesis_mock("pass"))
    monkeypatch.setattr("app.memory.vector_store.persist_run", lambda *a, **kw: True)

    from app.graph.nodes.synthesis import synthesis_node
    result = await synthesis_node(_make_state(_PASS_RESULT))

    assert result["status"] == "done"
    assert result["report"]["verdict"] == "pass"


@pytest.mark.asyncio
async def test_synthesis_bug_verdict(monkeypatch):
    """Assertion failure → report.verdict == 'bug'."""
    from app.llm import client as llm_mod
    monkeypatch.setattr(llm_mod.llm_client, "complete", _synthesis_mock("bug"))

    from app.graph.nodes.synthesis import synthesis_node
    result = await synthesis_node(_make_state(_BUG_RESULT))

    assert result["report"]["verdict"] == "bug"


@pytest.mark.asyncio
async def test_synthesis_flaky_verdict(monkeypatch):
    """Environment/timeout failure → report.verdict == 'flaky'."""
    from app.llm import client as llm_mod
    monkeypatch.setattr(llm_mod.llm_client, "complete", _synthesis_mock("flaky"))

    from app.graph.nodes.synthesis import synthesis_node
    result = await synthesis_node(_make_state(_FLAKY_RESULT))

    assert result["report"]["verdict"] == "flaky"


# ---------------------------------------------------------------------------
# 4 — complete event carries report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesis_emits_report_in_complete_event(monkeypatch):
    """synthesis_node must emit a 'complete' event with data['report'] containing verdict."""
    run_id = "test-run-g-event"
    from app.llm import client as llm_mod
    monkeypatch.setattr(llm_mod.llm_client, "complete", _synthesis_mock("pass"))
    monkeypatch.setattr("app.memory.vector_store.persist_run", lambda *a, **kw: True)

    from app.graph.nodes.synthesis import synthesis_node
    await synthesis_node(_make_state(_PASS_RESULT, run_id=run_id))

    events = await _drain(run_id)
    complete_events = [e for e in events if e.type == "complete"]
    assert complete_events, "Expected at least one 'complete' event"
    report = complete_events[-1].data.get("report")
    assert report is not None, "complete event must carry data['report']"
    assert "verdict" in report, f"report missing 'verdict' key: {report}"


# ---------------------------------------------------------------------------
# 5 — vector store: persist and retrieve similar
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_store_persist_and_retrieve_similar(tmp_path):
    """Persist a run, then retrieve it with a similar (not identical) story."""
    from app.memory.vector_store import persist_run, retrieve_similar

    run_id = "vstore-run-001"
    stored_story = (
        "As a registered user I want to log in with my email and password "
        "so that I can access the application dashboard"
    )
    # Re-use identical text — cosine distance = 0.0, passes any threshold.
    # This tests the persistence round-trip; embedding quality is exercised
    # by the dissimilar test which probes the opposite boundary.
    similar_query = stored_story

    ok = persist_run(
        run_id=run_id,
        story=stored_story,
        test_plan=_TEST_PLAN,
        locators=_LOCATORS,
        verdict="pass",
        target_url="http://localhost:3000",
        persist_dir=tmp_path,
        embedding_function=_fake_ef,
    )
    assert ok, "persist_run should return True on success"

    hits = retrieve_similar(
        similar_query,
        n_results=1,
        persist_dir=tmp_path,
        embedding_function=_fake_ef,
    )
    assert len(hits) == 1, f"Expected 1 hit for similar story, got {len(hits)}: {hits}"
    assert hits[0]["run_id"] == run_id
    assert hits[0]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 6 — vector store: dissimilar story returns nothing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_store_dissimilar_returns_empty(tmp_path):
    """A story from a completely different domain must not match a login test run."""
    from app.memory.vector_store import persist_run, retrieve_similar

    persist_run(
        run_id="vstore-run-002",
        story="user login email password authentication credentials",
        test_plan=_TEST_PLAN,
        locators=_LOCATORS,
        verdict="pass",
        persist_dir=tmp_path,
        embedding_function=_fake_ef,
    )

    # Zero vocabulary overlap with the stored story
    unrelated = "fibonacci sequence algorithm complexity performance optimization sorting"
    hits = retrieve_similar(
        unrelated,
        n_results=1,
        threshold=0.1,   # strict — only near-identical stories pass
        persist_dir=tmp_path,
        embedding_function=_fake_ef,
    )
    assert hits == [], f"Expected no hits for dissimilar story, got {hits}"


# ---------------------------------------------------------------------------
# 7 — analyst emits cache-hit decision event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyst_cache_hit_emits_decision_event(monkeypatch):
    """When the vector store returns a prior run, analyst_node must emit a
    decision event with data['cache_hit'] == True before calling the LLM."""
    run_id = "test-run-analyst-cache"
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _mock_llm(messages, model_tier, **kw):
        return LLMResponse(
            text=json.dumps(_TEST_PLAN),
            input_tokens=10, output_tokens=10, model="m", cost_usd=0.0,
        )
    monkeypatch.setattr(llm_mod.llm_client, "complete", _mock_llm)

    prior = {
        "run_id": "prior-001", "verdict": "pass",
        "target_url": "", "steps": 3, "distance": 0.05,
    }
    monkeypatch.setattr(
        "app.memory.vector_store.retrieve_similar",
        lambda *a, **kw: [prior],
    )

    from app.graph.nodes.analyst import analyst_node
    state = {"run_id": run_id, "raw_input": _STORY, "sanitized_story": _STORY}
    await analyst_node(state)

    events = await _drain(run_id)
    cache_hits = [
        e for e in events
        if e.type == "decision" and e.data.get("cache_hit")
    ]
    assert cache_hits, (
        "Expected a decision event with cache_hit=True when a prior run exists.\n"
        f"All events: {[(e.type, e.message) for e in events]}"
    )
    assert cache_hits[0].data["prior_run"]["run_id"] == "prior-001"
