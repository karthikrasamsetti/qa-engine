"""Tests for per-tier LLM provider/model routing.

Covers:
  1. effective_provider falls back to llm_provider when no per-tier vars are set
  2. per-tier REASONING_PROVIDER / FAST_PROVIDER override llm_provider independently
  3. effective_model resolves to the right default for each provider/tier combination
  4. effective_model per-tier model override (REASONING_MODEL / FAST_MODEL)
  5. DispatchingClient routes reasoning and fast tiers to different backends
  6. DispatchingClient singleton (llm_client) is a DispatchingClient
"""
from __future__ import annotations

import pytest

from app.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**kw) -> Settings:
    """Create a Settings instance isolated from the .env file.

    pydantic-settings loads .env values at a higher priority than model defaults,
    so a .env file with REASONING_PROVIDER=anthropic would silently override the
    None default and break tests that exercise the llm_provider fallback path.
    Passing _env_file=None disables .env loading for these unit tests so they
    depend only on explicit constructor arguments and model defaults.
    """
    return Settings(
        _env_file=None,
        anthropic_api_key="test-anth",
        openai_api_key="test-oai",
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. effective_provider — fallback to llm_provider
# ---------------------------------------------------------------------------

def test_effective_provider_defaults_to_llm_provider_anthropic():
    s = _settings(llm_provider="anthropic")
    assert s.effective_provider("reasoning") == "anthropic"
    assert s.effective_provider("fast") == "anthropic"


def test_effective_provider_defaults_to_llm_provider_openai():
    s = _settings(llm_provider="openai")
    assert s.effective_provider("reasoning") == "openai"
    assert s.effective_provider("fast") == "openai"


# ---------------------------------------------------------------------------
# 2. effective_provider — per-tier overrides
# ---------------------------------------------------------------------------

def test_reasoning_provider_overrides_llm_provider():
    s = _settings(llm_provider="anthropic", reasoning_provider="openai")
    assert s.effective_provider("reasoning") == "openai"
    # fast tier must still see the llm_provider fallback
    assert s.effective_provider("fast") == "anthropic"


def test_fast_provider_overrides_llm_provider():
    s = _settings(llm_provider="anthropic", fast_provider="openai")
    assert s.effective_provider("fast") == "openai"
    assert s.effective_provider("reasoning") == "anthropic"


def test_both_tiers_can_use_different_providers():
    s = _settings(
        llm_provider="anthropic",  # irrelevant when both overridden
        reasoning_provider="anthropic",
        fast_provider="openai",
    )
    assert s.effective_provider("reasoning") == "anthropic"
    assert s.effective_provider("fast") == "openai"


# ---------------------------------------------------------------------------
# 3. effective_model — default resolution by provider/tier
# ---------------------------------------------------------------------------

def test_effective_model_reasoning_anthropic_default():
    s = _settings(llm_provider="anthropic")
    assert s.effective_model("reasoning") == s.model_reasoning


def test_effective_model_fast_anthropic_default():
    s = _settings(llm_provider="anthropic")
    assert s.effective_model("fast") == s.model_fast


def test_effective_model_reasoning_openai_default():
    s = _settings(llm_provider="openai")
    assert s.effective_model("reasoning") == s.openai_model_reasoning


def test_effective_model_fast_openai_default():
    s = _settings(llm_provider="openai")
    assert s.effective_model("fast") == s.openai_model_fast


def test_effective_model_mixed_providers():
    # reasoning → Anthropic, fast → OpenAI
    s = _settings(
        llm_provider="anthropic",
        reasoning_provider="anthropic",
        fast_provider="openai",
    )
    assert s.effective_model("reasoning") == s.model_reasoning
    assert s.effective_model("fast") == s.openai_model_fast


# ---------------------------------------------------------------------------
# 4. effective_model — explicit per-tier model override
# ---------------------------------------------------------------------------

def test_reasoning_model_override():
    s = _settings(llm_provider="anthropic", reasoning_model="my-custom-model")
    assert s.effective_model("reasoning") == "my-custom-model"
    # fast must still use the default
    assert s.effective_model("fast") == s.model_fast


def test_fast_model_override():
    s = _settings(llm_provider="openai", fast_model="gpt-4o-mini-special")
    assert s.effective_model("fast") == "gpt-4o-mini-special"
    assert s.effective_model("reasoning") == s.openai_model_reasoning


def test_both_model_overrides():
    s = _settings(
        llm_provider="anthropic",
        reasoning_model="r-model",
        fast_model="f-model",
    )
    assert s.effective_model("reasoning") == "r-model"
    assert s.effective_model("fast") == "f-model"


# ---------------------------------------------------------------------------
# 5. DispatchingClient routes tiers to the correct backend
# ---------------------------------------------------------------------------

async def test_dispatching_client_routes_reasoning_and_fast_independently(monkeypatch):
    """DispatchingClient must route each tier to the provider configured for that tier.

    We monkeypatch get_settings() to return split-provider config, then intercept
    the backend .complete() calls and verify provider identity.
    """
    from app.llm.client import DispatchingClient, LLMResponse
    from app.config import get_settings

    reasoning_calls: list[str] = []
    fast_calls: list[str] = []

    class _FakeAnthropicClient:
        async def complete(self, messages, model_tier, **kw):
            reasoning_calls.append(model_tier)
            return LLMResponse(text="anthr", input_tokens=1, output_tokens=1,
                               model="anthr-model", cost_usd=0.0)

    class _FakeOpenAIClient:
        async def complete(self, messages, model_tier, **kw):
            fast_calls.append(model_tier)
            return LLMResponse(text="oai", input_tokens=1, output_tokens=1,
                               model="oai-model", cost_usd=0.0)

    split_settings = _settings(
        llm_provider="anthropic",
        reasoning_provider="anthropic",
        fast_provider="openai",
    )

    monkeypatch.setattr("app.llm.client.get_settings", lambda: split_settings)

    client = DispatchingClient()
    # Inject fake backends directly — bypasses real API calls.
    client._backends["anthropic"] = _FakeAnthropicClient()
    client._backends["openai"] = _FakeOpenAIClient()

    msgs = [{"role": "user", "content": "hello"}]
    await client.complete(msgs, "reasoning")
    await client.complete(msgs, "fast")

    assert reasoning_calls == ["reasoning"], (
        "reasoning tier must route to Anthropic backend"
    )
    assert fast_calls == ["fast"], (
        "fast tier must route to OpenAI backend"
    )


# ---------------------------------------------------------------------------
# 6. Module-level singleton is a DispatchingClient
# ---------------------------------------------------------------------------

def test_llm_client_singleton_is_dispatching_client():
    from app.llm.client import llm_client, DispatchingClient
    assert isinstance(llm_client, DispatchingClient), (
        f"llm_client must be a DispatchingClient, got {type(llm_client).__name__}"
    )
