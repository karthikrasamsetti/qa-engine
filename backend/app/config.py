"""Application configuration and model routing.

Reads from environment / .env.  The model router maps a task tier to a concrete
model ID; cost computation uses the PRICING table.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ModelTier = Literal["reasoning", "fast"]
LlmProvider = Literal["openai", "anthropic"]

# ---------------------------------------------------------------------------
# Pricing (USD per 1 M tokens). Add new models here; unknown models get $0 cost
# and the client emits a warning event rather than crashing.
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-8":              {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":            {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":    {"input":  0.25, "output":  1.25},
    # OpenAI
    "gpt-4o":                       {"input":  5.00, "output": 15.00},
    "gpt-4o-mini":                  {"input":  0.15, "output":  0.60},
    "o3":                           {"input": 10.00, "output": 40.00},
    "o4-mini":                      {"input":  1.10, "output":  4.40},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    """Return (cost_usd, is_known).  Callers should warn when is_known=False."""
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0, False
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return cost, True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- provider selection ---
    llm_provider: LlmProvider = "anthropic"

    # --- Anthropic ---
    anthropic_api_key: str = ""
    model_reasoning: str = "claude-sonnet-4-6"
    model_fast: str = "claude-haiku-4-5-20251001"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model_reasoning: str = "gpt-4o"
    openai_model_fast: str = "gpt-4o-mini"

    # --- external integrations (stubbed in Stage A/B) ---
    jira_mcp_url: str = ""
    target_url: str = ""

    # --- limits / safety caps ---
    max_reflection_loops: int = 3
    max_heal_attempts: int = 3

    def model_for(self, tier: ModelTier) -> str:
        """Resolve a tier to the configured Anthropic model ID."""
        return self.model_reasoning if tier == "reasoning" else self.model_fast

    def openai_model_for(self, tier: ModelTier) -> str:
        """Resolve a tier to the configured OpenAI model ID."""
        return self.openai_model_reasoning if tier == "reasoning" else self.openai_model_fast


@lru_cache
def get_settings() -> Settings:
    return Settings()
