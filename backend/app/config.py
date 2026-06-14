"""Application configuration and model routing.

Reads from environment / .env. The model router maps a task tier to a concrete
Anthropic model so reasoning-heavy nodes use a frontier model and cheap parsing
nodes use a fast model (resource-aware optimization).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ModelTier = Literal["reasoning", "fast"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""

    # Model routing: reasoning tasks (INVEST, Critic, Synthesis) vs.
    # fast/cheap tasks (DOM parsing, extraction).
    model_reasoning: str = "claude-sonnet-4-6"
    model_fast: str = "claude-haiku-4-5-20251001"

    # External integrations (stubbed in Stage A/B).
    jira_mcp_url: str = ""
    target_url: str = ""

    # Limits / safety caps used by later stages.
    max_reflection_loops: int = 3
    max_heal_attempts: int = 3

    def model_for(self, tier: ModelTier) -> str:
        return self.model_reasoning if tier == "reasoning" else self.model_fast


@lru_cache
def get_settings() -> Settings:
    return Settings()
