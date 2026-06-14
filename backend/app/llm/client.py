"""Anthropic LLM wrapper with model-tier routing.

Usage:
    from app.llm.client import llm_client
    text = await llm_client.complete("reasoning", system=SYS, user=USR)
"""
from __future__ import annotations

import logging

import anthropic

from app.config import ModelTier, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=get_settings().anthropic_api_key
            )
        return self._client

    def _model(self, tier: ModelTier) -> str:
        return get_settings().model_for(tier)

    async def complete(
        self,
        tier: ModelTier,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> str:
        model = self._model(tier)
        logger.info("LLMClient.complete: tier=%s model=%s", tier, model)
        message = await self._get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


llm_client = LLMClient()
