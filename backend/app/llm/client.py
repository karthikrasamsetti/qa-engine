"""Provider-agnostic LLM client factory.

Usage (nodes):
    from app.llm.client import llm_client, LLMResponse
    resp: LLMResponse = await llm_client.complete(
        messages=[{"role": "system", "content": SYS},
                  {"role": "user",   "content": USR}],
        model_tier="reasoning",
        run_id=run_id,          # optional — enables cost tracking & events
    )
    text = resp.text

Active provider is read from LLM_PROVIDER env ("anthropic" | "openai").
Default is "anthropic".  Swap by setting LLM_PROVIDER=openai in .env.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

from app.config import ModelTier, compute_cost, get_settings
from app.llm.cost_store import cost_store

logger = logging.getLogger(__name__)

Message = dict[str, Any]  # {"role": "system"|"user"|"assistant", "content": str|list}


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    tool_calls: list[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """All providers expose exactly this interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model_tier: ModelTier,
        *,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        run_id: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM; return a fully-populated LLMResponse.

        Args:
            messages:        Conversation turns.  System prompts use role="system".
            model_tier:      "reasoning" | "fast" — resolved to a concrete model
                             by the Settings router.
            response_format: OpenAI-style format hint ({"type": "json_object"} etc.).
                             Anthropic ignores this; use prompt engineering instead.
            tools:           OpenAI-style tool definitions.  Clients auto-convert to
                             the provider's native schema.
            max_tokens:      Maximum tokens to generate.
            run_id:          When provided, cost is tallied in CostStore and a
                             tool_result event is emitted to the run's SSE stream.
        """

    async def _record_cost(
        self,
        run_id: str | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute cost, store it, emit events.  Returns cost_usd."""
        cost, known = compute_cost(model, input_tokens, output_tokens)

        if run_id is not None:
            # Import lazily to avoid circular imports at module load time.
            from app.streaming.events import emitter
            running_total = cost_store.add(run_id, cost)
            if not known:
                await emitter.emit(
                    run_id, "System", 0, "decision",
                    f"Unknown pricing for model {model!r} — cost recorded as $0.00",
                )
            await emitter.emit(
                run_id, "System", 0, "tool_result",
                f"LLM {model}: ${cost:.6f} "
                f"(in={input_tokens} out={output_tokens}) | "
                f"run total: ${running_total:.4f}",
                data={
                    "model": model,
                    "cost_usd": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "run_total_usd": running_total,
                },
            )
        return cost


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool defs to the Anthropic tool schema."""
    result = []
    for t in tools:
        if t.get("type") == "function":
            f = t["function"]
            result.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "input_schema": f.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            })
        else:
            result.append(t)  # already Anthropic-shaped
    return result


class AnthropicClient(LLMClient):
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
        messages: list[Message],
        model_tier: ModelTier,
        *,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        run_id: str | None = None,
    ) -> LLMResponse:
        model = self._model(model_tier)
        logger.info("AnthropicClient: tier=%s model=%s", model_tier, model)

        # Anthropic separates the system prompt from the turn list.
        system_parts: list[str] = []
        turns: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(str(m["content"]))
            else:
                turns.append({"role": m["role"], "content": m["content"]})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        resp = await self._get_client().messages.create(**kwargs)

        text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if hasattr(block, "text"):
                text = block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost = await self._record_cost(run_id, model, in_tok, out_tok)

        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=model,
            cost_usd=cost,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=get_settings().openai_api_key
            )
        return self._client

    def _model(self, tier: ModelTier) -> str:
        return get_settings().openai_model_for(tier)

    async def complete(
        self,
        messages: list[Message],
        model_tier: ModelTier,
        *,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        run_id: str | None = None,
    ) -> LLMResponse:
        model = self._model(model_tier)
        logger.info("OpenAIClient: tier=%s model=%s", model_tier, model)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools

        resp = await self._get_client().chat.completions.create(**kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        cost = await self._record_cost(run_id, model, in_tok, out_tok)

        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=model,
            cost_usd=cost,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# Factory + module-level singleton
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    provider = get_settings().llm_provider
    if provider == "openai":
        return OpenAIClient()
    return AnthropicClient()


# Nodes import this directly.
llm_client: LLMClient = get_llm_client()
