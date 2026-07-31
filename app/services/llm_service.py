"""
Provider-agnostic LLM service with **cross-provider** fallback.

Ported from the Week 3 agent, with the one change that matters for a multi-agent workload:
Week 3 fell back across *models within one provider*. A rate-limited key therefore killed the
whole run. Here the fallback chain is ``(provider, model)`` pairs spanning every configured
provider, so an exhausted Groq quota continues on OpenRouter mid-workflow instead of failing
the run (§22, "Model API failure"). That behaviour is demonstrable — revoke a key mid-run and
the workflow completes, with the switch recorded in the trace.

Retained from Week 3 because each was earned the hard way:
- **choices: null guard** — some providers return an empty ``choices`` array; the resulting
  empty content is detected and raised as a clean :class:`LLMError` rather than crashing.
- **retry + backoff** on transient 429/timeout errors.
- **user-safe errors** — every failure maps to a short message; keys never leak into UI text.
- **JSON-prompt fallback** when a model's native function-calling refuses a nested schema.

Added here:
- **per-agent metering** — every call is recorded on a :class:`~app.services.usage.UsageTracker`
  with tokens, latency and provider, and the run budget is checked *before* each call.
- **per-agent model tiering** — ``agent_id`` selects the model via ``config.model_for_agent``.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import PROVIDERS, model_for_agent, settings
from app.services.usage import UsageTracker

T = TypeVar("T", bound=BaseModel)

# Substrings that indicate a transient, worth-retrying failure.
_TRANSIENT = ("429", "rate limit", "rate-limit", "timeout", "timed out", "temporarily",
              "overloaded", "502", "503", "504")


class LLMError(RuntimeError):
    """A user-safe LLM failure carrying a short, display-ready message."""


def _friendly(exc: Exception) -> LLMError:
    """Map any provider exception to a short, safe message (no keys, no stack traces)."""
    if isinstance(exc, LLMError):
        return exc
    msg = str(exc).lower()
    if "401" in msg or "invalid api key" in msg or "authentication" in msg:
        return LLMError("The AI provider rejected the API key. Check your credentials.")
    if "403" in msg or "permission" in msg:
        return LLMError("The AI provider denied access to this model (region/permission).")
    if "429" in msg or "rate" in msg or "quota" in msg:
        return LLMError("AI rate limit reached. Falling back to the next provider.")
    if "timeout" in msg or "timed out" in msg:
        return LLMError("The AI request timed out.")
    if "connection" in msg or "network" in msg or "getaddrinfo" in msg:
        return LLMError("Could not reach the AI provider. Check your connection.")
    if "not found" in msg or "404" in msg:
        return LLMError("That model id was not found at the provider.")
    if "did not match sc" in msg or "failed to call a function" in msg:
        return LLMError("The model could not produce the required structured output.")
    return LLMError("The AI service failed to produce a valid response.")


def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object from a model response (tolerates code fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise LLMError("Model did not return JSON.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMError("Model returned truncated JSON.")


def _tokens(resp) -> tuple[int, int]:
    meta = getattr(resp, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


class LLMService:
    """Thin wrapper over ``ChatOpenAI`` that spans providers.

    ``agent_id`` is not decoration: it selects the model tier and attributes every token to a
    role in the usage report, which is what makes the §29 per-agent cost table real rather
    than estimated.
    """

    def __init__(self, agent_id: str = "system", usage: UsageTracker | None = None):
        self.agent_id = agent_id
        self.usage = usage
        self.last_used_model: str | None = None
        self.last_used_provider: str | None = None

    # ------------------------------------------------------------------- chain
    def _chain(self) -> list[tuple[str, str]]:
        """``(provider, model)`` pairs to try, in order.

        Within the active provider the agent's tiered model comes first, then that provider's
        remaining models; then each fallback provider's own models. A fallback provider never
        inherits a Groq-specific model id.
        """
        pairs: list[tuple[str, str]] = []
        for i, provider in enumerate(settings.provider_chain()):
            cfg = PROVIDERS[provider]
            models = list(cfg.models)
            if i == 0:
                preferred = model_for_agent(self.agent_id)
                if preferred:
                    models = [preferred] + [m for m in models if m != preferred]
            for m in models:
                if (provider, m) not in pairs:
                    pairs.append((provider, m))
        return pairs

    def describe(self) -> str:
        chain = self._chain()
        head = f"{chain[0][0]}:{chain[0][1]}" if chain else "none"
        return f"{head} (+{max(0, len(chain) - 1)} fallback)"

    # ------------------------------------------------------------------ client
    def _client_for(self, provider: str, model: str, temperature: float | None = None):
        from langchain_openai import ChatOpenAI

        cfg = PROVIDERS[provider]
        key = os.getenv(cfg.api_key_env)
        if not key:
            raise LLMError(f"Missing API key: set {cfg.api_key_env} in your environment / .env.")
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=cfg.base_url,
            temperature=settings.temperature if temperature is None else temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.agent_timeout_seconds,
            max_retries=0,  # we own retry/backoff so behaviour is explicit and testable
        )

    # ------------------------------------------------------- retry & fallback
    def _run_with_retry(self, fn):
        """Retry one (provider, model) on transient errors; raise a user-safe error otherwise."""
        attempts = settings.max_retries_per_call + 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                return fn()
            except LLMError:
                raise  # our own guard failures are already user-safe; don't retry
            except Exception as e:  # noqa: BLE001
                last = e
                if any(t in str(e).lower() for t in _TRANSIENT) and i < attempts - 1:
                    time.sleep(1.5 * (i + 1))  # linear backoff
                    continue
                raise _friendly(e) from e
        raise _friendly(last or LLMError("unknown error"))

    def _try_chain(self, per_pair):
        """Run ``per_pair(provider, model)`` across the chain until one succeeds.

        Every attempt — successful or not — is metered, so the usage report shows the cost of
        failed fallbacks rather than hiding it.
        """
        if self.usage:
            self.usage.check_budget()

        last: LLMError | None = None
        for provider, model in self._chain():
            t0 = time.perf_counter()
            try:
                result, tin, tout = per_pair(provider, model)
            except LLMError as e:
                last = e
                if self.usage:
                    self.usage.record(
                        agent_id=self.agent_id, provider=provider, model=model,
                        seconds=time.perf_counter() - t0, ok=False, error=str(e),
                    )
                continue
            self.last_used_model, self.last_used_provider = model, provider
            if self.usage:
                self.usage.record(
                    agent_id=self.agent_id, provider=provider, model=model,
                    input_tokens=tin, output_tokens=tout,
                    seconds=time.perf_counter() - t0, ok=True,
                )
            return result
        raise last or LLMError("All configured providers and models failed.")

    # ------------------------------------------------------------------- calls
    def complete(self, system: str, user: str) -> str:
        def per_pair(provider: str, model: str):
            def call():
                resp = self._client_for(provider, model).invoke(
                    [("system", system), ("user", user)]
                )
                content = getattr(resp, "content", None)
                if not content:  # choices:null guard
                    raise LLMError("The AI returned an empty response (no choices).")
                tin, tout = _tokens(resp)
                return (content if isinstance(content, str) else str(content)), tin, tout

            return self._run_with_retry(call)

        return self._try_chain(per_pair)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        """Return a validated ``schema`` instance.

        Three escalating strategies per (provider, model): native function calling, then explicit
        JSON prompting for models whose function-calling rejects nested schemas, then the next
        pair in the chain. The probe showed several Groq models fail native calling on the nested
        TaskPlan while handling the JSON form, so this path is load-bearing, not defensive.
        """
        def per_pair(provider: str, model: str):
            def native():
                client = self._client_for(provider, model).with_structured_output(
                    schema, method="function_calling", include_raw=True
                )
                out = client.invoke([("system", system), ("user", user)])
                parsed, raw = out.get("parsed"), out.get("raw")
                if parsed is None:
                    raise LLMError("The AI returned no structured result.")
                tin, tout = _tokens(raw) if raw is not None else (0, 0)
                if isinstance(parsed, schema):
                    return parsed, tin, tout
                if isinstance(parsed, dict):
                    return schema.model_validate(parsed), tin, tout
                raise LLMError("The AI returned an unexpected structured type.")

            try:
                return self._run_with_retry(native)
            except LLMError:
                pass  # same pair: fall back to explicit JSON prompting

            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            json_system = (
                f"{system}\n\nReturn ONLY a JSON object that validates against this JSON Schema. "
                f"No prose, no code fences.\n\nJSON Schema:\n{schema_json}"
            )

            def as_json():
                resp = self._client_for(provider, model).invoke(
                    [("system", json_system), ("user", user)]
                )
                content = getattr(resp, "content", None)
                if not content:
                    raise LLMError("The AI returned an empty response (no choices).")
                tin, tout = _tokens(resp)
                try:
                    return schema.model_validate_json(_extract_json(str(content))), tin, tout
                except ValidationError as e:
                    raise LLMError("The AI response did not match the required structure.") from e

            return self._run_with_retry(as_json)

        return self._try_chain(per_pair)

    def invoke_tools(self, messages: list, tools: list):
        """Tool-calling invocation, with cross-provider fallback."""
        def per_pair(provider: str, model: str):
            def call():
                resp = self._client_for(provider, model).bind_tools(tools).invoke(messages)
                tin, tout = _tokens(resp)
                return resp, tin, tout

            return self._run_with_retry(call)

        return self._try_chain(per_pair)


# ---------------------------------------------------------------- convenience
def get_llm(agent_id: str = "system", usage: UsageTracker | None = None) -> LLMService:
    """Factory used by every agent node. ``agent_id`` drives model tier and cost attribution."""
    return LLMService(agent_id=agent_id, usage=usage)


def configured_providers() -> dict[str, bool]:
    """Provider → whether a key is present. Drives the UI's provider status panel."""
    return {name: bool(os.getenv(cfg.api_key_env)) for name, cfg in PROVIDERS.items()}
