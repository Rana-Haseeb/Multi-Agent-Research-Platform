"""
Environment-based configuration. No hard-coded secrets.

Two things distinguish this from the Week 3 config:

1. **Five providers, not two.** Every one of them speaks the OpenAI wire protocol, so a single
   client drives all of them and switching is one env var. More importantly, ``LLMService``
   falls back *across* providers, not just across models within one — a rate-limited Groq key
   continues on OpenRouter mid-workflow instead of killing the run (Week 4 §22, "Model API
   failure").

2. **Explicit workflow budgets.** A multi-agent graph can loop, fan out, and re-plan, so every
   one of those degrees of freedom gets a hard cap here: revision cycles, research rounds, total
   agent calls, wall clock, and estimated spend. These are the numbers quoted in the README's
   runaway-loop and cost-control sections, so they live in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Project root = parent of the ``app`` package.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class ProviderConfig(BaseModel):
    """A single OpenAI-compatible LLM provider."""

    label: str
    base_url: str
    api_key_env: str
    default_model: str
    models: list[str]                     # in-provider fallback chain, best first
    probe_models: list[str] = []          # candidates for scripts/probe_providers.py
    # USD per 1M tokens, for measured cost estimates (§29). 0.0 = free tier.
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0

    def probes(self) -> list[str]:
        return self.probe_models or self.models


PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        label="Groq (free tier — primary)",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        # Chosen by measurement, not reputation — see scripts/probe_planning_quality.py and
        # scripts/probe_results.json. All five candidates emit *a* structured object; only these
        # two reliably emit the nested TaskPlan schema the Supervisor actually needs.
        default_model="openai/gpt-oss-120b",
        models=["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        probe_models=[
            "openai/gpt-oss-120b",       # 10/10 planning probe
            "openai/gpt-oss-20b",        #  5/10 — 400s on the nested schema
            "llama-3.3-70b-versatile",   #  9/10, 2-3x faster
            "llama-3.1-8b-instant",      #  4/10
            "qwen/qwen3.6-27b",          #  5/10
        ],
    ),
    "google": ProviderConfig(
        label="Google AI Studio (free tier)",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-2.0-flash",
        models=["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    ),
    "openrouter": ProviderConfig(
        label="OpenRouter (free — fallback only, 50 req/day)",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key_env="OPENROUTER_API_KEY",
        default_model="cohere/north-mini-code:free",
        models=["cohere/north-mini-code:free", "nvidia/nemotron-3-nano-30b-a3b:free"],
    ),
    "xai": ProviderConfig(
        label="xAI Grok",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        default_model="grok-3-mini",
        models=["grok-3-mini", "grok-3"],
    ),
    "openai": ProviderConfig(
        label="OpenAI (paid)",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        models=["gpt-4o-mini", "gpt-4o"],
        price_in_per_m=0.15,
        price_out_per_m=0.60,
    ),
}


# --------------------------------------------------------------- per-agent models
# Not every agent needs the same brain, and pretending otherwise wastes latency where it is
# multiplied. Two tiers, both measured by scripts/probe_planning_quality.py:
#
#   REASONING (gpt-oss-120b, 10/10, ~2.5-3.1s) — roles whose *judgement* is the product.
#       The Supervisor's plan determines every downstream task; the Analyst's comparison is the
#       deliverable's substance; the Critic exists specifically to catch weak reasoning, so it
#       must not be the weakest reasoner in the system.
#
#   THROUGHPUT (llama-3.3-70b-versatile, 9/10, ~0.6-1.2s) — roles that transform already-decided
#       inputs. Researchers run N-way in parallel so their latency sets the fan-out's critical
#       path; the Fact-Checker runs once per claim; the Writer synthesises text that the Critic
#       has already approved.
#
# The split is not a guess. scripts/probe_critic_quality.py plants six defects in an analysis and
# scores detection: gpt-oss-120b finds 6/6, llama-3.3-70b finds 3/6 — and *which* three is the
# point. llama caught every citation defect (unsupported claim, fabricated evidence id, evidence
# that doesn't support its claim) and missed every reasoning defect (internal contradiction,
# overgeneralisation from one data point, conclusion resting on an unresearched criterion).
# So llama is sound where the work is mechanical and unsound where it is judgement, which is
# exactly the line these two tiers are drawn along.
#
# Rate limits (measured from Groq response headers, not documentation):
#   gpt-oss-120b            1,000 req/day   8,000 tok/min
#   llama-3.3-70b-versatile 1,000 req/day  12,000 tok/min   <- 50% more TPM headroom, and the
#                                                              4-way researcher fan-out bursts
#                                                              ~8k tokens at once
#   llama-3.1-8b-instant   14,400 req/day   6,000 tok/min   <- high quota, but 400s on our
#                                                              nested schemas; unusable here
# Buckets are per-model, and ~480 calls are planned for the whole week, so quota is not binding.
#
# Experiment 4 measures whether this tiering costs output quality relative to all-120b.
AGENT_MODELS: dict[str, str] = {
    "supervisor":   "openai/gpt-oss-120b",
    "analyst":      "openai/gpt-oss-120b",
    "critic":       "openai/gpt-oss-120b",
    "researcher":   "llama-3.3-70b-versatile",
    "fact_checker": "llama-3.3-70b-versatile",
    "writer":       "llama-3.3-70b-versatile",
    "baseline":     "openai/gpt-oss-120b",   # single-agent control for Experiment 1
}


def model_for_agent(agent_id: str) -> str | None:
    """Model id for an agent, or None to use the provider default.

    Returns None for any provider other than the tiered one, since these ids are Groq-specific
    and a fallback provider must fall back to *its* own default rather than a missing model id.
    """
    if settings.provider != "groq" or settings.model_override:
        return settings.model_override
    return AGENT_MODELS.get(agent_id)


class Settings(BaseModel):
    """Typed, validated view over the environment."""

    # --- LLM ---
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    model_override: str | None = Field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    temperature: float = Field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.0))
    max_tokens: int = Field(default_factory=lambda: _int("LLM_MAX_TOKENS", 2048))
    fallback_providers: list[str] = Field(
        default_factory=lambda: [
            p.strip() for p in os.getenv("LLM_FALLBACK_PROVIDERS", "").split(",") if p.strip()
        ]
    )

    # --- Workflow budgets (§18 revision cap, §22 runaway control, §29 cost) ---
    max_revision_cycles: int = Field(default_factory=lambda: _int("MAX_REVISION_CYCLES", 2))
    max_research_rounds: int = Field(default_factory=lambda: _int("MAX_RESEARCH_ROUNDS", 2))
    max_agent_calls_per_run: int = Field(default_factory=lambda: _int("MAX_AGENT_CALLS_PER_RUN", 40))
    max_parallel_researchers: int = Field(default_factory=lambda: _int("MAX_PARALLEL_RESEARCHERS", 4))
    agent_timeout_seconds: int = Field(default_factory=lambda: _int("AGENT_TIMEOUT_SECONDS", 60))
    max_run_seconds: int = Field(default_factory=lambda: _int("MAX_RUN_SECONDS", 600))
    max_cost_usd_per_run: float = Field(default_factory=lambda: _float("MAX_COST_USD_PER_RUN", 0.50))
    max_retries_per_call: int = 2

    # --- Retrieval / research source ---
    retrieval_mode: str = Field(default_factory=lambda: os.getenv("RETRIEVAL_MODE", "bm25"))
    embedding_provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "none"))
    enable_live_search: bool = Field(default_factory=lambda: _bool("ENABLE_LIVE_SEARCH", False))

    # --- Context budgets (§21) — per-agent, enforced by the context builders ---
    evidence_snippet_chars: int = 400     # how much supporting text an agent ever sees per item
    max_evidence_items_analyst: int = 24
    max_evidence_items_critic: int = 40   # critic sees the index, not full bodies

    # --- Data ---
    database_url: str | None = Field(default_factory=lambda: os.getenv("DATABASE_URL"))
    embedding_dim: int = 384

    # ------------------------------------------------------------------ helpers
    def provider_config(self, name: str | None = None) -> ProviderConfig:
        key = name or self.provider
        if key not in PROVIDERS:
            raise ValueError(f"Unknown provider {key!r}. Choose one of {list(PROVIDERS)}.")
        return PROVIDERS[key]

    def active_model(self) -> str:
        """Model to use: explicit override wins, else the provider default."""
        return self.model_override or self.provider_config().default_model

    def provider_chain(self) -> list[str]:
        """Active provider first, then configured fallbacks that actually have a key set."""
        chain = [self.provider]
        for name in self.fallback_providers:
            if name in PROVIDERS and name not in chain and os.getenv(PROVIDERS[name].api_key_env):
                chain.append(name)
        return chain

    def db_configured(self) -> bool:
        url = self.database_url or ""
        return bool(url) and "<PASSWORD>" not in url and "<project-ref>" not in url


# Import this singleton everywhere.
settings = Settings()
