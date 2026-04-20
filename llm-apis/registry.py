"""
Master registry for all free LLM API providers.
Each provider module exposes: MODELS, DEFAULT_MODEL, client(), chat().
"""

from __future__ import annotations

import importlib
from typing import Any

_PROVIDER_MODULES = {
    "cerebras":      "llm_apis.providers.cerebras",
    "cloudflare":    "llm_apis.providers.cloudflare",
    "cohere":        "llm_apis.providers.cohere",
    "github":        "llm_apis.providers.github_models",
    "gemini":        "llm_apis.providers.google_gemini",
    "groq":          "llm_apis.providers.groq",
    "huggingface":   "llm_apis.providers.huggingface",
    "kilocode":      "llm_apis.providers.kilocode",
    "llm7":          "llm_apis.providers.llm7",
    "mistral":       "llm_apis.providers.mistral",
    "modelscope":    "llm_apis.providers.modelscope",
    "nvidia":        "llm_apis.providers.nvidia_nim",
    "ollama":        "llm_apis.providers.ollama_cloud",
    "openrouter":    "llm_apis.providers.openrouter",
    "siliconflow":   "llm_apis.providers.siliconflow",
    "zai":           "llm_apis.providers.zai",
}

# Human-readable metadata for every provider (sourced from awesome-free-llm-apis)
PROVIDER_INFO: dict[str, dict] = {
    "cerebras":    {"category": "inference_provider", "country": "US", "free_tier": "1M tokens/day",       "card_required": False, "key_url": "https://cloud.cerebras.ai/"},
    "cloudflare":  {"category": "inference_provider", "country": "US", "free_tier": "10K neurons/day",     "card_required": False, "key_url": "https://dash.cloudflare.com/profile/api-tokens"},
    "cohere":      {"category": "provider_api",       "country": "CA", "free_tier": "1,000 calls/month",   "card_required": False, "key_url": "https://dashboard.cohere.com/api-keys"},
    "github":      {"category": "inference_provider", "country": "US", "free_tier": "50–150 RPD",          "card_required": False, "key_url": "https://github.com/settings/tokens"},
    "gemini":      {"category": "provider_api",       "country": "US", "free_tier": "250–1,000 RPD",       "card_required": False, "key_url": "https://aistudio.google.com/app/apikey"},
    "groq":        {"category": "inference_provider", "country": "US", "free_tier": "14,400 RPD",          "card_required": False, "key_url": "https://console.groq.com/keys"},
    "huggingface": {"category": "inference_provider", "country": "US", "free_tier": "~$0.10/month credits","card_required": False, "key_url": "https://huggingface.co/settings/tokens"},
    "kilocode":    {"category": "inference_provider", "country": "US", "free_tier": "~200 req/hr",         "card_required": False, "key_url": "https://kilo.ai"},
    "llm7":        {"category": "inference_provider", "country": "GB", "free_tier": "30–120 RPM",          "card_required": False, "key_url": "https://token.llm7.io"},
    "mistral":     {"category": "provider_api",       "country": "FR", "free_tier": "~1B tokens/month",    "card_required": False, "key_url": "https://console.mistral.ai/api-keys"},
    "modelscope":  {"category": "inference_provider", "country": "CN", "free_tier": "2,000 RPD",           "card_required": False, "key_url": "https://modelscope.cn/my/myaccesstoken"},
    "nvidia":      {"category": "inference_provider", "country": "US", "free_tier": "~40 RPM, no daily cap","card_required": False, "key_url": "https://build.nvidia.com/explore/discover"},
    "ollama":      {"category": "inference_provider", "country": "US", "free_tier": "Session/weekly limits","card_required": False, "key_url": "https://ollama.com/settings/keys"},
    "openrouter":  {"category": "inference_provider", "country": "US", "free_tier": "200 RPD (35+ models)","card_required": False, "key_url": "https://openrouter.ai/keys"},
    "siliconflow": {"category": "inference_provider", "country": "CN", "free_tier": "14 CNY + free models","card_required": False, "key_url": "https://cloud.siliconflow.cn/account/ak"},
    "zai":         {"category": "provider_api",       "country": "CN", "free_tier": "Permanent free",      "card_required": False, "key_url": "https://open.bigmodel.cn/usercenter/apikeys"},
}


def _load(provider: str):
    if provider not in _PROVIDER_MODULES:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list_providers()}")
    return importlib.import_module(_PROVIDER_MODULES[provider])


def list_providers() -> list[str]:
    return sorted(_PROVIDER_MODULES.keys())


def list_models(provider: str) -> dict[str, Any]:
    return _load(provider).MODELS


def info(provider: str) -> dict:
    return PROVIDER_INFO.get(provider, {})


def chat(
    prompt: str,
    provider: str,
    model: str | None = None,
    system: str = "You are a helpful assistant.",
    **kwargs,
) -> str:
    """
    Send a chat prompt to any provider.

    Args:
        prompt:   The user message.
        provider: One of the keys from list_providers().
        model:    Specific model ID (defaults to each provider's DEFAULT_MODEL).
        system:   System prompt.
        **kwargs: Passed through to the underlying completions call.
    """
    mod = _load(provider)
    call_model = model or mod.DEFAULT_MODEL
    return mod.chat(prompt, model=call_model, system=system, **kwargs)


def all_models_table() -> list[dict]:
    """Returns a flat list of all models across all providers for easy inspection."""
    rows = []
    for provider in list_providers():
        try:
            mod = _load(provider)
            for alias, spec in mod.MODELS.items():
                rows.append({
                    "provider": provider,
                    "alias": alias,
                    **spec,
                })
        except Exception:
            pass
    return rows
