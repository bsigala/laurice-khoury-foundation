"""
Master registry for all free LLM API providers.

Features:
- Unified chat() / async_chat() / stream() across all 16 providers
- Automatic .env loading
- Fallback chains: if provider A rate-limits, roll to B, C ...
- Rate limit tracker per provider
- Health checker: test which providers are reachable
- Benchmarker: rank all providers by latency on a given prompt
"""

from __future__ import annotations

import importlib
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator, Iterator

# Auto-load .env from project root or llm-apis directory
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        for candidate in [
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
        ]:
            if candidate.exists():
                load_dotenv(candidate)
                return
        load_dotenv()  # fallback: search CWD
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment

_load_env()

# ──────────────────────────────────────────────────────────────────────────────
# Provider module map
# ──────────────────────────────────────────────────────────────────────────────

_PROVIDER_MODULES = {
    "cerebras":    "llm_apis.providers.cerebras",
    "cloudflare":  "llm_apis.providers.cloudflare",
    "cohere":      "llm_apis.providers.cohere",
    "github":      "llm_apis.providers.github_models",
    "gemini":      "llm_apis.providers.google_gemini",
    "groq":        "llm_apis.providers.groq",
    "huggingface": "llm_apis.providers.huggingface",
    "kilocode":    "llm_apis.providers.kilocode",
    "llm7":        "llm_apis.providers.llm7",
    "mistral":     "llm_apis.providers.mistral",
    "modelscope":  "llm_apis.providers.modelscope",
    "nvidia":      "llm_apis.providers.nvidia_nim",
    "ollama":      "llm_apis.providers.ollama_cloud",
    "openrouter":  "llm_apis.providers.openrouter",
    "siliconflow": "llm_apis.providers.siliconflow",
    "zai":         "llm_apis.providers.zai",
}

# ──────────────────────────────────────────────────────────────────────────────
# Rich metadata
# ──────────────────────────────────────────────────────────────────────────────

PROVIDER_INFO: dict[str, dict] = {
    "cerebras":    {"category": "inference_provider", "country": "US", "free_tier": "1M tokens/day",        "card_required": False, "key_url": "https://cloud.cerebras.ai/",                         "speed": "ultra-fast (~2,600 tok/s)", "strengths": ["speed", "text"]},
    "cloudflare":  {"category": "inference_provider", "country": "US", "free_tier": "10K neurons/day",      "card_required": False, "key_url": "https://dash.cloudflare.com/profile/api-tokens",    "speed": "fast",                      "strengths": ["text", "vision", "multimodal"]},
    "cohere":      {"category": "provider_api",       "country": "CA", "free_tier": "1,000 calls/month",    "card_required": False, "key_url": "https://dashboard.cohere.com/api-keys",              "speed": "moderate",                  "strengths": ["text", "embeddings", "reranking"]},
    "github":      {"category": "inference_provider", "country": "US", "free_tier": "50–150 RPD",           "card_required": False, "key_url": "https://github.com/settings/tokens",                "speed": "fast",                      "strengths": ["text", "vision", "reasoning"]},
    "gemini":      {"category": "provider_api",       "country": "US", "free_tier": "250–1,000 RPD",        "card_required": False, "key_url": "https://aistudio.google.com/app/apikey",             "speed": "fast",                      "strengths": ["text", "vision", "audio", "video", "long-context"]},
    "groq":        {"category": "inference_provider", "country": "US", "free_tier": "14,400 RPD",           "card_required": False, "key_url": "https://console.groq.com/keys",                     "speed": "ultra-fast (LPU)",           "strengths": ["speed", "text", "audio"]},
    "huggingface": {"category": "inference_provider", "country": "US", "free_tier": "~$0.10/month credits", "card_required": False, "key_url": "https://huggingface.co/settings/tokens",            "speed": "variable",                  "strengths": ["text", "thousands of models"]},
    "kilocode":    {"category": "inference_provider", "country": "US", "free_tier": "~200 req/hr",          "card_required": False, "key_url": "https://kilo.ai",                                   "speed": "fast",                      "strengths": ["text", "code"]},
    "llm7":        {"category": "inference_provider", "country": "GB", "free_tier": "30–120 RPM",           "card_required": False, "key_url": "https://token.llm7.io",                             "speed": "fast",                      "strengths": ["text", "no-signup"]},
    "mistral":     {"category": "provider_api",       "country": "FR", "free_tier": "~1B tokens/month",     "card_required": False, "key_url": "https://console.mistral.ai/api-keys",               "speed": "fast",                      "strengths": ["text", "code", "vision", "large-context"]},
    "modelscope":  {"category": "inference_provider", "country": "CN", "free_tier": "2,000 RPD",            "card_required": False, "key_url": "https://modelscope.cn/my/myaccesstoken",            "speed": "moderate",                  "strengths": ["text", "vision", "image-gen"]},
    "nvidia":      {"category": "inference_provider", "country": "US", "free_tier": "~40 RPM, no daily cap","card_required": False, "key_url": "https://build.nvidia.com/explore/discover",          "speed": "fast",                      "strengths": ["text", "vision", "video", "100+ models"]},
    "ollama":      {"category": "inference_provider", "country": "US", "free_tier": "Session/weekly limits","card_required": False, "key_url": "https://ollama.com/settings/keys",                  "speed": "moderate",                  "strengths": ["text", "400+ models"]},
    "openrouter":  {"category": "inference_provider", "country": "US", "free_tier": "200 RPD (35+ :free)",  "card_required": False, "key_url": "https://openrouter.ai/keys",                        "speed": "fast",                      "strengths": ["text", "multimodal", "routing"]},
    "siliconflow": {"category": "inference_provider", "country": "CN", "free_tier": "1,000 RPM, 50K TPM",   "card_required": False, "key_url": "https://cloud.siliconflow.cn/account/ak",           "speed": "fast",                      "strengths": ["text", "reasoning", "vision", "ocr"]},
    "zai":         {"category": "provider_api",       "country": "CN", "free_tier": "Permanent free",       "card_required": False, "key_url": "https://open.bigmodel.cn/usercenter/apikeys",       "speed": "moderate",                  "strengths": ["text", "vision", "permanent-free"]},
}

# Task-type → recommended provider priority order
TASK_ROUTING: dict[str, list[str]] = {
    "speed":        ["cerebras", "groq", "llm7", "siliconflow"],
    "reasoning":    ["github", "openrouter", "groq", "nvidia"],
    "code":         ["mistral", "kilocode", "openrouter", "groq"],
    "vision":       ["gemini", "github", "cloudflare", "nvidia"],
    "long-context": ["gemini", "mistral", "openrouter", "nvidia"],
    "embeddings":   ["cohere", "huggingface", "siliconflow"],
    "audio":        ["groq"],
    "free":         ["llm7", "zai", "siliconflow", "cerebras"],
    "default":      ["groq", "openrouter", "cerebras", "mistral", "gemini"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Rate limit tracker
# ──────────────────────────────────────────────────────────────────────────────

class RateLimitTracker:
    """Lightweight in-process call counter per provider."""

    def __init__(self) -> None:
        self._counts: dict[str, list[float]] = defaultdict(list)

    def record(self, provider: str) -> None:
        self._counts[provider].append(time.time())

    def calls_in_window(self, provider: str, window_seconds: int = 60) -> int:
        cutoff = time.time() - window_seconds
        self._counts[provider] = [t for t in self._counts[provider] if t > cutoff]
        return len(self._counts[provider])

    def report(self) -> dict[str, dict]:
        now = time.time()
        result = {}
        for provider, timestamps in self._counts.items():
            result[provider] = {
                "total_calls": len(timestamps),
                "calls_last_60s": sum(1 for t in timestamps if now - t < 60),
                "calls_last_hour": sum(1 for t in timestamps if now - t < 3600),
            }
        return result


_tracker = RateLimitTracker()

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load(provider: str):
    if provider not in _PROVIDER_MODULES:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list_providers()}")
    return importlib.import_module(_PROVIDER_MODULES[provider])


def _env_key_set(provider: str) -> bool:
    """Return True if the provider's API key env var is set."""
    try:
        mod = _load(provider)
        env_var = getattr(mod, "ENV_VAR", None)
        if not env_var:
            return True  # provider like llm7 doesn't require a key
        return bool(os.environ.get(env_var))
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def list_providers() -> list[str]:
    return sorted(_PROVIDER_MODULES.keys())


def list_models(provider: str) -> dict[str, Any]:
    return _load(provider).MODELS


def info(provider: str) -> dict:
    return PROVIDER_INFO.get(provider, {})


def available_providers() -> list[str]:
    """Return providers whose API key env var is currently set."""
    return [p for p in list_providers() if _env_key_set(p)]


def chat(
    prompt: str,
    provider: str,
    model: str | None = None,
    system: str = "You are a helpful assistant.",
    **kwargs,
) -> str:
    """Send a chat prompt to a specific provider."""
    mod = _load(provider)
    call_model = model or mod.DEFAULT_MODEL
    _tracker.record(provider)
    return mod.chat(prompt, model=call_model, system=system, **kwargs)


def chat_with_fallback(
    prompt: str,
    providers: list[str] | None = None,
    task: str = "default",
    model: str | None = None,
    system: str = "You are a helpful assistant.",
    **kwargs,
) -> dict[str, Any]:
    """
    Try providers in order; return on first success.

    Args:
        prompt:    The user message.
        providers: Explicit ordered list. If omitted, uses TASK_ROUTING[task].
        task:      One of: speed, reasoning, code, vision, long-context,
                           embeddings, audio, free, default.
    Returns:
        {"provider": str, "model": str, "response": str, "attempts": list}
    """
    chain = providers or TASK_ROUTING.get(task, TASK_ROUTING["default"])
    attempts = []
    for provider in chain:
        if not _env_key_set(provider):
            attempts.append({"provider": provider, "error": "no API key set"})
            continue
        try:
            mod = _load(provider)
            call_model = model or mod.DEFAULT_MODEL
            _tracker.record(provider)
            response = mod.chat(prompt, model=call_model, system=system, **kwargs)
            return {
                "provider": provider,
                "model": call_model,
                "response": response,
                "attempts": attempts,
            }
        except Exception as e:
            attempts.append({"provider": provider, "error": str(e)})
    raise RuntimeError(
        f"All providers failed.\nAttempts: {attempts}"
    )


def stream(
    prompt: str,
    provider: str,
    model: str | None = None,
    system: str = "You are a helpful assistant.",
    **kwargs,
) -> Generator[str, None, None]:
    """
    Stream tokens from a provider. Yields one content chunk at a time.
    Works with any OpenAI-compatible provider.

    Usage:
        for chunk in registry.stream("Tell me a story", provider="groq"):
            print(chunk, end="", flush=True)
    """
    mod = _load(provider)
    call_model = model or mod.DEFAULT_MODEL
    c = mod.client()
    _tracker.record(provider)
    response = c.chat.completions.create(
        model=call_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        **kwargs,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def stream_with_fallback(
    prompt: str,
    providers: list[str] | None = None,
    task: str = "default",
    model: str | None = None,
    system: str = "You are a helpful assistant.",
    **kwargs,
) -> Iterator[str]:
    """Stream with automatic provider fallback."""
    chain = providers or TASK_ROUTING.get(task, TASK_ROUTING["default"])
    for provider in chain:
        if not _env_key_set(provider):
            continue
        try:
            yield from stream(prompt, provider=provider, model=model, system=system, **kwargs)
            return
        except Exception:
            continue
    raise RuntimeError("All providers failed during streaming.")


def health_check(providers: list[str] | None = None, timeout: float = 8.0) -> dict[str, dict]:
    """
    Ping each provider with a minimal request and report latency.
    Only tests providers with API keys set.

    Returns:
        {"groq": {"status": "ok", "latency_ms": 312, "model": "..."}, ...}
    """
    import concurrent.futures

    targets = providers or available_providers()
    results = {}

    def _ping(provider: str) -> tuple[str, dict]:
        start = time.time()
        try:
            mod = _load(provider)
            mod.chat("Hi", model=mod.DEFAULT_MODEL, system="Reply with one word.")
            latency = round((time.time() - start) * 1000)
            return provider, {"status": "ok", "latency_ms": latency, "model": mod.DEFAULT_MODEL}
        except Exception as e:
            latency = round((time.time() - start) * 1000)
            return provider, {"status": "error", "latency_ms": latency, "error": str(e)[:120]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_ping, p): p for p in targets}
        for future in concurrent.futures.as_completed(futures, timeout=timeout + 2):
            try:
                provider, result = future.result(timeout=timeout)
                results[provider] = result
            except Exception as e:
                results[futures[future]] = {"status": "timeout", "error": str(e)[:80]}

    return dict(sorted(results.items(), key=lambda x: x[1].get("latency_ms", 99999)))


def benchmark(
    prompt: str = "Write one sentence about the universe.",
    providers: list[str] | None = None,
) -> list[dict]:
    """
    Run the same prompt across all available providers and rank by latency.

    Returns:
        Sorted list of {"rank", "provider", "model", "latency_ms", "tokens", "response"}
    """
    import concurrent.futures

    targets = providers or available_providers()
    rows = []

    def _run(provider: str) -> dict:
        start = time.time()
        try:
            mod = _load(provider)
            resp = mod.chat(prompt, model=mod.DEFAULT_MODEL)
            latency = round((time.time() - start) * 1000)
            tokens = len(resp.split())
            return {
                "provider": provider,
                "model": mod.DEFAULT_MODEL,
                "latency_ms": latency,
                "approx_tokens": tokens,
                "response": resp[:200],
                "error": None,
            }
        except Exception as e:
            return {
                "provider": provider,
                "model": getattr(_load(provider), "DEFAULT_MODEL", "?"),
                "latency_ms": round((time.time() - start) * 1000),
                "approx_tokens": 0,
                "response": None,
                "error": str(e)[:120],
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_run, p) for p in targets]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda x: x["latency_ms"] if not x["error"] else 99999)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def route(task: str) -> list[str]:
    """Return the recommended provider priority list for a given task type."""
    return TASK_ROUTING.get(task, TASK_ROUTING["default"])


def usage_report() -> dict[str, dict]:
    """Return per-provider call counts tracked in this process."""
    return _tracker.report()


def all_models_table() -> list[dict]:
    """Returns a flat list of all models across all providers for inspection."""
    rows = []
    for provider in list_providers():
        try:
            mod = _load(provider)
            for alias, spec in mod.MODELS.items():
                rows.append({"provider": provider, "alias": alias, **spec})
        except Exception:
            pass
    return rows
