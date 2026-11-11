"""
OpenRouter – Inference Provider  🇺🇸
Free tier key: https://openrouter.ai/keys
Limits: 20 RPM · 200 RPD (1,000 RPD after a one-time $10+ credit purchase) · 35+ :free models
Note: Use `:free` suffix on model IDs. `openrouter/free` auto-routes across all free models.
"""

import os
from openai import OpenAI

BASE_URL = "https://openrouter.ai/api/v1"
KEY_URL  = "https://openrouter.ai/keys"
ENV_VAR  = "OPENROUTER_API_KEY"

MODELS = {
    "deepseek-r1":         {"id": "deepseek/deepseek-r1-0528:free",            "context": "163K", "output": "~163K", "modality": "Text (reasoning)", "rate": "20 RPM, 200 RPD"},
    "deepseek-chat-v3":    {"id": "deepseek/deepseek-chat-v3-0324:free",       "context": "163K", "output": "163K",  "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "qwen3.6-plus":        {"id": "qwen/qwen3.6-plus:free",                    "context": "1M",   "output": "65K",   "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "qwen3-coder-480b":    {"id": "qwen/qwen3-coder-480b-a35b:free",           "context": "262K", "output": "~32K",  "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "llama-4-scout":       {"id": "meta-llama/llama-4-scout:free",             "context": "10M",  "output": "16K",   "modality": "Multimodal",       "rate": "20 RPM, 200 RPD"},
    "llama-4-maverick":    {"id": "meta-llama/llama-4-maverick:free",          "context": "1M",   "output": "16K",   "modality": "Multimodal",       "rate": "20 RPM, 200 RPD"},
    "llama-3.3-70b":       {"id": "meta-llama/llama-3.3-70b-instruct:free",    "context": "65K",  "output": "~16K",  "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "gemma-4-31b":         {"id": "google/gemma-4-31b-it:free",                "context": "256K", "output": "~8K",   "modality": "Multimodal",       "rate": "20 RPM, 200 RPD"},
    "nemotron-super-120b": {"id": "nvidia/nemotron-3-super-120b-a12b:free",    "context": "1M",   "output": "~32K",  "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "gpt-oss-120b":        {"id": "openai/gpt-oss-120b:free",                  "context": "131K", "output": "131K",  "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "minimax-m2.5":        {"id": "minimax/minimax-m2.5:free",                 "context": "196K", "output": "8K",    "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "devstral-2512":       {"id": "mistralai/devstral-2512:free",              "context": "256K", "output": "~32K",  "modality": "Text",             "rate": "20 RPM, 200 RPD"},
    "auto-free":           {"id": "openrouter/free",                           "context": "Varies","output": "Varies","modality": "Text",            "rate": "20 RPM, 200 RPD"},
}

DEFAULT_MODEL = "meta-llama/llama-4-scout:free"


def client(site_url: str = "", site_name: str = "") -> OpenAI:
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Set {ENV_VAR} (get key at {KEY_URL})")
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_name:
        extra_headers["X-Title"] = site_name
    return OpenAI(api_key=api_key, base_url=BASE_URL, default_headers=extra_headers)


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "You are a helpful assistant.", **kwargs) -> str:
    c = client()
    response = c.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Say hello from OpenRouter!"))
