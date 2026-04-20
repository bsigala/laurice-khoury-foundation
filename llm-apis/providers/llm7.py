"""
LLM7.io – Inference Provider  🇬🇧
Zero-friction gateway: https://token.llm7.io
Limits: 30 RPM without token · 120 RPM with token · 30+ models · No registration for basic access
"""

import os
from openai import OpenAI

BASE_URL = "https://api.llm7.io/v1"
KEY_URL  = "https://token.llm7.io"
ENV_VAR  = "LLM7_API_KEY"

MODELS = {
    "deepseek-r1":          {"id": "deepseek-r1-0528",        "modality": "Text (reasoning)", "rate": "30 RPM (120 with token)"},
    "deepseek-v3":          {"id": "deepseek-v3-0324",        "modality": "Text",             "rate": "30 RPM (120 with token)"},
    "gemini-2.5-flash-lite":{"id": "gemini-2.5-flash-lite",   "modality": "Text+Vision",      "rate": "30 RPM (120 with token)"},
    "gpt-4o-mini":          {"id": "gpt-4o-mini",             "modality": "Text+Vision",      "rate": "30 RPM (120 with token)"},
    "mistral-small-3.1":    {"id": "mistral-small-3.1-24b",   "context": "32K", "modality": "Text", "rate": "30 RPM (120 with token)"},
    "qwen2.5-coder-32b":    {"id": "qwen2.5-coder-32b",       "modality": "Text (code)",      "rate": "30 RPM (120 with token)"},
}

DEFAULT_MODEL = "deepseek-v3-0324"


def client() -> OpenAI:
    # Token is optional — basic access works without it, but raises rate limit
    api_key = os.environ.get(ENV_VAR, "no-key-required")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "You are a helpful assistant.", **kwargs) -> str:
    c = client()
    response = c.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Say hello from LLM7.io!"))
