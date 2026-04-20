"""
GitHub Models – Inference Provider  🇺🇸
Free for all GitHub users: https://github.com/marketplace/models
Limits: 10–15 RPM · 50–150 RPD · 8K input / 4K output per request
Requires a GitHub Personal Access Token (classic or fine-grained).
"""

import os
from openai import OpenAI

BASE_URL = "https://models.inference.ai.azure.com"
KEY_URL  = "https://github.com/settings/tokens"
ENV_VAR  = "GITHUB_TOKEN"

MODELS = {
    "gpt-4.1":             {"id": "gpt-4.1",                   "context": "1M",   "output": "32K",  "modality": "Text",            "rate": "10 RPM, 50 RPD"},
    "gpt-4.1-mini":        {"id": "gpt-4.1-mini",              "context": "1M",   "output": "32K",  "modality": "Text",            "rate": "15 RPM, 150 RPD"},
    "gpt-4o":              {"id": "gpt-4o",                    "context": "128K",  "output": "16K",  "modality": "Text+Vision",     "rate": "10 RPM, 50 RPD"},
    "o3-mini":             {"id": "o3-mini",                   "context": "200K",  "output": "100K", "modality": "Text (reasoning)","rate": "10 RPM, 50 RPD"},
    "o4-mini":             {"id": "o4-mini",                   "context": "200K",  "output": "100K", "modality": "Text (reasoning)","rate": "10 RPM, 50 RPD"},
    "llama-4-scout":       {"id": "Llama-4-Scout-17B-16E",     "context": "512K",  "output": "~4K",  "modality": "Text+Vision",     "rate": "15 RPM, 150 RPD"},
    "llama-4-maverick":    {"id": "Llama-4-Maverick-17B-128E", "context": "256K",  "output": "~4K",  "modality": "Text+Vision",     "rate": "10 RPM, 50 RPD"},
    "llama-3.3-70b":       {"id": "Meta-Llama-3.3-70B",        "context": "131K",  "output": "~4K",  "modality": "Text",            "rate": "15 RPM, 150 RPD"},
    "deepseek-r1":         {"id": "DeepSeek-R1",               "context": "64K",   "output": "8K",   "modality": "Text (reasoning)","rate": "15 RPM, 150 RPD"},
    "mistral-small-3.1":   {"id": "Mistral-Small-3.1",         "context": "128K",  "output": "~4K",  "modality": "Text+Vision",     "rate": "15 RPM, 150 RPD"},
}

DEFAULT_MODEL = "gpt-4.1-mini"


def client() -> OpenAI:
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Set {ENV_VAR} – a GitHub Personal Access Token (get one at {KEY_URL})")
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
    print(chat("Say hello from GitHub Models!"))
