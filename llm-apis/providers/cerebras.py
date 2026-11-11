"""
Cerebras – Inference Provider  🇺🇸
Free tier key: https://cloud.cerebras.ai/
Limits: 30 RPM · 14,400 RPD · 1M tokens/day · Ultra-fast ~2,600 tok/s
Note: Context window capped at 8K on free tier (models support 128K+ paid).
"""

import os
from openai import OpenAI

BASE_URL = "https://api.cerebras.ai/v1"
KEY_URL  = "https://cloud.cerebras.ai/"
ENV_VAR  = "CEREBRAS_API_KEY"

MODELS = {
    "llama3.1-8b":    {"id": "llama3.1-8b",                       "context": "8K (128K paid)",  "output": "8K", "modality": "Text", "rate": "30 RPM, 14,400 RPD, 1M TPD"},
    "gpt-oss-120b":   {"id": "gpt-oss-120b",                      "context": "8K (128K paid)",  "output": "8K", "modality": "Text", "rate": "30 RPM, 14,400 RPD, 1M TPD"},
    "qwen3-235b":     {"id": "qwen-3-235b-a22b-instruct-2507",    "context": "8K (131K paid)",  "output": "8K", "modality": "Text", "rate": "30 RPM, 14,400 RPD, 1M TPD"},
    "zai-glm-4.7":    {"id": "zai-glm-4.7",                       "context": "8K (128K paid)",  "output": "8K", "modality": "Text", "rate": "10 RPM, 100 RPD, 1M TPD"},
}

DEFAULT_MODEL = "llama3.1-8b"


def client() -> OpenAI:
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Set {ENV_VAR} (get key at {KEY_URL})")
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
    print(chat("Say hello from Cerebras!"))
