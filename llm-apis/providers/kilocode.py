"""
Kilo Code – Inference Provider  🇺🇸
Free models (no credit card): https://kilo.ai
Limits: ~200 req/hr per model
Note: `kilo-auto/free` auto-routes to minimax/minimax-m2.5:free (80%) and stepfun/step-3.5-flash:free (20%).
      nvidia/nemotron-3-super-120b-a12b:free is trial-use only — prompts are logged by NVIDIA.
"""

import os
from openai import OpenAI

BASE_URL = "https://api.kilo.ai/api/gateway"
KEY_URL  = "https://kilo.ai"
ENV_VAR  = "KILOCODE_API_KEY"

MODELS = {
    "dola-seed-2.0-pro":    {"id": "bytedance-seed/dola-seed-2.0-pro:free",         "modality": "Text",            "rate": "~200 req/hr"},
    "grok-code-fast-1":     {"id": "x-ai/grok-code-fast-1:optimized:free",          "modality": "Text (code)",     "rate": "~200 req/hr"},
    "nemotron-120b":        {"id": "nvidia/nemotron-3-super-120b-a12b:free",         "context": "262K", "output": "32K", "modality": "Text", "rate": "~200 req/hr"},
    "trinity-thinking":     {"id": "arcee-ai/trinity-large-thinking:free",           "modality": "Text (reasoning)","rate": "~200 req/hr"},
    "openrouter-free":      {"id": "openrouter/free",                                "modality": "Text",            "rate": "~200 req/hr"},
    "kilo-auto":            {"id": "kilo-auto/free",                                 "modality": "Text",            "rate": "~200 req/hr"},
}

DEFAULT_MODEL = "kilo-auto/free"


def client() -> OpenAI:
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Set {ENV_VAR} (sign up at {KEY_URL})")
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
    print(chat("Say hello from Kilo Code!"))
