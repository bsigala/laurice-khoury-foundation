"""
Mistral AI – Provider API  🇫🇷
Free "Experiment" key: https://console.mistral.ai/api-keys
Limits: ~1 RPS · 500K TPM · ~1B tokens/month · No credit card required
"""

import os
from openai import OpenAI

BASE_URL = "https://api.mistral.ai/v1"
KEY_URL  = "https://console.mistral.ai/api-keys"
ENV_VAR  = "MISTRAL_API_KEY"

MODELS = {
    "mistral-small-4":  {"id": "mistral-small-2603",  "context": "256K", "output": "256K", "modality": "Text+Image+Code", "rate": "~1 RPS, 500K TPM"},
    "mistral-medium-3": {"id": "mistral-medium-2505", "context": "128K", "output": "128K", "modality": "Text",            "rate": "~1 RPS, 500K TPM"},
    "mistral-large-3":  {"id": "mistral-large-2411",  "context": "256K", "output": "256K", "modality": "Text",            "rate": "~1 RPS, 500K TPM"},
    "mistral-nemo":     {"id": "open-mistral-nemo",   "context": "128K", "output": "128K", "modality": "Text",            "rate": "~1 RPS, 500K TPM"},
    "codestral":        {"id": "codestral-2501",       "context": "256K", "output": "256K", "modality": "Code",            "rate": "~1 RPS, 500K TPM"},
    "pixtral-large":    {"id": "pixtral-large-2411",  "context": "128K", "output": "128K", "modality": "Text+Image",      "rate": "~1 RPS, 500K TPM"},
}

DEFAULT_MODEL = "mistral-small-2603"


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
    print(chat("Say hello from Mistral AI!"))
