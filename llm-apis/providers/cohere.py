"""
Cohere – Provider API  🇨🇦
Free "Trial" key: https://dashboard.cohere.com/api-keys
Limits: 1,000 API calls/month · Non-commercial use only
"""

import os
from openai import OpenAI

BASE_URL = "https://api.cohere.com/v2"
KEY_URL  = "https://dashboard.cohere.com/api-keys"
ENV_VAR  = "COHERE_API_KEY"

MODELS = {
    "command-a":     {"id": "command-a-03-2025",       "context": "256K", "output": "4K",  "modality": "Text",                    "rate": "20 RPM"},
    "command-r-plus":{"id": "command-r-plus-08-2024",  "context": "128K", "output": "4K",  "modality": "Text",                    "rate": "20 RPM"},
    "command-r":     {"id": "command-r-08-2024",       "context": "128K", "output": "4K",  "modality": "Text",                    "rate": "20 RPM"},
    "command-r7b":   {"id": "command-r7b-12-2024",     "context": "128K", "output": "4K",  "modality": "Text",                    "rate": "20 RPM"},
    "embed-v4":      {"id": "embed-v4.0",              "context": "—",    "output": "—",   "modality": "Embeddings (Text+Image)", "rate": "2,000 inputs/min"},
    "rerank-3.5":    {"id": "rerank-v3.5",             "context": "—",    "output": "—",   "modality": "Reranking",               "rate": "10 RPM"},
}

DEFAULT_MODEL = "command-a-03-2025"


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
    print(chat("Say hello from Cohere!"))
