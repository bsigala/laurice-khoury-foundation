"""
Google Gemini – Provider API  🇺🇸
Free tier key: https://aistudio.google.com/app/apikey
Limits: 10–15 RPM · 250–1,000 RPD · NOT available in EU/UK/Switzerland
Note: Free-tier prompts may be used by Google to improve products.
"""

import os
from openai import OpenAI

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
KEY_URL  = "https://aistudio.google.com/app/apikey"
ENV_VAR  = "GEMINI_API_KEY"

MODELS = {
    "gemini-2.5-flash":      {"id": "gemini-2.5-flash",      "context": "1M",  "output": "65K", "modality": "Text+Image+Audio+Video", "rate": "10 RPM, 250 RPD"},
    "gemini-2.5-flash-lite": {"id": "gemini-2.5-flash-lite", "context": "1M",  "output": "65K", "modality": "Text+Image+Audio+Video", "rate": "15 RPM, 1,000 RPD"},
}

DEFAULT_MODEL = "gemini-2.5-flash"


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
    print(chat("Say hello from Google Gemini!"))
