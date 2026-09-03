"""
Groq – Inference Provider  🇺🇸
Free tier key: https://console.groq.com/keys
Limits: 30 RPM · 14,400 RPD (most models) · Ultra-fast LPU inference
Note: Llama 4 Maverick is capped at 500 RPD.
"""

import os
from openai import OpenAI

BASE_URL = "https://api.groq.com/openai/v1"
KEY_URL  = "https://console.groq.com/keys"
ENV_VAR  = "GROQ_API_KEY"

MODELS = {
    "llama-3.3-70b":          {"id": "llama-3.3-70b-versatile",            "context": "131K", "output": "32K",  "modality": "Text",          "rate": "30 RPM, 14,400 RPD"},
    "llama-3.1-8b":           {"id": "llama-3.1-8b-instant",               "context": "131K", "output": "131K", "modality": "Text",          "rate": "30 RPM, 14,400 RPD"},
    "llama-4-scout":          {"id": "llama-4-scout-17b-16e-instruct",     "context": "131K", "output": "8K",   "modality": "Text+Vision",   "rate": "30 RPM, 14,400 RPD"},
    "llama-4-maverick":       {"id": "llama-4-maverick-17b-128e-instruct", "context": "131K", "output": "8K",   "modality": "Text+Vision",   "rate": "15 RPM, 500 RPD"},
    "qwen3-32b":              {"id": "qwen3-32b",                          "context": "131K", "output": "131K", "modality": "Text",          "rate": "30 RPM, 14,400 RPD"},
    "gpt-oss-120b":           {"id": "gpt-oss-120b",                       "context": "131K", "output": "32K",  "modality": "Text",          "rate": "30 RPM, 14,400 RPD"},
    "kimi-k2":                {"id": "kimi-k2-instruct",                   "context": "262K", "output": "262K", "modality": "Text",          "rate": "30 RPM, 14,400 RPD"},
    "deepseek-r1-70b":        {"id": "deepseek-r1-distill-70b",            "context": "131K", "output": "8K",   "modality": "Text",          "rate": "30 RPM, 14,400 RPD"},
    "whisper-large-v3":       {"id": "whisper-large-v3",                   "context": "—",    "output": "—",    "modality": "Audio→Text",    "rate": "20 RPM, 2,000 RPD"},
    "whisper-large-v3-turbo": {"id": "whisper-large-v3-turbo",             "context": "—",    "output": "—",    "modality": "Audio→Text",    "rate": "20 RPM, 2,000 RPD"},
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"


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


def transcribe(audio_file_path: str, model: str = "whisper-large-v3-turbo") -> str:
    """Transcribe an audio file using Groq's Whisper endpoint."""
    c = client()
    with open(audio_file_path, "rb") as f:
        result = c.audio.transcriptions.create(model=model, file=f)
    return result.text


if __name__ == "__main__":
    print(chat("Say hello from Groq!"))
