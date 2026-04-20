"""
NVIDIA NIM – Inference Provider  🇺🇸
Free with NVIDIA Developer Program: https://build.nvidia.com/explore/discover
Limits: ~40 RPM · No daily token cap · 100+ models available
"""

import os
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
KEY_URL  = "https://build.nvidia.com/explore/discover"
ENV_VAR  = "NVIDIA_API_KEY"

MODELS = {
    "deepseek-r1":          {"id": "deepseek-ai/deepseek-r1",               "context": "128K", "output": "~163K", "modality": "Text (reasoning)",      "rate": "~40 RPM"},
    "nemotron-ultra-253b":  {"id": "nvidia/llama-3.1-nemotron-ultra-253b-v1","context": "128K", "output": "4K",    "modality": "Text",                  "rate": "~40 RPM"},
    "nemotron-super-120b":  {"id": "nvidia/nemotron-3-super-120b-a12b",      "context": "262K", "output": "262K",  "modality": "Text",                  "rate": "~40 RPM"},
    "nemotron-nano-30b":    {"id": "nvidia/nemotron-3-nano-30b-a3b",         "context": "128K", "output": "32K",   "modality": "Text",                  "rate": "~40 RPM"},
    "llama-3.1-405b":       {"id": "meta/llama-3.1-405b-instruct",           "context": "128K", "output": "4K",    "modality": "Text",                  "rate": "~40 RPM"},
    "qwen2.5-72b":          {"id": "qwen/qwen2.5-72b-instruct",              "context": "128K", "output": "8K",    "modality": "Text",                  "rate": "~40 RPM"},
    "gemma-4-31b":          {"id": "google/gemma-4-31b",                     "context": "128K", "output": "8K",    "modality": "Text",                  "rate": "~40 RPM"},
    "mistral-large-2":      {"id": "mistralai/mistral-large-2-instruct",     "context": "128K", "output": "4K",    "modality": "Text",                  "rate": "~40 RPM"},
    "nemotron-nano-vl":     {"id": "nvidia/nemotron-nano-2-vl",              "context": "128K", "output": "8K",    "modality": "Vision+Text+Video",     "rate": "~40 RPM"},
    "minimax-m2.7":         {"id": "minimax/minimax-m2.7",                   "context": "128K", "output": "8K",    "modality": "Text",                  "rate": "~40 RPM"},
}

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


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
    print(chat("Say hello from NVIDIA NIM!"))
