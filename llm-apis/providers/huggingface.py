"""
Hugging Face – Inference Provider  🇺🇸
Free Serverless Inference API: https://huggingface.co/settings/tokens
Limits: ~1,000 RPD · ~$0.10/month free credits · Thousands of models available
"""

import os
from openai import OpenAI

BASE_URL = "https://api-inference.huggingface.co/v1"
KEY_URL  = "https://huggingface.co/settings/tokens"
ENV_VAR  = "HF_API_TOKEN"

MODELS = {
    "llama-3.1-8b":       {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct",    "context": "128K", "output": "~4K", "modality": "Text", "rate": "~1,000 RPD"},
    "mistral-7b":         {"id": "mistralai/Mistral-7B-Instruct-v0.3",        "context": "32K",  "output": "~4K", "modality": "Text", "rate": "~1,000 RPD"},
    "mixtral-8x7b":       {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1",      "context": "32K",  "output": "~4K", "modality": "Text", "rate": "~1,000 RPD"},
    "phi-3.5-mini":       {"id": "microsoft/Phi-3.5-mini-instruct",            "context": "128K", "output": "~4K", "modality": "Text", "rate": "~1,000 RPD"},
    "qwen2.5-7b":         {"id": "Qwen/Qwen2.5-7B-Instruct",                  "context": "131K", "output": "~4K", "modality": "Text", "rate": "~1,000 RPD"},
}

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def client(model_id: str | None = None) -> OpenAI:
    """
    Returns an OpenAI-compatible client pointed at the given model's endpoint.
    Pass model_id to target a specific model (e.g. any HF model ID).
    """
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Set {ENV_VAR} (get token at {KEY_URL})")
    base = BASE_URL
    return OpenAI(api_key=api_key, base_url=base)


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "You are a helpful assistant.", **kwargs) -> str:
    c = client()
    response = c.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Say hello from Hugging Face!"))
