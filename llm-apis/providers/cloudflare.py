"""
Cloudflare Workers AI – Inference Provider  🇺🇸
Free tier token: https://dash.cloudflare.com/profile/api-tokens
Limits: 10,000 Neurons/day shared across all models · 50+ models available
Requires CLOUDFLARE_ACCOUNT_ID in addition to API token.
"""

import os
from openai import OpenAI

KEY_URL = "https://dash.cloudflare.com/profile/api-tokens"
ENV_VAR = "CLOUDFLARE_API_TOKEN"
ACCOUNT_ENV_VAR = "CLOUDFLARE_ACCOUNT_ID"

MODELS = {
    "llama-3.3-70b":          {"id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",          "context": "131K",    "modality": "Text",          "rate": "10K neurons/day (shared)"},
    "llama-3.1-8b":           {"id": "@cf/meta/llama-3.1-8b-instruct-fp8-fast",           "context": "131K",    "modality": "Text",          "rate": "10K neurons/day (shared)"},
    "llama-3.2-11b-vision":   {"id": "@cf/meta/llama-3.2-11b-vision-instruct",            "context": "131K",    "modality": "Text+Vision",   "rate": "10K neurons/day (shared)"},
    "llama-4-scout":          {"id": "@cf/meta/llama-4-scout-17b-16e-instruct",           "context": "Up to 10M","modality": "Multimodal",    "rate": "10K neurons/day (shared)"},
    "mistral-small-3.1":      {"id": "@cf/mistralai/mistral-small-3.1-24b-instruct",      "context": "128K",    "modality": "Text",          "rate": "10K neurons/day (shared)"},
    "gemma-4-26b":            {"id": "@cf/google/gemma-4-26b-a4b-it",                     "context": "256K",    "modality": "Text",          "rate": "10K neurons/day (shared)"},
    "qwq-32b":                {"id": "@cf/qwen/qwq-32b",                                  "context": "32K",     "modality": "Text",          "rate": "10K neurons/day (shared)"},
    "deepseek-r1-distill-32b":{"id": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",      "context": "32K",     "modality": "Text",          "rate": "10K neurons/day (shared)"},
}

DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def _base_url() -> str:
    account_id = os.environ.get(ACCOUNT_ENV_VAR)
    if not account_id:
        raise EnvironmentError(f"Set {ACCOUNT_ENV_VAR} (your Cloudflare account ID)")
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def client() -> OpenAI:
    api_token = os.environ.get(ENV_VAR)
    if not api_token:
        raise EnvironmentError(f"Set {ENV_VAR} (get token at {KEY_URL})")
    return OpenAI(api_key=api_token, base_url=_base_url())


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "You are a helpful assistant.", **kwargs) -> str:
    c = client()
    response = c.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Say hello from Cloudflare Workers AI!"))
