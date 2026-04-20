"""
ModelScope – Inference Provider  🇨🇳
Free API-Inference: https://modelscope.cn/my/myaccesstoken
Limits: 2,000 RPD total · ≤500 RPD/model (dynamic) · Dynamic concurrency limits
Requirements: Alibaba Cloud account binding + real-name verification
"""

import os
from openai import OpenAI

BASE_URL = "https://api-inference.modelscope.cn/v1"
KEY_URL  = "https://modelscope.cn/my/myaccesstoken"
ENV_VAR  = "MODELSCOPE_API_KEY"

MODELS = {
    "qwen3.5-35b":    {"id": "Qwen/Qwen3.5-35B-A3B",  "modality": "Text+Vision",       "rate": "2,000 RPD total; <=500 RPD/model"},
    "qwen3.5-27b":    {"id": "Qwen/Qwen3.5-27B",      "modality": "Text",              "rate": "2,000 RPD total; <=500 RPD/model"},
    "qwen-image":     {"id": "Qwen/Qwen-Image",        "modality": "Image Generation",  "rate": "2,000 RPD total; AIGC-specific caps"},
}

DEFAULT_MODEL = "Qwen/Qwen3.5-27B"


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
    print(chat("Say hello from ModelScope!"))
