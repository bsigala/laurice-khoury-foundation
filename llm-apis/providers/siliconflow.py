"""
SiliconFlow – Inference Provider  🇨🇳
Free tier key: https://cloud.siliconflow.cn/account/ak
Limits: 1,000 RPM · 50K TPM · 14 CNY signup credits + permanently free models
"""

import os
from openai import OpenAI

BASE_URL = "https://api.siliconflow.cn/v1"
KEY_URL  = "https://cloud.siliconflow.cn/account/ak"
ENV_VAR  = "SILICONFLOW_API_KEY"

MODELS = {
    "qwen3-8b":                  {"id": "Qwen/Qwen3-8B",                            "context": "131K",  "output": "131K",         "modality": "Text",            "rate": "1,000 RPM, 50K TPM"},
    "deepseek-r1-0528-qwen3-8b": {"id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",   "context": "~33K",  "output": "16K",           "modality": "Text (reasoning)","rate": "1,000 RPM, 50K TPM"},
    "deepseek-r1-distill-qwen7b":{"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  "context": "131K",  "output": "Configurable",  "modality": "Text (reasoning)","rate": "1,000 RPM, 50K TPM"},
    "glm-4-9b":                  {"id": "THUDM/glm-4-9b-chat",                      "context": "32K",   "output": "32K",           "modality": "Text",            "rate": "1,000 RPM, 50K TPM"},
    "glm-4.1v-9b-thinking":      {"id": "THUDM/GLM-4.1V-9B-Thinking",              "context": "66K",   "output": "66K",           "modality": "Vision+Text",     "rate": "1,000 RPM, 50K TPM"},
    "deepseek-ocr":              {"id": "deepseek-ai/DeepSeek-OCR",                 "context": "—",     "output": "8K",            "modality": "Vision (OCR)",    "rate": "1,000 RPM, 50K TPM"},
}

DEFAULT_MODEL = "Qwen/Qwen3-8B"


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
    print(chat("Say hello from SiliconFlow!"))
