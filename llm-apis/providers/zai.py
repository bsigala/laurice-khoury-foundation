"""
Z AI / Zhipu AI – Provider API  🇨🇳
Free permanent models: https://open.bigmodel.cn/usercenter/apikeys
Limits: 1 concurrent request per model · No credit card required
"""

import os
from openai import OpenAI

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
KEY_URL  = "https://open.bigmodel.cn/usercenter/apikeys"
ENV_VAR  = "ZAI_API_KEY"

MODELS = {
    "glm-4.7-flash": {"id": "glm-4.7-flash",  "context": "200K", "output": "128K", "modality": "Text",        "rate": "1 concurrent request"},
    "glm-4.5-flash": {"id": "glm-4.5-flash",  "context": "128K", "output": "~8K",  "modality": "Text",        "rate": "1 concurrent request"},
    "glm-4.6v-flash":{"id": "glm-4.6v-flash", "context": "128K", "output": "~4K",  "modality": "Text+Image",  "rate": "1 concurrent request"},
}

DEFAULT_MODEL = "glm-4.7-flash"


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
    print(chat("Say hello from Z AI!"))
