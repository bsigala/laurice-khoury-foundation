"""
Ollama Cloud – Inference Provider  🇺🇸
Free tier key: https://ollama.com/settings/keys
Limits: Session limits reset every 5 hrs · Weekly limits reset every 7 days · 400+ models
NOTE: NOT OpenAI SDK-compatible. Uses the native Ollama REST API.
Install: pip install ollama
"""

import os

KEY_URL = "https://ollama.com/settings/keys"
ENV_VAR = "OLLAMA_API_KEY"

MODELS = {
    "llama3.1":    {"id": "llama3.1:cloud",    "context": "128K", "modality": "Text",            "rate": "Session/weekly limits"},
    "deepseek-r1": {"id": "deepseek-r1:cloud", "context": "128K", "modality": "Text (reasoning)","rate": "Session/weekly limits"},
    "qwen2.5":     {"id": "qwen2.5:cloud",     "context": "128K", "modality": "Text",            "rate": "Session/weekly limits"},
    "gemma2":      {"id": "gemma2:cloud",      "context": "8K",   "modality": "Text",            "rate": "Session/weekly limits"},
    "mistral":     {"id": "mistral:cloud",     "context": "32K",  "modality": "Text",            "rate": "Session/weekly limits"},
}

DEFAULT_MODEL = "llama3.1:cloud"
OLLAMA_CLOUD_HOST = "https://api.ollama.com"


def client():
    """Returns a configured ollama.Client pointing at Ollama Cloud."""
    try:
        import ollama
    except ImportError:
        raise ImportError("Run: pip install ollama")
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        raise EnvironmentError(f"Set {ENV_VAR} (get key at {KEY_URL})")
    return ollama.Client(
        host=OLLAMA_CLOUD_HOST,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "You are a helpful assistant.", **kwargs) -> str:
    c = client()
    response = c.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **kwargs,
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print(chat("Say hello from Ollama Cloud!"))
