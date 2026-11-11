"""
llm-apis – Ready-to-use clients for every free LLM API.
Source: https://github.com/mnfst/awesome-free-llm-apis

Quick start:
    from llm_apis import registry
    print(registry.list_providers())
    reply = registry.chat("Hello!", provider="groq")

Or use a provider directly:
    from llm_apis.providers import groq
    print(groq.chat("Hello!"))
"""

from . import registry

__all__ = ["registry"]
