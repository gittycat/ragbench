from typing import Optional

from anthropic import Anthropic

from .settings import init_settings, get_anthropic_key

_CLIENT: Optional[Anthropic] = None


def get_client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        init_settings()
        _CLIENT = Anthropic(api_key=get_anthropic_key())
    return _CLIENT


def call_anthropic(messages, model: str = "claude-3-5-sonnet-latest", **kwargs):
    client = get_client()
    return client.messages.create(
        model=model,
        messages=messages,
        **kwargs,
    )
