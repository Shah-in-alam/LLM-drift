from collections.abc import Callable

from drift.providers.anthropic_provider import AnthropicChatProvider
from drift.providers.base import ChatProvider
from drift.providers.openai_embed import embed as _openai_embed
from drift.providers.openai_provider import OpenAIChatProvider

VALID_PROVIDERS = ("openai", "anthropic")


def get_provider(name: str) -> ChatProvider:
    if name == "openai":
        return OpenAIChatProvider()
    if name == "anthropic":
        return AnthropicChatProvider()
    raise ValueError(
        f"Unknown provider: {name!r}. Choices: {', '.join(VALID_PROVIDERS)}."
    )


def get_embedder() -> Callable[[str], list[float]]:
    return _openai_embed


__all__ = ["ChatProvider", "VALID_PROVIDERS", "get_provider", "get_embedder"]
