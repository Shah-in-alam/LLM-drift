import pytest

from drift.providers import VALID_PROVIDERS, get_provider
from drift.providers.anthropic_provider import AnthropicChatProvider
from drift.providers.openai_provider import OpenAIChatProvider


def test_get_provider_openai():
    p = get_provider("openai")
    assert isinstance(p, OpenAIChatProvider)
    assert p.name == "openai"
    assert p.chat_model == "gpt-4o-mini"


def test_get_provider_anthropic():
    p = get_provider("anthropic")
    assert isinstance(p, AnthropicChatProvider)
    assert p.name == "anthropic"
    assert p.chat_model == "claude-sonnet-4-6"


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("cohere")


def test_valid_providers_contains_both():
    assert set(VALID_PROVIDERS) == {"openai", "anthropic"}
