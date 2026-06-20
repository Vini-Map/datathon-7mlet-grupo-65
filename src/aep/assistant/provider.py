"""Pluggable LLM provider interface (Stage 5).

The assistant talks to an LLM only through :class:`LLMProvider`, so the demo can
run fully offline (:class:`MockProvider`, deterministic, no API key) or against
the Anthropic API (:class:`AnthropicProvider`). In the Azure target architecture
the same interface maps to Azure OpenAI — no caller code changes.

Selection is driven by ``AEP_LLM_PROVIDER`` (``mock`` | ``anthropic``).
"""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod

from aep.config import get_settings


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Return a completion for the given system + user prompt."""


class MockProvider(LLMProvider):
    """Deterministic, offline provider.

    It does not call any model: it returns a structured, readable echo of the
    grounded context it is given. This keeps the demo reproducible and free of
    secrets while exercising the exact same code path as a real LLM.
    """

    name = "mock"

    def complete(self, system: str, prompt: str) -> str:
        body = textwrap.shorten(prompt.replace("\n", " "), width=600, placeholder=" ...")
        return (
            "[mock-llm] "
            + body
            + "\n\n(This deterministic summary is produced offline by MockProvider; "
            "set AEP_LLM_PROVIDER=anthropic with an API key for a generative answer.)"
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (used when an API key is configured)."""

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or get_settings().anthropic_model

    def complete(self, system: str, prompt: str) -> str:
        import anthropic  # imported lazily so the demo never needs the dependency

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        msg = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory: return the configured provider (defaults to settings)."""
    choice = (name or get_settings().llm_provider).lower()
    if choice == "anthropic":
        return AnthropicProvider()
    return MockProvider()
