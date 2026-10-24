"""
Base adapter interface and model registry.

All image generation adapters conform to the same interface so the
server can treat them interchangeably. The registry is driven by a
single MODEL_CATALOG table: adding a model is a one-line change.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def resolve_key(env_key: str, alt_env_keys: tuple = ()) -> str:
    """Return the first non-empty value among env_key and its aliases."""
    for name in (env_key, *alt_env_keys):
        if name and os.environ.get(name):
            return os.environ[name]
    return ""


@dataclass
class ModelInfo:
    """Metadata about an available model."""
    id: str                  # unique key for API/frontend
    name: str                # display name
    provider: str            # "gemini", "openai", "xai"
    cost_per_image: float    # USD per generated image (estimate)
    api_model: str = ""      # exact model string sent to the provider
    env_key: str = ""        # env var holding the provider API key
    default: bool = False    # preselected in the UI
    alt_env_keys: tuple = () # other accepted env var names for the same key

    @property
    def available(self) -> bool:
        return bool(resolve_key(self.env_key, self.alt_env_keys))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "cost_per_image": self.cost_per_image,
            "api_model": self.api_model,
            "default": self.default,
            "available": self.available,
        }


class BaseAdapter(ABC):
    """Interface that all image generation adapters must implement."""

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Return metadata about this adapter's model."""
        ...

    @abstractmethod
    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        """
        Send an image + prompt to the API and return the edited image bytes.

        Raises:
            Exception: If the API key is missing or the API call fails.
        """
        ...


@dataclass
class CatalogEntry:
    id: str
    name: str
    provider: str
    api_model: str
    cost_per_image: float
    default: bool = False
    options: dict = field(default_factory=dict)  # provider-specific request options


# Latest image-editing models per provider (verified Oct 2026).
# Costs are per-image estimates at ~1K resolution, used for the UI cost hint.
MODEL_CATALOG: list[CatalogEntry] = [
    CatalogEntry("nano-banana-2.1", "Nano Banana 2.1", "gemini",
                 "gemini-nano-banana-2.1", 0.034, default=True),
    CatalogEntry("nano-banana-pro", "Nano Banana Pro", "gemini",
                 "gemini-3-pro-image", 0.134),
    CatalogEntry("gpt-image-2.5-sunburst", "GPT Image 2.5 Sunburst", "openai",
                 "gpt-image-2.5-sunburst", 0.17, options={"quality": "high"}),
    CatalogEntry("gpt-image-2.5-flare", "GPT Image 2.5 Flare", "openai",
                 "gpt-image-2.5-flare", 0.05, options={"quality": "medium"}),
    CatalogEntry("grok-imagine-2.0", "Grok Imagine 2.0", "xai",
                 "grok-imagine-image-2.0", 0.05),
]

# Old ids that older clients may still send
LEGACY_ALIASES = {"gemini": "nano-banana-2.1"}


def build_adapter(entry: CatalogEntry) -> BaseAdapter:
    from .gemini import GeminiAdapter
    from .openai_adapter import OpenAIAdapter
    from .xai import XAIAdapter

    cls = {"gemini": GeminiAdapter, "openai": OpenAIAdapter, "xai": XAIAdapter}[entry.provider]
    return cls(entry)


def default_entry(provider: str) -> CatalogEntry:
    """First catalog entry for a provider (used when an adapter is built bare)."""
    return next(e for e in MODEL_CATALOG if e.provider == provider)


class AdapterRegistry:
    """Registry mapping model IDs to adapter instances."""

    def __init__(self):
        self._adapters: dict[str, BaseAdapter] = {
            e.id: build_adapter(e) for e in MODEL_CATALOG
        }

    def list_models(self) -> list[ModelInfo]:
        """Return info about all registered models, in catalog order."""
        return [a.model_info for a in self._adapters.values()]

    def get_adapter(self, model_id: str) -> BaseAdapter | None:
        """Get an adapter by model ID (or legacy alias), or None if not found."""
        model_id = LEGACY_ALIASES.get(model_id, model_id)
        return self._adapters.get(model_id)
