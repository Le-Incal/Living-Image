"""
Base adapter interface and model registry.

All image generation adapters conform to the same interface so the
server can treat them interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Metadata about an available model."""
    id: str                  # unique key for API/frontend
    name: str                # display name
    provider: str            # "gemini", "openai", "xai"
    cost_per_image: float    # USD per generated image

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "cost_per_image": self.cost_per_image,
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

        Args:
            image_bytes: The original image as raw bytes.
            prompt: The relighting instruction prompt.
            mime_type: MIME type of the input image.

        Returns:
            The edited image as raw bytes (PNG or JPEG).

        Raises:
            Exception: If the API call fails.
        """
        ...


class AdapterRegistry:
    """Registry mapping model IDs to adapter instances."""

    def __init__(self):
        # Import here to avoid circular deps
        from .gemini import GeminiAdapter
        from .openai_adapter import OpenAIAdapter
        from .xai import XAIAdapter

        self._adapters: dict[str, BaseAdapter] = {
            "gemini": GeminiAdapter(),
            "openai": OpenAIAdapter(),
            "xai": XAIAdapter(),
        }

    def list_models(self) -> list[ModelInfo]:
        """Return info about all registered models."""
        return [a.model_info for a in self._adapters.values()]

    def get_adapter(self, model_id: str) -> BaseAdapter | None:
        """Get an adapter by model ID, or None if not found."""
        return self._adapters.get(model_id)
