"""
OpenAI GPT Image editing adapter.

Uses the Images API edit endpoint (image in + prompt -> image out)
with the configured GPT Image 2.5 model (Sunburst or Flare).
"""

import base64
import os

import httpx

from .base import BaseAdapter, CatalogEntry, ModelInfo, default_entry

ENV_KEY = "OPENAI_API_KEY"

# Reinforce relight-only so output keeps same composition and framing
RELIGHT_ONLY_PREFIX = "Relight only; preserve exact composition, framing, and all structure. "


class OpenAIAdapter(BaseAdapter):

    API_URL = "https://api.openai.com/v1/images/edits"

    def __init__(self, entry: CatalogEntry | None = None):
        # Bare construction defaults to the cheaper Flare model
        self.entry = entry or next(
            (e for e in _openai_entries() if e.id == "gpt-image-2.5-flare"),
            default_entry("openai"),
        )

    @property
    def model_info(self) -> ModelInfo:
        e = self.entry
        return ModelInfo(
            id=e.id, name=e.name, provider="openai", cost_per_image=e.cost_per_image,
            api_model=e.api_model, env_key=ENV_KEY, default=e.default,
        )

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        api_key = os.environ.get(ENV_KEY, "")
        if not api_key:
            raise ValueError(f"{ENV_KEY} environment variable not set")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.entry.api_model,
            "images": [{"image_url": f"data:{mime_type};base64,{b64_image}"}],
            "prompt": RELIGHT_ONLY_PREFIX + prompt,
            "n": 1,
            "size": "auto",
            "quality": self.entry.options.get("quality", "high"),
            "output_format": "png",
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                self.API_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            if resp.is_error:
                raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text[:300]}")
            data = resp.json()

        results = data.get("data", [])
        if results and results[0].get("b64_json"):
            return base64.b64decode(results[0]["b64_json"])

        raise RuntimeError("No image found in OpenAI response")


def _openai_entries():
    from .base import MODEL_CATALOG
    return [e for e in MODEL_CATALOG if e.provider == "openai"]
