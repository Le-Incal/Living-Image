"""
xAI Grok Imagine adapter for image relighting.

Uses the xAI Images Edits endpoint so the model edits the given image in place
rather than generating a new scene from the prompt.
"""

import base64
import os

import httpx

from .base import BaseAdapter, CatalogEntry, ModelInfo, default_entry, resolve_key

ENV_KEY = "XAI_API_KEY"
ALT_ENV_KEYS = ("GROK_API_KEY",)

# Prefix to force edit-in-place: same building/scene, only lighting changes
RELIGHT_ONLY_PREFIX = (
    "Only change the lighting in this exact image. "
    "Keep the same building, composition, camera angle, and every structural detail. "
)


class XAIAdapter(BaseAdapter):

    API_URL = "https://api.x.ai/v1/images/edits"

    def __init__(self, entry: CatalogEntry | None = None):
        self.entry = entry or default_entry("xai")

    @property
    def model_info(self) -> ModelInfo:
        e = self.entry
        return ModelInfo(
            id=e.id, name=e.name, provider="xai", cost_per_image=e.cost_per_image,
            api_model=e.api_model, env_key=ENV_KEY, default=e.default,
            alt_env_keys=ALT_ENV_KEYS,
        )

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        api_key = resolve_key(ENV_KEY, ALT_ENV_KEYS)
        if not api_key:
            raise ValueError(f"{ENV_KEY} or {ALT_ENV_KEYS[0]} environment variable not set")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.entry.api_model,
            "prompt": RELIGHT_ONLY_PREFIX + prompt,
            "image": {"url": f"data:{mime_type};base64,{b64_image}", "type": "image_url"},
            "response_format": "b64_json",
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                self.API_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            if resp.is_error:
                raise RuntimeError(f"xAI {resp.status_code}: {resp.text[:300]}")
            data = resp.json()

        results = data.get("data", [])
        if not results:
            raise RuntimeError("No data in xAI response")

        b64_result = results[0].get("b64_json") or results[0].get("image")
        if b64_result:
            return base64.b64decode(b64_result)

        url = results[0].get("url")
        if url:
            async with httpx.AsyncClient(timeout=60.0) as client:
                img_resp = await client.get(url)
                img_resp.raise_for_status()
                return img_resp.content

        raise RuntimeError("No image found in xAI response")
