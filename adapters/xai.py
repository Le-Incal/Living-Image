"""
xAI Grok Imagine adapter for image relighting.

Uses the xAI API's image editing endpoint.
"""

import os
import base64
import httpx
from .base import BaseAdapter, ModelInfo


class XAIAdapter(BaseAdapter):

    API_URL = "https://api.x.ai/v1/images/generations"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="xai",
            name="Grok Imagine (Aurora)",
            provider="xai",
            cost_per_image=0.10,
        )

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        api_key = os.environ.get("XAI_API_KEY", "")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": "grok-imagine-image",
            "prompt": prompt,
            "image_url": f"data:{mime_type};base64,{b64_image}",
            "n": 1,
            "image_format": "base64",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract image from response
        results = data.get("data", [])
        if not results:
            raise RuntimeError("No data in xAI response")

        b64_result = results[0].get("b64_json") or results[0].get("image")
        if b64_result:
            return base64.b64decode(b64_result)

        # If URL is returned instead, fetch it
        url = results[0].get("url")
        if url:
            async with httpx.AsyncClient(timeout=60.0) as client:
                img_resp = await client.get(url)
                img_resp.raise_for_status()
                return img_resp.content

        raise RuntimeError("No image found in xAI response")
