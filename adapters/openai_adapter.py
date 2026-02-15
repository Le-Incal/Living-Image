"""
OpenAI GPT Image editing adapter.

Uses the OpenAI Images API edit endpoint (image in + prompt → image out).
Supports gpt-image-1, gpt-image-1.5, gpt-image-1-mini.
"""

import os
import base64
import httpx
from .base import BaseAdapter, ModelInfo

# Reinforce relight-only so output keeps same composition and framing
RELIGHT_ONLY_PREFIX = "Relight only; preserve exact composition, framing, and all structure. "


class OpenAIAdapter(BaseAdapter):

    API_URL = "https://api.openai.com/v1/images/edits"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="openai",
            name="GPT Image (gpt-image-1)",
            provider="openai",
            cost_per_image=0.08,
        )

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64_image}"

        # input_fidelity="high" keeps the same composition, geometry, and framing (relight only)
        payload = {
            "model": "gpt-image-1",
            "images": [{"image_url": image_url}],
            "prompt": RELIGHT_ONLY_PREFIX + prompt,
            "n": 1,
            "size": "auto",
            "quality": "high",
            "output_format": "png",
            "input_fidelity": "high",
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

        results = data.get("data", [])
        if not results:
            raise RuntimeError("No image in OpenAI response")

        b64_result = results[0].get("b64_json")
        if b64_result:
            return base64.b64decode(b64_result)

        raise RuntimeError("No image found in OpenAI response")
