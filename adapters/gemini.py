"""
Gemini (Nano Banana) adapter for image relighting.

Uses the Gemini 2.5 Flash Image model via Google's genai SDK.
"""

import os
import base64
import httpx
from .base import BaseAdapter, ModelInfo


class GeminiAdapter(BaseAdapter):

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="gemini",
            name="Nano Banana (Gemini 2.5 Flash Image)",
            provider="gemini",
            cost_per_image=0.039,
        )

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": b64_image,
                            }
                        },
                        {
                            "text": prompt,
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.API_URL}?key={api_key}",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract the image from the response
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates in Gemini response")

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part:
                img_data = part["inlineData"]["data"]
                return base64.b64decode(img_data)

        raise RuntimeError("No image found in Gemini response")
