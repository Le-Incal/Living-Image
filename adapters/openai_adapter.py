"""
OpenAI GPT-4o image editing adapter.

Uses the OpenAI API's image generation endpoint with image input.
"""

import os
import base64
import httpx
from .base import BaseAdapter, ModelInfo


class OpenAIAdapter(BaseAdapter):

    API_URL = "https://api.openai.com/v1/images/edits"
    CHAT_URL = "https://api.openai.com/v1/chat/completions"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="openai",
            name="GPT-4o Image",
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

        # Use chat completions with image input and image generation
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "modalities": ["text", "image"],
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.CHAT_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract image from response
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices in OpenAI response")

        message = choices[0].get("message", {})
        content = message.get("content", [])

        # Content can be a list of parts or a string
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        # Extract base64 from data URL
                        b64_part = url.split(",", 1)[1]
                        return base64.b64decode(b64_part)

        raise RuntimeError("No image found in OpenAI response")
