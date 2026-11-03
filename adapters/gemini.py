"""
Gemini (Nano Banana family) adapter for image relighting.

Calls generateContent on the configured Gemini image model.
The API key is sent in the x-goog-api-key header, never in the URL.
"""

from __future__ import annotations

import base64
import os

import httpx

from .base import BaseAdapter, CatalogEntry, ModelInfo, default_entry

ENV_KEY = "GEMINI_API_KEY"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiAdapter(BaseAdapter):

    def __init__(self, entry: CatalogEntry | None = None):
        self.entry = entry or default_entry("gemini")

    @property
    def model_info(self) -> ModelInfo:
        e = self.entry
        return ModelInfo(
            id=e.id, name=e.name, provider="gemini", cost_per_image=e.cost_per_image,
            api_model=e.api_model, env_key=ENV_KEY, default=e.default,
            prices=dict(e.prices),
        )

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> bytes:
        image, _ = await self.edit_image_with_usage(image_bytes, prompt, mime_type)
        return image

    async def edit_image_with_usage(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> tuple[bytes, dict]:
        api_key = os.environ.get(ENV_KEY, "")
        if not api_key:
            raise ValueError(f"{ENV_KEY} environment variable not set")

        payload = {
            "contents": [
                {
                    "parts": [
                        {"inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{API_BASE}/{self.entry.api_model}:generateContent",
                json=payload,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            )
            if resp.is_error:
                raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:300]}")
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates in Gemini response")

        for part in candidates[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"]), _usage(data.get("usageMetadata"))

        raise RuntimeError("No image found in Gemini response")


def _by_modality(details) -> dict:
    out = {}
    for d in details or []:
        out[d.get("modality", "").upper()] = out.get(d.get("modality", "").upper(), 0) + d.get("tokenCount", 0)
    return out


def _usage(meta: dict | None) -> dict:
    """Normalize Gemini usageMetadata. Thinking tokens bill at the text-output rate."""
    if not meta:
        return {}
    prompt = meta.get("promptTokenCount", 0)
    cands = meta.get("candidatesTokenCount", 0)
    thoughts = meta.get("thoughtsTokenCount", 0)
    pin = _by_modality(meta.get("promptTokensDetails"))
    pout = _by_modality(meta.get("candidatesTokensDetails"))
    if pin:
        in_text, in_image = pin.get("TEXT", 0), pin.get("IMAGE", 0)
    else:
        in_text, in_image = prompt, 0
    if pout:
        out_image, out_text = pout.get("IMAGE", 0), pout.get("TEXT", 0)
    else:
        out_image, out_text = cands, 0
    return {
        "input_tokens": prompt,
        "output_tokens": cands + thoughts,
        "input_text_tokens": in_text,
        "input_image_tokens": in_image,
        "output_image_tokens": out_image,
        "output_text_tokens": out_text + thoughts,
    }
