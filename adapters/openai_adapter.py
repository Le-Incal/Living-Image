"""
OpenAI GPT Image editing adapter.

Uses the Images API edit endpoint (image in + prompt -> image out)
with the configured GPT Image 2.5 model (Sunburst or Flare).
"""

from __future__ import annotations

import base64
import io
import math
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
        options: dict | None = None,
    ) -> tuple[bytes, dict]:
        api_key = os.environ.get(ENV_KEY, "")
        if not api_key:
            raise ValueError(f"{ENV_KEY} environment variable not set")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.entry.api_model,
            "images": [{"image_url": f"data:{mime_type};base64,{b64_image}"}],
            "prompt": RELIGHT_ONLY_PREFIX + prompt,
            "n": 1,
            "size": request_size(image_bytes),
            "quality": (options or {}).get("quality") or self.entry.options.get("quality", "high"),
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
            return base64.b64decode(results[0]["b64_json"]), _usage(data.get("usage"))

        raise RuntimeError("No image found in OpenAI response")


# Custom sizes: both sides divisible by 16, aspect between 1:3 and 3:1, and
# above 2560x1440 is experimental (OpenAI Images API reference, Nov 2026).
SIZE_STEP = 16
MAX_LONG, MAX_SHORT = 2560, 1440
MIN_PIXELS = 1024 * 1024  # the smallest standard size; an untested floor


def request_size(image_bytes: bytes) -> str:
    """An explicit output size that follows the input's shape, or "auto"."""
    try:
        from PIL import Image
        w, h = Image.open(io.BytesIO(image_bytes)).size
    except Exception:
        return "auto"
    if not w or not h or not (1 / 3 <= w / h <= 3):
        return "auto"
    long_side, short_side = max(w, h), min(w, h)
    scale = min(1.0, MAX_LONG / long_side, MAX_SHORT / short_side)
    if w * h * scale * scale < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (w * h))
        rw, rh = (math.ceil(w * scale / SIZE_STEP) * SIZE_STEP,
                  math.ceil(h * scale / SIZE_STEP) * SIZE_STEP)
    else:
        rw, rh = (round(w * scale / SIZE_STEP) * SIZE_STEP,
                  round(h * scale / SIZE_STEP) * SIZE_STEP)
    while max(rw, rh) > MAX_LONG or min(rw, rh) > MAX_SHORT:
        if max(rw, rh) > MAX_LONG:
            rw, rh = (rw - SIZE_STEP, rh) if rw >= rh else (rw, rh - SIZE_STEP)
        else:
            rw, rh = (rw - SIZE_STEP, rh) if rw < rh else (rw, rh - SIZE_STEP)
    return f"{rw}x{rh}"


def _openai_entries():
    from .base import MODEL_CATALOG
    return [e for e in MODEL_CATALOG if e.provider == "openai"]


def _usage(u: dict | None) -> dict:
    """Normalize Images API usage. Output tokens are image tokens."""
    if not u:
        return {}
    ind = u.get("input_tokens_details") or {}
    outd = u.get("output_tokens_details") or {}
    inp, out = u.get("input_tokens", 0), u.get("output_tokens", 0)
    in_text = ind.get("text_tokens", inp if not ind else 0)
    in_image = ind.get("image_tokens", 0)
    out_image = outd.get("image_tokens", out)
    out_text = outd.get("text_tokens", 0)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "input_text_tokens": in_text,
        "input_image_tokens": in_image,
        "output_image_tokens": out_image,
        "output_text_tokens": out_text,
    }
