"""
Living Image Crossfade Test Harness - FastAPI Server

Serves the frontend and handles image generation requests
across multiple AI model providers.
"""

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response

from prompts import generate_time_slots, build_prompt
from adapters.base import AdapterRegistry
from metrics import RunMetricsStore, clean_error

app = FastAPI(title="Living Image Crossfade Test Harness")

# Directory to store generated images
GENERATED_DIR = Path(__file__).parent / "generated_images"
GENERATED_DIR.mkdir(exist_ok=True)

# Adapter registry (singleton)
registry = AdapterRegistry()

# Backend-only run metrics (read via /api/admin/*)
metrics_store = RunMetricsStore(
    os.environ.get("METRICS_PATH") or (GENERATED_DIR / "run_metrics.jsonl")
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/keys-check")
async def keys_check():
    """
    Report which API keys are set (values never exposed).
    Use this to verify Railway env vars are loaded.
    """
    return {
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "XAI_API_KEY": bool(os.environ.get("XAI_API_KEY")),
        "GROK_API_KEY": bool(os.environ.get("GROK_API_KEY")),
    }


@app.get("/api/models")
async def list_models():
    """Return available model options for the dropdown."""
    return [m.to_dict() for m in registry.list_models()]


@app.get("/api/time-slots")
async def time_slots():
    """Return the 12 time slots with metadata."""
    slots = generate_time_slots()
    return [
        {
            "hour": s.hour,
            "label": s.label,
            "sun_elevation": s.sun_elevation,
            "color_temp_k": s.color_temp_k,
        }
        for s in slots
    ]


@app.get("/api/prompt-preview")
async def prompt_preview():
    """Preview all 12 prompts for debugging/iteration."""
    slots = generate_time_slots()
    return [
        {
            "hour": s.hour,
            "label": s.label,
            "prompt": build_prompt(s),
        }
        for s in slots
    ]


@app.post("/api/generate")
async def generate_variants(
    image: UploadFile = File(...),
    model_id: str = Form("nano-banana-2.1"),
):
    """
    Generate relit variants of the uploaded image with the chosen model.
    """
    adapter = registry.get_adapter(model_id)
    if adapter is None:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    info = adapter.model_info
    if not info.available:
        raise HTTPException(
            status_code=400,
            detail=f"{info.name} is unavailable: {info.env_key} is not set on the server",
        )

    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    # Create a job directory
    job_id = str(uuid.uuid4())[:8]
    job_dir = GENERATED_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save the original
    original_path = job_dir / "original.png"
    original_path.write_bytes(image_bytes)

    slots = generate_time_slots()
    timings: dict[int, float] = {}
    started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()

    # Fire all requests concurrently
    async def generate_one(slot, index):
        prompt = build_prompt(slot)
        t0 = time.perf_counter()
        try:
            result_bytes = await adapter.edit_image(image_bytes, prompt, mime_type)
            timings[index] = (time.perf_counter() - t0) * 1000
            filename = f"slot_{index:02d}_h{slot.hour}.png"
            filepath = job_dir / filename
            filepath.write_bytes(result_bytes)
            return {
                "index": index,
                "hour": slot.hour,
                "label": slot.label,
                "image_url": f"/api/images/{job_id}/{filename}",
                "status": "success",
            }
        except Exception as e:
            timings[index] = (time.perf_counter() - t0) * 1000
            err_msg = clean_error(e)
            logger.warning(
                "Generation failed for slot %s (hour %s): %s",
                index, slot.hour, err_msg,
            )
            return {
                "index": index,
                "hour": slot.hour,
                "label": slot.label,
                "image_url": None,
                "status": "error",
                "error": err_msg,
            }

    tasks = [generate_one(slot, i) for i, slot in enumerate(slots)]
    results = await asyncio.gather(*tasks)
    total_ms = (time.perf_counter() - run_start) * 1000

    # Sort by index
    results.sort(key=lambda r: r["index"])

    success_count = sum(1 for r in results if r["status"] == "success")

    run = {
        "job_id": job_id,
        "model_id": info.id,
        "model_name": info.name,
        "provider": info.provider,
        "started_at": started_at,
        "total_ms": round(total_ms, 1),
        "frames_requested": len(slots),
        "frames_ok": success_count,
        "frames_failed": len(slots) - success_count,
        "cost_per_image": info.cost_per_image,
        "est_cost_usd": round(success_count * info.cost_per_image, 4),
        "input_bytes": len(image_bytes),
        "frames": [
            {
                "hour": r["hour"],
                "ms": round(timings.get(r["index"], 0.0), 1),
                "status": r["status"],
                "error": r.get("error", ""),
            }
            for r in results
        ],
    }
    try:
        metrics_store.record(run)
    except OSError:
        logger.exception("Could not write run metrics")
    logger.info("run_metrics %s", json.dumps(run, separators=(",", ":")))

    return {
        "job_id": job_id,
        "model": info.id,
        "model_name": info.name,
        "cost_per_image": info.cost_per_image,
        "total": len(slots),
        "success": success_count,
        "errors": len(slots) - success_count,
        "original_url": f"/api/images/{job_id}/original.png",
        "variants": results,
    }


@app.get("/api/images/{job_id}/animation.gif")
async def get_animation_gif(job_id: str):
    """Build and return a looped animated GIF (8am–8pm with crossfade) for download."""
    import io
    from PIL import Image

    job_dir = GENERATED_DIR / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    slots = generate_time_slots()
    frames = []
    for i, slot in enumerate(slots):
        filename = f"slot_{i:02d}_h{slot.hour}.png"
        filepath = job_dir / filename
        if not filepath.exists():
            continue
        img = Image.open(filepath).convert("RGB")
        frames.append(img)

    if not frames:
        raise HTTPException(status_code=404, detail="No slot images found")

    # Crossfade: N blended frames between each pair
    N_BLEND = 4
    MAX_SIZE = 512
    composed = []
    for i in range(len(frames)):
        composed.append(frames[i])
        if i + 1 < len(frames):
            for k in range(1, N_BLEND + 1):
                t = k / (N_BLEND + 1)
                a = frames[i].copy()
                b = frames[i + 1].copy()
                if a.size != b.size:
                    b = b.resize(a.size, Image.Resampling.LANCZOS)
                blended = Image.blend(a, b, alpha=t)
                composed.append(blended)

    # Resize to max dimension
    w, h = composed[0].size
    if max(w, h) > MAX_SIZE:
        ratio = MAX_SIZE / max(w, h)
        nw, nh = int(w * ratio), int(h * ratio)
        composed = [f.resize((nw, nh), Image.Resampling.LANCZOS) for f in composed]

    # Save as looped GIF (duration in ms)
    buf = io.BytesIO()
    duration_ms = 180
    composed[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=composed[1:],
        loop=0,
        duration=duration_ms,
    )
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="image/gif",
        headers={"Content-Disposition": f'attachment; filename="living-image-{job_id}.gif"'},
    )


@app.get("/api/images/{job_id}/{filename}")
async def get_image(job_id: str, filename: str):
    """Serve a generated image."""
    filepath = GENERATED_DIR / job_id / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/png")


@app.get("/api/images/{job_id}/{filename}/upscale")
async def upscale_image(job_id: str, filename: str):
    """Return a 2x upscaled version of the image for download."""
    import io
    from PIL import Image

    filepath = GENERATED_DIR / job_id / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(filepath).convert("RGB")
    w, h = img.size
    upscaled = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    upscaled.save(buf, format="PNG")
    download_name = filename.removesuffix(".png") + "_2x.png"
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


# Serve the frontend
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")



# ──────────────────────────────────────────────
# Admin: backend-only run metrics
# ──────────────────────────────────────────────

def _require_admin(request: Request) -> None:
    """404 unless ADMIN_TOKEN is configured; 403 unless the header matches."""
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    supplied = request.headers.get("X-Admin-Token", "")
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/api/admin/runs")
async def admin_runs(request: Request):
    _require_admin(request)
    runs = metrics_store.all()
    runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return runs


@app.get("/api/admin/summary")
async def admin_summary(request: Request):
    _require_admin(request)
    return metrics_store.summary()


@app.get("/admin")
async def admin_page():
    return FileResponse(STATIC_DIR / "admin.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
