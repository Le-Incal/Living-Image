# Living Image - Crossfade Test Harness

Transform static architectural renders into dynamic time-of-day visualizations. Upload one image, generate 12 relit variants (7am to 6pm), and scrub through the day with a smooth crossfade slider.

## What This Is

This is a test harness to evaluate whether AI image editing APIs can reliably relight architectural images while preserving structural fidelity. Four models from two providers sit behind one adapter interface:

| Model | Provider | Est. cost/image | Est. cost/run (13) |
|-------|----------|-----------------|--------------------|
| Nano Banana 2.1 (default) | Google | $0.034 | $0.44 |
| Nano Banana Pro | Google | $0.134 | $1.74 |
| GPT Image 2.5 Flare | OpenAI | $0.05 | $0.65 |
| GPT Image 2.5 Sunburst | OpenAI | $0.17 | $2.21 |

Grok Imagine 2.0 (xAI) was tested and removed on Oct 31, 2026. xAI's entry-tier limit is 6 to 12 image requests per second; a run sends 13 at once, so every Grok run lost frames (7 of 13 returned in testing).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env and add your keys (you only need the ones you want to test)

# 3. Run
bash run.sh
# Or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 4. Open http://localhost:8000
```

## How It Works

### Upload and Generate

1. Select a model from the dropdown
2. Upload an architectural render (PNG or JPG)
3. The server fires 12 parallel API calls, one per hour from 7am to 6pm
4. Each call sends the original image with a time-specific relighting prompt

### The Slider

The slider maps linearly from 7am to 6pm. At any position, two adjacent keyframe images are blended by opacity. For example, at 9:30am the 9am image shows at full opacity with the 10am image at 50% opacity on top. The result is a smooth visual transition across the day.

### The Prompts

Every statement in a prompt comes from one solar model, so a prompt cannot contradict itself (enforced by `tests/test_prompt_consistency.py`):

- Sun position for 40N on the June solstice, solar noon at 1 PM: elevation from the spherical formula (peak 73.4 degrees), azimuth from the hour angle (15 degrees per hour, even motion across the frame)
- Frame convention: the camera faces north, so morning light comes from the RIGHT and evening light from the LEFT. The prompt uses frame language only, never compass words, because the building's real orientation is unknown
- Shadows always fall opposite the light, with length stated as a multiple of object height (cot of elevation)
- Color temperature, lighting, and sky text follow elevation, not the clock (about 2700K at 8 PM to 5700K at solar noon)
- A darker-image instruction only when the sun is below 8 degrees
- Explicit lists of what may change (light, shadows, sky, reflections) and what may not (geometry, windows, materials, vegetation, camera, text, objects)

Click "View prompts" in the UI to inspect the full prompt text for each time slot.

### Prompt v3 and test variants

v3 (`prompts_v3.py`, the default) keeps the same solar model and answers the second-round reviews of v2:

- Direction as a camera-relative clock reading in half-hour steps plus graded words ("slightly", "partly", "halfway"), so every hour reads differently. v2's eight 45-degree sectors gave 9, 10 and 11 AM identical words
- AHEAD and BEHIND defined relative to the camera
- The render's baked-in lighting is replaced, keeping contact shadows and shading in corners
- Below 8 degrees: long but soft-edged, low-contrast shadows and a darker image
- Shadow length and color from the unrounded elevation (8 PM is 13.5 times height)
- Windows keep their original lit or unlit state, so independently generated frames cannot blink
- Tree trunks and canopy outlines fixed, no crop, zoom, pan or recenter, white balance and reflectivity kept

| Variant | What changes |
|---------|--------------|
| `v2` | The previous prompt, unchanged, as the baseline |
| `v3` | Default. Even sun motion, 15 degrees per hour |
| `v3-true-sun` | True solar azimuth (80.3 degrees at 8 AM, fastest near noon) |
| `v3-glow` | Even warm glow in every window when the sun is below 20 degrees (7 and 8 PM) |

Pick a variant, and for OpenAI models a quality override, through the app URL. There is no UI control:

```
/?prompt_variant=v3-glow
/?prompt_variant=v3&quality=xhigh     (OpenAI only: low, medium, high, xhigh, max)
```

Each run records its variant, quality, input size and per-frame output size, and the admin summary keeps them apart.

### Matched resolution

- OpenAI: an explicit `size` that follows the input (sides divisible by 16, within 2560x1440), instead of `auto`, which gave Sunburst 1678 or 1679 pixel widths. A 2064x1152 render is requested at 2064x1152
- Gemini: `imageConfig` at `2K` with the nearest supported aspect ratio, instead of the 1K default (1376x768). If a model rejects `imageConfig`, the frame retries once without it

## Project Structure

```
living-image-crossfade/
├── server.py              # FastAPI server with all endpoints
├── prompts.py             # Prompt v2 (baseline) and the shared solar model
├── prompts_v3.py          # Prompt v3 and its test variants
├── adapters/
│   ├── base.py            # Abstract adapter interface and model registry
│   ├── gemini.py          # Google Gemini / Nano Banana adapter
│   ├── openai_adapter.py  # OpenAI GPT-4o image editing adapter
├── static/
│   └── index.html         # Frontend UI (upload, slider, crossfade viewer)
├── tests/
│   ├── test_prompts.py    # Tests for prompt generation and time slots
│   ├── test_adapters.py   # Tests for adapter interfaces and registry
│   └── test_server.py     # Tests for API endpoints
├── generated_images/      # Generated variants stored here per job
├── requirements.txt
├── .env.example           # API key template
└── run.sh                 # Quick start script
```

## Run Metrics (admin only)

Every generation run records per-frame timing, failed frames with their error, and estimated cost from successful frames. Metrics never appear in the public API or the app UI.

- Set `ADMIN_TOKEN` in Railway variables to enable. Without it the admin endpoints return 404.
- Open `/admin` (not linked from the app), enter the token. It is sent as an `X-Admin-Token` header, never in the URL.
- `GET /api/admin/summary`: per-model runs, failure rate, mean run time, median and p95 frame time, mean cost per run
- `GET /api/admin/runs`: every run, newest first, with per-frame detail
- Each frame records input and output tokens from the provider's response, split into text and image tokens (Gemini thinking tokens count as text output). True cost per run is those tokens at each model's published per-1M-token rates, set in `MODEL_CATALOG`; the catalog per-image estimate is kept alongside for comparison.
- Runs and images reset on redeploy unless they live on a volume: attach a Railway Volume at `/data` and set `GENERATED_DIR=/data/generated_images`. Each run is also logged as a `run_metrics {...}` line, which Railway keeps.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models` | GET | List available models for the dropdown |
| `/api/time-slots` | GET | Return the 12 time slots with metadata |
| `/api/prompt-preview` | GET | Preview all 13 prompts; `?variant=` picks a variant (default v3) |
| `/api/generate` | POST | Upload image + model choice (+ optional `prompt_variant`, `quality`), returns 13 variants |
| `/api/images/{job_id}/{filename}` | GET | Serve a generated image |

## Running Tests

```bash
python -m pytest tests/ -v
```

42 tests covering prompt generation, adapter interfaces, and server endpoints.

## What We're Evaluating

The critical question: **can these models relight an image while keeping the architecture pixel-perfect?**

When comparing results across models, look for:

- **Structural consistency**: Do windows, edges, and geometry stay in exactly the same position?
- **Shadow accuracy**: Do shadows move in a physically plausible direction across the day?
- **Color temperature**: Does the light feel right for each time of day?
- **Crossfade smoothness**: When scrubbing the slider, do transitions feel natural or ghostly?
- **Artifact introduction**: Does the model add, remove, or alter any architectural elements?

## Next Steps

- [ ] Wire up API keys and run first real test
- [ ] Iterate on prompt template based on results
- [ ] Add side-by-side comparison mode (two models on split screen)
- [ ] Add per-image prompt override for fine-tuning
- [ ] Evaluate IC-Light as a fourth option if API models lack structural fidelity
- [ ] If results validate, build toward production architecture
