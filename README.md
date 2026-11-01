# Living Image - Crossfade Test Harness

Transform static architectural renders into dynamic time-of-day visualizations. Upload one image, generate 12 relit variants (7am to 6pm), and scrub through the day with a smooth crossfade slider.

## What This Is

This is a test harness to evaluate whether AI image editing APIs can reliably relight architectural images while preserving structural fidelity. We're testing three models side by side:

| Model | Provider | Cost/Image | Key Strength |
|-------|----------|-----------|--------------|
| Nano Banana (Gemini 2.5 Flash Image) | Google | ~$0.039 | 3D spatial understanding, scene preservation |
| GPT-4o Image | OpenAI | ~$0.08 | Battle-tested editing, massive training data |
| Grok Imagine (Aurora) | xAI | ~$0.10 | Fast generation, chainable edits |

A full 12-variant generation costs roughly $0.47 to $1.20 depending on the model.

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

## Project Structure

```
living-image-crossfade/
├── server.py              # FastAPI server with all endpoints
├── prompts.py             # Prompt template system and time slot generation
├── adapters/
│   ├── base.py            # Abstract adapter interface and model registry
│   ├── gemini.py          # Google Gemini / Nano Banana adapter
│   ├── openai_adapter.py  # OpenAI GPT-4o image editing adapter
│   └── xai.py             # xAI Grok Imagine adapter
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
- The metrics file resets on redeploy. Each run is also logged as a `run_metrics {...}` line, which Railway keeps.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models` | GET | List available models for the dropdown |
| `/api/time-slots` | GET | Return the 12 time slots with metadata |
| `/api/prompt-preview` | GET | Preview all 12 prompts (for debugging) |
| `/api/generate` | POST | Upload image + model choice, returns 12 variants |
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
