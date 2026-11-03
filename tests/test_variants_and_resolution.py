"""
Tests for the v3 benchmark plumbing.

- Matched resolution: OpenAI gets an explicit size that follows the input
  (multiples of 16, within the non-experimental 2560x1440 bound), instead
  of "auto", which gave Sunburst 1678 or 1679 pixel widths. Gemini gets
  imageConfig at 2K with the nearest supported aspect ratio, instead of
  its 1K default.
- /api/generate takes a hidden prompt_variant and, for OpenAI models only,
  a quality override. Both are recorded with the run, and the admin
  summary keeps variants apart.
"""

import base64
import io

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from main import app
from adapters import openai_adapter, gemini
from adapters.base import MODEL_CATALOG

ALL_KEYS = ["GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"]


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from metrics import RunMetricsStore
    s = RunMetricsStore(tmp_path / "runs.jsonl")
    monkeypatch.setattr(main, "metrics_store", s)
    return s


@pytest.fixture
def keys(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")


def _capture(monkeypatch, out_png=None, first_status=None, first_text=""):
    """Fake both providers; record every payload sent."""
    sent = []
    out = base64.b64encode(out_png or b"result").decode()

    async def fake_post(self, url, json=None, headers=None, **kw):
        sent.append({"url": url, "json": json})
        req = httpx.Request("POST", url)
        if first_status and len(sent) == 1:
            return httpx.Response(first_status, text=first_text, request=req)
        if "googleapis" in url:
            body = {"candidates": [{"content": {"parts": [{"inlineData": {"data": out}}]}}]}
        else:
            body = {"data": [{"b64_json": out}]}
        return httpx.Response(200, json=body, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return sent


def _entry(model_id):
    return next(e for e in MODEL_CATALOG if e.id == model_id)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ──────────────────────────────────────────────
# OpenAI explicit size
# ──────────────────────────────────────────────

class TestOpenAISize:

    @pytest.mark.parametrize("w, h, expected", [
        (2064, 1152, "2064x1152"),
        (1152, 2064, "1152x2064"),
        (1376, 768, "1376x768"),
        (4128, 2304, "2560x1424"),
    ])
    def test_size_follows_input(self, w, h, expected):
        assert openai_adapter.request_size(_png(w, h)) == expected

    @pytest.mark.parametrize("w, h", [(800, 450), (1000, 1000), (3000, 1100), (1500, 2000)])
    def test_size_rules(self, w, h):
        size = openai_adapter.request_size(_png(w, h))
        rw, rh = map(int, size.split("x"))
        assert rw % 16 == 0 and rh % 16 == 0
        assert max(rw, rh) <= 2560 and min(rw, rh) <= 1440
        assert rw * rh >= 1_000_000
        assert abs(rw / rh - w / h) / (w / h) < 0.03

    def test_unreadable_image_falls_back_to_auto(self):
        assert openai_adapter.request_size(b"not an image") == "auto"

    def test_extreme_aspect_falls_back_to_auto(self):
        assert openai_adapter.request_size(_png(4000, 1000)) == "auto"

    def test_payload_uses_explicit_size(self, keys, monkeypatch):
        sent = _capture(monkeypatch)
        a = openai_adapter.OpenAIAdapter(_entry("gpt-image-2.5-sunburst"))
        _run(a.edit_image_with_usage(_png(2064, 1152), "p"))
        assert sent[0]["json"]["size"] == "2064x1152"

    def test_quality_override(self, keys, monkeypatch):
        sent = _capture(monkeypatch)
        a = openai_adapter.OpenAIAdapter(_entry("gpt-image-2.5-sunburst"))
        _run(a.edit_image_with_usage(_png(2064, 1152), "p", options={"quality": "xhigh"}))
        assert sent[0]["json"]["quality"] == "xhigh"

    def test_catalog_quality_by_default(self, keys, monkeypatch):
        sent = _capture(monkeypatch)
        a = openai_adapter.OpenAIAdapter(_entry("gpt-image-2.5-sunburst"))
        _run(a.edit_image_with_usage(_png(2064, 1152), "p"))
        assert sent[0]["json"]["quality"] == "high"


# ──────────────────────────────────────────────
# Gemini imageConfig
# ──────────────────────────────────────────────

class TestGeminiImageConfig:

    @pytest.mark.parametrize("w, h, ratio", [
        (2064, 1152, "16:9"), (1000, 1000, "1:1"), (1152, 2064, "9:16"),
        (1600, 1200, "4:3"), (2520, 1080, "21:9"),
    ])
    def test_nearest_aspect_ratio(self, w, h, ratio):
        assert gemini.image_config(_png(w, h)) == {"imageSize": "2K", "aspectRatio": ratio}

    def test_unreadable_image_sets_size_only(self):
        assert gemini.image_config(b"nope") == {"imageSize": "2K"}

    def test_payload_carries_image_config(self, keys, monkeypatch):
        sent = _capture(monkeypatch)
        a = gemini.GeminiAdapter(_entry("nano-banana-pro"))
        _run(a.edit_image_with_usage(_png(2064, 1152), "p"))
        cfg = sent[0]["json"]["generationConfig"]
        assert cfg["imageConfig"] == {"imageSize": "2K", "aspectRatio": "16:9"}
        assert cfg["responseModalities"] == ["TEXT", "IMAGE"]

    def test_retries_once_without_image_config_if_rejected(self, keys, monkeypatch):
        sent = _capture(monkeypatch, first_status=400,
                        first_text='{"error": {"message": "Unknown field imageSize"}}')
        a = gemini.GeminiAdapter(_entry("nano-banana-2.1"))
        img, _ = _run(a.edit_image_with_usage(_png(2064, 1152), "p"))
        assert img == b"result"
        assert len(sent) == 2
        assert "imageConfig" not in sent[1]["json"]["generationConfig"]

    def test_other_errors_are_not_retried(self, keys, monkeypatch):
        sent = _capture(monkeypatch, first_status=429, first_text="quota")
        a = gemini.GeminiAdapter(_entry("nano-banana-2.1"))
        with pytest.raises(RuntimeError):
            _run(a.edit_image_with_usage(_png(2064, 1152), "p"))
        assert len(sent) == 1


# ──────────────────────────────────────────────
# /api/generate: variant and quality
# ──────────────────────────────────────────────

def _generate(client, model_id="gpt-image-2.5-flare", image=None, **data):
    return client.post(
        "/api/generate",
        files={"image": ("t.png", image or _png(2064, 1152), "image/png")},
        data={"model_id": model_id, **data},
    )


class TestGenerateVariant:

    def test_default_variant_is_v3_and_recorded(self, client, store, keys, monkeypatch):
        sent = _capture(monkeypatch)
        assert _generate(client).status_code == 200
        run = store.all()[-1]
        assert run["prompt_variant"] == "v3"
        assert all("AHEAD means into the scene" in s["json"]["prompt"] for s in sent)

    @pytest.mark.parametrize("variant", ["v2", "v3", "v3-true-sun", "v3-glow"])
    def test_variant_selects_prompt(self, client, store, keys, monkeypatch, variant):
        import prompts_v3
        sent = _capture(monkeypatch)
        assert _generate(client, prompt_variant=variant).status_code == 200
        assert store.all()[-1]["prompt_variant"] == variant
        expected = {prompts_v3.build(s, variant) for s in prompts_v3.generate_slots(variant)}
        got = {s["json"]["prompt"].removeprefix(openai_adapter.RELIGHT_ONLY_PREFIX) for s in sent}
        assert got == expected

    def test_unknown_variant_is_400(self, client, store, keys, monkeypatch):
        _capture(monkeypatch)
        assert _generate(client, prompt_variant="v9").status_code == 400

    def test_quality_override_for_openai(self, client, store, keys, monkeypatch):
        sent = _capture(monkeypatch)
        assert _generate(client, "gpt-image-2.5-sunburst", quality="xhigh").status_code == 200
        assert {s["json"]["quality"] for s in sent} == {"xhigh"}
        assert store.all()[-1]["quality"] == "xhigh"

    def test_catalog_quality_recorded_by_default(self, client, store, keys, monkeypatch):
        _capture(monkeypatch)
        _generate(client, "gpt-image-2.5-sunburst")
        assert store.all()[-1]["quality"] == "high"

    def test_gemini_has_no_quality(self, client, store, keys, monkeypatch):
        _capture(monkeypatch)
        _generate(client, "nano-banana-2.1")
        assert store.all()[-1]["quality"] is None

    @pytest.mark.parametrize("model_id, quality", [
        ("gpt-image-2.5-sunburst", "ultra"),
        ("nano-banana-2.1", "high"),
    ])
    def test_bad_quality_is_400(self, client, store, keys, monkeypatch, model_id, quality):
        _capture(monkeypatch)
        assert _generate(client, model_id, quality=quality).status_code == 400

    def test_output_size_recorded_per_frame(self, client, store, keys, monkeypatch):
        _capture(monkeypatch, out_png=_png(2064, 1152))
        _generate(client)
        assert {f["output_wh"] for f in store.all()[-1]["frames"]} == {"2064x1152"}

    def test_unreadable_output_size_is_blank(self, client, store, keys, monkeypatch):
        _capture(monkeypatch)
        _generate(client)
        assert {f["output_wh"] for f in store.all()[-1]["frames"]} == {""}

    def test_prompt_preview_takes_variant(self, client):
        rows = client.get("/api/prompt-preview?variant=v3-glow").json()
        assert len(rows) == 13
        assert "interior glow" in rows[-1]["prompt"]
        assert client.get("/api/prompt-preview?variant=v9").status_code == 400

    def test_prompt_preview_defaults_to_v3(self, client):
        rows = client.get("/api/prompt-preview").json()
        assert "AHEAD means into the scene" in rows[0]["prompt"]


# ──────────────────────────────────────────────
# Admin summary keeps variants apart
# ──────────────────────────────────────────────

class TestSummaryByVariant:

    def _seed(self, store):
        base = {"model_id": "gpt-image-2.5-sunburst", "total_ms": 1000,
                "frames_requested": 1, "frames_ok": 1, "frames_failed": 0,
                "est_cost_usd": 0.1, "frames": [{"ms": 100, "status": "success"}]}
        store.record({**base, "job_id": "old"})  # before variants existed
        store.record({**base, "job_id": "a", "prompt_variant": "v3", "quality": "high"})
        store.record({**base, "job_id": "b", "prompt_variant": "v3", "quality": "xhigh"})
        store.record({**base, "job_id": "c", "prompt_variant": "v3-glow", "quality": "high"})

    def test_rows_split_by_variant_and_quality(self, client, store, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "t")
        self._seed(store)
        rows = client.get("/api/admin/summary", headers={"X-Admin-Token": "t"}).json()
        keys = {(r["model_id"], r["prompt_variant"], r["quality"]) for r in rows}
        assert keys == {
            ("gpt-image-2.5-sunburst", "v2", None),
            ("gpt-image-2.5-sunburst", "v3", "high"),
            ("gpt-image-2.5-sunburst", "v3", "xhigh"),
            ("gpt-image-2.5-sunburst", "v3-glow", "high"),
        }
        assert all(r["runs"] == 1 for r in rows)


# ──────────────────────────────────────────────
# Hidden controls: URL options, admin columns
# ──────────────────────────────────────────────

class TestHiddenControls:

    def test_index_passes_url_options_through(self):
        html = (main.Path(main.__file__).parent / "static" / "index.html").read_text()
        assert "URLSearchParams" in html
        assert "'prompt_variant'" in html
        assert "'quality'" in html

    def test_admin_shows_variant_and_quality(self):
        html = (main.Path(main.__file__).parent / "static" / "admin.html").read_text()
        assert "prompt_variant" in html
        assert "quality" in html
