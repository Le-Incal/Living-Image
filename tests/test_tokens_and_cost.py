"""
Tests for token capture, true cost, and persistent storage.

Every provider response states the tokens it used. We record them per
frame, price them with each model's published per-token rates, and
total them per run, so a run's cost is what the provider will bill
rather than a catalog estimate. Runs and images can live on a mounted
volume (GENERATED_DIR) so they survive redeploys.

Price sources (per 1M tokens), checked Nov 3, 2026:
- Nano Banana 2.1: $1.50 input, $30 image output, $7.50 text/thinking output
- Nano Banana Pro: $2 input, $120 image output, $12 text/thinking output
- GPT Image 2.5 Flare and Sunburst: $5 text input, $8 image input, $30 image output
"""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from adapters.base import AdapterRegistry, cost_from_usage
from main import app

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
IMG_B64 = base64.b64encode(b"result-bytes").decode()
ALL_KEYS = ["GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"]

GEMINI_RESP = {
    "candidates": [{"content": {"parts": [{"inlineData": {"data": IMG_B64}}]}}],
    "usageMetadata": {
        "promptTokenCount": 1433,
        "candidatesTokenCount": 1290,
        "thoughtsTokenCount": 921,
        "promptTokensDetails": [
            {"modality": "TEXT", "tokenCount": 323},
            {"modality": "IMAGE", "tokenCount": 1110},
        ],
        "candidatesTokensDetails": [
            {"modality": "IMAGE", "tokenCount": 1290},
        ],
    },
}
OPENAI_RESP = {
    "data": [{"b64_json": IMG_B64}],
    "usage": {
        "input_tokens": 1839,
        "output_tokens": 279,
        "total_tokens": 2118,
        "input_tokens_details": {"text_tokens": 330, "image_tokens": 1509},
    },
}


def _patch_post(monkeypatch, response_json):
    async def fake_post(self, url, json=None, headers=None, **kw):
        return httpx.Response(200, json=response_json, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)


@pytest.fixture
def keys(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")


@pytest.fixture
def store(tmp_path, monkeypatch):
    from metrics import RunMetricsStore
    s = RunMetricsStore(tmp_path / "runs.jsonl")
    monkeypatch.setattr(main, "metrics_store", s)
    return s


@pytest.fixture
def client():
    return TestClient(app)


# ──────────────────────────────────────────────
# Prices and the cost formula
# ──────────────────────────────────────────────

class TestPrices:

    @pytest.mark.parametrize("model_id", [
        "nano-banana-2.1", "nano-banana-pro", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst",
    ])
    def test_every_model_has_per_token_prices(self, model_id):
        prices = AdapterRegistry().get_adapter(model_id).model_info.prices
        assert set(prices) == {"text_in", "image_in", "image_out", "text_out"}
        assert prices["image_out"] > 0

    def test_published_rates(self):
        reg = AdapterRegistry()
        p = lambda m: reg.get_adapter(m).model_info.prices
        assert p("nano-banana-2.1") == {"text_in": 1.50, "image_in": 1.50, "image_out": 30.0, "text_out": 7.50}
        assert p("nano-banana-pro") == {"text_in": 2.0, "image_in": 2.0, "image_out": 120.0, "text_out": 12.0}
        for m in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
            assert p(m) == {"text_in": 5.0, "image_in": 8.0, "image_out": 30.0, "text_out": 0.0}


class TestCostFormula:

    def test_cost_sums_each_token_type_at_its_rate(self):
        usage = {"input_text_tokens": 1_000_000, "input_image_tokens": 1_000_000,
                 "output_image_tokens": 1_000_000, "output_text_tokens": 1_000_000}
        prices = {"text_in": 1, "image_in": 2, "image_out": 3, "text_out": 4}
        assert cost_from_usage(usage, prices) == pytest.approx(10.0)

    def test_missing_usage_costs_nothing(self):
        assert cost_from_usage({}, {"text_in": 5, "image_in": 8, "image_out": 30, "text_out": 0}) == 0.0

    def test_reproduces_openai_bill_for_flare_nov_3(self):
        """Nov 3 Flare run, billed $0.287 (0.021 text in + 0.157 image in + 0.109 out)."""
        prices = AdapterRegistry().get_adapter("gpt-image-2.5-flare").model_info.prices
        usage = {"input_text_tokens": 4200, "input_image_tokens": 19703, "output_image_tokens": 3627}
        assert cost_from_usage(usage, prices) == pytest.approx(0.287, abs=0.003)

    def test_reproduces_openai_bill_for_sunburst_nov_3(self):
        """Nov 3 Sunburst run, billed $0.630 (0.021 + 0.157 + 0.452)."""
        prices = AdapterRegistry().get_adapter("gpt-image-2.5-sunburst").model_info.prices
        usage = {"input_text_tokens": 4200, "input_image_tokens": 19703, "output_image_tokens": 15054}
        assert cost_from_usage(usage, prices) == pytest.approx(0.630, abs=0.003)


# ──────────────────────────────────────────────
# Adapters report usage
# ──────────────────────────────────────────────

class TestGeminiUsage:

    async def test_parses_modalities_and_thinking(self, keys, monkeypatch):
        _patch_post(monkeypatch, GEMINI_RESP)
        img, usage = await AdapterRegistry().get_adapter("nano-banana-2.1").edit_image_with_usage(FAKE_PNG, "p")
        assert img == b"result-bytes"
        assert usage == {
            "input_tokens": 1433, "output_tokens": 2211,
            "input_text_tokens": 323, "input_image_tokens": 1110,
            "output_image_tokens": 1290, "output_text_tokens": 921,
        }

    async def test_falls_back_when_details_missing(self, keys, monkeypatch):
        resp = {"candidates": GEMINI_RESP["candidates"],
                "usageMetadata": {"promptTokenCount": 1400, "candidatesTokenCount": 1290}}
        _patch_post(monkeypatch, resp)
        _, usage = await AdapterRegistry().get_adapter("nano-banana-pro").edit_image_with_usage(FAKE_PNG, "p")
        assert usage["input_text_tokens"] == 1400
        assert usage["output_image_tokens"] == 1290
        assert usage["output_text_tokens"] == 0

    async def test_no_usage_block_gives_empty_usage(self, keys, monkeypatch):
        _patch_post(monkeypatch, {"candidates": GEMINI_RESP["candidates"]})
        _, usage = await AdapterRegistry().get_adapter("nano-banana-2.1").edit_image_with_usage(FAKE_PNG, "p")
        assert usage == {}

    async def test_edit_image_still_returns_bytes(self, keys, monkeypatch):
        _patch_post(monkeypatch, GEMINI_RESP)
        out = await AdapterRegistry().get_adapter("nano-banana-2.1").edit_image(FAKE_PNG, "p")
        assert out == b"result-bytes"


class TestOpenAIUsage:

    async def test_parses_text_and_image_input(self, keys, monkeypatch):
        _patch_post(monkeypatch, OPENAI_RESP)
        img, usage = await AdapterRegistry().get_adapter("gpt-image-2.5-flare").edit_image_with_usage(FAKE_PNG, "p")
        assert img == b"result-bytes"
        assert usage == {
            "input_tokens": 1839, "output_tokens": 279,
            "input_text_tokens": 330, "input_image_tokens": 1509,
            "output_image_tokens": 279, "output_text_tokens": 0,
        }

    async def test_edit_image_still_returns_bytes(self, keys, monkeypatch):
        _patch_post(monkeypatch, OPENAI_RESP)
        out = await AdapterRegistry().get_adapter("gpt-image-2.5-sunburst").edit_image(FAKE_PNG, "p")
        assert out == b"result-bytes"


# ──────────────────────────────────────────────
# Runs record tokens and true cost
# ──────────────────────────────────────────────

def _generate(client, model_id):
    return client.post(
        "/api/generate",
        files={"image": ("t.png", FAKE_PNG, "image/png")},
        data={"model_id": model_id},
    )


class TestRunTokensAndCost:

    def test_frames_carry_tokens_and_cost(self, client, store, keys, monkeypatch):
        _patch_post(monkeypatch, OPENAI_RESP)
        _generate(client, "gpt-image-2.5-flare")
        frames = store.all()[0]["frames"]
        prices = AdapterRegistry().get_adapter("gpt-image-2.5-flare").model_info.prices
        for f in frames:
            assert f["input_tokens"] == 1839
            assert f["output_tokens"] == 279
            expected = cost_from_usage(
                {"input_text_tokens": 330, "input_image_tokens": 1509, "output_image_tokens": 279}, prices)
            assert f["cost_usd"] == pytest.approx(expected, abs=1e-6)

    def test_run_totals(self, client, store, keys, monkeypatch):
        _patch_post(monkeypatch, GEMINI_RESP)
        _generate(client, "nano-banana-2.1")
        run = store.all()[0]
        assert run["input_tokens"] == 13 * 1433
        assert run["output_tokens"] == 13 * 2211
        assert run["output_image_tokens"] == 13 * 1290
        assert run["output_text_tokens"] == 13 * 921
        assert run["cost_usd"] == pytest.approx(sum(f["cost_usd"] for f in run["frames"]), abs=1e-6)
        assert "est_cost_usd" in run  # catalog estimate kept for comparison

    def test_gemini_thinking_billed_at_text_rate(self, client, store, keys, monkeypatch):
        _patch_post(monkeypatch, GEMINI_RESP)
        _generate(client, "nano-banana-2.1")
        per_frame = store.all()[0]["frames"][0]["cost_usd"]
        expected = (1433 * 1.50 + 1290 * 30 + 921 * 7.50) / 1_000_000
        assert per_frame == pytest.approx(expected, abs=1e-7)

    def test_failed_frames_add_no_tokens_or_cost(self, client, store, keys, monkeypatch):
        calls = {"n": 0}

        async def flaky(self, url, json=None, headers=None, **kw):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise httpx.HTTPError("boom")
            return httpx.Response(200, json=OPENAI_RESP, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", flaky)
        _generate(client, "gpt-image-2.5-flare")
        run = store.all()[0]
        assert run["frames_ok"] == 7
        assert run["input_tokens"] == 7 * 1839
        for f in run["frames"]:
            if f["status"] == "error":
                assert f["input_tokens"] == 0 and f["cost_usd"] == 0

    def test_public_response_has_no_tokens_or_cost(self, client, store, keys, monkeypatch):
        _patch_post(monkeypatch, OPENAI_RESP)
        body = _generate(client, "gpt-image-2.5-flare").json()
        text = json.dumps(body)
        for field in ("input_tokens", "output_tokens", "cost_usd"):
            assert field not in text


class TestSummaryAndAdmin:

    def test_summary_reports_true_cost_and_tokens(self, client, store, keys, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "t")
        _patch_post(monkeypatch, OPENAI_RESP)
        _generate(client, "gpt-image-2.5-flare")
        _generate(client, "gpt-image-2.5-flare")
        row = client.get("/api/admin/summary", headers={"X-Admin-Token": "t"}).json()[0]
        assert row["mean_input_tokens"] == 13 * 1839
        assert row["mean_output_tokens"] == 13 * 279
        assert row["mean_true_cost_usd"] == pytest.approx(store.all()[0]["cost_usd"], abs=1e-6)

    def test_summary_tolerates_runs_recorded_before_tokens(self, client, store, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "t")
        store.record({"job_id": "old", "model_id": "m", "frames_requested": 13, "frames_failed": 0,
                      "total_ms": 1, "est_cost_usd": 0.4, "frames": []})
        row = client.get("/api/admin/summary", headers={"X-Admin-Token": "t"}).json()[0]
        assert row["mean_true_cost_usd"] is None

    def test_admin_page_shows_tokens_and_true_cost(self):
        html = (main.STATIC_DIR / "admin.html").read_text()
        for field in ("mean_true_cost_usd", "cost_usd", "input_tokens", "output_tokens"):
            assert field in html


# ──────────────────────────────────────────────
# Persistent storage
# ──────────────────────────────────────────────

class TestStorageLocation:

    def test_generated_dir_from_env(self, tmp_path, monkeypatch):
        target = tmp_path / "data" / "generated_images"
        monkeypatch.setenv("GENERATED_DIR", str(target))
        assert main.resolve_generated_dir() == target
        assert target.is_dir()

    def test_generated_dir_default_is_repo_folder(self, monkeypatch):
        monkeypatch.delenv("GENERATED_DIR", raising=False)
        assert main.resolve_generated_dir().name == "generated_images"

    def test_metrics_live_with_images_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("METRICS_PATH", raising=False)
        assert main.resolve_metrics_path(tmp_path) == tmp_path / "run_metrics.jsonl"
