"""
Tests for multi-model selection.

Covers the model catalog (latest image models per provider), key-gated
availability, the exact API model string each adapter sends, and the
frontend hooks for picking a model and rerendering.
"""

import base64
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from adapters.base import AdapterRegistry
from main import app

EXPECTED_MODELS = {
    # model_id: (provider, api_model, env_key)
    "nano-banana-2.1": ("gemini", "gemini-nano-banana-2.1", "GEMINI_API_KEY"),
    "nano-banana-pro": ("gemini", "gemini-3-pro-image", "GEMINI_API_KEY"),
    "gpt-image-2.5-sunburst": ("openai", "gpt-image-2.5-sunburst", "OPENAI_API_KEY"),
    "gpt-image-2.5-flare": ("openai", "gpt-image-2.5-flare", "OPENAI_API_KEY"),
    "grok-imagine-2.0": ("xai", "grok-imagine-image-2.0", "XAI_API_KEY"),
}
ALL_KEYS = ["GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"]
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def no_keys(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)


# ──────────────────────────────────────────────
# Catalog
# ──────────────────────────────────────────────

class TestCatalog:

    def test_registry_lists_latest_models_in_order(self):
        ids = [m.id for m in AdapterRegistry().list_models()]
        assert ids == list(EXPECTED_MODELS)

    @pytest.mark.parametrize("model_id", list(EXPECTED_MODELS))
    def test_each_model_targets_expected_api_model(self, model_id):
        provider, api_model, env_key = EXPECTED_MODELS[model_id]
        info = AdapterRegistry().get_adapter(model_id).model_info
        assert info.provider == provider
        assert info.api_model == api_model
        assert info.env_key == env_key

    def test_exactly_one_default_and_it_is_nano_banana_2_1(self):
        models = AdapterRegistry().list_models()
        defaults = [m.id for m in models if m.default]
        assert defaults == ["nano-banana-2.1"]

    def test_legacy_gemini_id_resolves_to_default(self):
        adapter = AdapterRegistry().get_adapter("gemini")
        assert adapter is not None
        assert adapter.model_info.id == "nano-banana-2.1"

    def test_no_outdated_models_remain(self):
        api_models = {m.api_model for m in AdapterRegistry().list_models()}
        for old in ["gemini-2.5-flash-image", "gpt-image-1", "grok-imagine-image"]:
            assert old not in api_models


# ──────────────────────────────────────────────
# /api/models availability
# ──────────────────────────────────────────────

class TestModelsAvailability:

    def test_all_models_listed_with_available_flag(self, client, no_keys):
        data = client.get("/api/models").json()
        assert [m["id"] for m in data] == list(EXPECTED_MODELS)
        assert all(m["available"] is False for m in data)

    def test_only_gemini_key_enables_gemini_models(self, client, no_keys, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        data = client.get("/api/models").json()
        available = {m["id"] for m in data if m["available"]}
        assert available == {"nano-banana-2.1", "nano-banana-pro"}

    def test_all_keys_enable_all_models(self, client, monkeypatch):
        for k in ALL_KEYS:
            monkeypatch.setenv(k, "test")
        data = client.get("/api/models").json()
        assert all(m["available"] for m in data)

    def test_models_payload_never_leaks_key_values(self, client, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "sk-secret-abc123")
        assert "sk-secret" not in client.get("/api/models").text

    def test_models_payload_includes_default_flag(self, client):
        data = client.get("/api/models").json()
        assert sum(1 for m in data if m["default"]) == 1


# ──────────────────────────────────────────────
# Adapters send the right model string
# ──────────────────────────────────────────────

class _Capture:
    def __init__(self, response_json):
        self.response_json = response_json
        self.url = None
        self.json = None
        self.headers = None


def _patch_post(monkeypatch, response_json):
    cap = _Capture(response_json)

    async def fake_post(self, url, json=None, headers=None, **kw):
        cap.url, cap.json, cap.headers = str(url), json, headers or {}
        return httpx.Response(200, json=cap.response_json, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return cap


IMG_B64 = base64.b64encode(b"result-bytes").decode()
GEMINI_RESP = {"candidates": [{"content": {"parts": [{"inlineData": {"data": IMG_B64}}]}}]}
OPENAI_RESP = {"data": [{"b64_json": IMG_B64}]}
XAI_RESP = {"data": [{"b64_json": IMG_B64}]}


class TestAdapterRequests:

    @pytest.mark.parametrize("model_id", ["nano-banana-2.1", "nano-banana-pro"])
    async def test_gemini_calls_model_in_url_with_header_key(self, model_id, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "secret-gem")
        cap = _patch_post(monkeypatch, GEMINI_RESP)
        out = await AdapterRegistry().get_adapter(model_id).edit_image(FAKE_PNG, "relight")
        assert out == b"result-bytes"
        assert f"/models/{EXPECTED_MODELS[model_id][1]}:generateContent" in cap.url
        # Key travels in a header, never in the URL (URLs end up in logs)
        assert "secret-gem" not in cap.url
        assert cap.headers.get("x-goog-api-key") == "secret-gem"

    @pytest.mark.parametrize("model_id", ["gpt-image-2.5-sunburst", "gpt-image-2.5-flare"])
    async def test_openai_sends_model_string(self, model_id, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "secret-oai")
        cap = _patch_post(monkeypatch, OPENAI_RESP)
        out = await AdapterRegistry().get_adapter(model_id).edit_image(FAKE_PNG, "relight")
        assert out == b"result-bytes"
        assert cap.json["model"] == EXPECTED_MODELS[model_id][1]
        # input_fidelity is a gpt-image-1.x parameter; not sent to 2.5 models
        assert "input_fidelity" not in cap.json

    async def test_xai_sends_model_string(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "secret-xai")
        cap = _patch_post(monkeypatch, XAI_RESP)
        out = await AdapterRegistry().get_adapter("grok-imagine-2.0").edit_image(FAKE_PNG, "relight")
        assert out == b"result-bytes"
        assert cap.json["model"] == "grok-imagine-image-2.0"


# ──────────────────────────────────────────────
# /api/generate
# ──────────────────────────────────────────────

class TestGenerateWithModel:

    def test_missing_key_rejected_before_any_calls(self, client, no_keys):
        resp = client.post(
            "/api/generate",
            files={"image": ("t.png", FAKE_PNG, "image/png")},
            data={"model_id": "gpt-image-2.5-flare"},
        )
        assert resp.status_code == 400
        assert "OPENAI_API_KEY" in resp.json()["detail"]

    def test_response_reports_model_used(self, client, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "x")
        _patch_post(monkeypatch, XAI_RESP)
        resp = client.post(
            "/api/generate",
            files={"image": ("t.png", FAKE_PNG, "image/png")},
            data={"model_id": "grok-imagine-2.0"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "grok-imagine-2.0"
        assert body["model_name"]
        assert body["success"] == body["total"]


# ──────────────────────────────────────────────
# Frontend hooks
# ──────────────────────────────────────────────

INDEX = (Path(__file__).parent.parent / "static" / "index.html").read_text()


class TestFrontend:

    def test_landing_has_model_picker(self):
        assert 'id="model-select"' in INDEX

    def test_viewer_has_rerender_controls(self):
        assert 'id="rerender-model-select"' in INDEX
        assert 'id="rerender-btn"' in INDEX
        assert 'id="run-tabs"' in INDEX

    def test_model_not_hardcoded(self):
        assert "formData.append('model_id', 'gemini')" not in INDEX

    def test_frontend_loads_models_from_api(self):
        assert "/api/models" in INDEX


class TestLandingLayout:

    def test_no_model_label_or_cost_line(self):
        assert 'id="model-cost"' not in INDEX
        assert '<label for="model-select">' not in INDEX

    def test_price_shown_right_aligned_inside_dropdown(self):
        # Custom dropdown renders name + price columns per option
        assert "ms-name" in INDEX and "ms-price" in INDEX
        assert 'role="listbox"' in INDEX or "'listbox'" in INDEX

    def test_native_selects_kept_as_value_source(self):
        # Hidden native selects stay so form logic and a11y fallbacks work
        assert 'id="model-select"' in INDEX
        assert 'id="rerender-model-select"' in INDEX

    def test_model_picker_sits_below_upload(self):
        assert INDEX.index('id="browse-btn"') < INDEX.index('id="model-select"')


class TestGrokKeyAlias:

    def test_grok_api_key_enables_grok(self, client, no_keys, monkeypatch):
        monkeypatch.delenv("GROK_API_KEY", raising=False)
        monkeypatch.setenv("GROK_API_KEY", "g")
        data = {m["id"]: m for m in client.get("/api/models").json()}
        assert data["grok-imagine-2.0"]["available"] is True

    async def test_xai_adapter_uses_grok_api_key(self, no_keys, monkeypatch):
        monkeypatch.setenv("GROK_API_KEY", "secret-grok")
        cap = _patch_post(monkeypatch, XAI_RESP)
        await AdapterRegistry().get_adapter("grok-imagine-2.0").edit_image(FAKE_PNG, "relight")
        assert cap.headers["Authorization"] == "Bearer secret-grok"

    def test_keys_check_reports_grok_key(self, client):
        assert "GROK_API_KEY" in client.get("/api/keys-check").json()


class TestControlsLayout:

    def _section(self, start_marker, end_marker):
        s = INDEX.index(start_marker)
        return INDEX[s:INDEX.index(end_marker, s)]

    def test_rerender_controls_share_row_with_new_image(self):
        footer = self._section('class="controls-footer"', 'id="btn-prompts"')
        assert 'id="btn-reset"' in footer
        assert 'id="rerender-model-select"' in footer
        assert 'id="rerender-btn"' in footer

    def test_run_tabs_have_their_own_row(self):
        row = self._section('class="model-row"', 'class="controls-footer"')
        assert 'id="run-tabs"' in row
        assert 'id="rerender-btn"' not in row

    def test_panel_wide_enough_for_five_tabs(self):
        import re
        m = re.search(r"\.controls-inner\s*\{[^}]*max-width:\s*(\d+)px", INDEX)
        assert m and int(m.group(1)) >= 760
