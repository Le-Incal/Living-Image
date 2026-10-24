"""
Tests for Living Image API endpoints.

Smoke tests that verify endpoint availability, response structure,
and correct behavior without requiring live API keys.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


# ──────────────────────────────────────────────
# Health and Info Endpoints
# ──────────────────────────────────────────────

class TestHealthEndpoints:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_keys_check_returns_booleans(self, client):
        resp = client.get("/api/keys-check")
        assert resp.status_code == 200
        data = resp.json()
        assert "GEMINI_API_KEY" in data
        assert "OPENAI_API_KEY" in data
        assert "XAI_API_KEY" in data
        # Values must be booleans, never the actual key
        for key, val in data.items():
            assert isinstance(val, bool), f"{key} should be bool, got {type(val)}"

    def test_keys_check_never_exposes_values(self, client, monkeypatch):
        """Even with keys set, the endpoint must only return True/False."""
        monkeypatch.setenv("GEMINI_API_KEY", "sk-secret-test-key-12345")
        resp = client.get("/api/keys-check")
        data = resp.json()
        response_text = str(data)
        assert "sk-secret" not in response_text
        assert "12345" not in response_text


# ──────────────────────────────────────────────
# Model Listing
# ──────────────────────────────────────────────

class TestModelsEndpoint:

    def test_models_returns_list(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_model_has_required_fields(self, client):
        resp = client.get("/api/models")
        for model in resp.json():
            assert "id" in model
            assert "name" in model
            assert "provider" in model
            assert "cost_per_image" in model


# ──────────────────────────────────────────────
# Time Slots
# ──────────────────────────────────────────────

class TestTimeSlotsEndpoint:

    def test_returns_13_slots(self, client):
        resp = client.get("/api/time-slots")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 13

    def test_slot_structure(self, client):
        resp = client.get("/api/time-slots")
        for slot in resp.json():
            assert "hour" in slot
            assert "label" in slot
            assert "sun_elevation" in slot
            assert "color_temp_k" in slot

    def test_hours_span_8am_to_8pm(self, client):
        resp = client.get("/api/time-slots")
        hours = [s["hour"] for s in resp.json()]
        assert hours[0] == 8
        assert hours[-1] == 20


# ──────────────────────────────────────────────
# Prompt Preview
# ──────────────────────────────────────────────

class TestPromptPreviewEndpoint:

    def test_returns_13_prompts(self, client):
        resp = client.get("/api/prompt-preview")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 13

    def test_prompt_structure(self, client):
        resp = client.get("/api/prompt-preview")
        for item in resp.json():
            assert "hour" in item
            assert "label" in item
            assert "prompt" in item
            assert len(item["prompt"]) > 100

    def test_prompts_contain_preservation_instructions(self, client):
        """Every prompt must tell the model what NOT to change."""
        resp = client.get("/api/prompt-preview")
        for item in resp.json():
            assert "DO NOT CHANGE" in item["prompt"]


# ──────────────────────────────────────────────
# Generate Endpoint (Validation Only)
# ──────────────────────────────────────────────

class TestGenerateValidation:

    def test_rejects_unknown_model(self, client):
        """Requesting a model that doesn't exist should return 400."""
        # Create a minimal fake image
        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp = client.post(
            "/api/generate",
            files={"image": ("test.png", fake_image, "image/png")},
            data={"model_id": "nonexistent_model"},
        )
        assert resp.status_code == 400


# ──────────────────────────────────────────────
# Image Serving
# ──────────────────────────────────────────────

class TestImageServing:

    def test_missing_job_returns_404(self, client):
        resp = client.get("/api/images/nonexistent/test.png")
        assert resp.status_code == 404

    def test_missing_upscale_returns_404(self, client):
        resp = client.get("/api/images/nonexistent/test.png/upscale")
        assert resp.status_code == 404

    def test_missing_gif_returns_404(self, client):
        resp = client.get("/api/images/nonexistent/animation.gif")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Frontend Serving
# ──────────────────────────────────────────────

class TestFrontend:

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
