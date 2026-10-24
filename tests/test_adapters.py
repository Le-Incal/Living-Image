"""
Tests for the adapter registry and adapter interface.

Verifies that the adapter pattern is correctly implemented:
- Registry discovers and serves adapters
- All adapters conform to the BaseAdapter interface
- Model info is complete and consistent
- Adapters fail gracefully without API keys
"""

import pytest
from adapters.base import AdapterRegistry, BaseAdapter, ModelInfo
from adapters.gemini import GeminiAdapter
from adapters.openai_adapter import OpenAIAdapter
from adapters.xai import XAIAdapter


# ──────────────────────────────────────────────
# ModelInfo
# ──────────────────────────────────────────────

class TestModelInfo:

    def test_to_dict_has_required_fields(self):
        info = ModelInfo(id="test", name="Test Model", provider="test", cost_per_image=0.05)
        d = info.to_dict()
        assert "id" in d
        assert "name" in d
        assert "provider" in d
        assert "cost_per_image" in d

    def test_to_dict_values_match(self):
        info = ModelInfo(id="x", name="X Model", provider="xprov", cost_per_image=0.01)
        d = info.to_dict()
        assert d["id"] == "x"
        assert d["name"] == "X Model"
        assert d["provider"] == "xprov"
        assert d["cost_per_image"] == 0.01


# ──────────────────────────────────────────────
# AdapterRegistry
# ──────────────────────────────────────────────

class TestAdapterRegistry:

    def test_registry_has_gemini_registered(self):
        reg = AdapterRegistry()
        adapter = reg.get_adapter("gemini")
        assert adapter is not None

    def test_registry_returns_none_for_unknown(self):
        reg = AdapterRegistry()
        assert reg.get_adapter("nonexistent") is None

    def test_list_models_returns_model_info(self):
        reg = AdapterRegistry()
        models = reg.list_models()
        assert len(models) >= 1
        assert all(isinstance(m, ModelInfo) for m in models)

    def test_gemini_model_info_is_complete(self):
        reg = AdapterRegistry()
        adapter = reg.get_adapter("gemini")  # legacy alias -> default Gemini model
        info = adapter.model_info
        assert info.id == "nano-banana-2.1"
        assert info.provider == "gemini"
        assert info.cost_per_image > 0
        assert len(info.name) > 0


# ──────────────────────────────────────────────
# Adapter Interface Compliance
# ──────────────────────────────────────────────

class TestAdapterInterface:
    """Verify all three adapters implement BaseAdapter correctly."""

    @pytest.fixture(params=[GeminiAdapter, OpenAIAdapter, XAIAdapter])
    def adapter(self, request):
        return request.param()

    def test_is_base_adapter_subclass(self, adapter):
        assert isinstance(adapter, BaseAdapter)

    def test_has_model_info(self, adapter):
        info = adapter.model_info
        assert isinstance(info, ModelInfo)
        assert len(info.id) > 0
        assert len(info.name) > 0
        assert len(info.provider) > 0
        assert info.cost_per_image > 0

    def test_model_ids_are_unique(self):
        adapters = [GeminiAdapter(), OpenAIAdapter(), XAIAdapter()]
        ids = [a.model_info.id for a in adapters]
        assert len(ids) == len(set(ids)), f"Duplicate model IDs found: {ids}"

    def test_providers_are_distinct(self):
        adapters = [GeminiAdapter(), OpenAIAdapter(), XAIAdapter()]
        providers = [a.model_info.provider for a in adapters]
        assert len(providers) == len(set(providers)), f"Duplicate providers: {providers}"

    @pytest.mark.asyncio
    async def test_edit_image_fails_without_api_key(self, adapter, monkeypatch):
        """All adapters should raise when their API key env var is not set."""
        # Clear all possible API key env vars
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        with pytest.raises(Exception):
            await adapter.edit_image(b"fake_image_bytes", "relight this", "image/png")


# ──────────────────────────────────────────────
# Adapter-Specific: Cost Sanity
# ──────────────────────────────────────────────

class TestAdapterCosts:
    """Verify cost-per-image is within expected ranges for each provider."""

    def test_gemini_cost(self):
        cost = GeminiAdapter().model_info.cost_per_image
        assert 0.01 <= cost <= 0.10, f"Gemini cost {cost} seems wrong"

    def test_openai_cost(self):
        cost = OpenAIAdapter().model_info.cost_per_image
        assert 0.01 <= cost <= 0.20, f"OpenAI cost {cost} seems wrong"

    def test_xai_cost(self):
        cost = XAIAdapter().model_info.cost_per_image
        assert 0.01 <= cost <= 0.20, f"xAI cost {cost} seems wrong"

    def test_full_run_cost_is_reasonable(self):
        """13 images per run should cost less than $2 for any provider."""
        for adapter_cls in [GeminiAdapter, OpenAIAdapter, XAIAdapter]:
            cost = adapter_cls().model_info.cost_per_image * 13
            assert cost < 2.0, (
                f"{adapter_cls.__name__} full run costs ${cost:.2f}, seems too expensive"
            )
