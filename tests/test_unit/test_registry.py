"""Unit tests for ModelRegistry, ModelSpec, and ModelTier."""

import pytest
from dataclasses import FrozenInstanceError

from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier


class TestModelSpec:
    """Test ModelSpec dataclass."""

    def test_model_spec_effective_tokenizer_defaults_to_model_id(self):
        spec = ModelSpec(model_id="my-org/model", tier=ModelTier.DRAFT)
        assert spec.effective_tokenizer_id == "my-org/model"

    def test_model_spec_effective_tokenizer_uses_override(self):
        spec = ModelSpec(
            model_id="my-org/model",
            tier=ModelTier.DRAFT,
            tokenizer_id="my-org/tokenizer",
        )
        assert spec.effective_tokenizer_id == "my-org/tokenizer"

    def test_model_spec_frozen(self):
        spec = ModelSpec(model_id="my-org/model", tier=ModelTier.DRAFT)
        with pytest.raises(FrozenInstanceError):
            spec.model_id = "other"

    def test_model_spec_defaults(self):
        spec = ModelSpec(model_id="x", tier=ModelTier.TARGET)
        assert spec.tokenizer_id is None
        assert spec.dtype == "float16"
        assert spec.trust_remote_code is True
        assert spec.revision is None


class TestModelRegistry:
    """Test ModelRegistry operations."""

    def test_register_and_get(self):
        registry = ModelRegistry()
        spec = ModelSpec(model_id="test/model", tier=ModelTier.DRAFT)
        registry.register(spec)
        retrieved = registry.get_tier(ModelTier.DRAFT)
        assert retrieved is spec
        assert retrieved.model_id == "test/model"

    def test_get_tier_by_string(self):
        registry = ModelRegistry()
        spec = ModelSpec(model_id="test/model", tier=ModelTier.TARGET)
        registry.register(spec)
        retrieved = registry.get_tier("target")
        assert retrieved is spec

    def test_get_missing_tier(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="qualifier"):
            registry.get_tier(ModelTier.QUALIFIER)

    def test_get_missing_tier_by_string(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="qualifier"):
            registry.get_tier("qualifier")

    def test_has_tier_true(self):
        registry = ModelRegistry()
        spec = ModelSpec(model_id="test/model", tier=ModelTier.DRAFT)
        registry.register(spec)
        assert registry.has_tier(ModelTier.DRAFT) is True

    def test_has_tier_false(self):
        registry = ModelRegistry()
        assert registry.has_tier(ModelTier.QUALIFIER) is False

    def test_has_tier_by_string(self):
        registry = ModelRegistry()
        spec = ModelSpec(model_id="test/model", tier=ModelTier.DRAFT)
        registry.register(spec)
        assert registry.has_tier("draft") is True
        assert registry.has_tier("qualifier") is False

    def test_from_settings_two_model(self, default_settings):
        registry = ModelRegistry.from_settings(default_settings)
        assert registry.has_tier(ModelTier.DRAFT) is True
        assert registry.has_tier(ModelTier.QUALIFIER) is False
        assert registry.has_tier(ModelTier.TARGET) is True

    def test_from_settings_three_model(self, three_model_settings):
        registry = ModelRegistry.from_settings(three_model_settings)
        assert registry.has_tier(ModelTier.DRAFT) is True
        assert registry.has_tier(ModelTier.QUALIFIER) is True
        assert registry.has_tier(ModelTier.TARGET) is True
        assert registry.get_tier(ModelTier.QUALIFIER).model_id == "microsoft/phi-2"

    def test_tiers_property_two_model(self, default_registry):
        tiers = default_registry.tiers
        assert tiers == [ModelTier.DRAFT, ModelTier.TARGET]

    def test_tiers_property_three_model(self, three_model_registry):
        tiers = three_model_registry.tiers
        assert tiers == [ModelTier.DRAFT, ModelTier.QUALIFIER, ModelTier.TARGET]

    def test_register_overwrites(self):
        registry = ModelRegistry()
        spec1 = ModelSpec(model_id="first", tier=ModelTier.DRAFT)
        spec2 = ModelSpec(model_id="second", tier=ModelTier.DRAFT)
        registry.register(spec1)
        registry.register(spec2)
        assert registry.get_tier(ModelTier.DRAFT).model_id == "second"

    def test_from_settings_preserves_dtype(self):
        from momo_kibidango.config.settings import DecoderSettings

        settings = DecoderSettings(draft_dtype="bfloat16", target_dtype="int8")
        registry = ModelRegistry.from_settings(settings)
        assert registry.get_tier(ModelTier.DRAFT).dtype == "bfloat16"
        assert registry.get_tier(ModelTier.TARGET).dtype == "int8"
