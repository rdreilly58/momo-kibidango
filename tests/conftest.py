"""Shared test fixtures for momo-kibidango."""

import sys
from unittest.mock import MagicMock

import pytest

# Mock transformers (heavyweight, rarely installed in CI) if not present.
# torch is expected to be installed (at least CPU-only).
for _mod_name in ("transformers",):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from momo_kibidango.config.settings import DecoderSettings
from momo_kibidango.models.registry import ModelRegistry, ModelSpec, ModelTier
from momo_kibidango.models.loader import LoadedModel
from momo_kibidango.monitoring.metrics import MetricsCollector
from momo_kibidango.core.adaptive import AdaptiveThreshold
from momo_kibidango.core.decoder import GenerationRequest


@pytest.fixture
def default_settings():
    return DecoderSettings()


@pytest.fixture
def three_model_settings():
    return DecoderSettings(qualifier_model_id="microsoft/phi-2")


@pytest.fixture
def default_registry(default_settings):
    return ModelRegistry.from_settings(default_settings)


@pytest.fixture
def three_model_registry(three_model_settings):
    return ModelRegistry.from_settings(three_model_settings)


@pytest.fixture
def metrics_collector():
    return MetricsCollector()


@pytest.fixture
def adaptive_threshold():
    return AdaptiveThreshold()


@pytest.fixture
def sample_request():
    return GenerationRequest(prompt="Hello world", max_new_tokens=32)


@pytest.fixture
def mock_model():
    """Mock transformer model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tok = MagicMock()
    tok.encode = MagicMock(return_value=[1, 2, 3])
    tok.decode = MagicMock(return_value="hello")
    tok.eos_token_id = 2
    tok.pad_token = None
    tok.eos_token = "</s>"
    return tok
