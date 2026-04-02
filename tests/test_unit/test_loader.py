"""Unit tests for ModelLoader (with mocks to avoid actual model downloads)."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from momo_kibidango.models.loader import ModelLoader, LoadedModel, _DTYPE_BPP
from momo_kibidango.models.registry import ModelSpec, ModelTier
from momo_kibidango.exceptions import ModelLoadError


class TestResolveDevice:
    """Test ModelLoader._resolve_device static method."""

    def test_resolve_device_explicit(self):
        assert ModelLoader._resolve_device("cpu") == "cpu"
        assert ModelLoader._resolve_device("cuda") == "cuda"
        assert ModelLoader._resolve_device("mps") == "mps"

    @patch("momo_kibidango.models.loader.torch")
    def test_resolve_device_cuda(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        assert ModelLoader._resolve_device("auto") == "cuda"

    @patch("momo_kibidango.models.loader.torch")
    def test_resolve_device_mps(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert ModelLoader._resolve_device("auto") == "mps"

    @patch("momo_kibidango.models.loader.torch")
    def test_resolve_device_cpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert ModelLoader._resolve_device("auto") == "cpu"


class TestGuessParamCount:
    """Test the heuristic parameter count extraction."""

    def test_guess_7b(self):
        count = ModelLoader._guess_param_count("Qwen/Qwen2.5-7B-Instruct")
        assert count == pytest.approx(7e9)

    def test_guess_0_5b(self):
        count = ModelLoader._guess_param_count("Qwen/Qwen2.5-0.5B-Instruct")
        assert count == pytest.approx(0.5e9)

    def test_guess_125m(self):
        count = ModelLoader._guess_param_count("gpt2-125m")
        assert count == pytest.approx(125e6)

    def test_guess_no_match(self):
        count = ModelLoader._guess_param_count("some-model-name")
        assert count == pytest.approx(1e9)  # default fallback


class TestEstimateMemory:
    """Test memory estimation heuristic."""

    @patch("momo_kibidango.models.loader.torch")
    def test_estimate_memory_7b_float16(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")
        estimate = loader._estimate_memory("test/model-7b", "float16")
        # 7e9 params * 2 bytes / (1024^3) * 1.2 overhead
        expected = (7e9 * 2.0 / (1024 ** 3)) * 1.2
        assert estimate == pytest.approx(expected, rel=0.01)

    @patch("momo_kibidango.models.loader.torch")
    def test_estimate_memory_int4(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")
        estimate = loader._estimate_memory("test/model-7b", "int4")
        expected = (7e9 * 0.5 / (1024 ** 3)) * 1.2
        assert estimate == pytest.approx(expected, rel=0.01)


class TestLoadCaching:
    """Test that second load returns cached object."""

    @patch("momo_kibidango.models.loader.torch")
    def test_load_caches(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")

        spec = ModelSpec(model_id="test/model-0.5b", tier=ModelTier.DRAFT)

        # Pre-populate the cache to avoid actual model loading
        fake_loaded = LoadedModel(
            model=MagicMock(),
            tokenizer=MagicMock(),
            spec=spec,
            memory_footprint_gb=1.0,
        )
        loader._loaded["test/model-0.5b"] = fake_loaded

        result = loader.load(spec)
        assert result is fake_loaded

    @patch("momo_kibidango.models.loader.torch")
    def test_load_returns_cached_on_second_call(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")

        spec = ModelSpec(model_id="cached/model", tier=ModelTier.DRAFT)
        fake_loaded = LoadedModel(
            model=MagicMock(), tokenizer=MagicMock(),
            spec=spec, memory_footprint_gb=0.5,
        )
        loader._loaded["cached/model"] = fake_loaded

        result1 = loader.load(spec)
        result2 = loader.load(spec)
        assert result1 is result2


class TestUnload:
    """Test model unloading."""

    @patch("momo_kibidango.models.loader.torch")
    def test_unload(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")

        spec = ModelSpec(model_id="to-unload", tier=ModelTier.DRAFT)
        fake_loaded = LoadedModel(
            model=MagicMock(), tokenizer=MagicMock(),
            spec=spec, memory_footprint_gb=1.0,
        )
        loader._loaded["to-unload"] = fake_loaded

        loader.unload("to-unload")
        assert "to-unload" not in loader._loaded

    @patch("momo_kibidango.models.loader.torch")
    def test_unload_missing_model(self, mock_torch):
        """Unloading a model that is not loaded should not raise."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")
        loader.unload("nonexistent")  # Should not raise

    @patch("momo_kibidango.models.loader.torch")
    def test_unload_all(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")

        for name in ["model-a", "model-b", "model-c"]:
            spec = ModelSpec(model_id=name, tier=ModelTier.DRAFT)
            loader._loaded[name] = LoadedModel(
                model=MagicMock(), tokenizer=MagicMock(),
                spec=spec, memory_footprint_gb=0.5,
            )

        assert len(loader._loaded) == 3
        loader.unload_all()
        assert len(loader._loaded) == 0


class TestBuildFallbackChain:
    """Test dtype fallback chain construction."""

    @patch("momo_kibidango.models.loader.torch")
    def test_fallback_from_float16(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")
        chain = loader._build_fallback_chain("float16")
        assert chain == ["float16", "int8", "int4"]

    @patch("momo_kibidango.models.loader.torch")
    def test_fallback_from_int8(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")
        chain = loader._build_fallback_chain("int8")
        assert chain == ["int8", "int4"]

    @patch("momo_kibidango.models.loader.torch")
    def test_fallback_from_nonstandard(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        loader = ModelLoader(device="cpu")
        chain = loader._build_fallback_chain("bfloat16")
        assert chain == ["bfloat16", "float16", "int8", "int4"]
