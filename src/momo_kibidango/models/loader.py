"""Memory-aware model loader with automatic dtype degradation fallback."""

from __future__ import annotations
import gc
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from momo_kibidango.exceptions import ModelLoadError
from momo_kibidango.models.registry import ModelSpec

logger = logging.getLogger(__name__)

# Bytes-per-parameter for each dtype used in memory estimation.
_DTYPE_BPP: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Ordered fallback chain when the preferred dtype does not fit in memory.
_FALLBACK_CHAIN: list[str] = ["float16", "int8", "int4"]


@dataclass
class LoadedModel:
    """Container for a loaded model, its tokenizer, and related metadata."""
    model: Any
    tokenizer: Any
    spec: ModelSpec
    memory_footprint_gb: float = 0.0


class ModelLoader:
    """Loads HuggingFace models with automatic dtype fallback and memory awareness.

    The loader attempts to load the model in the dtype specified by the
    ``ModelSpec``.  If the estimated memory footprint exceeds available memory
    (minus ``memory_headroom_gb``), it walks a degradation chain
    (float16 -> int8 -> int4) until a feasible dtype is found or raises
    ``ModelLoadError``.
    """

    def __init__(self, device: str = "auto", memory_headroom_gb: float = 2.0) -> None:
        self._device = self._resolve_device(device)
        self._headroom = memory_headroom_gb
        self._loaded: dict[str, LoadedModel] = {}
        logger.info(
            "ModelLoader initialised  device=%s  headroom=%.1f GB",
            self._device,
            self._headroom,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, spec: ModelSpec, device: str | None = None) -> LoadedModel:
        """Load a model following the dtype fallback chain if necessary.

        Parameters
        ----------
        spec:
            The ``ModelSpec`` describing which model and dtype to load.

        Returns
        -------
        LoadedModel
            A wrapper around the model, tokenizer, and memory metadata.

        Raises
        ------
        ModelLoadError
            When every dtype in the fallback chain has been exhausted.
        """
        if spec.model_id in self._loaded:
            logger.info("Model %s already loaded — returning cached instance", spec.model_id)
            return self._loaded[spec.model_id]

        # Build the fallback chain starting from the requested dtype.
        chain = self._build_fallback_chain(spec.dtype)
        logger.info(
            "Loading %s (%s tier) — fallback chain: %s",
            spec.model_id,
            spec.tier.value,
            chain,
        )

        last_error: Exception | None = None
        for dtype in chain:
            estimated = self._estimate_memory(spec.model_id, dtype)
            available = self._available_memory_gb()
            logger.info(
                "Trying dtype=%s  estimated=%.2f GB  available=%.2f GB  headroom=%.2f GB",
                dtype,
                estimated,
                available,
                self._headroom,
            )
            if estimated > (available - self._headroom):
                logger.warning(
                    "Skipping dtype=%s — estimated %.2f GB exceeds available %.2f GB "
                    "(headroom %.2f GB)",
                    dtype,
                    estimated,
                    available,
                    self._headroom,
                )
                continue

            try:
                loaded = self._do_load(spec, dtype)
                self._loaded[spec.model_id] = loaded
                logger.info(
                    "Successfully loaded %s as %s (%.2f GB)",
                    spec.model_id,
                    dtype,
                    loaded.memory_footprint_gb,
                )
                return loaded
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Failed to load %s with dtype=%s: %s", spec.model_id, dtype, exc
                )

        raise ModelLoadError(
            f"All dtype options exhausted for {spec.model_id}. "
            f"Last error: {last_error}"
        )

    def unload(self, model_id: str) -> None:
        """Unload a single model and free its memory."""
        loaded = self._loaded.pop(model_id, None)
        if loaded is None:
            logger.warning("Model %s is not loaded — nothing to unload", model_id)
            return

        del loaded.model
        del loaded.tokenizer
        self._gc_collect()
        logger.info("Unloaded model %s", model_id)

    def unload_all(self) -> None:
        """Unload every loaded model."""
        model_ids = list(self._loaded.keys())
        for model_id in model_ids:
            self.unload(model_id)
        logger.info("All models unloaded")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``'auto'`` to the best available device string."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            resolved = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
        logger.info("Auto-resolved device to '%s'", resolved)
        return resolved

    def _available_memory_gb(self) -> float:
        """Return an estimate of free accelerator / system memory in GB."""
        if self._device == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info()
                return free / (1024 ** 3)
            except Exception:  # noqa: BLE001
                logger.debug("torch.cuda.mem_get_info() unavailable, falling back to psutil")

        # For MPS or CPU, fall back to system RAM via psutil.
        try:
            import psutil  # noqa: WPS433 (optional import)
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except ImportError:
            logger.warning("psutil not installed — assuming 8 GB available memory")
            return 8.0

    def _estimate_memory(self, model_id: str, dtype: str) -> float:
        """Heuristic memory estimate in GB based on parameter count and dtype."""
        params = self._guess_param_count(model_id)
        bpp = _DTYPE_BPP.get(dtype, 2.0)
        # Multiply by 1.2 to account for KV-cache and framework overhead.
        estimate = (params * bpp / (1024 ** 3)) * 1.2
        return estimate

    @staticmethod
    def _guess_param_count(model_id: str) -> float:
        """Extract an approximate parameter count from the model name.

        Common patterns: ``7b``, ``1.5B``, ``0.5b``, ``125m``, ``350M``.
        Falls back to 1 billion if no pattern matches.
        """
        name = model_id.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*([bm])", name)
        if match:
            number = float(match.group(1))
            unit = match.group(2)
            if unit == "b":
                return number * 1e9
            return number * 1e6
        logger.debug(
            "Could not guess param count from '%s' — defaulting to 1B", model_id
        )
        return 1e9

    def _do_load(self, spec: ModelSpec, dtype: str) -> LoadedModel:
        """Perform the actual HuggingFace model and tokenizer load."""
        tokenizer_id = spec.effective_tokenizer_id
        logger.info("Loading tokenizer from %s", tokenizer_id)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=spec.trust_remote_code,
            revision=spec.revision,
        )

        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": spec.model_id,
            "trust_remote_code": spec.trust_remote_code,
            "revision": spec.revision,
        }

        if dtype in ("int8", "int4"):
            quant_config = self._make_quantization_config(dtype)
            load_kwargs["quantization_config"] = quant_config
            # quantised models require device_map for bitsandbytes
            load_kwargs["device_map"] = "auto"
        else:
            torch_dtype = self._torch_dtype(dtype)
            load_kwargs["torch_dtype"] = torch_dtype
            if self._device == "cuda":
                load_kwargs["device_map"] = "auto"

        logger.info("Loading model %s with dtype=%s", spec.model_id, dtype)
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Move to device when not using device_map (e.g. MPS, CPU).
        if "device_map" not in load_kwargs and self._device != "cpu":
            model = model.to(self._device)

        footprint = self._model_memory_footprint(model)
        logger.info("Model memory footprint: %.2f GB", footprint)

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            memory_footprint_gb=footprint,
        )

    # ------------------------------------------------------------------
    # Dtype / quantisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_dtype(dtype: str) -> torch.dtype:
        """Convert a string dtype to a ``torch.dtype``."""
        mapping: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype, torch.float16)

    @staticmethod
    def _make_quantization_config(dtype: str) -> BitsAndBytesConfig:
        """Create a ``BitsAndBytesConfig`` for int8 or int4 quantisation."""
        if dtype == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        # int4
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    @staticmethod
    def _model_memory_footprint(model: Any) -> float:
        """Return the model's actual GPU/CPU memory footprint in GB."""
        try:
            return model.get_memory_footprint() / (1024 ** 3)
        except AttributeError:
            # Fallback: sum parameter sizes.
            total_bytes = sum(
                p.nelement() * p.element_size() for p in model.parameters()
            )
            return total_bytes / (1024 ** 3)

    def _build_fallback_chain(self, preferred: str) -> list[str]:
        """Return the dtype fallback chain starting from *preferred*."""
        if preferred in _FALLBACK_CHAIN:
            idx = _FALLBACK_CHAIN.index(preferred)
            return _FALLBACK_CHAIN[idx:]
        # Non-standard dtype: try it first, then the full chain.
        return [preferred] + _FALLBACK_CHAIN

    @staticmethod
    def _gc_collect() -> None:
        """Force garbage collection and clear CUDA cache if available."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
