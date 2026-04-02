"""Pydantic settings models for momo-kibidango configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator

from momo_kibidango.utils import get_device


class DecoderSettings(BaseModel):
    """Settings for the speculative decoding engine."""

    # ── Model identifiers ──────────────────────────────────────────────
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    qualifier_model_id: Optional[str] = None  # None = 2-model mode
    target_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # ── Per-model overrides ────────────────────────────────────────────
    draft_tokenizer_id: Optional[str] = None  # defaults to draft_model_id
    qualifier_tokenizer_id: Optional[str] = None  # defaults to qualifier_model_id
    target_tokenizer_id: Optional[str] = None  # defaults to target_model_id
    draft_dtype: str = "float16"
    qualifier_dtype: str = "float16"
    target_dtype: str = "float16"

    # ── Generation parameters ──────────────────────────────────────────
    max_draft_tokens: int = 5
    temperature: float = 0.7
    top_p: float = 0.9

    # ── Verification thresholds ────────────────────────────────────────
    stage1_threshold: float = 0.10
    stage2_threshold: float = 0.03

    # ── Adaptive threshold tuning ──────────────────────────────────────
    adaptive_enabled: bool = True
    adaptive_target_rate: float = 0.70
    adaptive_ema_alpha: float = 0.05
    adaptive_warmup: int = 20

    # ── Resource management ────────────────────────────────────────────
    device: str = "auto"
    memory_headroom_gb: float = 2.0

    # ── Safety / limits ────────────────────────────────────────────────
    rate_limit_per_minute: int = 60
    max_prompt_length: int = 4096
    max_output_length: int = 4096
    request_timeout_seconds: int = 300

    # ── Validators ─────────────────────────────────────────────────────
    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {v}")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {v}")
        return v

    @field_validator("stage1_threshold", "stage2_threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v

    @field_validator("adaptive_target_rate")
    @classmethod
    def _check_adaptive_target_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"adaptive_target_rate must be in [0, 1], got {v}")
        return v

    @field_validator("adaptive_ema_alpha")
    @classmethod
    def _check_adaptive_ema_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"adaptive_ema_alpha must be in [0, 1], got {v}")
        return v

    @field_validator("max_draft_tokens")
    @classmethod
    def _check_max_draft_tokens(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError(f"max_draft_tokens must be in [1, 20], got {v}")
        return v

    # ── Helpers ────────────────────────────────────────────────────────
    def resolve_device(self) -> str:
        """Return the concrete device string, resolving 'auto' if needed."""
        if self.device == "auto":
            return get_device()
        return self.device

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecoderSettings":
        """Load settings from a YAML file, with env-var overrides.

        Environment variables are checked with the prefix ``MOMO_`` and
        upper-cased field names (e.g. ``MOMO_TEMPERATURE``).  They take
        precedence over values in the YAML file.
        """
        path = Path(path)
        with open(path, "r") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # Flatten: allow a top-level "decoder" key in the YAML
        if "decoder" in raw and isinstance(raw["decoder"], dict):
            data = raw["decoder"]
        else:
            data = raw

        # Apply env-var overrides (MOMO_<FIELD_NAME>)
        for field_name in cls.model_fields:
            env_key = f"MOMO_{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                data[field_name] = env_val

        return cls(**data)


class ServerSettings(BaseModel):
    """Settings for the HTTP / API server layer."""

    host: str = "0.0.0.0"
    port: int = 7779
    metrics_enabled: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ServerSettings":
        """Load server settings from a YAML file, with env-var overrides."""
        path = Path(path)
        with open(path, "r") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        if "server" in raw and isinstance(raw["server"], dict):
            data = raw["server"]
        else:
            data = raw

        for field_name in cls.model_fields:
            env_key = f"MOMO_SERVER_{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                data[field_name] = env_val

        return cls(**data)
