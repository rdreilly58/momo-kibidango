"""Unit tests for DecoderSettings and ServerSettings."""

import pytest
import yaml
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from momo_kibidango.config.settings import DecoderSettings, ServerSettings


class TestDecoderSettingsDefaults:
    """Verify all default values are correct."""

    def test_defaults(self, default_settings):
        s = default_settings
        assert s.draft_model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert s.qualifier_model_id is None
        assert s.target_model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert s.draft_tokenizer_id is None
        assert s.qualifier_tokenizer_id is None
        assert s.target_tokenizer_id is None
        assert s.draft_dtype == "float16"
        assert s.qualifier_dtype == "float16"
        assert s.target_dtype == "float16"
        assert s.max_draft_tokens == 5
        assert s.temperature == 0.7
        assert s.top_p == 0.9
        assert s.stage1_threshold == 0.10
        assert s.stage2_threshold == 0.03
        assert s.adaptive_enabled is True
        assert s.adaptive_target_rate == 0.70
        assert s.adaptive_ema_alpha == 0.05
        assert s.adaptive_warmup == 20
        assert s.device == "auto"
        assert s.memory_headroom_gb == 2.0
        assert s.rate_limit_per_minute == 60
        assert s.max_prompt_length == 4096
        assert s.max_output_length == 4096
        assert s.request_timeout_seconds == 300


class TestDecoderSettingsFromYaml:
    """Test loading settings from YAML files."""

    def test_from_yaml(self, tmp_path):
        config = {
            "temperature": 1.0,
            "top_p": 0.8,
            "max_draft_tokens": 10,
            "draft_model_id": "my-org/small-model",
        }
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(config))

        settings = DecoderSettings.from_yaml(yaml_file)
        assert settings.temperature == 1.0
        assert settings.top_p == 0.8
        assert settings.max_draft_tokens == 10
        assert settings.draft_model_id == "my-org/small-model"
        # Unchanged defaults are preserved
        assert settings.target_model_id == "Qwen/Qwen2.5-7B-Instruct"

    def test_from_yaml_nested_decoder_key(self, tmp_path):
        config = {"decoder": {"temperature": 1.5, "top_p": 0.6}}
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text(yaml.dump(config))

        settings = DecoderSettings.from_yaml(yaml_file)
        assert settings.temperature == 1.5
        assert settings.top_p == 0.6

    def test_from_yaml_env_override(self, tmp_path, monkeypatch):
        config = {"temperature": 0.5}
        yaml_file = tmp_path / "env.yaml"
        yaml_file.write_text(yaml.dump(config))
        monkeypatch.setenv("MOMO_TEMPERATURE", "1.2")

        settings = DecoderSettings.from_yaml(yaml_file)
        assert settings.temperature == 1.2

    def test_from_yaml_empty_file(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        # Should work with all defaults
        settings = DecoderSettings.from_yaml(yaml_file)
        assert settings.temperature == 0.7


class TestDecoderSettingsValidation:
    """Test field validators."""

    def test_validation_temperature_range_too_high(self):
        with pytest.raises(ValidationError, match="temperature"):
            DecoderSettings(temperature=2.5)

    def test_validation_temperature_range_too_low(self):
        with pytest.raises(ValidationError, match="temperature"):
            DecoderSettings(temperature=-0.1)

    def test_validation_temperature_boundary_zero(self):
        s = DecoderSettings(temperature=0.0)
        assert s.temperature == 0.0

    def test_validation_temperature_boundary_two(self):
        s = DecoderSettings(temperature=2.0)
        assert s.temperature == 2.0

    def test_validation_top_p_range_too_high(self):
        with pytest.raises(ValidationError, match="top_p"):
            DecoderSettings(top_p=1.1)

    def test_validation_top_p_range_too_low(self):
        with pytest.raises(ValidationError, match="top_p"):
            DecoderSettings(top_p=-0.1)

    def test_validation_top_p_boundary_zero(self):
        s = DecoderSettings(top_p=0.0)
        assert s.top_p == 0.0

    def test_validation_top_p_boundary_one(self):
        s = DecoderSettings(top_p=1.0)
        assert s.top_p == 1.0

    def test_validation_thresholds_stage1_too_high(self):
        with pytest.raises(ValidationError, match="threshold"):
            DecoderSettings(stage1_threshold=1.5)

    def test_validation_thresholds_stage1_too_low(self):
        with pytest.raises(ValidationError, match="threshold"):
            DecoderSettings(stage1_threshold=-0.1)

    def test_validation_thresholds_stage2_too_high(self):
        with pytest.raises(ValidationError, match="threshold"):
            DecoderSettings(stage2_threshold=1.5)

    def test_validation_thresholds_stage2_too_low(self):
        with pytest.raises(ValidationError, match="threshold"):
            DecoderSettings(stage2_threshold=-0.01)

    def test_validation_max_draft_tokens_too_high(self):
        with pytest.raises(ValidationError, match="max_draft_tokens"):
            DecoderSettings(max_draft_tokens=21)

    def test_validation_max_draft_tokens_too_low(self):
        with pytest.raises(ValidationError, match="max_draft_tokens"):
            DecoderSettings(max_draft_tokens=0)

    def test_validation_max_draft_tokens_boundaries(self):
        s1 = DecoderSettings(max_draft_tokens=1)
        assert s1.max_draft_tokens == 1
        s20 = DecoderSettings(max_draft_tokens=20)
        assert s20.max_draft_tokens == 20


class TestDecoderSettingsResolveDevice:
    """Test resolve_device with mocked torch backends."""

    def test_resolve_device_explicit(self):
        s = DecoderSettings(device="cuda")
        assert s.resolve_device() == "cuda"

    def test_resolve_device_auto_cuda(self):
        s = DecoderSettings(device="auto")
        with patch("momo_kibidango.config.settings.get_device", return_value="cuda"):
            result = s.resolve_device()
        assert result == "cuda"

    def test_resolve_device_auto_mps(self):
        s = DecoderSettings(device="auto")
        with patch("momo_kibidango.config.settings.get_device", return_value="mps"):
            result = s.resolve_device()
        assert result == "mps"

    def test_resolve_device_auto_cpu(self):
        s = DecoderSettings(device="auto")
        with patch("momo_kibidango.config.settings.get_device", return_value="cpu"):
            result = s.resolve_device()
        assert result == "cpu"


class TestDecoderSettingsQualifier:
    """Test qualifier_model_id behavior."""

    def test_qualifier_none_means_two_model(self, default_settings):
        assert default_settings.qualifier_model_id is None

    def test_qualifier_set_means_three_model(self, three_model_settings):
        assert three_model_settings.qualifier_model_id == "microsoft/phi-2"


class TestServerSettings:
    """Test ServerSettings defaults and YAML loading."""

    def test_server_settings_defaults(self):
        s = ServerSettings()
        assert s.host == "0.0.0.0"
        assert s.port == 7779
        assert s.metrics_enabled is True

    def test_server_settings_from_yaml(self, tmp_path):
        config = {"server": {"host": "127.0.0.1", "port": 8080}}
        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text(yaml.dump(config))

        s = ServerSettings.from_yaml(yaml_file)
        assert s.host == "127.0.0.1"
        assert s.port == 8080

    def test_server_settings_env_override(self, tmp_path, monkeypatch):
        config = {"host": "0.0.0.0", "port": 7779}
        yaml_file = tmp_path / "server_env.yaml"
        yaml_file.write_text(yaml.dump(config))
        monkeypatch.setenv("MOMO_SERVER_PORT", "9999")

        s = ServerSettings.from_yaml(yaml_file)
        assert s.port == 9999
