"""Unit tests for configuration management."""

import pytest
import os
from unittest.mock import patch

from medical_transcription.config import Config


class TestConfig:
    """Test Config class."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        assert Config.LLM_PROVIDER == "ollama"
        assert Config.OLLAMA_BASE_URL == "http://localhost:11434"
        assert Config.OLLAMA_MODEL == "llama3.1:8b"
        assert Config.WHISPER_MODEL == "small"
        assert Config.WHISPER_DEVICE == "cpu"
        assert Config.MAX_EXTRACTION_ITERATIONS == 2
        assert Config.MAX_TRANSCRIPTION_ATTEMPTS == 1
        assert Config.QUALITY_THRESHOLD == 0.8

    def test_validate_success(self):
        """Test successful validation with valid configuration."""
        Config.validate()  # Should not raise

    def test_validate_unsupported_provider(self):
        """Test validation fails with unsupported LLM provider."""
        # Reset validated flag
        Config._validated = False
        original_provider = Config.LLM_PROVIDER

        try:
            Config.LLM_PROVIDER = "unsupported"
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                Config.validate()
        finally:
            Config.LLM_PROVIDER = original_provider
            Config._validated = False

    def test_validate_unsupported_device(self):
        """Test validation fails with unsupported Whisper device."""
        # Reset validated flag
        Config._validated = False
        original_device = Config.WHISPER_DEVICE

        try:
            Config.WHISPER_DEVICE = "tpu"
            with pytest.raises(ValueError, match="Unsupported Whisper device"):
                Config.validate()
        finally:
            Config.WHISPER_DEVICE = original_device
            Config._validated = False

    def test_output_dir_created(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        from pathlib import Path
        original_dir = Config.OUTPUT_DIR
        try:
            Config.OUTPUT_DIR = Path(tmp_path / "new_output")
            Config.ensure_output_dir()
            assert (tmp_path / "new_output").exists()
        finally:
            Config.OUTPUT_DIR = original_dir

    @patch.dict(os.environ, {"OLLAMA_TIMEOUT": "120"})
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        # Need to reload Config to pick up env changes
        from importlib import reload
        import medical_transcription.config as config_module

        with patch.dict(os.environ, {"OLLAMA_TIMEOUT": "120"}):
            reload(config_module)
            from medical_transcription.config import Config as ReloadedConfig
            assert ReloadedConfig.OLLAMA_TIMEOUT == 120

    def test_config_thresholds(self):
        """Test configuration threshold values are reasonable."""
        assert 0 <= Config.QUALITY_THRESHOLD <= 1
        assert Config.MAX_EXTRACTION_ITERATIONS >= 1
        assert Config.MAX_TRANSCRIPTION_ATTEMPTS >= 1
        assert Config.MAX_UPLOAD_SIZE_MB > 0
        assert Config.OLLAMA_TIMEOUT > 0
