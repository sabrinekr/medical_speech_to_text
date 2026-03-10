"""Unit tests for ProviderFactory."""

import pytest
from unittest.mock import patch

from medical_transcription.llm.provider_factory import ProviderFactory
from medical_transcription.llm.ollama_provider import OllamaProvider


class TestProviderFactory:
    """Test ProviderFactory class."""

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_create_ollama_provider(self, mock_client):
        """Test creating Ollama provider."""
        provider = ProviderFactory.create("ollama")

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.1:8b"  # Default from Config

    def test_create_unsupported_provider_raises(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            ProviderFactory.create("unsupported_provider")

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_create_empty_provider_name_uses_default(self, mock_client):
        """Test that empty provider name uses default from Config."""
        # Empty string falls back to Config.LLM_PROVIDER which is "ollama"
        provider = ProviderFactory.create("")

        assert isinstance(provider, OllamaProvider)

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_create_ollama_case_insensitive(self, mock_client):
        """Test that provider name is case-insensitive."""
        provider1 = ProviderFactory.create("ollama")
        provider2 = ProviderFactory.create("OLLAMA")
        provider3 = ProviderFactory.create("Ollama")

        assert isinstance(provider1, OllamaProvider)
        assert isinstance(provider2, OllamaProvider)
        assert isinstance(provider3, OllamaProvider)
