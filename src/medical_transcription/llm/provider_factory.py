"""Factory for creating LLM provider instances."""

import logging
from typing import Optional

from medical_transcription.llm.base import BaseLLMProvider
from medical_transcription.llm.ollama_provider import OllamaProvider
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating LLM provider instances."""

    @staticmethod
    def create(provider_name: Optional[str] = None) -> BaseLLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider_name: "ollama" (default: from Config.LLM_PROVIDER)

        Returns:
            OllamaProvider instance

        Raises:
            ValueError: If provider name is not "ollama"
        """
        provider_name = (provider_name or Config.LLM_PROVIDER).lower()

        logger.info(f"Creating LLM provider: {provider_name}")

        if provider_name == "ollama":
            return OllamaProvider()
        else:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Only 'ollama' is supported."
            )
