"""LLM-based structured extraction from transcripts."""

from pathlib import Path
import logging
from typing import Optional

from medical_transcription.llm.base import BaseLLMProvider
from medical_transcription.llm.ollama_provider import OllamaProvider
from medical_transcription.models.clinical_summary import ClinicalSummary
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extract structured clinical summaries from transcripts using LLM."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        """
        Initialize LLM extractor.

        Args:
            provider: LLM provider instance (defaults to Ollama based on config)
        """
        if provider is None:
            provider = self._create_default_provider()
        self.provider = provider
        self.prompt_template = self._load_prompt_template()

    def _create_default_provider(self) -> BaseLLMProvider:
        """Create default LLM provider based on configuration."""
        provider_name = Config.LLM_PROVIDER.lower()

        if provider_name == "ollama":
            return OllamaProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        prompt_path = Config.get_prompt_path()

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract(self, transcript: str) -> ClinicalSummary:
        """
        Extract structured clinical summary from transcript.

        Args:
            transcript: The transcribed medical dictation text

        Returns:
            Structured ClinicalSummary object

        Raises:
            ValueError: If transcript is invalid
            Exception: If extraction fails
        """
        # Validate transcript is not empty
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")

        # Validate transcript length
        transcript_length = len(transcript.strip())
        if transcript_length < Config.MIN_TRANSCRIPT_LENGTH:
            raise ValueError(
                f"Transcript too short ({transcript_length} characters). "
                f"Minimum length: {Config.MIN_TRANSCRIPT_LENGTH} characters"
            )

        if transcript_length > Config.MAX_TRANSCRIPT_LENGTH:
            raise ValueError(
                f"Transcript too long ({transcript_length} characters). "
                f"Maximum length: {Config.MAX_TRANSCRIPT_LENGTH} characters"
            )

        logger.info(f"Extracting clinical summary from transcript ({transcript_length} characters)")

        try:
            summary = self.provider.extract_clinical_summary(
                transcript=transcript,
                prompt=self.prompt_template
            )

            logger.info("Clinical summary extraction successful")
            return summary

        except Exception as e:
            logger.error(f"Failed to extract clinical summary: {e}")
            raise
