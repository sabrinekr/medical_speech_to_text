"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from medical_transcription.models.clinical_summary import ClinicalSummary


class BaseLLMProvider(ABC):
    """Abstract base class for LLM provider implementations."""

    @abstractmethod
    def extract_clinical_summary(self, transcript: str, prompt: str) -> ClinicalSummary:
        """
        Extract structured clinical summary from transcript.

        Args:
            transcript: The transcribed medical dictation text
            prompt: The prompt template for extraction

        Returns:
            Structured ClinicalSummary object

        Raises:
            Exception: If extraction fails
        """
        pass
