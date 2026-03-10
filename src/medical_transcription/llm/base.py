"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict
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

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text response from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def generate_json(
        self, prompt: str, system_prompt: Optional[str] = None, schema: Optional[Dict] = None
    ) -> Dict:
        """
        Generate structured JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            schema: Optional JSON schema for validation

        Returns:
            Parsed JSON dictionary
        """
        pass
