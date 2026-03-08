"""Ollama LLM provider implementation."""

import json
import logging
from typing import Optional
import ollama

from medical_transcription.llm.base import BaseLLMProvider
from medical_transcription.models.clinical_summary import ClinicalSummary
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama-based LLM provider for local inference."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama3.1:8b")
            base_url: Ollama API base URL
        """
        self.model = model or Config.OLLAMA_MODEL
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.client = ollama.Client(host=self.base_url)

        logger.info(f"OllamaProvider initialized with model={self.model}, base_url={self.base_url}")

    def _sanitize_transcript(self, transcript: str) -> str:
        """
        Sanitize transcript to prevent prompt injection attacks.

        Args:
            transcript: Raw transcript text

        Returns:
            Sanitized transcript safe for prompt formatting
        """
        # Remove control characters (keep only printable chars + newlines/tabs/carriage returns)
        sanitized = ''.join(
            c for c in transcript
            if ord(c) >= 32 or c in '\n\t\r'
        )

        # Escape curly braces to prevent template injection
        sanitized = sanitized.replace('{', '{{').replace('}', '}}')

        # Log warning for suspicious patterns that might indicate injection attempts
        suspicious_patterns = ['ignore previous', 'ignore all', 'new instructions', 'disregard']
        for pattern in suspicious_patterns:
            if pattern.lower() in sanitized.lower():
                logger.warning(f"Suspicious pattern detected in transcript: '{pattern}'")

        return sanitized

    def extract_clinical_summary(self, transcript: str, prompt: str) -> ClinicalSummary:
        """
        Extract structured clinical summary from transcript using Ollama.

        Args:
            transcript: The transcribed medical dictation text
            prompt: The prompt template for extraction

        Returns:
            Structured ClinicalSummary object

        Raises:
            Exception: If extraction fails
        """
        # Sanitize transcript to prevent prompt injection
        sanitized_transcript = self._sanitize_transcript(transcript)

        # Format prompt with sanitized transcript
        full_prompt = prompt.format(transcript=sanitized_transcript)

        logger.info(f"Sending request to Ollama model: {self.model}")

        try:
            # Call Ollama with JSON format and timeout
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                format="json",  # Request JSON output
                options={
                    "temperature": 0.1,  # Low temperature for consistent output
                    "timeout": Config.OLLAMA_TIMEOUT,  # Timeout in seconds
                }
            )

            response_text = response["message"]["content"]
            logger.debug(f"Ollama response: {response_text}")

            # Parse JSON response
            try:
                clinical_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                # Try to extract JSON from response (in case there's extra text)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    clinical_data = json.loads(response_text[json_start:json_end])
                else:
                    raise ValueError("Could not extract valid JSON from response")

            # Validate and create ClinicalSummary object
            summary = ClinicalSummary(**clinical_data)
            logger.info("Clinical summary extracted successfully")

            return summary

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running and the model '{self.model}' is available. "
                f"Error: {e}"
            )
        except TimeoutError as e:
            logger.error(f"Ollama request timed out after {Config.OLLAMA_TIMEOUT}s")
            raise Exception(
                f"Ollama request timed out after {Config.OLLAMA_TIMEOUT} seconds. "
                f"The model may be overloaded or the transcript may be too long."
            )
        except Exception as e:
            logger.error(f"Error extracting clinical summary: {e}")
            raise
