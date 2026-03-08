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

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Robustly extract JSON from LLM response.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        # Strategy 1: Try direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find outermost {} and parse
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(response_text[json_start:json_end])
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try to find JSON code block (```json ... ```)
        import re
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # All strategies failed
        logger.error(f"Could not extract valid JSON. Response: {response_text[:200]}...")
        raise ValueError(
            "Could not extract valid JSON from LLM response. "
            "The model may not be following the JSON format instruction."
        )

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

            # Parse JSON response using robust extraction
            clinical_data = self._extract_json_from_response(response_text)

            # Validate and create ClinicalSummary object
            try:
                summary = ClinicalSummary(**clinical_data)
            except ValueError as e:
                logger.error(f"Pydantic validation error: {e}")
                logger.error(f"Clinical data: {clinical_data}")
                raise ValueError(f"Invalid clinical data structure: {e}")
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
