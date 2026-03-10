"""Medical transcription agent - main entry point."""

import logging
from pathlib import Path
from typing import Optional
import tempfile

from medical_transcription.agent.graph import create_agent_graph
from medical_transcription.agent.state import AgentState
from medical_transcription.llm.provider_factory import ProviderFactory
from medical_transcription.models.clinical_summary import ClinicalSummary
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


class MedicalAgent:
    """Autonomous agent for end-to-end medical transcription and extraction."""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize medical agent.

        Args:
            provider: LLM provider name ("claude", "openai", "ollama")
                     If None, uses Config.LLM_PROVIDER
        """
        # Validate configuration
        Config.validate()

        # Create LLM provider
        self.provider_name = provider or Config.LLM_PROVIDER
        self.llm_provider = ProviderFactory.create(self.provider_name)

        # Create agent graph
        self.graph = create_agent_graph(self.llm_provider)

        logger.info(f"MedicalAgent initialized with provider: {self.provider_name}")

    def process(self, audio_file_path: str) -> ClinicalSummary:
        """
        Process audio file end-to-end using agent.

        This method:
        1. Converts audio to WAV format
        2. Transcribes audio using Whisper
        3. Assesses transcript quality (may re-transcribe if poor)
        4. Extracts medical entities
        5. Structures findings and diagnosis
        6. Plans treatment
        7. Validates quality (may refine if issues found)
        8. Synthesizes final clinical summary

        Args:
            audio_file_path: Path to audio file (any supported format: WAV, MP3, M4A, OGG, FLAC)

        Returns:
            Structured ClinicalSummary object

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format not supported
            Exception: If processing fails
        """
        audio_path = Path(audio_file_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting medical transcription for: {audio_path.name}")
        logger.info(f"Provider: {self.provider_name}")

        # Initialize state
        initial_state: AgentState = {
            # Input
            "audio_path": str(audio_path),
            "wav_path": None,
            "wav_duration": None,
            "is_temp_wav": None,
            # Transcription
            "transcript": None,
            "transcript_segments": None,
            "transcript_confidence": None,
            "transcription_attempts": 0,
            "max_transcription_attempts": Config.MAX_TRANSCRIPTION_ATTEMPTS,
            # Entity extraction
            "entities": None,
            # Intermediate clinical data
            "patient_complaint": None,
            "findings": None,
            "diagnosis": None,
            "next_steps": None,
            "medications": None,
            "additional_notes": None,
            # Quality control
            "validation_result": None,
            "iteration_count": 0,
            "max_iterations": Config.MAX_EXTRACTION_ITERATIONS,
            # Metadata
            "provider_name": self.provider_name,
            "model_name": getattr(self.llm_provider, "model", "unknown"),
            "execution_path": [],
            # Final output
            "final_summary": None
        }

        try:
            # Execute agent graph
            logger.info("Executing agent graph...")
            final_state = self.graph.invoke(initial_state)

            # Extract final summary
            final_summary_data = final_state["final_summary"]

            if not final_summary_data:
                raise ValueError("Agent failed to produce final summary")

            # Create ClinicalSummary object
            summary = ClinicalSummary(
                patient_complaint=final_summary_data["patient_complaint"],
                findings=final_summary_data["findings"],
                diagnosis=final_summary_data["diagnosis"],
                next_steps=final_summary_data["next_steps"],
                medications=final_summary_data["medications"],
                additional_notes=final_summary_data["additional_notes"]
            )

            logger.info("✅ Medical transcription completed successfully")
            logger.info(f"Execution path: {' -> '.join(final_state['execution_path'])}")
            logger.info(
                f"Quality metrics: "
                f"confidence={final_summary_data.get('confidence_score', 'N/A')}, "
                f"iterations={final_state['iteration_count']}, "
                f"transcription_attempts={final_state['transcription_attempts']}"
            )

            # Cleanup temporary WAV file if needed
            if final_state.get("is_temp_wav") and final_state.get("wav_path"):
                try:
                    wav_file = Path(final_state["wav_path"])
                    if wav_file.exists():
                        wav_file.unlink()
                        logger.debug(f"Cleaned up temporary WAV file: {wav_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary WAV file: {e}")

            return summary

        except Exception as e:
            logger.error(f"❌ Error during medical transcription: {e}")

            # Cleanup temporary WAV file on error
            if initial_state.get("is_temp_wav") and initial_state.get("wav_path"):
                try:
                    wav_file = Path(initial_state["wav_path"])
                    if wav_file.exists():
                        wav_file.unlink()
                        logger.debug(f"Cleaned up temporary WAV file: {wav_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary WAV file: {cleanup_error}")

            raise

    def get_provider_info(self) -> dict:
        """
        Get information about the current LLM provider.

        Returns:
            Dictionary with provider name and model
        """
        return {
            "provider": self.provider_name,
            "model": getattr(self.llm_provider, "model", "unknown"),
            "class": self.llm_provider.__class__.__name__
        }
