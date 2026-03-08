"""Speech-to-text transcription using Whisper."""

from pathlib import Path
from typing import Optional, Dict, Any
from faster_whisper import WhisperModel
import logging

from medical_transcription.config import Config

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribe audio files using faster-whisper."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        """
        Initialize transcriber with Whisper model.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            compute_type: Compute type (int8, float16, float32)
        """
        self.model_name = model_name or Config.WHISPER_MODEL
        self.device = device or Config.WHISPER_DEVICE
        self.compute_type = compute_type or Config.WHISPER_COMPUTE_TYPE
        self._model: Optional[WhisperModel] = None

        logger.info(
            f"Transcriber initialized with model={self.model_name}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    @property
    def model(self) -> WhisperModel:
        """Lazy load Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Whisper model loaded successfully")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str = "de",
        beam_size: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "de" for German)
            beam_size: Beam size for decoding (higher = more accurate but slower)

        Returns:
            Dictionary with transcript and metadata
        """
        logger.info(f"Transcribing audio file: {audio_path}")

        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,  # Voice activity detection filter
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Collect all segments
        transcript_segments = []
        full_text = []

        for segment in segments:
            transcript_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": segment.avg_logprob
            })
            full_text.append(segment.text.strip())

        transcript = " ".join(full_text)

        logger.info(f"Transcription complete. Length: {len(transcript)} characters")

        return {
            "transcript": transcript,
            "segments": transcript_segments,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }

    def transcribe_text_only(self, audio_path: Path, language: str = "de") -> str:
        """
        Transcribe audio file and return only the text.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "de" for German)

        Returns:
            Transcribed text
        """
        result = self.transcribe(audio_path, language=language)
        return result["transcript"]
