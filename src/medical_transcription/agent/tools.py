"""Agent tools for audio processing and transcription."""

import logging
from pathlib import Path
from typing import Dict, Any
from langchain_core.tools import tool

from medical_transcription.core.audio_processor import AudioProcessor
from medical_transcription.core.transcriber import Transcriber
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


@tool
def process_audio_tool(audio_path: str) -> Dict[str, Any]:
    """Convert audio file to WAV format for transcription.

    This tool converts any supported audio format (MP3, M4A, OGG, FLAC, WAV)
    to standardized WAV format (16kHz, mono) required by Whisper.

    Args:
        audio_path: Path to the audio file

    Returns:
        Dictionary with:
            - wav_path: Path to converted WAV file (str)
            - duration: Audio duration in seconds (float)
            - is_temp: Whether file should be cleaned up (bool)
    """
    try:
        logger.info(f"Processing audio file: {audio_path}")
        processor = AudioProcessor()

        # Convert to WAV
        wav_path, is_temp = processor.convert_to_wav(Path(audio_path))

        # Get duration
        duration = processor.get_audio_duration(wav_path)

        logger.info(
            f"Audio processed successfully: {wav_path} "
            f"(duration={duration:.2f}s, temp={is_temp})"
        )

        return {
            "wav_path": str(wav_path),
            "duration": duration,
            "is_temp": is_temp
        }

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise RuntimeError(f"Failed to process audio file: {e}")


@tool
def transcribe_audio_tool(
    wav_path: str,
    language: str = "de",
    model_size: str | None = None
) -> Dict[str, Any]:
    """Transcribe German medical audio using Whisper.

    This tool uses faster-whisper to transcribe audio with voice activity
    detection and beam search for optimal accuracy.

    Args:
        wav_path: Path to WAV audio file
        language: Language code (default: "de" for German)
        model_size: Whisper model size (tiny, base, small, medium, large)
                   If None, uses Config.WHISPER_MODEL

    Returns:
        Dictionary with:
            - transcript: Full transcribed text (str)
            - segments: List of timestamped segments (List[Dict])
            - language: Detected language (str)
            - language_probability: Confidence in language detection (float)
            - duration: Audio duration in seconds (float)
    """
    try:
        logger.info(f"Transcribing audio: {wav_path} (language={language})")

        # Initialize transcriber with optional model size override
        transcriber = Transcriber(model_name=model_size)

        # Transcribe with VAD filtering
        result = transcriber.transcribe(
            Path(wav_path),
            language=language,
            beam_size=Config.WHISPER_BEAM_SIZE
        )

        logger.info(
            f"Transcription complete: {len(result['transcript'])} characters, "
            f"{len(result['segments'])} segments"
        )

        return result

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise RuntimeError(f"Failed to transcribe audio: {e}")
