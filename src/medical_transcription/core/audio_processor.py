"""Audio preprocessing and format conversion."""

from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
import tempfile
from typing import Tuple


class AudioProcessor:
    """Process and convert audio files for Whisper."""

    SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

    @staticmethod
    def validate_audio_file(audio_path: Path) -> None:
        """Validate that the audio file exists and has a supported format."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in AudioProcessor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {', '.join(AudioProcessor.SUPPORTED_FORMATS)}"
            )

    @staticmethod
    def convert_to_wav(audio_path: Path) -> Tuple[Path, bool]:
        """
        Convert audio file to WAV format required by Whisper.

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (wav_path, is_temp) where is_temp indicates if the file should be cleaned up
        """
        AudioProcessor.validate_audio_file(audio_path)

        # If already WAV, check if it needs conversion
        if audio_path.suffix.lower() == ".wav":
            # Check sample rate and channels
            info = sf.info(str(audio_path))
            if info.samplerate == 16000 and info.channels == 1:
                return audio_path, False

        # Load audio file with pydub
        audio = AudioSegment.from_file(str(audio_path))

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Convert to 16kHz sample rate
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        # Export as WAV
        audio.export(str(temp_path), format="wav")

        return temp_path, True

    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        info = sf.info(str(audio_path))
        return info.duration
