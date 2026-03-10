"""Unit tests for AudioProcessor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import wave
import struct

from medical_transcription.core.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_supported_formats(self):
        """Test that supported formats are defined correctly."""
        expected_formats = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]
        assert AudioProcessor.SUPPORTED_FORMATS == expected_formats

    def test_validate_audio_file_exists(self, temp_audio_file):
        """Test validation of existing audio file."""
        # Create a test WAV file
        with wave.open(str(temp_audio_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack('<h', 0) * 16000)

        # Should not raise any exception
        AudioProcessor.validate_audio_file(temp_audio_file)

    def test_validate_audio_file_nonexistent_raises(self):
        """Test validation fails for nonexistent file."""
        nonexistent = Path("/nonexistent/file.wav")
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            AudioProcessor.validate_audio_file(nonexistent)

    def test_validate_audio_file_unsupported_format_raises(self, tmp_path):
        """Test validation fails for unsupported format."""
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("not audio")

        with pytest.raises(ValueError, match="Unsupported audio format"):
            AudioProcessor.validate_audio_file(unsupported_file)

    def test_convert_to_wav_already_correct_format(self, temp_audio_file):
        """Test converting WAV file that's already in correct format."""
        # Create a test WAV file with correct format (16kHz, mono)
        with wave.open(str(temp_audio_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack('<h', 0) * 16000)

        result_path, is_temp = AudioProcessor.convert_to_wav(temp_audio_file)

        assert result_path == temp_audio_file
        assert is_temp is False

    def test_get_audio_duration(self, temp_audio_file):
        """Test getting audio duration."""
        # Create a test WAV file with 1 second of audio
        with wave.open(str(temp_audio_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack('<h', 0) * 16000)

        duration = AudioProcessor.get_audio_duration(temp_audio_file)
        assert duration > 0
        assert abs(duration - 1.0) < 0.1  # Should be close to 1 second
