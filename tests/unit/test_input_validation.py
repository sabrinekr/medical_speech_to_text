"""Unit tests for input validation."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import wave
import struct

from medical_transcription.agent.medical_agent import MedicalAgent
from medical_transcription.agent.nodes import AgentNodes
from medical_transcription.config import Config


class TestFileValidation:
    """Test file size validation in MedicalAgent."""

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_file_size_within_limit(self, mock_client, temp_audio_file):
        """Test that files within size limit are accepted."""
        # Create a small test WAV file (< 1 MB)
        with wave.open(str(temp_audio_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            # Write 1 second of audio (~32KB)
            wav.writeframes(struct.pack('<h', 0) * 16000)

        agent = MedicalAgent()

        # Mock the graph to avoid full execution
        with patch.object(agent.graph, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "final_summary": {
                    "patient_complaint": "Test",
                    "findings": ["Finding 1"],
                    "diagnosis": "Test diagnosis",
                    "next_steps": ["Step 1"],
                    "medications": ["Med 1"],
                    "additional_notes": "Test notes"
                },
                "execution_path": ["test"],
                "iteration_count": 0,
                "transcription_attempts": 1,
                "is_temp_wav": False,
                "wav_path": None
            }

            # Should not raise
            summary = agent.process(str(temp_audio_file))
            assert summary.patient_complaint == "Test"

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_file_size_exceeds_limit(self, mock_client, tmp_path):
        """Test that files exceeding size limit are rejected."""
        # Create a file larger than MAX_UPLOAD_SIZE_MB
        large_file = tmp_path / "large_audio.wav"

        # Create file larger than the limit (default 100 MB)
        # We'll just create a file with metadata claiming it's too large
        with wave.open(str(large_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack('<h', 0) * 16000)

        # Mock the file size to be larger than limit
        original_stat = large_file.stat

        class MockStat:
            def __init__(self):
                self.st_size = (Config.MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024

        with patch.object(Path, 'stat', return_value=MockStat()):
            agent = MedicalAgent()

            with pytest.raises(ValueError, match="Audio file too large"):
                agent.process(str(large_file))

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_nonexistent_file_raises(self, mock_client):
        """Test that nonexistent files raise FileNotFoundError."""
        agent = MedicalAgent()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            agent.process("/nonexistent/file.wav")


class TestTranscriptValidation:
    """Test transcript length validation in AgentNodes."""

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_transcript_within_length_limits(self, mock_client):
        """Test that transcripts within length limits are accepted."""
        from medical_transcription.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        nodes = AgentNodes(provider)

        # Create state with valid transcript length
        state = {
            "wav_path": "/test/audio.wav",
            "execution_path": [],
            "transcription_attempts": 0
        }

        # Mock transcribe_audio_tool
        with patch("medical_transcription.agent.nodes.transcribe_audio_tool") as mock_tool:
            mock_tool.invoke.return_value = {
                "transcript": "A" * 100,  # Valid length (between MIN and MAX)
                "segments": [
                    {"confidence": 0.9},
                    {"confidence": 0.85}
                ]
            }

            result = nodes.transcribe_audio_node(state)

            assert result["transcript"] == "A" * 100
            assert result["transcript_confidence"] > 0

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_transcript_too_short_raises(self, mock_client):
        """Test that transcripts below minimum length are rejected."""
        from medical_transcription.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        nodes = AgentNodes(provider)

        state = {
            "wav_path": "/test/audio.wav",
            "execution_path": [],
            "transcription_attempts": 0
        }

        # Mock transcribe_audio_tool with too-short transcript
        with patch("medical_transcription.agent.nodes.transcribe_audio_tool") as mock_tool:
            mock_tool.invoke.return_value = {
                "transcript": "Hi",  # Too short (< MIN_TRANSCRIPT_LENGTH)
                "segments": []
            }

            with pytest.raises(ValueError, match="Transcript too short"):
                nodes.transcribe_audio_node(state)

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_transcript_too_long_raises(self, mock_client):
        """Test that transcripts exceeding maximum length are rejected."""
        from medical_transcription.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        nodes = AgentNodes(provider)

        state = {
            "wav_path": "/test/audio.wav",
            "execution_path": [],
            "transcription_attempts": 0
        }

        # Mock transcribe_audio_tool with too-long transcript
        with patch("medical_transcription.agent.nodes.transcribe_audio_tool") as mock_tool:
            # Create transcript longer than MAX_TRANSCRIPT_LENGTH
            long_transcript = "A" * (Config.MAX_TRANSCRIPT_LENGTH + 1)
            mock_tool.invoke.return_value = {
                "transcript": long_transcript,
                "segments": []
            }

            with pytest.raises(ValueError, match="Transcript too long"):
                nodes.transcribe_audio_node(state)

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_empty_transcript_raises(self, mock_client):
        """Test that empty transcripts are rejected."""
        from medical_transcription.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        nodes = AgentNodes(provider)

        state = {
            "wav_path": "/test/audio.wav",
            "execution_path": [],
            "transcription_attempts": 0
        }

        # Mock transcribe_audio_tool with empty transcript
        with patch("medical_transcription.agent.nodes.transcribe_audio_tool") as mock_tool:
            mock_tool.invoke.return_value = {
                "transcript": "",  # Empty
                "segments": []
            }

            with pytest.raises(ValueError, match="Transcript too short"):
                nodes.transcribe_audio_node(state)


class TestValidationErrorMessages:
    """Test that validation error messages are helpful."""

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_file_size_error_includes_actual_size(self, mock_client, tmp_path):
        """Test that file size error includes actual size."""
        large_file = tmp_path / "test.wav"
        with wave.open(str(large_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack('<h', 0) * 16000)

        # Mock file size
        class MockStat:
            st_size = 150 * 1024 * 1024  # 150 MB

        with patch.object(Path, 'stat', return_value=MockStat()):
            agent = MedicalAgent()

            try:
                agent.process(str(large_file))
                assert False, "Should have raised ValueError"
            except ValueError as e:
                error_msg = str(e)
                assert "150.00 MB" in error_msg
                assert "100 MB" in error_msg
                assert "splitting" in error_msg.lower()

    @patch("medical_transcription.llm.ollama_provider.ollama.Client")
    def test_transcript_length_error_includes_limits(self, mock_client):
        """Test that transcript length errors include limits."""
        from medical_transcription.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        nodes = AgentNodes(provider)

        state = {
            "wav_path": "/test/audio.wav",
            "execution_path": [],
            "transcription_attempts": 0
        }

        # Test too short
        with patch("medical_transcription.agent.nodes.transcribe_audio_tool") as mock_tool:
            mock_tool.invoke.return_value = {
                "transcript": "Hi",
                "segments": []
            }

            try:
                nodes.transcribe_audio_node(state)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                error_msg = str(e)
                assert "2 characters" in error_msg
                assert str(Config.MIN_TRANSCRIPT_LENGTH) in error_msg
