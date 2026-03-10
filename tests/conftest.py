"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Write minimal WAV header (44 bytes)
        tmp.write(b"RIFF")
        tmp.write((36).to_bytes(4, "little"))  # Chunk size
        tmp.write(b"WAVE")
        tmp.write(b"fmt ")
        tmp.write((16).to_bytes(4, "little"))  # Subchunk1 size
        tmp.write((1).to_bytes(2, "little"))  # Audio format (PCM)
        tmp.write((1).to_bytes(2, "little"))  # Num channels (mono)
        tmp.write((16000).to_bytes(4, "little"))  # Sample rate
        tmp.write((32000).to_bytes(4, "little"))  # Byte rate
        tmp.write((2).to_bytes(2, "little"))  # Block align
        tmp.write((16).to_bytes(2, "little"))  # Bits per sample
        tmp.write(b"data")
        tmp.write((0).to_bytes(4, "little"))  # Data size
        path = Path(tmp.name)

    yield path

    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    mock = Mock()
    mock.model = "test-model"
    mock.generate_json = Mock(return_value={"test": "response"})
    mock.generate_text = Mock(return_value="test response")
    return mock


@pytest.fixture
def sample_transcript():
    """Sample German medical transcript."""
    return """Der Patient klagt über Kopfschmerzen seit drei Tagen.
Fieber von 38.5 Grad. Keine Übelkeit. Blutdruck 120/80."""


@pytest.fixture
def sample_entities():
    """Sample extracted medical entities."""
    return {
        "symptoms": ["Kopfschmerzen", "Fieber"],
        "vital_signs": [
            {"name": "Körpertemperatur", "value": "38.5", "unit": "°C"},
            {"name": "Blutdruck", "value": "120/80", "unit": "mmHg"}
        ],
        "medications": [],
        "procedures": [],
        "temporal_info": ["seit drei Tagen"]
    }


@pytest.fixture
def sample_clinical_summary():
    """Sample clinical summary data."""
    return {
        "patient_complaint": "Kopfschmerzen seit drei Tagen mit Fieber",
        "findings": ["Fieber 38.5°C", "Blutdruck 120/80 mmHg", "keine Übelkeit"],
        "diagnosis": "Verdacht auf virale Infektion",
        "next_steps": ["Ruhe", "Flüssigkeitszufuhr", "Kontrolle in 2 Tagen"],
        "medications": ["Paracetamol 500mg bei Bedarf"],
        "additional_notes": "Patient ist bei klarem Bewusstsein"
    }


@pytest.fixture
def sample_agent_state():
    """Sample agent state for testing."""
    return {
        "audio_path": "/test/audio.wav",
        "wav_path": "/test/audio.wav",
        "is_temp_wav": False,
        "wav_duration": 10.5,
        "transcript": "Test transcript",
        "transcript_segments": [],
        "transcript_confidence": 0.95,
        "transcript_quality": {"is_acceptable": True, "quality_score": 0.9},
        "transcription_attempts": 1,
        "entities": {},
        "structured_findings": {},
        "diagnosis": {},
        "treatment_plan": {},
        "quality_validation": {},
        "iteration_count": 0,
        "final_summary": None,
        "execution_path": []
    }
