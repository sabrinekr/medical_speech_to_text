"""State schema for the medical transcription agent."""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """State that flows through the agent graph."""

    # Input
    audio_path: str
    wav_path: Optional[str]
    wav_duration: Optional[float]
    is_temp_wav: Optional[bool]

    # Transcription
    transcript: Optional[str]
    transcript_segments: Optional[List[Dict[str, Any]]]
    transcript_confidence: Optional[float]
    transcription_attempts: int
    max_transcription_attempts: int

    # Entity extraction output
    entities: Optional[Dict[str, Any]]

    # Intermediate clinical data
    patient_complaint: Optional[str]
    findings: Optional[List[str]]
    diagnosis: Optional[str]
    next_steps: Optional[List[str]]
    medications: Optional[List[str]]
    additional_notes: Optional[str]

    # Quality control
    validation_result: Optional[Dict[str, Any]]
    iteration_count: int
    max_iterations: int

    # Metadata
    provider_name: str
    model_name: str
    execution_path: List[str]

    # Final output
    final_summary: Optional[Dict[str, Any]]
