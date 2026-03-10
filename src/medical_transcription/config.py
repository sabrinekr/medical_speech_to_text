"""Configuration management using environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Application configuration from environment variables."""

    # LLM Provider Configuration (Ollama only)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Whisper Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Output Configuration
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Request Timeouts
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))

    # Input Validation
    MIN_TRANSCRIPT_LENGTH: int = int(os.getenv("MIN_TRANSCRIPT_LENGTH", "10"))
    MAX_TRANSCRIPT_LENGTH: int = int(os.getenv("MAX_TRANSCRIPT_LENGTH", "50000"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))

    # Whisper parameters (previously hardcoded)
    WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    WHISPER_VAD_MIN_SILENCE_MS: int = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "500"))

    # Agent Configuration
    MAX_EXTRACTION_ITERATIONS: int = int(os.getenv("MAX_EXTRACTION_ITERATIONS", "2"))
    MAX_TRANSCRIPTION_ATTEMPTS: int = int(os.getenv("MAX_TRANSCRIPTION_ATTEMPTS", "1"))
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "0.8"))

    _validated: bool = False

    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if cls._validated:
            return

        # Validate LLM provider (only Ollama supported)
        if cls.LLM_PROVIDER != "ollama":
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}. Only 'ollama' is supported.")

        # Validate Whisper device
        if cls.WHISPER_DEVICE not in ["cpu", "cuda"]:
            raise ValueError(f"Unsupported Whisper device: {cls.WHISPER_DEVICE}")

        cls._validated = True

    @classmethod
    def ensure_output_dir(cls) -> None:
        """Ensure output directory exists."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_prompt_path(cls) -> Path:
        """Get path to prompt template."""
        return Path(__file__).parent.parent.parent / "prompts" / "clinical_extraction.txt"
