"""Configuration management using environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Application configuration from environment variables."""

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Whisper Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Output Configuration
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if cls.LLM_PROVIDER not in ["ollama"]:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")

        if cls.WHISPER_DEVICE not in ["cpu", "cuda"]:
            raise ValueError(f"Unsupported Whisper device: {cls.WHISPER_DEVICE}")

        # Ensure output directory exists
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_prompt_path(cls) -> Path:
        """Get path to prompt template."""
        return Path(__file__).parent.parent.parent / "prompts" / "clinical_extraction.txt"


# Validate configuration on import
Config.validate()
