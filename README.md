# Medical Speech-to-Text System

Transform German medical dictations into structured clinical summaries using AI.

## Overview

This application processes German medical audio dictations through a two-stage pipeline:

1. **Speech-to-Text**: Transcribe audio files (WAV/MP3) using Whisper
2. **Structured Extraction**: Extract clinical summaries as JSON using a local LLM (Ollama)

**Key Features:**
- 🎤 High-quality German speech recognition with Whisper
- 🏥 Structured clinical data extraction (patient complaint, findings, diagnosis, next steps, medications)
- 🖥️ Both CLI and web interface (Streamlit)
- 🐳 Docker support for easy deployment
- 🔒 Privacy-focused with local processing (no external APIs required)

## Architecture

```
Audio File (WAV/MP3)
    ↓
[Audio Processor] → Convert to 16kHz mono WAV
    ↓
[Whisper Transcriber] → German speech-to-text
    ↓
[LLM Extractor] → Structured JSON extraction (via Ollama)
    ↓
Clinical Summary (JSON)
```

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager)
- **Ollama** (for local LLM inference)
- **Docker** (optional, for containerized deployment)
- **ffmpeg** (for audio processing)

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Ollama and Pull Model

```bash
# Install Ollama (see https://ollama.com for your OS)
# For Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull the required model
ollama pull llama3.1:8b
```

### 3. Setup Project

```bash
# Clone or navigate to project directory
cd /home/sabrinekr/kim/medical_speech_to_text

# Install dependencies with uv
uv sync

# Copy environment template
cp .env.example .env

# (Optional) Download Whisper model in advance
uv run python scripts/download_models.py

# (Optional) Generate sample audio for testing
uv run python scripts/create_sample_audio.py
```

## Usage

### CLI Interface

Basic usage:

```bash
# Transcribe audio file
uv run medical-transcribe audio.wav

# Save output to JSON file
uv run medical-transcribe audio.wav --output result.json

# Verbose output with progress details
uv run medical-transcribe audio.wav --verbose
```

Example with sample audio:

```bash
# Generate sample audio first
uv run python scripts/create_sample_audio.py

# Process it
uv run medical-transcribe examples/sample_medical_de.mp3 -o output.json -v
```

### Streamlit Web Interface

Launch the web interface:

```bash
uv run streamlit run src/medical_transcription/app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload audio files via drag & drop
- Play audio before processing
- View transcript and structured summary
- Download results as JSON

### Docker Usage

#### Build Docker Image

```bash
docker build -t medical-transcription .
```

#### Run CLI in Docker

```bash
# Process a file from examples directory
docker run -v $(pwd)/examples:/data medical-transcription /data/sample_medical_de.mp3

# Process with output
docker run -v $(pwd):/workspace medical-transcription \
  /workspace/audio.wav --output /workspace/result.json
```

#### Run Streamlit in Docker

```bash
docker run -p 8501:8501 \
  --entrypoint uv \
  medical-transcription \
  run streamlit run src/medical_transcription/app.py
```

**Note:** For the LLM to work in Docker, you'll need to connect to Ollama running on your host:

```bash
# On Linux, use host.docker.internal or your host IP
docker run -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  medical-transcription ...
```

## Configuration

Configuration is managed through environment variables. Create a `.env` file or set them directly:

```bash
# LLM Configuration
LLM_PROVIDER=ollama                    # LLM provider (only ollama supported currently)
OLLAMA_BASE_URL=http://localhost:11434 # Ollama API URL
OLLAMA_MODEL=llama3.1:8b               # Ollama model name

# Whisper Configuration
WHISPER_MODEL=small                    # Model size: tiny, base, small, medium, large
WHISPER_DEVICE=cpu                     # Device: cpu or cuda
WHISPER_COMPUTE_TYPE=int8              # Compute type: int8, float16, float32

# Output
OUTPUT_DIR=./output                    # Default output directory
LOG_LEVEL=INFO                         # Logging level
```

### Model Selection

**Whisper Models:**
- `tiny` (39M) - Fastest but least accurate
- `base` (74M) - Fast but lower accuracy
- `small` (244M) - **Recommended** - Good balance of speed and accuracy
- `medium` (769M) - Better accuracy but slower
- `large` (1.5G) - Best accuracy but very slow on CPU

**Ollama Models:**
- `llama3.1:8b` - **Recommended** - Good instruction following, fast
- `mistral:7b` - Alternative, also works well
- `llama3.1:70b` - Better quality but requires more resources

## Output Schema

The application outputs JSON with the following structure:

```json
{
  "audio_file": "path/to/audio.wav",
  "duration_seconds": 45.2,
  "transcript": "Full transcribed text...",
  "clinical_summary": {
    "patient_complaint": "Chief complaint or reason for visit",
    "findings": [
      "Clinical finding 1",
      "Clinical finding 2"
    ],
    "diagnosis": "Primary diagnosis or differential diagnoses",
    "next_steps": [
      "Treatment plan step 1",
      "Follow-up instruction"
    ],
    "medications": [
      "Medication 1 with dosage",
      "Medication 2 with dosage"
    ],
    "additional_notes": "Any other relevant information"
  }
}
```

## Performance

**Expected performance on modern CPU (Intel i7 or similar):**

| Task | Duration for 30s audio |
|------|------------------------|
| Audio preprocessing | <1s |
| Whisper transcription (small) | 60-90s |
| LLM extraction (llama3.1:8b) | 5-10s |
| **Total** | **~70-100s** |

**Tips for better performance:**
- Use GPU for Whisper: Set `WHISPER_DEVICE=cuda` (requires NVIDIA GPU + CUDA)
- Use smaller Whisper model: `WHISPER_MODEL=base` (faster but less accurate)
- Use faster-whisper (already default): 4x faster than openai-whisper

## Troubleshooting

### Issue: "Failed to connect to Ollama"

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list

# Pull model if needed
ollama pull llama3.1:8b
```

### Issue: "Whisper model download takes too long"

**Solution:**
```bash
# Pre-download model before first use
uv run python scripts/download_models.py
```

### Issue: "Audio file format not supported"

**Solution:**
- Ensure ffmpeg is installed: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)
- Supported formats: WAV, MP3, M4A, OGG, FLAC

### Issue: "Transcription is inaccurate"

**Solutions:**
- Ensure audio quality is good (clear speech, minimal background noise)
- Try a larger Whisper model: `WHISPER_MODEL=medium`
- Check that the language is German (this is set by default)

### Issue: "LLM returns incomplete or malformed JSON"

**Solutions:**
- Try a different model: `OLLAMA_MODEL=mistral:7b`
- Ensure you're using a model with good instruction-following (llama3.1 works well)
- Check Ollama logs: `journalctl -u ollama -f`

## Development

### Project Structure

```
.
├── src/
│   └── medical_transcription/
│       ├── __init__.py
│       ├── main.py                   # CLI entry point
│       ├── app.py                    # Streamlit web interface
│       ├── config.py                 # Configuration management
│       ├── core/
│       │   ├── audio_processor.py   # Audio format conversion
│       │   ├── transcriber.py       # Whisper speech-to-text
│       │   └── llm_extractor.py     # LLM-based extraction
│       ├── models/
│       │   └── clinical_summary.py  # Pydantic data models
│       └── llm/
│           ├── base.py              # Abstract LLM provider
│           └── ollama_provider.py   # Ollama implementation
├── prompts/
│   └── clinical_extraction.txt      # German prompt template
├── examples/
│   ├── sample_medical_de.mp3        # Sample audio
│   └── expected_output.json         # Expected output
├── scripts/
│   ├── download_models.py           # Pre-download Whisper model
│   └── create_sample_audio.py       # Generate test audio
├── pyproject.toml                   # UV dependencies
├── Dockerfile                       # Container configuration
└── README.md                        # This file
```

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests (when implemented)
uv run pytest
```

## Privacy & Security

- **All processing is local**: No data is sent to external APIs (when using Ollama)
- **PHI handling**: Audio files and transcripts contain Protected Health Information (PHI)
- **Recommendations**:
  - Use local Ollama for HIPAA compliance
  - Do not commit audio files or transcripts to version control
  - Ensure proper access controls on output directory
  - Consider encrypting stored results

## Future Enhancements

- [ ] Batch processing mode for multiple files
- [ ] GPU acceleration support for faster inference
- [ ] Fine-tuned Whisper model for German medical terminology
- [ ] Additional LLM providers (Anthropic Claude API, OpenAI API)
- [ ] Web API mode (FastAPI) for integration
- [ ] Real-time streaming transcription
- [ ] Multi-language support

## License

This project is provided as-is for medical documentation purposes.

## Support

For issues, questions, or contributions, please open an issue in the project repository.

---

**Built with:**
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) by Guillaume Klein
- [Ollama](https://ollama.com) for local LLM inference
- [Streamlit](https://streamlit.io) for web interface
- [uv](https://github.com/astral-sh/uv) for Python dependency management
