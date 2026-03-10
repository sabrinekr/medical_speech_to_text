# Medical Speech-to-Text System

Transform German medical dictations into structured clinical summaries using AI.

## Overview

This application processes German medical audio dictations using an **agentic AI architecture** with LangGraph:

1. **Speech-to-Text**: Transcribe audio files (WAV/MP3/M4A/OGG/FLAC) using Whisper
2. **Agentic Extraction**: Multi-step LLM reasoning with quality checks and self-correction
3. **Structured Output**: Clinical summaries as validated JSON

**Key Features:**
- 🤖 **Autonomous agent** with multi-step reasoning and self-correction
- 🎤 High-quality German speech recognition with Whisper
- 🏥 Structured clinical data extraction (patient complaint, findings, diagnosis, next steps, medications)
- 🔄 Quality validation with automatic refinement
- 🖥️ Both CLI and web interface (Streamlit)
- 🐳 Docker support for easy deployment
- 🔒 Privacy-focused with local processing (no external APIs required)
- 🆓 **Free to use** - runs entirely on your local machine with Ollama

## Architecture

**Agentic LangGraph Workflow:**

```
Audio File → Medical Agent → Clinical Summary
                  ↓
        [LangGraph - 9 Node Pipeline]
          ├─ Tool: process_audio (convert to WAV)
          ├─ Tool: transcribe_audio (Whisper)
          ├─ Quality Assessment
          ├─ Entity Extraction (LLM)
          ├─ Findings Structuring (LLM)
          ├─ Diagnosis Synthesis (LLM)
          ├─ Treatment Planning (LLM)
          ├─ Quality Check (LLM)
          └─ Synthesis
```

**Key Innovation:** The agent autonomously controls the entire pipeline, with built-in quality checks and self-correction loops. If the extracted summary is incomplete, the agent automatically refines it (up to 2 iterations).

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
cd /medical_speech_to_text

# Install dependencies with uv
uv sync

# Copy environment template
cp .env.example .env

# (Optional) Download Whisper model in advance
uv run python scripts/download_models.py

# Generate test audio files (8 German medical scenarios)
uv run python generate_test_audio.py
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

Example with test audio:

```bash
# Generate test audio files (8 German medical scenarios)
uv run python generate_test_audio.py

# Process a test file
uv run medical-transcribe test_audio/01_bronchitis.mp3 -o output.json -v

# Try other scenarios
uv run medical-transcribe test_audio/02_hypertonie.mp3 -v
uv run medical-transcribe test_audio/05_asthma.mp3 -v
```

**Available test scenarios:**
1. `01_bronchitis.mp3` - Acute bronchitis with antibiotics
2. `02_hypertonie.mp3` - Hypertension control visit
3. `03_gastritis.mp3` - NSAID-induced gastritis
4. `04_diabetes_kontrolle.mp3` - Diabetes quarterly check
5. `05_asthma.mp3` - Allergic asthma exacerbation
6. `06_rueckenschmerzen.mp3` - Acute lower back pain
7. `07_harnwegsinfekt.mp3` - Urinary tract infection
8. `08_angina.mp3` - Streptococcal tonsillitis

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

**Important:** Docker containers need to connect to Ollama running on your host machine. Choose one of the following methods:

**Method 1: Using Host Network (Recommended for Linux)**

```bash
# Process a test audio file
docker run --network=host \
  -v $(pwd)/test_audio:/data \
  medical-transcription /data/01_bronchitis.mp3

# Process with output file
docker run --network=host \
  -v $(pwd)/test_audio:/data \
  -v $(pwd)/output:/output \
  medical-transcription /data/01_bronchitis.mp3 --output /output/result.json
```

**Method 2: Using host.docker.internal (Works on all platforms)**

```bash
# On Linux, add host gateway
docker run --add-host=host.docker.internal:host-gateway \
  -v $(pwd)/test_audio:/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  medical-transcription /data/01_bronchitis.mp3

# On macOS/Windows, host.docker.internal works by default
docker run \
  -v $(pwd)/test_audio:/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  medical-transcription /data/01_bronchitis.mp3
```

**Method 3: Using Host IP (Alternative)**

```bash
# Replace 192.168.1.100 with your actual host IP
docker run \
  -v $(pwd)/test_audio:/data \
  -e OLLAMA_BASE_URL=http://192.168.1.100:11434 \
  medical-transcription /data/01_bronchitis.mp3
```

#### Run Streamlit in Docker

```bash
# Using host network (simplest for Linux)
docker run --network=host \
  --entrypoint uv \
  medical-transcription \
  run streamlit run src/medical_transcription/app.py

# Using port mapping with host.docker.internal
docker run --add-host=host.docker.internal:host-gateway \
  -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --entrypoint uv \
  medical-transcription \
  run streamlit run src/medical_transcription/app.py
```

Then open your browser to `http://localhost:8501`

**Docker Networking Notes:**
- Ollama must be running on your host machine (`ollama serve`)
- The model must be downloaded (`ollama pull llama3.1:8b`)
- `--network=host` is the simplest option on Linux but bypasses Docker's network isolation
- `--add-host=host.docker.internal:host-gateway` enables `host.docker.internal` on Linux
- On macOS/Windows, `host.docker.internal` works by default without extra flags

## Configuration

Configuration is managed through environment variables. Create a `.env` file or set them directly:

```bash
# LLM Configuration (Ollama only)
LLM_PROVIDER=ollama                    # LLM provider (only ollama supported)
OLLAMA_BASE_URL=http://localhost:11434 # Ollama API URL
OLLAMA_MODEL=llama3.1:8b               # Ollama model name

# Whisper Configuration
WHISPER_MODEL=small                    # Model size: tiny, base, small, medium, large
WHISPER_DEVICE=cpu                     # Device: cpu or cuda
WHISPER_COMPUTE_TYPE=int8              # Compute type: int8, float16, float32

# Agent Configuration
MAX_EXTRACTION_ITERATIONS=2            # Max refinement iterations
MAX_TRANSCRIPTION_ATTEMPTS=1           # Max retranscription attempts
QUALITY_THRESHOLD=0.8                  # Quality validation threshold

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
| Agent extraction (5-7 LLM calls) | 50-150s |
| **Total** | **~2-4 minutes** |

**Note:** The agentic architecture uses multiple LLM calls for better quality:
- Without refinement: ~50-80s LLM time (5 calls)
- With 1 refinement iteration: +30-50% time
- Trade-off: Slower but more accurate and complete extractions

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
│       ├── agent/                    # Agentic LangGraph implementation
│       │   ├── __init__.py
│       │   ├── medical_agent.py     # Main agent entry point
│       │   ├── state.py             # Agent state schema
│       │   ├── nodes.py             # 9 node implementations
│       │   ├── graph.py             # LangGraph definition
│       │   ├── tools.py             # Tool wrappers
│       │   └── prompts.py           # German prompts for each node
│       ├── core/
│       │   ├── audio_processor.py   # Audio format conversion
│       │   └── transcriber.py       # Whisper speech-to-text
│       ├── models/
│       │   └── clinical_summary.py  # Pydantic data models
│       └── llm/
│           ├── base.py              # Abstract LLM provider
│           ├── ollama_provider.py   # Ollama implementation
│           └── provider_factory.py  # Provider creation
├── test_audio/                      # Generated test audio files
│   ├── 01_bronchitis.mp3
│   ├── 02_hypertonie.mp3
│   ├── ... (8 scenarios total)
│   └── 08_angina.mp3
├── generate_test_audio.py           # Generate test audio
├── pyproject.toml                   # UV dependencies
├── Dockerfile                       # Container configuration
└── README.md                        # This file
```

### Running Tests

The project includes comprehensive unit tests for core components.

```bash
# Install dev dependencies (includes pytest, pytest-cov, pytest-mock)
uv sync --dev

# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_ollama_provider.py

# Run with coverage report
uv run pytest --cov=src/medical_transcription --cov-report=term-missing

# Run tests matching a pattern
uv run pytest -k "test_sanitize"

# Run only unit tests (using markers)
uv run pytest -m unit
```

**Test Structure:**
```
tests/
├── conftest.py                      # Shared fixtures
└── unit/
    ├── test_audio_processor.py      # Audio processing tests (6 tests)
    ├── test_clinical_summary.py     # Pydantic model tests (6 tests)
    ├── test_config.py              # Configuration tests (7 tests)
    ├── test_ollama_provider.py     # LLM provider tests (13 tests)
    └── test_provider_factory.py    # Factory pattern tests (4 tests)
```

**Current Test Coverage:**
- ✅ 36 unit tests covering critical components
- ✅ Config validation and environment handling
- ✅ Pydantic models and data structures
- ✅ LLM provider (Ollama) with JSON extraction
- ✅ Audio processing and validation
- ✅ Provider factory pattern

## Privacy & Security

- **All processing is local**: No data is sent to external APIs (when using Ollama)
- **PHI handling**: Audio files and transcripts contain Protected Health Information (PHI)
- **Recommendations**:
  - Use local Ollama for HIPAA compliance
  - Do not commit audio files or transcripts to version control
  - Ensure proper access controls on output directory
  - Consider encrypting stored results

## Technical Details

### Agentic Architecture

The application uses **LangGraph** to implement an autonomous agent with:

- **9 specialized nodes**: Each handles a specific task (audio processing, transcription, entity extraction, etc.)
- **Quality-driven iteration**: Automatic refinement if extracted summary is incomplete
- **Conditional routing**: Agent decides next steps based on quality assessment
- **Tool integration**: Agent calls audio processing and transcription as tools
- **State management**: Full execution state tracked through pipeline

### Why Agentic?

Traditional single-shot LLM extraction often misses details or produces incomplete summaries. The agentic approach:

✅ **Better completeness** - Multiple specialized extraction steps catch more details
✅ **Self-correction** - Quality checks trigger automatic refinement
✅ **Transparency** - Execution path shows decision-making
✅ **Adaptability** - Can re-transcribe if needed

## Future Enhancements

- [ ] Batch processing mode for multiple files
- [ ] GPU acceleration support for faster inference
- [ ] Fine-tuned Whisper model for German medical terminology
- [ ] Web API mode (FastAPI) for integration
- [ ] Real-time streaming transcription
- [ ] Multi-language support
- [ ] Export to EHR formats (HL7 FHIR, etc.)

## License

This project is provided as-is for medical documentation purposes.

## Support

For issues, questions, or contributions, please open an issue in the project repository.

---

**Built with:**
- [LangGraph](https://github.com/langchain-ai/langgraph) for agentic workflow orchestration
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) by Guillaume Klein
- [Ollama](https://ollama.com) for local LLM inference (free)
- [Streamlit](https://streamlit.io) for web interface
- [uv](https://github.com/astral-sh/uv) for Python dependency management
- [gTTS](https://github.com/pndurette/gTTS) for test audio generation
