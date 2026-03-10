FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files and source code (needed for building)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install dependencies
RUN uv sync --frozen --no-dev

# Pre-download Whisper model (reduces first-run time)
RUN uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" || echo "Whisper model will be downloaded on first run"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8

# Expose Streamlit port
EXPOSE 8501

# Default: Run CLI
# Override with: docker run ... streamlit run src/medical_transcription/app.py
ENTRYPOINT ["uv", "run", "medical-transcribe"]
