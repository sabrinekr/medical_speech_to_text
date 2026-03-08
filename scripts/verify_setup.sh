#!/bin/bash
# Verify project setup

echo "🔍 Medical Transcription - Setup Verification"
echo "=============================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python3 --version || echo "❌ Python 3 not found"
echo ""

# Check uv
echo "✓ Checking uv..."
if command -v uv &> /dev/null; then
    uv --version
else
    echo "❌ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
echo ""

# Check Ollama
echo "✓ Checking Ollama..."
if command -v ollama &> /dev/null; then
    ollama --version
    echo "Checking if Ollama is running..."
    curl -s http://localhost:11434/api/tags > /dev/null && echo "  ✓ Ollama is running" || echo "  ⚠ Ollama not running. Start with: ollama serve"
    echo "Available models:"
    ollama list 2>/dev/null || echo "  ⚠ Could not list models"
else
    echo "❌ Ollama not found. Install from: https://ollama.com"
fi
echo ""

# Check ffmpeg
echo "✓ Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg -version | head -n1
else
    echo "❌ ffmpeg not found. Install with: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)"
fi
echo ""

# Check Docker
echo "✓ Checking Docker (optional)..."
if command -v docker &> /dev/null; then
    docker --version
else
    echo "⚠ Docker not found (optional)"
fi
echo ""

echo "=============================================="
echo "Next steps:"
echo "1. Install missing dependencies (see above)"
echo "2. Run: uv sync"
echo "3. Run: ollama pull llama3.1:8b"
echo "4. Run: uv run python scripts/download_models.py"
echo "5. Run: uv run python scripts/create_sample_audio.py"
echo "6. Test: uv run medical-transcribe examples/sample_medical_de.mp3 -v"
