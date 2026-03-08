#!/usr/bin/env python3
"""Pre-download Whisper model to avoid delays on first run."""

from faster_whisper import WhisperModel
import sys

def main():
    """Download Whisper model."""
    model_name = "small"
    device = "cpu"
    compute_type = "int8"

    print(f"Downloading Whisper '{model_name}' model...")
    print(f"Device: {device}, Compute type: {compute_type}")
    print("This may take a few minutes...")

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("✓ Model downloaded successfully!")
        print("Model is ready to use.")
        return 0
    except Exception as e:
        print(f"✗ Error downloading model: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
