"""Streamlit web interface for medical transcription."""

import streamlit as st
import json
import tempfile
from pathlib import Path
import logging

from medical_transcription.core.audio_processor import AudioProcessor
from medical_transcription.core.transcriber import Transcriber
from medical_transcription.core.llm_extractor import LLMExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Medical Speech-to-Text",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def process_audio(audio_file):
    """Process audio file and extract clinical summary."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
            tmp_file.write(audio_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        # Step 1: Process audio
        st.info("🔄 Processing audio file...")
        processor = AudioProcessor()
        wav_path, is_temp = processor.convert_to_wav(tmp_path)
        duration = processor.get_audio_duration(wav_path)

        st.success(f"✓ Audio processed: {duration:.1f}s duration")

        # Step 2: Transcribe
        st.info("🎤 Transcribing audio (this may take a few minutes)...")
        transcriber = Transcriber()
        transcript_result = transcriber.transcribe(wav_path, language="de")
        transcript = transcript_result["transcript"]

        st.success(f"✓ Transcription complete: {len(transcript)} characters")

        # Step 3: Extract structured summary
        st.info("🤖 Extracting structured clinical summary...")
        extractor = LLMExtractor()
        summary = extractor.extract(transcript)

        st.success("✓ Clinical summary extracted!")

        # Clean up temporary files
        tmp_path.unlink()
        if is_temp:
            wav_path.unlink()

        return transcript, summary, duration

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logging.exception("Error processing audio")
        return None, None, None


# Header
st.title("🏥 Medical Speech-to-Text System")
st.markdown("""
Transform German medical dictations into structured clinical summaries using AI.

**Features:**
- Speech-to-text transcription with Whisper
- Structured extraction with local LLM (Ollama)
- Export results as JSON
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    **Supported Audio Formats:**
    - WAV
    - MP3
    - M4A
    - OGG
    - FLAC

    **Requirements:**
    - Ollama must be running
    - Model llama3.1:8b must be available

    **Privacy:**
    All processing is done locally. No data is sent to external servers.
    """)

    st.divider()

    st.header("⚙️ Settings")
    st.markdown("""
    Current configuration:
    - **Whisper Model:** small
    - **LLM Provider:** Ollama
    - **Language:** German
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Input")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Upload a German medical dictation audio file"
    )

    if uploaded_file:
        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix[1:]}")

        # Process button
        if st.button("🚀 Transcribe & Extract", type="primary", use_container_width=True):
            st.session_state.processing = True
            transcript, summary, duration = process_audio(uploaded_file)

            if transcript and summary:
                st.session_state.transcript = transcript
                st.session_state.summary = summary
                st.session_state.duration = duration
            st.session_state.processing = False

with col2:
    st.header("📊 Results")

    if st.session_state.transcript:
        # Transcript
        with st.expander("📝 Transcript", expanded=False):
            st.text_area(
                "Transcribed text",
                value=st.session_state.transcript,
                height=200,
                label_visibility="collapsed"
            )

        # Clinical Summary
        st.subheader("🏥 Clinical Summary")

        summary = st.session_state.summary

        # Display structured summary
        st.markdown(f"**Patient Complaint:**")
        st.info(summary.patient_complaint)

        if summary.findings:
            st.markdown("**Findings:**")
            for finding in summary.findings:
                st.write(f"- {finding}")

        if summary.diagnosis:
            st.markdown(f"**Diagnosis:**")
            st.warning(summary.diagnosis)

        if summary.next_steps:
            st.markdown("**Next Steps:**")
            for step in summary.next_steps:
                st.write(f"- {step}")

        if summary.medications:
            st.markdown("**Medications:**")
            for med in summary.medications:
                st.write(f"- {med}")

        if summary.additional_notes:
            st.markdown(f"**Additional Notes:**")
            st.write(summary.additional_notes)

        st.divider()

        # JSON export
        st.subheader("💾 Export")

        output_data = {
            "duration_seconds": st.session_state.duration,
            "transcript": st.session_state.transcript,
            "clinical_summary": summary.model_dump()
        }

        json_str = json.dumps(output_data, ensure_ascii=False, indent=2)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="clinical_summary.json",
                mime="application/json",
                use_container_width=True
            )

        with col_b:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.transcript = None
                st.session_state.summary = None
                st.rerun()

    elif not st.session_state.processing:
        st.info("👈 Upload an audio file and click 'Transcribe & Extract' to begin.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Medical Speech-to-Text v0.1.0 | Built with Whisper & Ollama</small>
</div>
""", unsafe_allow_html=True)
