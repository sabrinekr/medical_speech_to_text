"""Streamlit web interface for medical transcription."""

import streamlit as st
import json
import tempfile
from pathlib import Path
import logging

from medical_transcription.agent import MedicalAgent
from medical_transcription.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Medical Speech-to-Text",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "summary" not in st.session_state:
    st.session_state.summary = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def process_audio(audio_file):
    """Process audio file and extract clinical summary using Medical Agent."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
            tmp_file.write(audio_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        # Process with Medical Agent
        st.info("🤖 Medical agent processing audio with Ollama...")

        # Initialize agent (always uses Ollama)
        agent = MedicalAgent()

        # Process end-to-end
        summary = agent.process(str(tmp_path))

        st.success("✅ Processing complete!")

        # Clean up temporary file
        tmp_path.unlink()

        return summary

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logging.exception("Error processing audio")
        return None


# Header
st.title("🏥 Medical Speech-to-Text System")
st.markdown("""
Transform German medical dictations into structured clinical summaries using **Agentic AI with Ollama**.

**Features:**
- 🎤 Speech-to-text transcription with Whisper
- 🤖 Multi-step agentic extraction with LangGraph
- ✅ Automatic quality checks and self-correction
- 💻 Free local processing with Ollama
- 📥 Export results as JSON
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    st.info("💻 **Ollama** (Local)\n- Free\n- Requires Ollama running\n- Model: " + Config.OLLAMA_MODEL)

    st.divider()

    st.header("ℹ️ Information")
    st.markdown("""
    **Supported Audio Formats:**
    - WAV
    - MP3
    - M4A
    - OGG
    - FLAC

    **How it works:**
    1. Audio → WAV conversion
    2. Whisper transcription
    3. Quality assessment
    4. Multi-step AI extraction
    5. Quality validation & refinement
    6. Final clinical summary

    **Agentic Features:**
    - Automatic quality checks
    - Self-correction via refinement
    - Adaptive processing
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
        if st.button("🚀 Process Audio", type="primary", use_container_width=True):
            st.session_state.processing = True
            summary = process_audio(uploaded_file)

            if summary:
                st.session_state.summary = summary
            st.session_state.processing = False

with col2:
    st.header("📊 Results")

    if st.session_state.summary:
        # Clinical Summary
        st.subheader("🏥 Clinical Summary")

        summary = st.session_state.summary

        # Display structured summary
        st.markdown("**Patient Complaint:**")
        st.info(summary.patient_complaint)

        if summary.findings:
            st.markdown("**Findings:**")
            for finding in summary.findings:
                st.write(f"- {finding}")

        if summary.diagnosis:
            st.markdown("**Diagnosis:**")
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
            st.markdown("**Additional Notes:**")
            st.write(summary.additional_notes)

        st.divider()

        # JSON export
        st.subheader("💾 Export")

        output_data = {
            "provider": "ollama",
            "model": Config.OLLAMA_MODEL,
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
                st.session_state.summary = None
                st.rerun()

    elif not st.session_state.processing:
        st.info("👈 Upload an audio file and click 'Process Audio' to begin.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Medical Speech-to-Text v1.0.0 | Agentic AI with LangGraph | Whisper + Ollama</small>
</div>
""", unsafe_allow_html=True)
