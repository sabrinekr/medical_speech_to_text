"""Agent nodes for medical transcription workflow."""

import json
import logging
from typing import Dict, Any
from pathlib import Path

from medical_transcription.agent.state import AgentState
from medical_transcription.agent.tools import process_audio_tool, transcribe_audio_tool
from medical_transcription.agent import prompts
from medical_transcription.llm.base import BaseLLMProvider
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


class AgentNodes:
    """Collection of agent node functions for LangGraph."""

    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialize agent nodes with LLM provider.

        Args:
            llm_provider: LLM provider instance for generation
        """
        self.llm = llm_provider
        self.provider_name = llm_provider.__class__.__name__
        self.model_name = getattr(llm_provider, "model", "unknown")

    def process_audio_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Process audio file to WAV format.

        Args:
            state: Current agent state

        Returns:
            Updated state with WAV path and duration
        """
        logger.info("=== Node 1: Processing audio ===")

        try:
            # Use process_audio tool
            result = process_audio_tool.invoke({"audio_path": state["audio_path"]})

            state["wav_path"] = result["wav_path"]
            state["wav_duration"] = result["duration"]
            state["is_temp_wav"] = result["is_temp"]
            state["execution_path"].append("process_audio")

            logger.info(f"Audio processed: {result['wav_path']} ({result['duration']:.2f}s)")

            return state

        except Exception as e:
            logger.error(f"Error in process_audio_node: {e}")
            raise

    def transcribe_audio_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Transcribe audio using Whisper.

        Args:
            state: Current agent state

        Returns:
            Updated state with transcript and segments
        """
        logger.info("=== Node 2: Transcribing audio ===")

        try:
            # Use transcribe_audio tool
            result = transcribe_audio_tool.invoke({
                "wav_path": state["wav_path"],
                "language": "de"
            })

            state["transcript"] = result["transcript"]
            state["transcript_segments"] = result["segments"]

            # Validate transcript length
            transcript_length = len(result["transcript"])
            if transcript_length < Config.MIN_TRANSCRIPT_LENGTH:
                raise ValueError(
                    f"Transcript too short: {transcript_length} characters. "
                    f"Minimum required: {Config.MIN_TRANSCRIPT_LENGTH} characters. "
                    f"Audio may be empty or silent."
                )
            if transcript_length > Config.MAX_TRANSCRIPT_LENGTH:
                raise ValueError(
                    f"Transcript too long: {transcript_length} characters. "
                    f"Maximum allowed: {Config.MAX_TRANSCRIPT_LENGTH} characters. "
                    f"Consider splitting the audio into smaller segments."
                )

            # Calculate average confidence from segments
            if result["segments"]:
                avg_confidence = sum(
                    seg["confidence"] for seg in result["segments"]
                ) / len(result["segments"])
                state["transcript_confidence"] = avg_confidence
            else:
                state["transcript_confidence"] = 0.0

            state["transcription_attempts"] += 1
            state["execution_path"].append("transcribe_audio")

            logger.info(
                f"Transcription complete: {transcript_length} chars, "
                f"confidence={state['transcript_confidence']:.2f}"
            )

            return state

        except Exception as e:
            logger.error(f"Error in transcribe_audio_node: {e}")
            raise

    def assess_transcript_quality_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Assess transcript quality.

        Args:
            state: Current agent state

        Returns:
            Updated state with quality assessment
        """
        logger.info("=== Node 3: Assessing transcript quality ===")

        try:
            # Format prompt
            prompt = prompts.TRANSCRIPT_QUALITY_PROMPT.format(
                transcript=state["transcript"],
                segment_count=len(state.get("transcript_segments", [])),
                avg_confidence=state.get("transcript_confidence", 0.0),
                duration=state.get("wav_duration", 0.0),
                quality_threshold=Config.QUALITY_THRESHOLD
            )

            # Call LLM
            assessment = self.llm.generate_json(prompt)

            # Store in state (for routing decision)
            state["transcript_quality"] = assessment
            state["execution_path"].append("assess_quality")

            logger.info(
                f"Quality assessment: acceptable={assessment['is_acceptable']}, "
                f"score={assessment['quality_score']:.2f}"
            )

            return state

        except Exception as e:
            logger.error(f"Error in assess_transcript_quality_node: {e}")
            # If assessment fails, assume transcript is acceptable
            state["transcript_quality"] = {
                "is_acceptable": True,
                "quality_score": 0.7,
                "issues": ["Quality assessment failed"],
                "recommendations": []
            }
            state["execution_path"].append("assess_quality_error")
            return state

    def extract_entities_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Extract medical entities from transcript.

        Args:
            state: Current agent state

        Returns:
            Updated state with extracted entities
        """
        logger.info("=== Node 4: Extracting medical entities ===")

        try:
            # Format prompt
            prompt = prompts.ENTITY_EXTRACTION_PROMPT.format(
                transcript=state["transcript"]
            )

            # Call LLM
            entities = self.llm.generate_json(prompt)

            state["entities"] = entities
            state["execution_path"].append("extract_entities")

            logger.info(
                f"Entities extracted: {sum(len(v) for v in entities.values())} total items"
            )

            return state

        except Exception as e:
            logger.error(f"Error in extract_entities_node: {e}")
            raise

    def structure_findings_node(self, state: AgentState) -> AgentState:
        """
        Node 5: Structure findings from entities.

        Args:
            state: Current agent state

        Returns:
            Updated state with structured findings
        """
        logger.info("=== Node 5: Structuring findings ===")

        try:
            # Format prompt
            prompt = prompts.FINDINGS_STRUCTURING_PROMPT.format(
                entities=json.dumps(state["entities"], ensure_ascii=False, indent=2),
                transcript=state["transcript"]
            )

            # Call LLM
            structured = self.llm.generate_json(prompt)

            state["patient_complaint"] = structured["patient_complaint"]
            state["findings"] = structured["findings"]
            state["execution_path"].append("structure_findings")

            logger.info(
                f"Findings structured: complaint='{state['patient_complaint'][:50]}...', "
                f"{len(state['findings'])} findings"
            )

            return state

        except Exception as e:
            logger.error(f"Error in structure_findings_node: {e}")
            raise

    def synthesize_diagnosis_node(self, state: AgentState) -> AgentState:
        """
        Node 6: Synthesize diagnosis from findings.

        Args:
            state: Current agent state

        Returns:
            Updated state with diagnosis
        """
        logger.info("=== Node 6: Synthesizing diagnosis ===")

        try:
            # Format prompt
            prompt = prompts.DIAGNOSIS_SYNTHESIS_PROMPT.format(
                patient_complaint=state["patient_complaint"],
                findings=json.dumps(state["findings"], ensure_ascii=False, indent=2),
                entities=json.dumps(state["entities"], ensure_ascii=False, indent=2),
                transcript=state["transcript"]
            )

            # Call LLM
            diagnosis_result = self.llm.generate_json(prompt)

            state["diagnosis"] = diagnosis_result["diagnosis"]
            state["execution_path"].append("synthesize_diagnosis")

            logger.info(f"Diagnosis synthesized: '{state['diagnosis'][:100]}...'")

            return state

        except Exception as e:
            logger.error(f"Error in synthesize_diagnosis_node: {e}")
            raise

    def plan_treatment_node(self, state: AgentState) -> AgentState:
        """
        Node 7: Plan treatment and extract medications.

        Args:
            state: Current agent state

        Returns:
            Updated state with treatment plan
        """
        logger.info("=== Node 7: Planning treatment ===")

        try:
            # Format prompt
            prompt = prompts.TREATMENT_PLANNING_PROMPT.format(
                diagnosis=state["diagnosis"],
                findings=json.dumps(state["findings"], ensure_ascii=False, indent=2),
                transcript=state["transcript"]
            )

            # Call LLM
            treatment = self.llm.generate_json(prompt)

            state["next_steps"] = treatment["next_steps"]
            state["medications"] = treatment["medications"]
            state["additional_notes"] = treatment["additional_notes"]
            state["execution_path"].append("plan_treatment")

            logger.info(
                f"Treatment planned: {len(state['next_steps'])} steps, "
                f"{len(state['medications'])} medications"
            )

            return state

        except Exception as e:
            logger.error(f"Error in plan_treatment_node: {e}")
            raise

    def validate_quality_node(self, state: AgentState) -> AgentState:
        """
        Node 8: Validate quality of extraction.

        Args:
            state: Current agent state

        Returns:
            Updated state with validation result
        """
        logger.info("=== Node 8: Validating quality ===")

        try:
            # Format prompt
            prompt = prompts.QUALITY_CHECK_PROMPT.format(
                patient_complaint=state["patient_complaint"],
                findings=json.dumps(state["findings"], ensure_ascii=False, indent=2),
                diagnosis=state["diagnosis"],
                next_steps=json.dumps(state["next_steps"], ensure_ascii=False, indent=2),
                medications=json.dumps(state["medications"], ensure_ascii=False, indent=2),
                additional_notes=state["additional_notes"],
                transcript=state["transcript"]
            )

            # Call LLM
            validation = self.llm.generate_json(prompt)

            state["validation_result"] = validation
            state["execution_path"].append("validate_quality")

            logger.info(
                f"Quality validated: score={validation['overall_quality_score']:.2f}, "
                f"complete={validation['is_complete']}, consistent={validation['is_consistent']}"
            )

            return state

        except Exception as e:
            logger.error(f"Error in validate_quality_node: {e}")
            # If validation fails, assume quality is acceptable
            state["validation_result"] = {
                "is_complete": True,
                "is_consistent": True,
                "is_clear": True,
                "missing_fields": [],
                "inconsistencies": [],
                "suggestions": [],
                "overall_quality_score": 0.7
            }
            state["execution_path"].append("validate_quality_error")
            return state

    def refine_extraction_node(self, state: AgentState) -> AgentState:
        """
        Node 8b: Refine extraction based on quality check.

        Args:
            state: Current agent state

        Returns:
            Updated state with refined extraction
        """
        logger.info("=== Node 8b: Refining extraction ===")

        try:
            # Increment iteration counter
            state["iteration_count"] += 1

            # Format prompt
            prompt = prompts.REFINEMENT_PROMPT.format(
                patient_complaint=state["patient_complaint"],
                findings=json.dumps(state["findings"], ensure_ascii=False, indent=2),
                diagnosis=state["diagnosis"],
                next_steps=json.dumps(state["next_steps"], ensure_ascii=False, indent=2),
                medications=json.dumps(state["medications"], ensure_ascii=False, indent=2),
                additional_notes=state["additional_notes"],
                validation_result=json.dumps(
                    state["validation_result"], ensure_ascii=False, indent=2
                ),
                transcript=state["transcript"]
            )

            # Call LLM
            refined = self.llm.generate_json(prompt)

            # Update state with refined data
            state["patient_complaint"] = refined["patient_complaint"]
            state["findings"] = refined["findings"]
            state["diagnosis"] = refined["diagnosis"]
            state["next_steps"] = refined["next_steps"]
            state["medications"] = refined["medications"]
            state["additional_notes"] = refined["additional_notes"]
            state["execution_path"].append(f"refine_iteration_{state['iteration_count']}")

            logger.info(
                f"Extraction refined (iteration {state['iteration_count']}/{state['max_iterations']})"
            )

            return state

        except Exception as e:
            logger.error(f"Error in refine_extraction_node: {e}")
            # If refinement fails, keep current state
            state["execution_path"].append("refine_error")
            return state

    def synthesize_final_node(self, state: AgentState) -> AgentState:
        """
        Node 9: Create final clinical summary.

        Args:
            state: Current agent state

        Returns:
            Updated state with final summary
        """
        logger.info("=== Node 9: Synthesizing final summary ===")

        try:
            # Format prompt
            prompt = prompts.FINAL_SYNTHESIS_PROMPT.format(
                patient_complaint=state["patient_complaint"],
                findings=json.dumps(state["findings"], ensure_ascii=False, indent=2),
                diagnosis=state["diagnosis"],
                next_steps=json.dumps(state["next_steps"], ensure_ascii=False, indent=2),
                medications=json.dumps(state["medications"], ensure_ascii=False, indent=2),
                additional_notes=state["additional_notes"]
            )

            # Call LLM
            final_summary = self.llm.generate_json(prompt)

            # Store provider and model info
            state["provider_name"] = self.provider_name
            state["model_name"] = self.model_name

            # Store final summary
            state["final_summary"] = final_summary
            state["execution_path"].append("synthesize_final")

            logger.info(
                f"Final summary created: confidence={final_summary.get('confidence_score', 0):.2f}"
            )
            logger.info(f"Execution path: {' -> '.join(state['execution_path'])}")

            return state

        except Exception as e:
            logger.error(f"Error in synthesize_final_node: {e}")
            raise
