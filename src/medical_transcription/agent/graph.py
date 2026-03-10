"""LangGraph definition for medical transcription agent."""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from medical_transcription.agent.state import AgentState
from medical_transcription.agent.nodes import AgentNodes
from medical_transcription.llm.base import BaseLLMProvider
from medical_transcription.config import Config

logger = logging.getLogger(__name__)


def should_refine(state: AgentState) -> Literal["refine", "synthesize"]:
    """
    Conditional edge: Determine if refinement is needed.

    Args:
        state: Current agent state

    Returns:
        "refine" if quality check failed and iterations remaining, else "synthesize"
    """
    validation = state.get("validation_result", {})
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", Config.MAX_EXTRACTION_ITERATIONS)

    # Check if quality is acceptable
    quality_score = validation.get("overall_quality_score", 1.0)
    is_complete = validation.get("is_complete", True)
    is_consistent = validation.get("is_consistent", True)

    # Quality threshold check
    quality_acceptable = (
        quality_score >= Config.QUALITY_THRESHOLD
        and is_complete
        and is_consistent
    )

    # Decide routing
    if not quality_acceptable and iteration_count < max_iterations:
        logger.info(
            f"Quality check failed (score={quality_score:.2f}), "
            f"routing to refinement (iteration {iteration_count + 1}/{max_iterations})"
        )
        return "refine"
    else:
        if not quality_acceptable:
            logger.warning(
                f"Quality check failed but max iterations reached ({max_iterations}), "
                f"proceeding to synthesis"
            )
        else:
            logger.info(f"Quality check passed (score={quality_score:.2f}), proceeding to synthesis")
        return "synthesize"


def should_retranscribe(state: AgentState) -> Literal["retranscribe", "extract"]:
    """
    Conditional edge: Determine if re-transcription is needed.

    Args:
        state: Current agent state

    Returns:
        "retranscribe" if quality check failed and attempts remaining, else "extract"
    """
    quality = state.get("transcript_quality", {})
    attempts = state.get("transcription_attempts", 0)
    max_attempts = state.get("max_transcription_attempts", Config.MAX_TRANSCRIPTION_ATTEMPTS)

    is_acceptable = quality.get("is_acceptable", True)

    # Decide routing
    if not is_acceptable and attempts < max_attempts:
        logger.info(
            f"Transcript quality insufficient, routing to re-transcription "
            f"(attempt {attempts + 1}/{max_attempts})"
        )
        return "retranscribe"
    else:
        if not is_acceptable:
            logger.warning(
                f"Transcript quality insufficient but max attempts reached ({max_attempts}), "
                f"proceeding to extraction"
            )
        else:
            logger.info("Transcript quality acceptable, proceeding to extraction")
        return "extract"


def create_agent_graph(llm_provider: BaseLLMProvider) -> StateGraph:
    """
    Create LangGraph for medical transcription agent.

    Args:
        llm_provider: LLM provider instance

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating agent graph...")

    # Initialize nodes
    nodes = AgentNodes(llm_provider)

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("process_audio", nodes.process_audio_node)
    workflow.add_node("transcribe_audio", nodes.transcribe_audio_node)
    workflow.add_node("assess_quality", nodes.assess_transcript_quality_node)
    workflow.add_node("extract_entities", nodes.extract_entities_node)
    workflow.add_node("structure_findings", nodes.structure_findings_node)
    workflow.add_node("synthesize_diagnosis", nodes.synthesize_diagnosis_node)
    workflow.add_node("plan_treatment", nodes.plan_treatment_node)
    workflow.add_node("validate_quality", nodes.validate_quality_node)
    workflow.add_node("refine_extraction", nodes.refine_extraction_node)
    workflow.add_node("synthesize_final", nodes.synthesize_final_node)

    # Set entry point
    workflow.set_entry_point("process_audio")

    # Add edges - Linear flow for nodes 1-2
    workflow.add_edge("process_audio", "transcribe_audio")
    workflow.add_edge("transcribe_audio", "assess_quality")

    # Conditional edge - Re-transcribe or proceed to extraction
    workflow.add_conditional_edges(
        "assess_quality",
        should_retranscribe,
        {
            "retranscribe": "transcribe_audio",  # Loop back with different params
            "extract": "extract_entities"
        }
    )

    # Add edges - Linear flow for extraction nodes 4-7
    workflow.add_edge("extract_entities", "structure_findings")
    workflow.add_edge("structure_findings", "synthesize_diagnosis")
    workflow.add_edge("synthesize_diagnosis", "plan_treatment")
    workflow.add_edge("plan_treatment", "validate_quality")

    # Conditional edge - Refine or proceed to synthesis
    workflow.add_conditional_edges(
        "validate_quality",
        should_refine,
        {
            "refine": "refine_extraction",
            "synthesize": "synthesize_final"
        }
    )

    # After refinement, validate again
    workflow.add_edge("refine_extraction", "validate_quality")

    # Final node leads to END
    workflow.add_edge("synthesize_final", END)

    # Compile graph
    graph = workflow.compile()

    logger.info("Agent graph created successfully")
    logger.info(
        f"Graph nodes: {len(workflow.nodes)}, "
        f"Max transcription attempts: {Config.MAX_TRANSCRIPTION_ATTEMPTS}, "
        f"Max refinement iterations: {Config.MAX_EXTRACTION_ITERATIONS}"
    )

    return graph
