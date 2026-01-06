"""
Pipeline orchestrator using LangGraph.
"""
import os
import uuid
import json
import time
import logging
from typing import Optional
from langgraph.graph import StateGraph, END

from src.core.state import PipelineState, CallAnalysisResult, ProcessingStatus
from src.core.config import get_settings, Settings
from src.services.transcription import TranscriptionService
from src.services.refinement import RefinementService
from src.services.correction import CorrectionService
from src.services.classification import ClassificationService
from src.services.sentiment import SentimentService


class CallAnalysisPipeline:
    """Main pipeline orchestrator for call center analysis."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.logger = self._create_logger()
        
        # Initialize services
        self.transcription_service = TranscriptionService(self.settings.whisper)
        self.refinement_service = RefinementService(self.settings.gemini)
        # Temporarily disable correction service
        # self.correction_service = CorrectionService(self.settings.qwen)
        self.correction_service = None
        self.classification_service = ClassificationService(
            self.settings.vllm,
            self.settings.classification
        )
        self.sentiment_service = SentimentService(self.settings.vllm)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _create_logger(self) -> logging.Logger:
        """Create logger for the pipeline."""
        logger = logging.getLogger("pipeline.orchestrator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("transcribe", self._transcribe_node)
        workflow.add_node("refine", self._refine_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("correct", self._correct_node)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("analyze_sentiment", self._sentiment_node)
        workflow.add_node("save_result", self._save_node)
        
        # Set entry point
        workflow.set_entry_point("transcribe")
        
        # Add edges
        workflow.add_edge("transcribe", "refine")
        workflow.add_edge("refine", "verify")
        
        # Conditional routing from verify
        workflow.add_conditional_edges(
            "verify",
            self._route_decision,
            {
                "proceed": "classify",
                "manual_review": "save_result"
            }
        )
        
        # Correction step temporarily disabled - node kept for future re-enablement
        # workflow.add_edge("correct", "save_result")
        
        # Success path with error checking
        workflow.add_conditional_edges(
            "classify",
            self._check_error,
            {
                "continue": "analyze_sentiment",
                "error": "save_result"
            }
        )
        workflow.add_conditional_edges(
            "analyze_sentiment",
            self._check_error,
            {
                "continue": "save_result",
                "error": "save_result"
            }
        )
        
        # Final edge
        workflow.add_edge("save_result", END)
        
        return workflow.compile()
    
    def _transcribe_node(self, state: PipelineState) -> PipelineState:
        """Transcribe audio node."""
        result = state["result"]
        audio_path = state["audio_path"]
        run_count = state["run_count"]
        
        # If already corrected, recalculate confidence
        if result.is_corrected and result.transcript:
            confidence = self.transcription_service.recalculate_confidence(result.transcript)
            result.confidence_score = confidence
            return {
                "result": result,
                "run_count": run_count + 1
            }
        
        # Fresh transcription
        service_result = self.transcription_service.process(
            audio_path,
            self.settings.pipeline.audio_sample_rate
        )
        
        if service_result.success:
            result.transcript = service_result.data["transcript"]
            result.confidence_score = service_result.data["confidence"]
        else:
            result.transcript = f"Error: {service_result.error}"
            result.confidence_score = 0.0
            result.status = ProcessingStatus.ERROR
            result.error_message = service_result.error
        
        return {
            "result": result,
            "run_count": run_count + 1
        }
    
    def _refine_node(self, state: PipelineState) -> PipelineState:
        """Refine transcript node."""
        result = state["result"]
        
        # Check if already in error state, skip processing
        if result.status == ProcessingStatus.ERROR:
            return {"result": result}
        
        service_result = self.refinement_service.process(result.transcript)
        
        if not service_result.success:
            # Service failed, stop pipeline
            result.status = ProcessingStatus.ERROR
            result.error_message = service_result.error or "Refinement service failed"
            result.refined_transcript = result.transcript
            result.refinement_score = 0.0
            self.logger.error(f"Refinement failed: {result.error_message}")
            return {"result": result}
        
        # Check if refinement actually failed (score = 0.0 and no API key or API error)
        refinement_score = service_result.data.get("refinement_score", 0.0)
        if refinement_score == 0.0 and not self.settings.gemini.api_key:
            # No API key, but this is expected - not an error
            result.refined_transcript = service_result.data.get("refined_transcript", result.transcript)
            result.refinement_score = 0.0
        elif refinement_score == 0.0:
            # API key exists but refinement failed - check if it's a critical error
            # If refinement_score is 0.0 and we have API key, it might be an error
            # But we'll let routing handle this (will go to manual_review)
            result.refined_transcript = service_result.data.get("refined_transcript", result.transcript)
            result.refinement_score = 0.0
        else:
            # Refinement succeeded
            result.refined_transcript = service_result.data["refined_transcript"]
            result.refinement_score = refinement_score
        
        return {"result": result}
    
    def _verify_node(self, state: PipelineState) -> PipelineState:
        """Verify confidence and refinement quality node."""
        result = state["result"]
        result.status = ProcessingStatus.IN_PROGRESS
        
        # Log confidence and refinement scores
        self.logger.info(f"Transcription confidence: {result.confidence_score:.3f}, Threshold: {self.settings.pipeline.confidence_threshold}")
        self.logger.info(f"Refinement score: {result.refinement_score:.3f}, Threshold: {self.settings.pipeline.refinement_threshold}")
        
        return {"result": result}
    
    def _check_error(self, state: PipelineState) -> str:
        """Check if pipeline is in error state."""
        result = state["result"]
        if result.status == ProcessingStatus.ERROR:
            return "error"
        return "continue"
    
    def _route_decision(self, state: PipelineState) -> str:
        """Routing decision based on confidence and refinement quality."""
        result = state["result"]
        
        # If already in error state, go directly to save
        if result.status == ProcessingStatus.ERROR:
            self.logger.info("Routing decision: Error detected, routing to 'save_result'")
            return "manual_review"  # Use manual_review route which goes to save_result
        
        confidence = result.confidence_score
        refinement_score = result.refinement_score
        
        # Check both transcription confidence and refinement quality
        # If refinement score is too low, it means the transcript has no meaningful content
        # Route to manual review instead of proceeding to classification/sentiment
        if refinement_score < self.settings.pipeline.refinement_threshold:
            self.logger.info(f"Routing decision: refinement_score={refinement_score:.3f} < threshold={self.settings.pipeline.refinement_threshold}, routing to 'manual_review'")
            return "manual_review"
        
        # Also check transcription confidence
        if confidence < self.settings.pipeline.confidence_threshold:
            self.logger.info(f"Routing decision: confidence={confidence:.3f} < threshold={self.settings.pipeline.confidence_threshold}, routing to 'manual_review'")
            return "manual_review"
        
        # Both scores are acceptable, proceed to classification
        self.logger.info(f"Routing decision: confidence={confidence:.3f}, refinement_score={refinement_score:.3f}, routing to 'proceed' (classification)")
        return "proceed"
    
    def _correct_node(self, state: PipelineState) -> PipelineState:
        """Correct transcript node."""
        result = state["result"]
        
        # Use refined transcript if available, otherwise original
        transcript_to_correct = result.refined_transcript or result.transcript
        
        try:
            service_result = self.correction_service.process(transcript_to_correct)
            
            if service_result.success:
                result.transcript = service_result.data["corrected_transcript"]
                result.is_corrected = True
            else:
                # Keep original on error
                result.is_corrected = False
                self.logger.warning(f"Correction failed: {service_result.error}. Using original transcript.")
        except Exception as e:
            # If correction service is not available, skip correction
            result.is_corrected = False
            self.logger.warning(f"Correction service unavailable: {e}. Skipping correction step.")
        
        return {"result": result}
    
    def _classify_node(self, state: PipelineState) -> PipelineState:
        """Classify subject node."""
        result = state["result"]
        
        # Check if already in error state, skip processing
        if result.status == ProcessingStatus.ERROR:
            return {"result": result}
        
        transcript = result.refined_transcript or result.transcript
        
        self.logger.info(f"Classification node called with transcript length: {len(transcript)}")
        
        service_result = self.classification_service.process(transcript)
        
        if not service_result.success:
            # Service failed, stop pipeline
            result.status = ProcessingStatus.ERROR
            result.error_message = service_result.error or "Classification service failed"
            result.subject = "UNKNOWN"
            result.sub_subject = "N/A"
            self.logger.error(f"Classification failed: {result.error_message}")
            return {"result": result}
        
        # Check if classification actually failed (subject = "OTHER" with 0.0 confidence might indicate error)
        # But we'll accept the result since service returned success
        result.subject = service_result.data.get("subject", "UNKNOWN")
        result.sub_subject = service_result.data.get("sub_subject", "N/A")
        
        # Additional check: if we get "UNKNOWN" or empty subject, it might be an error
        # But since the service returned success, we'll trust it
        
        return {"result": result}
    
    def _sentiment_node(self, state: PipelineState) -> PipelineState:
        """Analyze sentiment node."""
        result = state["result"]
        
        # Check if already in error state, skip processing
        if result.status == ProcessingStatus.ERROR:
            return {"result": result}
        
        transcript = result.refined_transcript or result.transcript
        
        service_result = self.sentiment_service.process(transcript)
        
        if not service_result.success:
            # Service failed, stop pipeline
            result.status = ProcessingStatus.ERROR
            result.error_message = service_result.error or "Sentiment analysis service failed"
            result.satisfaction_score = 0.0
            self.logger.error(f"Sentiment analysis failed: {result.error_message}")
            return {"result": result}
        
        # Service succeeded
        result.satisfaction_score = service_result.data.get("satisfaction_score", 0.0)
        
        return {"result": result}
    
    def _save_node(self, state: PipelineState) -> PipelineState:
        """Save result node."""
        result = state["result"]
        audio_path = state["audio_path"]
        
        # Determine final status
        if result.status == ProcessingStatus.ERROR:
            pass  # Keep error status - pipeline stopped due to error
        elif result.refinement_score < self.settings.pipeline.refinement_threshold:
            # Refinement failed or quality too low
            result.status = ProcessingStatus.MANUAL_REVIEW
        elif result.confidence_score < self.settings.pipeline.confidence_threshold:
            if state["run_count"] >= self.settings.pipeline.max_retry_attempts:
                result.status = ProcessingStatus.MANUAL_REVIEW
            else:
                result.status = ProcessingStatus.LOW_CONFIDENCE
        else:
            # Both scores are acceptable and all steps completed
            result.status = ProcessingStatus.COMPLETE
        
        # Save to file
        os.makedirs(self.settings.pipeline.output_dir, exist_ok=True)
        
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        call_id = state.get("call_id", result.call_id) or "unknown"
        unique_id = call_id.split("_")[-1] if "_" in call_id else call_id[-8:] if len(call_id) > 8 else call_id
        
        output_path = os.path.join(
            self.settings.pipeline.output_dir,
            f"{audio_basename}_{unique_id}_result.json"
        )
        
        # Convert to dict
        result_dict = result.model_dump()
        result_dict["status"] = result.status.value
        result_dict["audio_path"] = audio_path
        result_dict["run_count"] = state["run_count"]
        result_dict["call_id"] = result.call_id
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Log summary
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("📋 CALL ANALYSIS RESULT")
        self.logger.info("=" * 60)
        self.logger.info(f"Call ID: {result.call_id}")
        self.logger.info(f"Audio: {audio_path}")
        self.logger.info(f"Status: {result.status.value}")
        self.logger.info(f"Confidence: {result.confidence_score:.3f}")
        self.logger.info(f"Refinement Score: {result.refinement_score:.3f}")
        self.logger.info(f"Subject: {result.subject} / {result.sub_subject}")
        self.logger.info(f"Satisfaction: {result.satisfaction_score:.3f}")
        self.logger.info(f"Result saved to: {output_path}")
        self.logger.info("=" * 60)
        self.logger.info("")
        
        return {"result": result}
    
    def run(self, audio_path: str, call_id: Optional[str] = None) -> PipelineState:
        """
        Run the complete pipeline.
        
        Args:
            audio_path: Path to audio file
            call_id: Optional call identifier
        
        Returns:
            Final pipeline state
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate call_id if not provided
        if call_id is None:
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            call_id = f"call_{audio_basename}_{uuid.uuid4().hex[:8]}"
        
        # Initialize state
        initial_state: PipelineState = {
            "audio_path": audio_path,
            "call_id": call_id,
            "run_count": 0,
            "result": CallAnalysisResult(
                call_id=call_id,
                status=ProcessingStatus.PENDING,
                satisfaction_score=0.0  # Default value (not analyzed)
            )
        }
        
        # Run pipeline
        start_time = time.time()
        final_state = self.graph.invoke(initial_state)
        duration = time.time() - start_time
        
        self.logger.info("")
        self.logger.info(f"✅ Pipeline completed in {duration:.2f}s")
        
        return final_state

