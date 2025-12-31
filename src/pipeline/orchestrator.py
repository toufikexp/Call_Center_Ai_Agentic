"""
Pipeline orchestrator using LangGraph.
"""
import os
import uuid
import json
import time
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
        
        # Initialize services
        self.transcription_service = TranscriptionService(self.settings.whisper)
        self.refinement_service = RefinementService(self.settings.gemini)
        self.correction_service = CorrectionService(self.settings.qwen)
        self.classification_service = ClassificationService(
            self.settings.dziribert_classifier,
            self.settings.classification
        )
        self.sentiment_service = SentimentService(self.settings.dziribert_sentiment)
        
        # Build graph
        self.graph = self._build_graph()
    
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
                "correct": "correct",
                "manual_review": "save_result"
            }
        )
        
        # Loop back from correct
        workflow.add_edge("correct", "transcribe")
        
        # Success path
        workflow.add_edge("classify", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "save_result")
        
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
        
        # service_result = self.refinement_service.process(result.transcript)
        service_result = self.refinement_service.process("")
        
        if service_result.success:
            result.refined_transcript = service_result.data["refined_transcript"]
        else:
            # Keep original if refinement fails
            result.refined_transcript = result.transcript
        
        return {"result": result}
    
    def _verify_node(self, state: PipelineState) -> PipelineState:
        """Verify confidence node."""
        result = state["result"]
        result.status = ProcessingStatus.IN_PROGRESS
        return {"result": result}
    
    def _route_decision(self, state: PipelineState) -> str:
        """Routing decision based on confidence and run count."""
        result = state["result"]
        run_count = state["run_count"]
        confidence = result.confidence_score
        
        if confidence >= self.settings.pipeline.confidence_threshold:
            return "proceed"
        elif run_count <= 1:
            return "correct"
        else:
            return "manual_review"
    
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
                print(f"[WARNING] Correction failed: {service_result.error}. Using original transcript.")
        except Exception as e:
            # If correction service is not available, skip correction
            result.is_corrected = False
            print(f"[WARNING] Correction service unavailable: {e}. Skipping correction step.")
        
        return {"result": result}
    
    def _classify_node(self, state: PipelineState) -> PipelineState:
        """Classify subject node."""
        result = state["result"]
        transcript = result.refined_transcript or result.transcript
        
        service_result = self.classification_service.process(transcript)
        
        if service_result.success:
            result.subject = service_result.data["subject"]
            result.sub_subject = service_result.data["sub_subject"]
        else:
            result.subject = "UNKNOWN"
            result.sub_subject = "N/A"
        
        return {"result": result}
    
    def _sentiment_node(self, state: PipelineState) -> PipelineState:
        """Analyze sentiment node."""
        result = state["result"]
        transcript = result.refined_transcript or result.transcript
        
        service_result = self.sentiment_service.process(transcript)
        
        if service_result.success:
            result.satisfaction_score = service_result.data["satisfaction_score"]
        else:
            result.satisfaction_score = 5.0  # Default to neutral (middle of 1-10 scale)
        
        return {"result": result}
    
    def _save_node(self, state: PipelineState) -> PipelineState:
        """Save result node."""
        result = state["result"]
        audio_path = state["audio_path"]
        
        # Determine final status
        if result.status == ProcessingStatus.ERROR:
            pass  # Keep error status
        elif result.confidence_score < self.settings.pipeline.confidence_threshold:
            if state["run_count"] >= self.settings.pipeline.max_retry_attempts:
                result.status = ProcessingStatus.MANUAL_REVIEW
            else:
                result.status = ProcessingStatus.LOW_CONFIDENCE
        else:
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
        print("\n" + "=" * 60)
        print("CALL ANALYSIS RESULT")
        print("=" * 60)
        print(f"Call ID: {result.call_id}")
        print(f"Audio: {audio_path}")
        print(f"Status: {result.status.value}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Subject: {result.subject} / {result.sub_subject}")
        print(f"Satisfaction: {result.satisfaction_score:.3f}")
        print(f"Result saved to: {output_path}")
        print("=" * 60 + "\n")
        
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
                satisfaction_score=5.0  # Default to neutral (middle of 1-10 scale)
            )
        }
        
        # Run pipeline
        start_time = time.time()
        final_state = self.graph.invoke(initial_state)
        duration = time.time() - start_time
        
        print(f"\n✅ Pipeline completed in {duration:.2f}s")
        
        return final_state

