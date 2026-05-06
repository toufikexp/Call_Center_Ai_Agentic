"""
Pipeline orchestrator using LangGraph.
"""
import os
import uuid
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from langgraph.graph import StateGraph, END

from src.core.state import PipelineState, CallAnalysisResult, SegmentResult, ProcessingStatus
from src.core.config import get_settings, Settings
from src.services.preprocessing import PreprocessingService
from src.services.transcription import TranscriptionService
from src.services.refinement import RefinementService
from src.services.classification import ClassificationService
from src.services.sentiment import SentimentService


class CallAnalysisPipeline:
    """Main pipeline orchestrator for call center analysis."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.logger = self._create_logger()

        # Initialize services
        self.preprocessing_service = PreprocessingService(self.settings.preprocessing)
        self.transcription_service = TranscriptionService(self.settings.whisper)
        self.refinement_service = RefinementService(self.settings.gemini)
        self.classification_service = ClassificationService(
            self.settings.vllm,
            self.settings.classification
        )
        self.sentiment_service = SentimentService(self.settings.vllm)

        # Optional persistence layer. Failures here never block the pipeline:
        # JSON output under data/results/ remains the durable per-call artifact.
        self._results_store = None
        self._current_batch_id: Optional[str] = None
        if self.settings.storage.enable:
            try:
                from src.storage import ResultsStore  # lazy import — psycopg only required when used
                self._results_store = ResultsStore(self.settings.storage.database_url)
            except Exception as e:
                self.logger.warning(
                    f"Results storage disabled (init failed): {e}. "
                    f"JSON output under {self.settings.pipeline.output_dir} is unaffected."
                )

        # Build graph
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Batch lifecycle (optional, only meaningful when storage is enabled)
    # ------------------------------------------------------------------
    def start_batch(
        self,
        file_count: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Optional[str]:
        """Begin a batch context. Subsequent run() calls record into this
        batch. Returns the batch_id, or None when storage is disabled."""
        if not self._results_store:
            return None
        batch_id = str(uuid.uuid4())
        self._results_store.start_batch(batch_id, file_count=file_count, notes=notes)
        self._current_batch_id = batch_id
        self.logger.info(f"📦 Batch started: {batch_id}")
        return batch_id

    def finish_batch(self) -> None:
        """Stamp the current batch as finished and aggregate counts."""
        if not self._results_store or not self._current_batch_id:
            return
        self._results_store.finish_batch(self._current_batch_id)
        self.logger.info(f"📦 Batch finished: {self._current_batch_id}")
        self._current_batch_id = None
    
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
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("transcribe", self._transcribe_node)
        workflow.add_node("refine", self._refine_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("analyze_sentiment", self._sentiment_node)
        workflow.add_node("save_result", self._save_node)

        # Set entry point
        workflow.set_entry_point("preprocess")

        # Preprocessing failure short-circuits to save with ERROR status.
        workflow.add_conditional_edges(
            "preprocess",
            self._check_error,
            {
                "continue": "transcribe",
                "error": "save_result",
            },
        )
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
    
    def _preprocess_node(self, state: PipelineState) -> PipelineState:
        """Channel split + VAD segmentation."""
        result = state["result"]
        audio_path = state["audio_path"]

        service_result = self.preprocessing_service.process(audio_path)

        if not service_result.success:
            result.status = ProcessingStatus.ERROR
            result.error_message = service_result.error or "Preprocessing failed"
            self.logger.error(f"Preprocessing failed: {result.error_message}")
            return {"result": result, "segments": []}

        data = service_result.data
        # Audio metadata: persisted in calls.duration_s / calls.channel_count
        # via the storage layer, and surfaced in the JSON output for analytics.
        result.audio_duration_s = float(data.get("audio_duration_s", 0.0))
        result.channel_count = int(data.get("channel_count", 0))

        segments = data.get("segments", [])
        if not segments:
            # No speech detected — treat as low-quality input headed to manual review.
            self.logger.warning("No speech segments detected; routing to manual review")
            result.transcript = ""
            result.confidence_score = 0.0
        return {"result": result, "segments": segments}

    def _transcribe_node(self, state: PipelineState) -> PipelineState:
        """Transcribe pre-segmented audio."""
        result = state["result"]
        run_count = state.get("run_count", 0)
        segments = state.get("segments", [])

        if not segments:
            # Nothing to transcribe — leave result empty, routing will pick it up.
            return {"result": result, "run_count": run_count + 1}

        service_result = self.transcription_service.process(segments)

        if service_result.success:
            data = service_result.data
            result.transcript = data["transcript"]
            result.confidence_score = data["confidence"]
            result.whisper_adapter_version = data.get("adapter_version", "")
            result.segments = [
                SegmentResult(**s) for s in data.get("segments", [])
            ]
        else:
            result.transcript = f"Error: {service_result.error}"
            result.confidence_score = 0.0
            result.status = ProcessingStatus.ERROR
            result.error_message = service_result.error

        return {"result": result, "run_count": run_count + 1}


    def _refine_node(self, state: PipelineState) -> PipelineState:
        """Refine transcript node."""
        result = state["result"]

        # Check if already in error state, skip processing
        if result.status == ProcessingStatus.ERROR:
            return {"result": result}

        # Empty transcript (e.g. preprocessing detected no speech) → skip
        # the Gemini call and let routing send the run to manual review.
        if not result.transcript.strip():
            self.logger.info("Empty transcript; skipping refinement")
            result.refined_transcript = ""
            result.refinement_score = 0.0
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
        
        result.subject = service_result.data.get("subject", "UNKNOWN")
        result.sub_subject = service_result.data.get("sub_subject", "N/A")
        result.classification_confidence = float(service_result.data.get("confidence", 0.0))

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
        result.sentiment_label = service_result.data.get("sentiment_label", "")
        result.sentiment_reasoning = service_result.data.get("reasoning", "")

        return {"result": result}
    
    def _save_node(self, state: PipelineState) -> PipelineState:
        """Save result node."""
        result = state["result"]
        audio_path = state["audio_path"]
        
        # Determine final status
        if result.status == ProcessingStatus.ERROR:
            pass  # Keep error status - pipeline stopped due to error
        elif result.refinement_score < self.settings.pipeline.refinement_threshold:
            result.status = ProcessingStatus.MANUAL_REVIEW
        elif result.confidence_score < self.settings.pipeline.confidence_threshold:
            result.status = ProcessingStatus.MANUAL_REVIEW
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
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("📋 CALL ANALYSIS RESULT")
        self.logger.info("=" * 60)
        self.logger.info(f"Call ID: {result.call_id}")
        self.logger.info(f"Audio: {audio_path}")
        self.logger.info(f"Status: {result.status.value}")
        self.logger.info(f"Confidence: {result.confidence_score:.3f}")
        self.logger.info(f"Refinement Score: {result.refinement_score:.3f}")
        self.logger.info(f"Segments: {len(result.segments)}")
        self.logger.info(f"Adapter: {result.whisper_adapter_version or '<merged>'}")
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
            "segments": [],
            "result": CallAnalysisResult(
                call_id=call_id,
                status=ProcessingStatus.PENDING,
                satisfaction_score=0.0  # Default value (not analyzed)
            )
        }

        # Run pipeline
        started_at = datetime.now(timezone.utc)
        start_time = time.time()
        final_state = self.graph.invoke(initial_state)
        duration = time.time() - start_time
        finished_at = datetime.now(timezone.utc)

        self.logger.info("")
        self.logger.info(f"✅ Pipeline completed in {duration:.2f}s")

        # Persist to PostgreSQL when storage is enabled. JSON output under
        # data/results/ has already been written by _save_node and stays as
        # the source of truth for replay. DB problems are logged inside the
        # store and never raise.
        if self._results_store:
            self._results_store.record_attempt(
                call_id=final_state["result"].call_id,
                audio_path=audio_path,
                result=final_state["result"],
                batch_id=self._current_batch_id,
                started_at=started_at,
                finished_at=finished_at,
            )

        return final_state

