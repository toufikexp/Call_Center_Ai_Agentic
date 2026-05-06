"""
Pipeline state management.
"""
from typing import TypedDict, Optional, List, Any
from pydantic import BaseModel, Field
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    ERROR = "ERROR"


class SegmentResult(BaseModel):
    """Per-segment transcription result.

    A segment is a VAD-cut speech region from one channel of the input
    audio. Segments are produced by `PreprocessingService` and consumed
    (and annotated with text + confidence) by `TranscriptionService`.
    """
    channel: str = Field(default="unknown", description="Speaker channel: 'agent', 'client', or 'unknown'")
    start_ms: int = Field(default=0, ge=0, description="Segment start in milliseconds from the start of the call")
    end_ms: int = Field(default=0, ge=0, description="Segment end in milliseconds from the start of the call")
    text: str = Field(default="", description="Transcribed text for this segment")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Per-segment Whisper confidence (exp mean token log-prob)")


class CallAnalysisResult(BaseModel):
    """Result of call analysis."""
    call_id: str = Field(default="", description="Unique call identifier")
    transcript: str = Field(default="", description="Reconstructed dialogue: speaker-tagged turns concatenated in chronological order")
    refined_transcript: str = Field(default="", description="Refined transcript")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average per-segment Whisper confidence")
    refinement_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Refinement quality score")
    subject: str = Field(default="", description="Primary classification category")
    sub_subject: str = Field(default="", description="Sub-category classification")
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Classifier self-reported confidence")
    satisfaction_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Satisfaction score (0-10 scale, 0 = not analyzed)")
    sentiment_label: str = Field(default="", description="Sentiment label (POSITIVE, NEUTRAL, NEGATIVE)")
    sentiment_reasoning: str = Field(default="", description="One-sentence justification for the satisfaction score")
    segments: List[SegmentResult] = Field(default_factory=list, description="Per-segment transcription details")
    audio_duration_s: float = Field(default=0.0, ge=0.0, description="Original audio duration in seconds (computed during preprocessing)")
    channel_count: int = Field(default=0, ge=0, description="Number of audio channels in the input file")
    whisper_adapter_version: str = Field(default="", description="Identifier of the LoRA adapter used (folder name), or '' if a full merged checkpoint was used")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if any")


class PipelineState(TypedDict, total=False):
    """State carried through the pipeline."""
    audio_path: str
    call_id: str
    run_count: int
    result: CallAnalysisResult
    # Populated by `preprocess`, consumed by `transcribe`. Each entry is a dict
    # with keys: channel (str), start_ms (int), end_ms (int), audio (np.ndarray).
    # Kept out of `CallAnalysisResult` because it carries raw audio arrays.
    segments: List[Any]
