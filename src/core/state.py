"""
Pipeline state management.
"""
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    ERROR = "ERROR"


class CallAnalysisResult(BaseModel):
    """Result of call analysis."""
    call_id: str = Field(default="", description="Unique call identifier")
    transcript: str = Field(default="", description="Transcribed text")
    refined_transcript: str = Field(default="", description="Refined transcript")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    is_corrected: bool = Field(default=False, description="Whether transcript was corrected")
    subject: str = Field(default="", description="Primary classification category")
    sub_subject: str = Field(default="", description="Sub-category classification")
    satisfaction_score: float = Field(default=5.0, ge=1.0, le=10.0, description="Satisfaction score (1-10 scale)")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if any")


class PipelineState(TypedDict):
    """State carried through the pipeline."""
    audio_path: str
    call_id: str
    run_count: int
    result: CallAnalysisResult

