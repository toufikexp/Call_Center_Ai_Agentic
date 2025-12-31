"""
Sentiment analysis service using DziriBERT.
"""
from transformers import pipeline, AutoTokenizer

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, ModelSettings


class SentimentService(BaseService):
    """Service for analyzing customer satisfaction."""
    
    def __init__(self, model_settings: ModelSettings = None):
        super().__init__("sentiment")
        self.model_settings = model_settings or get_settings().dziribert_sentiment
        self._pipeline = None
        self._tokenizer = None
        self._max_length = None
    
    def initialize(self) -> None:
        """Load sentiment analysis pipeline."""
        if self._initialized:
            return
        
        self.logger.info(f"Loading sentiment model: {self.model_settings.model_path}")
        
        device = 0 if (self.model_settings.device == "auto" and self._check_cuda()) else -1
        
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_settings.model_path,
            tokenizer=self.model_settings.model_path,
            device=device
        )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_settings.model_path)
        self._max_length = self.model_settings.max_length or getattr(
            self._tokenizer,
            'model_max_length',
            512
        )
        
        self.logger.info("✅ Sentiment model loaded")
        self._initialized = True
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _truncate_transcript(self, transcript: str) -> str:
        """Truncate transcript to fit model's max length."""
        if not transcript:
            return transcript
        
        tokens = self._tokenizer.encode(transcript, add_special_tokens=False)
        effective_max = max(1, self._max_length - 10)
        
        if len(tokens) <= effective_max:
            return transcript
        
        # Use sliding window: first 60% + last 40%
        first_portion = int(effective_max * 0.6)
        last_portion = effective_max - first_portion
        
        first_tokens = tokens[:first_portion]
        last_tokens = tokens[-last_portion:] if last_portion > 0 else []
        
        combined_tokens = first_tokens + last_tokens
        truncated = self._tokenizer.decode(combined_tokens, skip_special_tokens=True)
        
        self.logger.warning(
            f"Transcript truncated from {len(tokens)} to {len(combined_tokens)} tokens"
        )
        
        return truncated
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Analyze sentiment and return satisfaction score.
        
        Args:
            transcript: Transcript to analyze
        
        Returns:
            ServiceResult with satisfaction score (1-10 scale)
        """
        def _analyze():
            self.ensure_initialized()
            
            processed_transcript = self._truncate_transcript(transcript)
            
            try:
                result = self._pipeline(
                    processed_transcript,
                    truncation=True,
                    max_length=self._max_length
                )
            except Exception as e:
                self.logger.error(f"Sentiment analysis failed: {e}")
                return {"satisfaction_score": 5.0}  # Neutral score (middle of 1-10)
            
            # Extract sentiment score
            if isinstance(result, list) and len(result) > 0:
                sentiment_label = result[0].get("label", "NEUTRAL")
                sentiment_score = result[0].get("score", 0.5)
                
                # Map to satisfaction score (1-10 scale)
                # Positive: 5-10 (neutral to very positive)
                # Negative: 1-5 (very negative to neutral)
                # Neutral: 5.0
                if "POSITIVE" in sentiment_label.upper() or "POS" in sentiment_label.upper():
                    # Map sentiment_score (0-1) to 5-10 range
                    satisfaction = 5.0 + (sentiment_score * 5.0)
                elif "NEGATIVE" in sentiment_label.upper() or "NEG" in sentiment_label.upper():
                    # Map sentiment_score (0-1) to 1-5 range
                    satisfaction = 5.0 - (sentiment_score * 4.0)
                else:
                    # Neutral sentiment
                    satisfaction = 5.0
            else:
                satisfaction = 5.0  # Default to neutral
            
            # Ensure score is within 1-10 range
            satisfaction = max(1.0, min(10.0, satisfaction))
            
            return {"satisfaction_score": float(satisfaction)}
        
        return self._execute_with_timing(_analyze)

