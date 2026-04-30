"""
Sentiment analysis service using local vLLM API (Qwen3-4B).
"""
import json
from typing import Optional
from openai import OpenAI

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, VLLMSettings


class SentimentService(BaseService):
    """Service for analyzing customer satisfaction using local vLLM API (Qwen3-4B)."""
    
    def __init__(
        self,
        vllm_settings: VLLMSettings = None
    ):
        super().__init__("sentiment")
        self.vllm_settings = vllm_settings or get_settings().vllm
        self._api_configured = False
        self._client: Optional[OpenAI] = None
    
    def initialize(self) -> None:
        """Initialize the OpenAI Client for vLLM API."""
        if self._initialized:
            return
        
        try:
            # Initialize the OpenAI Client for vLLM
            self._client = OpenAI(
                base_url=self.vllm_settings.base_url,
                api_key=self.vllm_settings.api_key  # vLLM doesn't require a key by default
            )
            self._api_configured = True
            self.logger.info(f"✅ vLLM API configured for sentiment analysis with model: {self.vllm_settings.model_name} at {self.vllm_settings.base_url}")
        except Exception as e:
            self.logger.error(f"Failed to configure vLLM API: {e}")
            self.logger.warning("Sentiment analysis will be skipped if API is unavailable.")
        
        self._initialized = True
    
    def _build_sentiment_prompt(self, transcript: str) -> str:
        """Build sentiment analysis prompt."""
        prompt = f"""You are an expert in analyzing customer satisfaction for a telecom company (Ooredoo) in Algeria.

Analyze the sentiment and customer satisfaction level from the following call transcript between a subscriber and a customer service agent.

INSTRUCTIONS:
1. Read the transcript carefully and understand the overall tone, emotion, and satisfaction level of the customer.
2. Consider:
   - The customer's tone (polite, frustrated, angry, happy, satisfied, etc.)
   - The resolution of their issue or request
   - Their level of satisfaction with the service received
   - Expressions of gratitude, complaints, or appreciation
   - Overall sentiment throughout the conversation
3. Rate the customer satisfaction on a scale of 1 to 10, where:
   - 1-3: Very dissatisfied (angry, frustrated, unresolved issues, complaints)
   - 4-5: Somewhat dissatisfied (neutral to negative, minor issues)
   - 6-7: Neutral to somewhat satisfied (issues addressed but not exceptional)
   - 8-9: Satisfied (positive experience, issues resolved, polite interaction)
   - 10: Very satisfied (highly positive, grateful, excellent service)

OUTPUT FORMAT:
You MUST return a valid JSON object with exactly these keys:
- "satisfaction_score": A number between 1.0 and 10.0 representing customer satisfaction
- "sentiment_label": One of "POSITIVE", "NEUTRAL", or "NEGATIVE"
- "confidence": A number between 0.0 and 1.0 indicating your confidence in the analysis
- "reasoning": A brief explanation (one sentence) of why this score was assigned

Example outputs:
{{"satisfaction_score": 8.5, "sentiment_label": "POSITIVE", "confidence": 0.92, "reasoning": "Customer expressed gratitude and the issue was resolved satisfactorily"}}
{{"satisfaction_score": 3.0, "sentiment_label": "NEGATIVE", "confidence": 0.88, "reasoning": "Customer was frustrated and the issue was not fully resolved"}}
{{"satisfaction_score": 6.0, "sentiment_label": "NEUTRAL", "confidence": 0.75, "reasoning": "Neutral interaction with standard service delivery"}}

Call Transcript:
{transcript}

JSON Output:"""
        
        return prompt
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Analyze sentiment and return satisfaction score using local vLLM API (Qwen3-4B).
        
        Args:
            transcript: Transcript to analyze
            
        Returns:
            ServiceResult with satisfaction score (1-10 scale, default 0.0 on error)
        """
        def _analyze():
            self.ensure_initialized()
            
            if not self._api_configured or self._client is None:
                self.logger.warning("vLLM API not configured. Using fallback sentiment analysis.")
                return {
                    "satisfaction_score": 0.0,
                    "sentiment_label": "",
                    "confidence": 0.0,
                    "reasoning": "",
                }
            
            sentiment_label = "N/A"
            confidence = 0.0
            reasoning = "N/A"
            satisfaction = 0.0  # Default value
            
            try:
                # Build sentiment analysis prompt
                prompt = self._build_sentiment_prompt(transcript)
                
                # Call vLLM API using OpenAI-compatible interface
                response = self._client.chat.completions.create(
                    model=self.vllm_settings.model_name,  # Must match the model name in Docker
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.vllm_settings.temperature  # Lower temperature for better accuracy
                )
                
                # Extract response text
                raw_text = response.choices[0].message.content
                
                # Clean and parse JSON response
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                
                # Extract sentiment results
                satisfaction = float(data.get("satisfaction_score", 0.0))
                sentiment_label = data.get("sentiment_label", "N/A").strip()
                confidence = float(data.get("confidence", 0.0))
                reasoning = data.get("reasoning", "N/A").strip()
                
                # Validate satisfaction score is in 1-10 range
                if satisfaction < 1.0 or satisfaction > 10.0:
                    self.logger.warning(
                        f"vLLM returned invalid satisfaction score '{satisfaction}'. "
                        f"Valid range is 1.0-10.0. Using 0.0 (default)."
                    )
                    satisfaction = 0.0
                    confidence = 0.0
                else:
                    # Ensure score is within 1-10 range (clamp if needed)
                    satisfaction = max(1.0, min(10.0, satisfaction))
                
                # Validate sentiment label
                valid_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
                if sentiment_label.upper() not in valid_labels:
                    self.logger.warning(
                        f"vLLM returned invalid sentiment label '{sentiment_label}'. "
                        f"Valid labels: {valid_labels}. Using ''."
                    )
                    sentiment_label = ""
                else:
                    sentiment_label = sentiment_label.upper()
                
                # Log sentiment results
                self.logger.info("=" * 60)
                self.logger.info("😊 SENTIMENT RESULT")
                self.logger.info("=" * 60)
                self.logger.info(f"Satisfaction Score: {satisfaction:.2f}/10.0")
                self.logger.info(f"Sentiment Label: {sentiment_label}")
                self.logger.info(f"Confidence: {confidence:.4f}")
                if reasoning != "N/A":
                    self.logger.info(f"Reasoning: {reasoning}")
                self.logger.info("=" * 60)

                return {
                    "satisfaction_score": float(satisfaction),
                    "sentiment_label": sentiment_label,
                    "confidence": float(confidence),
                    "reasoning": reasoning,
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse vLLM sentiment response as JSON: {e}")
                if 'raw_text' in locals():
                    self.logger.error(f"Raw response: {raw_text[:200]}...")
                # Raise exception so orchestrator can detect failure
                raise RuntimeError(f"Sentiment JSON parse error: {str(e)}")
            except Exception as e:
                self.logger.error(f"vLLM Sentiment Analysis Error: {str(e)}")
                # Raise exception so orchestrator can detect failure
                raise RuntimeError(f"Sentiment analysis service error: {str(e)}")
        
        return self._execute_with_timing(_analyze)
