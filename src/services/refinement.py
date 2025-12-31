"""
Transcript refinement service using the new google-genai SDK.
"""
import os
import json
from typing import Optional
from google import genai

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, GeminiSettings


class RefinementService(BaseService):
    """Service for refining transcripts using the updated Gemini API."""
    
    def __init__(self, settings: GeminiSettings = None):
        super().__init__("refinement")
        self.settings = settings or get_settings().gemini
        self._api_configured = False
        self._client: Optional[genai.Client] = None
    
    def initialize(self) -> None:
        """Initialize the new Gemini Client."""
        if self._initialized:
            return
        
        if not self.settings.api_key:
            self.logger.warning("Gemini API key not configured. Refinement will be skipped.")
        else:
            try:
                # Initialize the new Client
                self._client = genai.Client(api_key=self.settings.api_key)
                self._api_configured = True
                self.logger.info(f"✅ Gemini API (v2.0) configured with model: {self.settings.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to configure Gemini API: {e}")
        
        self._initialized = True
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Refine transcript using the new google-genai Client.
        """
        def _refine():
            if not self.settings.api_key:
                return {"refined_transcript": transcript, "refinement_score": 0.0}
            
            self.ensure_initialized()
            
            if not self._api_configured or self._client is None:
                return {"refined_transcript": transcript, "refinement_score": 0.0}
            
            try:
                # Create refinement prompt
                prompt = self.settings.refinement_prompt_template.format(transcript=transcript)
                
                # Use the new models.generate_content syntax
                # Note: Config moves to a dictionary or GenerateConfig object
                response = self._client.models.generate_content(
                    model=self.settings.model_name,
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
                
                # Get text from the new response object
                raw_text = response.text
                
                # Clean and parse JSON
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                
                refined_text = data.get("refined_text", transcript).strip()
                score = float(data.get("score", self.settings.default_refinement_score))
                
                # Length validation
                min_length = len(transcript) * self.settings.min_refinement_length_ratio
                if not refined_text or len(refined_text) < min_length:
                    self.logger.warning("Refinement resulted in truncated text, using original.")
                    return {"refined_transcript": transcript, "refinement_score": 0.0}
                
                # Detailed logging for Algerian dialect monitoring
                self.logger.info("-" * 30)
                self.logger.info(f"✨ Refinement Score: {score:.2f}")
                self.logger.info(f"Original: {transcript[:100]}...")
                self.logger.info(f"Refined:  {refined_text[:100]}...")
                self.logger.info("-" * 30)
                
                return {"refined_transcript": refined_text, "refinement_score": score}
                
            except Exception as e:
                self.logger.error(f"Gemini Refinement Error: {str(e)}")
                return {"refined_transcript": transcript, "refinement_score": 0.0}
        
        return self._execute_with_timing(_refine)
