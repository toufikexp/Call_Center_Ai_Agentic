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
            refined_text = transcript
            score = 0.0
            status = "FAILED"
            
            if not self.settings.api_key:
                # No API key is expected - not an error, just skip refinement
                status = "SKIPPED (no API key)"
            else:
                self.ensure_initialized()
                
                if not self._api_configured or self._client is None:
                    # API not configured is expected - not an error
                    status = "SKIPPED (API not configured)"
                else:
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
                            refined_text = transcript
                            score = 0.0
                            status = "FAILED (truncated text)"
                        else:
                            status = "SUCCESS"
                        
                    except json.JSONDecodeError as e:
                        refined_text = transcript
                        score = 0.0
                        status = f"FAILED (JSON parse error: {str(e)})"
                        # Raise exception for critical errors (API connection issues)
                        error_str = str(e).lower()
                        if "connection" in error_str or "timeout" in error_str or "name resolution" in error_str or "errno" in error_str:
                            raise RuntimeError(f"Refinement API connection error: {str(e)}")
                    except Exception as e:
                        refined_text = transcript
                        score = 0.0
                        status = f"FAILED (API error: {str(e)})"
                        # Raise exception for critical errors (API connection issues)
                        error_str = str(e).lower()
                        if "connection" in error_str or "timeout" in error_str or "name resolution" in error_str or "errno" in error_str:
                            raise RuntimeError(f"Refinement API connection error: {str(e)}")
            
            # Unified logging format
            self.logger.info("=" * 60)
            self.logger.info("✨ REFINEMENT RESULT")
            self.logger.info("=" * 60)
            self.logger.info(f"Status: {status}")
            self.logger.info(f"Refinement Score: {score:.2f}")
            self.logger.info(f"Original: {transcript[:100]}...")
            self.logger.info(f"Refined:  {refined_text[:100]}...")
            self.logger.info("=" * 60)
                
            return {"refined_transcript": refined_text, "refinement_score": score}
        
        return self._execute_with_timing(_refine)
