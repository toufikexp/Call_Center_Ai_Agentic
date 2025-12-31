"""
Transcript refinement service using Gemini API (temporary).
"""
import os
import json
from typing import Optional

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, GeminiSettings


class RefinementService(BaseService):
    """Service for refining transcripts using Gemini API."""
    
    def __init__(self, settings: GeminiSettings = None):
        super().__init__("refinement")
        self.settings = settings or get_settings().gemini
        self._api_configured = False
        self._client = None
    
    def initialize(self) -> None:
        """Initialize Gemini API configuration."""
        if self._initialized:
            return
        
        if not self.settings.api_key:
            self.logger.warning("Gemini API key not configured. Refinement will be skipped.")
        else:
            try:
                # Set API key as environment variable for google.genai
                os.environ["GOOGLE_API_KEY"] = self.settings.api_key
                self._api_configured = True
                
                # Create and cache client using google.genai API
                try:
                    # Try new API first (google-genai package)
                    from google import genai
                    self._client = genai.Client()
                except ImportError:
                    # Fallback: The new API might not be available, use old API temporarily
                    import google.generativeai as genai_old
                    genai_old.configure(api_key=self.settings.api_key)
                    self._client = genai_old
                    self.logger.warning("Using deprecated google.generativeai API. Please install google-genai package for new API.")
                
                self.logger.info(f"✅ Gemini API configured with model: {self.settings.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to configure Gemini API: {e}")
        
        self._initialized = True
    
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Refine transcript using Gemini API.
        
        Args:
            transcript: Raw transcript to refine
        
        Returns:
            ServiceResult with refined transcript and quality score
        """
        def _refine():
            if not self.settings.api_key:
                self.logger.warning("Gemini API key not set, returning original transcript")
                return {"refined_transcript": transcript, "refinement_score": 0.0}
            
            self.ensure_initialized()
            
            if not self._api_configured:
                return {"refined_transcript": transcript, "refinement_score": 0.0}
            
            try:
                # Use cached client (created during initialization)
                if self._client is None:
                    try:
                        from google import genai
                        self._client = genai.Client()
                    except ImportError:
                        import google.generativeai as genai_old
                        genai_old.configure(api_key=self.settings.api_key)
                        self._client = genai_old
                
                # Create refinement prompt from config template
                prompt = self.settings.refinement_prompt_template.format(transcript=transcript)
                
                # Use model from config (gemini-2.0-flash-exp)
                model_name = self.settings.model_name
                
                # Check which API we're using and call accordingly
                if hasattr(self._client, 'models') and hasattr(self._client.models, 'generate_content'):
                    # New API: google.genai
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                elif hasattr(self._client, 'GenerativeModel'):
                    # Old API: google.generativeai
                    model = self._client.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "response_mime_type": "application/json"
                        }
                    )
                else:
                    raise Exception("Unknown API structure")
                
                # Extract JSON from response
                raw_text = ""
                if hasattr(response, 'text'):
                    raw_text = response.text
                elif hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, str):
                        raw_text = content
                    elif isinstance(content, list) and len(content) > 0:
                        first_part = content[0]
                        if hasattr(first_part, 'text'):
                            raw_text = first_part.text
                        elif isinstance(first_part, str):
                            raw_text = first_part
                elif isinstance(response, str):
                    raw_text = response
                else:
                    raw_text = str(response)
                
                # Clean potential markdown backticks and parse JSON
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                
                refined_text = data.get("refined_text", transcript).strip()
                score = float(data.get("score", self.settings.default_refinement_score))
                
                # Validate refinement length using config threshold
                min_length = len(transcript) * self.settings.min_refinement_length_ratio
                if not refined_text or len(refined_text) < min_length:
                    self.logger.warning("Refinement failed or was too short, using original")
                    return {"refined_transcript": transcript, "refinement_score": 0.0}
                
                # Log refinement results
                self.logger.info("=" * 60)
                self.logger.info(f"📝 TRANSCRIPT REFINEMENT RESULT (Score: {score:.2f})")
                self.logger.info("=" * 60)
                self.logger.info(f"Original length: {len(transcript)} characters")
                self.logger.info(f"Refined length: {len(refined_text)} characters")
                self.logger.info("-" * 60)
                self.logger.info("Original transcript:")
                self.logger.info(transcript[:500] + ("..." if len(transcript) > 500 else ""))
                self.logger.info("-" * 60)
                self.logger.info("Refined transcript:")
                self.logger.info(refined_text[:500] + ("..." if len(refined_text) > 500 else ""))
                self.logger.info("=" * 60)
                self.logger.info(f"Quality score: {score:.2f}")
                self.logger.info("=" * 60)
                
                return {"refined_transcript": refined_text, "refinement_score": score}
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                self.logger.error(f"Raw response: {raw_text[:500] if 'raw_text' in locals() else 'N/A'}")
                return {"refined_transcript": transcript, "refinement_score": 0.0}
            except Exception as e:
                self.logger.error(f"Refinement failed: {e}")
                return {"refined_transcript": transcript, "refinement_score": 0.0}
        
        return self._execute_with_timing(_refine)

