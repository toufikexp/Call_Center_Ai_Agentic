"""
Transcript refinement service using Gemini API (temporary).
"""
import os
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
        self._cached_model = None
    
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
                
                # Create and cache client using new google.genai API
                from google import genai
                self._client = genai.Client()
                
                # Cache available model (only once during initialization)
                self._cached_model = self._find_available_model()
                
                self.logger.info("✅ Gemini API configured")
            except Exception as e:
                self.logger.error(f"Failed to configure Gemini API: {e}")
        
        self._initialized = True
    
    def _find_available_model(self) -> Optional[str]:
        """Find and cache an available Gemini model (called once during init)."""
        if self._client is None:
            return None
        
        try:
            # List available models using new API
            models = self._client.models.list()
            available_models = [m.name for m in models if hasattr(m, 'name')]
            self.logger.info(f"Found {len(available_models)} available Gemini models")
            
            # Look for gemini-2.0-flash-exp (preferred model)
            preferred_name = "gemini-2.0-flash-exp"
            for avail in available_models:
                if preferred_name in avail or avail.endswith(preferred_name):
                    self.logger.info(f"Cached model: {avail}")
                    return avail
            
            # If not found, log warning and return None (will use default in process)
            self.logger.warning(f"Preferred model {preferred_name} not found in available models")
            return None
        except Exception as e:
            self.logger.warning(f"Could not list models: {e}. Will use default model name.")
            return None
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Refine transcript using Gemini API.
        
        Args:
            transcript: Raw transcript to refine
        
        Returns:
            ServiceResult with refined transcript
        """
        def _refine():
            if not self.settings.api_key:
                self.logger.warning("Gemini API key not set, returning original transcript")
                return {"refined_transcript": transcript}
            
            self.ensure_initialized()
            
            if not self._api_configured:
                return {"refined_transcript": transcript}
            
            try:
                # Use cached client (created during initialization)
                if self._client is None:
                    from google import genai
                    self._client = genai.Client()
                
                # Create refinement prompt
                prompt = f"""You are a text refinement expert for Algerian Arabic (Darija) call center transcripts.

Your task is to refine and improve the following transcript by:
1. Fixing spelling and grammatical errors
2. Adding appropriate punctuation
3. Normalizing dialectal variations where appropriate
4. Improving readability while preserving the original meaning
5. Removing obvious repetitions
6. DO NOT add information that is not in the original transcript

Original transcript:
{transcript}

Refined transcript:"""
                
                # Use gemini-2.0-flash-exp model (single model, no iteration)
                model_name = "gemini-2.0-flash-exp"
                
                # Try with cached model first if available
                if self._cached_model:
                    model_name = self._cached_model
                
                try:
                    # Use new google.genai API
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                except Exception as e:
                    # If cached model fails, try default model name
                    if model_name != "gemini-2.0-flash-exp":
                        try:
                            self.logger.warning(f"Cached model {model_name} failed, trying default: gemini-2.0-flash-exp")
                            response = self._client.models.generate_content(
                                model="gemini-2.0-flash-exp",
                                contents=prompt
                            )
                        except Exception as e2:
                            raise Exception(f"Failed to use Gemini model: {e2}")
                    else:
                        raise Exception(f"Failed to use Gemini model {model_name}: {e}")
                
                # Extract text from response
                # The new API response structure: check for text attribute or direct string
                refined_text = ""
                if hasattr(response, 'text'):
                    refined_text = response.text
                elif hasattr(response, 'content'):
                    # Sometimes content is a list or object
                    content = response.content
                    if isinstance(content, str):
                        refined_text = content
                    elif isinstance(content, list) and len(content) > 0:
                        # Extract text from first content part
                        first_part = content[0]
                        if hasattr(first_part, 'text'):
                            refined_text = first_part.text
                        elif isinstance(first_part, str):
                            refined_text = first_part
                elif isinstance(response, str):
                    refined_text = response
                else:
                    # Fallback: try to convert to string
                    refined_text = str(response)
                
                refined_text = refined_text.strip()
                
                # Validate refinement
                if not refined_text or len(refined_text) < len(transcript) * 0.5:
                    self.logger.warning("Refinement too short, using original")
                    return {"refined_transcript": transcript}
                
                # Log refinement results
                self.logger.info("=" * 60)
                self.logger.info("📝 TRANSCRIPT REFINEMENT RESULT")
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
                
                return {"refined_transcript": refined_text}
                
            except Exception as e:
                self.logger.error(f"Refinement failed: {e}")
                return {"refined_transcript": transcript}
        
        return self._execute_with_timing(_refine)

