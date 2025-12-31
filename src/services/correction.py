"""
Text correction service using local LLM (Qwen).
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, LLMSettings


class CorrectionService(BaseService):
    """Service for correcting transcripts using local LLM."""
    
    def __init__(self, settings: LLMSettings = None):
        super().__init__("correction")
        self.settings = settings or get_settings().qwen
        self._tokenizer = None
        self._model = None
        self._device = None
    
    def _get_device(self) -> str:
        """Determine device to use."""
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.settings.device
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        return torch.float16 if self.settings.dtype == "float16" else torch.float32
    
    def initialize(self) -> None:
        """Load Qwen model."""
        if self._initialized:
            return
        
        self.logger.info(f"Loading correction model: {self.settings.model_path}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_path,
                trust_remote_code=self.settings.trust_remote_code
            )
            
            dtype = self._get_dtype()
            self._device = self._get_device()
            
            # Try loading with dtype parameter (new API)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_path,
                    trust_remote_code=self.settings.trust_remote_code,
                    dtype=dtype,
                    device_map="auto" if self._device == "cuda" else None
                )
            except (ImportError, AttributeError) as e:
                # Fallback: try without device_map and move manually
                self.logger.warning(f"Loading without device_map due to: {e}")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_path,
                    trust_remote_code=self.settings.trust_remote_code,
                    torch_dtype=dtype  # Fallback to old parameter name
                )
                if self._device == "cuda":
                    self._model = self._model.to(self._device)
            
            self._model.eval()
            self.logger.info(f"✅ Correction model loaded on: {self._device.upper()}")
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to load correction model: {e}")
            self.logger.warning("Correction service will be unavailable. Continuing without correction capability.")
            raise
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Correct and normalize transcript.
        
        Args:
            transcript: Original transcript to correct
        
        Returns:
            ServiceResult with corrected transcript
        """
        def _correct():
            self.ensure_initialized()
            
            # Create correction prompt
            prompt = f"""You are a text normalization expert for Algerian Arabic (Darija). 
Your task is to correct and normalize the following transcript, adding proper punctuation, fixing spelling errors, and improving readability while preserving the original meaning.

Transcript: {transcript}

Instructions:
1. Fix any spelling or grammatical errors
2. Add appropriate punctuation
3. Normalize dialectal variations to standard forms where appropriate
4. Preserve the original meaning and context
5. Do not add information that is not in the original transcript

Corrected transcript:"""
            
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self._device == "cuda":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Generate correction
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.max_length or 512,
                    temperature=self.settings.temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode
            corrected = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract corrected part
            if "Corrected transcript:" in corrected:
                corrected = corrected.split("Corrected transcript:")[-1].strip()
            else:
                # Fallback: use original if extraction fails
                corrected = transcript
            
            return {"corrected_transcript": corrected}
        
        return self._execute_with_timing(_correct)

