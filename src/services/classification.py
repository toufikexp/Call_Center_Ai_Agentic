"""
Classification service using local vLLM API (Qwen3-4B).
"""
import json
import re
from typing import Optional
from openai import OpenAI

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, ClassificationSettings, VLLMSettings


class ClassificationService(BaseService):
    """Service for classifying call subjects using local vLLM API (Qwen3-4B)."""
    
    def __init__(
        self,
        vllm_settings: VLLMSettings = None,
        classification_settings: ClassificationSettings = None
    ):
        super().__init__("classification")
        self.vllm_settings = vllm_settings or get_settings().vllm
        self.classification_settings = classification_settings or get_settings().classification
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
            self.logger.info(f"✅ vLLM API configured for classification with model: {self.vllm_settings.model_name} at {self.vllm_settings.base_url}")
        except Exception as e:
            self.logger.error(f"Failed to configure vLLM API: {e}")
            self.logger.warning("Classification will be skipped if API is unavailable.")
        
        self._initialized = True
    
    def _build_classification_prompt(self, transcript: str) -> str:
        """Build classification prompt with predefined categories and sub-categories from config."""
        categories = self.classification_settings.primary_categories
        
        # Reorder categories to put OTHER first (to emphasize it as default)
        other_cat = self.classification_settings.other_category_name
        if other_cat in categories:
            categories = [other_cat] + [c for c in categories if c != other_cat]
        
        # Build category mapping with descriptions and sub-categories
        category_mapping = []
        for category in categories:
            # Get description if available
            description = self.classification_settings.category_descriptions.get(category, "")
            
            # Get sub-categories if available
            subcategories = self.classification_settings.category_subcategories.get(category, [])
            
            # Build category entry
            cat_entry = f"• {category}"
            if description:
                cat_entry += f"\n  Description: {description}"
            
            if subcategories and subcategories != ["N/A"]:
                cat_entry += f"\n  Sub-categories:"
                # Get subcategory descriptions if available
                subcat_descriptions = self.classification_settings.subcategory_descriptions.get(category, {})
                for subcat in subcategories:
                    subcat_desc = subcat_descriptions.get(subcat, "")
                    if subcat_desc:
                        cat_entry += f"\n    - {subcat}: {subcat_desc}"
                    else:
                        cat_entry += f"\n    - {subcat}"
            else:
                cat_entry += f"\n  Sub-categories: N/A"
            
            category_mapping.append(cat_entry)
        
        categories_text = "\n\n".join(category_mapping)
        
        # Clean transcript - remove speaker labels for classification (they might confuse the model)
        clean_transcript = transcript
        # Remove common speaker label patterns
        clean_transcript = re.sub(r'^(Agent|Customer|Speaker):\s*', '', clean_transcript, flags=re.MULTILINE)
        clean_transcript = re.sub(r'\n(Agent|Customer|Speaker):\s*', ' ', clean_transcript)
        clean_transcript = clean_transcript.strip()
        
        prompt = f"""You are an expert classifier for telecom customer service calls (Ooredoo, Algeria).

TASK: Classify the call transcript into ONE category and ONE sub-category.

AVAILABLE CATEGORIES:
{categories_text}

CLASSIFICATION RULES (READ CAREFULLY):
1. Read the ENTIRE transcript to understand the MAIN topic/subject.
2. Choose the category that BEST matches the PRIMARY purpose of the call.
3. DEFAULT TO "{self.classification_settings.other_category_name}" if uncertain - DO NOT guess!
4. DO NOT use "Customer Service Support Topic" unless the call is EXPLICITLY about:
   - Support process issues
   - Formal complaints about service delivery
   - Support staff interactions/problems
   - Service quality complaints
5. Category selection guide:
   - Network problems (coverage, speed, dropped calls) → "Network"
   - Product/service questions (plans, bundles, features) → "Product Category"
   - Pricing/tariff questions → "Pricing"
   - Channel/platform issues (app, website, store) → "Channel"
   - Device issues (phone, tablet, modem) → "Mobile Device"
   - Brand/loyalty questions → "Brand"
   - Information requests → "Information"
   - Process/registration → "Customer Service Process Related"
   - Support/complaints about service → "Customer Service Support Topic"
   - Anything unclear or doesn't fit → "{self.classification_settings.other_category_name}"
6. For sub-category: Select the MOST SPECIFIC sub-category that matches. If none fit, use "N/A".
7. IMPORTANT: Use EXACT sub-category names as listed (including spaces and dashes).

CRITICAL: 
- When in doubt, choose "{self.classification_settings.other_category_name}" - it's better to be conservative
- "Customer Service Support Topic" is ONLY for support process issues, NOT for general questions
- Network issues go to "Network", NOT "Customer Service Support Topic"
- Product questions go to "Product Category", NOT "Customer Service Support Topic"

OUTPUT FORMAT (JSON only):
{{"subject": "EXACT_CATEGORY_NAME", "sub_subject": "EXACT_SUBCATEGORY_NAME_OR_N/A", "confidence": 0.0-1.0}}

Call Transcript:
{clean_transcript}

JSON Output:"""
        
        return prompt
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Classify transcript into subject categories using local vLLM API (Qwen3-4B).
        
        Args:
            transcript: Transcript to classify
            
        Returns:
            ServiceResult with subject and sub_subject
        """
        def _classify():
            self.ensure_initialized()
            
            if not self._api_configured or self._client is None:
                self.logger.warning("vLLM API not configured. Using fallback classification.")
                return {
                    "subject": self.classification_settings.other_category_name,
                    "sub_subject": "N/A",
                    "confidence": 0.0
                }
            
            try:
                # Build classification prompt
                prompt = self._build_classification_prompt(transcript)
                
                # Call vLLM API using OpenAI-compatible interface
                response = self._client.chat.completions.create(
                    model=self.vllm_settings.model_name,  # Must match the model name in Docker
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.vllm_settings.temperature  # Lower temperature for better classification accuracy
                )
                
                # Extract response text
                raw_text = response.choices[0].message.content
                
                # Clean and parse JSON response
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                
                # Extract classification results
                predicted_subject = data.get("subject", "").strip()
                predicted_sub_subject = data.get("sub_subject", "N/A").strip()
                confidence = float(data.get("confidence", 0.5))
                
                # Validate subject is in predefined categories
                valid_categories = self.classification_settings.primary_categories
                if predicted_subject not in valid_categories:
                    self.logger.warning(
                        f"vLLM returned invalid category '{predicted_subject}'. "
                        f"Valid categories: {valid_categories}. Using '{self.classification_settings.other_category_name}'."
                    )
                    predicted_subject = self.classification_settings.other_category_name
                    predicted_sub_subject = "N/A"
                    confidence = 0.0
                
                # If subject is OTHER, sub_subject should always be N/A
                if predicted_subject == self.classification_settings.other_category_name:
                    predicted_sub_subject = "N/A"
                
                # Validate sub_subject is in allowed sub-categories for the subject
                allowed_subcategories = self.classification_settings.category_subcategories.get(
                    predicted_subject,
                    ["N/A"]
                )
                
                # Normalize sub-category for flexible matching (handle spaces around dashes)
                def normalize_subcat(subcat: str) -> str:
                    """Normalize sub-category by removing extra spaces around dashes."""
                    return re.sub(r'\s*-\s*', ' -', subcat.strip())
                
                normalized_predicted = normalize_subcat(predicted_sub_subject)
                normalized_allowed = {normalize_subcat(sc): sc for sc in allowed_subcategories}
                
                if normalized_predicted not in normalized_allowed:
                    # Try to find a close match (case-insensitive, space-tolerant)
                    predicted_lower = normalized_predicted.lower()
                    matched = None
                    for norm_allowed, original in normalized_allowed.items():
                        if norm_allowed.lower() == predicted_lower:
                            matched = original
                            break
                    
                    if matched:
                        predicted_sub_subject = matched
                        original_sub_subject = data.get("sub_subject", "")
                        self.logger.info(f"Matched sub-category '{predicted_sub_subject}' (normalized from '{original_sub_subject}')")
                    else:
                        self.logger.warning(
                            f"vLLM returned invalid sub-category '{predicted_sub_subject}' for '{predicted_subject}'. "
                            f"Allowed: {allowed_subcategories}. Using default or 'N/A'."
                        )
                        # Use default sub-category if available, otherwise first available or N/A
                        default_sub = self.classification_settings.default_subcategory.get(predicted_subject)
                        if default_sub and default_sub in allowed_subcategories:
                            predicted_sub_subject = default_sub
                        elif allowed_subcategories and allowed_subcategories != ["N/A"]:
                            non_na = [sc for sc in allowed_subcategories if sc != "N/A"]
                            predicted_sub_subject = non_na[0] if non_na else "N/A"
                        else:
                            predicted_sub_subject = "N/A"
                
                # Log classification results
                self.logger.info("=" * 60)
                self.logger.info("📊 CLASSIFICATION RESULT")
                self.logger.info("=" * 60)
                self.logger.info(f"Subject: {predicted_subject}")
                self.logger.info(f"Sub-subject: {predicted_sub_subject}")
                self.logger.info(f"Confidence: {confidence:.4f}")
                self.logger.info("=" * 60)
                
                return {
                    "subject": predicted_subject,
                    "sub_subject": predicted_sub_subject,
                    "confidence": confidence
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse vLLM classification response as JSON: {e}")
                if 'raw_text' in locals():
                    self.logger.error(f"Raw response: {raw_text[:200]}...")
                # Raise exception so orchestrator can detect failure
                raise RuntimeError(f"Classification JSON parse error: {str(e)}")
            except Exception as e:
                self.logger.error(f"vLLM Classification Error: {str(e)}")
                # Raise exception so orchestrator can detect failure
                raise RuntimeError(f"Classification service error: {str(e)}")
        
        return self._execute_with_timing(_classify)
