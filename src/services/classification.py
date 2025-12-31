"""
Classification service using DziriBERT.
"""
import torch
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.core.base import BaseService, ServiceResult
from src.core.config import get_settings, ModelSettings, ClassificationSettings


class ClassificationService(BaseService):
    """Service for classifying call subjects."""
    
    def __init__(
        self,
        model_settings: ModelSettings = None,
        classification_settings: ClassificationSettings = None
    ):
        super().__init__("classification")
        self.model_settings = model_settings or get_settings().dziribert_classifier
        self.classification_settings = classification_settings or get_settings().classification
        self._tokenizer = None
        self._model = None
        self._device = None
    
    def _get_device(self) -> str:
        """Determine device to use."""
        if self.model_settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model_settings.device
    
    def initialize(self) -> None:
        """Load classification model."""
        if self._initialized:
            return
        
        self.logger.info(f"Loading classification model: {self.model_settings.model_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_settings.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_settings.model_path
        )
        
        self._device = self._get_device()
        self._model.eval()
        if self._device == "cuda":
            self._model = self._model.to(self._device)
        
        self.logger.info(f"✅ Classification model loaded on: {self._device.upper()}")
        self._initialized = True
    
    def process(self, transcript: str) -> ServiceResult:
        """
        Classify transcript into subject categories.
        
        Args:
            transcript: Transcript to classify
        
        Returns:
            ServiceResult with subject and sub_subject
        """
        def _classify():
            self.ensure_initialized()
            
            # Tokenize
            inputs = self._tokenizer(
                transcript,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_settings.max_length or 512,
                padding=True
            )
            
            if self._device == "cuda":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Classify
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                probabilities = torch.softmax(logits, dim=-1)[0]
            
            max_confidence = probabilities[predicted_class].item()
            
            # Get model's label mapping if available
            model_labels = None
            if hasattr(self._model.config, 'id2label') and self._model.config.id2label:
                model_labels = self._model.config.id2label
                self.logger.info(f"Model has {len(model_labels)} labels: {model_labels}")
            elif hasattr(self._model.config, 'label2id') and self._model.config.label2id:
                # Reverse label2id to get id2label
                model_labels = {v: k for k, v in self._model.config.label2id.items()}
                self.logger.info(f"Model has {len(model_labels)} labels: {model_labels}")
            
            # Log prediction details
            self.logger.info("=" * 60)
            self.logger.info("📊 CLASSIFICATION PREDICTION")
            self.logger.info("=" * 60)
            self.logger.info(f"Predicted class index: {predicted_class}")
            self.logger.info(f"Confidence: {max_confidence:.4f}")
            if model_labels:
                model_label = model_labels.get(predicted_class, f"Unknown class {predicted_class}")
                self.logger.info(f"Model label: {model_label}")
            
            # Get all probabilities for debugging
            all_probs = {i: prob.item() for i, prob in enumerate(probabilities)}
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            self.logger.info("Top 3 predictions:")
            for idx, (class_idx, prob) in enumerate(sorted_probs[:3]):
                model_label_str = ""
                if model_labels:
                    model_label_str = f" ({model_labels.get(class_idx, 'Unknown')})"
                self.logger.info(f"  {idx+1}. Class {class_idx}{model_label_str}: {prob:.4f}")
            self.logger.info("=" * 60)
            
            # Determine subject
            subject_labels = self.classification_settings.primary_categories
            
            # If model has its own labels, try to map them
            if model_labels and predicted_class in model_labels:
                model_label = model_labels[predicted_class]
                # Try to find matching category (case-insensitive, partial match)
                matched_subject = None
                for cat in subject_labels:
                    if cat.upper() in model_label.upper() or model_label.upper() in cat.upper():
                        matched_subject = cat
                        break
                
                if matched_subject:
                    subject = matched_subject
                    self.logger.info(f"Mapped model label '{model_label}' to category '{subject}'")
                elif max_confidence < self.classification_settings.other_category_threshold:
                    # Low confidence, use OTHER
                    subject = self.classification_settings.other_category_name
                    self.logger.info(f"Low confidence ({max_confidence:.4f}), using {subject}")
                else:
                    # Model label doesn't match any category, use OTHER
                    subject = self.classification_settings.other_category_name
                    self.logger.warning(f"Model label '{model_label}' doesn't match any category, using {subject}")
            else:
                # Fallback: map by index (original behavior)
                if (predicted_class >= len(subject_labels) or
                    (self.classification_settings.other_category_threshold > 0.0 and
                     max_confidence < self.classification_settings.other_category_threshold)):
                    subject = self.classification_settings.other_category_name
                    self.logger.info(f"Using {subject} (class {predicted_class} out of range or low confidence)")
                else:
                    subject = subject_labels[predicted_class]
                    self.logger.info(f"Mapped class index {predicted_class} to '{subject}'")
            
            # Determine sub-subject
            allowed_subcategories = self.classification_settings.category_subcategories.get(
                subject,
                ["N/A"]
            )
            
            if len(allowed_subcategories) > 1 or (
                len(allowed_subcategories) == 1 and allowed_subcategories[0] != "N/A"
            ):
                default_sub = self.classification_settings.default_subcategory.get(subject)
                if default_sub and default_sub in allowed_subcategories:
                    sub_subject = default_sub
                else:
                    non_na = [sc for sc in allowed_subcategories if sc != "N/A"]
                    sub_subject = non_na[0] if non_na else allowed_subcategories[0]
            else:
                sub_subject = "N/A"
            
            return {
                "subject": subject,
                "sub_subject": sub_subject,
                "confidence": max_confidence
            }
        
        return self._execute_with_timing(_classify)

