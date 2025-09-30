"""
Emotion Transformer - Enterprise-grade emotion detection using BERT/RoBERTa models.

This module provides ML-based emotion detection with:
- Pre-trained transformer models (BERT, RoBERTa)
- Real-time emotion classification
- Adaptive threshold learning
- Multi-model ensemble fusion
- Production-ready error handling

Author: MasterX AI Team
Version: 1.0 (Enhanced from v9.0)
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Transformer imports with graceful fallback
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        logging as transformers_logging
    )
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available - using fallback methods")

# ML imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .emotion_core import (
    EmotionCategory,
    LearningReadiness,
    EmotionalTrajectory,
    EmotionConstants
)

logger = logging.getLogger(__name__)


# ============================================================================
# ML-BASED EMOTION CLASSIFIER
# ============================================================================

class EmotionClassifier(nn.Module):
    """Neural network classifier for emotion detection with PAD model."""
    
    def __init__(self, hidden_size: int = 768, num_emotions: int = 13, dropout: float = 0.1):
        """
        Initialize emotion classifier.
        
        Args:
            hidden_size: Size of input features from transformer
            num_emotions: Number of emotion categories
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        
        # Feature projections for different transformer models
        self.bert_projection = nn.Linear(hidden_size, hidden_size)
        self.roberta_projection = nn.Linear(hidden_size, hidden_size)
        
        # Multi-head attention for model fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Main emotion classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        # PAD (Pleasure-Arousal-Dominance) dimensional model
        self.arousal_head = self._create_regressor(hidden_size)
        self.valence_head = self._create_regressor(hidden_size)
        self.dominance_head = self._create_regressor(hidden_size)
        
        # Learning state predictor
        self.learning_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 5)  # 5 learning states
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def _create_regressor(self, hidden_size: int) -> nn.Module:
        """Create a regressor head for dimensional emotion values."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        bert_features: Optional[torch.Tensor] = None,
        roberta_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the classifier.
        
        Args:
            bert_features: BERT embeddings [batch, hidden_size]
            roberta_features: RoBERTa embeddings [batch, hidden_size]
            
        Returns:
            Dictionary containing all predictions
        """
        features = []
        
        # Project features from different models
        if bert_features is not None:
            bert_proj = self.bert_projection(bert_features)
            features.append(bert_proj.unsqueeze(1))
        
        if roberta_features is not None:
            roberta_proj = self.roberta_projection(roberta_features)
            features.append(roberta_proj.unsqueeze(1))
        
        # Handle case with no features
        if not features:
            batch_size = 1
            features = [torch.zeros(batch_size, 1, self.hidden_size)]
        
        # Multi-model fusion with attention
        if len(features) > 1:
            combined = torch.cat(features, dim=1)
            fused, attn_weights = self.attention(combined, combined, combined)
            pooled = fused.mean(dim=1)
        else:
            pooled = features[0].squeeze(1)
            attn_weights = None
        
        # Apply normalization and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Generate all predictions
        return {
            'emotion_logits': self.classifier(pooled),
            'arousal': self.arousal_head(pooled),
            'valence': self.valence_head(pooled),
            'dominance': self.dominance_head(pooled),
            'learning_logits': self.learning_predictor(pooled),
            'confidence': self.confidence_head(pooled),
            'attention_weights': attn_weights,
            'features': pooled
        }


# ============================================================================
# ADAPTIVE THRESHOLD MANAGER
# ============================================================================

@dataclass
class UserThresholds:
    """Per-user adaptive thresholds for emotion detection."""
    
    confidence_threshold: float = 0.7
    intervention_threshold: float = 0.4
    bert_weight: float = 1.0
    roberta_weight: float = 1.0
    total_predictions: int = 0
    successful_predictions: int = 0
    learning_rate: float = 0.1
    last_updated: Optional[datetime] = None
    
    def update_learning_rate(self) -> None:
        """Decay learning rate over time for stability."""
        self.learning_rate = max(0.01, self.learning_rate * 0.99)
    
    def update_confidence_threshold(self, prediction_confidence: float) -> None:
        """Adapt confidence threshold based on prediction quality."""
        lr = self.learning_rate
        
        if prediction_confidence > 0.8:
            # High confidence - lower threshold
            target = prediction_confidence - 0.1
        elif prediction_confidence < 0.5:
            # Low confidence - raise threshold
            target = prediction_confidence + 0.2
        else:
            return
        
        self.confidence_threshold = (1 - lr) * self.confidence_threshold + lr * target
        self.confidence_threshold = max(0.3, min(0.9, self.confidence_threshold))
    
    def update_model_weight(self, model_type: str, confidence: float) -> None:
        """Update model-specific performance weights."""
        lr = self.learning_rate
        
        if model_type == 'bert':
            self.bert_weight = (1 - lr) * self.bert_weight + lr * confidence
            self.bert_weight = max(0.1, min(2.0, self.bert_weight))
        elif model_type == 'roberta':
            self.roberta_weight = (1 - lr) * self.roberta_weight + lr * confidence
            self.roberta_weight = max(0.1, min(2.0, self.roberta_weight))


class AdaptiveThresholdManager:
    """Manages per-user adaptive thresholds for emotion detection."""
    
    def __init__(self):
        self.thresholds: Dict[str, UserThresholds] = {}
        logger.info("Adaptive threshold manager initialized")
    
    def get_thresholds(self, user_id: Optional[str]) -> UserThresholds:
        """Get thresholds for a user, creating default if needed."""
        if not user_id:
            return UserThresholds()
        
        if user_id not in self.thresholds:
            self.thresholds[user_id] = UserThresholds(last_updated=datetime.utcnow())
        
        return self.thresholds[user_id]
    
    def update_thresholds(
        self,
        user_id: str,
        prediction_confidence: float,
        model_type: str
    ) -> None:
        """Update thresholds based on prediction results."""
        thresholds = self.get_thresholds(user_id)
        
        thresholds.total_predictions += 1
        thresholds.update_learning_rate()
        thresholds.update_confidence_threshold(prediction_confidence)
        thresholds.update_model_weight(model_type, prediction_confidence)
        thresholds.last_updated = datetime.utcnow()
        
        self.thresholds[user_id] = thresholds


# ============================================================================
# MAIN EMOTION TRANSFORMER
# ============================================================================

class EmotionTransformer:
    """
    Enterprise-grade emotion detection using transformer models.
    
    Features:
    - Multi-model support (BERT, RoBERTa)
    - Adaptive threshold learning
    - Ensemble prediction fusion
    - Graceful fallback mechanisms
    - Production-ready error handling
    """
    
    def __init__(self):
        """Initialize emotion transformer."""
        self.bert_model: Optional[Any] = None
        self.roberta_model: Optional[Any] = None
        self.bert_tokenizer: Optional[Any] = None
        self.roberta_tokenizer: Optional[Any] = None
        self.classifier: Optional[EmotionClassifier] = None
        self.threshold_manager = AdaptiveThresholdManager()
        self.is_initialized = False
        
        # Model configuration
        self.config = {
            'bert_model': 'bert-base-uncased',
            'roberta_model': 'roberta-base',
            'max_length': 512,
            'hidden_size': 768,
            'num_emotions': len(EmotionCategory),
            'batch_size': 16,
            'dropout': 0.1
        }
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'transformer_predictions': 0,
            'fallback_predictions': 0,
            'avg_confidence': 0.0,
            'initialization_time': 0.0
        }
        
        logger.info("EmotionTransformer initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize transformer models and classifier.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
        
        start_time = time.time()
        logger.info("Initializing transformer models...")
        
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers library not available - using fallback only")
                self.is_initialized = True
                return True
            
            # Initialize BERT
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(
                    self.config['bert_model'],
                    use_fast=True
                )
                self.bert_model = AutoModel.from_pretrained(self.config['bert_model'])
                self.bert_model.eval()
                logger.info("✓ BERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BERT: {e}")
                self.bert_model = None
            
            # Initialize RoBERTa
            try:
                self.roberta_tokenizer = AutoTokenizer.from_pretrained(
                    self.config['roberta_model'],
                    use_fast=True
                )
                self.roberta_model = AutoModel.from_pretrained(self.config['roberta_model'])
                self.roberta_model.eval()
                logger.info("✓ RoBERTa model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RoBERTa: {e}")
                self.roberta_model = None
            
            # Initialize classifier if at least one model loaded
            if self.bert_model or self.roberta_model:
                self.classifier = EmotionClassifier(
                    hidden_size=self.config['hidden_size'],
                    num_emotions=self.config['num_emotions'],
                    dropout=self.config['dropout']
                )
                self.classifier.eval()
                logger.info("✓ Emotion classifier initialized")
            
            init_time = time.time() - start_time
            self.stats['initialization_time'] = init_time
            self.is_initialized = True
            
            models_loaded = sum([
                self.bert_model is not None,
                self.roberta_model is not None
            ])
            logger.info(f"Initialization complete ({init_time:.2f}s, {models_loaded} models loaded)")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.is_initialized = True  # Allow fallback mode
            return False
    
    async def predict(
        self,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict emotion from input data.
        
        Args:
            input_data: Dictionary containing 'text_data' key
            user_id: Optional user ID for adaptive thresholds
            
        Returns:
            Dictionary with emotion predictions
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Extract and validate text
            text = self._extract_text(input_data)
            if not text:
                return self._get_neutral_prediction()
            
            # Get user thresholds
            thresholds = self.threshold_manager.get_thresholds(user_id)
            
            # Generate predictions from available models
            predictions = []
            
            if self.bert_model and self.bert_tokenizer:
                bert_pred = await self._predict_bert(text, thresholds)
                if bert_pred:
                    predictions.append(bert_pred)
            
            if self.roberta_model and self.roberta_tokenizer:
                roberta_pred = await self._predict_roberta(text, thresholds)
                if roberta_pred:
                    predictions.append(roberta_pred)
            
            # Fuse predictions or use fallback
            if predictions:
                result = self._fuse_predictions(predictions, thresholds)
                self.stats['transformer_predictions'] += 1
            else:
                result = await self._predict_fallback(text)
                self.stats['fallback_predictions'] += 1
            
            # Update adaptive thresholds
            if user_id and predictions:
                self.threshold_manager.update_thresholds(
                    user_id,
                    result['confidence'],
                    result['model_type']
                )
            
            # Add metadata
            prediction_time = (time.time() - start_time) * 1000
            result['metadata'] = {
                'prediction_time_ms': round(prediction_time, 2),
                'models_used': len(predictions),
                'adaptive_threshold': thresholds.confidence_threshold,
                'version': '1.0'
            }
            
            # Update statistics
            self.stats['total_predictions'] += 1
            self._update_stats(result['confidence'])
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return self._get_neutral_prediction(error=str(e))
    
    async def _predict_bert(
        self,
        text: str,
        thresholds: UserThresholds
    ) -> Optional[Dict[str, Any]]:
        """Generate prediction using BERT model."""
        try:
            # Tokenize
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config['max_length'],
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Classify
            if self.classifier:
                outputs = self.classifier(bert_features=embeddings)
                return self._process_classifier_output(
                    outputs,
                    thresholds,
                    model_type='bert'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"BERT prediction failed: {e}")
            return None
    
    async def _predict_roberta(
        self,
        text: str,
        thresholds: UserThresholds
    ) -> Optional[Dict[str, Any]]:
        """Generate prediction using RoBERTa model."""
        try:
            # Tokenize
            inputs = self.roberta_tokenizer(
                text,
                return_tensors="pt",
                max_length=self.config['max_length'],
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Classify
            if self.classifier:
                outputs = self.classifier(roberta_features=embeddings)
                return self._process_classifier_output(
                    outputs,
                    thresholds,
                    model_type='roberta'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"RoBERTa prediction failed: {e}")
            return None
    
    def _process_classifier_output(
        self,
        outputs: Dict[str, torch.Tensor],
        thresholds: UserThresholds,
        model_type: str
    ) -> Dict[str, Any]:
        """Process classifier output into prediction dictionary."""
        # Get emotion probabilities
        emotion_probs = F.softmax(outputs['emotion_logits'], dim=-1)
        emotion_categories = list(EmotionCategory)
        
        # Build emotion distribution
        emotion_dist = {
            emotion_categories[i].value: float(emotion_probs[0][i])
            for i in range(len(emotion_categories))
        }
        
        # Get primary emotion and confidence
        primary_emotion = max(emotion_dist.keys(), key=lambda k: emotion_dist[k])
        confidence = emotion_dist[primary_emotion]
        
        # Apply threshold
        if confidence < thresholds.confidence_threshold:
            primary_emotion = EmotionCategory.NEUTRAL.value
            confidence = emotion_dist[primary_emotion]
        
        # Get learning state
        learning_probs = F.softmax(outputs['learning_logits'], dim=-1)
        learning_states = list(LearningReadiness)
        learning_idx = torch.argmax(learning_probs, dim=-1).item()
        learning_state = learning_states[learning_idx].value
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_distribution': emotion_dist,
            'confidence': float(confidence),
            'arousal': float(outputs['arousal'][0].item()),
            'valence': float(outputs['valence'][0].item()),
            'dominance': float(outputs['dominance'][0].item()),
            'learning_state': learning_state,
            'model_type': model_type
        }
    
    def _fuse_predictions(
        self,
        predictions: List[Dict[str, Any]],
        thresholds: UserThresholds
    ) -> Dict[str, Any]:
        """Fuse multiple model predictions with adaptive weighting."""
        if len(predictions) == 1:
            return predictions[0]
        
        # Calculate weights based on confidence and model performance
        weights = []
        for pred in predictions:
            confidence = pred['confidence']
            model_type = pred['model_type']
            
            if model_type == 'bert':
                weight = confidence * thresholds.bert_weight
            elif model_type == 'roberta':
                weight = confidence * thresholds.roberta_weight
            else:
                weight = confidence
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        # Fuse emotion distributions
        fused_dist = {}
        for emotion in EmotionCategory:
            emotion_value = emotion.value
            score = sum(
                pred['emotion_distribution'].get(emotion_value, 0.0) * weights[i]
                for i, pred in enumerate(predictions)
            )
            fused_dist[emotion_value] = score
        
        # Normalize distribution
        total = sum(fused_dist.values())
        if total > 0:
            fused_dist = {k: v / total for k, v in fused_dist.items()}
        
        # Get primary emotion
        primary_emotion = max(fused_dist.keys(), key=lambda k: fused_dist[k])
        
        # Fuse other metrics
        result = {
            'primary_emotion': primary_emotion,
            'emotion_distribution': fused_dist,
            'confidence': sum(pred['confidence'] * weights[i] for i, pred in enumerate(predictions)),
            'arousal': sum(pred['arousal'] * weights[i] for i, pred in enumerate(predictions)),
            'valence': sum(pred['valence'] * weights[i] for i, pred in enumerate(predictions)),
            'dominance': sum(pred['dominance'] * weights[i] for i, pred in enumerate(predictions)),
            'learning_state': max(predictions, key=lambda p: p['confidence'])['learning_state'],
            'model_type': 'ensemble',
            'ensemble_weights': dict(zip([p['model_type'] for p in predictions], weights))
        }
        
        return result
    
    async def _predict_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback prediction using pattern matching."""
        text_lower = text.lower()
        
        # Initialize scores
        emotion_scores = {emotion.value: 0.0 for emotion in EmotionCategory}
        
        # Pattern-based scoring
        patterns = {
            EmotionCategory.JOY.value: ['happy', 'excited', 'joy', 'great', 'awesome', 'love', 'amazing', 'wonderful'],
            EmotionCategory.SATISFACTION.value: ['satisfied', 'good', 'nice', 'pleased', 'content'],
            EmotionCategory.FRUSTRATION.value: ['frustrated', 'annoyed', 'irritated', 'difficult', 'hard'],
            EmotionCategory.CONFUSION.value: ['confused', 'unclear', 'lost', "don't understand", "don't get", 'puzzled'],
            EmotionCategory.BREAKTHROUGH_MOMENT.value: ['understand', 'got it', 'makes sense', 'clear now', 'aha', 'finally'],
            EmotionCategory.SADNESS.value: ['sad', 'unhappy', 'disappointed', 'down'],
            EmotionCategory.ANXIETY.value: ['worried', 'anxious', 'nervous', 'scared', 'afraid'],
        }
        
        # Score based on patterns
        for emotion, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 0.2
        
        # Normalize
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
        else:
            emotion_scores[EmotionCategory.NEUTRAL.value] = 1.0
        
        # Get primary emotion
        primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_distribution': emotion_scores,
            'confidence': emotion_scores[primary_emotion],
            'arousal': 0.5,
            'valence': 0.5,
            'dominance': 0.5,
            'learning_state': LearningReadiness.MODERATE_READINESS.value,
            'model_type': 'fallback'
        }
    
    def _extract_text(self, input_data: Dict[str, Any]) -> str:
        """Extract text from input data."""
        text_data = input_data.get('text_data', '')
        
        if isinstance(text_data, dict):
            text_data = text_data.get('content', '') or str(text_data)
        
        return str(text_data).strip()
    
    def _get_neutral_prediction(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Get neutral prediction for error cases."""
        result = {
            'primary_emotion': EmotionCategory.NEUTRAL.value,
            'emotion_distribution': {EmotionCategory.NEUTRAL.value: 1.0},
            'confidence': 0.5,
            'arousal': 0.5,
            'valence': 0.5,
            'dominance': 0.5,
            'learning_state': LearningReadiness.MODERATE_READINESS.value,
            'model_type': 'neutral_fallback'
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def _update_stats(self, confidence: float) -> None:
        """Update running statistics."""
        n = self.stats['total_predictions']
        if n > 0:
            current_avg = self.stats['avg_confidence']
            self.stats['avg_confidence'] = (current_avg * (n - 1) + confidence) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        models_available = []
        if self.bert_model:
            models_available.append('BERT')
        if self.roberta_model:
            models_available.append('RoBERTa')
        
        return {
            **self.stats,
            'models_available': models_available,
            'total_users': len(self.threshold_manager.thresholds),
            'is_initialized': self.is_initialized
        }


__all__ = ['EmotionTransformer', 'EmotionClassifier', 'AdaptiveThresholdManager']