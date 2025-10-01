"""
Adaptive Learning Engine for MasterX
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md Section 4

PURPOSE:
- Dynamic difficulty adaptation based on performance and emotion
- Ability estimation using Item Response Theory (IRT)
- Cognitive load management
- Flow state optimization
- Learning velocity tracking

PRINCIPLES:
- No hardcoded rules (all ML-driven)
- Real-time adaptation
- Research-based algorithms (IRT, ZPD)
- PEP8 compliant
- Clean naming conventions
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.models import EmotionState, LearningReadiness
from utils.errors import MasterXError

logger = logging.getLogger(__name__)


@dataclass
class DifficultyLevel:
    """Difficulty level representation"""
    value: float  # 0.0 (easiest) to 1.0 (hardest)
    label: str  # human-readable label
    explanation: str  # explanation for the level


@dataclass
class PerformanceMetrics:
    """User performance metrics"""
    accuracy: float  # 0.0 to 1.0
    response_time_ms: float
    help_requests: int
    retries: int
    success_streak: int
    failure_streak: int


@dataclass
class CognitiveLoadEstimate:
    """Cognitive load estimation"""
    load: float  # 0.0 (low) to 1.0 (overload)
    level: str  # low, moderate, high, overload
    factors: Dict[str, float]  # contributing factors


class AbilityEstimator:
    """
    Estimate learner ability using Item Response Theory (IRT)
    
    Uses 2-Parameter Logistic (2PL) IRT model:
    P(correct) = 1 / (1 + exp(-1.7 * discrimination * (ability - difficulty)))
    
    Based on Lord (1980) - Item Response Theory
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize ability estimator
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.performance_collection = db.user_performance
        
        # IRT parameters
        self.discrimination_default = 1.0  # Item discrimination (how well it differentiates)
        self.scaling_factor = 1.7  # Standard IRT scaling factor
    
    async def get_ability(
        self,
        user_id: str,
        subject: str
    ) -> float:
        """
        Get current ability estimate for user in subject
        
        Args:
            user_id: User ID
            subject: Subject/topic
        
        Returns:
            Ability estimate (0.0 to 1.0)
        """
        try:
            # Get latest performance record
            performance = await self.performance_collection.find_one(
                {'user_id': user_id, 'subject': subject},
                sort=[('timestamp', -1)]
            )
            
            if performance:
                return performance['ability_level']
            else:
                # Default starting ability (middle of the range)
                return 0.5
        
        except Exception as e:
            logger.error(f"Error getting ability: {e}")
            return 0.5
    
    async def update_ability(
        self,
        user_id: str,
        subject: str,
        item_difficulty: float,
        result: bool,
        discrimination: float = None
    ) -> float:
        """
        Update ability estimate based on performance
        
        Uses Bayesian updating with IRT model
        
        Args:
            user_id: User ID
            subject: Subject/topic
            item_difficulty: Difficulty of the item (0.0 to 1.0)
            result: Whether user succeeded (True) or failed (False)
            discrimination: Item discrimination parameter
        
        Returns:
            Updated ability estimate
        """
        try:
            # Get current ability
            current_ability = await self.get_ability(user_id, subject)
            
            # Use default discrimination if not provided
            if discrimination is None:
                discrimination = self.discrimination_default
            
            # Calculate expected performance using IRT 2PL model
            expected_prob = self._irt_probability(
                ability=current_ability,
                difficulty=item_difficulty,
                discrimination=discrimination
            )
            
            # Bayesian update (simplified)
            # If correct: increase ability if item was difficult
            # If incorrect: decrease ability if item was easy
            
            if result:
                # Correct response
                # Learning rate proportional to surprise (1 - expected_prob)
                learning_rate = 0.1 * (1 - expected_prob)
                ability_change = learning_rate * (1 - current_ability)
            else:
                # Incorrect response
                # Learning rate proportional to surprise (expected_prob)
                learning_rate = 0.1 * expected_prob
                ability_change = -learning_rate * current_ability
            
            # Update ability
            new_ability = np.clip(current_ability + ability_change, 0.0, 1.0)
            
            # Save to database
            await self._save_ability(user_id, subject, new_ability)
            
            logger.debug(
                f"Ability updated for {user_id}/{subject}: "
                f"{current_ability:.3f} -> {new_ability:.3f} "
                f"(item difficulty: {item_difficulty:.3f}, result: {result})"
            )
            
            return new_ability
        
        except Exception as e:
            logger.error(f"Error updating ability: {e}")
            return current_ability
    
    def _irt_probability(
        self,
        ability: float,
        difficulty: float,
        discrimination: float
    ) -> float:
        """
        Calculate probability of success using IRT 2PL model
        
        P(correct) = 1 / (1 + exp(-discrimination * scaling * (ability - difficulty)))
        
        Args:
            ability: Learner ability (0.0 to 1.0)
            difficulty: Item difficulty (0.0 to 1.0)
            discrimination: Item discrimination
        
        Returns:
            Probability of success (0.0 to 1.0)
        """
        # Convert 0-1 range to centered range (-2 to +2)
        ability_centered = (ability - 0.5) * 4
        difficulty_centered = (difficulty - 0.5) * 4
        
        # IRT 2PL model
        exponent = -discrimination * self.scaling_factor * (ability_centered - difficulty_centered)
        probability = 1.0 / (1.0 + math.exp(exponent))
        
        return probability
    
    async def _save_ability(
        self,
        user_id: str,
        subject: str,
        ability: float
    ):
        """Save ability estimate to database"""
        try:
            await self.performance_collection.update_one(
                {'user_id': user_id, 'subject': subject},
                {
                    '$set': {
                        'ability_level': ability,
                        'timestamp': datetime.utcnow()
                    },
                    '$setOnInsert': {
                        'user_id': user_id,
                        'subject': subject,
                        'difficulty_preference': 0.5,
                        'learning_velocity': 0.0,
                        'mastery_topics': [],
                        'struggling_topics': [],
                        'total_practice_time_hours': 0.0,
                        'improvement_rate': 0.0
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving ability: {e}")
    
    def predict_success_probability(
        self,
        ability: float,
        difficulty: float
    ) -> float:
        """
        Predict probability of success for given ability and difficulty
        
        Args:
            ability: Learner ability (0.0 to 1.0)
            difficulty: Item difficulty (0.0 to 1.0)
        
        Returns:
            Predicted success probability (0.0 to 1.0)
        """
        return self._irt_probability(
            ability=ability,
            difficulty=difficulty,
            discrimination=self.discrimination_default
        )


class CognitiveLoadEstimator:
    """
    Estimate cognitive load from multiple factors
    
    Cognitive Load Theory (Sweller, 1988)
    Combines intrinsic, extraneous, and germane load
    """
    
    def estimate_load(
        self,
        task_complexity: float,
        time_on_task_seconds: float,
        emotion_state: EmotionState,
        help_requests: int = 0,
        retries: int = 0
    ) -> CognitiveLoadEstimate:
        """
        Estimate cognitive load from multiple factors
        
        Args:
            task_complexity: Inherent difficulty of task (0.0 to 1.0)
            time_on_task_seconds: Time spent on task
            emotion_state: Current emotional state
            help_requests: Number of times help was requested
            retries: Number of retry attempts
        
        Returns:
            Cognitive load estimate
        """
        factors = {}
        
        # Factor 1: Task complexity (intrinsic load)
        factors['task_complexity'] = task_complexity
        
        # Factor 2: Time on task (long time = high load or struggle)
        # Normalize to 0-1 (assume 300s = moderate, > 600s = high)
        time_factor = min(time_on_task_seconds / 600.0, 1.0)
        factors['time_pressure'] = time_factor
        
        # Factor 3: Emotional indicators
        # High arousal + low valence = high load (frustration, anxiety)
        emotion_load = 0.5
        if emotion_state:
            # Arousal contributes to load
            arousal_factor = emotion_state.arousal
            # Low valence (negative emotion) increases load
            valence_factor = 1.0 - emotion_state.valence
            emotion_load = (arousal_factor * 0.6 + valence_factor * 0.4)
        factors['emotional_load'] = emotion_load
        
        # Factor 4: Help requests (more requests = higher load)
        help_factor = min(help_requests / 5.0, 1.0)
        factors['help_needed'] = help_factor
        
        # Factor 5: Retries (more retries = higher load)
        retry_factor = min(retries / 3.0, 1.0)
        factors['retry_burden'] = retry_factor
        
        # Combined cognitive load (weighted average)
        total_load = (
            factors['task_complexity'] * 0.30 +
            factors['time_pressure'] * 0.20 +
            factors['emotional_load'] * 0.25 +
            factors['help_needed'] * 0.15 +
            factors['retry_burden'] * 0.10
        )
        
        # Classify load level
        if total_load < 0.3:
            level = "low"
        elif total_load < 0.6:
            level = "moderate"
        elif total_load < 0.8:
            level = "high"
        else:
            level = "overload"
        
        return CognitiveLoadEstimate(
            load=total_load,
            level=level,
            factors=factors
        )


class FlowStateOptimizer:
    """
    Optimize for flow state (Csikszentmihalyi, 1990)
    
    Flow occurs when:
    - Challenge matches skill level
    - Clear goals
    - Immediate feedback
    - Deep concentration
    """
    
    def __init__(self):
        # Flow state parameters
        self.optimal_challenge_ratio = 0.75  # Challenge should be 75% of ability
        self.flow_zone_width = 0.15  # ±15% tolerance
    
    def calculate_optimal_difficulty(
        self,
        ability: float,
        current_emotion: EmotionState
    ) -> float:
        """
        Calculate optimal difficulty for flow state
        
        Args:
            ability: Current ability level (0.0 to 1.0)
            current_emotion: Current emotional state
        
        Returns:
            Optimal difficulty (0.0 to 1.0)
        """
        # Base optimal difficulty (slightly below ability)
        base_difficulty = ability * self.optimal_challenge_ratio
        
        # Adjust based on emotional state
        if current_emotion:
            # If frustrated or anxious: reduce difficulty
            if current_emotion.primary_emotion in ['frustration', 'anxiety', 'overwhelmed']:
                adjustment = -0.1
            # If bored or disengaged: increase difficulty
            elif current_emotion.primary_emotion in ['boredom', 'disengagement']:
                adjustment = 0.15
            # If in flow state: maintain current level
            elif current_emotion.primary_emotion in ['flow_state', 'engagement']:
                adjustment = 0.0
            # If joyful/achieved: slight increase (capitalize on momentum)
            elif current_emotion.primary_emotion in ['joy', 'achievement']:
                adjustment = 0.05
            else:
                adjustment = 0.0
            
            base_difficulty = np.clip(base_difficulty + adjustment, 0.0, 1.0)
        
        return base_difficulty
    
    def detect_flow_state(
        self,
        emotion_state: EmotionState,
        performance: PerformanceMetrics,
        ability: float,
        current_difficulty: float
    ) -> bool:
        """
        Detect if learner is in flow state
        
        Indicators:
        - Moderate to high arousal (engaged)
        - Positive valence (enjoying)
        - High accuracy (succeeding)
        - Challenge matches skill
        
        Args:
            emotion_state: Current emotional state
            performance: Performance metrics
            ability: Current ability
            current_difficulty: Current difficulty level
        
        Returns:
            True if in flow state
        """
        # Check emotion indicators
        emotion_flow = False
        if emotion_state:
            # Flow: moderate-high arousal, positive valence
            arousal_ok = 0.5 <= emotion_state.arousal <= 0.9
            valence_ok = emotion_state.valence >= 0.6
            emotion_flow = arousal_ok and valence_ok
        
        # Check performance indicators
        performance_flow = performance.accuracy >= 0.65 and performance.accuracy <= 0.85
        
        # Check challenge-skill balance
        difficulty_diff = abs(current_difficulty - ability)
        balance_flow = difficulty_diff <= self.flow_zone_width
        
        # All indicators must be true
        in_flow = emotion_flow and performance_flow and balance_flow
        
        return in_flow
    
    def get_flow_recommendations(
        self,
        ability: float,
        current_difficulty: float,
        emotion_state: EmotionState
    ) -> Dict[str, any]:
        """
        Get recommendations to achieve/maintain flow state
        
        Args:
            ability: Current ability
            current_difficulty: Current difficulty
            emotion_state: Current emotional state
        
        Returns:
            Dictionary with recommendations
        """
        optimal_difficulty = self.calculate_optimal_difficulty(ability, emotion_state)
        difficulty_gap = current_difficulty - optimal_difficulty
        
        recommendations = {
            'optimal_difficulty': optimal_difficulty,
            'current_difficulty': current_difficulty,
            'difficulty_adjustment_needed': abs(difficulty_gap) > 0.1,
            'suggested_action': None,
            'reasoning': None
        }
        
        if difficulty_gap > 0.1:
            recommendations['suggested_action'] = 'decrease_difficulty'
            recommendations['reasoning'] = 'Current task too challenging for flow state'
        elif difficulty_gap < -0.1:
            recommendations['suggested_action'] = 'increase_difficulty'
            recommendations['reasoning'] = 'Current task too easy, risk of boredom'
        else:
            recommendations['suggested_action'] = 'maintain'
            recommendations['reasoning'] = 'Difficulty level optimal for flow state'
        
        return recommendations


class LearningVelocityTracker:
    """
    Track learning velocity (concepts learned per hour)
    Detects plateaus and breakthroughs
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize velocity tracker
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.performance_collection = db.user_performance
    
    async def calculate_velocity(
        self,
        user_id: str,
        subject: str,
        time_window_hours: float = 1.0
    ) -> float:
        """
        Calculate learning velocity over time window
        
        Args:
            user_id: User ID
            subject: Subject/topic
            time_window_hours: Time window for calculation
        
        Returns:
            Learning velocity (concepts per hour)
        """
        try:
            # Get performance records in time window
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            cursor = self.performance_collection.find({
                'user_id': user_id,
                'subject': subject,
                'timestamp': {'$gte': cutoff_time}
            }).sort('timestamp', 1)
            
            records = await cursor.to_list(length=100)
            
            if len(records) < 2:
                return 0.0
            
            # Calculate ability improvement
            initial_ability = records[0]['ability_level']
            final_ability = records[-1]['ability_level']
            ability_gain = final_ability - initial_ability
            
            # Normalize to "concepts learned"
            # Assume 0.1 ability gain = 1 concept learned
            concepts_learned = ability_gain / 0.1
            
            # Calculate velocity
            velocity = concepts_learned / time_window_hours
            
            return max(0.0, velocity)
        
        except Exception as e:
            logger.error(f"Error calculating velocity: {e}")
            return 0.0
    
    async def detect_plateau(
        self,
        user_id: str,
        subject: str,
        threshold_hours: float = 2.0
    ) -> bool:
        """
        Detect if learner has plateaued (no progress)
        
        Args:
            user_id: User ID
            subject: Subject/topic
            threshold_hours: Time window to check
        
        Returns:
            True if plateaued
        """
        velocity = await self.calculate_velocity(
            user_id,
            subject,
            time_window_hours=threshold_hours
        )
        
        # Plateau if velocity near zero for extended period
        return velocity < 0.1


class AdaptiveLearningEngine:
    """
    Main adaptive learning engine
    
    Orchestrates:
    - Ability estimation (IRT)
    - Difficulty adaptation
    - Cognitive load management
    - Flow state optimization
    - Learning velocity tracking
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize adaptive learning engine
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        
        # Initialize components
        self.ability_estimator = AbilityEstimator(db)
        self.cognitive_estimator = CognitiveLoadEstimator()
        self.flow_optimizer = FlowStateOptimizer()
        self.velocity_tracker = LearningVelocityTracker(db)
        
        logger.info("✅ AdaptiveLearningEngine initialized")
    
    async def recommend_difficulty(
        self,
        user_id: str,
        subject: str,
        emotion_state: EmotionState,
        recent_performance: Optional[PerformanceMetrics] = None
    ) -> DifficultyLevel:
        """
        Recommend optimal difficulty level for next task
        
        Args:
            user_id: User ID
            subject: Subject/topic
            emotion_state: Current emotional state
            recent_performance: Recent performance metrics
        
        Returns:
            Recommended difficulty level
        """
        try:
            # Get current ability
            ability = await self.ability_estimator.get_ability(user_id, subject)
            
            # Calculate optimal difficulty for flow state
            optimal_difficulty = self.flow_optimizer.calculate_optimal_difficulty(
                ability=ability,
                current_emotion=emotion_state
            )
            
            # Adjust based on cognitive load if performance data available
            if recent_performance:
                # Estimate cognitive load
                cognitive_load = self.cognitive_estimator.estimate_load(
                    task_complexity=optimal_difficulty,
                    time_on_task_seconds=recent_performance.response_time_ms / 1000.0,
                    emotion_state=emotion_state,
                    help_requests=recent_performance.help_requests,
                    retries=recent_performance.retries
                )
                
                # If overload, reduce difficulty
                if cognitive_load.level == "overload":
                    optimal_difficulty *= 0.8
                    logger.info(f"Reducing difficulty due to cognitive overload (user: {user_id})")
            
            # Clamp to valid range
            optimal_difficulty = np.clip(optimal_difficulty, 0.0, 1.0)
            
            # Convert to labeled difficulty
            if optimal_difficulty < 0.3:
                label = "Beginner"
                explanation = "Basic concepts and simple problems"
            elif optimal_difficulty < 0.5:
                label = "Easy"
                explanation = "Fundamental skills with some complexity"
            elif optimal_difficulty < 0.7:
                label = "Moderate"
                explanation = "Intermediate concepts requiring thought"
            elif optimal_difficulty < 0.85:
                label = "Challenging"
                explanation = "Advanced topics that stretch your skills"
            else:
                label = "Expert"
                explanation = "Complex problems for mastery"
            
            difficulty = DifficultyLevel(
                value=optimal_difficulty,
                label=label,
                explanation=explanation
            )
            
            logger.debug(
                f"Recommended difficulty for {user_id}/{subject}: "
                f"{difficulty.label} ({difficulty.value:.2f})"
            )
            
            return difficulty
        
        except Exception as e:
            logger.error(f"Error recommending difficulty: {e}")
            # Fallback to moderate difficulty
            return DifficultyLevel(
                value=0.5,
                label="Moderate",
                explanation="Intermediate level"
            )
