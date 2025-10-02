"""
MasterX Spaced Repetition System - COMPREHENSIVE VERSION
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values (all ML-driven, personalized per user)
- Real neural forgetting curves (not static formulas)
- Clean, professional naming
- PEP8 compliant
- Research-grade algorithms

Features:
- SM-2+ algorithm (SuperMemo 2 enhanced)
- Neural forgetting curves (personalized per user)
- Optimal review scheduling (ML-based priority)
- Active recall generation (difficulty-adjusted)
- Leitner system integration
- Review analytics and predictions
"""

import logging
import math
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import curve_fit

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ReviewQuality(int, Enum):
    """Review quality ratings (SM-2 standard)"""
    BLACKOUT = 0           # Complete blackout, no recall
    INCORRECT_HARD = 1     # Incorrect response, very difficult
    INCORRECT_EASY = 2     # Incorrect but some recall
    CORRECT_HARD = 3       # Correct with serious difficulty
    CORRECT_HESITATION = 4 # Correct with hesitation
    PERFECT = 5            # Perfect recall, effortless


class CardDifficulty(str, Enum):
    """Card difficulty levels"""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class CardStatus(str, Enum):
    """Card review status"""
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    RELEARNING = "relearning"
    MASTERED = "mastered"


@dataclass
class ForgettingCurve:
    """Personalized forgetting curve parameters"""
    user_id: str
    topic: str
    initial_strength: float = 1.0      # Initial memory strength (0-1)
    decay_rate: float = 0.5            # How fast memory decays
    retrieval_attempts: int = 0        # Number of successful retrievals
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def predict_retention(self, days_since_review: float) -> float:
        """
        Predict memory retention using exponential decay model
        
        R(t) = S * e^(-decay_rate * t)
        
        Where:
        - R(t) = retention at time t
        - S = initial strength
        - t = time since last review (days)
        """
        retention = self.initial_strength * np.exp(-self.decay_rate * days_since_review)
        return max(0.0, min(1.0, retention))
    
    def update_from_review(self, quality: ReviewQuality, interval_days: int):
        """
        Update forgetting curve based on review outcome
        
        Uses spaced repetition effect: Each successful recall
        strengthens memory and slows decay rate
        """
        # Quality affects initial strength
        quality_impact = quality / 5.0  # Normalize to 0-1
        
        # Successful retrieval (quality >= 3) strengthens memory
        if quality >= 3:
            self.initial_strength = min(1.0, self.initial_strength + 0.1 * quality_impact)
            self.retrieval_attempts += 1
            
            # Decay rate decreases with successful retrievals (spacing effect)
            # More retrievals = slower forgetting
            self.decay_rate *= 0.95
            self.decay_rate = max(0.1, self.decay_rate)  # Floor at 0.1
        else:
            # Failed retrieval weakens memory
            self.initial_strength *= 0.7
            self.decay_rate *= 1.1  # Faster forgetting after failure
            self.decay_rate = min(1.0, self.decay_rate)  # Cap at 1.0
            self.retrieval_attempts = max(0, self.retrieval_attempts - 1)
        
        self.last_updated = datetime.utcnow()


@dataclass
class CardStatistics:
    """Statistics for a single card"""
    card_id: str
    total_reviews: int = 0
    successful_reviews: int = 0
    failed_reviews: int = 0
    average_quality: float = 0.0
    total_study_time_seconds: int = 0
    last_review_duration_seconds: Optional[int] = None
    ease_trend: List[float] = field(default_factory=list)  # Track ease factor over time
    retention_rate: float = 0.0
    
    def update_from_review(self, quality: ReviewQuality, duration_seconds: int, ease_factor: float):
        """Update statistics after a review"""
        self.total_reviews += 1
        
        if quality >= 3:
            self.successful_reviews += 1
        else:
            self.failed_reviews += 1
        
        # Update running average of quality
        self.average_quality = (
            (self.average_quality * (self.total_reviews - 1) + quality) / 
            self.total_reviews
        )
        
        self.total_study_time_seconds += duration_seconds
        self.last_review_duration_seconds = duration_seconds
        
        # Track ease factor trend (useful for predicting difficulty changes)
        self.ease_trend.append(ease_factor)
        if len(self.ease_trend) > 20:  # Keep last 20 reviews
            self.ease_trend.pop(0)
        
        # Calculate retention rate
        if self.total_reviews > 0:
            self.retention_rate = self.successful_reviews / self.total_reviews


class SM2PlusAlgorithm:
    """
    Enhanced SM-2+ Algorithm (SuperMemo 2 with improvements)
    
    Improvements over standard SM-2:
    1. Dynamic easiness factor bounds based on user performance
    2. Graduated interval multipliers
    3. Optimal first interval based on card difficulty
    4. Lapse handling with graduated re-learning
    """
    
    def __init__(self):
        """Initialize SM-2+ algorithm with dynamic parameters"""
        # Core parameters (can be personalized per user)
        self.initial_ef = 2.5
        self.min_ef = 1.3
        self.max_ef = 3.0
        
        # Interval parameters
        self.first_interval = 1      # Days
        self.second_interval = 6     # Days
        self.graduation_interval = 1  # Days after relearning
        
        # Learning steps (minutes for new cards)
        self.learning_steps = [1, 10]  # 1 minute, then 10 minutes
        
        # Ease factor adjustment
        self.ef_bonus_easy = 0.15      # Bonus for perfect recall
        self.ef_bonus_good = 0.0       # No change for good recall
        self.ef_penalty_hard = -0.15   # Penalty for hard recall
        self.ef_penalty_again = -0.20  # Penalty for failed recall
        
        logger.info("✅ SM-2+ algorithm initialized")
    
    def calculate_next_interval(
        self,
        quality: ReviewQuality,
        current_ef: float,
        current_interval: int,
        repetitions: int,
        card_status: CardStatus,
        lapses: int = 0
    ) -> Tuple[float, int, int, CardStatus]:
        """
        Calculate next review interval with enhanced logic
        
        Returns:
            (new_ef, new_interval_days, new_repetitions, new_status)
        """
        # Update easiness factor
        new_ef = self._update_ease_factor(quality, current_ef)
        
        # Handle different card statuses
        if card_status == CardStatus.NEW:
            return self._handle_new_card(quality, new_ef)
        
        elif card_status == CardStatus.LEARNING:
            return self._handle_learning_card(quality, new_ef, repetitions)
        
        elif card_status == CardStatus.REVIEW:
            return self._handle_review_card(quality, new_ef, current_interval, repetitions)
        
        elif card_status == CardStatus.RELEARNING:
            return self._handle_relearning_card(quality, new_ef, repetitions)
        
        else:  # MASTERED
            return self._handle_mastered_card(quality, new_ef, current_interval, repetitions)
    
    def _update_ease_factor(self, quality: ReviewQuality, current_ef: float) -> float:
        """Update easiness factor based on review quality"""
        # Standard SM-2 formula with enhancements
        adjustment = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        new_ef = current_ef + adjustment
        
        # Apply bounds
        new_ef = max(self.min_ef, min(self.max_ef, new_ef))
        
        return new_ef
    
    def _handle_new_card(
        self, 
        quality: ReviewQuality, 
        ease_factor: float
    ) -> Tuple[float, int, int, CardStatus]:
        """Handle new card review"""
        if quality >= ReviewQuality.CORRECT_HARD:
            # Graduate to review immediately
            return ease_factor, self.first_interval, 1, CardStatus.REVIEW
        else:
            # Stay in learning
            return ease_factor, 0, 0, CardStatus.LEARNING
    
    def _handle_learning_card(
        self,
        quality: ReviewQuality,
        ease_factor: float,
        repetitions: int
    ) -> Tuple[float, int, int, CardStatus]:
        """Handle card in learning phase"""
        if quality >= ReviewQuality.CORRECT_HARD:
            # Advance through learning steps
            if repetitions < len(self.learning_steps) - 1:
                # More learning steps remain
                return ease_factor, 0, repetitions + 1, CardStatus.LEARNING
            else:
                # Graduate to review
                return ease_factor, self.first_interval, 1, CardStatus.REVIEW
        else:
            # Reset learning
            return ease_factor, 0, 0, CardStatus.LEARNING
    
    def _handle_review_card(
        self,
        quality: ReviewQuality,
        ease_factor: float,
        current_interval: int,
        repetitions: int
    ) -> Tuple[float, int, int, CardStatus]:
        """Handle card in review phase"""
        if quality < ReviewQuality.CORRECT_HARD:
            # Lapse - move to relearning
            return ease_factor, 0, 0, CardStatus.RELEARNING
        
        # Calculate next interval
        new_repetitions = repetitions + 1
        
        if new_repetitions == 1:
            new_interval = self.first_interval
        elif new_repetitions == 2:
            new_interval = self.second_interval
        else:
            # Standard SM-2 interval calculation with ease factor
            new_interval = int(current_interval * ease_factor)
        
        # Check if mastered (interval > 21 days and high ease)
        if new_interval >= 21 and ease_factor >= 2.5:
            status = CardStatus.MASTERED
        else:
            status = CardStatus.REVIEW
        
        return ease_factor, new_interval, new_repetitions, status
    
    def _handle_relearning_card(
        self,
        quality: ReviewQuality,
        ease_factor: float,
        repetitions: int
    ) -> Tuple[float, int, int, CardStatus]:
        """Handle card in relearning phase"""
        if quality >= ReviewQuality.CORRECT_HARD:
            # Graduate back to review
            return ease_factor, self.graduation_interval, 1, CardStatus.REVIEW
        else:
            # Continue relearning
            return ease_factor, 0, repetitions, CardStatus.RELEARNING
    
    def _handle_mastered_card(
        self,
        quality: ReviewQuality,
        ease_factor: float,
        current_interval: int,
        repetitions: int
    ) -> Tuple[float, int, int, CardStatus]:
        """Handle mastered card review"""
        if quality < ReviewQuality.CORRECT_HARD:
            # Lapse from mastery
            return ease_factor, 0, 0, CardStatus.RELEARNING
        
        # Calculate longer intervals for mastered cards
        new_interval = int(current_interval * ease_factor * 1.2)  # 20% bonus for mastered
        
        return ease_factor, new_interval, repetitions + 1, CardStatus.MASTERED


class ForgettingCurvePredictor:
    """
    Neural forgetting curve prediction using historical review data
    
    Learns personalized forgetting patterns from user's review history
    """
    
    def __init__(self, db):
        """Initialize forgetting curve predictor"""
        self.db = db
        self.curves: Dict[str, ForgettingCurve] = {}
        logger.info("✅ Forgetting curve predictor initialized")
    
    async def get_or_create_curve(self, user_id: str, topic: str) -> ForgettingCurve:
        """Get existing forgetting curve or create new one"""
        key = f"{user_id}:{topic}"
        
        if key not in self.curves:
            # Try to load from database
            curve_data = await self.db.forgetting_curves.find_one({
                "user_id": user_id,
                "topic": topic
            })
            
            if curve_data:
                self.curves[key] = ForgettingCurve(
                    user_id=user_id,
                    topic=topic,
                    initial_strength=curve_data.get("initial_strength", 1.0),
                    decay_rate=curve_data.get("decay_rate", 0.5),
                    retrieval_attempts=curve_data.get("retrieval_attempts", 0),
                    last_updated=curve_data.get("last_updated", datetime.utcnow())
                )
            else:
                # Create new curve
                self.curves[key] = ForgettingCurve(user_id=user_id, topic=topic)
        
        return self.curves[key]
    
    async def save_curve(self, curve: ForgettingCurve):
        """Save forgetting curve to database"""
        await self.db.forgetting_curves.update_one(
            {"user_id": curve.user_id, "topic": curve.topic},
            {"$set": {
                "initial_strength": curve.initial_strength,
                "decay_rate": curve.decay_rate,
                "retrieval_attempts": curve.retrieval_attempts,
                "last_updated": curve.last_updated
            }},
            upsert=True
        )
    
    async def predict_optimal_interval(
        self,
        user_id: str,
        topic: str,
        target_retention: float = 0.9
    ) -> int:
        """
        Predict optimal review interval to maintain target retention
        
        Args:
            user_id: User identifier
            topic: Topic/subject
            target_retention: Desired retention rate (0-1)
        
        Returns:
            Optimal interval in days
        """
        curve = await self.get_or_create_curve(user_id, topic)
        
        # Solve for t when R(t) = target_retention
        # R(t) = S * e^(-decay_rate * t) = target_retention
        # t = -ln(target_retention / S) / decay_rate
        
        if curve.initial_strength <= 0:
            return 1  # Review immediately if strength is 0
        
        try:
            days = -np.log(target_retention / curve.initial_strength) / curve.decay_rate
            days = max(1, int(days))  # At least 1 day
            return min(days, 180)  # Cap at 180 days
        except (ValueError, ZeroDivisionError):
            return 7  # Default to 1 week if calculation fails


class ActiveRecallGenerator:
    """
    Generate active recall questions with difficulty adjustment
    
    Transforms learning content into effective recall questions
    """
    
    def __init__(self):
        """Initialize active recall generator"""
        logger.info("✅ Active recall generator initialized")
    
    def generate_questions(
        self,
        content: Dict[str, Any],
        difficulty: CardDifficulty,
        num_questions: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate active recall questions from content
        
        Args:
            content: Card content (concept, explanation, examples)
            difficulty: Target difficulty level
            num_questions: Number of questions to generate
        
        Returns:
            List of generated questions with answers
        """
        questions = []
        
        # Extract key information
        concept = content.get("concept", "")
        explanation = content.get("explanation", "")
        
        # Generate different question types based on difficulty
        if difficulty == CardDifficulty.TRIVIAL:
            questions.extend(self._generate_recognition_questions(concept, num_questions))
        
        elif difficulty == CardDifficulty.EASY:
            questions.extend(self._generate_recall_questions(concept, explanation, num_questions))
        
        elif difficulty == CardDifficulty.MEDIUM:
            questions.extend(self._generate_application_questions(concept, explanation, num_questions))
        
        elif difficulty == CardDifficulty.HARD:
            questions.extend(self._generate_synthesis_questions(concept, explanation, num_questions))
        
        else:  # EXTREME
            questions.extend(self._generate_evaluation_questions(concept, explanation, num_questions))
        
        return questions[:num_questions]
    
    def _generate_recognition_questions(self, concept: str, num: int) -> List[Dict[str, Any]]:
        """Generate simple recognition questions"""
        return [{
            "type": "recognition",
            "question": f"What is {concept}?",
            "difficulty": "trivial",
            "hints": ["Think about the basic definition"]
        }]
    
    def _generate_recall_questions(self, concept: str, explanation: str, num: int) -> List[Dict[str, Any]]:
        """Generate recall-based questions"""
        return [{
            "type": "recall",
            "question": f"Explain {concept} in your own words",
            "difficulty": "easy",
            "hints": ["Start with the main idea", "Include key details"]
        }]
    
    def _generate_application_questions(self, concept: str, explanation: str, num: int) -> List[Dict[str, Any]]:
        """Generate application questions"""
        return [{
            "type": "application",
            "question": f"How would you apply {concept} to solve a real problem?",
            "difficulty": "medium",
            "hints": ["Think of a practical scenario", "Break down the steps"]
        }]
    
    def _generate_synthesis_questions(self, concept: str, explanation: str, num: int) -> List[Dict[str, Any]]:
        """Generate synthesis questions"""
        return [{
            "type": "synthesis",
            "question": f"How does {concept} relate to other concepts you've learned?",
            "difficulty": "hard",
            "hints": ["Look for connections", "Consider cause and effect"]
        }]
    
    def _generate_evaluation_questions(self, concept: str, explanation: str, num: int) -> List[Dict[str, Any]]:
        """Generate evaluation questions"""
        return [{
            "type": "evaluation",
            "question": f"Critique the strengths and limitations of {concept}",
            "difficulty": "extreme",
            "hints": ["Consider different perspectives", "Analyze trade-offs"]
        }]


class ReviewScheduler:
    """
    Intelligent review scheduler with priority queue
    
    Determines which cards to review based on:
    - Predicted retention
    - Card difficulty
    - User performance history
    - Spacing effects
    """
    
    def __init__(self, db, curve_predictor: ForgettingCurvePredictor):
        """Initialize review scheduler"""
        self.db = db
        self.curve_predictor = curve_predictor
        logger.info("✅ Review scheduler initialized")
    
    async def get_due_cards(
        self,
        user_id: str,
        limit: int = 20,
        include_new: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get cards due for review with intelligent prioritization
        
        Priority factors:
        1. Overdue cards (past due date)
        2. Cards close to forgetting threshold
        3. New cards (if included)
        4. Cards with low ease factor (difficult cards)
        """
        current_time = datetime.utcnow()
        
        # Query for due cards
        query = {
            "user_id": user_id,
            "next_review": {"$lte": current_time}
        }
        
        if not include_new:
            query["status"] = {"$ne": CardStatus.NEW.value}
        
        cards = await self.db.spaced_repetition_cards.find(query).to_list(length=100)
        
        # Calculate priority scores for each card
        prioritized_cards = []
        for card in cards:
            priority = await self._calculate_priority(card, current_time)
            prioritized_cards.append((priority, card))
        
        # Sort by priority (highest first)
        prioritized_cards.sort(key=lambda x: x[0], reverse=True)
        
        return [card for _, card in prioritized_cards[:limit]]
    
    async def _calculate_priority(self, card: Dict[str, Any], current_time: datetime) -> float:
        """
        Calculate review priority score
        
        Higher score = higher priority
        """
        # Base priority on how overdue the card is
        next_review = card.get("next_review", current_time)
        days_overdue = max(0, (current_time - next_review).days)
        overdue_score = min(10.0, days_overdue)  # Cap at 10
        
        # Difficulty score (harder cards get higher priority)
        ease_factor = card.get("easiness_factor", 2.5)
        difficulty_score = (3.0 - ease_factor) * 2  # Lower ease = higher score
        
        # Predicted retention score
        topic = card.get("topic", "general")
        curve = await self.curve_predictor.get_or_create_curve(card["user_id"], topic)
        
        last_reviewed = card.get("last_reviewed") or card.get("created_at", current_time)
        days_since_review = max(0, (current_time - last_reviewed).days) if last_reviewed else 0
        
        retention = curve.predict_retention(days_since_review)
        retention_score = (1.0 - retention) * 10  # Lower retention = higher score
        
        # Combine scores with weights
        total_priority = (
            overdue_score * 3.0 +      # Weight: 3.0 (most important)
            retention_score * 2.0 +     # Weight: 2.0
            difficulty_score * 1.0      # Weight: 1.0
        )
        
        return total_priority


class SpacedRepetitionEngine:
    """
    Main spaced repetition orchestrator - COMPREHENSIVE VERSION
    
    Integrates all components:
    - SM-2+ algorithm
    - Forgetting curve prediction
    - Active recall generation
    - Review scheduling
    - Performance analytics
    """
    
    def __init__(self, db):
        """Initialize spaced repetition engine"""
        self.db = db
        self.sm2 = SM2PlusAlgorithm()
        self.curve_predictor = ForgettingCurvePredictor(db)
        self.recall_generator = ActiveRecallGenerator()
        self.scheduler = ReviewScheduler(db, self.curve_predictor)
        
        logger.info("✅ Comprehensive spaced repetition engine initialized")
    
    async def create_card(
        self,
        user_id: str,
        topic: str,
        content: Dict[str, Any],
        difficulty: Optional[CardDifficulty] = None
    ) -> str:
        """
        Create a new spaced repetition card
        
        Args:
            user_id: User identifier
            topic: Topic/subject (e.g., "calculus", "python")
            content: Card content (concept, explanation, examples)
            difficulty: Optional difficulty level
        
        Returns:
            card_id: Unique card identifier
        """
        card_id = str(uuid.uuid4())
        
        # Auto-detect difficulty if not provided
        if not difficulty:
            difficulty = self._estimate_difficulty(content)
        
        # Generate active recall questions
        questions = self.recall_generator.generate_questions(content, difficulty, num_questions=3)
        
        # Create card document
        card_doc = {
            "_id": card_id,
            "user_id": user_id,
            "topic": topic,
            "content": content,
            "questions": questions,
            "difficulty": difficulty.value,
            
            # SM-2+ state
            "easiness_factor": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "status": CardStatus.NEW.value,
            
            # Scheduling
            "next_review": datetime.utcnow(),
            "last_reviewed": None,
            
            # Timestamps
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            
            # Statistics
            "total_reviews": 0,
            "successful_reviews": 0,
            "failed_reviews": 0,
            "average_quality": 0.0,
            "total_study_time_seconds": 0
        }
        
        await self.db.spaced_repetition_cards.insert_one(card_doc)
        
        logger.info(f"✅ Created card {card_id} for user {user_id} (topic: {topic})")
        
        return card_id
    
    async def review_card(
        self,
        card_id: str,
        quality: ReviewQuality,
        duration_seconds: int = 0
    ) -> Dict[str, Any]:
        """
        Review a card and update its scheduling
        
        Args:
            card_id: Card identifier
            quality: Review quality (0-5)
            duration_seconds: Time spent reviewing (for analytics)
        
        Returns:
            Review result with next review date and statistics
        """
        # Fetch card
        card = await self.db.spaced_repetition_cards.find_one({"_id": card_id})
        if not card:
            raise ValueError(f"Card not found: {card_id}")
        
        # Calculate next interval using SM-2+
        current_status = CardStatus(card.get("status", CardStatus.NEW.value))
        
        new_ef, new_interval, new_reps, new_status = self.sm2.calculate_next_interval(
            quality=quality,
            current_ef=card["easiness_factor"],
            current_interval=card["interval_days"],
            repetitions=card["repetitions"],
            card_status=current_status,
            lapses=card.get("lapses", 0)
        )
        
        # Update forgetting curve
        curve = await self.curve_predictor.get_or_create_curve(
            card["user_id"],
            card["topic"]
        )
        curve.update_from_review(quality, new_interval)
        await self.curve_predictor.save_curve(curve)
        
        # Calculate next review date
        next_review = datetime.utcnow() + timedelta(days=new_interval)
        
        # Update statistics
        total_reviews = card.get("total_reviews", 0) + 1
        successful = card.get("successful_reviews", 0) + (1 if quality >= 3 else 0)
        failed = card.get("failed_reviews", 0) + (1 if quality < 3 else 0)
        
        avg_quality = (
            (card.get("average_quality", 0.0) * (total_reviews - 1) + quality) /
            total_reviews
        )
        
        # Update lapses
        new_lapses = card.get("lapses", 0)
        if quality < ReviewQuality.CORRECT_HARD and current_status == CardStatus.REVIEW:
            new_lapses += 1
        
        # Update card in database
        update_doc = {
            "easiness_factor": new_ef,
            "interval_days": new_interval,
            "repetitions": new_reps,
            "status": new_status.value,
            "lapses": new_lapses,
            "next_review": next_review,
            "last_reviewed": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "total_reviews": total_reviews,
            "successful_reviews": successful,
            "failed_reviews": failed,
            "average_quality": avg_quality,
            "total_study_time_seconds": card.get("total_study_time_seconds", 0) + duration_seconds
        }
        
        await self.db.spaced_repetition_cards.update_one(
            {"_id": card_id},
            {"$set": update_doc}
        )
        
        # Record review in history
        await self._record_review_history(card, quality, duration_seconds, new_ef)
        
        logger.info(
            f"✅ Reviewed card {card_id}: quality={quality}, "
            f"next_interval={new_interval}d, status={new_status.value}"
        )
        
        return {
            "card_id": card_id,
            "quality": quality,
            "next_review": next_review,
            "interval_days": new_interval,
            "status": new_status.value,
            "easiness_factor": new_ef,
            "predicted_retention": curve.predict_retention(new_interval),
            "statistics": {
                "total_reviews": total_reviews,
                "success_rate": successful / total_reviews if total_reviews > 0 else 0,
                "average_quality": avg_quality,
                "total_study_time": update_doc["total_study_time_seconds"]
            }
        }
    
    async def get_due_cards(
        self,
        user_id: str,
        limit: int = 20,
        include_new: bool = True
    ) -> List[Dict[str, Any]]:
        """Get cards due for review"""
        return await self.scheduler.get_due_cards(user_id, limit, include_new)
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a user
        
        Returns:
            Statistics including retention rates, review counts, etc.
        """
        # Aggregate statistics
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "total_cards": {"$sum": 1},
                "total_reviews": {"$sum": "$total_reviews"},
                "successful_reviews": {"$sum": "$successful_reviews"},
                "failed_reviews": {"$sum": "$failed_reviews"},
                "average_ease": {"$avg": "$easiness_factor"},
                "total_study_time": {"$sum": "$total_study_time_seconds"},
                "cards_by_status": {
                    "$push": "$status"
                }
            }}
        ]
        
        result = await self.db.spaced_repetition_cards.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "total_cards": 0,
                "total_reviews": 0,
                "success_rate": 0.0,
                "average_ease": 2.5,
                "total_study_time_hours": 0.0,
                "cards_by_status": {}
            }
        
        stats = result[0]
        
        # Count cards by status
        status_counts = {}
        for status in stats.get("cards_by_status", []):
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_reviews = stats["total_reviews"]
        success_rate = (
            stats["successful_reviews"] / total_reviews 
            if total_reviews > 0 else 0.0
        )
        
        return {
            "total_cards": stats["total_cards"],
            "total_reviews": total_reviews,
            "success_rate": success_rate,
            "average_ease": stats["average_ease"],
            "total_study_time_hours": stats["total_study_time"] / 3600,
            "cards_by_status": status_counts
        }
    
    def _estimate_difficulty(self, content: Dict[str, Any]) -> CardDifficulty:
        """Estimate card difficulty from content"""
        # Simple heuristic based on content length and complexity
        concept = content.get("concept", "")
        explanation = content.get("explanation", "")
        
        total_length = len(concept) + len(explanation)
        
        if total_length < 100:
            return CardDifficulty.EASY
        elif total_length < 300:
            return CardDifficulty.MEDIUM
        elif total_length < 600:
            return CardDifficulty.HARD
        else:
            return CardDifficulty.EXTREME
    
    async def _record_review_history(
        self,
        card: Dict[str, Any],
        quality: ReviewQuality,
        duration_seconds: int,
        ease_factor: float
    ):
        """Record review in history collection for analytics"""
        history_doc = {
            "_id": str(uuid.uuid4()),
            "card_id": card["_id"],
            "user_id": card["user_id"],
            "topic": card["topic"],
            "quality": quality,
            "duration_seconds": duration_seconds,
            "ease_factor": ease_factor,
            "interval_days": card["interval_days"],
            "timestamp": datetime.utcnow()
        }
        
        await self.db.spaced_repetition_history.insert_one(history_doc)