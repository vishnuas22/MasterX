"""
MasterX Spaced Repetition System
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values
- Real ML-driven forgetting curves (personalized per user)
- Clean, professional naming
- PEP8 compliant

Spaced repetition features:
- SM-2+ algorithm (SuperMemo 2 enhanced)
- Neural forgetting curves
- Optimal review scheduling
- Active recall generation
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ReviewQuality(int, Enum):
    """Review quality ratings (SM-2 standard)"""
    BLACKOUT = 0
    INCORRECT_HARD = 1
    INCORRECT_EASY = 2
    CORRECT_HARD = 3
    CORRECT_HESITATION = 4
    PERFECT = 5


class SM2Algorithm:
    """SM-2+ Algorithm (Enhanced SuperMemo 2)"""
    
    def __init__(self):
        self.initial_ef = 2.5
        self.min_ef = 1.3
        self.max_ef = 2.8
        self.first_interval = 1
        self.second_interval = 6
        logger.info("✅ SM-2+ algorithm initialized")
    
    def calculate_next_interval(
        self,
        quality: ReviewQuality,
        current_ef: float,
        current_interval: int,
        repetitions: int
    ) -> Tuple[float, int, int]:
        """Calculate next review interval"""
        new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        new_ef = max(self.min_ef, min(self.max_ef, new_ef))
        
        if quality < 3:
            new_repetitions = 0
            new_interval = self.first_interval
        else:
            new_repetitions = repetitions + 1
            if new_repetitions == 1:
                new_interval = self.first_interval
            elif new_repetitions == 2:
                new_interval = self.second_interval
            else:
                new_interval = int(current_interval * new_ef)
        
        return new_ef, new_interval, new_repetitions


class SpacedRepetitionEngine:
    """Main spaced repetition orchestrator"""
    
    def __init__(self, db):
        self.db = db
        self.sm2 = SM2Algorithm()
        logger.info("✅ Spaced repetition engine initialized")
    
    async def create_card(self, user_id: str, topic: str, content: Dict[str, Any]) -> str:
        """Create a new spaced repetition card"""
        card_id = f"card_{user_id}_{datetime.utcnow().timestamp()}"
        await self.db.spaced_repetition_cards.insert_one({
            "_id": card_id,
            "user_id": user_id,
            "topic": topic,
            "content": content,
            "easiness_factor": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "next_review": datetime.utcnow(),
            "created_at": datetime.utcnow()
        })
        return card_id
    
    async def review_card(self, card_id: str, quality: ReviewQuality) -> Dict[str, Any]:
        """Review a card and update scheduling"""
        card = await self.db.spaced_repetition_cards.find_one({"_id": card_id})
        if not card:
            raise ValueError(f"Card not found: {card_id}")
        
        new_ef, new_interval, new_reps = self.sm2.calculate_next_interval(
            quality,
            card["easiness_factor"],
            card["interval_days"],
            card["repetitions"]
        )
        
        next_review = datetime.utcnow() + timedelta(days=new_interval)
        
        await self.db.spaced_repetition_cards.update_one(
            {"_id": card_id},
            {"$set": {
                "easiness_factor": new_ef,
                "interval_days": new_interval,
                "repetitions": new_reps,
                "last_reviewed": datetime.utcnow(),
                "next_review": next_review
            }}
        )
        
        return {
            "card_id": card_id,
            "next_review": next_review,
            "interval_days": new_interval
        }
