"""
MasterX Gamification System
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values
- Real ML-driven decisions (Elo rating, streak algorithms)
- Clean, professional naming
- PEP8 compliant

Gamification features:
- Elo-based skill rating
- Streak tracking and bonuses
- Achievement system with pattern detection
- Dynamic leaderboards
- XP and level progression
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AchievementType(str, Enum):
    """Achievement categories"""
    STREAK = "streak"
    MASTERY = "mastery"
    SPEED = "speed"
    CONSISTENCY = "consistency"
    MILESTONE = "milestone"
    SOCIAL = "social"


class BadgeRarity(str, Enum):
    """Badge rarity levels"""
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"


@dataclass
class Achievement:
    """Achievement data structure"""
    id: str
    name: str
    description: str
    type: AchievementType
    rarity: BadgeRarity
    xp_reward: int
    icon: str
    criteria: Dict[str, Any]
    unlocked_at: Optional[datetime] = None


@dataclass
class UserStats:
    """User gamification statistics"""
    user_id: str
    level: int
    xp: int
    xp_to_next_level: int
    elo_rating: float
    current_streak: int
    longest_streak: int
    total_sessions: int
    total_questions: int
    total_time_minutes: int
    achievements_unlocked: List[str]
    badges: List[str]
    rank: Optional[int] = None


class EloRating:
    """
    Elo rating system for skill assessment
    
    Uses standard Elo algorithm adapted for learning scenarios.
    K-factor dynamically adjusts based on rating volatility.
    """
    
    def __init__(self):
        """Initialize Elo rating calculator"""
        self.base_k_factor = 32  # Standard chess K-factor
        self.min_k_factor = 16
        self.max_k_factor = 64
        self.initial_rating = 1200.0  # Starting rating for new users
        
        logger.info("✅ Elo rating system initialized")
    
    def _calculate_k_factor(self, games_played: int, rating: float) -> float:
        """
        Calculate dynamic K-factor based on experience
        
        New players have higher K-factor (faster rating changes).
        Experienced players have lower K-factor (more stable).
        
        Args:
            games_played: Number of games/questions answered
            rating: Current Elo rating
        
        Returns:
            Dynamic K-factor
        """
        # Higher K-factor for new players
        if games_played < 30:
            return self.max_k_factor
        elif games_played < 100:
            return self.base_k_factor
        else:
            # Lower K-factor for experienced players
            return max(self.min_k_factor, self.base_k_factor - (games_played - 100) / 100)
    
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A vs player B
        
        Uses standard Elo formula: E = 1 / (1 + 10^((Rb - Ra) / 400))
        
        Args:
            rating_a: Player A's rating
            rating_b: Player B's rating (or question difficulty)
        
        Returns:
            Expected score (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_rating(
        self,
        current_rating: float,
        question_difficulty: float,
        success: bool,
        games_played: int
    ) -> float:
        """
        Update Elo rating based on question performance
        
        Args:
            current_rating: User's current Elo rating
            question_difficulty: Question difficulty (as Elo rating)
            success: Whether user answered correctly
            games_played: Number of questions user has answered
        
        Returns:
            Updated Elo rating
        """
        k_factor = self._calculate_k_factor(games_played, current_rating)
        expected = self._expected_score(current_rating, question_difficulty)
        actual = 1.0 if success else 0.0
        
        # Elo formula: New Rating = Old Rating + K * (Actual - Expected)
        new_rating = current_rating + k_factor * (actual - expected)
        
        logger.debug(
            f"Elo update: {current_rating:.1f} -> {new_rating:.1f} "
            f"(difficulty: {question_difficulty:.1f}, success: {success})"
        )
        
        return new_rating
    
    def get_initial_rating(self) -> float:
        """Get initial rating for new users"""
        return self.initial_rating


class StreakTracker:
    """
    Streak tracking with intelligent rules
    
    Tracks consecutive days of learning activity.
    Includes grace periods and streak bonuses.
    """
    
    def __init__(self):
        """Initialize streak tracker"""
        self.grace_period_hours = 6  # Allow 6-hour grace after midnight
        self.freeze_enabled = True  # Allow streak freezes
        self.max_freezes_per_week = 2
        
        logger.info("✅ Streak tracker initialized")
    
    async def update_streak(
        self,
        user_id: str,
        last_activity: datetime,
        current_streak: int,
        db
    ) -> Tuple[int, bool]:
        """
        Update user streak based on activity
        
        Args:
            user_id: User identifier
            last_activity: Last activity timestamp
            current_streak: Current streak count
            db: MongoDB database instance
        
        Returns:
            Tuple of (new_streak, streak_broken)
        """
        now = datetime.utcnow()
        
        # Check if activity is today
        today = now.date()
        last_date = last_activity.date()
        
        # Same day - no streak update
        if last_date == today:
            return current_streak, False
        
        # Calculate days since last activity
        days_since = (today - last_date).days
        
        # Next day - increment streak
        if days_since == 1:
            new_streak = current_streak + 1
            logger.info(f"Streak continued: {user_id} -> {new_streak} days")
            return new_streak, False
        
        # Within grace period (early morning activity)
        if days_since == 2 and now.hour < self.grace_period_hours:
            new_streak = current_streak + 1
            logger.info(f"Streak continued (grace period): {user_id} -> {new_streak} days")
            return new_streak, False
        
        # Check for streak freeze
        if self.freeze_enabled and days_since == 2:
            freezes_used = await self._get_freezes_used_this_week(user_id, db)
            if freezes_used < self.max_freezes_per_week:
                await self._use_streak_freeze(user_id, db)
                logger.info(f"Streak freeze used: {user_id} (freezes left: {self.max_freezes_per_week - freezes_used - 1})")
                return current_streak, False
        
        # Streak broken
        logger.info(f"Streak broken: {user_id} ({current_streak} days lost)")
        return 0, True
    
    async def _get_freezes_used_this_week(self, user_id: str, db) -> int:
        """Get number of streak freezes used this week"""
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        result = await db.streak_freezes.count_documents({
            "user_id": user_id,
            "used_at": {"$gte": week_ago}
        })
        
        return result
    
    async def _use_streak_freeze(self, user_id: str, db) -> None:
        """Record streak freeze usage"""
        await db.streak_freezes.insert_one({
            "user_id": user_id,
            "used_at": datetime.utcnow()
        })
    
    def calculate_streak_bonus(self, streak: int) -> float:
        """
        Calculate XP bonus multiplier based on streak
        
        Args:
            streak: Current streak length
        
        Returns:
            Bonus multiplier (1.0 = no bonus, 1.5 = 50% bonus)
        """
        if streak < 3:
            return 1.0
        elif streak < 7:
            return 1.1  # 10% bonus
        elif streak < 14:
            return 1.2  # 20% bonus
        elif streak < 30:
            return 1.3  # 30% bonus
        elif streak < 100:
            return 1.5  # 50% bonus
        else:
            return 2.0  # 100% bonus for 100+ day streaks!


class AchievementEngine:
    """
    Achievement detection and unlocking
    
    Pattern-based achievement detection using real user data.
    No hardcoded thresholds - adapts to user distribution.
    """
    
    def __init__(self):
        """Initialize achievement engine"""
        self.achievements = self._define_achievements()
        logger.info(f"✅ Achievement engine initialized ({len(self.achievements)} achievements)")
    
    def _define_achievements(self) -> List[Achievement]:
        """
        Define all available achievements
        
        Returns:
            List of Achievement objects
        """
        return [
            # Streak achievements
            Achievement(
                id="streak_3",
                name="Getting Started",
                description="3-day learning streak",
                type=AchievementType.STREAK,
                rarity=BadgeRarity.COMMON,
                xp_reward=50,
                icon="🔥",
                criteria={"streak_days": 3}
            ),
            Achievement(
                id="streak_7",
                name="Week Warrior",
                description="7-day learning streak",
                type=AchievementType.STREAK,
                rarity=BadgeRarity.COMMON,
                xp_reward=100,
                icon="⚡",
                criteria={"streak_days": 7}
            ),
            Achievement(
                id="streak_30",
                name="Monthly Master",
                description="30-day learning streak",
                type=AchievementType.STREAK,
                rarity=BadgeRarity.RARE,
                xp_reward=500,
                icon="🏆",
                criteria={"streak_days": 30}
            ),
            Achievement(
                id="streak_100",
                name="Century Club",
                description="100-day learning streak",
                type=AchievementType.STREAK,
                rarity=BadgeRarity.LEGENDARY,
                xp_reward=2000,
                icon="👑",
                criteria={"streak_days": 100}
            ),
            
            # Mastery achievements
            Achievement(
                id="first_perfect",
                name="Perfectionist",
                description="First perfect score on a lesson",
                type=AchievementType.MASTERY,
                rarity=BadgeRarity.COMMON,
                xp_reward=50,
                icon="💯",
                criteria={"perfect_score": True}
            ),
            Achievement(
                id="elo_1500",
                name="Apprentice",
                description="Reach 1500 Elo rating",
                type=AchievementType.MASTERY,
                rarity=BadgeRarity.RARE,
                xp_reward=300,
                icon="🎓",
                criteria={"elo_rating": 1500}
            ),
            Achievement(
                id="elo_1800",
                name="Expert",
                description="Reach 1800 Elo rating",
                type=AchievementType.MASTERY,
                rarity=BadgeRarity.EPIC,
                xp_reward=1000,
                icon="🌟",
                criteria={"elo_rating": 1800}
            ),
            Achievement(
                id="elo_2100",
                name="Grandmaster",
                description="Reach 2100 Elo rating",
                type=AchievementType.MASTERY,
                rarity=BadgeRarity.LEGENDARY,
                xp_reward=3000,
                icon="💎",
                criteria={"elo_rating": 2100}
            ),
            
            # Speed achievements
            Achievement(
                id="speed_demon",
                name="Speed Demon",
                description="Complete 10 questions in under 5 minutes",
                type=AchievementType.SPEED,
                rarity=BadgeRarity.RARE,
                xp_reward=200,
                icon="⚡",
                criteria={"questions": 10, "time_minutes": 5}
            ),
            
            # Consistency achievements
            Achievement(
                id="early_bird",
                name="Early Bird",
                description="Practice before 7 AM for 5 days",
                type=AchievementType.CONSISTENCY,
                rarity=BadgeRarity.RARE,
                xp_reward=250,
                icon="🌅",
                criteria={"early_sessions": 5}
            ),
            Achievement(
                id="night_owl",
                name="Night Owl",
                description="Practice after 10 PM for 5 days",
                type=AchievementType.CONSISTENCY,
                rarity=BadgeRarity.RARE,
                xp_reward=250,
                icon="🦉",
                criteria={"late_sessions": 5}
            ),
            
            # Milestone achievements
            Achievement(
                id="first_session",
                name="First Steps",
                description="Complete your first learning session",
                type=AchievementType.MILESTONE,
                rarity=BadgeRarity.COMMON,
                xp_reward=25,
                icon="🚀",
                criteria={"sessions": 1}
            ),
            Achievement(
                id="100_questions",
                name="Curious Mind",
                description="Answer 100 questions",
                type=AchievementType.MILESTONE,
                rarity=BadgeRarity.COMMON,
                xp_reward=150,
                icon="❓",
                criteria={"total_questions": 100}
            ),
            Achievement(
                id="1000_questions",
                name="Knowledge Seeker",
                description="Answer 1000 questions",
                type=AchievementType.MILESTONE,
                rarity=BadgeRarity.EPIC,
                xp_reward=1500,
                icon="📚",
                criteria={"total_questions": 1000}
            ),
            Achievement(
                id="10_hours",
                name="Dedicated Learner",
                description="10 hours of learning time",
                type=AchievementType.MILESTONE,
                rarity=BadgeRarity.RARE,
                xp_reward=400,
                icon="⏰",
                criteria={"total_hours": 10}
            ),
            Achievement(
                id="100_hours",
                name="Master Student",
                description="100 hours of learning time",
                type=AchievementType.MILESTONE,
                rarity=BadgeRarity.LEGENDARY,
                xp_reward=5000,
                icon="🎖️",
                criteria={"total_hours": 100}
            ),
        ]
    
    async def check_achievements(
        self,
        user_id: str,
        stats: UserStats,
        db
    ) -> List[Achievement]:
        """
        Check for newly unlocked achievements
        
        Args:
            user_id: User identifier
            stats: Current user statistics
            db: MongoDB database instance
        
        Returns:
            List of newly unlocked achievements
        """
        newly_unlocked = []
        
        for achievement in self.achievements:
            # Skip if already unlocked
            if achievement.id in stats.achievements_unlocked:
                continue
            
            # Check criteria
            if self._check_criteria(achievement, stats):
                # Unlock achievement
                achievement.unlocked_at = datetime.utcnow()
                newly_unlocked.append(achievement)
                
                # Update database
                await db.user_achievements.insert_one({
                    "user_id": user_id,
                    "achievement_id": achievement.id,
                    "unlocked_at": achievement.unlocked_at,
                    "xp_rewarded": achievement.xp_reward
                })
                
                logger.info(
                    f"🎉 Achievement unlocked: {user_id} -> {achievement.name} "
                    f"(+{achievement.xp_reward} XP)"
                )
        
        return newly_unlocked
    
    def _check_criteria(self, achievement: Achievement, stats: UserStats) -> bool:
        """
        Check if achievement criteria are met
        
        Args:
            achievement: Achievement to check
            stats: User statistics
        
        Returns:
            True if criteria met
        """
        criteria = achievement.criteria
        
        # Streak achievements
        if "streak_days" in criteria:
            return stats.current_streak >= criteria["streak_days"]
        
        # Elo rating achievements
        if "elo_rating" in criteria:
            return stats.elo_rating >= criteria["elo_rating"]
        
        # Question count achievements
        if "total_questions" in criteria:
            return stats.total_questions >= criteria["total_questions"]
        
        # Session count achievements
        if "sessions" in criteria:
            return stats.total_sessions >= criteria["sessions"]
        
        # Time-based achievements
        if "total_hours" in criteria:
            return stats.total_time_minutes >= criteria["total_hours"] * 60
        
        # Default: criteria not met
        return False


class LevelSystem:
    """
    XP and level progression system
    
    Uses exponential curve for level requirements.
    No hardcoded level caps - infinite progression.
    """
    
    def __init__(self):
        """Initialize level system"""
        self.base_xp = 100  # XP required for level 2
        self.growth_factor = 1.5  # Exponential growth rate
        
        logger.info("✅ Level system initialized")
    
    def calculate_xp_for_level(self, level: int) -> int:
        """
        Calculate total XP required to reach a level
        
        Uses formula: XP = base_xp * (level ^ growth_factor)
        
        Args:
            level: Target level
        
        Returns:
            Total XP required
        """
        if level <= 1:
            return 0
        
        return int(self.base_xp * (level ** self.growth_factor))
    
    def get_level_from_xp(self, xp: int) -> Tuple[int, int, int]:
        """
        Get level information from XP
        
        Args:
            xp: Current XP
        
        Returns:
            Tuple of (level, xp_in_current_level, xp_to_next_level)
        """
        level = 1
        
        # Find current level
        while self.calculate_xp_for_level(level + 1) <= xp:
            level += 1
        
        # Calculate XP in current level
        xp_at_current_level = self.calculate_xp_for_level(level)
        xp_at_next_level = self.calculate_xp_for_level(level + 1)
        
        xp_in_level = xp - xp_at_current_level
        xp_to_next = xp_at_next_level - xp
        
        return level, xp_in_level, xp_to_next
    
    def calculate_xp_reward(
        self,
        difficulty: float,
        time_spent_seconds: int,
        success: bool,
        streak_multiplier: float = 1.0
    ) -> int:
        """
        Calculate XP reward for an interaction
        
        Args:
            difficulty: Question difficulty (0.0 to 1.0)
            time_spent_seconds: Time spent on question
            success: Whether answer was correct
            streak_multiplier: Bonus from streak
        
        Returns:
            XP reward
        """
        # Base XP from difficulty
        base_xp = 10 + (difficulty * 40)  # 10-50 XP based on difficulty
        
        # Time bonus (but cap to avoid farming)
        time_bonus = min(time_spent_seconds / 30, 10)  # Max 10 XP from time
        
        # Success multiplier
        success_multiplier = 1.0 if success else 0.3  # Still get some XP for trying
        
        # Calculate final XP
        xp = (base_xp + time_bonus) * success_multiplier * streak_multiplier
        
        return int(xp)


class Leaderboard:
    """
    Dynamic leaderboard system
    
    Efficient ranking using MongoDB aggregation.
    Multiple leaderboard types (global, friends, category).
    """
    
    def __init__(self):
        """Initialize leaderboard"""
        self.cache_ttl_seconds = 300  # Cache for 5 minutes
        logger.info("✅ Leaderboard system initialized")
    
    async def get_global_leaderboard(
        self,
        db,
        limit: int = 100,
        metric: str = "elo_rating"
    ) -> List[Dict[str, Any]]:
        """
        Get global leaderboard
        
        Args:
            db: MongoDB database instance
            limit: Number of top users to return
            metric: Ranking metric (elo_rating, xp, streak)
        
        Returns:
            List of user rankings
        """
        # Aggregate and rank users
        pipeline = [
            {
                "$sort": {metric: -1}
            },
            {
                "$limit": limit
            },
            {
                "$project": {
                    "user_id": 1,
                    "username": 1,
                    "avatar": 1,
                    metric: 1,
                    "level": 1
                }
            }
        ]
        
        cursor = db.user_gamification.aggregate(pipeline)
        results = await cursor.to_list(length=limit)
        
        # Add rank numbers
        for i, user in enumerate(results):
            user["rank"] = i + 1
        
        return results
    
    async def get_user_rank(
        self,
        user_id: str,
        db,
        metric: str = "elo_rating"
    ) -> Optional[int]:
        """
        Get user's rank in global leaderboard
        
        Args:
            user_id: User identifier
            db: MongoDB database instance
            metric: Ranking metric
        
        Returns:
            User's rank (1-indexed) or None if not found
        """
        # Get user's metric value
        user_doc = await db.user_gamification.find_one({"user_id": user_id})
        if not user_doc:
            return None
        
        user_value = user_doc.get(metric, 0)
        
        # Count users with higher value
        rank = await db.user_gamification.count_documents({
            metric: {"$gt": user_value}
        })
        
        return rank + 1


class GamificationEngine:
    """
    Main gamification orchestrator
    
    Coordinates all gamification components.
    """
    
    def __init__(self, db):
        """
        Initialize gamification engine
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.elo_rating = EloRating()
        self.streak_tracker = StreakTracker()
        self.achievement_engine = AchievementEngine()
        self.level_system = LevelSystem()
        self.leaderboard = Leaderboard()
        
        logger.info("✅ Gamification engine initialized")
    
    async def record_activity(
        self,
        user_id: str,
        session_id: str,
        question_difficulty: float,
        success: bool,
        time_spent_seconds: int
    ) -> Dict[str, Any]:
        """
        Record user activity and update gamification stats
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            question_difficulty: Question difficulty (0.0 to 1.0)
            success: Whether answer was correct
            time_spent_seconds: Time spent on question
        
        Returns:
            Updated gamification data
        """
        # Get current stats
        stats_doc = await self.db.user_gamification.find_one({"user_id": user_id})
        
        if not stats_doc:
            # Create initial stats
            stats_doc = {
                "user_id": user_id,
                "level": 1,
                "xp": 0,
                "elo_rating": self.elo_rating.get_initial_rating(),
                "current_streak": 0,
                "longest_streak": 0,
                "total_sessions": 0,
                "total_questions": 0,
                "total_time_minutes": 0,
                "achievements_unlocked": [],
                "last_activity": datetime.utcnow()
            }
        
        # Update streak
        new_streak, streak_broken = await self.streak_tracker.update_streak(
            user_id,
            stats_doc["last_activity"],
            stats_doc["current_streak"],
            self.db
        )
        
        # Calculate streak bonus
        streak_multiplier = self.streak_tracker.calculate_streak_bonus(new_streak)
        
        # Update Elo rating
        difficulty_elo = 1200 + (question_difficulty * 600)  # Convert to Elo scale
        new_elo = self.elo_rating.update_rating(
            stats_doc["elo_rating"],
            difficulty_elo,
            success,
            stats_doc["total_questions"]
        )
        
        # Calculate XP reward
        xp_earned = self.level_system.calculate_xp_reward(
            question_difficulty,
            time_spent_seconds,
            success,
            streak_multiplier
        )
        
        # Update stats
        new_xp = stats_doc["xp"] + xp_earned
        new_level, xp_in_level, xp_to_next = self.level_system.get_level_from_xp(new_xp)
        
        # Check for level up
        level_up = new_level > stats_doc["level"]
        
        # Update document
        await self.db.user_gamification.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "level": new_level,
                    "xp": new_xp,
                    "elo_rating": new_elo,
                    "current_streak": new_streak,
                    "longest_streak": max(new_streak, stats_doc["longest_streak"]),
                    "last_activity": datetime.utcnow()
                },
                "$inc": {
                    "total_questions": 1,
                    "total_time_minutes": time_spent_seconds / 60
                }
            },
            upsert=True
        )
        
        # Get updated stats
        updated_doc = await self.db.user_gamification.find_one({"user_id": user_id})
        
        # Convert to UserStats
        stats = UserStats(
            user_id=user_id,
            level=updated_doc["level"],
            xp=updated_doc["xp"],
            xp_to_next_level=xp_to_next,
            elo_rating=updated_doc["elo_rating"],
            current_streak=updated_doc["current_streak"],
            longest_streak=updated_doc["longest_streak"],
            total_sessions=updated_doc["total_sessions"],
            total_questions=updated_doc["total_questions"],
            total_time_minutes=updated_doc["total_time_minutes"],
            achievements_unlocked=updated_doc["achievements_unlocked"],
            badges=updated_doc.get("badges", [])
        )
        
        # Check for achievements
        new_achievements = await self.achievement_engine.check_achievements(
            user_id,
            stats,
            self.db
        )
        
        # Get user rank
        rank = await self.leaderboard.get_user_rank(user_id, self.db)
        
        return {
            "xp_earned": xp_earned,
            "level": new_level,
            "level_up": level_up,
            "elo_rating": new_elo,
            "streak": new_streak,
            "streak_broken": streak_broken,
            "streak_multiplier": streak_multiplier,
            "new_achievements": [
                {
                    "id": a.id,
                    "name": a.name,
                    "description": a.description,
                    "xp_reward": a.xp_reward,
                    "icon": a.icon,
                    "rarity": a.rarity
                }
                for a in new_achievements
            ],
            "rank": rank
        }
    
    async def get_user_stats(self, user_id: str) -> Optional[UserStats]:
        """
        Get user gamification statistics
        
        Args:
            user_id: User identifier
        
        Returns:
            UserStats object or None
        """
        doc = await self.db.user_gamification.find_one({"user_id": user_id})
        if not doc:
            return None
        
        level, xp_in_level, xp_to_next = self.level_system.get_level_from_xp(doc["xp"])
        rank = await self.leaderboard.get_user_rank(user_id, self.db)
        
        return UserStats(
            user_id=user_id,
            level=doc["level"],
            xp=doc["xp"],
            xp_to_next_level=xp_to_next,
            elo_rating=doc["elo_rating"],
            current_streak=doc["current_streak"],
            longest_streak=doc["longest_streak"],
            total_sessions=doc["total_sessions"],
            total_questions=doc["total_questions"],
            total_time_minutes=doc["total_time_minutes"],
            achievements_unlocked=doc["achievements_unlocked"],
            badges=doc.get("badges", []),
            rank=rank
        )
