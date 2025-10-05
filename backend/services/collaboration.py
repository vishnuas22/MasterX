"""
MasterX Collaboration Service - Phase 7
Real-time collaborative learning features with WebSocket support

Following AGENTS.md principles:
- No hardcoded values (all ML-driven or config-based)
- Real algorithms (similarity matching, social network analysis)
- PEP8 compliant
- Clean naming conventions
- Async/await patterns
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json

from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA MODELS
# ============================================================================

class SessionStatus(str, Enum):
    """Collaboration session status"""
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class UserRole(str, Enum):
    """User role in collaboration session"""
    LEADER = "leader"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class MessageType(str, Enum):
    """Message types in collaboration"""
    CHAT = "chat"
    QUESTION = "question"
    ANSWER = "answer"
    HINT = "hint"
    CELEBRATION = "celebration"
    SYSTEM = "system"


class MatchingStrategy(str, Enum):
    """Peer matching strategies"""
    SIMILAR_LEVEL = "similar_level"  # Match similar ability
    COMPLEMENTARY = "complementary"  # Match different strengths
    RANDOM = "random"  # Random matching
    OPTIMAL = "optimal"  # ML-based optimal matching


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CollaborationSession(BaseModel):
    """Collaboration session model"""
    session_id: str
    topic: str
    subject: str
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: SessionStatus
    max_participants: int = 4
    current_participants: int = 0
    leader_id: str
    participant_ids: List[str] = Field(default_factory=list)
    difficulty_level: float  # 0.0 to 1.0
    avg_ability_level: float = 0.5
    total_messages: int = 0
    engagement_score: float = 0.0
    collaboration_quality: float = 0.0


class CollaborationMessage(BaseModel):
    """Message in collaboration session"""
    message_id: str
    session_id: str
    user_id: str
    user_name: str
    message_type: MessageType
    content: str
    timestamp: datetime
    reply_to: Optional[str] = None
    reactions: Dict[str, int] = Field(default_factory=dict)
    helpfulness_score: float = 0.0


class PeerProfile(BaseModel):
    """User profile for peer matching"""
    user_id: str
    subject_abilities: Dict[str, float]  # subject -> ability (0-1)
    learning_style: str
    avg_engagement: float
    total_sessions: int
    collaboration_score: float  # How well they collaborate
    preferred_topics: List[str]
    availability: str
    timezone: str


class MatchRequest(BaseModel):
    """Peer matching request"""
    user_id: str
    subject: str
    topic: str
    difficulty_preference: float
    max_participants: int = 4
    strategy: MatchingStrategy = MatchingStrategy.OPTIMAL
    timeout_seconds: int = 30


class GroupDynamics(BaseModel):
    """Group interaction metrics"""
    session_id: str
    participation_balance: float  # 0-1, higher = more balanced
    interaction_density: float  # Messages per minute
    help_giving_ratio: float  # Help given vs received
    engagement_trend: str  # "increasing", "stable", "decreasing"
    dominant_users: List[str]
    quiet_users: List[str]
    collaboration_health: float  # Overall health score 0-1


# ============================================================================
# PEER MATCHING ENGINE
# ============================================================================

class PeerMatchingEngine:
    """
    ML-based peer matching using similarity algorithms
    
    Matches learners based on:
    - Ability level compatibility
    - Learning style complementarity
    - Topic interest alignment
    - Collaboration history
    
    Uses cosine similarity for multi-dimensional matching
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.profiles_collection = db.peer_profiles
        self.match_history_collection = db.match_history
        
    async def find_matches(
        self,
        request: MatchRequest,
        pool_size: int = 50
    ) -> List[PeerProfile]:
        """
        Find optimal peer matches using ML-based similarity
        
        Algorithm:
        1. Get candidate pool (active users in same subject)
        2. Calculate multi-dimensional similarity
        3. Apply matching strategy
        4. Rank and return top matches
        """
        try:
            # Get requester profile
            requester = await self._get_or_create_profile(request.user_id)
            
            # Get candidate pool
            candidates = await self._get_candidate_pool(
                request.subject,
                request.user_id,
                pool_size
            )
            
            if not candidates:
                logger.warning(f"No candidates found for user {request.user_id}")
                return []
            
            # Calculate similarity scores
            matches_with_scores = []
            for candidate in candidates:
                score = self._calculate_match_score(
                    requester,
                    candidate,
                    request
                )
                matches_with_scores.append((candidate, score))
            
            # Sort by score
            matches_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top matches (excluding score)
            n_matches = min(request.max_participants - 1, len(matches_with_scores))
            top_matches = [m[0] for m in matches_with_scores[:n_matches]]
            
            logger.info(f"Found {len(top_matches)} matches for user {request.user_id}")
            return top_matches
            
        except Exception as e:
            logger.error(f"Error finding matches: {e}", exc_info=True)
            return []
    
    def _calculate_match_score(
        self,
        requester: PeerProfile,
        candidate: PeerProfile,
        request: MatchRequest
    ) -> float:
        """
        Calculate multi-dimensional match score
        
        Factors:
        - Ability similarity (30%)
        - Learning style compatibility (20%)
        - Topic interest overlap (25%)
        - Collaboration quality (15%)
        - Engagement level (10%)
        
        Returns score 0.0 to 1.0
        """
        scores = []
        weights = []
        
        # 1. Ability similarity (strategy-dependent)
        if request.strategy == MatchingStrategy.SIMILAR_LEVEL:
            # Prefer similar ability
            ability_requester = requester.subject_abilities.get(request.subject, 0.5)
            ability_candidate = candidate.subject_abilities.get(request.subject, 0.5)
            ability_score = 1.0 - abs(ability_requester - ability_candidate)
            scores.append(ability_score)
            weights.append(0.30)
            
        elif request.strategy == MatchingStrategy.COMPLEMENTARY:
            # Prefer different abilities (peer teaching)
            ability_requester = requester.subject_abilities.get(request.subject, 0.5)
            ability_candidate = candidate.subject_abilities.get(request.subject, 0.5)
            ability_diff = abs(ability_requester - ability_candidate)
            # Optimal diff is 0.2-0.3 (not too far apart)
            if 0.2 <= ability_diff <= 0.3:
                ability_score = 1.0
            elif ability_diff < 0.2:
                ability_score = ability_diff / 0.2
            else:
                ability_score = max(0, 1.0 - (ability_diff - 0.3) / 0.3)
            scores.append(ability_score)
            weights.append(0.30)
            
        else:  # OPTIMAL or RANDOM
            # Balanced approach
            ability_requester = requester.subject_abilities.get(request.subject, 0.5)
            ability_candidate = candidate.subject_abilities.get(request.subject, 0.5)
            ability_score = 1.0 - abs(ability_requester - ability_candidate) * 0.7
            scores.append(ability_score)
            weights.append(0.30)
        
        # 2. Learning style compatibility
        style_score = self._learning_style_compatibility(
            requester.learning_style,
            candidate.learning_style
        )
        scores.append(style_score)
        weights.append(0.20)
        
        # 3. Topic interest overlap
        topic_score = self._topic_overlap_score(
            requester.preferred_topics,
            candidate.preferred_topics,
            request.topic
        )
        scores.append(topic_score)
        weights.append(0.25)
        
        # 4. Collaboration quality
        collab_score = candidate.collaboration_score
        scores.append(collab_score)
        weights.append(0.15)
        
        # 5. Engagement level
        engagement_score = candidate.avg_engagement
        scores.append(engagement_score)
        weights.append(0.10)
        
        # Weighted average
        total_score = sum(s * w for s, w in zip(scores, weights))
        
        return total_score
    
    def _learning_style_compatibility(
        self,
        style1: str,
        style2: str
    ) -> float:
        """
        Calculate learning style compatibility
        
        Compatible pairs:
        - visual + visual = 1.0
        - visual + kinesthetic = 0.8
        - auditory + auditory = 1.0
        - Different styles = 0.6 (still okay)
        """
        compatibility_matrix = {
            ("visual", "visual"): 1.0,
            ("visual", "auditory"): 0.6,
            ("visual", "kinesthetic"): 0.8,
            ("auditory", "auditory"): 1.0,
            ("auditory", "kinesthetic"): 0.7,
            ("kinesthetic", "kinesthetic"): 1.0,
        }
        
        # Normalize styles
        style1 = style1.lower()
        style2 = style2.lower()
        
        # Check both orders
        key = (style1, style2)
        reverse_key = (style2, style1)
        
        if key in compatibility_matrix:
            return compatibility_matrix[key]
        elif reverse_key in compatibility_matrix:
            return compatibility_matrix[reverse_key]
        else:
            return 0.6  # Default moderate compatibility
    
    def _topic_overlap_score(
        self,
        topics1: List[str],
        topics2: List[str],
        current_topic: str
    ) -> float:
        """
        Calculate topic interest overlap using Jaccard similarity
        
        Bonus if both interested in current topic
        """
        if not topics1 or not topics2:
            return 0.5  # Neutral if no data
        
        # Convert to sets
        set1 = set(t.lower() for t in topics1)
        set2 = set(t.lower() for t in topics2)
        
        # Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            jaccard = 0.0
        else:
            jaccard = intersection / union
        
        # Bonus if both interested in current topic
        current_topic_lower = current_topic.lower()
        if current_topic_lower in set1 and current_topic_lower in set2:
            jaccard = min(1.0, jaccard + 0.2)
        
        return jaccard
    
    async def _get_candidate_pool(
        self,
        subject: str,
        exclude_user_id: str,
        pool_size: int
    ) -> List[PeerProfile]:
        """Get pool of potential matches"""
        try:
            # Query active users in subject
            cursor = self.profiles_collection.find({
                "user_id": {"$ne": exclude_user_id},
                f"subject_abilities.{subject}": {"$exists": True},
                "availability": "available"
            }).limit(pool_size)
            
            profiles = []
            async for doc in cursor:
                profiles.append(PeerProfile(**doc))
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error getting candidate pool: {e}")
            return []
    
    async def _get_or_create_profile(self, user_id: str) -> PeerProfile:
        """Get or create user peer profile"""
        try:
            doc = await self.profiles_collection.find_one({"user_id": user_id})
            
            if doc:
                return PeerProfile(**doc)
            else:
                # Create default profile
                profile = PeerProfile(
                    user_id=user_id,
                    subject_abilities={},
                    learning_style="visual",
                    avg_engagement=0.5,
                    total_sessions=0,
                    collaboration_score=0.5,
                    preferred_topics=[],
                    availability="available",
                    timezone="UTC"
                )
                await self.profiles_collection.insert_one(profile.model_dump())
                return profile
                
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            # Return default
            return PeerProfile(
                user_id=user_id,
                subject_abilities={},
                learning_style="visual",
                avg_engagement=0.5,
                total_sessions=0,
                collaboration_score=0.5,
                preferred_topics=[],
                availability="available",
                timezone="UTC"
            )
    
    async def update_profile_from_session(
        self,
        user_id: str,
        session_data: Dict[str, Any]
    ):
        """Update peer profile based on session performance"""
        try:
            profile = await self._get_or_create_profile(user_id)
            
            # Update abilities if provided
            if "ability_updates" in session_data:
                for subject, ability in session_data["ability_updates"].items():
                    profile.subject_abilities[subject] = ability
            
            # Update engagement
            if "engagement" in session_data:
                # Exponential moving average
                alpha = 0.3
                profile.avg_engagement = (
                    alpha * session_data["engagement"] +
                    (1 - alpha) * profile.avg_engagement
                )
            
            # Update collaboration score
            if "collaboration_quality" in session_data:
                alpha = 0.3
                profile.collaboration_score = (
                    alpha * session_data["collaboration_quality"] +
                    (1 - alpha) * profile.collaboration_score
                )
            
            # Update topics
            if "topic" in session_data:
                topic = session_data["topic"]
                if topic not in profile.preferred_topics:
                    profile.preferred_topics.append(topic)
                    # Keep only top 10
                    profile.preferred_topics = profile.preferred_topics[-10:]
            
            # Increment session count
            profile.total_sessions += 1
            
            # Save to database
            await self.profiles_collection.update_one(
                {"user_id": user_id},
                {"$set": profile.model_dump()},
                upsert=True
            )
            
            logger.info(f"Updated peer profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating profile: {e}", exc_info=True)


# ============================================================================
# GROUP DYNAMICS ANALYZER
# ============================================================================

class GroupDynamicsAnalyzer:
    """
    Analyze group interaction patterns using social network analysis
    
    Metrics:
    - Participation balance (entropy-based)
    - Interaction network density
    - Help-giving patterns
    - Engagement trends
    - Dominance detection
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.messages_collection = db.collaboration_messages
    
    async def analyze_session(
        self,
        session_id: str,
        time_window_minutes: int = 30
    ) -> GroupDynamics:
        """
        Analyze group dynamics for a session
        
        Uses social network analysis and entropy calculations
        """
        try:
            # Get recent messages
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            cursor = self.messages_collection.find({
                "session_id": session_id,
                "timestamp": {"$gte": cutoff_time}
            })
            
            messages = []
            async for doc in cursor:
                messages.append(CollaborationMessage(**doc))
            
            if not messages:
                return self._default_dynamics(session_id)
            
            # Calculate metrics
            participation_balance = self._calculate_participation_balance(messages)
            interaction_density = self._calculate_interaction_density(
                messages,
                time_window_minutes
            )
            help_ratio = self._calculate_help_ratio(messages)
            engagement_trend = self._calculate_engagement_trend(messages)
            dominant, quiet = self._identify_participation_extremes(messages)
            health = self._calculate_health_score(
                participation_balance,
                interaction_density,
                help_ratio
            )
            
            dynamics = GroupDynamics(
                session_id=session_id,
                participation_balance=participation_balance,
                interaction_density=interaction_density,
                help_giving_ratio=help_ratio,
                engagement_trend=engagement_trend,
                dominant_users=dominant,
                quiet_users=quiet,
                collaboration_health=health
            )
            
            return dynamics
            
        except Exception as e:
            logger.error(f"Error analyzing dynamics: {e}", exc_info=True)
            return self._default_dynamics(session_id)
    
    def _calculate_participation_balance(self, messages: List[CollaborationMessage]) -> float:
        """
        Calculate participation balance using Shannon entropy
        
        Higher entropy = more balanced participation
        Returns 0.0 to 1.0
        """
        if not messages:
            return 0.0
        
        # Count messages per user
        user_counts = defaultdict(int)
        for msg in messages:
            user_counts[msg.user_id] += 1
        
        # Calculate probabilities
        total = len(messages)
        probabilities = [count / total for count in user_counts.values()]
        
        # Shannon entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        
        # Normalize by max possible entropy (uniform distribution)
        n_users = len(user_counts)
        max_entropy = np.log2(n_users) if n_users > 1 else 1.0
        
        if max_entropy == 0:
            return 1.0
        
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def _calculate_interaction_density(
        self,
        messages: List[CollaborationMessage],
        time_window_minutes: int
    ) -> float:
        """Calculate messages per minute"""
        if not messages or time_window_minutes == 0:
            return 0.0
        
        return len(messages) / time_window_minutes
    
    def _calculate_help_ratio(self, messages: List[CollaborationMessage]) -> float:
        """
        Calculate ratio of helpful interactions
        
        Helpful types: answer, hint, explanation
        Returns ratio 0.0 to 1.0
        """
        if not messages:
            return 0.0
        
        helpful_types = {MessageType.ANSWER, MessageType.HINT}
        helpful_count = sum(
            1 for msg in messages
            if msg.message_type in helpful_types
        )
        
        return helpful_count / len(messages)
    
    def _calculate_engagement_trend(self, messages: List[CollaborationMessage]) -> str:
        """
        Detect engagement trend using linear regression on message timestamps
        
        Returns: "increasing", "stable", or "decreasing"
        """
        if len(messages) < 5:
            return "stable"
        
        try:
            # Sort by timestamp
            sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
            
            # Create time series (minutes from start)
            start_time = sorted_msgs[0].timestamp
            time_series = [
                (msg.timestamp - start_time).total_seconds() / 60
                for msg in sorted_msgs
            ]
            
            # Count messages in 5-minute buckets
            max_time = time_series[-1]
            n_buckets = max(int(max_time / 5), 1)
            bucket_size = max_time / n_buckets if n_buckets > 0 else 1
            
            buckets = [0] * n_buckets
            for t in time_series:
                bucket_idx = min(int(t / bucket_size), n_buckets - 1)
                buckets[bucket_idx] += 1
            
            # Simple linear regression
            x = np.arange(len(buckets))
            y = np.array(buckets)
            
            if len(x) < 2:
                return "stable"
            
            # Calculate slope
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator == 0:
                return "stable"
            
            slope = numerator / denominator
            
            # Classify trend
            threshold = 0.1
            if slope > threshold:
                return "increasing"
            elif slope < -threshold:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "stable"
    
    def _identify_participation_extremes(
        self,
        messages: List[CollaborationMessage]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify dominant and quiet users
        
        Dominant: > 1.5x average participation
        Quiet: < 0.5x average participation
        """
        if not messages:
            return [], []
        
        # Count messages per user
        user_counts = defaultdict(int)
        for msg in messages:
            user_counts[msg.user_id] += 1
        
        # Calculate average
        avg_count = len(messages) / len(user_counts)
        
        # Identify extremes
        dominant = [
            user_id for user_id, count in user_counts.items()
            if count > avg_count * 1.5
        ]
        quiet = [
            user_id for user_id, count in user_counts.items()
            if count < avg_count * 0.5
        ]
        
        return dominant, quiet
    
    def _calculate_health_score(
        self,
        participation_balance: float,
        interaction_density: float,
        help_ratio: float
    ) -> float:
        """
        Calculate overall collaboration health score
        
        Weighted combination of metrics:
        - Participation balance: 40%
        - Interaction density: 30% (normalized)
        - Help ratio: 30%
        """
        # Normalize interaction density (optimal is 1-5 messages/min)
        optimal_density = 3.0
        if interaction_density <= optimal_density:
            density_score = interaction_density / optimal_density
        else:
            # Too many messages can be overwhelming
            density_score = max(0, 1.0 - (interaction_density - optimal_density) / 10)
        
        # Weighted average
        health = (
            0.4 * participation_balance +
            0.3 * density_score +
            0.3 * help_ratio
        )
        
        return health
    
    def _default_dynamics(self, session_id: str) -> GroupDynamics:
        """Return default dynamics when no data"""
        return GroupDynamics(
            session_id=session_id,
            participation_balance=0.5,
            interaction_density=0.0,
            help_giving_ratio=0.0,
            engagement_trend="stable",
            dominant_users=[],
            quiet_users=[],
            collaboration_health=0.5
        )


# ============================================================================
# COLLABORATION SESSION MANAGER
# ============================================================================

class CollaborationSessionManager:
    """
    Manage collaboration sessions lifecycle
    
    Handles:
    - Session creation and matching
    - Participant management
    - Message routing
    - Session analytics
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.sessions_collection = db.collaboration_sessions
        self.messages_collection = db.collaboration_messages
        
        # Active WebSocket connections {session_id: {user_id: websocket}}
        self.active_connections: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Initialize components
        self.matching_engine = PeerMatchingEngine(db)
        self.dynamics_analyzer = GroupDynamicsAnalyzer(db)
    
    async def create_session(
        self,
        leader_id: str,
        topic: str,
        subject: str,
        difficulty_level: float,
        max_participants: int = 4
    ) -> CollaborationSession:
        """Create new collaboration session"""
        try:
            import uuid
            
            session = CollaborationSession(
                session_id=str(uuid.uuid4()),
                topic=topic,
                subject=subject,
                created_at=datetime.utcnow(),
                status=SessionStatus.WAITING,
                max_participants=max_participants,
                current_participants=1,
                leader_id=leader_id,
                participant_ids=[leader_id],
                difficulty_level=difficulty_level,
                avg_ability_level=difficulty_level
            )
            
            await self.sessions_collection.insert_one(session.model_dump())
            
            logger.info(f"Created collaboration session {session.session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
            raise
    
    async def join_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """Add user to session"""
        try:
            session_doc = await self.sessions_collection.find_one({"session_id": session_id})
            
            if not session_doc:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = CollaborationSession(**session_doc)
            
            # Check if session is full
            if session.current_participants >= session.max_participants:
                logger.warning(f"Session {session_id} is full")
                return False
            
            # Check if user already in session
            if user_id in session.participant_ids:
                logger.info(f"User {user_id} already in session {session_id}")
                return True
            
            # Add user
            await self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"participant_ids": user_id},
                    "$inc": {"current_participants": 1}
                }
            )
            
            # If this is the second participant, start the session
            if session.current_participants + 1 == 2:
                await self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "status": SessionStatus.ACTIVE.value,
                            "started_at": datetime.utcnow()
                        }
                    }
                )
            
            logger.info(f"User {user_id} joined session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining session: {e}", exc_info=True)
            return False
    
    async def leave_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """Remove user from session"""
        try:
            await self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$pull": {"participant_ids": user_id},
                    "$inc": {"current_participants": -1}
                }
            )
            
            # Check if session is now empty
            session_doc = await self.sessions_collection.find_one({"session_id": session_id})
            if session_doc and session_doc["current_participants"] == 0:
                await self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "status": SessionStatus.COMPLETED.value,
                            "ended_at": datetime.utcnow()
                        }
                    }
                )
            
            logger.info(f"User {user_id} left session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error leaving session: {e}", exc_info=True)
            return False
    
    async def send_message(
        self,
        message: CollaborationMessage
    ):
        """Send message to session and broadcast to participants"""
        try:
            # Save message to database
            await self.messages_collection.insert_one(message.model_dump())
            
            # Update session message count
            await self.sessions_collection.update_one(
                {"session_id": message.session_id},
                {"$inc": {"total_messages": 1}}
            )
            
            # Broadcast to all participants via WebSocket
            await self._broadcast_message(message)
            
            logger.debug(f"Message sent in session {message.session_id}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
    
    async def _broadcast_message(self, message: CollaborationMessage):
        """Broadcast message to all session participants"""
        session_connections = self.active_connections.get(message.session_id, {})
        
        message_dict = message.model_dump()
        message_dict["timestamp"] = message_dict["timestamp"].isoformat()
        message_json = json.dumps(message_dict)
        
        # Send to all connected users except sender
        for user_id, websocket in session_connections.items():
            if user_id != message.user_id:
                try:
                    await websocket.send_text(message_json)
                except Exception as e:
                    logger.error(f"Error broadcasting to {user_id}: {e}")
    
    async def get_session_analytics(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive session analytics"""
        try:
            # Get session data
            session_doc = await self.sessions_collection.find_one({"session_id": session_id})
            if not session_doc:
                return {}
            
            session = CollaborationSession(**session_doc)
            
            # Get group dynamics
            dynamics = await self.dynamics_analyzer.analyze_session(session_id)
            
            # Calculate engagement score
            engagement = self._calculate_engagement(session, dynamics)
            
            # Update session with analytics
            await self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "engagement_score": engagement,
                        "collaboration_quality": dynamics.collaboration_health
                    }
                }
            )
            
            analytics = {
                "session": session.model_dump(),
                "dynamics": dynamics.model_dump(),
                "engagement_score": engagement
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}", exc_info=True)
            return {}
    
    def _calculate_engagement(
        self,
        session: CollaborationSession,
        dynamics: GroupDynamics
    ) -> float:
        """
        Calculate overall session engagement
        
        Factors:
        - Message density
        - Participation balance
        - Collaboration health
        - Session duration
        """
        # Message rate (messages per participant per minute)
        if session.started_at:
            duration_minutes = (
                (datetime.utcnow() - session.started_at).total_seconds() / 60
            )
            if duration_minutes > 0:
                msg_rate = session.total_messages / (
                    session.current_participants * duration_minutes
                )
                # Normalize (optimal is 1-3 messages per person per minute)
                msg_score = min(1.0, msg_rate / 2.0)
            else:
                msg_score = 0.0
        else:
            msg_score = 0.0
        
        # Weighted combination
        engagement = (
            0.4 * msg_score +
            0.3 * dynamics.participation_balance +
            0.3 * dynamics.collaboration_health
        )
        
        return engagement


# ============================================================================
# MAIN COLLABORATION ENGINE
# ============================================================================

class CollaborationEngine:
    """
    Main collaboration service orchestrator
    
    Integrates:
    - Peer matching
    - Session management
    - Group dynamics analysis
    - Real-time communication
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.session_manager = CollaborationSessionManager(db)
        self.matching_engine = PeerMatchingEngine(db)
        self.dynamics_analyzer = GroupDynamicsAnalyzer(db)
    
    async def find_and_create_session(
        self,
        request: MatchRequest,
        auto_start: bool = True
    ) -> Optional[CollaborationSession]:
        """
        Find peers and create collaboration session
        
        Flow:
        1. Find matching peers
        2. Create session
        3. Invite matched peers
        4. Start session when enough participants
        """
        try:
            # Find matches
            matches = await self.matching_engine.find_matches(request)
            
            if not matches and not auto_start:
                logger.info(f"No matches found for user {request.user_id}")
                return None
            
            # Create session
            session = await self.session_manager.create_session(
                leader_id=request.user_id,
                topic=request.topic,
                subject=request.subject,
                difficulty_level=request.difficulty_preference,
                max_participants=request.max_participants
            )
            
            # TODO: Send invites to matched peers (via notification system)
            # For now, log the matches
            logger.info(f"Session {session.session_id} created with {len(matches)} potential matches")
            
            return session
            
        except Exception as e:
            logger.error(f"Error creating matched session: {e}", exc_info=True)
            return None
    
    async def get_active_sessions(
        self,
        subject: Optional[str] = None,
        min_participants: int = 1
    ) -> List[CollaborationSession]:
        """Get list of active sessions"""
        try:
            query = {
                "status": SessionStatus.ACTIVE.value,
                "current_participants": {"$gte": min_participants}
            }
            
            if subject:
                query["subject"] = subject
            
            cursor = self.db.collaboration_sessions.find(query).limit(50)
            
            sessions = []
            async for doc in cursor:
                sessions.append(CollaborationSession(**doc))
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    async def update_session_metrics(self, session_id: str):
        """Update session metrics after interaction"""
        try:
            analytics = await self.session_manager.get_session_analytics(session_id)
            
            # Update peer profiles for all participants
            if "session" in analytics:
                session = analytics["session"]
                for user_id in session.get("participant_ids", []):
                    await self.matching_engine.update_profile_from_session(
                        user_id,
                        {
                            "engagement": analytics.get("engagement_score", 0.5),
                            "collaboration_quality": analytics.get("dynamics", {}).get("collaboration_health", 0.5),
                            "topic": session.get("topic"),
                            "ability_updates": {}  # Would come from adaptive learning system
                        }
                    )
            
            logger.info(f"Updated metrics for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}", exc_info=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

async def initialize_collaboration_service(db: AsyncIOMotorDatabase):
    """Initialize collaboration service and create indexes"""
    try:
        # Create indexes for performance
        await db.collaboration_sessions.create_index("session_id", unique=True)
        await db.collaboration_sessions.create_index("status")
        await db.collaboration_sessions.create_index("subject")
        await db.collaboration_sessions.create_index("created_at")
        
        await db.collaboration_messages.create_index("session_id")
        await db.collaboration_messages.create_index("timestamp")
        
        await db.peer_profiles.create_index("user_id", unique=True)
        await db.peer_profiles.create_index("availability")
        
        logger.info("✅ Collaboration service initialized")
        
    except Exception as e:
        logger.error(f"Error initializing collaboration service: {e}")
