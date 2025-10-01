"""
Session Manager - Provider Consistency & Topic Tracking
Following specifications from 4.DYNAMIC_AI_ROUTING_SYSTEM.md

Maintains provider consistency within topics/sessions:
- Tracks current provider per session
- Tracks current topic per session
- Prevents unnecessary provider switching
- Stores session metadata in MongoDB
"""

import logging
from typing import Optional, Dict
from datetime import datetime
from utils.database import get_database

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manage provider assignments per session
    
    Key responsibilities:
    - Track which provider is assigned to each session
    - Track current topic/category for session
    - Maintain topic history
    - Prevent unnecessary provider switches
    """
    
    def __init__(self):
        self.db = None
        self.sessions_collection = None
        logger.info("✅ SessionManager initialized")
    
    async def initialize_db(self):
        """Initialize database connection"""
        if not self.db:
            self.db = get_database()
            self.sessions_collection = self.db['sessions']
            logger.info("✅ SessionManager database initialized")
    
    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get current session information
        
        Args:
            session_id: Session UUID
        
        Returns:
            Dict with session info or None if not found
            {
                'session_id': str,
                'current_provider': str,
                'current_topic': str,
                'topic_history': List[str],
                'provider_history': List[str],
                'updated_at': datetime
            }
        """
        
        await self.initialize_db()
        
        try:
            session = await self.sessions_collection.find_one({'_id': session_id})
            
            if session:
                return {
                    'session_id': session['_id'],
                    'current_provider': session.get('assigned_provider'),
                    'current_topic': session.get('current_topic'),
                    'topic_history': session.get('topic_history', []),
                    'provider_history': session.get('provider_history', []),
                    'updated_at': session.get('updated_at')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return None
    
    async def update_session(
        self,
        session_id: str,
        provider: str,
        topic: str
    ):
        """
        Update session with provider and topic
        
        Args:
            session_id: Session UUID
            provider: Selected AI provider name
            topic: Detected topic/category
        """
        
        await self.initialize_db()
        
        try:
            # Get existing session to track history
            existing = await self.sessions_collection.find_one({'_id': session_id})
            
            if existing:
                # Update existing session
                update_data = {
                    '$set': {
                        'assigned_provider': provider,
                        'current_topic': topic,
                        'updated_at': datetime.utcnow()
                    }
                }
                
                # Add to history if different from current
                if existing.get('assigned_provider') != provider:
                    update_data['$addToSet'] = {
                        'provider_history': provider
                    }
                
                if existing.get('current_topic') != topic:
                    if '$addToSet' not in update_data:
                        update_data['$addToSet'] = {}
                    update_data['$addToSet']['topic_history'] = topic
                
                await self.sessions_collection.update_one(
                    {'_id': session_id},
                    update_data
                )
                
                logger.info(f"📝 Session {session_id[:8]} updated: provider={provider}, topic={topic}")
            
            else:
                # Session doesn't exist yet, just log (it will be created by server.py)
                logger.warning(f"⚠️ Session {session_id[:8]} not found in database")
        
        except Exception as e:
            logger.error(f"Error updating session: {e}")
    
    async def get_session_provider(self, session_id: str) -> Optional[str]:
        """
        Get current provider for session
        
        Args:
            session_id: Session UUID
        
        Returns:
            Provider name or None
        """
        
        session_info = await self.get_session_info(session_id)
        
        if session_info:
            return session_info.get('current_provider')
        
        return None
    
    async def get_session_topic(self, session_id: str) -> Optional[str]:
        """
        Get current topic for session
        
        Args:
            session_id: Session UUID
        
        Returns:
            Topic/category name or None
        """
        
        session_info = await self.get_session_info(session_id)
        
        if session_info:
            return session_info.get('current_topic')
        
        return None
    
    async def detect_topic_change(
        self,
        session_id: str,
        new_topic: str
    ) -> bool:
        """
        Detect if topic has changed from current session topic
        
        Args:
            session_id: Session UUID
            new_topic: Newly detected topic
        
        Returns:
            True if topic changed, False otherwise
        """
        
        current_topic = await self.get_session_topic(session_id)
        
        if not current_topic:
            # No current topic means this is first message
            return True
        
        return current_topic != new_topic
    
    async def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """
        Get statistics for a session
        
        Args:
            session_id: Session UUID
        
        Returns:
            Dict with session statistics
        """
        
        session_info = await self.get_session_info(session_id)
        
        if not session_info:
            return None
        
        return {
            'session_id': session_id,
            'providers_used': len(session_info.get('provider_history', [])),
            'topics_covered': len(session_info.get('topic_history', [])),
            'current_provider': session_info.get('current_provider'),
            'current_topic': session_info.get('current_topic'),
            'last_updated': session_info.get('updated_at')
        }
