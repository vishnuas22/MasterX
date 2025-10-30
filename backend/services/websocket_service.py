"""
MasterX WebSocket Service - Real-time bidirectional communication

Features:
- Token authentication
- Session management
- Real-time emotion updates
- Typing indicators
- Message broadcasting
- Connection tracking

Following AGENTS.md:
- No hardcoded values
- Clean code structure
- Proper error handling
- Async/await patterns
"""

import json
import logging
import asyncio
from typing import Dict, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from jose import jwt, JWTError
import os

logger = logging.getLogger(__name__)

# Get JWT settings from environment
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = "HS256"


class ConnectionManager:
    """
    Manages WebSocket connections
    
    Features:
    - Multiple connections per user
    - Session-based message routing
    - Broadcast to all/session/user
    - Connection tracking
    """
    
    def __init__(self):
        # Active connections: {user_id: {connection_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        
        # Session membership: {session_id: {user_id}}
        self.sessions: Dict[str, Set[str]] = {}
        
        # User to sessions mapping: {user_id: {session_id}}
        self.user_sessions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        """
        Register new WebSocket connection
        """
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        
        self.active_connections[user_id][connection_id] = websocket
        
        logger.info(f"✓ WebSocket connected: user={user_id}, conn={connection_id}")
        
        # Send connection confirmation
        await self.send_personal_message(user_id, {
            'type': 'connect',
            'data': {
                'user_id': user_id,
                'connection_id': connection_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
    
    def disconnect(self, user_id: str, connection_id: str):
        """
        Remove WebSocket connection
        """
        if user_id in self.active_connections:
            if connection_id in self.active_connections[user_id]:
                del self.active_connections[user_id][connection_id]
                
                # Remove user if no more connections
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                    
                    # Remove from all sessions
                    if user_id in self.user_sessions:
                        for session_id in self.user_sessions[user_id]:
                            if session_id in self.sessions:
                                self.sessions[session_id].discard(user_id)
                        del self.user_sessions[user_id]
        
        logger.info(f"✗ WebSocket disconnected: user={user_id}, conn={connection_id}")
    
    async def send_personal_message(self, user_id: str, message: Dict[str, Any]):
        """
        Send message to all connections of a specific user
        """
        if user_id not in self.active_connections:
            return
        
        disconnected = []
        
        for connection_id, websocket in self.active_connections[user_id].items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}, conn {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected
        for connection_id in disconnected:
            self.disconnect(user_id, connection_id)
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """
        Send message to all users in a session
        """
        if session_id not in self.sessions:
            return
        
        for user_id in self.sessions[session_id]:
            if exclude_user and user_id == exclude_user:
                continue
            await self.send_personal_message(user_id, message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Send message to all connected users
        """
        for user_id in list(self.active_connections.keys()):
            await self.send_personal_message(user_id, message)
    
    def join_session(self, user_id: str, session_id: str):
        """
        Add user to session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = set()
        
        self.sessions[session_id].add(user_id)
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        
        self.user_sessions[user_id].add(session_id)
        
        logger.info(f"User {user_id} joined session {session_id}")
    
    def leave_session(self, user_id: str, session_id: str):
        """
        Remove user from session
        """
        if session_id in self.sessions:
            self.sessions[session_id].discard(user_id)
            
            # Remove empty session
            if not self.sessions[session_id]:
                del self.sessions[session_id]
        
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
            
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        logger.info(f"User {user_id} left session {session_id}")
    
    def is_connected(self, user_id: str) -> bool:
        """
        Check if user has any active connections
        """
        return user_id in self.active_connections and len(self.active_connections[user_id]) > 0
    
    def get_session_users(self, session_id: str) -> Set[str]:
        """
        Get all users in a session
        """
        return self.sessions.get(session_id, set())


# Global connection manager instance
manager = ConnectionManager()


def verify_token(token: str) -> Optional[str]:
    """
    Verify JWT token and return user_id
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None


async def handle_websocket_message(user_id: str, data: Dict[str, Any]):
    """
    Handle incoming WebSocket message based on type
    """
    message_type = data.get('type')
    message_data = data.get('data', {})
    
    if message_type == 'join_session':
        # Join a chat session
        session_id = message_data.get('session_id')
        if session_id:
            manager.join_session(user_id, session_id)
            
            # Notify user
            await manager.send_personal_message(user_id, {
                'type': 'session_joined',
                'data': {'session_id': session_id}
            })
            
            # Notify other session members
            await manager.send_to_session(session_id, {
                'type': 'user_joined',
                'data': {'user_id': user_id, 'session_id': session_id}
            }, exclude_user=user_id)
    
    elif message_type == 'leave_session':
        # Leave a chat session
        session_id = message_data.get('session_id')
        if session_id:
            manager.leave_session(user_id, session_id)
            
            # Notify other session members
            await manager.send_to_session(session_id, {
                'type': 'user_left',
                'data': {'user_id': user_id, 'session_id': session_id}
            })
    
    elif message_type == 'user_typing':
        # User is typing indicator
        session_id = message_data.get('session_id')
        is_typing = message_data.get('isTyping', False)
        
        if session_id:
            await manager.send_to_session(session_id, {
                'type': 'typing_indicator',
                'data': {
                    'user_id': user_id,
                    'isTyping': is_typing
                }
            }, exclude_user=user_id)
    
    elif message_type == 'message_sent':
        # Message sent notification (for multi-device sync)
        session_id = message_data.get('sessionId')
        if session_id:
            await manager.send_to_session(session_id, {
                'type': 'session_update',
                'data': {
                    'session_id': session_id,
                    'user_id': user_id,
                    'action': 'message_sent'
                }
            }, exclude_user=user_id)
    
    elif message_type == 'ping':
        # Heartbeat response
        await manager.send_personal_message(user_id, {
            'type': 'pong',
            'data': {'timestamp': datetime.utcnow().isoformat()}
        })
    
    else:
        logger.warning(f"Unknown message type: {message_type}")


async def send_emotion_update(user_id: str, message_id: str, emotion_data: Dict[str, Any]):
    """
    Send real-time emotion update to user
    
    Called by chat endpoint after emotion detection
    """
    if not manager.is_connected(user_id):
        return
    
    await manager.send_personal_message(user_id, {
        'type': 'emotion_update',
        'data': {
            'message_id': message_id,
            'emotion': emotion_data
        }
    })
    
    logger.info(f"Sent emotion update to user {user_id}: {emotion_data.get('primary_emotion')}")


async def send_message_received(session_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
    """
    Broadcast new message to session participants
    """
    await manager.send_to_session(session_id, {
        'type': 'message_received',
        'data': {'message': message}
    }, exclude_user=exclude_user)


async def send_notification(user_id: str, notification: Dict[str, Any]):
    """
    Send notification to user
    """
    await manager.send_personal_message(user_id, {
        'type': 'notification',
        'data': notification
    })


# Export connection manager for use in routes
__all__ = [
    'manager',
    'verify_token',
    'handle_websocket_message',
    'send_emotion_update',
    'send_message_received',
    'send_notification'
]
