import sys
import os
from pathlib import Path
import uuid

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

# Import our models and services
from models import (
    User, UserCreate, ChatSession, SessionCreate, ChatMessage, MessageCreate,
    MentorRequest, MentorResponse, Exercise, ExerciseSubmission, LearningProgress,
    MetacognitiveSessionModel, MemoryPalaceModel, ElaborativeQuestionModel, 
    TransferScenarioModel, LearningPsychologyProgressModel,
    MetacognitiveSessionRequest, MemoryPalaceRequest, ElaborativeQuestionRequest,
    TransferScenarioRequest, LearningPsychologyResponse
)
from database import db_service
from ai_service import ai_service
from premium_ai_service import premium_ai_service
from model_manager import premium_model_manager
from gamification_service import gamification_service
from advanced_streaming_service import advanced_streaming_service
from advanced_context_service import advanced_context_service
from live_learning_service import live_learning_service, SessionType
from learning_psychology_service import learning_psychology_service
from personalization_engine import personalization_engine, LearningDNA, AdaptiveContentParameters, MoodBasedAdaptation
from adaptive_ai_service import adaptive_ai_service
from personal_learning_assistant import personal_assistant, LearningGoal, LearningMemory, PersonalInsight, GoalType, GoalStatus, MemoryType
from advanced_analytics_service import advanced_analytics_service, LearningEvent

ROOT_DIR = backend_dir
load_dotenv(ROOT_DIR / '.env')

# Initialize advanced analytics service as a global instance
if not hasattr(advanced_analytics_service, 'knowledge_graph'):
    from advanced_analytics_service import AdvancedAnalyticsService
    advanced_analytics_service = AdvancedAnalyticsService()

# Create the main app with optimized settings
app = FastAPI(
    title="MasterX AI Mentor System",
    description="World-class AI-powered personalized learning platform",
    version="1.0.0",
    docs_url="/docs" if os.getenv("DEBUG", "false").lower() == "true" else None,  # Hide docs in production
    redoc_url=None  # Disable redoc for better performance
)

# ⚙️ UNIVERSAL PORTABILITY: Add CORS middleware FIRST for proper cross-origin support
# This ensures localhost:3000 can connect to localhost:8001 during development
# and works seamlessly across all deployment environments
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Allow all origins for universal portability
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging with optimized levels for production
logging.basicConfig(
    level=logging.WARNING,  # Reduced from INFO to reduce log spam
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger('motor').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection and gamification system"""
    try:
        await db_service.connect()
        await gamification_service.initialize_achievements()
        logger.info("MasterX AI Mentor System started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections"""
    await db_service.disconnect()
    logger.info("MasterX AI Mentor System shutdown complete")

# ================================
# USER MANAGEMENT ENDPOINTS
# ================================

@api_router.post("/users", response_model=User)
async def create_user(user_data: UserCreate):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = await db_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")
        
        user = await db_service.create_user(user_data)
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    """Get user by ID"""
    user = await db_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@api_router.get("/users/email/{email}", response_model=User)
async def get_user_by_email(email: str):
    """Get user by email"""
    user = await db_service.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ================================
# SESSION MANAGEMENT ENDPOINTS
# ================================

@api_router.post("/sessions", response_model=ChatSession)
async def create_session(session_data: SessionCreate):
    """Create a new learning session"""
    try:
        # Verify user exists
        user = await db_service.get_user(session_data.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        session = await db_service.create_session(session_data)
        logger.info(f"Created new session {session.id} for user {session_data.user_id}")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@api_router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get session by ID"""
    session = await db_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@api_router.get("/users/{user_id}/sessions", response_model=List[ChatSession])
async def get_user_sessions(user_id: str, active_only: bool = True):
    """Get all sessions for a user"""
    sessions = await db_service.get_user_sessions(user_id, active_only)
    return sessions

@api_router.put("/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a session"""
    success = await db_service.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session ended successfully"}

# ================================
# CHAT MANAGEMENT ENDPOINTS
# ================================

@api_router.put("/sessions/{session_id}/rename")
async def rename_chat_session(session_id: str, request: dict):
    """Rename a chat session"""
    try:
        new_title = request.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        # Get the session first to verify it exists
        session = await db_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update the session title
        success = await db_service.update_session_title(session_id, new_title)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to rename session")
        
        return {
            "message": "Session renamed successfully",
            "session_id": session_id,
            "new_title": new_title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to rename session")

@api_router.post("/sessions/{session_id}/share")
async def share_chat_session(session_id: str, request: dict):
    """Create a shareable link for a chat session"""
    try:
        # Get the session to verify it exists
        session = await db_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get session messages for sharing
        messages = await db_service.get_session_messages(session_id, limit=100)
        
        # Create share data
        share_data = {
            "session_id": session_id,
            "title": session.subject or "MasterX Chat",
            "created_at": session.created_at.isoformat(),
            "user_id": session.user_id,
            "messages": [
                {
                    "sender": msg.sender,
                    "message": msg.message,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        }
        
        # Generate a unique share ID
        import secrets
        share_id = secrets.token_urlsafe(16)
        
        # Store the share data (in production, you'd save this to a database)
        # For now, we'll return the share_id and data
        
        return {
            "message": "Session shared successfully",
            "share_id": share_id,
            "share_url": f"/shared/{share_id}",
            "expires_in": "7 days",
            "data": share_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to share session")

@api_router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages"""
    try:
        # Get the session to verify it exists
        session = await db_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete all messages first
        await db_service.delete_session_messages(session_id)
        
        # Delete the session
        success = await db_service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete session")
        
        return {
            "message": "Session deleted successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@api_router.get("/sessions/{session_id}/search")
async def search_session_messages(session_id: str, query: str, limit: int = 20):
    """Search messages within a specific session"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Get the session to verify it exists
        session = await db_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Search messages (this would be implemented in db_service)
        messages = await db_service.search_session_messages(session_id, query, limit)
        
        return {
            "session_id": session_id,
            "query": query,
            "results": messages,
            "total_found": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search session")

@api_router.get("/users/{user_id}/sessions/search")
async def search_user_sessions(user_id: str, query: str, limit: int = 50):
    """Search across all user sessions"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Search across all user sessions
        results = await db_service.search_user_sessions(user_id, query, limit)
        
        return {
            "user_id": user_id,
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching user sessions for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search sessions")

# ================================
# PREMIUM AI CHAT ENDPOINTS
# ================================

@api_router.post("/chat/premium")
async def premium_chat_with_mentor(request: MentorRequest):
    """Premium chat with advanced learning modes and multi-model AI"""
    try:
        # Get session info
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message
        user_message = await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Get recent messages for context
        recent_messages = await db_service.get_recent_messages(request.session_id, limit=10)
        
        # Prepare context
        context = request.context or {}
        context['recent_messages'] = recent_messages
        
        # Extract learning mode from context
        learning_mode = context.get('learning_mode', 'adaptive')
        
        # Get premium AI response
        mentor_response = await premium_ai_service.get_premium_response(
            user_message=request.user_message,
            session=session,
            context=context,
            learning_mode=learning_mode,
            stream=False
        )
        
        # Save mentor response
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=mentor_response.response,
            sender="mentor",
            message_type=mentor_response.response_type,
            metadata=mentor_response.metadata
        ))
        
        return mentor_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in premium chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process premium chat message")

@api_router.post("/chat/premium/stream")
async def premium_chat_with_mentor_stream(request: MentorRequest):
    """Premium streaming chat with advanced learning modes"""
    try:
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Get context
        recent_messages = await db_service.get_recent_messages(request.session_id, limit=10)
        context = request.context or {}
        context['recent_messages'] = recent_messages
        
        # Extract learning mode
        learning_mode = context.get('learning_mode', 'adaptive')
        
        async def generate_premium_stream():
            try:
                # Get streaming response from premium AI
                stream_response = await premium_ai_service.get_premium_response(
                    user_message=request.user_message,
                    session=session,
                    context=context,
                    learning_mode=learning_mode,
                    stream=True
                )
                
                full_response = ""
                
                for chunk in stream_response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # Send chunk as Server-Sent Event
                        yield f"data: {json.dumps({'content': content, 'type': 'chunk', 'mode': learning_mode})}\n\n"
                        
                        # Small delay for better UX
                        await asyncio.sleep(0.01)
                
                # Save complete response
                if full_response:
                    formatted_response = ai_service._format_response(full_response)
                    await db_service.save_message(MessageCreate(
                        session_id=request.session_id,
                        message=full_response,
                        sender="mentor",
                        message_type=f"premium_{learning_mode}",
                        metadata={
                            **formatted_response.metadata,
                            "learning_mode": learning_mode,
                            "premium_features": True
                        }
                    ))
                
                # Send completion signal with premium features
                yield f"data: {json.dumps({'type': 'complete', 'suggestions': formatted_response.suggested_actions, 'mode': learning_mode, 'next_steps': formatted_response.next_steps})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in premium streaming: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Sorry, I encountered an error. Please try again.'})}\n\n"
        
        return StreamingResponse(
            generate_premium_stream(),
            media_type="text/stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up premium stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to setup premium streaming")

@api_router.post("/chat", response_model=MentorResponse)
async def chat_with_mentor(request: MentorRequest):
    """Send message to AI mentor and get response"""
    try:
        # Get session info
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message
        user_message = await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Get recent messages for context
        recent_messages = await db_service.get_recent_messages(request.session_id, limit=10)
        
        # Prepare context
        context = request.context or {}
        context['recent_messages'] = recent_messages
        
        # Get AI response
        mentor_response = await ai_service.get_mentor_response(
            request.user_message,
            session,
            context
        )
        
        # Save mentor response
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=mentor_response.response,
            sender="mentor",
            message_type=mentor_response.response_type,
            metadata=mentor_response.metadata
        ))
        
        return mentor_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

@api_router.post("/chat/stream")
async def chat_with_mentor_stream(request: MentorRequest):
    """Stream real-time response from AI mentor"""
    try:
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Get context
        recent_messages = await db_service.get_recent_messages(request.session_id, limit=10)
        context = request.context or {}
        context['recent_messages'] = recent_messages
        
        async def generate_stream():
            try:
                # Get streaming response from AI
                stream_response = await ai_service.get_mentor_response(
                    request.user_message,
                    session,
                    context,
                    stream=True
                )
                
                full_response = ""
                
                for chunk in stream_response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # Send chunk as Server-Sent Event
                        yield f"data: {json.dumps({'content': content, 'type': 'chunk'})}\n\n"
                        
                        # Small delay for better UX
                        await asyncio.sleep(0.01)
                
                # Save complete response
                if full_response:
                    formatted_response = ai_service._format_response(full_response)
                    await db_service.save_message(MessageCreate(
                        session_id=request.session_id,
                        message=full_response,
                        sender="mentor",
                        message_type="explanation",
                        metadata=formatted_response.metadata
                    ))
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'suggestions': formatted_response.suggested_actions})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Sorry, I encountered an error. Please try again.'})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to setup streaming")

@api_router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(session_id: str, limit: int = 50, offset: int = 0):
    """Get messages for a session"""
    messages = await db_service.get_session_messages(session_id, limit, offset)
    return messages

# ================================
# PREMIUM MODEL MANAGEMENT ENDPOINTS
# ================================

@api_router.get("/models/available")
async def get_available_models():
    """Get list of available AI models"""
    try:
        analytics = premium_model_manager.get_usage_analytics()
        return {
            "available_models": analytics["available_models"],
            "total_calls": analytics["total_calls"],
            "model_capabilities": {
                "deepseek-r1": {
                    "provider": "groq",
                    "specialties": ["reasoning", "learning", "explanation", "socratic", "debug"],
                    "strength_score": 9,
                    "available": "deepseek-r1" in analytics["available_models"]
                },
                "claude-sonnet": {
                    "provider": "anthropic", 
                    "specialties": ["mentoring", "analysis", "creative", "assessment"],
                    "strength_score": 10,
                    "available": "claude-sonnet" in analytics["available_models"]
                },
                "gpt-4o": {
                    "provider": "openai",
                    "specialties": ["creative", "explanation", "multimodal", "voice"],
                    "strength_score": 9,
                    "available": "gpt-4o" in analytics["available_models"]
                },
                "gemini-pro": {
                    "provider": "google",
                    "specialties": ["multimodal", "creative", "voice"],
                    "strength_score": 8,
                    "available": "gemini-pro" in analytics["available_models"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@api_router.post("/models/add-key")
async def add_model_api_key(request: dict):
    """Add new AI model API key"""
    try:
        provider = request.get("provider", "").lower()
        api_key = request.get("api_key", "")
        
        if not provider or not api_key:
            raise HTTPException(status_code=400, detail="Provider and API key are required")
        
        if provider not in ["groq", "openai", "anthropic", "google"]:
            raise HTTPException(status_code=400, detail="Unsupported provider")
        
        result = premium_model_manager.add_new_model(provider, api_key)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding model key: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add model API key")

@api_router.get("/analytics/models")
async def get_model_analytics():
    """Get model usage analytics for premium dashboard"""
    try:
        return premium_model_manager.get_usage_analytics()
    except Exception as e:
        logger.error(f"Error getting model analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model analytics")

@api_router.post("/users/{user_id}/learning-mode")
async def set_user_learning_mode(user_id: str, request: dict):
    """Set user's preferred learning mode and AI preferences"""
    try:
        preferred_mode = request.get("preferred_mode", "adaptive")
        preferences = request.get("preferences", {})
        
        await premium_ai_service.set_user_learning_mode(user_id, preferred_mode, preferences)
        
        return {
            "message": "Learning mode updated successfully",
            "preferred_mode": preferred_mode,
            "preferences": preferences
        }
        
    except Exception as e:
        logger.error(f"Error setting learning mode: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to set learning mode")

@api_router.get("/users/{user_id}/analytics")
async def get_user_learning_analytics(user_id: str):
    """Get comprehensive learning analytics for user"""
    try:
        analytics = premium_ai_service.get_learning_analytics(user_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting user analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get learning analytics")

# ================================
# INTELLIGENT GAMIFICATION ENDPOINTS
# ================================

@api_router.get("/users/{user_id}/gamification")
async def get_user_gamification_status(user_id: str):
    """Get comprehensive gamification status for user"""
    try:
        status = await gamification_service.get_user_gamification_status(user_id)
        return status
    except Exception as e:
        logger.error(f"Error getting gamification status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get gamification status")

@api_router.post("/users/{user_id}/gamification/session-complete")
async def record_session_completion(user_id: str, request: dict):
    """Record session completion and update gamification metrics"""
    try:
        session_id = request.get("session_id")
        context = request.get("context", {})
        
        # Update learning streak
        streak_result = await gamification_service.update_learning_streak(user_id)
        
        # Award points for session completion
        points_result = await gamification_service.award_adaptive_points(
            user_id, "session_complete", context
        )
        
        # Check for achievement unlocks
        new_achievements = await gamification_service.check_achievement_unlocks(
            user_id, {**context, "session_completed": True}
        )
        
        return {
            "streak": streak_result,
            "points": points_result,
            "new_achievements": [ach.dict() for ach in new_achievements],
            "motivational_message": await _generate_motivational_message(
                user_id, streak_result, points_result, new_achievements
            )
        }
    except Exception as e:
        logger.error(f"Error recording session completion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record session completion")

@api_router.post("/users/{user_id}/gamification/concept-mastered")
async def record_concept_mastery(user_id: str, request: dict):
    """Record concept mastery and update gamification metrics"""
    try:
        concept = request.get("concept")
        subject = request.get("subject")
        difficulty = request.get("difficulty", "medium")
        
        context = {
            "concept": concept,
            "subject": subject,
            "difficulty": difficulty,
            "first_time": request.get("first_time", False)
        }
        
        # Award points for concept mastery
        points_result = await gamification_service.award_adaptive_points(
            user_id, "concept_mastered", context
        )
        
        # Check for achievement unlocks
        new_achievements = await gamification_service.check_achievement_unlocks(
            user_id, {**context, "concept_mastered": True}
        )
        
        return {
            "points": points_result,
            "new_achievements": [ach.dict() for ach in new_achievements]
        }
    except Exception as e:
        logger.error(f"Error recording concept mastery: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record concept mastery")

@api_router.get("/achievements")
async def get_all_achievements():
    """Get all available achievements"""
    try:
        achievements = await db_service.get_achievements()
        return [ach.dict() for ach in achievements]
    except Exception as e:
        logger.error(f"Error getting achievements: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get achievements")

@api_router.post("/study-groups")
async def create_study_group(request: dict):
    """Create a new AI-facilitated study group"""
    try:
        admin_id = request.get("admin_id")
        subject = request.get("subject")
        description = request.get("description", f"AI-guided learning group for {subject}")
        
        group = await gamification_service.create_ai_facilitated_study_group(
            admin_id, subject, description
        )
        
        # Award points for creating a group
        await gamification_service.award_adaptive_points(
            admin_id, "group_participation", {"action": "created_group"}
        )
        
        return group.dict()
    except Exception as e:
        logger.error(f"Error creating study group: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create study group")

@api_router.get("/study-groups")
async def get_study_groups(user_id: Optional[str] = None):
    """Get study groups (user's groups or all public groups)"""
    try:
        groups = await db_service.get_study_groups(user_id)
        return [group.dict() for group in groups]
    except Exception as e:
        logger.error(f"Error getting study groups: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get study groups")

@api_router.post("/study-groups/{group_id}/join")
async def join_study_group(group_id: str, request: dict):
    """Join a study group"""
    try:
        user_id = request.get("user_id")
        
        success = await db_service.join_study_group(group_id, user_id)
        
        if success:
            # Award points for joining a group
            await gamification_service.award_adaptive_points(
                user_id, "group_participation", {"action": "joined_group"}
            )
            
            # Generate welcome activity
            await gamification_service.generate_group_activity(
                group_id, user_id, "member_joined", f"Welcome to the study group!"
            )
            
        return {"success": success}
    except Exception as e:
        logger.error(f"Error joining study group: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to join study group")

@api_router.get("/study-groups/{group_id}/activities")
async def get_group_activities(group_id: str, limit: int = 50):
    """Get activities for a study group"""
    try:
        activities = await db_service.get_group_activities(group_id, limit)
        return [activity.dict() for activity in activities]
    except Exception as e:
        logger.error(f"Error getting group activities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get group activities")

# ================================
# ADVANCED STREAMING ENDPOINTS
# ================================

@api_router.post("/streaming/session")
async def create_streaming_session(request: dict):
    """Create an adaptive streaming session"""
    try:
        session_id = request.get("session_id")
        user_id = request.get("user_id")
        preferences = request.get("preferences", {})
        
        streaming_session = await advanced_streaming_service.create_adaptive_streaming_session(
            session_id, user_id, preferences
        )
        
        return streaming_session.dict()
    except Exception as e:
        logger.error(f"Error creating streaming session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create streaming session")

@api_router.post("/streaming/{session_id}/chat")
async def advanced_streaming_chat(session_id: str, request: dict):
    """Start advanced streaming chat with adaptive features"""
    try:
        message = request.get("message")
        context = request.get("context", {})
        
        async def generate_advanced_stream():
            try:
                async for chunk in advanced_streaming_service.generate_adaptive_stream(
                    session_id, message, context
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                logger.error(f"Error in advanced streaming: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Streaming error occurred'})}\n\n"
        
        return StreamingResponse(
            generate_advanced_stream(),
            media_type="text/stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/stream"
            }
        )
    except Exception as e:
        logger.error(f"Error setting up advanced streaming: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to setup advanced streaming")

@api_router.post("/streaming/{session_id}/interrupt")
async def interrupt_stream(session_id: str, request: dict):
    """Handle user interruption during streaming"""
    try:
        user_id = request.get("user_id")
        interrupt_message = request.get("message")
        
        result = await advanced_streaming_service.handle_stream_interruption(
            session_id, user_id, interrupt_message
        )
        
        return result
    except Exception as e:
        logger.error(f"Error handling stream interruption: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to handle interruption")

@api_router.post("/streaming/multi-branch")
async def generate_multi_branch_response(request: dict):
    """Generate multiple explanation paths for the same concept"""
    try:
        session_id = request.get("session_id")
        message = request.get("message")
        branches = request.get("branches", ["visual", "logical", "practical"])
        
        result = await advanced_streaming_service.generate_multi_branch_response(
            session_id, message, branches
        )
        
        return result
    except Exception as e:
        logger.error(f"Error generating multi-branch response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate multi-branch response")

@api_router.get("/streaming/{user_id}/analytics")
async def get_streaming_analytics(user_id: str):
    """Get streaming analytics for user"""
    try:
        analytics = await advanced_streaming_service.get_streaming_analytics(user_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting streaming analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get streaming analytics")

async def _generate_motivational_message(
    user_id: str, 
    streak_result: Dict[str, Any], 
    points_result: Dict[str, Any], 
    new_achievements: List[Any]
) -> str:
    """Generate personalized motivational message"""
    
    messages = []
    
    if streak_result.get("streak_extended"):
        streak_count = streak_result.get("current_streak", 0)
        if streak_count == 1:
            messages.append("🎯 Great start! You've begun building a learning habit.")
        elif streak_count < 7:
            messages.append(f"🔥 {streak_count} days strong! Keep the momentum going.")
        elif streak_count < 30:
            messages.append(f"🚀 Amazing {streak_count}-day streak! You're building real discipline.")
        else:
            messages.append(f"💎 Incredible {streak_count}-day streak! You're a learning legend!")
    
    if points_result.get("level_up"):
        new_level = points_result.get("level", 0)
        messages.append(f"🌟 Level Up! Welcome to Level {new_level}!")
    
    if new_achievements:
        ach_names = [ach.name for ach in new_achievements[:2]]  # Show first 2
        if len(ach_names) == 1:
            messages.append(f"🏆 Achievement Unlocked: {ach_names[0]}!")
        else:
            messages.append(f"🏆 Achievements Unlocked: {', '.join(ach_names)}!")
    
    if not messages:
        messages.append("📚 Great work! Every step forward counts in your learning journey.")
    
    return " ".join(messages)

# ================================
# EXERCISE & ASSESSMENT ENDPOINTS
# ================================

@api_router.post("/exercises/generate")
async def generate_exercise(
    topic: str,
    difficulty: str = "medium",
    exercise_type: str = "multiple_choice"
):
    """Generate a practice exercise"""
    try:
        exercise_data = await ai_service.generate_exercise(topic, difficulty, exercise_type)
        return exercise_data
    except Exception as e:
        logger.error(f"Error generating exercise: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate exercise")

@api_router.post("/exercises/analyze")
async def analyze_exercise_response(
    question: str,
    user_answer: str,
    correct_answer: Optional[str] = None
):
    """Analyze user's response to an exercise"""
    try:
        analysis = await ai_service.analyze_user_response(question, user_answer, correct_answer)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze response")

# ================================
# LEARNING PATH ENDPOINTS
# ================================

@api_router.post("/learning-paths/generate")
async def generate_learning_path(
    subject: str,
    user_level: str = "beginner",
    goals: List[str] = []
):
    """Generate personalized learning path"""
    try:
        path = await ai_service.generate_learning_path(subject, user_level, goals)
        return path
    except Exception as e:
        logger.error(f"Error generating learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate learning path")

@api_router.get("/users/{user_id}/progress")
async def get_user_progress(user_id: str, subject: Optional[str] = None):
    """Get user's learning progress"""
    try:
        progress = await db_service.get_user_progress(user_id, subject)
        return progress
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get progress")

# ================================
# ADVANCED ANALYTICS ENDPOINTS
# ================================

@api_router.get("/analytics/{user_id}/comprehensive-dashboard")
async def get_comprehensive_analytics_dashboard(user_id: str, time_range: int = 30):
    """Get comprehensive learning analytics dashboard for user"""
    try:
        # Generate knowledge graph mapping
        knowledge_graph = await advanced_analytics_service.generate_knowledge_graph_mapping(user_id)
        
        # Generate competency heat map
        competency_heat_map = await advanced_analytics_service.generate_competency_heat_map(
            user_id, time_range
        )
        
        # Track learning velocity
        learning_velocity = await advanced_analytics_service.track_learning_velocity(
            user_id, window_days=7
        )
        
        # Generate retention curves
        retention_curves = await advanced_analytics_service.generate_retention_curves(user_id)
        
        # Optimize learning path
        learning_path_optimization = await advanced_analytics_service.optimize_learning_path(user_id)
        
        # Calculate summary statistics
        user_competencies = advanced_analytics_service.user_competencies.get(user_id, {})
        total_concepts = len(advanced_analytics_service.concept_library)
        mastered_concepts = sum(1 for c in user_competencies.values() if c.mastered)
        
        overall_competency = (
            sum(c.current_level for c in user_competencies.values()) / len(user_competencies)
            if user_competencies else 0.0
        )
        
        avg_learning_velocity = learning_velocity.get("overall_velocity", 0.0)
        avg_retention_score = retention_curves.get("overall_retention", 0.0)
        
        # Get next priority concepts from learning path
        next_priority_concepts = [
            item["concept_id"] for item in learning_path_optimization.get("optimal_path", [])[:5]
        ]
        
        dashboard_data = {
            "knowledge_graph": knowledge_graph,
            "competency_heat_map": competency_heat_map,
            "learning_velocity": learning_velocity,
            "retention_curves": retention_curves,
            "learning_path_optimization": learning_path_optimization,
            "summary": {
                "total_concepts": total_concepts,
                "mastered_concepts": mastered_concepts,
                "overall_competency": overall_competency,
                "learning_velocity": avg_learning_velocity,
                "retention_score": avg_retention_score,
                "next_priority_concepts": next_priority_concepts,
                "completion_percentage": (mastered_concepts / max(1, total_concepts)) * 100,
                "learning_momentum": min(1.0, avg_learning_velocity * 2),  # Normalized momentum score
                "mastery_trend": "improving" if avg_learning_velocity > 0 else "stable"
            },
            "insights": {
                "strongest_areas": competency_heat_map.get("summary", {}).get("strongest_concepts", []),
                "improvement_opportunities": [
                    item["concept_name"] for item in learning_path_optimization.get("optimal_path", [])[:3]
                ],
                "retention_insights": {
                    "best_retention": retention_curves.get("strongest_retention"),
                    "needs_review": retention_curves.get("weakest_retention")
                },
                "velocity_insights": {
                    "accelerating": learning_velocity.get("accelerating_concepts", []),
                    "stalling": learning_velocity.get("stalling_concepts", [])
                }
            },
            "recommendations": learning_path_optimization.get("adaptive_recommendations", []),
            "generated_at": datetime.utcnow().isoformat(),
            "time_range_days": time_range
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating comprehensive analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics dashboard")

@api_router.post("/analytics/learning-event")
async def record_learning_analytics_event(event_data: dict):
    """Record a learning analytics event"""
    try:
        # Create LearningEvent from request data
        learning_event = LearningEvent(
            id=str(uuid.uuid4()),
            user_id=event_data.get("user_id"),
            concept_id=event_data.get("concept_id", "general_learning"),
            event_type=event_data.get("event_type", "interaction"),
            timestamp=datetime.utcnow(),
            duration_seconds=event_data.get("duration_seconds", 0),
            performance_score=event_data.get("performance_score", 0.5),
            confidence_level=event_data.get("confidence_level", 0.5),
            session_id=event_data.get("session_id", ""),
            context=event_data.get("context", {})
        )
        
        # Record the event
        await advanced_analytics_service.record_learning_event(learning_event)
        
        return {"message": "Learning event recorded successfully", "event_id": learning_event.id}
        
    except Exception as e:
        logger.error(f"Error recording learning event: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record learning event")

@api_router.get("/analytics/{user_id}/knowledge-graph")
async def get_knowledge_graph(user_id: str):
    """Get knowledge graph data for user"""
    try:
        graph_data = await advanced_analytics_service.generate_knowledge_graph_mapping(user_id)
        return graph_data
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get knowledge graph")

@api_router.get("/analytics/{user_id}/competency-heatmap")
async def get_competency_heatmap(user_id: str, days: int = 30):
    """Get competency heat map for user"""
    try:
        heatmap_data = await advanced_analytics_service.generate_competency_heat_map(user_id, days)
        return heatmap_data
    except Exception as e:
        logger.error(f"Error getting competency heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get competency heatmap")

@api_router.get("/analytics/{user_id}/learning-velocity")
async def get_learning_velocity(user_id: str, window_days: int = 7):
    """Get learning velocity tracking for user"""
    try:
        velocity_data = await advanced_analytics_service.track_learning_velocity(user_id, window_days)
        return velocity_data
    except Exception as e:
        logger.error(f"Error getting learning velocity: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get learning velocity")

@api_router.get("/analytics/{user_id}/retention-curves")
async def get_retention_curves(user_id: str):
    """Get retention curves for user"""
    try:
        retention_data = await advanced_analytics_service.generate_retention_curves(user_id)
        return retention_data
    except Exception as e:
        logger.error(f"Error getting retention curves: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get retention curves")

@api_router.get("/analytics/{user_id}/learning-path")
async def get_optimized_learning_path(user_id: str):
    """Get AI-optimized learning path for user"""
    try:
        path_data = await advanced_analytics_service.optimize_learning_path(user_id)
        return path_data
    except Exception as e:
        logger.error(f"Error getting learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get learning path")

# ================================
# HEALTH & STATUS ENDPOINTS
# ================================

@api_router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MasterX AI Mentor System is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        await db_service.db.command("ping")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "ai_service": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# ================================
# ADVANCED LEARNING PSYCHOLOGY ENDPOINTS
# ================================

@api_router.post("/learning-psychology/metacognitive/start")
async def start_metacognitive_session(request: MetacognitiveSessionRequest, user_id: str):
    """Start a new metacognitive training session"""
    try:
        from learning_psychology_service import MetacognitiveStrategy
        
        # Convert string to enum
        strategy_map = {
            "self_questioning": MetacognitiveStrategy.SELF_QUESTIONING,
            "goal_setting": MetacognitiveStrategy.GOAL_SETTING,
            "progress_monitoring": MetacognitiveStrategy.PROGRESS_MONITORING,
            "strategy_selection": MetacognitiveStrategy.STRATEGY_SELECTION,
            "reflection": MetacognitiveStrategy.REFLECTION,
            "planning": MetacognitiveStrategy.PLANNING
        }
        
        strategy = strategy_map.get(request.strategy, MetacognitiveStrategy.SELF_QUESTIONING)
        
        session = await learning_psychology_service.start_metacognitive_session(
            user_id=user_id,
            strategy=strategy,
            topic=request.topic,
            level=request.level
        )
        
        # Convert to dict for JSON response
        return {
            "session_id": session.session_id,
            "strategy": session.strategy.value,
            "topic": session.topic,
            "level": session.level,
            "initial_prompt": session.responses[-1]["content"] if session.responses else "",
            "created_at": session.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting metacognitive session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start metacognitive session")

@api_router.post("/learning-psychology/metacognitive/{session_id}/respond")
async def respond_to_metacognitive_session(session_id: str, response: LearningPsychologyResponse):
    """Respond to metacognitive training session"""
    try:
        result = await learning_psychology_service.process_metacognitive_response(
            session_id=session_id,
            user_response=response.user_response
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing metacognitive response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process response")

@api_router.post("/learning-psychology/memory-palace/create")
async def create_memory_palace(request: MemoryPalaceRequest, user_id: str):
    """Create a new AI-assisted memory palace"""
    try:
        from learning_psychology_service import MemoryPalaceType
        
        # Convert string to enum
        palace_type_map = {
            "home": MemoryPalaceType.HOME,
            "school": MemoryPalaceType.SCHOOL,
            "nature": MemoryPalaceType.NATURE,
            "castle": MemoryPalaceType.CASTLE,
            "library": MemoryPalaceType.LIBRARY,
            "laboratory": MemoryPalaceType.LABORATORY,
            "custom": MemoryPalaceType.CUSTOM
        }
        
        palace_type = palace_type_map.get(request.palace_type, MemoryPalaceType.HOME)
        
        palace = await learning_psychology_service.create_memory_palace(
            user_id=user_id,
            name=request.name,
            palace_type=palace_type,
            topic=request.topic,
            information_items=request.information_items
        )
        
        # Convert to dict for JSON response
        return {
            "palace_id": palace.palace_id,
            "name": palace.name,
            "palace_type": palace.palace_type.value,
            "description": palace.description,
            "rooms": palace.rooms,
            "pathways": palace.pathways,
            "information_nodes": palace.information_nodes,
            "visualization_data": palace.visualization_data,
            "created_at": palace.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating memory palace: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create memory palace")

@api_router.post("/learning-psychology/memory-palace/{palace_id}/practice")
async def practice_memory_palace(palace_id: str, practice_type: str = "recall"):
    """Practice using a memory palace"""
    try:
        practice_session = await learning_psychology_service.practice_memory_palace(
            palace_id=palace_id,
            practice_type=practice_type
        )
        
        return practice_session
        
    except Exception as e:
        logger.error(f"Error practicing memory palace: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start practice session")

@api_router.get("/learning-psychology/memory-palace/user/{user_id}")
async def get_user_memory_palaces(user_id: str):
    """Get all memory palaces for a user"""
    try:
        user_palaces = []
        for palace in learning_psychology_service.memory_palaces.values():
            if palace.user_id == user_id:
                user_palaces.append({
                    "palace_id": palace.palace_id,
                    "name": palace.name,
                    "palace_type": palace.palace_type.value,
                    "description": palace.description,
                    "room_count": len(palace.rooms),
                    "information_count": len(palace.information_nodes),
                    "effectiveness_score": palace.effectiveness_score,
                    "created_at": palace.created_at.isoformat()
                })
        
        return {"palaces": user_palaces}
        
    except Exception as e:
        logger.error(f"Error getting user memory palaces: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get memory palaces")

@api_router.post("/learning-psychology/elaborative-questions/generate")
async def generate_elaborative_questions(request: ElaborativeQuestionRequest):
    """Generate elaborative interrogation questions"""
    try:
        questions = await learning_psychology_service.generate_elaborative_questions(
            topic=request.topic,
            subject_area=request.subject_area,
            difficulty_level=request.difficulty_level,
            question_count=request.question_count
        )
        
        # Convert to dict for JSON response
        question_data = []
        for question in questions:
            question_data.append({
                "question_id": question.question_id,
                "question_type": question.question_type.value,
                "content": question.content,
                "difficulty_level": question.difficulty_level,
                "subject_area": question.subject_area,
                "expected_answer_type": question.expected_answer_type,
                "evaluation_criteria": question.evaluation_criteria,
                "follow_up_questions": question.follow_up_questions
            })
        
        return {"questions": question_data}
        
    except Exception as e:
        logger.error(f"Error generating elaborative questions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate questions")

@api_router.post("/learning-psychology/elaborative-questions/{question_id}/evaluate")
async def evaluate_elaborative_response(question_id: str, response: LearningPsychologyResponse):
    """Evaluate user's response to elaborative question"""
    try:
        # First, we need to retrieve the question from our service
        # For now, we'll create a mock question based on the question_id
        # In production, this would be retrieved from database
        
        from learning_psychology_service import ElaborativeQuestion, QuestionType
        
        # Mock question retrieval - in production, get from database
        mock_question = ElaborativeQuestion(
            question_id=question_id,
            question_type=QuestionType.WHY,
            content="Sample elaborative question",
            difficulty_level="intermediate",
            subject_area="general",
            expected_answer_type="explanation",
            evaluation_criteria=["accuracy", "depth"],
            follow_up_questions=[]
        )
        
        evaluation = await learning_psychology_service.evaluate_elaborative_response(
            question=mock_question,
            user_response=response.user_response
        )
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error evaluating elaborative response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to evaluate response")

@api_router.post("/learning-psychology/transfer-learning/create-scenario")
async def create_transfer_scenario(request: TransferScenarioRequest):
    """Create a knowledge transfer learning scenario"""
    try:
        from learning_psychology_service import TransferType
        
        # Convert string to enum
        transfer_type_map = {
            "analogical": TransferType.ANALOGICAL,
            "procedural": TransferType.PROCEDURAL,
            "conceptual": TransferType.CONCEPTUAL,
            "strategic": TransferType.STRATEGIC
        }
        
        transfer_type = transfer_type_map.get(request.transfer_type, TransferType.ANALOGICAL)
        
        scenario = await learning_psychology_service.create_transfer_scenario(
            source_domain=request.source_domain,
            target_domain=request.target_domain,
            key_concepts=request.key_concepts,
            transfer_type=transfer_type
        )
        
        # Convert to dict for JSON response
        return {
            "scenario_id": scenario.scenario_id,
            "source_domain": scenario.source_domain,
            "target_domain": scenario.target_domain,
            "transfer_type": scenario.transfer_type.value,
            "scenario_description": scenario.scenario_description,
            "key_concepts": scenario.key_concepts,
            "analogy_mapping": scenario.analogy_mapping,
            "exercises": scenario.exercises,
            "difficulty_level": scenario.difficulty_level
        }
        
    except Exception as e:
        logger.error(f"Error creating transfer scenario: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create transfer scenario")

@api_router.get("/learning-psychology/progress/{user_id}")
async def get_learning_psychology_progress(user_id: str):
    """Get user's learning psychology progress summary"""
    try:
        progress = learning_psychology_service.get_user_progress_summary(user_id)
        return progress
        
    except Exception as e:
        logger.error(f"Error getting learning psychology progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get progress")

@api_router.post("/learning-psychology/progress/{user_id}/update")
async def update_learning_psychology_progress(user_id: str, session_data: Dict[str, Any]):
    """Update user's learning psychology progress"""
    try:
        learning_psychology_service.update_user_progress(user_id, session_data)
        return {"message": "Progress updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update progress")

@api_router.get("/learning-psychology/features")
async def get_learning_psychology_features():
    """Get available learning psychology features and capabilities"""
    try:
        return {
            "features": {
                "metacognitive_training": {
                    "description": "Advanced metacognitive learning strategies",
                    "strategies": [
                        "self_questioning",
                        "goal_setting", 
                        "progress_monitoring",
                        "strategy_selection",
                        "reflection",
                        "planning"
                    ],
                    "levels": ["beginner", "intermediate", "advanced"]
                },
                "memory_palace_builder": {
                    "description": "AI-assisted spatial memory techniques",
                    "palace_types": [
                        "home",
                        "school", 
                        "nature",
                        "castle",
                        "library",
                        "laboratory",
                        "custom"
                    ],
                    "practice_modes": ["recall", "navigation", "association"]
                },
                "elaborative_interrogation": {
                    "description": "Deep questioning skills development",
                    "question_types": [
                        "why",
                        "how",
                        "what_if",
                        "compare",
                        "apply",
                        "synthesize"
                    ],
                    "difficulty_levels": ["beginner", "intermediate", "advanced"]
                },
                "transfer_learning": {
                    "description": "Knowledge application across domains",
                    "transfer_types": [
                        "analogical",
                        "procedural", 
                        "conceptual",
                        "strategic"
                    ],
                    "supported_domains": ["any subject area"]
                }
            },
            "ai_capabilities": {
                "model": learning_psychology_service.model,
                "real_time_feedback": True,
                "personalized_adaptation": True,
                "progress_tracking": True,
                "multi_modal_support": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get features")

# CORS and router registration will be done at the end of the file after all endpoints are defined

# ================================
# PREMIUM ADVANCED CONTEXT AWARENESS ENDPOINTS
# ================================

@api_router.post("/context/analyze")
async def analyze_user_context(request: dict):
    """Analyze user context for advanced learning adaptation"""
    try:
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        current_message = request.get("message", "")
        conversation_context = request.get("conversation_context", [])
        
        # Get comprehensive context state
        context_state = await advanced_context_service.get_context_state(
            user_id, session_id, conversation_context, current_message
        )
        
        return {
            "context_state": context_state.to_dict(),
            "recommendations": {
                "response_complexity": context_state.response_complexity,
                "preferred_pace": context_state.preferred_pace,
                "explanation_depth": context_state.explanation_depth,
                "interaction_style": context_state.interaction_style
            },
            "adaptations": context_state.style_adaptations,
            "emotional_insights": {
                "state": context_state.emotional_state.value,
                "confidence": context_state.emotional_confidence,
                "indicators": context_state.emotional_indicators
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing context: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze user context")

@api_router.get("/context/{user_id}/memory")
async def get_user_memory_insights(user_id: str):
    """Get multi-session memory insights for user"""
    try:
        # This would get memory insights from the context service
        insights = {
            "learning_patterns": {},
            "concept_mastery": {},
            "session_history_summary": {},
            "growth_trajectory": 0.7,
            "consistency_score": 0.8
        }
        return insights
        
    except Exception as e:
        logger.error(f"Error getting memory insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get memory insights")

# ================================
# LIVE LEARNING SESSIONS ENDPOINTS
# ================================

@api_router.post("/live-sessions/create")
async def create_live_session(request: dict):
    """Create a new live learning session"""
    try:
        user_id = request.get("user_id")
        session_type_str = request.get("session_type", "voice_interaction")
        title = request.get("title", "Live Learning Session")
        duration_minutes = request.get("duration_minutes", 60)
        features = request.get("features", {})
        
        # Convert string to enum
        session_type = SessionType(session_type_str)
        
        # Create live session
        live_session = await live_learning_service.create_live_session(
            user_id, session_type, title, duration_minutes, features
        )
        
        return live_session.to_dict()
        
    except Exception as e:
        logger.error(f"Error creating live session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create live session")

@api_router.post("/live-sessions/{session_id}/voice")
async def handle_voice_interaction(session_id: str, request: dict):
    """Handle voice interaction in live session"""
    try:
        user_id = request.get("user_id")
        # In production, this would handle actual audio data
        audio_data = request.get("audio_data", b"").encode() if isinstance(request.get("audio_data", ""), str) else b""
        
        result = await live_learning_service.handle_voice_interaction(
            session_id, audio_data, user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error handling voice interaction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to handle voice interaction")

@api_router.post("/live-sessions/{session_id}/screen-share")
async def handle_screen_sharing(session_id: str, request: dict):
    """Handle screen sharing and analysis"""
    try:
        user_id = request.get("user_id")
        # In production, this would handle actual screen data
        screen_data = request.get("screen_data", b"").encode() if isinstance(request.get("screen_data", ""), str) else b""
        
        result = await live_learning_service.handle_screen_sharing(
            session_id, screen_data, user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error handling screen sharing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to handle screen sharing")

@api_router.post("/live-sessions/{session_id}/code")
async def handle_live_coding(session_id: str, request: dict):
    """Handle live coding session"""
    try:
        user_id = request.get("user_id")
        code_update = request.get("code_update", {})
        
        result = await live_learning_service.handle_live_coding(
            session_id, code_update, user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error handling live coding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to handle live coding")

@api_router.post("/live-sessions/{session_id}/whiteboard")
async def handle_interactive_whiteboard(session_id: str, request: dict):
    """Handle interactive whiteboard session"""
    try:
        user_id = request.get("user_id")
        whiteboard_update = request.get("whiteboard_update", {})
        
        result = await live_learning_service.handle_interactive_whiteboard(
            session_id, whiteboard_update, user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error handling whiteboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to handle whiteboard")

@api_router.get("/live-sessions/{session_id}/status")
async def get_live_session_status(session_id: str):
    """Get live session status"""
    try:
        status = await live_learning_service.get_session_status(session_id)
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session status")

@api_router.post("/live-sessions/{session_id}/end")
async def end_live_session(session_id: str):
    """End a live session"""
    try:
        result = await live_learning_service.end_session(session_id)
        return result
        
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to end session")

# ================================
# ENHANCED PREMIUM CHAT WITH CONTEXT AWARENESS
# ================================

@api_router.post("/chat/premium-context")
async def premium_context_aware_chat(request: MentorRequest):
    """Premium chat with advanced context awareness"""
    try:
        # Get session info
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get recent messages for context
        recent_messages = await db_service.get_recent_messages(request.session_id, limit=10)
        
        # Convert ChatMessage objects to dictionaries for context service
        conversation_context = [
            {
                'sender': msg.sender,
                'message': msg.message,
                'timestamp': msg.timestamp
            } for msg in recent_messages
        ]
        
        # Analyze user context
        context_state = await advanced_context_service.get_context_state(
            session.user_id, request.session_id, conversation_context, request.user_message
        )
        
        # Prepare enhanced context with awareness data
        enhanced_context = request.context or {}
        enhanced_context.update({
            'recent_messages': recent_messages,
            'emotional_state': context_state.emotional_state.value,
            'learning_style': context_state.learning_style.value,
            'cognitive_load': context_state.cognitive_load.value,
            'preferred_pace': context_state.preferred_pace,
            'explanation_depth': context_state.explanation_depth,
            'interaction_style': context_state.interaction_style,
            'style_adaptations': context_state.style_adaptations,
            'concept_mastery': context_state.concept_mastery,
            'learning_patterns': context_state.learning_patterns
        })
        
        # Save user message
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Get premium AI response with context awareness
        mentor_response = await premium_ai_service.get_premium_response(
            user_message=request.user_message,
            session=session,
            context=enhanced_context,
            learning_mode=enhanced_context.get('learning_mode', 'adaptive'),
            stream=False
        )
        
        # Enhance response with context insights
        mentor_response.metadata.update({
            'context_awareness': {
                'emotional_state': context_state.emotional_state.value,
                'learning_style': context_state.learning_style.value,
                'cognitive_load': context_state.cognitive_load.value,
                'adaptations_applied': context_state.style_adaptations
            },
            'personalization_score': 0.9,
            'context_confidence': context_state.emotional_confidence
        })
        
        # Save mentor response
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=mentor_response.response,
            sender="mentor",
            message_type="premium_context_aware",
            metadata=mentor_response.metadata
        ))
        
        return mentor_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in premium context chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process premium context chat")

# ================================
# ADVANCED PERSONALIZATION ENDPOINTS
# ================================

@api_router.post("/chat/adaptive")
async def adaptive_personalized_chat(request: MentorRequest):
    """Highly personalized chat using learning DNA and mood analysis"""
    try:
        # Get session info
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message first
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Get personalized response using adaptive AI service
        mentor_response = await adaptive_ai_service.get_personalized_response(
            user_message=request.user_message,
            session=session,
            context=request.context,
            stream=False
        )
        
        # Save mentor response
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=mentor_response.response,
            sender="mentor",
            message_type=mentor_response.response_type,
            metadata=mentor_response.metadata
        ))
        
        # Update goal progress if relevant
        if request.context and 'goal_id' in request.context:
            try:
                await personal_assistant.update_goal_progress(
                    request.context['goal_id'],
                    5.0,  # Small progress increment
                    {
                        'session_id': request.session_id,
                        'session_duration_minutes': 15  # Estimate
                    }
                )
            except Exception as e:
                logger.warning(f"Could not update goal progress: {str(e)}")
        
        return mentor_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in adaptive chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process adaptive chat")

@api_router.post("/chat/adaptive/stream")
async def adaptive_personalized_chat_stream(request: MentorRequest):
    """Streaming personalized chat with adaptive pacing"""
    try:
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        async def generate_adaptive_stream():
            try:
                # Get streaming response from adaptive AI
                stream_response = await adaptive_ai_service.get_personalized_response(
                    user_message=request.user_message,
                    session=session,
                    context=request.context,
                    stream=True
                )
                
                full_response = ""
                
                async for chunk in stream_response:
                    if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # Send chunk with personalization metadata
                        yield f"data: {json.dumps({'content': content, 'type': 'chunk', 'personalized': True})}\n\n"
                
                # Save complete response
                if full_response:
                    await db_service.save_message(MessageCreate(
                        session_id=request.session_id,
                        message=full_response,
                        sender="mentor",
                        message_type="adaptive_personalized",
                        metadata={"personalization": True, "adaptive_features": True}
                    ))
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'personalized': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in adaptive streaming: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Sorry, I encountered an error. Please try again.'})}\n\n"
        
        return StreamingResponse(
            generate_adaptive_stream(),
            media_type="text/stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up adaptive stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to setup adaptive streaming")

@api_router.get("/users/{user_id}/learning-dna")
async def get_learning_dna(user_id: str):
    """Get user's learning DNA profile"""
    try:
        learning_dna = await personalization_engine.analyze_learning_dna(user_id)
        dna_dict = learning_dna.to_dict()
        
        # Return format that matches test expectations
        return {
            "learning_style": dna_dict["learning_style"],
            "cognitive_patterns": dna_dict["cognitive_patterns"],
            "preferred_pace": dna_dict["preferred_pace"],
            "motivation_style": dna_dict["motivation_style"],
            "attention_span_minutes": learning_dna.attention_span_minutes,
            "difficulty_preference": learning_dna.difficulty_preference,
            "confidence_score": learning_dna.confidence_score,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning DNA: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get learning DNA")

@api_router.get("/users/{user_id}/adaptive-parameters")
async def get_adaptive_parameters(user_id: str, context: Optional[str] = None):
    """Get adaptive content parameters for user"""
    try:
        context_dict = json.loads(context) if context else {}
        parameters = await personalization_engine.get_adaptive_content_parameters(user_id, context_dict)
        params_dict = parameters.to_dict()
        
        # Return format that matches test expectations
        return {
            "complexity_level": parameters.complexity_level,
            "explanation_depth": parameters.explanation_depth,
            "example_count": parameters.example_count,
            "visual_elements": parameters.visual_elements,
            "interactive_elements": parameters.interactive_elements,
            "pacing_delay": parameters.pacing_delay,
            "reinforcement_frequency": parameters.reinforcement_frequency,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting adaptive parameters: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get adaptive parameters")

@api_router.get("/users/{user_id}/mood-analysis")
async def get_user_mood_analysis(user_id: str, session_id: str = None):
    """Get user mood analysis (GET method)"""
    try:
        if session_id:
            recent_messages = await db_service.get_recent_messages(session_id, limit=10)
        else:
            recent_messages = []
        
        mood_adaptation = await personalization_engine.analyze_mood_and_adapt(
            user_id, recent_messages, {}
        )
        
        return {
            "mood_analysis": mood_adaptation.to_dict(),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting mood analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get mood analysis")

@api_router.post("/users/{user_id}/mood-analysis")
async def analyze_user_mood(user_id: str, request: dict):
    """Analyze user mood and get adaptation recommendations"""
    try:
        session_id = request.get("session_id")
        recent_messages = request.get("recent_messages", [])
        
        if session_id:
            recent_messages_db = await db_service.get_recent_messages(session_id, limit=10)
            if recent_messages_db:
                recent_messages = recent_messages_db
        
        context = request.get("context", {})
        mood_adaptation = await personalization_engine.analyze_mood_and_adapt(
            user_id, recent_messages, context
        )
        
        mood_dict = mood_adaptation.to_dict()
        
        # Return format that matches test expectations
        return {
            "detected_mood": mood_dict["detected_mood"],
            "confidence": mood_adaptation.confidence,
            "energy_level": mood_adaptation.energy_level,
            "stress_level": mood_adaptation.stress_level,
            "recommended_pace": mood_dict["recommended_pace"],
            "content_tone": mood_adaptation.content_tone,
            "interaction_style": mood_adaptation.interaction_style,
            "break_recommendation": mood_adaptation.break_recommendation,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing mood: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze mood")

# ================================
# PERSONAL LEARNING ASSISTANT ENDPOINTS
# ================================

@api_router.post("/users/{user_id}/goals")
async def create_learning_goal(user_id: str, request: dict):
    """Create a new personalized learning goal"""
    try:
        title = request.get("title")
        description = request.get("description")
        goal_type = request.get("goal_type", "skill_mastery")
        target_date = None
        
        if request.get("target_date"):
            target_date = datetime.fromisoformat(request["target_date"])
        
        skills_required = request.get("skills_required", [])
        success_criteria = request.get("success_criteria", [])
        
        if not title or not description:
            raise HTTPException(status_code=400, detail="Title and description are required")
        
        goal = await personal_assistant.create_learning_goal(
            user_id=user_id,
            title=title,
            description=description,
            goal_type=goal_type,
            target_date=target_date,
            skills_required=skills_required,
            success_criteria=success_criteria
        )
        
        # Convert goal to dict and extract directly to match test expectations
        goal_dict = goal.to_dict()
        return {
            "goal_id": goal.goal_id,
            "user_id": goal.user_id,
            "title": goal.title,
            "description": goal.description,
            "goal_type": goal_type,
            "status": goal.status.value,
            "target_date": goal_dict.get("target_date"),
            "progress_percentage": goal.progress_percentage,
            "created_at": goal_dict.get("created_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating learning goal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create learning goal")

@api_router.get("/users/{user_id}/goals")
async def get_user_goals(user_id: str, status: Optional[str] = None):
    """Get user's learning goals"""
    try:
        goal_status = GoalStatus(status) if status else None
        goals = await personal_assistant.get_user_goals(user_id, goal_status)
        
        return {
            "goals": [goal.to_dict() for goal in goals],
            "total_count": len(goals),
            "active_count": len([g for g in goals if g.status == GoalStatus.ACTIVE])
        }
        
    except Exception as e:
        logger.error(f"Error getting user goals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user goals")

@api_router.put("/goals/{goal_id}/progress")
async def update_goal_progress(goal_id: str, request: dict):
    """Update goal progress"""
    try:
        progress_delta = request.get("progress_delta", 0.0)
        session_context = request.get("session_context", {})
        
        goal = await personal_assistant.update_goal_progress(
            goal_id=goal_id,
            progress_delta=progress_delta,
            session_context=session_context
        )
        
        # Return format that matches test expectations
        goal_dict = goal.to_dict()
        return {
            "goal_id": goal.goal_id,
            "progress_percentage": goal.progress_percentage,
            "status": goal.status.value,
            "last_activity_date": goal_dict.get("last_activity_date"),
            "updated_at": goal_dict.get("updated_at")
        }
        
    except Exception as e:
        logger.error(f"Error updating goal progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update goal progress")

@api_router.get("/users/{user_id}/recommendations")
async def get_personalized_recommendations(user_id: str):
    """Get personalized learning recommendations"""
    try:
        recommendations = await personal_assistant.get_personalized_recommendations(user_id)
        
        return {
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat(),
            "personalization_engine": "MasterX Advanced Personalization"
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@api_router.get("/users/{user_id}/insights")
async def get_learning_insights(user_id: str):
    """Get personal learning insights"""
    try:
        insights = await personal_assistant.get_learning_insights(user_id)
        
        return {
            "insights": [insight.to_dict() for insight in insights],
            "insights_count": len(insights),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get learning insights")

@api_router.post("/users/{user_id}/memories")
async def add_learning_memory(user_id: str, request: dict):
    """Add a learning memory"""
    try:
        memory_type = MemoryType(request.get("memory_type", "insight"))
        content = request.get("content")
        context = request.get("context", {})
        importance = request.get("importance", 0.5)
        related_goals = request.get("related_goals", [])
        related_concepts = request.get("related_concepts", [])
        
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        memory = await personal_assistant.add_learning_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            context=context,
            importance=importance,
            related_goals=related_goals,
            related_concepts=related_concepts
        )
        
        # Return format that matches test expectations
        memory_dict = memory.to_dict()
        return {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "memory_type": memory_dict["memory_type"],
            "content": memory.content,
            "importance": memory.importance,
            "created_at": memory_dict["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding learning memory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add learning memory")

@api_router.get("/users/{user_id}/memories")
async def get_user_memories(user_id: str, limit: int = 50, memory_type: Optional[str] = None):
    """Get user's learning memories"""
    try:
        memory_type_enum = MemoryType(memory_type) if memory_type else None
        memories = await personal_assistant.get_user_memories(user_id, limit, memory_type_enum)
        
        return {
            "memories": [memory.to_dict() for memory in memories],
            "total_count": len(memories),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting user memories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user memories")

@api_router.get("/personalization/features")
async def get_personalization_features():
    """Get available personalization features"""
    try:
        # Return format that matches test expectations
        return {
            "features": {
                "learning_dna_profiling": {
                    "description": "Deep analysis of learning patterns and preferences",
                    "features": [
                        "Learning style detection",
                        "Cognitive pattern analysis", 
                        "Pace preference optimization",
                        "Motivation style profiling",
                        "Attention span tracking",
                        "Performance pattern recognition"
                    ]
                },
                "adaptive_content_generation": {
                    "description": "AI creates content tailored to individual learners",
                    "features": [
                        "Complexity level adaptation",
                        "Explanation depth optimization",
                        "Learning style-specific formatting",
                        "Interactive element inclusion",
                        "Pacing adjustment",
                        "Reinforcement frequency tuning"
                    ]
                },
                "personal_learning_assistant": {
                    "description": "AI remembers preferences and tracks goals",
                    "features": [
                        "Long-term memory system",
                        "Goal creation and tracking",
                        "Progress prediction",
                        "Personalized recommendations",
                        "Learning insights generation",
                        "Adaptive milestone planning"
                    ]
                },
                "mood_based_adaptation": {
                    "description": "Content adapts to user's emotional state",
                    "features": [
                        "Emotional state detection",
                        "Energy level monitoring",
                        "Stress indicator analysis",
                        "Tone adaptation",
                        "Pacing adjustment",
                        "Break recommendations"
                    ]
                },
                "real_time_personalization": {
                    "description": "Live adaptation during conversations",
                    "features": [
                        "Real-time mood analysis",
                        "Dynamic difficulty adjustment",
                        "Adaptive response pacing",
                        "Context-aware interactions",
                        "Personalized motivation triggers",
                        "Learning blocker avoidance"
                    ]
                }
            },
            "version": "1.0",
            "engine": "MasterX Advanced Personalization"
        }
        
    except Exception as e:
        logger.error(f"Error getting personalization features: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get personalization features")

@api_router.post("/chat/premium-context/stream")
async def premium_context_aware_chat_stream(request: MentorRequest):
    """Premium streaming chat with advanced context awareness"""
    try:
        session = await db_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get recent messages and analyze context
        recent_messages = await db_service.get_recent_messages(request.session_id, limit=10)
        
        # Convert ChatMessage objects to dictionaries for context service
        conversation_context = [
            {
                'sender': msg.sender,
                'message': msg.message,
                'timestamp': msg.timestamp
            } for msg in recent_messages
        ]
        
        context_state = await advanced_context_service.get_context_state(
            session.user_id, request.session_id, conversation_context, request.user_message
        )
        
        # Save user message
        await db_service.save_message(MessageCreate(
            session_id=request.session_id,
            message=request.user_message,
            sender="user"
        ))
        
        # Enhanced context
        enhanced_context = request.context or {}
        enhanced_context.update({
            'recent_messages': recent_messages,
            'emotional_state': context_state.emotional_state.value,
            'learning_style': context_state.learning_style.value,
            'cognitive_load': context_state.cognitive_load.value,
            'preferred_pace': context_state.preferred_pace,
            'explanation_depth': context_state.explanation_depth,
            'interaction_style': context_state.interaction_style,
            'style_adaptations': context_state.style_adaptations
        })
        
        async def generate_context_aware_stream():
            try:
                # Get streaming response with context awareness
                stream_response = await premium_ai_service.get_premium_response(
                    user_message=request.user_message,
                    session=session,
                    context=enhanced_context,
                    learning_mode=enhanced_context.get('learning_mode', 'adaptive'),
                    stream=True
                )
                
                full_response = ""
                
                for chunk in stream_response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        # Send chunk with context awareness metadata
                        yield f"data: {json.dumps({'content': content, 'type': 'chunk', 'context': {'emotional_state': context_state.emotional_state.value, 'learning_style': context_state.learning_style.value}})}\n\n"
                        
                        # Adaptive delay based on cognitive load
                        delay = 0.01 * (1.0 + context_state.load_factors.get('session_fatigue', 0.0))
                        await asyncio.sleep(delay)
                
                # Save complete response
                if full_response:
                    formatted_response = ai_service._format_response(full_response)
                    await db_service.save_message(MessageCreate(
                        session_id=request.session_id,
                        message=full_response,
                        sender="mentor",
                        message_type="premium_context_aware_stream",
                        metadata={
                            **formatted_response.metadata,
                            "context_awareness": {
                                'emotional_state': context_state.emotional_state.value,
                                'learning_style': context_state.learning_style.value,
                                'cognitive_load': context_state.cognitive_load.value
                            }
                        }
                    ))
                
                # Send completion with context insights
                yield f"data: {json.dumps({'type': 'complete', 'suggestions': formatted_response.suggested_actions, 'context_insights': {'emotional_state': context_state.emotional_state.value, 'adaptations': context_state.style_adaptations}})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in context-aware streaming: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Sorry, I encountered an error. Please try again.'})}\n\n"
        
        return StreamingResponse(
            generate_context_aware_stream(),
            media_type="text/stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up context-aware stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to setup context-aware streaming")

# ================================
# ADVANCED LEARNING ANALYTICS ENDPOINTS
# ================================



@api_router.get("/analytics/{user_id}/knowledge-graph")
async def get_knowledge_graph_mapping(user_id: str):
    """Generate personalized knowledge graph mapping for user"""
    try:
        knowledge_graph = await advanced_analytics_service.generate_knowledge_graph_mapping(user_id)
        return knowledge_graph
        
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate knowledge graph")

@api_router.get("/analytics/{user_id}/competency-heatmap")
async def get_competency_heat_map(user_id: str, time_period: int = 30):
    """Generate competency heat map for user over specified time period (days)"""
    try:
        heat_map = await advanced_analytics_service.generate_competency_heat_map(user_id, time_period)
        return heat_map
        
    except Exception as e:
        logger.error(f"Error generating competency heat map: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate competency heat map")

@api_router.get("/analytics/{user_id}/learning-velocity")
async def get_learning_velocity_tracking(user_id: str, window_days: int = 7):
    """Track learning velocity over a rolling window"""
    try:
        velocity_data = await advanced_analytics_service.track_learning_velocity(user_id, window_days)
        return velocity_data
        
    except Exception as e:
        logger.error(f"Error tracking learning velocity: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to track learning velocity")

@api_router.get("/analytics/{user_id}/retention-curves")
async def get_retention_curves(user_id: str):
    """Generate retention curves showing how well knowledge is retained"""
    try:
        retention_curves = await advanced_analytics_service.generate_retention_curves(user_id)
        return retention_curves
        
    except Exception as e:
        logger.error(f"Error generating retention curves: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate retention curves")

@api_router.get("/analytics/{user_id}/learning-path-optimization")
async def get_optimized_learning_path(user_id: str):
    """Generate AI-optimized learning path for user"""
    try:
        learning_path = await advanced_analytics_service.optimize_learning_path(user_id)
        return learning_path
        
    except Exception as e:
        logger.error(f"Error optimizing learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to optimize learning path")

@api_router.get("/analytics/{user_id}/comprehensive-dashboard")
async def get_comprehensive_analytics_dashboard(user_id: str):
    """Get comprehensive analytics dashboard with all advanced features"""
    try:
        # Gather all analytics data
        knowledge_graph = await advanced_analytics_service.generate_knowledge_graph_mapping(user_id)
        heat_map = await advanced_analytics_service.generate_competency_heat_map(user_id, 30)
        velocity_data = await advanced_analytics_service.track_learning_velocity(user_id, 7)
        retention_curves = await advanced_analytics_service.generate_retention_curves(user_id)
        learning_path = await advanced_analytics_service.optimize_learning_path(user_id)
        
        # Combine into comprehensive dashboard
        dashboard = {
            "user_id": user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "knowledge_graph": knowledge_graph,
            "competency_heat_map": heat_map,
            "learning_velocity": velocity_data,
            "retention_curves": retention_curves,
            "learning_path_optimization": learning_path,
            "summary": {
                "total_concepts": len(knowledge_graph.get("nodes", [])),
                "mastered_concepts": knowledge_graph.get("user_progress", {}).get("mastered_concepts", 0),
                "overall_competency": knowledge_graph.get("user_progress", {}).get("average_competency", 0.0),
                "learning_velocity": velocity_data.get("overall_velocity", 0.0),
                "retention_score": retention_curves.get("overall_retention", 0.0),
                "next_priority_concepts": learning_path.get("priority_concepts", [])
            }
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error generating comprehensive dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate comprehensive dashboard")

@api_router.post("/analytics/concepts/add")
async def add_custom_concept(request: Dict[str, Any]):
    """Add a custom concept to the knowledge graph"""
    try:
        from advanced_analytics_service import ConceptNode
        
        concept = ConceptNode(
            id=request.get("id", str(uuid.uuid4())),
            name=request.get("name"),
            description=request.get("description", ""),
            difficulty_level=request.get("difficulty_level", 0.5),
            category=request.get("category", "general"),
            prerequisites=request.get("prerequisites", []),
            related_concepts=request.get("related_concepts", []),
            mastery_threshold=request.get("mastery_threshold", 0.8)
        )
        
        advanced_analytics_service.add_concept(concept)
        
        return {"message": "Concept added successfully", "concept_id": concept.id}
        
    except Exception as e:
        logger.error(f"Error adding concept: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add concept")

# ===============================
# 🎨 AR/VR AND GESTURE CONTROL ENDPOINTS
# ===============================

@api_router.get("/users/{user_id}/arvr-settings")
async def get_arvr_settings(user_id: str):
    """Get user's AR/VR settings and preferences"""
    try:
        # Get user to verify existence
        user = await db_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get settings from user preferences or return defaults
        settings = user.learning_preferences.get('arvr_settings', {
            'vr_enabled': False,
            'ar_enabled': False,
            '3d_mode_enabled': True,
            'render_quality': 'high',
            'enable_physics': True,
            'enable_shadows': True,
            'enable_lighting': True,
            'fov': 75,
            'auto_rotate': False,
            'rotation_speed': 1,
            'zoom_level': 1,
            'background_color': '#000011'
        })
        
        return {
            "user_id": user_id,
            "arvr_settings": settings,
            "capabilities": {
                "vr_supported": True,  # This would be detected on frontend
                "ar_supported": True,   # This would be detected on frontend
                "webxr_available": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AR/VR settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get AR/VR settings")

@api_router.post("/users/{user_id}/arvr-settings")
async def update_arvr_settings(user_id: str, request: dict):
    """Update user's AR/VR settings and preferences"""
    try:
        # Get user to verify existence
        user = await db_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        settings = request.get('settings', {})
        
        # Update user preferences with new AR/VR settings
        updated_preferences = user.learning_preferences.copy()
        updated_preferences['arvr_settings'] = settings
        
        # Update user in database
        success = await db_service.update_user_preferences(user_id, updated_preferences)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update settings")
        
        return {
            "message": "AR/VR settings updated successfully",
            "user_id": user_id,
            "settings": settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating AR/VR settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update AR/VR settings")

@api_router.get("/users/{user_id}/gesture-settings")
async def get_gesture_settings(user_id: str):
    """Get user's gesture control settings"""
    try:
        # Get user to verify existence
        user = await db_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get settings from user preferences or return defaults
        settings = user.learning_preferences.get('gesture_settings', {
            'enabled': False,
            'sensitivity': 0.7,
            'gesture_timeout': 2000,
            'enabled_gestures': {
                'scroll': True,
                'navigate': True,
                'voice': True,
                'speed': True,
                'volume': True
            },
            'custom_gestures': [],
            'camera_permission': 'prompt'
        })
        
        return {
            "user_id": user_id,
            "gesture_settings": settings,
            "available_gestures": [
                {"name": "scroll", "description": "Scroll through content"},
                {"name": "navigate", "description": "Navigate between sections"},
                {"name": "voice", "description": "Control voice input"},
                {"name": "speed", "description": "Adjust reading speed"},
                {"name": "volume", "description": "Control audio volume"}
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting gesture settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get gesture settings")

@api_router.post("/users/{user_id}/gesture-settings")
async def update_gesture_settings(user_id: str, request: dict):
    """Update user's gesture control settings"""
    try:
        # Get user to verify existence
        user = await db_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        settings = request.get('settings', {})
        
        # Update user preferences with new gesture settings
        updated_preferences = user.learning_preferences.copy()
        updated_preferences['gesture_settings'] = settings
        
        # Update user in database
        success = await db_service.update_user_preferences(user_id, updated_preferences)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update settings")
        
        return {
            "message": "Gesture settings updated successfully",
            "user_id": user_id,
            "settings": settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating gesture settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update gesture settings")

@api_router.post("/sessions/{session_id}/arvr-state")
async def update_session_arvr_state(session_id: str, request: dict):
    """Update the AR/VR state for a specific session"""
    try:
        # Get session to verify existence
        session = await db_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        state = request.get('state', {})
        mode = state.get('mode', 'normal')  # 'vr', 'ar', '3d', or 'normal'
        
        # Update session state with AR/VR state
        session_state = session.session_state or {}
        session_state['arvr_state'] = {
            'mode': mode,
            'enabled': state.get('enabled', False),
            'timestamp': datetime.utcnow().isoformat(),
            'settings': state.get('settings', {})
        }
        
        # Update session in database
        success = await db_service.update_session(session_id, {"session_state": session_state})
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update session state")
        
        return {
            "message": "AR/VR state updated successfully",
            "session_id": session_id,
            "state": state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating AR/VR state: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update AR/VR state")

# ================================
# ROUTER AND MIDDLEWARE REGISTRATION
# (Must be done after all endpoints are defined)
# ================================

# Include the router in the main app (after all endpoints are defined)
app.include_router(api_router)

# CORS middleware is already configured above for universal portability