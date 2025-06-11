import sys
import os
from pathlib import Path

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
    MentorRequest, MentorResponse, Exercise, ExerciseSubmission, LearningProgress
)
from database import db_service
from ai_service import ai_service

ROOT_DIR = backend_dir
load_dotenv(ROOT_DIR / '.env')

# Create the main app
app = FastAPI(
    title="MasterX AI Mentor System",
    description="World-class AI-powered personalized learning platform",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection"""
    try:
        await db_service.connect()
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
# CHAT & MENTORING ENDPOINTS
# ================================

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

# Include the router in the main app
app.include_router(api_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
