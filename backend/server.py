"""
MasterX FastAPI Server
Following specifications from 2.CRITICAL_INITIAL_SETUP.md and 5.DEVELOPMENT_HANDOFF_GUIDE.md

Phase 1: Core endpoints with MongoDB integration
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from core.models import ChatRequest, ChatResponse, EmotionState
from core.engine import MasterXEngine
from utils.database import (
    connect_to_mongodb,
    close_mongodb_connection,
    initialize_database,
    get_sessions_collection,
    get_messages_collection
)
from utils.logging_config import setup_logging
from utils.errors import MasterXError
from utils.cost_tracker import cost_tracker

# Setup logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("🚀 Starting MasterX server...")
    
    try:
        # Connect to MongoDB
        await connect_to_mongodb()
        
        # Initialize database (collections and indexes)
        await initialize_database()
        
        # Initialize engine
        app.state.engine = MasterXEngine()
        
        # Get database
        from utils.database import get_database
        db = get_database()
        
        # Initialize external benchmarking system (Phase 2)
        await app.state.engine.provider_manager.initialize_external_benchmarks(db)
        
        # Initialize intelligence layer (Phase 3: context + adaptive learning)
        app.state.engine.initialize_intelligence_layer(db)
        
        logger.info("✅ Phase 3 intelligence layer initialized (context + adaptive)")
        
        logger.info("✅ MasterX server started successfully")
        logger.info(f"📊 Available AI providers: {app.state.engine.get_available_providers()}")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down MasterX server...")
    await close_mongodb_connection()
    logger.info("✅ MasterX server shut down gracefully")


# Create FastAPI app
app = FastAPI(
    title="MasterX API",
    description="AI-Powered Adaptive Learning Platform with Emotion Detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global error handler
@app.exception_handler(MasterXError)
async def masterx_error_handler(request: Request, exc: MasterXError):
    """Handle all MasterX errors consistently"""
    
    logger.error(
        f"MasterX error: {exc.message}",
        extra={
            'error_type': type(exc).__name__,
            'details': exc.details,
            'path': request.url.path
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "type": type(exc).__name__,
            "details": exc.details
        }
    )


# Health check endpoints
@app.get("/api/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/health/detailed")
async def detailed_health():
    """Detailed health check with component status"""
    
    checks = {}
    
    # Check database
    try:
        from utils.database import get_database
        db = get_database()
        await db.command('ping')
        checks['database'] = 'healthy'
    except Exception as e:
        checks['database'] = f'unhealthy: {str(e)}'
    
    # Check AI providers
    try:
        providers = app.state.engine.get_available_providers()
        checks['ai_providers'] = {
            'status': 'healthy',
            'count': len(providers),
            'providers': providers
        }
    except Exception as e:
        checks['ai_providers'] = f'unhealthy: {str(e)}'
    
    # Check emotion detection
    try:
        emotion_engine = app.state.engine.emotion_engine
        checks['emotion_detection'] = 'healthy'
    except Exception as e:
        checks['emotion_detection'] = f'unhealthy: {str(e)}'
    
    all_healthy = all(
        isinstance(v, str) and 'healthy' in v or 
        isinstance(v, dict) and v.get('status') == 'healthy'
        for v in checks.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


# Main chat endpoint
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main learning interaction endpoint
    
    Process user message with emotion detection and AI response
    """
    
    logger.info(f"📨 Chat request from user: {request.user_id}")
    
    try:
        # Get or create session
        sessions_collection = get_sessions_collection()
        messages_collection = get_messages_collection()
        
        if request.session_id:
            session_id = request.session_id
            # Load existing session
            session = await sessions_collection.find_one({"_id": session_id})
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # Create new session
            session_id = str(uuid.uuid4())
            await sessions_collection.insert_one({
                "_id": session_id,
                "user_id": request.user_id,
                "started_at": datetime.utcnow(),
                "status": "active",
                "total_messages": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "emotion_trajectory": []
            })
            logger.info(f"✅ Created new session: {session_id}")
        
        # Save user message
        user_message_id = str(uuid.uuid4())
        await messages_collection.insert_one({
            "_id": user_message_id,
            "session_id": session_id,
            "user_id": request.user_id,
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow()
        })
        
        # Process request with engine
        ai_response = await app.state.engine.process_request(
            user_id=request.user_id,
            message=request.message,
            session_id=session_id,
            context=request.context
        )
        
        # Save AI message
        ai_message_id = str(uuid.uuid4())
        await messages_collection.insert_one({
            "_id": ai_message_id,
            "session_id": session_id,
            "user_id": request.user_id,
            "role": "assistant",
            "content": ai_response.content,
            "timestamp": datetime.utcnow(),
            "emotion_state": ai_response.emotion_state.model_dump() if ai_response.emotion_state else None,
            "provider_used": ai_response.provider,
            "response_time_ms": ai_response.response_time_ms,
            "tokens_used": ai_response.tokens_used,
            "cost": ai_response.cost
        })
        
        # Update session
        await sessions_collection.update_one(
            {"_id": session_id},
            {
                "$inc": {
                    "total_messages": 2,  # User + AI message
                    "total_tokens": ai_response.tokens_used,
                    "total_cost": ai_response.cost
                },
                "$push": {
                    "emotion_trajectory": ai_response.emotion_state.primary_emotion if ai_response.emotion_state else "neutral"
                }
            }
        )
        
        # Build response
        response = ChatResponse(
            session_id=session_id,
            message=ai_response.content,
            emotion_state=ai_response.emotion_state,
            provider_used=ai_response.provider,
            response_time_ms=ai_response.response_time_ms,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"✅ Chat response generated successfully (session: {session_id})")
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Cost dashboard endpoint (admin)
@app.get("/api/v1/admin/costs")
async def get_cost_dashboard():
    """Admin endpoint for cost monitoring"""
    
    try:
        return {
            "today": await cost_tracker.get_daily_cost(),
            "this_hour": await cost_tracker.get_hourly_cost(),
            "this_week": await cost_tracker.get_weekly_cost(),
            "breakdown": await cost_tracker.get_cost_breakdown(days=7),
            "top_users": await cost_tracker.get_top_users(days=7, limit=10)
        }
    except Exception as e:
        logger.error(f"Error fetching cost dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Providers endpoint
@app.get("/api/v1/providers")
async def get_providers():
    """Get list of available AI providers"""
    
    try:
        providers = app.state.engine.get_available_providers()
        return {
            "providers": providers,
            "count": len(providers)
        }
    except Exception as e:
        logger.error(f"Error fetching providers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "MasterX API",
        "version": "1.0.0",
        "description": "AI-Powered Adaptive Learning Platform with Emotion Detection",
        "status": "operational",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/v1/chat",
            "providers": "/api/v1/providers",
            "costs": "/api/v1/admin/costs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
