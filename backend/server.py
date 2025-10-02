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
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from core.models import ChatRequest, ChatResponse, EmotionState, ContextInfo, AbilityInfo
from core.engine import MasterXEngine
from services.gamification import GamificationEngine
from services.spaced_repetition import SpacedRepetitionEngine, ReviewQuality
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
        
        # Initialize gamification engine (Phase 5)
        app.state.gamification = GamificationEngine(db)
        logger.info("✅ Gamification engine initialized")
        
        # Initialize spaced repetition engine (Phase 5)
        app.state.spaced_repetition = SpacedRepetitionEngine(db)
        logger.info("✅ Spaced repetition engine initialized")
        
        # Initialize external benchmarking system (Phase 2)
        await app.state.engine.provider_manager.initialize_external_benchmarks(db)
        
        # Initialize intelligence layer (Phase 3: context + adaptive learning)
        app.state.engine.initialize_intelligence_layer(db)
        
        logger.info("✅ Phase 3 intelligence layer initialized (context + adaptive)")
        
        # Phase 4: Initialize optimization layer
        from optimization.caching import init_cache_manager
        from optimization.performance import init_performance_tracker
        
        app.state.cache_manager = init_cache_manager(db)
        app.state.performance_tracker = init_performance_tracker()
        
        logger.info("✅ Phase 4 optimization layer initialized (caching + performance)")
        
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
        
        # Build response with comprehensive metadata
        response = ChatResponse(
            session_id=session_id,
            message=ai_response.content,
            emotion_state=ai_response.emotion_state,
            provider_used=ai_response.provider,
            response_time_ms=ai_response.response_time_ms,
            timestamp=datetime.utcnow(),
            # Phase 2 metadata
            category_detected=ai_response.category,
            tokens_used=ai_response.tokens_used,
            cost=ai_response.cost,
            # Phase 3 metadata
            context_retrieved=ai_response.context_info,
            ability_info=ai_response.ability_info,
            ability_updated=ai_response.ability_updated,
            # Phase 4 metadata
            cached=False,  # Will be set by caching layer
            processing_breakdown=ai_response.processing_breakdown
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


# Phase 4: Performance monitoring endpoint
@app.get("/api/v1/admin/performance")
async def get_performance_dashboard():
    """Admin endpoint for performance monitoring"""
    
    try:
        from optimization.performance import get_performance_tracker
        
        tracker = get_performance_tracker()
        if not tracker:
            return {"error": "Performance tracker not initialized"}
        
        return tracker.get_dashboard_data()
    except Exception as e:
        logger.error(f"Error fetching performance dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Phase 4: Cache statistics endpoint
@app.get("/api/v1/admin/cache")
async def get_cache_stats():
    """Admin endpoint for cache statistics"""
    
    try:
        from optimization.caching import get_cache_manager
        
        cache_manager = get_cache_manager()
        if not cache_manager:
            return {"error": "Cache manager not initialized"}
        
        return await cache_manager.get_stats()
    except Exception as e:
        logger.error(f"Error fetching cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Phase 5: Gamification endpoints
@app.get("/api/v1/gamification/stats/{user_id}")
async def get_gamification_stats(user_id: str):
    """Get user gamification statistics"""
    try:
        stats = await app.state.gamification.get_user_stats(user_id)
        if not stats:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": stats.user_id,
            "level": stats.level,
            "xp": stats.xp,
            "xp_to_next_level": stats.xp_to_next_level,
            "elo_rating": stats.elo_rating,
            "current_streak": stats.current_streak,
            "longest_streak": stats.longest_streak,
            "total_sessions": stats.total_sessions,
            "total_questions": stats.total_questions,
            "total_time_minutes": stats.total_time_minutes,
            "achievements_unlocked": stats.achievements_unlocked,
            "badges": stats.badges,
            "rank": stats.rank
        }
    except Exception as e:
        logger.error(f"Error fetching gamification stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/gamification/leaderboard")
async def get_leaderboard(
    limit: int = 100,
    metric: str = "elo_rating"
):
    """Get global leaderboard"""
    try:
        from utils.database import get_database
        db = get_database()
        
        leaderboard = await app.state.gamification.leaderboard.get_global_leaderboard(
            db=db,
            limit=limit,
            metric=metric
        )
        
        return {
            "leaderboard": leaderboard,
            "metric": metric,
            "count": len(leaderboard)
        }
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/gamification/achievements")
async def get_achievements():
    """Get all available achievements"""
    try:
        achievements = app.state.gamification.achievement_engine.achievements
        
        return {
            "achievements": [
                {
                    "id": a.id,
                    "name": a.name,
                    "description": a.description,
                    "type": a.type,
                    "rarity": a.rarity,
                    "xp_reward": a.xp_reward,
                    "icon": a.icon,
                    "criteria": a.criteria
                }
                for a in achievements
            ],
            "count": len(achievements)
        }
    except Exception as e:
        logger.error(f"Error fetching achievements: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/gamification/record-activity")
async def record_gamification_activity(
    user_id: str,
    session_id: str,
    question_difficulty: float,
    success: bool,
    time_spent_seconds: int
):
    """Record user activity for gamification"""
    try:
        result = await app.state.gamification.record_activity(
            user_id=user_id,
            session_id=session_id,
            question_difficulty=question_difficulty,
            success=success,
            time_spent_seconds=time_spent_seconds
        )
        
        return result
    except Exception as e:
        logger.error(f"Error recording activity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Phase 5: Spaced Repetition endpoints
@app.get("/api/v1/spaced-repetition/due-cards/{user_id}")
async def get_due_cards(user_id: str, limit: int = 50):
    """Get cards due for review"""
    try:
        cards = await app.state.spaced_repetition.get_daily_review_session(
            user_id=user_id,
            limit=limit
        )
        
        return {
            "user_id": user_id,
            "cards": cards,
            "count": len(cards)
        }
    except Exception as e:
        logger.error(f"Error fetching due cards: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/spaced-repetition/create-card")
async def create_spaced_repetition_card(
    user_id: str,
    topic: str,
    content: Dict[str, Any]
):
    """Create a new spaced repetition card"""
    try:
        card_id = await app.state.spaced_repetition.create_card(
            user_id=user_id,
            topic=topic,
            content=content
        )
        
        return {
            "card_id": card_id,
            "user_id": user_id,
            "topic": topic,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating card: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/spaced-repetition/review-card")
async def review_spaced_repetition_card(
    card_id: str,
    quality: int,
    time_spent_seconds: int = 60
):
    """Review a card and update scheduling"""
    try:
        # Validate quality (0-5)
        if quality < 0 or quality > 5:
            raise HTTPException(
                status_code=400,
                detail="Quality must be between 0 and 5"
            )
        
        result = await app.state.spaced_repetition.review_card(
            card_id=card_id,
            quality=ReviewQuality(quality),
            time_spent_seconds=time_spent_seconds
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error reviewing card: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/spaced-repetition/stats/{user_id}")
async def get_spaced_repetition_stats(user_id: str):
    """Get user's spaced repetition statistics"""
    try:
        from utils.database import get_database
        db = get_database()
        
        stats = await app.state.spaced_repetition.scheduler.get_review_stats(
            user_id=user_id,
            db=db
        )
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching spaced repetition stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "MasterX API",
        "version": "1.0.0",
        "description": "AI-Powered Adaptive Learning Platform with Emotion Detection - Phase 5 In Progress",
        "status": "operational",
        "phase": "5 - Enhanced Features (Gamification + Spaced Repetition)",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/v1/chat",
            "providers": "/api/v1/providers",
            "admin": {
                "costs": "/api/v1/admin/costs",
                "performance": "/api/v1/admin/performance",
                "cache": "/api/v1/admin/cache"
            },
            "gamification": {
                "stats": "/api/v1/gamification/stats/{user_id}",
                "leaderboard": "/api/v1/gamification/leaderboard",
                "achievements": "/api/v1/gamification/achievements",
                "record_activity": "/api/v1/gamification/record-activity"
            },
            "spaced_repetition": {
                "due_cards": "/api/v1/spaced-repetition/due-cards/{user_id}",
                "create_card": "/api/v1/spaced-repetition/create-card",
                "review_card": "/api/v1/spaced-repetition/review-card",
                "stats": "/api/v1/spaced-repetition/stats/{user_id}"
            }
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
