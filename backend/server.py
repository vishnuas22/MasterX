"""
🚀 ENHANCED MASTERX SERVER WITH QUANTUM INTELLIGENCE INTEGRATION
Revolutionary FastAPI server with breakthrough quantum intelligence system

BREAKTHROUGH FEATURES:
- Integrated Quantum Intelligence Engine with enhanced context management
- Breakthrough AI Provider Optimization with Groq Llama 3.3 70B primary
- Revolutionary Adaptive Learning with real-time difficulty adjustment
- Enhanced MongoDB integration with advanced analytics
- Comprehensive health monitoring and performance optimization

QUANTUM INTELLIGENCE ENDPOINTS:
- Quantum message processing with context intelligence
- Advanced user profiling with learning analytics
- Breakthrough AI provider optimization endpoints
- Revolutionary adaptive learning metrics
- Enhanced conversation analytics with quantum metrics

Author: MasterX Quantum Intelligence Team
Version: 3.0 - Enhanced Server with Quantum Integration
"""

from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'masterx_quantum')]

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="MasterX Quantum Intelligence API",
    description="Revolutionary AI-powered learning platform with quantum intelligence",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Create API router with /api prefix for Kubernetes ingress
api_router = APIRouter(prefix="/api")

# Import quantum intelligence components
try:
    from quantum_intelligence.core.integrated_quantum_engine import (
        get_integrated_quantum_engine, IntegratedQuantumIntelligenceEngine
    )
    from quantum_intelligence.core.breakthrough_ai_integration import TaskType
    from quantum_intelligence.core.enhanced_database_models import (
        AdvancedLearningProfile, AdvancedConversationSession
    )
    QUANTUM_INTELLIGENCE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Quantum Intelligence system loaded successfully")
except ImportError as e:
    QUANTUM_INTELLIGENCE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Quantum Intelligence system not available: {e}")

# Import legacy components for backward compatibility
try:
    from models import StatusCheck, StatusCheckCreate, User, UserCreate
    from database import get_database
    from health_checks import health_service
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    LEGACY_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Legacy components not available: {e}")

# Import performance monitoring
try:
    from quantum_intelligence.orchestration.performance_api import (
        router as performance_router,
        initialize_performance_api,
        shutdown_performance_api
    )
    PERFORMANCE_MONITORING_AVAILABLE = True
    logger.info("✅ Performance monitoring system loaded successfully")
except ImportError as e:
    PERFORMANCE_MONITORING_AVAILABLE = False
    logger.warning(f"⚠️ Performance monitoring system not available: {e}")

# Global quantum engine instance
quantum_engine: Optional[IntegratedQuantumIntelligenceEngine] = None

# ============================================================================
# QUANTUM INTELLIGENCE REQUEST/RESPONSE MODELS
# ============================================================================

class QuantumMessageRequest(BaseModel):
    """Request model for quantum intelligence message processing"""
    user_id: str
    message: str
    session_id: Optional[str] = None
    task_type: str = "general"
    priority: str = "balanced"  # speed, quality, balanced
    initial_context: Optional[Dict[str, Any]] = None

class QuantumMessageResponse(BaseModel):
    """Response model for quantum intelligence message processing"""
    response: Dict[str, Any]
    conversation: Dict[str, Any]
    analytics: Dict[str, Any]
    quantum_metrics: Dict[str, Any]
    performance: Dict[str, Any]
    recommendations: Dict[str, Any]

class UserPreferencesUpdate(BaseModel):
    """User preferences update model"""
    difficulty_preference: Optional[float] = None
    explanation_style: Optional[str] = None
    interaction_pace: Optional[str] = None
    feedback_frequency: Optional[str] = None
    learning_goals: Optional[List[str]] = None

class SystemStatusResponse(BaseModel):
    """System status response model"""
    system_info: Dict[str, Any]
    component_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    quantum_intelligence_metrics: Dict[str, Any]
    database_status: Dict[str, Any]
    health_score: float

# ============================================================================
# STARTUP AND INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize quantum intelligence system on startup"""
    global quantum_engine
    
    try:
        logger.info("🚀 Starting MasterX Quantum Intelligence Server...")
        
        if QUANTUM_INTELLIGENCE_AVAILABLE:
            # Initialize quantum engine
            quantum_engine = get_integrated_quantum_engine(db)
            
            # Prepare API keys
            api_keys = {
                "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
                "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"), 
                "EMERGENT_LLM_KEY": os.environ.get("EMERGENT_LLM_KEY"),
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY")
            }
            
            # Filter out None values
            api_keys = {k: v for k, v in api_keys.items() if v is not None}
            
            if api_keys:
                # Initialize quantum intelligence system
                success = await quantum_engine.initialize(api_keys)
                if success:
                    logger.info("✅ Quantum Intelligence Engine initialized successfully")
                else:
                    logger.error("❌ Quantum Intelligence Engine initialization failed")
            else:
                logger.warning("⚠️ No AI provider API keys found - quantum features limited")
        
        # Initialize performance monitoring
        if PERFORMANCE_MONITORING_AVAILABLE:
            try:
                await initialize_performance_api()
                logger.info("✅ Performance monitoring system initialized successfully")
            except Exception as e:
                logger.error(f"❌ Performance monitoring initialization failed: {e}")
        
        logger.info("🌟 MasterX Quantum Intelligence Server started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down MasterX Quantum Intelligence Server...")
    
    # Shutdown performance monitoring
    if PERFORMANCE_MONITORING_AVAILABLE:
        try:
            await shutdown_performance_api()
            logger.info("✅ Performance monitoring system shutdown successfully")
        except Exception as e:
            logger.error(f"❌ Performance monitoring shutdown failed: {e}")
    
    client.close()

# ============================================================================
# CORS CONFIGURATION
# ============================================================================

@api_router.options("/{path:path}")
async def options_handler():
    """Handle CORS preflight requests"""
    from fastapi import Response
    
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Max-Age"] = "3600"
    response.status_code = 200
    
    return {"status": "ok", "message": "CORS preflight successful"}

# ============================================================================
# QUANTUM INTELLIGENCE ENDPOINTS
# ============================================================================

@api_router.post("/quantum/message", response_model=QuantumMessageResponse)
async def process_quantum_message(request: QuantumMessageRequest):
    """
    Process user message with complete quantum intelligence pipeline
    
    Revolutionary features:
    - Enhanced context management with conversation history
    - Breakthrough AI provider selection optimization  
    - Real-time adaptive learning adjustments
    - Comprehensive analytics and personalization
    """
    try:
        if not quantum_engine:
            raise HTTPException(
                status_code=503, 
                detail="Quantum Intelligence Engine not available"
            )
        
        # Map task type string to enum
        task_type_mapping = {
            "general": TaskType.GENERAL,
            "emotional_support": TaskType.EMOTIONAL_SUPPORT,
            "complex_explanation": TaskType.COMPLEX_EXPLANATION,
            "quick_response": TaskType.QUICK_RESPONSE,
            "code_examples": TaskType.CODE_EXAMPLES,
            "beginner_concepts": TaskType.BEGINNER_CONCEPTS,
            "advanced_concepts": TaskType.ADVANCED_CONCEPTS,
            "personalized_learning": TaskType.PERSONALIZED_LEARNING,
            "creative_content": TaskType.CREATIVE_CONTENT,
            "analytical_reasoning": TaskType.ANALYTICAL_REASONING
        }
        
        task_type = task_type_mapping.get(request.task_type, TaskType.GENERAL)
        
        # Process message with quantum intelligence
        result = await quantum_engine.process_user_message(
            user_id=request.user_id,
            user_message=request.message,
            session_id=request.session_id,
            initial_context=request.initial_context,
            task_type=task_type,
            priority=request.priority
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result)
        
        return QuantumMessageResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Quantum message processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Quantum message processing failed: {str(e)}"
        )

@api_router.get("/quantum/user/{user_id}/profile")
async def get_user_quantum_profile(user_id: str):
    """Get comprehensive user learning profile with quantum analytics"""
    try:
        if not quantum_engine:
            raise HTTPException(
                status_code=503,
                detail="Quantum Intelligence Engine not available"
            )
        
        profile = await quantum_engine.get_user_learning_profile(user_id)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail="User profile not found"
            )
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get user profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user profile: {str(e)}"
        )

@api_router.put("/quantum/user/{user_id}/preferences")
async def update_user_quantum_preferences(user_id: str, preferences: UserPreferencesUpdate):
    """Update user learning preferences with quantum optimization"""
    try:
        if not quantum_engine:
            raise HTTPException(
                status_code=503,
                detail="Quantum Intelligence Engine not available"
            )
        
        # Convert to dictionary, excluding None values
        prefs_dict = {k: v for k, v in preferences.dict().items() if v is not None}
        
        success = await quantum_engine.update_user_preferences(user_id, prefs_dict)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to update user preferences"
            )
        
        return {"success": True, "message": "User preferences updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update user preferences: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user preferences: {str(e)}"
        )

@api_router.get("/quantum/conversation/{conversation_id}/analytics")
async def get_conversation_quantum_analytics(conversation_id: str):
    """Get comprehensive conversation analytics with quantum metrics"""
    try:
        if not quantum_engine:
            raise HTTPException(
                status_code=503,
                detail="Quantum Intelligence Engine not available"
            )
        
        analytics = await quantum_engine.get_conversation_analytics(conversation_id)
        
        if not analytics:
            raise HTTPException(
                status_code=404,
                detail="Conversation analytics not found"
            )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get conversation analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation analytics: {str(e)}"
        )

@api_router.get("/quantum/system/status", response_model=SystemStatusResponse)
async def get_quantum_system_status():
    """Get comprehensive quantum intelligence system status"""
    try:
        if not quantum_engine:
            # Return basic status if quantum engine not available
            return SystemStatusResponse(
                system_info={
                    "status": "limited",
                    "message": "Quantum Intelligence Engine not available"
                },
                component_status={},
                performance_metrics={},
                quantum_intelligence_metrics={},
                database_status={"status": "unknown"},
                health_score=0.3
            )
        
        status = await quantum_engine.get_system_status()
        
        return SystemStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"❌ Failed to get system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@api_router.get("/health")
async def basic_health_check():
    """Basic health check for load balancer"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "quantum_intelligence": QUANTUM_INTELLIGENCE_AVAILABLE
    }

@api_router.get("/health/comprehensive") 
async def comprehensive_health_check():
    """Comprehensive health check with quantum intelligence status"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "3.0.0",
            "components": {
                "quantum_intelligence": {
                    "available": QUANTUM_INTELLIGENCE_AVAILABLE,
                    "initialized": quantum_engine is not None and quantum_engine.is_initialized if quantum_engine else False
                },
                "database": {
                    "connected": True,
                    "type": "MongoDB"
                },
                "legacy_components": {
                    "available": LEGACY_COMPONENTS_AVAILABLE
                }
            }
        }
        
        # Add quantum intelligence detailed status if available
        if quantum_engine and quantum_engine.is_initialized:
            quantum_status = await quantum_engine.get_system_status()
            health_status["quantum_intelligence_details"] = {
                "health_score": quantum_status.get("health_score", 0.5),
                "total_requests": quantum_status.get("performance_metrics", {}).get("total_requests", 0),
                "success_rate": quantum_status.get("performance_metrics", {}).get("success_rate", 0.0),
                "ai_providers_healthy": quantum_status.get("component_status", {}).get("ai_manager", {}).get("healthy_providers", 0)
            }
        
        # Test database connectivity
        try:
            await db.command("ping")
            health_status["components"]["database"]["ping_success"] = True
        except Exception as e:
            health_status["components"]["database"]["ping_success"] = False
            health_status["components"]["database"]["error"] = str(e)
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Comprehensive health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@api_router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Test database connectivity
        await db.command("ping")
        
        ready_status = {
            "status": "ready",
            "timestamp": time.time(),
            "checks": {
                "database": {"status": "healthy"},
                "quantum_intelligence": {
                    "available": QUANTUM_INTELLIGENCE_AVAILABLE,
                    "initialized": quantum_engine.is_initialized if quantum_engine else False
                }
            }
        }
        
        return ready_status
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@api_router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {
        "status": "alive",
        "timestamp": time.time(),
        "version": "3.0.0"
    }

# ============================================================================
# PERFORMANCE MONITORING ENDPOINTS
# ============================================================================

@api_router.get("/metrics/performance")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        start_time = time.time()
        
        # System metrics
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_metrics_available = True
        except ImportError:
            memory = None
            disk = None
            cpu_percent = 0
            system_metrics_available = False
        
        # Application metrics
        metrics = {
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000,
            "version": "3.0.0",
            "system_metrics_available": system_metrics_available
        }
        
        if system_metrics_available:
            metrics["system_metrics"] = {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": round(memory.percent, 1),
                    "status": "healthy" if memory.percent < 85 else "warning" if memory.percent < 95 else "critical"
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1),
                    "status": "healthy" if disk.used / disk.total < 0.8 else "warning" if disk.used / disk.total < 0.9 else "critical"
                },
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "status": "healthy" if cpu_percent < 70 else "warning" if cpu_percent < 90 else "critical"
                }
            }
        
        # Quantum intelligence metrics
        if quantum_engine and quantum_engine.is_initialized:
            quantum_status = await quantum_engine.get_system_status()
            metrics["quantum_intelligence"] = {
                "total_requests": quantum_status.get("performance_metrics", {}).get("total_requests", 0),
                "successful_responses": quantum_status.get("performance_metrics", {}).get("successful_responses", 0),
                "success_rate": quantum_status.get("performance_metrics", {}).get("success_rate", 0.0),
                "health_score": quantum_status.get("health_score", 0.5)
            }
        
        # Database metrics
        try:
            db_start = time.time()
            await db.command("ping")
            db_ping_time = (time.time() - db_start) * 1000
            metrics["database"] = {
                "status": "healthy",
                "ping_time_ms": round(db_ping_time, 2)
            }
        except Exception as e:
            metrics["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Performance metrics failed: {e}")
        return {
            "error": "Failed to collect performance metrics",
            "details": str(e),
            "timestamp": time.time()
        }

# ============================================================================
# LEGACY COMPATIBILITY ENDPOINTS
# ============================================================================

@api_router.get("/")
async def root():
    """Root endpoint with quantum intelligence info"""
    return {
        "message": "MasterX Quantum Intelligence API",
        "version": "3.0.0",
        "quantum_intelligence": QUANTUM_INTELLIGENCE_AVAILABLE,
        "status": "operational"
    }

if LEGACY_COMPONENTS_AVAILABLE:
    @api_router.post("/status", response_model=StatusCheck)
    async def create_status_check(input: StatusCheckCreate):
        """Create status check (legacy compatibility)"""
        status_dict = input.dict()
        status_obj = StatusCheck(**status_dict)
        await db.status_checks.insert_one(status_obj.dict())
        return status_obj

    @api_router.get("/status", response_model=List[StatusCheck])
    async def get_status_checks():
        """Get status checks (legacy compatibility)"""
        status_checks = await db.status_checks.find().to_list(1000)
        return [StatusCheck(**status_check) for status_check in status_checks]

# ============================================================================
# AI TESTING ENDPOINTS
# ============================================================================

@api_router.post("/ai/test")
async def test_ai_integration(request: Dict[str, Any]):
    """Test AI integration with quantum intelligence"""
    try:
        if not quantum_engine:
            raise HTTPException(
                status_code=503,
                detail="Quantum Intelligence Engine not available"
            )
        
        user_message = request.get("message", "Hello, please respond with 'AI integration working' and nothing else.")
        test_user_id = "test_user_" + str(int(time.time()))
        
        # Process test message
        result = await quantum_engine.process_user_message(
            user_id=test_user_id,
            user_message=user_message,
            task_type=TaskType.QUICK_RESPONSE,
            priority="speed"
        )
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "fallback_available": True
            }
        
        return {
            "success": True,
            "response": result["response"]["content"],
            "provider": result["response"]["provider"],
            "model": result["response"]["model"],
            "confidence": result["response"]["confidence"],
            "processing_time_ms": result["performance"]["total_processing_time_ms"]
        }
        
    except Exception as e:
        logger.error(f"❌ AI integration test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "quantum_available": QUANTUM_INTELLIGENCE_AVAILABLE
        }

# ============================================================================
# BACKGROUND TASKS AND UTILITIES
# ============================================================================

async def cleanup_old_sessions():
    """Background task to cleanup old sessions"""
    try:
        # Cleanup sessions older than 30 days
        cutoff_date = time.time() - (30 * 24 * 60 * 60)
        
        result = await db.enhanced_conversations.delete_many({
            "last_activity": {"$lt": cutoff_date}
        })
        
        if result.deleted_count > 0:
            logger.info(f"🧹 Cleaned up {result.deleted_count} old conversations")
            
    except Exception as e:
        logger.error(f"❌ Session cleanup failed: {e}")

@api_router.post("/admin/cleanup")
async def trigger_cleanup(background_tasks: BackgroundTasks):
    """Trigger background cleanup (admin endpoint)"""
    background_tasks.add_task(cleanup_old_sessions)
    return {"message": "Cleanup task scheduled"}

# ============================================================================
# ROUTER REGISTRATION AND MIDDLEWARE
# ============================================================================

# Include API router
app.include_router(api_router)

# Include interactive API router if available
try:
    from interactive_api import router as interactive_router
    app.include_router(interactive_router)
    logger.info("✅ Interactive API router included")
except ImportError:
    logger.warning("⚠️ Interactive API router not available")

# Include performance monitoring router if available
if PERFORMANCE_MONITORING_AVAILABLE:
    app.include_router(performance_router)
    logger.info("✅ Performance monitoring router included")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Additional middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for monitoring"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log slow requests
    if process_time > 5.0:  # Log requests taking more than 5 seconds
        logger.warning(
            f"Slow request: {request.method} {request.url.path} "
            f"took {process_time:.2f}s"
        )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_server:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )