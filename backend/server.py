from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import uuid
from datetime import datetime

# Import Quantum Intelligence Engine (with proper path and fallback)
try:
    from quantum_intelligence.core.engine import QuantumLearningIntelligenceEngine
    from quantum_intelligence.config.dependencies import setup_dependencies, get_quantum_engine, cleanup_dependencies
    from quantum_intelligence.config.settings import get_config
    QUANTUM_ENGINE_AVAILABLE = True
    print("✅ Quantum Intelligence Engine loaded successfully")
except ImportError as e:
    print(f"⚠️ Quantum Intelligence Engine not available: {str(e)}")
    print("🔄 Using simplified AI response system")
    QUANTUM_ENGINE_AVAILABLE = False

# Import Interactive API (with fallback)
try:
    from interactive_api import router as interactive_router
    from interactive_service import InteractiveContentService
    INTERACTIVE_FEATURES_AVAILABLE = True
    print("✅ Interactive Features loaded successfully")
except ImportError as e:
    print(f"⚠️ Interactive Features not available: {str(e)}")
    print("🔄 Using basic message system")
    INTERACTIVE_FEATURES_AVAILABLE = False
    
from models import ChatSession, SessionCreate as ModelSessionCreate


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection with fallback
try:
    mongo_url = os.environ['MONGO_URL']
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ['DB_NAME']]
    MONGODB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ MongoDB not available: {e}")
    print("🔄 Using in-memory storage for development")
    MONGODB_AVAILABLE = False
    client = None
    db = None

# In-memory storage fallback
memory_storage = {
    'chat_sessions': {},
    'chat_messages': {},
    'status_checks': {},
    'learning_progress': {},
    'learning_streaks': {},
    'user_achievements': {},
    'learning_sessions': {}
}

# Helper functions for database operations with fallback
async def save_chat_message(message_data):
    """Save chat message with MongoDB fallback to memory"""
    if MONGODB_AVAILABLE and db is not None:
        try:
            await db.chat_messages.insert_one(message_data)
        except Exception as e:
            print(f"MongoDB save failed, using memory: {e}")
            memory_storage['chat_messages'][message_data['session_id']] = memory_storage['chat_messages'].get(message_data['session_id'], [])
            memory_storage['chat_messages'][message_data['session_id']].append(message_data)
    else:
        memory_storage['chat_messages'][message_data['session_id']] = memory_storage['chat_messages'].get(message_data['session_id'], [])
        memory_storage['chat_messages'][message_data['session_id']].append(message_data)

# Create the main app without a prefix
app = FastAPI(
    title="🚀 MasterX Quantum Intelligence API",
    description="Revolutionary AI learning platform with quantum intelligence and premium interactive experiences",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include interactive API routes if available
if INTERACTIVE_FEATURES_AVAILABLE:
    app.include_router(interactive_router)
    print("✅ Interactive API routes included")

# Add a simple health check at the root level for preview environment
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "platform": "MasterX Quantum Intelligence",
        "version": "3.0",
        "quantum_engine": "online" if QUANTUM_ENGINE_AVAILABLE else "offline",
        "interactive_features": "online" if INTERACTIVE_FEATURES_AVAILABLE else "offline",
        "features": {
            "quantum_intelligence": QUANTUM_ENGINE_AVAILABLE,
            "interactive_content": INTERACTIVE_FEATURES_AVAILABLE,
            "real_time_collaboration": INTERACTIVE_FEATURES_AVAILABLE,
            "advanced_analytics": INTERACTIVE_FEATURES_AVAILABLE
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def app_root():
    return {
        "message": "MasterX Quantum Intelligence Platform",
        "version": "3.0",
        "status": "online",
        "api_docs": "/docs",
        "api_base": "/api"
    }

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize Quantum Intelligence Engine (with proper dependency injection)
quantum_engine = None
if QUANTUM_ENGINE_AVAILABLE:
    try:
        # Setup dependencies first
        setup_dependencies()
        
        # Get configured quantum engine instance
        quantum_engine = get_quantum_engine()
        print("🚀 Quantum Intelligence Engine initialized with dependencies")
        
        # Verify configuration
        config = get_config()
        print(f"✅ Configuration loaded: {config.app_name} v{config.version}")
        print(f"✅ AI Providers available: {config.has_ai_provider}")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize Quantum Engine: {str(e)}")
        print(f"🔧 Error details: {type(e).__name__}")
        quantum_engine = None
        QUANTUM_ENGINE_AVAILABLE = False
else:
    quantum_engine = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket

    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            await self.user_connections[user_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Chat Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    sender: str  # "user" or "ai"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

class SessionCreate(BaseModel):
    user_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {
        "message": "MasterX Quantum Intelligence Engine API",
        "version": "3.0",
        "status": "online",
        "capabilities": [
            "quantum_intelligence",
            "multi_modal_ai",
            "adaptive_learning",
            "real_time_mentorship"
        ]
    }

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Chat Endpoints
@api_router.post("/chat/session", response_model=SessionResponse)
async def create_chat_session(session_data: SessionCreate):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "user_id": session_data.user_id,
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    await db.chat_sessions.insert_one(session)
    logger.info(f"Created new chat session: {session_id}")
    
    return SessionResponse(
        session_id=session_id,
        created_at=session["created_at"]
    )

@api_router.post("/chat/send", response_model=ChatResponse)
async def send_chat_message(chat_request: ChatRequest):
    """Send a message and get AI response"""
    try:
        # Store user message
        user_message = ChatMessage(
            session_id=chat_request.session_id or str(uuid.uuid4()),
            message=chat_request.message,
            sender="user"
        )
        await save_chat_message(user_message.model_dump())
        
        # Generate AI response using quantum intelligence
        ai_response_text, metadata = await generate_quantum_response(
            chat_request.message, 
            chat_request.session_id
        )
        
        # Store AI response
        ai_message = ChatMessage(
            session_id=user_message.session_id,
            message=ai_response_text,
            sender="ai",
            metadata=metadata
        )
        await save_chat_message(ai_message.model_dump())
        
        logger.info(f"Generated response for session: {user_message.session_id}")
        
        return ChatResponse(
            response=ai_response_text,
            session_id=user_message.session_id,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

# Streaming Chat Endpoint
@api_router.post("/chat/stream")
async def stream_chat_message(chat_request: ChatRequest):
    """Send a message and get streaming AI response"""
    try:
        # Store user message
        user_message = ChatMessage(
            session_id=chat_request.session_id or str(uuid.uuid4()),
            message=chat_request.message,
            sender="user"
        )
        await db.chat_messages.insert_one(user_message.dict())
        
        async def generate_stream():
            try:
                # Create or get existing session
                if chat_request.session_id:
                    session_data = await db.chat_sessions.find_one({"session_id": chat_request.session_id})
                    if session_data:
                        chat_session = ChatSession(
                            id=session_data["session_id"],
                            user_id=session_data.get("user_id", "anonymous"),
                            created_at=session_data.get("created_at", datetime.utcnow()),
                            updated_at=datetime.utcnow(),
                            is_active=True
                        )
                    else:
                        chat_session = ChatSession(
                            id=chat_request.session_id,
                            user_id="anonymous",
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow(),
                            is_active=True
                        )
                else:
                    new_session_id = str(uuid.uuid4())
                    chat_session = ChatSession(
                        id=new_session_id,
                        user_id="anonymous",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        is_active=True
                    )

                # Get streaming response from quantum engine
                try:
                    stream_response = await quantum_engine.get_quantum_response(
                        user_message=chat_request.message,
                        user_id=chat_session.user_id,
                        session_id=chat_session.id,
                        learning_dna=None,
                        context={}
                    )
                except Exception as e:
                    logger.error(f"Quantum streaming error: {str(e)}")
                    stream_response = f"I understand your question about: {chat_request.message}. Let me help you explore this topic with advanced learning techniques."

                full_response = ""
                if hasattr(stream_response, '__aiter__'):
                    async for chunk in stream_response:
                        if chunk:
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk, 'session_id': user_message.session_id})}\n\n"
                else:
                    # Fallback for non-streaming response
                    response_text = str(stream_response)
                    words = response_text.split()
                    for i, word in enumerate(words):
                        chunk = word + " "
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk, 'session_id': user_message.session_id})}\n\n"
                        await asyncio.sleep(0.05)  # Simulate typing delay

                # Store complete AI response
                ai_message = ChatMessage(
                    session_id=user_message.session_id,
                    message=full_response.strip(),
                    sender="ai",
                    metadata={"streaming": True, "learning_mode": "adaptive_quantum"}
                )
                await db.chat_messages.insert_one(ai_message.dict())
                
                yield f"data: {json.dumps({'done': True, 'session_id': user_message.session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process streaming chat message")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                # Process chat message
                user_message = message_data.get("message", "")
                session_id = message_data.get("session_id")
                
                # Generate quantum response
                ai_response, metadata = await generate_quantum_response(user_message, session_id)
                
                # Send response back to client
                response_data = {
                    "type": "chat_response",
                    "message": ai_response,
                    "session_id": session_id,
                    "metadata": metadata,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await manager.send_personal_message(json.dumps(response_data), user_id)
                
            elif message_data.get("type") == "ping":
                # Handle ping for connection keepalive
                await manager.send_personal_message(json.dumps({"type": "pong"}), user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user: {user_id}")

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(1000)
        
        return [ChatMessage(**msg) for msg in messages]
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")

# Learning Progress Tracking APIs
@api_router.get("/progress/{user_id}")
async def get_user_learning_progress(user_id: str):
    """Get comprehensive learning progress for a user"""
    try:
        # Get learning progress from database
        progress_data = await db.learning_progress.find({"user_id": user_id}).to_list(100)
        
        # Get learning streaks
        streak_data = await db.learning_streaks.find_one({"user_id": user_id})
        
        # Get achievements
        achievements = await db.user_achievements.find({"user_id": user_id}).to_list(100)
        
        # Calculate comprehensive stats
        total_sessions = len(progress_data)
        concepts_mastered = sum([len(p.get("concepts_mastered", [])) for p in progress_data])
        avg_competency = sum([p.get("competency_level", 0) for p in progress_data]) / max(len(progress_data), 1)
        
        return {
            "user_id": user_id,
            "total_sessions": total_sessions,
            "concepts_mastered": concepts_mastered,
            "average_competency": round(avg_competency, 2),
            "current_streak": streak_data.get("current_streak", 0) if streak_data else 0,
            "longest_streak": streak_data.get("longest_streak", 0) if streak_data else 0,
            "total_achievements": len(achievements),
            "progress_history": progress_data[:10],  # Last 10 sessions
            "recent_achievements": achievements[-5:] if achievements else []  # Last 5 achievements
        }
        
    except Exception as e:
        logger.error(f"Error fetching learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch learning progress")

@api_router.post("/progress/{user_id}")
async def update_learning_progress(user_id: str, progress_data: Dict[str, Any]):
    """Update learning progress for a user"""
    try:
        # Create progress entry
        progress_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "subject": progress_data.get("subject", "general"),
            "topic": progress_data.get("topic", ""),
            "competency_level": progress_data.get("competency_level", 0.0),
            "concepts_mastered": progress_data.get("concepts_mastered", []),
            "areas_for_improvement": progress_data.get("areas_for_improvement", []),
            "session_duration": progress_data.get("session_duration", 0),
            "learning_mode": progress_data.get("learning_mode", "adaptive_quantum"),
            "last_reviewed": datetime.utcnow(),
            "metadata": progress_data.get("metadata", {})
        }
        
        # Insert into database
        await db.learning_progress.insert_one(progress_entry)
        
        # Update learning streak
        await update_learning_streak(user_id)
        
        return {"message": "Learning progress updated successfully", "progress_id": progress_entry["id"]}
        
    except Exception as e:
        logger.error(f"Error updating learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update learning progress")

async def update_learning_streak(user_id: str):
    """Update learning streak for user"""
    try:
        today = datetime.utcnow().date()
        
        # Get current streak data
        streak_data = await db.learning_streaks.find_one({"user_id": user_id})
        
        if not streak_data:
            # Create new streak
            new_streak = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "current_streak": 1,
                "longest_streak": 1,
                "last_activity_date": today.isoformat(),
                "total_learning_days": 1,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await db.learning_streaks.insert_one(new_streak)
        else:
            # Update existing streak
            last_activity = datetime.fromisoformat(streak_data["last_activity_date"]).date()
            
            if last_activity == today:
                # Same day, no update needed
                return
            elif (today - last_activity).days == 1:
                # Consecutive day, increment streak
                new_current_streak = streak_data["current_streak"] + 1
                new_longest_streak = max(streak_data["longest_streak"], new_current_streak)
                
                await db.learning_streaks.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "current_streak": new_current_streak,
                            "longest_streak": new_longest_streak,
                            "last_activity_date": today.isoformat(),
                            "total_learning_days": streak_data["total_learning_days"] + 1,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            else:
                # Streak broken, reset
                await db.learning_streaks.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "current_streak": 1,
                            "last_activity_date": today.isoformat(),
                            "total_learning_days": streak_data["total_learning_days"] + 1,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
    except Exception as e:
        logger.error(f"Error updating learning streak: {str(e)}")

# Quantum Learning Modes API
@api_router.get("/learning-modes")
async def get_available_learning_modes():
    """Get all available quantum learning modes"""
    return {
        "learning_modes": [
            {
                "id": "adaptive_quantum",
                "name": "Adaptive Quantum",
                "description": "AI-driven adaptive learning with real-time personalization",
                "features": ["Dynamic difficulty", "Personalized content", "Real-time adaptation"]
            },
            {
                "id": "socratic_discovery", 
                "name": "Socratic Discovery",
                "description": "Question-based discovery learning through guided inquiry",
                "features": ["Guided questioning", "Self-discovery", "Critical thinking"]
            },
            {
                "id": "debug_mastery",
                "name": "Debug Mastery", 
                "description": "Knowledge gap identification and targeted remediation",
                "features": ["Gap analysis", "Targeted practice", "Misconception correction"]
            },
            {
                "id": "challenge_evolution",
                "name": "Challenge Evolution",
                "description": "Progressive difficulty evolution with optimal challenge",
                "features": ["Progressive difficulty", "Optimal challenge", "Mastery tracking"]
            },
            {
                "id": "mentor_wisdom",
                "name": "Mentor Wisdom",
                "description": "Professional mentorship with industry insights",
                "features": ["Expert guidance", "Industry context", "Career advice"]
            },
            {
                "id": "creative_synthesis",
                "name": "Creative Synthesis", 
                "description": "Creative learning through analogies and storytelling",
                "features": ["Creative analogies", "Storytelling", "Memorable connections"]
            },
            {
                "id": "analytical_precision",
                "name": "Analytical Precision",
                "description": "Structured analytical learning with logical frameworks",
                "features": ["Logical frameworks", "Step-by-step analysis", "Structured thinking"]
            }
        ]
    }

@api_router.post("/learning-modes/{mode_id}/session")
async def create_mode_specific_session(mode_id: str, session_data: Dict[str, Any]):
    """Create a learning session with specific quantum mode"""
    try:
        # Validate learning mode
        valid_modes = ["adaptive_quantum", "socratic_discovery", "debug_mastery", 
                      "challenge_evolution", "mentor_wisdom", "creative_synthesis", "analytical_precision"]
        
        if mode_id not in valid_modes:
            raise HTTPException(status_code=400, detail="Invalid learning mode")
        
        # Create specialized session
        session = {
            "session_id": str(uuid.uuid4()),
            "user_id": session_data.get("user_id", "anonymous"),
            "learning_mode": mode_id,
            "topic": session_data.get("topic", ""),
            "difficulty_level": session_data.get("difficulty_level", "intermediate"),
            "learning_objectives": session_data.get("learning_objectives", []),
            "created_at": datetime.utcnow(),
            "status": "active",
            "metadata": {
                "mode_specific_config": session_data.get("config", {}),
                "personalization_settings": session_data.get("personalization", {})
            }
        }
        
        await db.learning_sessions.insert_one(session)
        
        return {
            "session_id": session["session_id"],
            "learning_mode": mode_id,
            "message": f"Created {mode_id} learning session successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating mode-specific session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create learning session")

# Quantum Intelligence Integration
async def generate_quantum_response(user_message: str, session_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """
    Generate AI response using the REAL Quantum Intelligence Engine (with fallback)
    """
    try:
        # Create or get existing session
        if session_id:
            # Get existing session from database
            session_data = await db.chat_sessions.find_one({"session_id": session_id})
            if session_data:
                # Convert to ChatSession model
                chat_session = ChatSession(
                    id=session_data["session_id"],
                    user_id=session_data.get("user_id", "anonymous"),
                    created_at=session_data.get("created_at", datetime.utcnow()),
                    updated_at=datetime.utcnow(),
                    is_active=True
                )
            else:
                # Create new session if not found
                chat_session = ChatSession(
                    id=session_id,
                    user_id="anonymous",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    is_active=True
                )
        else:
            # Create new session
            new_session_id = str(uuid.uuid4())
            chat_session = ChatSession(
                id=new_session_id,
                user_id="anonymous",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True
            )

        # Use Quantum Intelligence Engine if available
        if QUANTUM_ENGINE_AVAILABLE and quantum_engine:
            try:
                quantum_response = await quantum_engine.get_quantum_response(
                    user_message=user_message,
                    user_id=chat_session.user_id,
                    session_id=chat_session.id,
                    learning_dna=None,  # Will be populated from user profile
                    context={}
                )

                if hasattr(quantum_response, 'content'):
                    # Extract response data from QuantumResponse object
                    response_text = quantum_response.content
                    metadata = {
                        "learning_mode": quantum_response.quantum_mode.value,
                        "concepts": quantum_response.concept_connections,
                        "confidence": quantum_response.personalization_score,
                        "session_context": session_id is not None,
                        "intelligence_level": quantum_response.intelligence_level.name,
                        "engagement_prediction": quantum_response.engagement_prediction,
                        "learning_velocity_boost": quantum_response.learning_velocity_boost,
                        "knowledge_gaps": quantum_response.knowledge_gaps_identified,
                        "next_concepts": quantum_response.next_optimal_concepts,
                        "emotional_resonance": quantum_response.emotional_resonance_score,
                        "quantum_powered": True,
                        "processing_time": getattr(quantum_response, 'processing_time', 0.0),
                        "quantum_analytics": quantum_response.quantum_analytics
                    }
                else:
                    # Fallback if quantum engine returns string
                    response_text = str(quantum_response)
                    metadata = {
                        "learning_mode": "adaptive_quantum",
                        "concepts": extract_concepts_from_message(user_message),
                        "confidence": 0.85,
                        "session_context": session_id is not None,
                        "quantum_powered": True
                    }
            except Exception as e:
                logger.error(f"Quantum engine error: {str(e)}")
                # Enhanced fallback with smart learning mode detection
                response_text, metadata = await generate_enhanced_fallback_response(user_message, session_id)
                metadata["quantum_powered"] = False
                metadata["fallback_reason"] = str(e)
        else:
            # Enhanced fallback with smart learning mode detection
            response_text, metadata = await generate_enhanced_fallback_response(user_message, session_id)
            metadata["quantum_powered"] = False

        return response_text, metadata

    except Exception as e:
        logger.error(f"Error in quantum intelligence generation: {str(e)}")
        # Fallback to basic response if quantum engine fails
        return f"I understand your question about: {user_message}. Let me help you explore this topic with advanced learning techniques.", {
            "learning_mode": "fallback",
            "concepts": extract_concepts_from_message(user_message),
            "confidence": 0.70,
            "session_context": session_id is not None,
            "error": "quantum_engine_fallback",
            "quantum_powered": False
        }

async def generate_enhanced_fallback_response(user_message: str, session_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Enhanced fallback response with intelligent learning mode detection"""
    
    message_lower = user_message.lower()
    
    # Intelligent mode detection
    if any(word in message_lower for word in ["why", "how does", "what if", "explain why"]):
        learning_mode = "socratic_discovery"
        response = f"That's a fascinating question! Let me guide you to discover the answer through Socratic questioning. When you think about '{user_message}', what underlying principles or connections come to mind? What do you already know that might relate to this topic?"
        
    elif any(word in message_lower for word in ["confused", "don't understand", "mistake", "wrong", "error", "help me fix"]):
        learning_mode = "debug_mastery"
        response = f"I can help you debug this concept step by step. Let's break down '{user_message}' into its core components. First, let me identify where the confusion might be coming from and then we'll build a solid foundation of understanding."
        
    elif any(word in message_lower for word in ["challenge", "harder", "difficult", "test me", "quiz", "practice"]):
        learning_mode = "challenge_evolution"
        response = f"Excellent! I can see you're ready for a challenge. Let me design a progressive learning experience around '{user_message}' that will push your understanding to the next level. Are you ready to dive deep?"
        
    elif any(word in message_lower for word in ["career", "professional", "industry", "job", "work", "real world"]):
        learning_mode = "mentor_wisdom" 
        response = f"Great question from a professional perspective! Let me share some industry insights about '{user_message}'. In the real world, this concept is particularly important because..."
        
    elif any(word in message_lower for word in ["creative", "imagine", "analogy", "story", "example", "metaphor"]):
        learning_mode = "creative_synthesis"
        response = f"Let's explore '{user_message}' through creative synthesis! I'll help you understand this through memorable analogies and creative connections. Imagine if..."
        
    elif any(word in message_lower for word in ["analyze", "compare", "evaluate", "break down", "systematic"]):
        learning_mode = "analytical_precision"
        response = f"Let me provide a structured analytical breakdown of '{user_message}'. We'll approach this systematically, examining each component with precision and logical reasoning."
        
    else:
        learning_mode = "adaptive_quantum"
        response = f"I understand you're exploring '{user_message}'. Using my adaptive learning algorithms, I'll personalize this explanation to match your learning style and current understanding level. Let me analyze the best approach for you..."

    # Extract concepts with enhanced intelligence
    concepts = extract_enhanced_concepts(user_message)
    
    # Calculate enhanced confidence based on message complexity and concept detection
    base_confidence = 0.75
    concept_boost = min(len(concepts) * 0.05, 0.15)
    complexity_factor = min(len(user_message.split()) / 20, 0.1)
    final_confidence = base_confidence + concept_boost + complexity_factor
    
    metadata = {
        "learning_mode": learning_mode,
        "concepts": concepts,
        "confidence": round(final_confidence, 2),
        "session_context": session_id is not None,
        "intelligence_level": "ENHANCED",
        "engagement_prediction": 0.85,
        "response_type": "enhanced_fallback",
        "processing_time": "instant"
    }
    
    return response, metadata

def extract_enhanced_concepts(message: str) -> List[str]:
    """Enhanced concept extraction with domain-specific intelligence"""
    
    # Domain-specific concept maps
    concept_domains = {
        'technology': ['AI', 'machine learning', 'programming', 'software', 'algorithm', 'data', 'neural', 'computer', 'digital', 'automation'],
        'science': ['physics', 'chemistry', 'biology', 'research', 'experiment', 'theory', 'hypothesis', 'analysis', 'scientific'],
        'mathematics': ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'equation', 'formula', 'number', 'calculation'],
        'business': ['strategy', 'marketing', 'management', 'finance', 'sales', 'profit', 'business', 'company', 'market'],
        'education': ['learning', 'teaching', 'study', 'education', 'knowledge', 'skill', 'training', 'development', 'academic'],
        'psychology': ['behavior', 'cognitive', 'mental', 'psychology', 'emotion', 'motivation', 'perception', 'memory'],
        'design': ['design', 'creative', 'visual', 'aesthetic', 'art', 'user experience', 'interface', 'graphic']
    }
    
    message_lower = message.lower()
    found_concepts = []
    
    # Find domain-specific concepts
    for domain, keywords in concept_domains.items():
        domain_matches = [keyword for keyword in keywords if keyword in message_lower]
        if domain_matches:
            found_concepts.extend(domain_matches[:2])  # Max 2 per domain
    
    # Remove duplicates and limit
    found_concepts = list(set(found_concepts))[:5]
    
    # If no domain concepts found, use general extraction
    if not found_concepts:
        found_concepts = extract_concepts_from_message(message)
    
    return found_concepts

def extract_concepts_from_message(message: str) -> List[str]:
    """Extract key concepts from user message"""
    # Simple keyword extraction (later replace with AI)
    keywords = {
        'learning', 'education', 'quantum', 'AI', 'machine learning', 
        'neural', 'algorithm', 'data', 'programming', 'science',
        'mathematics', 'physics', 'chemistry', 'biology', 'history'
    }
    
    message_lower = message.lower()
    found_concepts = [keyword for keyword in keywords if keyword in message_lower]
    
    return found_concepts[:5]  # Limit to top 5 concepts

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    """Cleanup database and quantum engine on shutdown"""
    client.close()
    
    # Cleanup quantum engine dependencies
    if QUANTUM_ENGINE_AVAILABLE:
        try:
            await cleanup_dependencies()
            print("🧹 Quantum Intelligence Engine dependencies cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up quantum engine: {e}")
