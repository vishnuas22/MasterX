from fastapi import FastAPI, APIRouter, HTTPException
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
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


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
        await db.chat_messages.insert_one(user_message.dict())
        
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
        await db.chat_messages.insert_one(ai_message.dict())
        
        logger.info(f"Generated response for session: {user_message.session_id}")
        
        return ChatResponse(
            response=ai_response_text,
            session_id=user_message.session_id,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

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

# Quantum Intelligence Integration
async def generate_quantum_response(user_message: str, session_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """
    Generate AI response using the Quantum Intelligence Engine
    This is a simplified version - later we'll integrate the full quantum engine
    """
    # Simulate quantum intelligence processing
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # Mock response based on quantum learning modes
    quantum_modes = [
        "adaptive_quantum",
        "socratic_discovery", 
        "debug_mastery",
        "creative_synthesis",
        "analytical_precision"
    ]
    
    selected_mode = quantum_modes[len(user_message) % len(quantum_modes)]
    
    # Mock concepts extraction
    concepts = extract_concepts_from_message(user_message)
    
    # Generate response based on mode
    response_templates = {
        "adaptive_quantum": f"Using adaptive quantum learning for: '{user_message}'. Let me analyze this through multiple cognitive dimensions...",
        "socratic_discovery": f"Excellent question! Let me guide you to discover this through Socratic questioning. What do you think might be the underlying principle here?",
        "debug_mastery": f"I can help you debug this concept. Let's break down '{user_message}' into its core components and identify any knowledge gaps...",
        "creative_synthesis": f"Let's explore '{user_message}' through creative synthesis. I can help you make analogies and connections...",
        "analytical_precision": f"Analyzing '{user_message}' with precision. Let me provide a structured, step-by-step breakdown..."
    }
    
    response = response_templates.get(selected_mode, "I understand your question about: " + user_message)
    
    metadata = {
        "learning_mode": selected_mode,
        "concepts": concepts,
        "confidence": 0.85 + (len(concepts) * 0.03),  # Mock confidence
        "session_context": session_id is not None
    }
    
    return response, metadata

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
    client.close()
