"""
Utility Classes for MasterX Quantum Intelligence Platform API

Comprehensive utility classes that provide LLM integration, response handling,
WebSocket management, and other essential API functionality.

🛠️ UTILITY CAPABILITIES:
- Multi-LLM integration (Groq, Gemini, OpenAI)
- Response handling and formatting
- WebSocket connection management
- API response standardization
- Error handling and logging
- Performance monitoring

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator, Set
from fastapi import WebSocket, WebSocketDisconnect
import aiohttp
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# LLM INTEGRATION
# ============================================================================

class LLMIntegration:
    """
    🤖 LLM INTEGRATION
    
    Multi-provider LLM integration supporting Groq, Gemini, and OpenAI
    with intelligent routing, fallback mechanisms, and response optimization.
    """
    
    def __init__(self):
        """Initialize LLM integration"""
        
        # API keys from environment
        self.groq_api_key = os.getenv('GROQ_API_KEY', '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')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCmV-mlB7rag8GurIDj07ijRDhPuNwOiVA')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Provider configurations
        self.providers = {
            'groq': {
                'api_key': self.groq_api_key,
                'base_url': 'https://api.groq.com/openai/v1',
                'models': ['mixtral-8x7b-32768', 'llama2-70b-4096'],
                'default_model': 'mixtral-8x7b-32768',
                'available': bool(self.groq_api_key)
            },
            'gemini': {
                'api_key': self.gemini_api_key,
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'models': ['gemini-pro', 'gemini-pro-vision'],
                'default_model': 'gemini-pro',
                'available': bool(self.gemini_api_key)
            },
            'openai': {
                'api_key': self.openai_api_key,
                'base_url': 'https://api.openai.com/v1',
                'models': ['gpt-4', 'gpt-3.5-turbo'],
                'default_model': 'gpt-3.5-turbo',
                'available': bool(self.openai_api_key)
            }
        }
        
        # Default provider priority
        self.provider_priority = ['groq', 'gemini', 'openai']
        
        # Performance tracking
        self.provider_stats = {
            provider: {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'avg_response_time': 0.0
            }
            for provider in self.providers.keys()
        }
        
        logger.info("🤖 LLM Integration initialized")
        logger.info(f"Available providers: {[p for p, config in self.providers.items() if config['available']]}")
    
    async def generate_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using specified or best available LLM provider"""
        
        try:
            # Select provider
            selected_provider = provider or self._select_best_provider()
            
            if not selected_provider:
                raise Exception("No LLM providers available")
            
            # Generate response based on provider
            if selected_provider == 'groq':
                return await self._generate_groq_response(message, context, user_id, model)
            elif selected_provider == 'gemini':
                return await self._generate_gemini_response(message, context, user_id, model)
            elif selected_provider == 'openai':
                return await self._generate_openai_response(message, context, user_id, model)
            else:
                raise Exception(f"Unsupported provider: {selected_provider}")
                
        except Exception as e:
            logger.error(f"LLM response generation error: {e}")
            
            # Try fallback provider
            if provider:  # If specific provider failed, try fallback
                fallback_provider = self._select_best_provider(exclude=[provider])
                if fallback_provider:
                    return await self.generate_response(message, context, user_id, fallback_provider)
            
            # Return fallback response
            return {
                'content': 'I apologize, but I encountered an issue generating a response. Please try again.',
                'provider': 'fallback',
                'suggestions': ['Try rephrasing your question', 'Check your internet connection'],
                'metadata': {'error': str(e)}
            }
    
    async def stream_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        provider: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from LLM provider"""
        
        try:
            # Select provider
            selected_provider = provider or self._select_best_provider()
            
            if not selected_provider:
                yield {'error': 'No LLM providers available'}
                return
            
            # Stream based on provider
            if selected_provider == 'groq':
                async for chunk in self._stream_groq_response(message, context, user_id):
                    yield chunk
            elif selected_provider == 'gemini':
                async for chunk in self._stream_gemini_response(message, context, user_id):
                    yield chunk
            else:
                # Fallback to non-streaming
                response = await self.generate_response(message, context, user_id, selected_provider)
                yield {'content': response['content'], 'done': True}
                
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield {'error': str(e)}
    
    def _select_best_provider(self, exclude: List[str] = None) -> Optional[str]:
        """Select the best available LLM provider"""
        
        exclude = exclude or []
        
        for provider in self.provider_priority:
            if provider not in exclude and self.providers[provider]['available']:
                return provider
        
        return None
    
    async def _generate_groq_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using Groq API"""
        
        start_time = time.time()
        provider = 'groq'
        
        try:
            self.provider_stats[provider]['requests'] += 1
            
            # Prepare system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare request
            model_name = model or self.providers[provider]['default_model']
            
            headers = {
                'Authorization': f'Bearer {self.providers[provider]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': message}
                ],
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)
                        
                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'suggestions': self._generate_suggestions(content),
                            'metadata': {
                                'response_time': response_time,
                                'tokens_used': data.get('usage', {}).get('total_tokens', 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Groq API error {response.status}: {error_text}")
                        
        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e
    
    async def _generate_gemini_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using Gemini API"""
        
        start_time = time.time()
        provider = 'gemini'
        
        try:
            self.provider_stats[provider]['requests'] += 1
            
            # Prepare context-aware prompt
            context_prompt = self._build_context_prompt(message, context)
            
            # Prepare request
            model_name = model or self.providers[provider]['default_model']
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                'contents': [{
                    'parts': [{'text': context_prompt}]
                }],
                'generationConfig': {
                    'temperature': 0.7,
                    'maxOutputTokens': 1000
                }
            }
            
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/models/{model_name}:generateContent?key={self.providers[provider]['api_key']}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        
                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)
                        
                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'suggestions': self._generate_suggestions(content),
                            'metadata': {
                                'response_time': response_time,
                                'safety_ratings': data['candidates'][0].get('safetyRatings', [])
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Gemini API error {response.status}: {error_text}")
                        
        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e
    
    async def _generate_openai_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        
        start_time = time.time()
        provider = 'openai'
        
        try:
            self.provider_stats[provider]['requests'] += 1
            
            # Prepare system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Prepare request
            model_name = model or self.providers[provider]['default_model']
            
            headers = {
                'Authorization': f'Bearer {self.providers[provider]["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': message}
                ],
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.providers[provider]['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Update stats
                        response_time = time.time() - start_time
                        self._update_provider_stats(provider, True, response_time)
                        
                        return {
                            'content': content,
                            'provider': provider,
                            'model': model_name,
                            'suggestions': self._generate_suggestions(content),
                            'metadata': {
                                'response_time': response_time,
                                'tokens_used': data.get('usage', {}).get('total_tokens', 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error {response.status}: {error_text}")
                        
        except Exception as e:
            self._update_provider_stats(provider, False, time.time() - start_time)
            raise e
    
    async def _stream_groq_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from Groq API"""
        
        try:
            # For now, simulate streaming by chunking the response
            response = await self._generate_groq_response(message, context, user_id)
            content = response['content']
            
            # Split into chunks and yield
            words = content.split()
            chunk_size = 3
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                yield {
                    'content': chunk,
                    'provider': 'groq',
                    'done': i + chunk_size >= len(words)
                }
                await asyncio.sleep(0.1)  # Simulate streaming delay
                
        except Exception as e:
            yield {'error': str(e)}
    
    async def _stream_gemini_response(
        self,
        message: str,
        context: Dict[str, Any],
        user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from Gemini API"""
        
        try:
            # For now, simulate streaming by chunking the response
            response = await self._generate_gemini_response(message, context, user_id)
            content = response['content']
            
            # Split into chunks and yield
            words = content.split()
            chunk_size = 3
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                yield {
                    'content': chunk,
                    'provider': 'gemini',
                    'done': i + chunk_size >= len(words)
                }
                await asyncio.sleep(0.1)  # Simulate streaming delay
                
        except Exception as e:
            yield {'error': str(e)}
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context"""
        
        base_prompt = """You are MasterX AI, an advanced quantum intelligence learning assistant. You provide personalized, adaptive learning support with deep understanding of each student's unique learning profile."""
        
        # Add personalization context
        if 'personalization' in context:
            personalization = context['personalization']
            base_prompt += f"\n\nUser Learning Profile:"
            base_prompt += f"\n- Learning Style: {personalization.get('learning_style', 'adaptive')}"
            base_prompt += f"\n- Preferred Pace: {personalization.get('preferred_pace', 'moderate')}"
            base_prompt += f"\n- Difficulty Preference: {personalization.get('difficulty_preference', 0.5)}"
        
        # Add learning context
        if 'learning' in context:
            learning = context['learning']
            base_prompt += f"\n\nCurrent Learning Context:"
            base_prompt += f"\n- Goals: {', '.join(learning.get('current_goals', []))}"
            base_prompt += f"\n- Recent Topics: {', '.join(learning.get('recent_topics', []))}"
            base_prompt += f"\n- Skill Levels: {learning.get('skill_levels', {})}"
        
        base_prompt += "\n\nProvide helpful, encouraging, and personalized responses that adapt to the user's learning style and current progress."
        
        return base_prompt
    
    def _build_context_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build context-aware prompt for Gemini"""
        
        system_context = self._build_system_prompt(context)
        return f"{system_context}\n\nUser Question: {message}"
    
    def _generate_suggestions(self, content: str) -> List[str]:
        """Generate follow-up suggestions based on response content"""
        
        # Simple suggestion generation (can be enhanced with ML)
        suggestions = []
        
        if 'python' in content.lower():
            suggestions.append("Can you show me a code example?")
            suggestions.append("What are the best practices for this?")
        
        if 'math' in content.lower() or 'equation' in content.lower():
            suggestions.append("Can you solve a similar problem?")
            suggestions.append("What's the step-by-step process?")
        
        if not suggestions:
            suggestions = [
                "Can you explain this in more detail?",
                "What should I learn next?",
                "Can you give me practice exercises?"
            ]
        
        return suggestions[:3]  # Return max 3 suggestions
    
    def _update_provider_stats(self, provider: str, success: bool, response_time: float):
        """Update provider performance statistics"""
        
        stats = self.provider_stats[provider]
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        # Update average response time
        total_requests = stats['successes'] + stats['failures']
        current_avg = stats['avg_response_time']
        stats['avg_response_time'] = ((current_avg * (total_requests - 1)) + response_time) / total_requests
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider performance statistics"""
        return self.provider_stats.copy()

# ============================================================================
# API RESPONSE HANDLER
# ============================================================================

class APIResponseHandler:
    """
    📋 API RESPONSE HANDLER
    
    Standardized response handling and formatting for all API endpoints.
    """
    
    def __init__(self):
        """Initialize response handler"""
        logger.info("📋 API Response Handler initialized")
    
    def success_response(
        self,
        data: Any,
        message: str = "Operation completed successfully",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized success response"""
        
        return {
            'success': True,
            'message': message,
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4())
        }
    
    def error_response(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        
        return {
            'success': False,
            'message': error_message,
            'error_code': error_code,
            'error_details': details or {},
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4())
        }

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class WebSocketManager:
    """
    🔌 WEBSOCKET MANAGER
    
    WebSocket connection management for real-time features.
    """
    
    def __init__(self):
        """Initialize WebSocket manager"""
        
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        
        logger.info("🔌 WebSocket Manager initialized")
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept WebSocket connection"""
        
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Disconnect WebSocket"""
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection"""
        
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id, "unknown")
    
    async def send_user_message(self, message: str, user_id: str):
        """Send message to all connections of a user"""
        
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        
        for connection_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, connection_id)
