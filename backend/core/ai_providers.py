"""
Dynamic Multi-AI Provider System
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md Section 2

REVOLUTIONARY APPROACH:
- Auto-discovers providers from .env (no hardcoding!)
- Supports unlimited AI providers
- Add/remove models by just editing .env
- No code changes needed

Phase 1 Implementation: Auto-discovery + Basic routing
Phase 2 (Week 2): Add benchmarking system
"""

import os
import logging
import time
from typing import Dict, Optional, List
from core.models import AIResponse, EmotionState
from utils.errors import ProviderError

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Auto-discover AI providers from environment variables"""
    
    def __init__(self):
        self.providers: Dict[str, Dict] = {}
        self.discover_providers()
    
    def discover_providers(self):
        """
        Scan .env for *_API_KEY and *_MODEL_NAME patterns
        Automatically initializes discovered providers
        """
        env_vars = list(os.environ.keys())
        
        # Find all API keys
        api_keys = [k for k in env_vars if k.endswith('_API_KEY') or k.endswith('_LLM_KEY')]
        
        logger.info(f"Scanning environment for AI providers...")
        
        for key_var in api_keys:
            # Extract provider name
            provider_name = key_var.replace('_API_KEY', '').replace('_LLM_KEY', '').lower()
            
            # Look for corresponding model name
            model_var_options = [
                f"{provider_name.upper()}_MODEL_NAME",
                f"{provider_name.upper()}_MODEL"
            ]
            
            model_name = None
            for model_var in model_var_options:
                if model_var in env_vars:
                    model_name = os.getenv(model_var)
                    break
            
            # Get API key
            api_key = os.getenv(key_var)
            
            if api_key:
                self.providers[provider_name] = {
                    'api_key': api_key,
                    'model_name': model_name,
                    'enabled': True,
                    'key_var': key_var
                }
                logger.info(f"✅ Discovered provider: {provider_name} (model: {model_name or 'default'})")
        
        if not self.providers:
            logger.warning("⚠️ No AI providers discovered in environment")
        else:
            logger.info(f"📊 Total providers discovered: {len(self.providers)}")
    
    def get_provider(self, name: str) -> Optional[Dict]:
        """Get provider configuration by name"""
        return self.providers.get(name.lower())
    
    def get_all_providers(self) -> Dict[str, Dict]:
        """Get all discovered providers"""
        return self.providers
    
    def is_available(self, name: str) -> bool:
        """Check if provider is available"""
        provider = self.get_provider(name)
        return provider is not None and provider.get('enabled', False)


class UniversalProvider:
    """Unified interface for all AI providers"""
    
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
        self._clients = {}
    
    def _get_client(self, provider_name: str):
        """Get or create client for provider"""
        if provider_name in self._clients:
            return self._clients[provider_name]
        
        provider = self.registry.get_provider(provider_name)
        if not provider:
            raise ProviderError(
                f"Provider {provider_name} not found",
                details={'provider': provider_name}
            )
        
        # Initialize client based on provider
        client = None
        
        try:
            if provider_name == 'groq':
                from groq import AsyncGroq
                client = AsyncGroq(api_key=provider['api_key'])
            
            elif provider_name == 'emergent':
                from emergentintegrations import EmergentLLM
                client = EmergentLLM(api_key=provider['api_key'])
            
            elif provider_name == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=provider['api_key'])
                client = genai
            
            elif provider_name == 'openai':
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=provider['api_key'])
            
            elif provider_name == 'anthropic':
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=provider['api_key'])
            
            else:
                logger.warning(f"Provider {provider_name} not yet implemented, using fallback")
                return None
            
            self._clients[provider_name] = client
            return client
        
        except Exception as e:
            logger.error(f"Error initializing {provider_name} client: {e}")
            raise ProviderError(
                f"Failed to initialize {provider_name}",
                details={'provider': provider_name, 'error': str(e)}
            )
    
    async def generate(
        self,
        provider_name: str,
        prompt: str,
        max_tokens: int = 1000
    ) -> AIResponse:
        """Generate response from specified provider"""
        
        start_time = time.time()
        provider = self.registry.get_provider(provider_name)
        
        if not provider:
            raise ProviderError(
                f"Provider {provider_name} not available",
                details={'provider': provider_name}
            )
        
        try:
            client = self._get_client(provider_name)
            model_name = provider['model_name'] or 'default'
            
            # Route to correct provider implementation
            if provider_name == 'groq':
                response = await self._groq_generate(client, model_name, prompt, max_tokens)
            
            elif provider_name == 'emergent':
                response = await self._emergent_generate(client, model_name, prompt, max_tokens)
            
            elif provider_name == 'gemini':
                response = await self._gemini_generate(client, model_name, prompt, max_tokens)
            
            elif provider_name == 'openai':
                response = await self._openai_generate(client, model_name, prompt, max_tokens)
            
            elif provider_name == 'anthropic':
                response = await self._anthropic_generate(client, model_name, prompt, max_tokens)
            
            else:
                raise ProviderError(
                    f"Provider {provider_name} not implemented",
                    details={'provider': provider_name}
                )
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"✅ {provider_name} response generated in {elapsed_ms:.0f}ms")
            
            return response
        
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ {provider_name} failed after {elapsed_ms:.0f}ms: {e}")
            raise ProviderError(
                f"Provider {provider_name} failed",
                details={
                    'provider': provider_name,
                    'error': str(e),
                    'elapsed_ms': elapsed_ms
                }
            )
    
    async def _groq_generate(self, client, model_name, prompt, max_tokens) -> AIResponse:
        """Generate using Groq"""
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        return AIResponse(
            content=response.choices[0].message.content,
            provider="groq",
            model_name=model_name,
            tokens_used=response.usage.total_tokens,
            cost=0.0,  # Calculate in cost tracker
            response_time_ms=0.0
        )
    
    async def _emergent_generate(self, client, model_name, prompt, max_tokens) -> AIResponse:
        """Generate using Emergent LLM"""
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        return AIResponse(
            content=response.choices[0].message.content,
            provider="emergent",
            model_name=model_name,
            tokens_used=response.usage.total_tokens,
            cost=0.0,
            response_time_ms=0.0
        )
    
    async def _gemini_generate(self, client, model_name, prompt, max_tokens) -> AIResponse:
        """Generate using Google Gemini"""
        model = client.GenerativeModel(model_name)
        response = await model.generate_content_async(prompt)
        
        # Estimate tokens (Gemini doesn't provide exact count in basic API)
        estimated_tokens = len(prompt.split()) + len(response.text.split())
        
        return AIResponse(
            content=response.text,
            provider="gemini",
            model_name=model_name,
            tokens_used=estimated_tokens,
            cost=0.0,
            response_time_ms=0.0
        )
    
    async def _openai_generate(self, client, model_name, prompt, max_tokens) -> AIResponse:
        """Generate using OpenAI"""
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        return AIResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model_name=model_name,
            tokens_used=response.usage.total_tokens,
            cost=0.0,
            response_time_ms=0.0
        )
    
    async def _anthropic_generate(self, client, model_name, prompt, max_tokens) -> AIResponse:
        """Generate using Anthropic Claude"""
        response = await client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return AIResponse(
            content=response.content[0].text,
            provider="anthropic",
            model_name=model_name,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            cost=0.0,
            response_time_ms=0.0
        )


class ProviderManager:
    """
    Main interface for AI provider management
    
    Features:
    - Auto-discovery of providers from .env
    - External benchmarking integration (Artificial Analysis API)
    - Smart routing based on real-world rankings
    - Automatic fallback on provider failure
    """
    
    def __init__(self):
        self.registry = ProviderRegistry()
        self.universal = UniversalProvider(self.registry)
        self.external_benchmarks = None  # Initialized in server startup
        self._default_provider = None
        self._set_default_provider()
        
        # Add universal provider reference for minimal manual tests
        self.universal_provider = self.universal
    
    def _set_default_provider(self):
        """Set default provider (first available)"""
        providers = self.registry.get_all_providers()
        if providers:
            # Prefer groq, emergent, or gemini if available
            for preferred in ['groq', 'emergent', 'gemini']:
                if preferred in providers:
                    self._default_provider = preferred
                    logger.info(f"🎯 Default provider set to: {preferred}")
                    return
            
            # Otherwise use first available
            self._default_provider = list(providers.keys())[0]
            logger.info(f"🎯 Default provider set to: {self._default_provider}")
        else:
            logger.error("❌ No AI providers available!")
    
    async def generate(
        self,
        prompt: str,
        provider_name: Optional[str] = None,
        max_tokens: int = 1000
    ) -> AIResponse:
        """
        Generate AI response
        Phase 1: Use specified provider or default
        Phase 2: Will use smart routing based on benchmarks
        """
        
        # Use specified provider or default
        target_provider = provider_name or self._default_provider
        
        if not target_provider:
            raise ProviderError(
                "No AI provider available",
                details={'requested': provider_name}
            )
        
        logger.info(f"🤖 Generating response using: {target_provider}")
        
        try:
            response = await self.universal.generate(
                provider_name=target_provider,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response
        
        except ProviderError as e:
            # Phase 1: Simple fallback to any other provider
            logger.warning(f"Provider {target_provider} failed, trying fallback...")
            
            for fallback_provider in self.registry.get_all_providers():
                if fallback_provider != target_provider:
                    try:
                        logger.info(f"🔄 Trying fallback provider: {fallback_provider}")
                        response = await self.universal.generate(
                            provider_name=fallback_provider,
                            prompt=prompt,
                            max_tokens=max_tokens
                        )
                        return response
                    except:
                        continue
            
            # All providers failed
            raise ProviderError(
                "All AI providers failed",
                details={'attempted': list(self.registry.get_all_providers().keys())}
            )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.registry.get_all_providers().keys())
    
    async def initialize_external_benchmarks(self, db):
        """
        Initialize external benchmarking system
        Called during server startup
        
        Args:
            db: MongoDB database instance
        """
        from core.external_benchmarks import get_external_benchmarks
        
        try:
            api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
            
            self.external_benchmarks = get_external_benchmarks(db, api_key)
            
            # Fetch initial rankings for all categories
            logger.info("Fetching initial rankings from external APIs...")
            for category in ["coding", "math", "reasoning", "research", "empathy", "general"]:
                await self.external_benchmarks.get_rankings(category)
            
            # Start background update task
            import asyncio
            asyncio.create_task(
                self.external_benchmarks.schedule_periodic_updates(interval_hours=12)
            )
            
            logger.info("✅ External benchmarking initialized with periodic updates")
            
        except Exception as e:
            logger.error(f"⚠️ External benchmarking initialization failed: {e}")
            logger.info("Will continue with simple routing")
    
    async def select_best_provider_for_category(
        self,
        category: str,
        emotion_state: Optional[EmotionState] = None
    ) -> str:
        """
        Select best provider for a category using external benchmarks
        
        Priority:
        1. Use external benchmarks (Artificial Analysis, LLM-Stats)
        2. Fall back to default provider if benchmarks unavailable
        
        Args:
            category: Task category (coding, math, reasoning, etc.)
            emotion_state: User's emotion state (for empathy routing)
        
        Returns:
            Provider name
        """
        
        # If external benchmarks available, use them
        if self.external_benchmarks:
            try:
                # Get rankings for category
                rankings = await self.external_benchmarks.get_rankings(category)
                
                if rankings:
                    # Get available providers
                    available = set(self.get_available_providers())
                    
                    # Find best available provider
                    for ranking in rankings:
                        if ranking.provider in available:
                            logger.info(
                                f"🎯 Selected {ranking.provider} for {category} "
                                f"(rank #{ranking.rank}, score: {ranking.score:.1f}, "
                                f"source: {ranking.source})"
                            )
                            return ranking.provider
                    
                    logger.warning(
                        f"No top-ranked providers available for {category}, "
                        f"using default"
                    )
            
            except Exception as e:
                logger.error(f"Error selecting provider from benchmarks: {e}")
        
        # Fallback to default provider
        return self._default_provider or self.get_available_providers()[0]
    
    def detect_category_from_message(self, message: str, emotion_state: Optional[EmotionState] = None) -> str:
        """
        Detect task category from message content
        
        Categories:
        - coding: Programming, algorithms, debugging
        - math: Mathematics, calculations
        - reasoning: Logic, analysis
        - research: Knowledge, facts, explanations
        - empathy: Emotional support (based on emotion state)
        - general: Default category
        
        Args:
            message: User's message
            emotion_state: User's emotion state
        
        Returns:
            Category name
        """
        
        message_lower = message.lower()
        
        # Coding indicators
        coding_keywords = [
            'code', 'program', 'function', 'algorithm', 'debug',
            'python', 'javascript', 'java', 'c++', 'recursion',
            'loop', 'variable', 'class', 'api', 'database', 'sql'
        ]
        if any(kw in message_lower for kw in coding_keywords):
            return "coding"
        
        # Math indicators
        math_keywords = [
            'calculate', 'solve', 'equation', 'formula', 'theorem',
            'integral', 'derivative', 'matrix', 'probability',
            'geometry', 'algebra', 'calculus', 'trigonometry'
        ]
        if any(kw in message_lower for kw in math_keywords):
            return "math"
        
        # Reasoning indicators
        reasoning_keywords = [
            'analyze', 'evaluate', 'compare', 'conclude', 'infer',
            'logic', 'argument', 'reasoning', 'critical thinking',
            'deduce', 'reasoning'
        ]
        if any(kw in message_lower for kw in reasoning_keywords):
            return "reasoning"
        
        # Research indicators
        research_keywords = [
            'what is', 'explain', 'describe', 'how does', 'why is',
            'history of', 'definition', 'meaning', 'overview', 'tell me about'
        ]
        if any(kw in message_lower for kw in research_keywords):
            return "research"
        
        # Empathy indicators (based on emotion)
        if emotion_state and hasattr(emotion_state, 'primary_emotion'):
            negative_emotions = [
                'frustration', 'anxiety', 'overwhelmed', 
                'sadness', 'confusion', 'stress'
            ]
            if emotion_state.primary_emotion in negative_emotions:
                return "empathy"
        
        # Default
        return "general"
