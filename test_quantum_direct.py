#!/usr/bin/env python3
"""
Test quantum intelligence initialization directly
"""

import asyncio
import sys
import os
import time

# Add backend to path
sys.path.insert(0, '/app/backend')

async def test_quantum_intelligence():
    print("🔍 Testing Quantum Intelligence Initialization...")
    
    try:
        # Import quantum intelligence components
        from quantum_intelligence.core.integrated_quantum_engine import UltraEnterpriseQuantumEngine
        from motor.motor_asyncio import AsyncIOMotorClient
        
        print("✅ Successfully imported UltraEnterpriseQuantumEngine")
        
        # Set up database connection
        print("🔗 Setting up database connection...")
        mongo_url = "mongodb://localhost:27017"
        mongo_client = AsyncIOMotorClient(mongo_url)
        db = mongo_client["masterx_quantum"]
        
        print("✅ Database connection established")
        
        # Set up API keys
        api_keys = {
            "GROQ_API_KEY": "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",
            "EMERGENT_LLM_KEY": "sk-emergent-f8dE0800cB67aAeAa2",
            "GEMINI_API_KEY": "AIzaSyCF0s5fIEz7hVgUxSiAO3MogQd_X2JWi2Q"
        }
        
        print("🚀 Initializing Quantum Intelligence...")
        
        # Initialize quantum intelligence with database
        quantum_ai = UltraEnterpriseQuantumEngine(database=db)
        
        # Initialize providers
        initialization_success = await quantum_ai.initialize(api_keys)
        
        if initialization_success:
            print("✅ Quantum Intelligence initialized successfully")
            
            # Test a simple message
            print("🧠 Testing message processing...")
            
            test_message = "Hello, can you help me understand quantum computing?"
            
            response = await quantum_ai.process_user_message(
                user_id="test_user_direct",
                user_message=test_message,
                initial_context={
                    "task_type": "general",
                    "priority": "quality",
                    "enable_caching": True
                }
            )
            
            print(f"✅ Response received from: {response.get('response', {}).get('provider', 'Unknown')}")
            print(f"✅ Response length: {len(response.get('response', {}).get('content', ''))}")
            print(f"✅ Response time: {response.get('performance', {}).get('total_processing_time_ms', 0):.2f}ms")
            print(f"✅ Circuit breaker status: {response.get('performance', {}).get('circuit_breaker_status', 'Unknown')}")
            
            # Test empathy detection
            print("💭 Testing empathy detection...")
            
            empathy_message = "I'm really struggling with my studies and feeling overwhelmed. Can you help?"
            
            empathy_response = await quantum_ai.process_user_message(
                user_id="test_user_empathy",
                user_message=empathy_message,
                initial_context={
                    "task_type": "emotional_support",
                    "priority": "quality",
                    "enable_caching": True
                }
            )
            
            empathy_score = empathy_response.get('response', {}).get('empathy_score', 0)
            print(f"✅ Empathy Score: {empathy_score}")
            print(f"✅ Provider used: {empathy_response.get('response', {}).get('provider', 'Unknown')}")
            
            # Test provider specialization
            print("⚡ Testing provider specialization...")
            
            quick_message = "What is 2+2?"
            
            quick_response = await quantum_ai.process_user_message(
                user_id="test_user_quick",
                user_message=quick_message,
                initial_context={
                    "task_type": "quick_response",
                    "priority": "speed",
                    "enable_caching": True
                }
            )
            
            print(f"✅ Quick response from: {quick_response.get('response', {}).get('provider', 'Unknown')}")
            print(f"✅ Quick response time: {quick_response.get('performance', {}).get('total_processing_time_ms', 0):.2f}ms")
            
            return True
            
        else:
            print("❌ Failed to initialize Quantum Intelligence")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Quantum Intelligence: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_quantum_intelligence())
    print(f"\n🎯 Test Result: {'SUCCESS' if success else 'FAILED'}")