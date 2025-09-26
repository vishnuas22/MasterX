#!/usr/bin/env python3
"""
🔍 AI PROVIDER INTEGRATION INVESTIGATION
Test to determine if MasterX is actually calling real AI providers or using mock responses
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_real_ai_integration():
    """Test if the quantum intelligence system is using real AI providers"""
    
    print("🔍 INVESTIGATING AI PROVIDER INTEGRATION")
    print("=" * 60)
    
    # Test 1: Simple question that should get different responses
    test_messages = [
        {
            "user_id": "test_investigator",
            "message": "Tell me exactly what time it is right now and generate a random number between 1 and 1000000",
            "session_id": "investigation_1", 
            "task_type": "general",
            "priority": "balanced",
            "enable_caching": False,  # Disable caching to ensure fresh responses
            "max_response_time_ms": 5000
        },
        {
            "user_id": "test_investigator", 
            "message": "Create a unique poem about quantum computing with exactly 4 lines and include the word 'nebula'",
            "session_id": "investigation_2",
            "task_type": "general", 
            "priority": "balanced",
            "enable_caching": False,
            "max_response_time_ms": 5000
        },
        {
            "user_id": "test_investigator",
            "message": "I'm feeling frustrated with learning algebra. Can you help me feel better?",
            "session_id": "investigation_3",
            "task_type": "emotional_support",
            "priority": "quality", 
            "enable_caching": False,
            "max_response_time_ms": 5000
        }
    ]
    
    responses = []
    
    async with aiohttp.ClientSession() as session:
        for i, test_msg in enumerate(test_messages, 1):
            print(f"📤 Test {i}: {test_msg['message'][:50]}...")
            start_time = time.time()
            
            try:
                async with session.post(
                    "http://localhost:8001/api/quantum/message",
                    json=test_msg,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        responses.append({
                            'test_id': i,
                            'message': test_msg['message'],
                            'task_type': test_msg['task_type'],
                            'response_time_ms': response_time,
                            'response_data': data,
                            'success': True
                        })
                        
                        # Extract key information
                        ai_response = data.get('response', {})
                        provider = ai_response.get('provider', 'unknown')
                        model = ai_response.get('model', 'unknown') 
                        content = ai_response.get('content', '')[:200] + '...' if len(ai_response.get('content', '')) > 200 else ai_response.get('content', '')
                        
                        print(f"✅ Response received ({response_time:.1f}ms)")
                        print(f"   🤖 Provider: {provider}")
                        print(f"   📱 Model: {model}")
                        print(f"   💬 Content: {content}")
                        
                        # Check for signs of real AI vs mock responses
                        performance_data = data.get('performance', {})
                        optimization_applied = data.get('processing_optimizations', [])
                        
                        if 'ultra_optimization' in optimization_applied:
                            print(f"   ⚠️  POTENTIAL MOCK: Ultra-optimization applied (possible optimized response)")
                        
                        if provider in ['groq', 'emergent_openai', 'emergent_anthropic']:
                            print(f"   ✅ REAL AI: Provider indicates actual AI service")
                        
                        if 'tokens_used' in ai_response and ai_response.get('tokens_used', 0) > 0:
                            print(f"   ✅ REAL AI: Token usage reported ({ai_response.get('tokens_used')} tokens)")
                        
                        print()
                        
                    else:
                        print(f"❌ Request failed: HTTP {response.status}")
                        error_text = await response.text()
                        print(f"   Error: {error_text[:200]}")
                        responses.append({
                            'test_id': i,
                            'message': test_msg['message'],
                            'success': False,
                            'error': f"HTTP {response.status}: {error_text}"
                        })
                        print()
                        
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                print(f"❌ Request failed: {str(e)}")
                responses.append({
                    'test_id': i,
                    'message': test_msg['message'],
                    'success': False,
                    'error': str(e),
                    'response_time_ms': response_time
                })
                print()
                
            # Small delay between requests
            await asyncio.sleep(2)
    
    # Analysis
    print("🔬 ANALYSIS RESULTS")
    print("=" * 60)
    
    successful_tests = [r for r in responses if r.get('success')]
    failed_tests = [r for r in responses if not r.get('success')]
    
    print(f"📊 Success Rate: {len(successful_tests)}/{len(responses)} tests passed")
    
    if successful_tests:
        print("\n🤖 AI PROVIDER ANALYSIS:")
        
        providers_used = set()
        models_used = set()
        real_ai_indicators = 0
        mock_indicators = 0
        
        for test in successful_tests:
            data = test.get('response_data', {})
            ai_response = data.get('response', {})
            provider = ai_response.get('provider', 'unknown')
            model = ai_response.get('model', 'unknown')
            
            providers_used.add(provider)
            models_used.add(model)
            
            # Look for real AI indicators
            if provider in ['groq', 'emergent_openai', 'emergent_anthropic', 'emergent_gemini']:
                real_ai_indicators += 1
            
            if ai_response.get('tokens_used', 0) > 0:
                real_ai_indicators += 1
                
            # Look for mock indicators
            optimization_applied = data.get('processing_optimizations', [])
            if 'ultra_optimization' in optimization_applied or 'response_optimization' in optimization_applied:
                mock_indicators += 1
        
        print(f"   Providers used: {', '.join(providers_used)}")
        print(f"   Models used: {', '.join(models_used)}")
        print(f"   Real AI indicators: {real_ai_indicators}")
        print(f"   Mock indicators: {mock_indicators}")
        
        # Final assessment
        if real_ai_indicators > mock_indicators:
            print("\n✅ CONCLUSION: System appears to be using REAL AI providers")
            print("   - Real provider names detected")
            print("   - Token usage reported")  
            print("   - Response patterns consistent with AI models")
        elif mock_indicators > real_ai_indicators:
            print("\n⚠️ CONCLUSION: System may be using OPTIMIZED/MOCK responses")
            print("   - Ultra-optimization flags detected")
            print("   - Potential pre-generated responses")
        else:
            print("\n🤔 CONCLUSION: Mixed indicators - further investigation needed")
        
        # Response uniqueness test
        if len(successful_tests) >= 2:
            print(f"\n📝 RESPONSE UNIQUENESS TEST:")
            responses_content = [test['response_data'].get('response', {}).get('content', '') for test in successful_tests]
            unique_responses = len(set(responses_content))
            print(f"   Unique responses: {unique_responses}/{len(successful_tests)}")
            
            if unique_responses == len(successful_tests):
                print("   ✅ All responses are unique (indicates real AI generation)")
            else:
                print("   ⚠️ Some responses are identical (possible mock/cached responses)")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for test in failed_tests:
            print(f"   Test {test['test_id']}: {test.get('error', 'Unknown error')}")
    
    return responses

if __name__ == "__main__":
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = asyncio.run(test_real_ai_integration())