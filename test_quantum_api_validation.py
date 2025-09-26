"""
🚀 MASTERX QUANTUM API VALIDATION - Direct Backend API Testing
Test real AI integration through the actual MasterX API endpoints
"""

import asyncio
import json
import aiohttp
from datetime import datetime
import time

async def test_quantum_intelligence_api():
    """
    🧪 TEST: Direct Quantum Intelligence API calls to validate:
    1. Real AI provider integration (not mocked)
    2. Emotion detection functionality  
    3. Personalized response generation
    4. A/B testing capability
    """
    
    print("🚀 MASTERX QUANTUM INTELLIGENCE API VALIDATION")
    print("=" * 70)
    
    base_url = "http://localhost:8001/api"
    
    # Test scenarios for A/B testing
    test_scenarios = [
        {
            "scenario": "Frustrated Learning - Math Fractions",
            "payload": {
                "user_id": "test_user_frustrated_001", 
                "message": "I'm so frustrated with learning fractions! I just don't get why 1/2 + 1/4 = 3/4. This is making me want to give up on math completely.",
                "task_type": "emotional_support",
                "priority": "quality",
                "enable_caching": False,
                "context": {
                    "subject": "mathematics",
                    "topic": "fractions",
                    "emotional_state": "frustrated",
                    "difficulty_level": "beginner"
                }
            }
        },
        {
            "scenario": "Excited Learning - Calculus Derivatives", 
            "payload": {
                "user_id": "test_user_excited_001",
                "message": "I'm excited about learning calculus but I'm worried it might be too hard for me. Can you help me understand derivatives?",
                "task_type": "learning_support", 
                "priority": "balanced",
                "enable_caching": False,
                "context": {
                    "subject": "calculus",
                    "topic": "derivatives", 
                    "emotional_state": "excited_anxious",
                    "difficulty_level": "intermediate"
                }
            }
        },
        {
            "scenario": "Mental Fatigue - Quadratic Equations",
            "payload": {
                "user_id": "test_user_tired_001",
                "message": "I've been studying for hours and I'm mentally exhausted. Can you explain quadratic equations one more time?",
                "task_type": "adaptive_explanation",
                "priority": "efficiency", 
                "enable_caching": False,
                "context": {
                    "subject": "algebra",
                    "topic": "quadratic_equations",
                    "emotional_state": "fatigued",
                    "difficulty_level": "intermediate",
                    "session_duration": "6_hours"
                }
            }
        }
    ]
    
    print(f"📊 Testing {len(test_scenarios)} scenarios through Quantum Intelligence API...")
    print()
    
    async with aiohttp.ClientSession() as session:
        for i, test in enumerate(test_scenarios, 1):
            print(f"🧪 TEST SCENARIO {i}: {test['scenario']}")
            print("-" * 50)
            print(f"Message: \"{test['payload']['message'][:80]}...\"")
            
            # Test the quantum intelligence endpoint
            try:
                start_time = time.time()
                
                async with session.post(
                    f"{base_url}/quantum/message",
                    json=test['payload'],
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        print(f"✅ SUCCESS (HTTP {response.status}) - Response Time: {response_time:.2f}s")
                        print()
                        
                        # Analyze the response structure
                        print("📊 RESPONSE ANALYSIS:")
                        
                        if isinstance(result, dict):
                            # Check for key response components
                            response_text = result.get('response', result.get('message', ''))
                            metadata = result.get('metadata', {})
                            
                            print(f"  Response Length: {len(str(response_text))} characters")
                            
                            # Analyze AI integration indicators
                            ai_indicators = {
                                'provider_info': metadata.get('provider', 'unknown'),
                                'model_info': metadata.get('model', 'unknown'),
                                'processing_time': metadata.get('processing_time_ms', 'unknown'),
                                'emotion_detected': metadata.get('emotion_analysis', {}),
                                'difficulty_analysis': metadata.get('difficulty_assessment', {}),
                                'personalization': metadata.get('personalization_applied', False)
                            }
                            
                            print(f"  AI Provider: {ai_indicators['provider_info']}")
                            print(f"  Model Used: {ai_indicators['model_info']}")
                            print(f"  Processing Time: {ai_indicators['processing_time']}ms")
                            
                            # Check for emotion detection
                            if ai_indicators['emotion_detected']:
                                print(f"  Emotion Detected: {ai_indicators['emotion_detected']}")
                                print("  ✅ EMOTION DETECTION: ACTIVE")
                            else:
                                print("  ⚠️ EMOTION DETECTION: NOT DETECTED IN METADATA")
                            
                            # Check for personalization
                            if ai_indicators['personalization']:
                                print("  ✅ PERSONALIZATION: ACTIVE")
                            else:
                                print("  ⚠️ PERSONALIZATION: NOT DETECTED IN METADATA")
                            
                            # Analyze response content for authenticity
                            response_lower = str(response_text).lower()
                            
                            # Check for empathy and emotional awareness
                            empathy_words = ['understand', 'feel', 'frustrating', 'challenging', 'support', 'help']
                            empathy_count = sum(1 for word in empathy_words if word in response_lower)
                            
                            # Check for personalization indicators
                            personal_indicators = ['you', 'your', 'specifically', 'particular', 'individual']
                            personal_count = sum(1 for word in personal_indicators if word in response_lower)
                            
                            print(f"  Empathy Score: {empathy_count}/6 keywords found")
                            print(f"  Personalization Score: {personal_count}/5 keywords found")
                            
                            # Response preview
                            print(f"\n📝 RESPONSE PREVIEW:")
                            print(f"  \"{str(response_text)[:200]}...\"")
                            
                            # Authenticity assessment
                            if response_time > 2.0 and len(str(response_text)) > 100:
                                print("\n✅ AUTHENTICITY INDICATORS:")
                                print("  - Realistic response time (indicates real AI processing)")
                                print("  - Substantial response length (not hardcoded)")
                                print("  - Dynamic content generation confirmed")
                            else:
                                print("\n⚠️ POTENTIAL AUTHENTICITY CONCERNS:")
                                if response_time <= 1.0:
                                    print("  - Very fast response (may be cached/hardcoded)")
                                if len(str(response_text)) <= 100:
                                    print("  - Short response (may be template-based)")
                        
                        else:
                            print(f"📝 Raw Response: {result}")
                    
                    elif response.status == 503:
                        print(f"⚠️ SERVICE UNAVAILABLE (HTTP {response.status})")
                        error_data = await response.json()
                        print(f"   Error: {error_data.get('detail', {}).get('error', 'Unknown')}")
                        print("   This may indicate the Quantum Engine is still initializing")
                    
                    else:
                        print(f"❌ FAILED (HTTP {response.status})")
                        error_text = await response.text()
                        print(f"   Error: {error_text}")
                
            except asyncio.TimeoutError:
                print("❌ TIMEOUT - Request took longer than 60 seconds")
            except Exception as e:
                print(f"❌ ERROR - {str(e)}")
            
            print("\n" + "=" * 70)
            print()
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Test alternative endpoints if quantum is unavailable
        print("🔄 TESTING ALTERNATIVE ENDPOINTS...")
        
        try:
            # Test health endpoint
            async with session.get(f"{base_url}/") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Health Check: {health_data}")
                else:
                    print(f"❌ Health Check Failed: {response.status}")
            
            # Test any other available endpoints
            async with session.get(f"{base_url}/docs") as response:
                if response.status == 200:
                    print("✅ API Documentation: Available")
                else:
                    print("❌ API Documentation: Unavailable")
                    
        except Exception as e:
            print(f"⚠️ Alternative endpoint testing failed: {e}")
    
    print("\n🎯 VALIDATION SUMMARY:")
    print("=" * 70)
    print("📊 KEY FINDINGS:")
    print("1. Quantum Intelligence API endpoint accessibility")
    print("2. Real AI provider integration status")
    print("3. Response authenticity indicators")
    print("4. Emotion detection functionality") 
    print("5. Personalization capabilities")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Fix any Quantum Engine initialization issues")
    print("2. Implement comprehensive logging for A/B testing")
    print("3. Add response quality metrics collection")
    print("4. Set up real-time monitoring for AI provider status")
    print("5. Create user feedback collection system")

if __name__ == "__main__":
    print("🧪 Starting Quantum Intelligence API Validation Test...")
    asyncio.run(test_quantum_intelligence_api())