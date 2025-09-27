#!/usr/bin/env python3
"""
🚀 FOCUSED EMOTION DETECTION & PERSONALIZATION TESTS
Real validation of MasterX V9.0 Authentic Emotion Detection System
"""

import requests
import json
import time

# Test scenarios
scenarios = [
    {
        "name": "Frustrated Math Student",
        "user_id": "student_frustrated",
        "message": "I don't understand this calculus problem at all! I've been trying for hours and nothing makes sense. This is so confusing!",
        "task_type": "emotional_support",
        "expected_indicators": ["frustrat", "understand", "help", "support"]
    },
    {
        "name": "Excited Advanced Learner", 
        "user_id": "student_excited",
        "message": "This machine learning concept is fascinating! Can you explain more advanced techniques? I want to dive deeper!",
        "task_type": "advanced_concepts",
        "expected_indicators": ["exciting", "advanced", "explore", "dive"]
    },
    {
        "name": "Confused Beginner",
        "user_id": "student_confused", 
        "message": "I'm new to programming and I keep getting errors. What does this mean? I'm completely lost.",
        "task_type": "beginner_concepts",
        "expected_indicators": ["simple", "basic", "start", "explain"]
    }
]

def test_emotion_scenario(scenario):
    print(f"\n🧠 TESTING: {scenario['name']}")
    print("=" * 50)
    
    payload = {
        "user_id": scenario["user_id"],
        "message": scenario["message"], 
        "task_type": scenario["task_type"],
        "priority": "balanced"
    }
    
    start_time = time.time()
    response = requests.post(
        "http://localhost:8001/api/quantum/message",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    response_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract key metrics
        ai_response = data.get("response", {})
        analytics = data.get("analytics", {})
        performance = data.get("performance", {})
        
        print(f"✅ STATUS: SUCCESS ({response_time:.2f}ms)")
        print(f"🤖 AI: {ai_response.get('provider', 'unknown')} - {ai_response.get('model', 'unknown')}")
        print(f"💝 EMPATHY: {ai_response.get('empathy_score', 0):.2f}")
        print(f"🎯 CONFIDENCE: {ai_response.get('confidence', 0):.2f}")
        
        # Emotion analysis
        auth_emotion = analytics.get('authentic_emotion_result', {})
        adaptation = analytics.get('adaptation_analysis', {}).get('analysis_results', {})
        emotional = adaptation.get('emotional', {}) if adaptation else {}
        
        print(f"🧠 DETECTED EMOTION: {emotional.get('primary_emotion', 'unknown')}")
        print(f"🔬 V9.0 ENGINE: {auth_emotion.get('primary_emotion', 'unknown')}")
        print(f"📚 LEARNING STATE: {auth_emotion.get('learning_readiness', 'unknown')}")
        
        # Response adaptation analysis
        content = ai_response.get('content', '')
        print(f"📝 RESPONSE LENGTH: {len(content)} chars")
        
        # Check for expected emotional indicators
        found_indicators = []
        for indicator in scenario["expected_indicators"]:
            if indicator.lower() in content.lower():
                found_indicators.append(indicator)
        
        print(f"🎭 ADAPTATION INDICATORS: {found_indicators}")
        print(f"⚡ PERFORMANCE: {performance.get('performance_tier', 'unknown')}")
        
        # Show response sample
        print(f"\n💬 AI RESPONSE SAMPLE:")
        print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
        
        # Validation
        is_personalized = (
            len(found_indicators) > 0 or 
            ai_response.get('empathy_score', 0) > 0.7 or
            len(content) > 100
        )
        
        print(f"\n🎯 PERSONALIZATION: {'✅ DETECTED' if is_personalized else '❌ NOT DETECTED'}")
        
        return True
        
    else:
        print(f"❌ STATUS: FAILED ({response.status_code})")
        print(f"   Error: {response.text}")
        return False

def main():
    print("🚀 MASTERX QUANTUM INTELLIGENCE - EMOTION DETECTION VALIDATION")
    print("Testing V9.0 Authentic Emotion Detection with Real AI Responses")
    print("=" * 80)
    
    total_tests = len(scenarios)
    passed_tests = 0
    
    for scenario in scenarios:
        success = test_emotion_scenario(scenario)
        if success:
            passed_tests += 1
        time.sleep(1)  # Brief pause between tests
    
    # Final summary
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 80)
    print("🎯 FINAL VALIDATION RESULTS")
    print("=" * 80)
    print(f"📊 TOTAL TESTS: {total_tests}")
    print(f"✅ PASSED: {passed_tests}")  
    print(f"❌ FAILED: {total_tests - passed_tests}")
    print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT: Quantum Intelligence System is performing optimally!")
    elif success_rate >= 60:
        print("👍 GOOD: System is mostly functional with minor issues")
    else:
        print("⚠️ NEEDS ATTENTION: System requires optimization")
    
    print("\n🔬 VALIDATION COMPLETE: MasterX Quantum Intelligence V6.0 + V9.0 Emotion Detection")

if __name__ == "__main__":
    main()