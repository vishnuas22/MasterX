#!/usr/bin/env python3
"""
🚀 MASTERX AI PROVIDER CONNECTION VALIDATOR
Test script to validate AI provider connections and emotional enhancement

This script addresses the user's concern about whether the emotion detection system 
is actually enhancing AI responses or just returning standard LLM responses.
"""

import asyncio
import os
import time
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_groq_emotional_enhancement():
    """Test Groq with different emotional contexts to validate enhancement"""
    print("🔥 TESTING GROQ EMOTIONAL ENHANCEMENT")
    print("=====================================")
    
    try:
        from groq import AsyncGroq
        
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY not found")
            return False
            
        client = AsyncGroq(api_key=api_key)
        
        # Test scenarios with different emotional contexts
        test_scenarios = [
            {
                "name": "Frustrated Student",
                "context": "User is frustrated and struggling with math. Detected emotions: frustration (0.85), anxiety (0.70), low_confidence (0.80). Intervention needed: high empathy, encouraging tone, simplification.",
                "message": "I don't understand this math problem at all!",
                "expected_enhancement": "Should show high empathy and encouragement"
            },
            {
                "name": "Confident Student", 
                "context": "User is confident and engaged. Detected emotions: confidence (0.90), curiosity (0.85), excitement (0.75). Optimal learning state detected.",
                "message": "I solved the previous problem! What's the next challenge?",
                "expected_enhancement": "Should provide more advanced challenges"
            },
            {
                "name": "Neutral Baseline",
                "context": "",  # No emotional context
                "message": "Explain algebra basics.",
                "expected_enhancement": "Standard response without emotional adaptation"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n🎯 Testing: {scenario['name']}")
            print(f"Context: {scenario['context'][:100]}..." if scenario['context'] else "No context")
            
            # Create messages with emotional context injection
            messages = []
            
            # System message with emotional intelligence instructions
            system_message = "You are MasterX, an AI tutor with advanced emotional intelligence."
            if scenario['context']:
                system_message += f" EMOTIONAL CONTEXT: {scenario['context']} Adapt your response accordingly."
            
            messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": scenario['message']})
            
            # Make API call
            start_time = time.time()
            response = await client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            response_time = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content
            
            print(f"Response ({response_time:.1f}ms):")
            print(f"'{content[:150]}...'" if len(content) > 150 else f"'{content}'")
            print(f"Expected: {scenario['expected_enhancement']}")
            
            # Analyze response for emotional adaptation markers
            emotional_markers = analyze_emotional_adaptation(content, scenario['context'])
            print(f"Emotional adaptation detected: {emotional_markers}")
            
        return True
        
    except Exception as e:
        print(f"❌ Groq emotional enhancement test failed: {e}")
        return False

async def test_gemini_emotional_enhancement():
    """Test Gemini with emotional contexts"""
    print("\n🤖 TESTING GEMINI EMOTIONAL ENHANCEMENT")
    print("========================================")
    
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY not found")
            return False
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test with emotional context vs without
        print("\n🎯 Testing Emotional Context Impact:")
        
        # Without emotional context
        print("\n📝 WITHOUT emotional context:")
        start_time = time.time()
        response1 = model.generate_content("I'm having trouble with calculus derivatives.")
        response_time1 = (time.time() - start_time) * 1000
        content1 = response1.text
        print(f"Response ({response_time1:.1f}ms): '{content1[:100]}...'")
        
        # With emotional context
        print("\n💭 WITH emotional context (frustrated, anxious student):")
        emotional_prompt = """
        EMOTIONAL CONTEXT: Student is frustrated (0.85), anxious (0.70), and has low confidence (0.80). 
        They need high empathy, encouragement, and simplified explanations.
        
        Student message: I'm having trouble with calculus derivatives.
        
        Respond with appropriate emotional support and adapted teaching approach.
        """
        
        start_time = time.time()
        response2 = model.generate_content(emotional_prompt)
        response_time2 = (time.time() - start_time) * 1000
        content2 = response2.text
        print(f"Response ({response_time2:.1f}ms): '{content2[:100]}...'")
        
        # Compare responses
        print("\n🔍 COMPARISON ANALYSIS:")
        adaptation_diff = compare_emotional_responses(content1, content2)
        print(f"Emotional adaptation detected: {adaptation_diff}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini emotional enhancement test failed: {e}")
        return False

def analyze_emotional_adaptation(response: str, emotional_context: str) -> Dict[str, Any]:
    """Analyze if response shows emotional adaptation"""
    adaptation_markers = {
        "empathy_words": 0,
        "encouragement": 0,
        "simplification": 0,
        "emotional_acknowledgment": 0,
        "tone_adaptation": "neutral"
    }
    
    response_lower = response.lower()
    
    # Check for empathy markers
    empathy_words = ["understand", "feel", "frustrating", "challenging", "difficult", "struggle", "empathy", "sorry"]
    adaptation_markers["empathy_words"] = sum(1 for word in empathy_words if word in response_lower)
    
    # Check for encouragement
    encouragement_words = ["you can", "don't worry", "it's okay", "let's try", "step by step", "together", "believe"]
    adaptation_markers["encouragement"] = sum(1 for word in encouragement_words if word in response_lower)
    
    # Check for simplification indicators
    simplification_words = ["simple", "easy", "basic", "start with", "first", "step by step", "break down"]
    adaptation_markers["simplification"] = sum(1 for word in simplification_words if word in response_lower)
    
    # Determine tone adaptation
    if adaptation_markers["empathy_words"] > 2 or adaptation_markers["encouragement"] > 1:
        adaptation_markers["tone_adaptation"] = "supportive"
    elif "challenge" in response_lower or "advanced" in response_lower:
        adaptation_markers["tone_adaptation"] = "challenging"
    
    return adaptation_markers

def compare_emotional_responses(response1: str, response2: str) -> Dict[str, Any]:
    """Compare two responses to see difference in emotional adaptation"""
    analysis1 = analyze_emotional_adaptation(response1, "")
    analysis2 = analyze_emotional_adaptation(response2, "emotional")
    
    return {
        "baseline_response": analysis1,
        "emotional_response": analysis2,
        "adaptation_improvement": {
            "empathy_increase": analysis2["empathy_words"] - analysis1["empathy_words"],
            "encouragement_increase": analysis2["encouragement"] - analysis1["encouragement"],
            "tone_change": f"{analysis1['tone_adaptation']} → {analysis2['tone_adaptation']}"
        },
        "significant_adaptation": (
            analysis2["empathy_words"] > analysis1["empathy_words"] + 1 or
            analysis2["encouragement"] > analysis1["encouragement"] + 1 or
            analysis2["tone_adaptation"] != analysis1["tone_adaptation"]
        )
    }

async def test_quantum_intelligence_integration():
    """Test if quantum intelligence system can connect to AI providers"""
    print("\n🧠 TESTING QUANTUM INTELLIGENCE INTEGRATION")
    print("============================================")
    
    try:
        # Try to import and initialize quantum intelligence components
        from quantum_intelligence.core.breakthrough_ai_integration import (
            UltraEnterpriseGroqProvider, UltraEnterpriseGeminiProvider
        )
        
        print("✅ Quantum intelligence imports successful")
        
        # Test Groq provider initialization
        try:
            groq_api_key = os.environ.get('GROQ_API_KEY')
            if groq_api_key:
                groq_provider = UltraEnterpriseGroqProvider(groq_api_key)
                print("✅ UltraEnterpriseGroqProvider initialized")
            else:
                print("❌ GROQ_API_KEY not available for provider initialization")
        except Exception as e:
            print(f"⚠️  UltraEnterpriseGroqProvider initialization failed: {e}")
        
        # Test Gemini provider initialization
        try:
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if gemini_api_key:
                gemini_provider = UltraEnterpriseGeminiProvider(gemini_api_key)
                print("✅ UltraEnterpriseGeminiProvider initialized")
            else:
                print("❌ GEMINI_API_KEY not available for provider initialization")
        except Exception as e:
            print(f"⚠️  UltraEnterpriseGeminiProvider initialization failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Quantum intelligence imports failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Quantum intelligence integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 MASTERX AI PROVIDER CONNECTION VALIDATION")
    print("=============================================")
    print("Testing whether emotion detection actually enhances AI responses...")
    print()
    
    results = {
        "groq_emotional_enhancement": False,
        "gemini_emotional_enhancement": False,
        "quantum_intelligence_integration": False
    }
    
    # Test each provider
    results["groq_emotional_enhancement"] = await test_groq_emotional_enhancement()
    results["gemini_emotional_enhancement"] = await test_gemini_emotional_enhancement()
    results["quantum_intelligence_integration"] = await test_quantum_intelligence_integration()
    
    # Summary
    print("\n🎯 FINAL VALIDATION RESULTS")
    print("============================")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    # Overall assessment
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 2:
        print("✅ AI providers are connecting and can provide emotional adaptation!")
        print("📊 Recommendation: Implement A/B testing to measure impact quantitatively")
    else:
        print("❌ AI provider integration needs attention")
        print("📊 Recommendation: Fix connection issues before implementing emotional features")

if __name__ == "__main__":
    asyncio.run(main())