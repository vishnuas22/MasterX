#!/usr/bin/env python3
"""
MasterX Comparative LLM Adaptive Learning Test
Test different AI providers to find the best for adaptive learning
"""

import requests
import json
import time
import sys
from datetime import datetime

class ComparativeLLMTester:
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/ai/test"
        self.provider_results = {}
        
    def force_provider_test(self, message, context, provider_preference=None):
        """Test with specific provider preference"""
        try:
            payload = {"message": message}
            if context:
                payload["context"] = context
            if provider_preference:
                payload["provider_preference"] = provider_preference
                
            response = requests.post(self.api_url, json=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return {
                        "success": True,
                        "content": data.get("response", ""),
                        "provider": data.get("provider", "unknown"),
                        "model": data.get("model", "unknown"),
                        "response_time": data.get("response_time", 0),
                        "tokens_used": data.get("tokens_used", 0),
                        "confidence": data.get("confidence", 0)
                    }
                    
        except Exception as e:
            pass
            
        return {"success": False, "error": "Failed"}

    def test_empathy_and_simplification(self, provider_name="unknown"):
        """Test empathy and simplification capabilities"""
        print(f"\n🧠 Testing {provider_name} - Empathy & Simplification")
        print("-" * 50)
        
        # Frustrated learner scenario
        response = self.force_provider_test(
            "I'm really frustrated! I've been trying to understand variables in Python for hours and I still don't get it. Everyone says it's easy but it makes no sense to me. I'm starting to think I'm too stupid for programming.",
            "The user is frustrated and has low confidence. Show empathy, provide encouragement, and break down the concept of variables into the simplest possible terms with relatable analogies."
        )
        
        if response["success"]:
            content = response["content"]
            
            # Analyze empathy
            empathy_indicators = [
                "understand", "frustrating", "don't worry", "it's okay", "normal",
                "many people", "not stupid", "everyone struggles", "take your time"
            ]
            empathy_score = sum(1 for indicator in empathy_indicators if indicator.lower() in content.lower())
            
            # Analyze simplification
            simple_analogies = [
                "like a", "imagine", "think of", "container", "box", "label",
                "name tag", "storage", "drawer", "envelope"
            ]
            analogy_score = sum(1 for analogy in simple_analogies if analogy.lower() in content.lower())
            
            # Check for technical jargon (should be minimal)
            tech_jargon = [
                "instantiate", "allocate", "reference", "pointer", "memory address",
                "object identifier", "namespace", "scope resolution"
            ]
            jargon_penalty = sum(1 for jargon in tech_jargon if jargon.lower() in content.lower())
            
            empathy_score = min(10, empathy_score * 2)
            simplification_score = min(10, analogy_score * 2.5)
            clarity_score = max(0, 10 - jargon_penalty * 2)
            
            overall_score = (empathy_score + simplification_score + clarity_score) / 3
            
            print(f"   Empathy Score: {empathy_score}/10")
            print(f"   Simplification Score: {simplification_score}/10")
            print(f"   Clarity Score: {clarity_score}/10")
            print(f"   Overall Empathy & Simplification: {overall_score:.1f}/10")
            
            return {
                "empathy_score": empathy_score,
                "simplification_score": simplification_score,
                "clarity_score": clarity_score,
                "overall_score": overall_score,
                "response_time": response["response_time"],
                "provider": response["provider"],
                "model": response["model"]
            }
            
        return {"overall_score": 0, "error": "Failed to get response"}

    def test_context_memory_and_personalization(self, provider_name="unknown"):
        """Test context memory and personalization"""
        print(f"\n🧠 Testing {provider_name} - Context Memory & Personalization")
        print("-" * 50)
        
        # Establish context
        context_response = self.force_provider_test(
            "Hi! I'm Alex, a 25-year-old graphic designer from Canada. I want to learn Python programming to automate my design workflows, specifically for batch image processing and creating data visualizations for my clients. I have no programming experience but I'm good with creative tools like Photoshop and Illustrator.",
            "Remember all details about this user for future interactions: name, age, profession, location, goals, and background."
        )
        
        time.sleep(2)
        
        # Test memory in follow-up
        memory_response = self.force_provider_test(
            "What Python libraries would be most useful for someone like me?",
            "The user asked a follow-up question. Use the context you learned about them to provide personalized recommendations."
        )
        
        if context_response["success"] and memory_response["success"]:
            memory_content = memory_response["content"]
            
            # Check for personal details retention
            personal_details = ["Alex", "graphic designer", "design", "Canada", "automate", "workflow"]
            personal_score = sum(2 for detail in personal_details if detail.lower() in memory_content.lower())
            
            # Check for relevant library suggestions
            relevant_libs = [
                "PIL", "Pillow", "opencv", "matplotlib", "seaborn", "plotly",
                "pandas", "numpy", "automation", "batch", "image processing"
            ]
            relevance_score = sum(1 for lib in relevant_libs if lib.lower() in memory_content.lower())
            
            # Check for personalized language
            personalized_language = [
                "for your", "as a designer", "in your field", "for design work",
                "graphics", "visual", "creative", "workflow"
            ]
            personalization_score = sum(1 for phrase in personalized_language if phrase.lower() in memory_content.lower())
            
            memory_score = min(10, personal_score)
            relevance_score = min(10, relevance_score * 1.5)
            personalization_score = min(10, personalization_score * 2)
            
            overall_score = (memory_score + relevance_score + personalization_score) / 3
            
            print(f"   Memory Retention Score: {memory_score}/10")
            print(f"   Relevance Score: {relevance_score}/10") 
            print(f"   Personalization Score: {personalization_score}/10")
            print(f"   Overall Context & Memory: {overall_score:.1f}/10")
            
            return {
                "memory_score": memory_score,
                "relevance_score": relevance_score,
                "personalization_score": personalization_score,
                "overall_score": overall_score,
                "response_time": (context_response["response_time"] + memory_response["response_time"]) / 2,
                "provider": memory_response["provider"],
                "model": memory_response["model"]
            }
            
        return {"overall_score": 0, "error": "Failed to complete context test"}

    def test_adaptive_complexity_scaling(self, provider_name="unknown"):
        """Test adaptive complexity scaling"""
        print(f"\n🧠 Testing {provider_name} - Adaptive Complexity Scaling")
        print("-" * 50)
        
        # Test beginner response
        beginner_response = self.force_provider_test(
            "I just started programming yesterday. Can you explain what a function is?",
            "This is a complete beginner who started programming yesterday. Use very simple language and basic analogies."
        )
        
        time.sleep(2)
        
        # Test advanced response  
        advanced_response = self.force_provider_test(
            "I'm a senior software engineer with 10 years experience. Can you explain advanced function concepts like decorators, closures, and higher-order functions in Python?",
            "This is an experienced programmer who wants advanced concepts. Use technical terminology and provide sophisticated examples."
        )
        
        if beginner_response["success"] and advanced_response["success"]:
            beginner_content = beginner_response["content"]
            advanced_content = advanced_response["content"]
            
            # Analyze beginner response simplicity
            beginner_simple_words = ["like", "think of", "imagine", "simply", "basically", "just", "easy"]
            beginner_simplicity = sum(1 for word in beginner_simple_words if word.lower() in beginner_content.lower())
            
            # Check for advanced technical terms in advanced response
            advanced_terms = [
                "closure", "decorator", "higher-order", "lambda", "scope", "namespace",
                "first-class", "callback", "functional programming", "lexical scoping"
            ]
            advanced_complexity = sum(1 for term in advanced_terms if term.lower() in advanced_content.lower())
            
            # Check word complexity difference
            beginner_words = len(beginner_content.split())
            advanced_words = len(advanced_content.split())
            complexity_ratio = advanced_words / max(beginner_words, 1) if beginner_words > 0 else 0
            
            simplicity_score = min(10, beginner_simplicity * 2)
            complexity_score = min(10, advanced_complexity * 1.5)
            scaling_score = min(10, complexity_ratio * 3) if complexity_ratio > 1 else 0
            
            overall_score = (simplicity_score + complexity_score + scaling_score) / 3
            
            print(f"   Beginner Simplicity Score: {simplicity_score}/10")
            print(f"   Advanced Complexity Score: {complexity_score}/10")
            print(f"   Adaptive Scaling Score: {scaling_score}/10")
            print(f"   Overall Complexity Adaptation: {overall_score:.1f}/10")
            
            return {
                "simplicity_score": simplicity_score,
                "complexity_score": complexity_score,
                "scaling_score": scaling_score,
                "overall_score": overall_score,
                "response_time": (beginner_response["response_time"] + advanced_response["response_time"]) / 2,
                "provider": advanced_response["provider"],
                "model": advanced_response["model"]
            }
            
        return {"overall_score": 0, "error": "Failed to complete complexity test"}

    def test_learning_style_adaptation(self, provider_name="unknown"):
        """Test learning style adaptation"""
        print(f"\n🧠 Testing {provider_name} - Learning Style Adaptation")
        print("-" * 50)
        
        response = self.force_provider_test(
            "I learn best through hands-on practice and examples rather than theory. Can you teach me Python loops by showing me practical code I can try, with step-by-step instructions I can follow?",
            "This user learns through hands-on practice. Provide practical code examples with clear step-by-step instructions they can follow and modify."
        )
        
        if response["success"]:
            content = response["content"]
            
            # Check for code examples
            code_indicators = ["```", "print(", "for ", "while ", "range(", ":", "def "]
            code_score = sum(2 for indicator in code_indicators if indicator in content)
            
            # Check for step-by-step structure
            step_indicators = ["step 1", "step 2", "first", "next", "then", "finally", "1.", "2.", "3."]
            step_score = sum(1 for indicator in step_indicators if indicator.lower() in content.lower())
            
            # Check for hands-on language
            hands_on_indicators = ["try", "practice", "run", "execute", "modify", "experiment", "test", "change"]
            hands_on_score = sum(1 for indicator in hands_on_indicators if indicator.lower() in content.lower())
            
            # Check for practical examples
            practical_indicators = ["example", "practical", "real", "useful", "project", "exercise", "activity"]
            practical_score = sum(1 for indicator in practical_indicators if indicator.lower() in content.lower())
            
            code_score = min(10, code_score)
            step_score = min(10, step_score * 1.5)
            hands_on_score = min(10, hands_on_score * 1.5)
            practical_score = min(10, practical_score * 1.5)
            
            overall_score = (code_score + step_score + hands_on_score + practical_score) / 4
            
            print(f"   Code Examples Score: {code_score}/10")
            print(f"   Step-by-Step Score: {step_score}/10")
            print(f"   Hands-on Language Score: {hands_on_score}/10")
            print(f"   Practical Examples Score: {practical_score}/10")
            print(f"   Overall Learning Style Adaptation: {overall_score:.1f}/10")
            
            return {
                "code_score": code_score,
                "step_score": step_score,
                "hands_on_score": hands_on_score,
                "practical_score": practical_score,
                "overall_score": overall_score,
                "response_time": response["response_time"],
                "provider": response["provider"],
                "model": response["model"]
            }
            
        return {"overall_score": 0, "error": "Failed to get response"}

    def run_comparative_test(self):
        """Run comparative test across multiple attempts to get different providers"""
        print("🧪 MASTERX COMPARATIVE LLM ADAPTIVE LEARNING TEST")
        print("=" * 80)
        print("Testing multiple AI providers to find the best for adaptive learning...")
        print()
        
        # Collect results from multiple test runs
        all_results = []
        
        # Run multiple iterations to potentially trigger different providers
        for iteration in range(8):
            print(f"🔄 Test Iteration {iteration + 1}/8")
            
            # Test all adaptive capabilities
            empathy_result = self.test_empathy_and_simplification(f"Iteration-{iteration+1}")
            context_result = self.test_context_memory_and_personalization(f"Iteration-{iteration+1}")
            complexity_result = self.test_adaptive_complexity_scaling(f"Iteration-{iteration+1}")
            learning_style_result = self.test_learning_style_adaptation(f"Iteration-{iteration+1}")
            
            # Calculate iteration summary
            iteration_scores = []
            if empathy_result.get("overall_score", 0) > 0:
                iteration_scores.append(empathy_result["overall_score"])
            if context_result.get("overall_score", 0) > 0:
                iteration_scores.append(context_result["overall_score"])
            if complexity_result.get("overall_score", 0) > 0:
                iteration_scores.append(complexity_result["overall_score"])
            if learning_style_result.get("overall_score", 0) > 0:
                iteration_scores.append(learning_style_result["overall_score"])
            
            avg_score = sum(iteration_scores) / len(iteration_scores) if iteration_scores else 0
            
            # Get provider info
            providers_used = set()
            for result in [empathy_result, context_result, complexity_result, learning_style_result]:
                if result.get("provider"):
                    providers_used.add(f"{result['provider']}:{result.get('model', 'unknown')}")
            
            iteration_data = {
                "iteration": iteration + 1,
                "empathy": empathy_result,
                "context": context_result,
                "complexity": complexity_result,
                "learning_style": learning_style_result,
                "avg_score": avg_score,
                "providers": list(providers_used)
            }
            
            all_results.append(iteration_data)
            print(f"   Iteration {iteration + 1} Average Score: {avg_score:.1f}/10")
            print(f"   Providers Used: {', '.join(providers_used) if providers_used else 'Unknown'}")
            print()
            
            # Brief pause between iterations
            time.sleep(1)
        
        # Analyze results by provider
        print("=" * 80)
        print("📊 COMPARATIVE ANALYSIS BY PROVIDER")
        print("=" * 80)
        
        provider_performance = {}
        
        for iteration_data in all_results:
            for test_type in ["empathy", "context", "complexity", "learning_style"]:
                result = iteration_data[test_type]
                if result.get("provider") and result.get("overall_score", 0) > 0:
                    provider_key = f"{result['provider']}:{result.get('model', 'unknown')}"
                    
                    if provider_key not in provider_performance:
                        provider_performance[provider_key] = {
                            "empathy_scores": [],
                            "context_scores": [],
                            "complexity_scores": [],
                            "learning_style_scores": [],
                            "response_times": [],
                            "total_tests": 0
                        }
                    
                    provider_performance[provider_key][f"{test_type}_scores"].append(result["overall_score"])
                    provider_performance[provider_key]["response_times"].append(result.get("response_time", 0))
                    provider_performance[provider_key]["total_tests"] += 1
        
        # Calculate provider averages and rankings
        provider_rankings = []
        
        for provider, data in provider_performance.items():
            empathy_avg = sum(data["empathy_scores"]) / len(data["empathy_scores"]) if data["empathy_scores"] else 0
            context_avg = sum(data["context_scores"]) / len(data["context_scores"]) if data["context_scores"] else 0
            complexity_avg = sum(data["complexity_scores"]) / len(data["complexity_scores"]) if data["complexity_scores"] else 0
            learning_avg = sum(data["learning_style_scores"]) / len(data["learning_style_scores"]) if data["learning_style_scores"] else 0
            response_time_avg = sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else 0
            
            overall_adaptive_score = (empathy_avg + context_avg + complexity_avg + learning_avg) / 4
            
            provider_rankings.append({
                "provider": provider,
                "overall_score": overall_adaptive_score,
                "empathy_avg": empathy_avg,
                "context_avg": context_avg,
                "complexity_avg": complexity_avg,
                "learning_avg": learning_avg,
                "response_time": response_time_avg,
                "total_tests": data["total_tests"]
            })
        
        # Sort by overall score
        provider_rankings.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Display results
        for i, provider_data in enumerate(provider_rankings):
            rank = i + 1
            provider = provider_data["provider"]
            
            print(f"🏆 RANK #{rank}: {provider}")
            print(f"   📊 Overall Adaptive Learning Score: {provider_data['overall_score']:.1f}/10")
            print(f"   🧠 Empathy & Simplification: {provider_data['empathy_avg']:.1f}/10")
            print(f"   🔍 Context & Memory: {provider_data['context_avg']:.1f}/10")
            print(f"   📈 Complexity Scaling: {provider_data['complexity_avg']:.1f}/10")
            print(f"   🎯 Learning Style Adaptation: {provider_data['learning_avg']:.1f}/10")
            print(f"   ⚡ Average Response Time: {provider_data['response_time']:.2f}s")
            print(f"   📋 Total Tests Completed: {provider_data['total_tests']}")
            print()
        
        # Recommendation
        if provider_rankings:
            best_provider = provider_rankings[0]
            
            print("🎯 RECOMMENDATION FOR PRIMARY AI PROVIDER:")
            print("=" * 60)
            print(f"🥇 BEST: {best_provider['provider']}")
            print(f"   Score: {best_provider['overall_score']:.1f}/10")
            
            if best_provider['overall_score'] >= 8.0:
                print("   ✅ EXCELLENT adaptive learning capabilities!")
            elif best_provider['overall_score'] >= 6.0:
                print("   ✅ GOOD adaptive learning with room for improvement")
            elif best_provider['overall_score'] >= 4.0:
                print("   ⚠️  MODERATE adaptive learning - consider optimization")
            else:
                print("   ❌ POOR adaptive learning - needs significant improvement")
            
            # Specific recommendations
            print(f"\n💡 SPECIFIC STRENGTHS:")
            if best_provider['empathy_avg'] >= 7.0:
                print("   ✅ Excellent empathy and emotional intelligence")
            if best_provider['context_avg'] >= 7.0:
                print("   ✅ Strong context memory and personalization")
            if best_provider['complexity_avg'] >= 7.0:
                print("   ✅ Good adaptive complexity scaling")
            if best_provider['learning_avg'] >= 7.0:
                print("   ✅ Effective learning style adaptation")
            
            print(f"\n📈 IMPROVEMENT OPPORTUNITIES:")
            if best_provider['empathy_avg'] < 6.0:
                print("   📈 Enhance empathy and emotional support")
            if best_provider['context_avg'] < 6.0:
                print("   📈 Improve context retention and memory")
            if best_provider['complexity_avg'] < 6.0:
                print("   📈 Better complexity scaling algorithms needed")
            if best_provider['learning_avg'] < 6.0:
                print("   📈 Enhance learning style recognition")
            
        else:
            print("❌ No provider data collected - check system configuration")
        
        return provider_rankings

def main():
    tester = ComparativeLLMTester()
    return tester.run_comparative_test()

if __name__ == "__main__":
    main()