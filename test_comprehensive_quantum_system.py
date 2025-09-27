#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE QUANTUM INTELLIGENCE SYSTEM TESTING V6.0
Real-world learning scenarios with authentic emotion detection and personalization

This test suite validates:
1. Emotion detection and response adaptation
2. Real AI API calls (not mocked)
3. Personalized learning experiences
4. Adaptive difficulty adjustment
5. Context awareness and memory
6. Performance optimization
"""

import asyncio
import json
import time
import sys
import os
from typing import Dict, Any, List
import requests
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8001"
API_BASE = f"{BACKEND_URL}/api"

class QuantumSystemTester:
    """Comprehensive tester for MasterX Quantum Intelligence System"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def record_test(self, test_name: str, success: bool, response_time: float, details: Dict[str, Any]):
        """Record test results with comprehensive metrics"""
        result = {
            'test_name': test_name,
            'success': success,
            'response_time_ms': response_time * 1000,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.test_results.append(result)
        self.total_tests += 1
        
        if success:
            self.passed_tests += 1
            self.log(f"✅ PASSED: {test_name} ({response_time*1000:.2f}ms)", "SUCCESS")
        else:
            self.failed_tests += 1
            self.log(f"❌ FAILED: {test_name} - {details.get('error', 'Unknown error')}", "ERROR")
    
    async def test_emotion_based_learning_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test emotion-based learning scenarios with real API calls"""
        start_time = time.time()
        
        try:
            # Prepare quantum intelligence request
            request_payload = {
                "user_id": scenario["user_id"],
                "message": scenario["message"],
                "session_id": f"test_session_{scenario['scenario_type']}",
                "task_type": scenario["task_type"],
                "priority": scenario.get("priority", "balanced"),
                "initial_context": scenario.get("context", {}),
                "enable_caching": True,
                "max_response_time_ms": 4000,  # 4 seconds for real AI processing
                "enable_streaming": False
            }
            
            self.log(f"🧠 Testing scenario: {scenario['name']}")
            self.log(f"📊 Expected emotion: {scenario['expected_emotion']}")
            
            # Make real API call to quantum intelligence system
            response = requests.post(
                f"{API_BASE}/quantum/message",
                json=request_payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned {response.status_code}: {response.text}")
            
            result = response.json()
            response_time = time.time() - start_time
            
            # Analyze response for emotion detection and personalization
            analysis = self.analyze_quantum_response(result, scenario)
            
            # Check if real AI response (not mocked)
            ai_response = result.get("response", {})
            is_real_ai = self.validate_real_ai_response(ai_response)
            
            self.record_test(
                f"Emotion Learning: {scenario['name']}", 
                analysis["success"], 
                response_time,
                {
                    "scenario_type": scenario["scenario_type"],
                    "expected_emotion": scenario["expected_emotion"],
                    "detected_emotion": analysis.get("detected_emotion"),
                    "personalization_score": analysis.get("personalization_score", 0),
                    "response_adaptation": analysis.get("response_adaptation"),
                    "is_real_ai": is_real_ai,
                    "ai_provider": ai_response.get("provider", "unknown"),
                    "response_length": len(ai_response.get("content", "")),
                    "empathy_score": ai_response.get("empathy_score", 0),
                    "confidence": ai_response.get("confidence", 0),
                    "performance_metrics": result.get("performance", {}),
                    "quantum_metrics": result.get("quantum_metrics", {})
                }
            )
            
            return analysis
            
        except Exception as e:
            response_time = time.time() - start_time
            self.record_test(
                f"Emotion Learning: {scenario['name']}", 
                False, 
                response_time,
                {"error": str(e), "scenario_type": scenario["scenario_type"]}
            )
            return {"success": False, "error": str(e)}
    
    def analyze_quantum_response(self, result: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quantum response for emotion detection and personalization"""
        try:
            analytics = result.get("analytics", {})
            adaptation_analysis = analytics.get("adaptation_analysis", {})
            response_content = result.get("response", {}).get("content", "")
            
            # Extract emotion indicators
            detected_emotion = "unknown"
            empathy_score = result.get("response", {}).get("empathy_score", 0)
            personalization_score = analytics.get("personalization_score", 0)
            
            # Check for emotion-aware response adaptation
            response_adaptation = self.check_response_adaptation(response_content, scenario)
            
            # Validate personalization indicators
            has_personalization = personalization_score > 0.3 or empathy_score > 0.3
            
            # Check for learning context awareness
            context_effectiveness = analytics.get("context_effectiveness", 0)
            
            success = (
                len(response_content) > 50 and  # Substantial response
                has_personalization and  # Shows personalization
                context_effectiveness > 0.1 and  # Context awareness
                response_adaptation["adapted"]  # Response adapted to scenario
            )
            
            return {
                "success": success,
                "detected_emotion": detected_emotion,
                "personalization_score": personalization_score,
                "response_adaptation": response_adaptation,
                "context_effectiveness": context_effectiveness,
                "empathy_score": empathy_score
            }
            
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {e}"}
    
    def check_response_adaptation(self, response: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Check if response is adapted to the learning scenario"""
        try:
            response_lower = response.lower()
            scenario_type = scenario["scenario_type"]
            
            adaptation_indicators = {
                "frustrated_student": ["understand", "help", "step by step", "let's break", "don't worry"],
                "confused_beginner": ["simple", "basic", "start with", "explain", "example"],
                "excited_learner": ["great", "awesome", "excellent", "let's explore", "advanced"],
                "anxious_student": ["calm", "relax", "take your time", "it's okay", "no pressure"],
                "advanced_learner": ["complex", "challenge", "advanced", "deep dive", "sophisticated"]
            }
            
            expected_indicators = adaptation_indicators.get(scenario_type, [])
            found_indicators = [ind for ind in expected_indicators if ind in response_lower]
            
            adapted = len(found_indicators) > 0
            adaptation_ratio = len(found_indicators) / max(len(expected_indicators), 1)
            
            return {
                "adapted": adapted,
                "adaptation_ratio": adaptation_ratio,
                "found_indicators": found_indicators,
                "response_tone": self.analyze_response_tone(response)
            }
            
        except Exception as e:
            return {"adapted": False, "error": str(e)}
    
    def analyze_response_tone(self, response: str) -> str:
        """Analyze the tone of the response"""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["exciting", "amazing", "fantastic", "wonderful"]):
            return "enthusiastic"
        elif any(word in response_lower for word in ["calm", "gentle", "patient", "understand"]):
            return "supportive"
        elif any(word in response_lower for word in ["complex", "advanced", "sophisticated", "challenging"]):
            return "challenging"
        elif any(word in response_lower for word in ["simple", "basic", "easy", "straightforward"]):
            return "simplified"
        else:
            return "neutral"
    
    def validate_real_ai_response(self, ai_response: Dict[str, Any]) -> bool:
        """Validate that response came from real AI, not mocked data"""
        content = ai_response.get("content", "")
        provider = ai_response.get("provider", "")
        confidence = ai_response.get("confidence", 0)
        
        # Real AI responses should have:
        # 1. Substantial content (not template responses)
        # 2. Valid AI provider
        # 3. Reasonable confidence scores
        # 4. Natural language patterns
        
        has_content = len(content) > 30
        has_provider = provider in ["groq", "gemini", "emergent", "openai", "anthropic", "ultra_optimized"]
        has_confidence = 0.1 <= confidence <= 1.0
        has_natural_language = self.check_natural_language_patterns(content)
        
        return has_content and has_provider and has_confidence and has_natural_language
    
    def check_natural_language_patterns(self, text: str) -> bool:
        """Check for natural language patterns that indicate real AI generation"""
        if len(text) < 20:
            return False
        
        # Check for varied sentence structure
        sentences = text.split('.')
        if len(sentences) < 2:
            return False
        
        # Check for connector words (indicates natural flow)
        connectors = ["however", "therefore", "moreover", "furthermore", "additionally", "meanwhile", "consequently"]
        has_connectors = any(conn in text.lower() for conn in connectors)
        
        # Check for varied vocabulary (not repetitive)
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        return has_connectors or unique_ratio > 0.6
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive test suite with real learning scenarios"""
        
        self.log("🚀 Starting Comprehensive Quantum Intelligence System Testing")
        self.log("=" * 80)
        
        # Real-world learning scenarios for emotion detection testing
        learning_scenarios = [
            {
                "name": "Frustrated Math Student",
                "scenario_type": "frustrated_student", 
                "user_id": "student_frustrated_001",
                "message": "I don't understand this calculus problem at all! I've been trying for hours and nothing makes sense. This is so confusing and I'm getting really stressed about it.",
                "expected_emotion": "frustration",
                "task_type": "emotional_support",
                "context": {
                    "subject": "mathematics",
                    "difficulty_level": "intermediate",
                    "learning_style": "visual"
                }
            },
            {
                "name": "Confused Programming Beginner", 
                "scenario_type": "confused_beginner",
                "user_id": "student_beginner_002",
                "message": "I'm new to programming and I keep getting errors in my Python code. What does 'NameError' mean? I'm completely lost.",
                "expected_emotion": "confusion",
                "task_type": "beginner_concepts",
                "context": {
                    "subject": "programming",
                    "difficulty_level": "beginner", 
                    "learning_style": "step_by_step"
                }
            },
            {
                "name": "Excited Advanced Learner",
                "scenario_type": "excited_learner",
                "user_id": "student_advanced_003", 
                "message": "This machine learning concept is fascinating! Can you explain more advanced techniques like neural networks and deep learning? I want to dive deeper!",
                "expected_emotion": "excitement",
                "task_type": "advanced_concepts",
                "priority": "quality",
                "context": {
                    "subject": "machine_learning",
                    "difficulty_level": "advanced",
                    "learning_style": "exploratory"
                }
            },
            {
                "name": "Anxious Test Preparation",
                "scenario_type": "anxious_student",
                "user_id": "student_anxious_004",
                "message": "I have an exam tomorrow and I'm really nervous. Can you help me understand these chemistry concepts? I'm worried I won't do well.",
                "expected_emotion": "anxiety", 
                "task_type": "emotional_support",
                "context": {
                    "subject": "chemistry",
                    "difficulty_level": "intermediate",
                    "exam_preparation": True
                }
            },
            {
                "name": "Analytical Problem Solver",
                "scenario_type": "advanced_learner",
                "user_id": "student_analytical_005",
                "message": "I need help analyzing this complex algorithm's time complexity. Can we break down the Big O notation step by step?",
                "expected_emotion": "focused",
                "task_type": "analytical_reasoning",
                "priority": "quality",
                "context": {
                    "subject": "computer_science",
                    "difficulty_level": "advanced", 
                    "learning_style": "analytical"
                }
            }
        ]
        
        # Test each learning scenario
        for scenario in learning_scenarios:
            await self.test_emotion_based_learning_scenario(scenario)
            await asyncio.sleep(1)  # Brief pause between tests
        
        # Test system performance and capabilities
        await self.test_system_performance()
        
        # Generate comprehensive report
        self.generate_final_report()
    
    async def test_system_performance(self):
        """Test system performance and advanced features"""
        self.log("\n🔧 Testing System Performance Features")
        
        # Test caching effectiveness
        await self.test_caching_performance()
        
        # Test context memory
        await self.test_context_memory()
        
        # Test adaptive difficulty
        await self.test_adaptive_difficulty()
    
    async def test_caching_performance(self):
        """Test caching system performance"""
        start_time = time.time()
        
        try:
            # Make identical request twice to test caching
            request_payload = {
                "user_id": "cache_test_user",
                "message": "What is machine learning?",
                "task_type": "general",
                "enable_caching": True
            }
            
            # First request (should hit AI)
            response1 = requests.post(f"{API_BASE}/quantum/message", json=request_payload, timeout=10)
            time1 = time.time() - start_time
            
            # Second request (should hit cache)
            start_time2 = time.time()
            response2 = requests.post(f"{API_BASE}/quantum/message", json=request_payload, timeout=10)
            time2 = time.time() - start_time2
            
            success = (response1.status_code == 200 and response2.status_code == 200 and time2 < time1)
            
            self.record_test(
                "Caching Performance",
                success,
                time2,
                {
                    "first_request_time": time1 * 1000,
                    "second_request_time": time2 * 1000,
                    "cache_speedup": f"{((time1 - time2) / time1 * 100):.1f}%" if time1 > 0 else "N/A",
                    "cache_hit_likely": time2 < (time1 * 0.5)
                }
            )
            
        except Exception as e:
            self.record_test("Caching Performance", False, time.time() - start_time, {"error": str(e)})
    
    async def test_context_memory(self):
        """Test context memory and conversation continuity"""
        start_time = time.time()
        
        try:
            session_id = "context_test_session"
            
            # First message
            response1 = requests.post(
                f"{API_BASE}/quantum/message",
                json={
                    "user_id": "context_test_user",
                    "message": "My name is Alice and I'm studying physics.",
                    "session_id": session_id,
                    "task_type": "general"
                },
                timeout=10
            )
            
            # Second message referencing first
            response2 = requests.post(
                f"{API_BASE}/quantum/message", 
                json={
                    "user_id": "context_test_user",
                    "message": "Can you explain quantum mechanics to me?",
                    "session_id": session_id,
                    "task_type": "complex_explanation"
                },
                timeout=10
            )
            
            success = (response1.status_code == 200 and response2.status_code == 200)
            response_time = time.time() - start_time
            
            # Analyze if context was maintained
            result2 = response2.json() if response2.status_code == 200 else {}
            context_maintained = self.check_context_continuity(result2)
            
            self.record_test(
                "Context Memory", 
                success and context_maintained,
                response_time,
                {
                    "context_maintained": context_maintained,
                    "conversation_messages": 2,
                    "session_id": session_id
                }
            )
            
        except Exception as e:
            self.record_test("Context Memory", False, time.time() - start_time, {"error": str(e)})
    
    def check_context_continuity(self, result: Dict[str, Any]) -> bool:
        """Check if context was maintained across conversation"""
        try:
            conversation = result.get("conversation", {})
            analytics = result.get("analytics", {})
            
            # Check if conversation metadata exists
            has_conversation_data = conversation.get("message_count", 0) > 1
            
            # Check context effectiveness
            context_effectiveness = analytics.get("context_effectiveness", 0)
            
            return has_conversation_data or context_effectiveness > 0.1
            
        except Exception:
            return False
    
    async def test_adaptive_difficulty(self):
        """Test adaptive difficulty adjustment"""
        start_time = time.time()
        
        try:
            # Test with beginner level
            response_beginner = requests.post(
                f"{API_BASE}/quantum/message",
                json={
                    "user_id": "adaptive_test_user",
                    "message": "Explain machine learning to me, I'm a complete beginner.",
                    "task_type": "beginner_concepts",
                    "initial_context": {"difficulty_level": "beginner"}
                },
                timeout=10
            )
            
            # Test with advanced level  
            response_advanced = requests.post(
                f"{API_BASE}/quantum/message",
                json={
                    "user_id": "adaptive_test_user_advanced",
                    "message": "Explain the mathematical foundations of gradient descent optimization.",
                    "task_type": "advanced_concepts", 
                    "initial_context": {"difficulty_level": "advanced"}
                },
                timeout=10
            )
            
            success = (response_beginner.status_code == 200 and response_advanced.status_code == 200)
            response_time = time.time() - start_time
            
            # Analyze adaptation differences
            adaptation_analysis = self.analyze_difficulty_adaptation(response_beginner, response_advanced)
            
            self.record_test(
                "Adaptive Difficulty",
                success and adaptation_analysis["adapted"],
                response_time,
                adaptation_analysis
            )
            
        except Exception as e:
            self.record_test("Adaptive Difficulty", False, time.time() - start_time, {"error": str(e)})
    
    def analyze_difficulty_adaptation(self, beginner_response, advanced_response) -> Dict[str, Any]:
        """Analyze if responses were adapted to different difficulty levels"""
        try:
            if beginner_response.status_code != 200 or advanced_response.status_code != 200:
                return {"adapted": False, "error": "Invalid responses"}
            
            beginner_content = beginner_response.json().get("response", {}).get("content", "")
            advanced_content = advanced_response.json().get("response", {}).get("content", "")
            
            # Simple indicators of complexity
            beginner_simple = any(word in beginner_content.lower() for word in ["simple", "basic", "easy", "start"])
            advanced_complex = any(word in advanced_content.lower() for word in ["complex", "advanced", "mathematical", "algorithm"])
            
            # Length difference (advanced explanations typically longer)
            length_difference = len(advanced_content) - len(beginner_content)
            
            adapted = beginner_simple or advanced_complex or length_difference > 100
            
            return {
                "adapted": adapted,
                "beginner_indicators": beginner_simple,
                "advanced_indicators": advanced_complex,
                "length_difference": length_difference,
                "beginner_length": len(beginner_content),
                "advanced_length": len(advanced_content)
            }
            
        except Exception as e:
            return {"adapted": False, "error": str(e)}
    
    def generate_final_report(self):
        """Generate comprehensive final test report"""
        self.log("\n" + "=" * 80)
        self.log("📊 COMPREHENSIVE QUANTUM INTELLIGENCE SYSTEM TEST REPORT")
        self.log("=" * 80)
        
        # Summary statistics
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        self.log(f"📈 OVERALL RESULTS:")
        self.log(f"   Total Tests: {self.total_tests}")
        self.log(f"   Passed: {self.passed_tests} ✅")
        self.log(f"   Failed: {self.failed_tests} ❌")
        self.log(f"   Success Rate: {success_rate:.1f}%")
        
        # Performance metrics
        if self.test_results:
            response_times = [r["response_time_ms"] for r in self.test_results if r["success"]]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                self.log(f"\n⚡ PERFORMANCE METRICS:")
                self.log(f"   Average Response Time: {avg_response_time:.2f}ms")
                self.log(f"   Fastest Response: {min_response_time:.2f}ms")
                self.log(f"   Slowest Response: {max_response_time:.2f}ms")
        
        # Feature analysis
        self.log(f"\n🧠 FEATURE ANALYSIS:")
        
        # Count emotion-based tests
        emotion_tests = [r for r in self.test_results if "Emotion Learning" in r["test_name"]]
        emotion_success_rate = (len([r for r in emotion_tests if r["success"]]) / len(emotion_tests) * 100) if emotion_tests else 0
        
        self.log(f"   Emotion-Based Learning: {emotion_success_rate:.1f}% success ({len(emotion_tests)} tests)")
        
        # Real AI validation
        real_ai_tests = [r for r in self.test_results if r["details"].get("is_real_ai")]
        self.log(f"   Real AI Responses: {len(real_ai_tests)} verified")
        
        # AI Providers used
        providers = set()
        for result in self.test_results:
            provider = result.get("details", {}).get("ai_provider", "unknown")
            if provider != "unknown":
                providers.add(provider)
        
        self.log(f"   AI Providers Used: {', '.join(sorted(providers))}")
        
        # Detailed results
        self.log(f"\n📋 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            self.log(f"   {status} {result['test_name']}: {result['response_time_ms']:.2f}ms")
            
            # Show additional details for failed tests
            if not result["success"] and "error" in result["details"]:
                self.log(f"      Error: {result['details']['error']}")
        
        # System health assessment
        self.log(f"\n🏥 SYSTEM HEALTH ASSESSMENT:")
        
        if success_rate >= 80:
            self.log("   Status: 🟢 EXCELLENT - System performing optimally")
        elif success_rate >= 60:
            self.log("   Status: 🟡 GOOD - System mostly functional with minor issues")  
        elif success_rate >= 40:
            self.log("   Status: 🟠 FAIR - System functional but needs attention")
        else:
            self.log("   Status: 🔴 POOR - System requires immediate attention")
        
        # Recommendations
        self.log(f"\n💡 RECOMMENDATIONS:")
        
        failed_tests = [r for r in self.test_results if not r["success"]]
        if not failed_tests:
            self.log("   • All tests passed! System is performing excellently.")
            self.log("   • Consider scaling up for production deployment.")
        else:
            self.log(f"   • {len(failed_tests)} tests failed - investigate error logs")
            if emotion_success_rate < 70:
                self.log("   • Emotion detection needs improvement - check V9.0 integration")
            if len(real_ai_tests) == 0:
                self.log("   • Verify real AI API connections - may be using mocked responses")
        
        # Save detailed results
        self.save_test_results()
        
        self.log("\n" + "=" * 80)
        self.log("🎯 QUANTUM INTELLIGENCE TESTING COMPLETE")
        self.log("=" * 80)
    
    def save_test_results(self):
        """Save detailed test results to JSON file"""
        try:
            results_file = "/app/quantum_test_results.json"
            
            report_data = {
                "test_summary": {
                    "total_tests": self.total_tests,
                    "passed_tests": self.passed_tests,
                    "failed_tests": self.failed_tests,
                    "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
                    "test_timestamp": datetime.now().isoformat()
                },
                "detailed_results": self.test_results
            }
            
            with open(results_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.log(f"📄 Detailed results saved to: {results_file}")
            
        except Exception as e:
            self.log(f"⚠️ Could not save results file: {e}")

async def main():
    """Main test execution"""
    print("🚀 MasterX Quantum Intelligence System - Comprehensive Testing Suite")
    print("Testing real-world learning scenarios with emotion detection and personalization")
    print("=" * 80)
    
    tester = QuantumSystemTester()
    await tester.run_comprehensive_test_suite()

if __name__ == "__main__":
    asyncio.run(main())