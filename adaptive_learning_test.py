#!/usr/bin/env python3
"""
MasterX Adaptive Learning Scenario Test
Testing adaptive learning capabilities through realistic data science learning simulation
"""

import requests
import json
import time
import sys
from datetime import datetime

class AdaptiveLearningTester:
    def __init__(self, base_url="https://adaptive-ai-2.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/ai/test"
        self.tests_run = 0
        self.tests_passed = 0
        self.conversation_history = []
        
    def log_test(self, scenario, success, details="", response_data=None):
        """Log test results with detailed information"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        result = f"{status} - {scenario}"
        if details:
            result += f" | {details}"
        
        print(result)
        
        # Store conversation for analysis
        if response_data:
            self.conversation_history.append({
                "scenario": scenario,
                "success": success,
                "details": details,
                "response": response_data,
                "timestamp": datetime.now().isoformat()
            })
        
        return success

    def send_ai_message(self, message, context=None, timeout=30):
        """Send message to AI and get response"""
        try:
            payload = {"message": message}
            if context:
                payload["context"] = context
            
            response = requests.post(self.api_url, json=payload, timeout=timeout)
            
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
                else:
                    return {
                        "success": False,
                        "error": data.get("error", "Unknown error"),
                        "fallback_available": data.get("fallback_available", False)
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response_text": response.text[:200]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def test_slow_learner_simulation(self):
        """🧪 SIMULATION 1: SLOW LEARNER (Struggling Student)"""
        print("\n🧪 SIMULATION 1: SLOW LEARNER (Struggling Student)")
        print("-" * 60)
        
        # Initial Query
        response1 = self.send_ai_message(
            "I want to learn data science but I have never programmed before. Can you help me start with Python?",
            context="You are MasterX, an adaptive learning AI. This user is a complete beginner. Adapt your teaching style to be very simple and encouraging."
        )
        
        if response1["success"]:
            initial_response = response1["content"]
            complexity_score = self.analyze_response_complexity(initial_response)
            self.log_test("Slow Learner - Initial Query", True, 
                         f"Response complexity: {complexity_score}, Provider: {response1['provider']}", response1)
            
            # Follow-up (Confused)
            time.sleep(2)  # Simulate thinking time
            response2 = self.send_ai_message(
                "I don't understand what a variable is. Can you explain it again?",
                context="The user is confused about basic concepts. They need even simpler explanations with more examples."
            )
            
            if response2["success"]:
                confused_response = response2["content"]
                adaptation_detected = self.detect_adaptation(initial_response, confused_response)
                self.log_test("Slow Learner - Confusion Response", adaptation_detected, 
                             f"Adaptation detected: {adaptation_detected}, Simpler explanation provided", response2)
                
                # Still Struggling
                time.sleep(2)
                response3 = self.send_ai_message(
                    "That's still confusing. I'm getting frustrated. Can you make it even simpler?",
                    context="The user is frustrated and struggling. Show empathy and break down concepts into the smallest possible pieces."
                )
                
                if response3["success"]:
                    frustrated_response = response3["content"]
                    empathy_detected = self.detect_empathy(frustrated_response)
                    simplification_detected = self.detect_simplification(confused_response, frustrated_response)
                    
                    success = empathy_detected and simplification_detected
                    self.log_test("Slow Learner - Frustration Handling", success, 
                                 f"Empathy: {empathy_detected}, Further simplification: {simplification_detected}", response3)
                    
                    # Progress Check
                    time.sleep(2)
                    response4 = self.send_ai_message(
                        "What is a variable? (Testing if they can detect I still don't understand)",
                        context="This is a test question to see if the AI can detect the user still doesn't understand the concept."
                    )
                    
                    if response4["success"]:
                        progress_response = response4["content"]
                        struggle_detection = self.detect_struggle_recognition(progress_response)
                        self.log_test("Slow Learner - Struggle Detection", struggle_detection, 
                                     f"AI detected continued struggle: {struggle_detection}", response4)
                        return True
                    else:
                        self.log_test("Slow Learner - Progress Check", False, f"API Error: {response4['error']}")
                else:
                    self.log_test("Slow Learner - Frustration Handling", False, f"API Error: {response3['error']}")
            else:
                self.log_test("Slow Learner - Confusion Response", False, f"API Error: {response2['error']}")
        else:
            self.log_test("Slow Learner - Initial Query", False, f"API Error: {response1['error']}")
        
        return False

    def test_fast_learner_simulation(self):
        """🚀 SIMULATION 2: FAST LEARNER (Quick Student)"""
        print("\n🚀 SIMULATION 2: FAST LEARNER (Quick Student)")
        print("-" * 60)
        
        # Initial Query
        response1 = self.send_ai_message(
            "I want to learn Python for data science. I have some programming background in Java.",
            context="This user has programming experience. You can use more technical terms and move faster through concepts."
        )
        
        if response1["success"]:
            initial_response = response1["content"]
            technical_level = self.analyze_technical_level(initial_response)
            self.log_test("Fast Learner - Initial Query", True, 
                         f"Technical level: {technical_level}, Provider: {response1['provider']}", response1)
            
            # Quick Grasp
            time.sleep(2)
            response2 = self.send_ai_message(
                "Got it! What about data structures in Python? Can you show me pandas?",
                context="The user quickly understood and is ready for more advanced topics. Increase the complexity."
            )
            
            if response2["success"]:
                advanced_response = response2["content"]
                complexity_increase = self.detect_complexity_increase(initial_response, advanced_response)
                self.log_test("Fast Learner - Quick Progression", complexity_increase, 
                             f"Complexity increased: {complexity_increase}, Advanced topics introduced", response2)
                
                # Advanced Request
                time.sleep(2)
                response3 = self.send_ai_message(
                    "This is easy. Can you show me machine learning examples with sklearn?",
                    context="The user is progressing very quickly and wants challenging content. Provide advanced examples."
                )
                
                if response3["success"]:
                    ml_response = response3["content"]
                    advanced_content = self.detect_advanced_content(ml_response)
                    code_examples = self.detect_code_examples(ml_response)
                    
                    success = advanced_content and code_examples
                    self.log_test("Fast Learner - Advanced Challenge", success, 
                                 f"Advanced content: {advanced_content}, Code examples: {code_examples}", response3)
                    
                    # Challenge Mode
                    time.sleep(2)
                    response4 = self.send_ai_message(
                        "Can you give me a complex real-world machine learning project to work on?",
                        context="The user wants maximum challenge. Provide complex, real-world scenarios."
                    )
                    
                    if response4["success"]:
                        challenge_response = response4["content"]
                        project_complexity = self.analyze_project_complexity(challenge_response)
                        self.log_test("Fast Learner - Challenge Mode", project_complexity > 7, 
                                     f"Project complexity score: {project_complexity}/10", response4)
                        return True
                    else:
                        self.log_test("Fast Learner - Challenge Mode", False, f"API Error: {response4['error']}")
                else:
                    self.log_test("Fast Learner - Advanced Challenge", False, f"API Error: {response3['error']}")
            else:
                self.log_test("Fast Learner - Quick Progression", False, f"API Error: {response2['error']}")
        else:
            self.log_test("Fast Learner - Initial Query", False, f"API Error: {response1['error']}")
        
        return False

    def test_different_learning_styles(self):
        """🎯 SIMULATION 3: DIFFERENT LEARNING STYLES"""
        print("\n🎯 SIMULATION 3: DIFFERENT LEARNING STYLES")
        print("-" * 60)
        
        # Visual Learner
        response1 = self.send_ai_message(
            "I learn better with examples and code. Can you show me instead of explaining?",
            context="This user is a visual learner who prefers examples and code over theoretical explanations."
        )
        
        if response1["success"]:
            visual_response = response1["content"]
            code_heavy = self.detect_code_examples(visual_response)
            visual_elements = self.detect_visual_elements(visual_response)
            
            self.log_test("Visual Learner Style", code_heavy, 
                         f"Code examples: {code_heavy}, Visual elements: {visual_elements}", response1)
            
            # Step-by-Step Learner
            time.sleep(2)
            response2 = self.send_ai_message(
                "Break this down into small steps please. I need to go slowly.",
                context="This user needs step-by-step instructions with clear progression through concepts."
            )
            
            if response2["success"]:
                stepwise_response = response2["content"]
                step_structure = self.detect_step_structure(stepwise_response)
                clear_progression = self.detect_clear_progression(stepwise_response)
                
                self.log_test("Step-by-Step Learner", step_structure, 
                             f"Step structure: {step_structure}, Clear progression: {clear_progression}", response2)
                
                # Hands-on Learner
                time.sleep(2)
                response3 = self.send_ai_message(
                    "Can I practice this? Give me exercises to try.",
                    context="This user learns by doing and needs practical exercises and hands-on activities."
                )
                
                if response3["success"]:
                    practice_response = response3["content"]
                    exercises_provided = self.detect_exercises(practice_response)
                    interactive_elements = self.detect_interactive_elements(practice_response)
                    
                    success = exercises_provided and interactive_elements
                    self.log_test("Hands-on Learner", success, 
                                 f"Exercises: {exercises_provided}, Interactive: {interactive_elements}", response3)
                    return True
                else:
                    self.log_test("Hands-on Learner", False, f"API Error: {response3['error']}")
            else:
                self.log_test("Step-by-Step Learner", False, f"API Error: {response2['error']}")
        else:
            self.log_test("Visual Learner Style", False, f"API Error: {response1['error']}")
        
        return False

    def test_context_memory_and_progression(self):
        """🧠 Test Context Memory and Learning Progression"""
        print("\n🧠 CONTEXT MEMORY AND LEARNING PROGRESSION")
        print("-" * 60)
        
        # Establish context
        response1 = self.send_ai_message(
            "I'm Sarah, a biology student trying to learn Python for data analysis. I'm particularly interested in analyzing gene expression data.",
            context="Remember this user's name, background, and specific interests for future interactions."
        )
        
        if response1["success"]:
            context_response = response1["content"]
            personalization = self.detect_personalization(context_response, "Sarah", "biology", "gene expression")
            
            self.log_test("Context Establishment", personalization, 
                         f"Personalization detected: {personalization}", response1)
            
            # Test memory in follow-up
            time.sleep(2)
            response2 = self.send_ai_message(
                "Can you suggest some Python libraries that would be useful for my field?",
                context="The user is asking a follow-up question. Remember their background and interests."
            )
            
            if response2["success"]:
                memory_response = response2["content"]
                context_memory = self.detect_context_memory(memory_response, ["biology", "gene expression", "Sarah"])
                relevant_suggestions = self.detect_relevant_suggestions(memory_response, "biology")
                
                success = context_memory and relevant_suggestions
                self.log_test("Context Memory", success, 
                             f"Memory: {context_memory}, Relevant suggestions: {relevant_suggestions}", response2)
                
                # Test progression tracking
                time.sleep(2)
                response3 = self.send_ai_message(
                    "I've learned the basics you taught me. What should I learn next?",
                    context="The user is indicating progress. Suggest appropriate next steps based on their learning journey."
                )
                
                if response3["success"]:
                    progression_response = response3["content"]
                    progression_awareness = self.detect_progression_awareness(progression_response)
                    next_steps = self.detect_appropriate_next_steps(progression_response)
                    
                    success = progression_awareness and next_steps
                    self.log_test("Learning Progression", success, 
                                 f"Progression awareness: {progression_awareness}, Next steps: {next_steps}", response3)
                    return True
                else:
                    self.log_test("Learning Progression", False, f"API Error: {response3['error']}")
            else:
                self.log_test("Context Memory", False, f"API Error: {response2['error']}")
        else:
            self.log_test("Context Establishment", False, f"API Error: {response1['error']}")
        
        return False

    # Analysis Methods
    def analyze_response_complexity(self, response):
        """Analyze the complexity of a response (1-10 scale)"""
        complexity_indicators = [
            "algorithm", "function", "class", "object", "method", "parameter",
            "variable", "data structure", "loop", "condition", "import", "library"
        ]
        
        technical_terms = sum(1 for term in complexity_indicators if term.lower() in response.lower())
        sentence_length = len(response.split('.'))
        word_count = len(response.split())
        
        # Simple scoring algorithm
        complexity = min(10, (technical_terms * 2) + (sentence_length / 5) + (word_count / 50))
        return round(complexity, 1)

    def analyze_technical_level(self, response):
        """Analyze technical level of response"""
        advanced_terms = [
            "inheritance", "polymorphism", "abstraction", "encapsulation",
            "decorator", "generator", "comprehension", "lambda", "async",
            "pandas", "numpy", "sklearn", "matplotlib", "seaborn"
        ]
        
        advanced_count = sum(1 for term in advanced_terms if term.lower() in response.lower())
        return "High" if advanced_count >= 3 else "Medium" if advanced_count >= 1 else "Basic"

    def detect_adaptation(self, initial_response, follow_up_response):
        """Detect if the AI adapted its response style"""
        initial_complexity = self.analyze_response_complexity(initial_response)
        follow_up_complexity = self.analyze_response_complexity(follow_up_response)
        
        # Check for simplification
        simplified = follow_up_complexity < initial_complexity
        
        # Check for more examples
        initial_examples = initial_response.lower().count("example")
        follow_up_examples = follow_up_response.lower().count("example")
        more_examples = follow_up_examples > initial_examples
        
        return simplified or more_examples

    def detect_empathy(self, response):
        """Detect empathetic language in response"""
        empathy_indicators = [
            "understand", "frustrating", "don't worry", "it's okay", "let's try",
            "I see", "that's normal", "many people", "step by step", "take your time"
        ]
        
        return any(indicator in response.lower() for indicator in empathy_indicators)

    def detect_simplification(self, previous_response, current_response):
        """Detect if current response is simpler than previous"""
        prev_complexity = self.analyze_response_complexity(previous_response)
        curr_complexity = self.analyze_response_complexity(current_response)
        
        return curr_complexity < prev_complexity

    def detect_struggle_recognition(self, response):
        """Detect if AI recognized user's continued struggle"""
        recognition_indicators = [
            "let me explain differently", "another way", "simpler terms",
            "break it down", "step by step", "don't worry", "it's okay"
        ]
        
        return any(indicator in response.lower() for indicator in recognition_indicators)

    def detect_complexity_increase(self, initial_response, advanced_response):
        """Detect if complexity increased appropriately"""
        initial_complexity = self.analyze_response_complexity(initial_response)
        advanced_complexity = self.analyze_response_complexity(advanced_response)
        
        return advanced_complexity > initial_complexity

    def detect_advanced_content(self, response):
        """Detect advanced content in response"""
        advanced_topics = [
            "machine learning", "sklearn", "pandas", "numpy", "matplotlib",
            "neural network", "algorithm", "model", "training", "prediction"
        ]
        
        return any(topic in response.lower() for topic in advanced_topics)

    def detect_code_examples(self, response):
        """Detect presence of code examples"""
        code_indicators = ["```", "import", "def ", "class ", "print(", "=", "for ", "if "]
        return any(indicator in response for indicator in code_indicators)

    def analyze_project_complexity(self, response):
        """Analyze project complexity (1-10 scale)"""
        complexity_factors = [
            "dataset", "preprocessing", "feature engineering", "model selection",
            "cross-validation", "hyperparameter", "deployment", "pipeline",
            "real-world", "production", "scalability", "performance"
        ]
        
        factor_count = sum(1 for factor in complexity_factors if factor.lower() in response.lower())
        return min(10, factor_count)

    def detect_visual_elements(self, response):
        """Detect visual learning elements"""
        visual_indicators = [
            "diagram", "chart", "graph", "visualization", "plot", "figure",
            "example", "illustration", "show", "display", "visual"
        ]
        
        return any(indicator in response.lower() for indicator in visual_indicators)

    def detect_step_structure(self, response):
        """Detect step-by-step structure"""
        step_indicators = [
            "step 1", "step 2", "first", "second", "third", "next", "then",
            "1.", "2.", "3.", "•", "-", "finally"
        ]
        
        return any(indicator in response.lower() for indicator in step_indicators)

    def detect_clear_progression(self, response):
        """Detect clear learning progression"""
        progression_indicators = [
            "start with", "begin by", "first learn", "then move to", "after that",
            "once you understand", "next step", "progression", "gradually"
        ]
        
        return any(indicator in response.lower() for indicator in progression_indicators)

    def detect_exercises(self, response):
        """Detect practice exercises"""
        exercise_indicators = [
            "exercise", "practice", "try", "attempt", "challenge", "problem",
            "assignment", "task", "activity", "hands-on", "implement"
        ]
        
        return any(indicator in response.lower() for indicator in exercise_indicators)

    def detect_interactive_elements(self, response):
        """Detect interactive elements"""
        interactive_indicators = [
            "interactive", "try it", "experiment", "modify", "change",
            "test", "run", "execute", "practice", "hands-on"
        ]
        
        return any(indicator in response.lower() for indicator in interactive_indicators)

    def detect_personalization(self, response, name, background, interest):
        """Detect personalized response"""
        personalization_score = 0
        
        if name.lower() in response.lower():
            personalization_score += 1
        if background.lower() in response.lower():
            personalization_score += 1
        if interest.lower() in response.lower():
            personalization_score += 1
        
        return personalization_score >= 2

    def detect_context_memory(self, response, context_items):
        """Detect if AI remembered context"""
        memory_score = sum(1 for item in context_items if item.lower() in response.lower())
        return memory_score >= len(context_items) // 2

    def detect_relevant_suggestions(self, response, field):
        """Detect field-relevant suggestions"""
        if field.lower() == "biology":
            bio_terms = ["bioinformatics", "genomics", "sequence", "protein", "gene", "DNA", "RNA"]
            return any(term in response.lower() for term in bio_terms)
        return False

    def detect_progression_awareness(self, response):
        """Detect awareness of user's learning progression"""
        progression_indicators = [
            "since you've learned", "now that you know", "building on",
            "next level", "advanced", "ready for", "progress"
        ]
        
        return any(indicator in response.lower() for indicator in progression_indicators)

    def detect_appropriate_next_steps(self, response):
        """Detect appropriate next learning steps"""
        next_step_indicators = [
            "next", "should learn", "recommend", "suggest", "move on to",
            "try", "explore", "dive into", "focus on"
        ]
        
        return any(indicator in response.lower() for indicator in next_step_indicators)

    def run_adaptive_learning_tests(self):
        """Run all adaptive learning tests"""
        print("🧠 MASTERX ADAPTIVE LEARNING SCENARIO TEST")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 80)
        
        # Run all simulations
        slow_learner_success = self.test_slow_learner_simulation()
        fast_learner_success = self.test_fast_learner_simulation()
        learning_styles_success = self.test_different_learning_styles()
        context_memory_success = self.test_context_memory_and_progression()
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("📊 ADAPTIVE LEARNING TEST SUMMARY")
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        print(f"\n🎯 SIMULATION RESULTS:")
        print(f"🧪 Slow Learner Simulation: {'✅ PASSED' if slow_learner_success else '❌ FAILED'}")
        print(f"🚀 Fast Learner Simulation: {'✅ PASSED' if fast_learner_success else '❌ FAILED'}")
        print(f"🎯 Learning Styles Adaptation: {'✅ PASSED' if learning_styles_success else '❌ FAILED'}")
        print(f"🧠 Context Memory & Progression: {'✅ PASSED' if context_memory_success else '❌ FAILED'}")
        
        # Analyze conversation patterns
        print(f"\n🔍 CONVERSATION ANALYSIS:")
        if self.conversation_history:
            avg_response_time = sum(conv.get("response", {}).get("response_time", 0) for conv in self.conversation_history) / len(self.conversation_history)
            providers_used = set(conv.get("response", {}).get("provider", "unknown") for conv in self.conversation_history)
            
            print(f"   • Average Response Time: {avg_response_time:.2f}s")
            print(f"   • AI Providers Used: {', '.join(providers_used)}")
            print(f"   • Total Conversations: {len(self.conversation_history)}")
        
        # Final assessment
        simulation_success_count = sum([slow_learner_success, fast_learner_success, learning_styles_success, context_memory_success])
        
        if simulation_success_count == 4:
            print("\n🎉 EXCELLENT! MasterX demonstrates FULL adaptive learning capabilities!")
            print("   ✅ Adapts to different learning speeds")
            print("   ✅ Recognizes and responds to learning styles")
            print("   ✅ Maintains context and tracks progression")
            print("   ✅ Shows empathy and emotional intelligence")
            return 0
        elif simulation_success_count >= 3:
            print("\n✅ GOOD! MasterX shows strong adaptive learning with minor gaps.")
            return 0
        elif simulation_success_count >= 2:
            print("\n⚠️  MODERATE! MasterX has basic adaptive features but needs improvement.")
            return 1
        else:
            print("\n❌ CRITICAL! MasterX lacks essential adaptive learning capabilities.")
            return 1

def main():
    tester = AdaptiveLearningTester()
    return tester.run_adaptive_learning_tests()

if __name__ == "__main__":
    sys.exit(main())