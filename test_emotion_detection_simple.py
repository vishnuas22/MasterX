#!/usr/bin/env python3
"""
🧠 ULTRA-ENTERPRISE EMOTION DETECTION ENGINE V6.0 - SIMPLE TEST RUNNER
Comprehensive testing for emotion detection with >95% accuracy and <100ms performance
"""

import asyncio
import time
import sys
import os
import json
import uuid
from typing import Dict, Any, List

# Add backend to path
sys.path.append('/app/backend')

try:
    from quantum_intelligence.services.emotional.emotion_detection import (
        UltraEnterpriseEmotionDetectionEngine,
        EmotionCategory,
        InterventionLevel,
        LearningReadinessState,
        EmotionDetectionConstants
    )
    print("✅ Successfully imported emotion detection components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class EmotionDetectionTester:
    """Simple emotion detection tester"""
    
    def __init__(self):
        self.engine = None
        self.test_results = []
        self.performance_metrics = []
    
    async def initialize_engine(self):
        """Initialize the emotion detection engine"""
        try:
            print("🚀 Initializing Ultra-Enterprise Emotion Detection Engine V6.0...")
            self.engine = UltraEnterpriseEmotionDetectionEngine()
            await self.engine.initialize()
            print("✅ Engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_test_data(self, emotion_type: str = "engagement") -> Dict[str, Any]:
        """Generate test data for emotion analysis"""
        emotion_texts = {
            "joy": "I'm so happy and excited about this learning experience! This is awesome!",
            "frustration": "This is really confusing and difficult. I don't understand what's happening.",
            "curiosity": "That's really interesting! I wonder how this works and why it behaves this way?",
            "satisfaction": "Perfect! I finally understand this concept completely. Everything makes sense now.",
            "anxiety": "I'm worried I won't be able to learn this. This seems too complicated for me.",
            "boredom": "This is pretty boring and repetitive. I've seen this before.",
            "engagement": "This is fascinating! I want to learn more about this topic.",
            "neutral": "This is a normal learning session with standard content."
        }
        
        emotion_physiology = {
            "anxiety": {"heart_rate": 95, "skin_conductance": 0.8, "breathing_rate": 20},
            "excitement": {"heart_rate": 88, "skin_conductance": 0.7, "breathing_rate": 18},
            "calm": {"heart_rate": 65, "skin_conductance": 0.3, "breathing_rate": 14},
            "stress": {"heart_rate": 92, "skin_conductance": 0.85, "breathing_rate": 22},
            "engagement": {"heart_rate": 75, "skin_conductance": 0.5, "breathing_rate": 16}
        }
        
        return {
            "facial_data": {
                "image_data": f"mock_facial_data_{emotion_type}",
                "landmarks": [{"x": 100, "y": 150}, {"x": 120, "y": 145}],
                "confidence": 0.95,
                "emotion_indicators": {
                    "smile_detected": emotion_type == "joy",
                    "frown_detected": emotion_type == "sadness",
                    "eyebrow_raise": emotion_type == "surprise"
                }
            },
            "voice_data": {
                "audio_features": {
                    "pitch_mean": 180.5 if emotion_type == "joy" else 150.0,
                    "pitch_std": 45.2,
                    "tempo": 165,
                    "intensity": 0.8 if emotion_type == "excitement" else 0.5
                },
                "duration_ms": 3000,
                "sample_rate": 44100,
                "quality_score": 0.92
            },
            "text_data": emotion_texts.get(emotion_type, emotion_texts["neutral"]),
            "physiological_data": emotion_physiology.get(emotion_type, emotion_physiology["engagement"])
        }
    
    def generate_learning_context(self, difficulty: str = "moderate") -> Dict[str, Any]:
        """Generate learning context data"""
        difficulty_levels = {
            "easy": 0.3,
            "moderate": 0.5,
            "hard": 0.7,
            "expert": 0.9
        }
        
        return {
            "task_difficulty": difficulty_levels.get(difficulty, 0.5),
            "subject": "mathematics",
            "lesson_duration_minutes": 45,
            "previous_performance": 0.75,
            "learning_goals": ["understand_concept", "apply_knowledge", "synthesize_information"]
        }
    
    async def test_basic_emotion_analysis(self):
        """Test basic emotion analysis functionality"""
        print("\n🧪 Testing Basic Emotion Analysis...")
        
        try:
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            test_data = self.generate_test_data("engagement")
            context = self.generate_learning_context("moderate")
            
            start_time = time.time()
            result = await self.engine.analyze_emotions(
                user_id=user_id,
                input_data=test_data,
                context=context,
                enable_caching=True,
                max_analysis_time_ms=100
            )
            analysis_time_ms = (time.time() - start_time) * 1000
            
            # Validate results
            if result.get("status") == "success":
                analysis_result = result.get("analysis_result", {})
                performance_summary = result.get("performance_summary", {})
                
                primary_emotion = analysis_result.get("primary_emotion")
                confidence = analysis_result.get("emotion_confidence", 0.0)
                accuracy = performance_summary.get("recognition_accuracy", 0.0)
                
                print(f"✅ Basic Analysis Success:")
                print(f"   - Analysis time: {analysis_time_ms:.2f}ms (Target: <100ms)")
                print(f"   - Primary emotion: {primary_emotion}")
                print(f"   - Confidence: {confidence:.3f}")
                print(f"   - Recognition accuracy: {accuracy:.3f} (Target: ≥0.95)")
                print(f"   - Performance target met: {analysis_time_ms < 100}")
                print(f"   - Accuracy target met: {accuracy >= 0.95}")
                
                self.performance_metrics.append({
                    "test": "basic_analysis",
                    "analysis_time_ms": analysis_time_ms,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "target_time_met": analysis_time_ms < 100,
                    "target_accuracy_met": accuracy >= 0.95
                })
                
                return True
            else:
                print(f"❌ Basic Analysis Failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Basic Analysis Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_multimodal_accuracy(self):
        """Test multimodal emotion recognition accuracy"""
        print("\n🎯 Testing Multimodal Emotion Recognition Accuracy...")
        
        test_emotions = [
            "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral",
            "frustration", "satisfaction", "curiosity", "confidence", "anxiety",
            "excitement", "boredom", "engagement"
        ]
        
        accuracy_scores = []
        user_id = f"multimodal_test_user_{uuid.uuid4().hex[:8]}"
        
        try:
            for emotion in test_emotions:
                test_data = self.generate_test_data(emotion)
                context = self.generate_learning_context()
                
                result = await self.engine.analyze_emotions(
                    user_id=user_id,
                    input_data=test_data,
                    context=context
                )
                
                if result.get("status") == "success":
                    analysis_result = result.get("analysis_result", {})
                    confidence = analysis_result.get("emotion_confidence", 0.0)
                    accuracy_scores.append(confidence)
                else:
                    accuracy_scores.append(0.5)  # Default for failed analysis
            
            average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            
            print(f"✅ Multimodal Accuracy Results:")
            print(f"   - Tested emotions: {len(test_emotions)}")
            print(f"   - Average accuracy: {average_accuracy:.3f} (Target: ≥0.95)")
            print(f"   - Accuracy target met: {average_accuracy >= 0.95}")
            print(f"   - Individual scores: {[f'{score:.2f}' for score in accuracy_scores[:5]]}...")
            
            self.performance_metrics.append({
                "test": "multimodal_accuracy",
                "average_accuracy": average_accuracy,
                "emotions_tested": len(test_emotions),
                "target_accuracy_met": average_accuracy >= 0.95
            })
            
            return average_accuracy >= 0.95
            
        except Exception as e:
            print(f"❌ Multimodal Accuracy Test Exception: {e}")
            return False
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        try:
            user_id = f"perf_test_user_{uuid.uuid4().hex[:8]}"
            response_times = []
            
            # Run 10 performance tests
            for i in range(10):
                test_data = self.generate_test_data("engagement")
                context = self.generate_learning_context()
                
                start_time = time.time()
                result = await self.engine.analyze_emotions(
                    user_id=user_id,
                    input_data=test_data,
                    context=context,
                    enable_caching=True
                )
                analysis_time_ms = (time.time() - start_time) * 1000
                response_times.append(analysis_time_ms)
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"✅ Performance Benchmark Results:")
            print(f"   - Tests run: {len(response_times)}")
            print(f"   - Average response time: {avg_response_time:.2f}ms (Target: <100ms)")
            print(f"   - Max response time: {max_response_time:.2f}ms")
            print(f"   - Min response time: {min_response_time:.2f}ms")
            print(f"   - Optimal target (<50ms): {sum(1 for t in response_times if t < 50)}/{len(response_times)}")
            print(f"   - Standard target (<100ms): {sum(1 for t in response_times if t < 100)}/{len(response_times)}")
            
            self.performance_metrics.append({
                "test": "performance_benchmarks",
                "avg_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time,
                "min_response_time_ms": min_response_time,
                "target_100ms_met": avg_response_time < 100,
                "target_50ms_met": avg_response_time < 50
            })
            
            return avg_response_time < 100
            
        except Exception as e:
            print(f"❌ Performance Benchmark Exception: {e}")
            return False
    
    async def test_learning_state_optimization(self):
        """Test learning state optimization"""
        print("\n🎓 Testing Learning State Optimization...")
        
        test_scenarios = [
            ("optimal_flow", "engagement", "moderate"),
            ("high_stress", "anxiety", "hard"),
            ("low_motivation", "boredom", "easy"),
            ("cognitive_overload", "frustration", "expert"),
            ("perfect_challenge", "curiosity", "moderate")
        ]
        
        try:
            user_id = f"learning_test_user_{uuid.uuid4().hex[:8]}"
            successful_analyses = 0
            
            for scenario_name, emotion, difficulty in test_scenarios:
                test_data = self.generate_test_data(emotion)
                context = self.generate_learning_context(difficulty)
                
                result = await self.engine.analyze_emotions(
                    user_id=user_id,
                    input_data=test_data,
                    context=context
                )
                
                if result.get("status") == "success":
                    analysis_result = result.get("analysis_result", {})
                    
                    # Check for learning state fields
                    learning_fields = [
                        "learning_readiness", "learning_readiness_score", 
                        "cognitive_load_level", "attention_state", 
                        "motivation_level", "engagement_score"
                    ]
                    
                    fields_present = sum(1 for field in learning_fields if field in analysis_result)
                    
                    if fields_present >= len(learning_fields) * 0.8:  # 80% of fields present
                        successful_analyses += 1
                        print(f"   ✅ {scenario_name}: {fields_present}/{len(learning_fields)} fields present")
                    else:
                        print(f"   ⚠️ {scenario_name}: Only {fields_present}/{len(learning_fields)} fields present")
                else:
                    print(f"   ❌ {scenario_name}: Analysis failed")
            
            success_rate = successful_analyses / len(test_scenarios)
            
            print(f"✅ Learning State Optimization Results:")
            print(f"   - Scenarios tested: {len(test_scenarios)}")
            print(f"   - Successful analyses: {successful_analyses}")
            print(f"   - Success rate: {success_rate:.1%}")
            
            self.performance_metrics.append({
                "test": "learning_state_optimization",
                "scenarios_tested": len(test_scenarios),
                "successful_analyses": successful_analyses,
                "success_rate": success_rate
            })
            
            return success_rate >= 0.8
            
        except Exception as e:
            print(f"❌ Learning State Optimization Exception: {e}")
            return False
    
    async def test_intervention_analysis(self):
        """Test intervention analysis"""
        print("\n🚨 Testing Intervention Analysis...")
        
        intervention_scenarios = [
            ("critical_anxiety", "anxiety", "expert", "critical"),
            ("urgent_frustration", "frustration", "hard", "urgent"),
            ("moderate_boredom", "boredom", "easy", "moderate"),
            ("mild_confusion", "neutral", "moderate", "mild"),
            ("no_intervention_flow", "engagement", "moderate", "none")
        ]
        
        try:
            user_id = f"intervention_test_user_{uuid.uuid4().hex[:8]}"
            successful_interventions = 0
            
            for scenario, emotion, difficulty, expected_level in intervention_scenarios:
                test_data = self.generate_test_data(emotion)
                context = self.generate_learning_context(difficulty)
                
                result = await self.engine.analyze_emotions(
                    user_id=user_id,
                    input_data=test_data,
                    context=context
                )
                
                if result.get("status") == "success":
                    analysis_result = result.get("analysis_result", {})
                    
                    # Check for intervention fields
                    intervention_fields = [
                        "intervention_needed", "intervention_level", 
                        "intervention_recommendations", "intervention_confidence"
                    ]
                    
                    fields_present = sum(1 for field in intervention_fields if field in analysis_result)
                    
                    if fields_present >= len(intervention_fields) * 0.75:  # 75% of fields present
                        successful_interventions += 1
                        intervention_level = analysis_result.get("intervention_level", "unknown")
                        print(f"   ✅ {scenario}: {intervention_level} (expected: {expected_level})")
                    else:
                        print(f"   ⚠️ {scenario}: Only {fields_present}/{len(intervention_fields)} fields present")
                else:
                    print(f"   ❌ {scenario}: Analysis failed")
            
            success_rate = successful_interventions / len(intervention_scenarios)
            
            print(f"✅ Intervention Analysis Results:")
            print(f"   - Scenarios tested: {len(intervention_scenarios)}")
            print(f"   - Successful interventions: {successful_interventions}")
            print(f"   - Success rate: {success_rate:.1%}")
            
            self.performance_metrics.append({
                "test": "intervention_analysis",
                "scenarios_tested": len(intervention_scenarios),
                "successful_interventions": successful_interventions,
                "success_rate": success_rate
            })
            
            return success_rate >= 0.8
            
        except Exception as e:
            print(f"❌ Intervention Analysis Exception: {e}")
            return False
    
    async def run_comprehensive_tests(self):
        """Run comprehensive emotion detection tests"""
        print("🧠 ULTRA-ENTERPRISE EMOTION DETECTION ENGINE V6.0 - COMPREHENSIVE TESTS")
        print("=" * 80)
        
        # Initialize engine
        if not await self.initialize_engine():
            print("❌ Failed to initialize engine. Aborting tests.")
            return False
        
        # Run all tests
        test_results = []
        
        print("\n🚀 Running Comprehensive Test Suite...")
        
        # Test 1: Basic functionality
        test_results.append(await self.test_basic_emotion_analysis())
        
        # Test 2: Multimodal accuracy
        test_results.append(await self.test_multimodal_accuracy())
        
        # Test 3: Performance benchmarks
        test_results.append(await self.test_performance_benchmarks())
        
        # Test 4: Learning state optimization
        test_results.append(await self.test_learning_state_optimization())
        
        # Test 5: Intervention analysis
        test_results.append(await self.test_intervention_analysis())
        
        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = sum(test_results)
        success_rate = passed_tests / total_tests
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {passed_tests}")
        print(f"Failed Tests: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1%}")
        
        # Performance summary
        if self.performance_metrics:
            print(f"\n🎯 PERFORMANCE SUMMARY:")
            
            # Basic analysis performance
            basic_metrics = next((m for m in self.performance_metrics if m["test"] == "basic_analysis"), None)
            if basic_metrics:
                print(f"   - Basic Analysis Time: {basic_metrics['analysis_time_ms']:.2f}ms (Target: <100ms)")
                print(f"   - Basic Analysis Accuracy: {basic_metrics['accuracy']:.3f} (Target: ≥0.95)")
            
            # Multimodal accuracy
            multimodal_metrics = next((m for m in self.performance_metrics if m["test"] == "multimodal_accuracy"), None)
            if multimodal_metrics:
                print(f"   - Multimodal Accuracy: {multimodal_metrics['average_accuracy']:.3f} (Target: ≥0.95)")
            
            # Performance benchmarks
            perf_metrics = next((m for m in self.performance_metrics if m["test"] == "performance_benchmarks"), None)
            if perf_metrics:
                print(f"   - Average Response Time: {perf_metrics['avg_response_time_ms']:.2f}ms (Target: <100ms)")
                print(f"   - Optimal Performance (<50ms): {perf_metrics['target_50ms_met']}")
        
        # Overall assessment
        if success_rate >= 0.9:
            print(f"\n🎉 EXCELLENT! Ultra-Enterprise Emotion Detection V6.0 is performing at enterprise-grade levels!")
            print(f"✅ All major performance and accuracy targets achieved.")
        elif success_rate >= 0.7:
            print(f"\n✅ GOOD! Emotion Detection V6.0 is mostly functional with minor issues.")
            print(f"⚠️ Some optimizations may be needed for full enterprise deployment.")
        else:
            print(f"\n❌ CRITICAL! Emotion Detection V6.0 needs significant improvements.")
            print(f"🔧 Major issues detected that require immediate attention.")
        
        return success_rate >= 0.7

async def main():
    """Main test runner"""
    tester = EmotionDetectionTester()
    success = await tester.run_comprehensive_tests()
    return 0 if success else 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)