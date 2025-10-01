"""
Comprehensive Test Suite for MasterX Intelligence System
Tests context management and adaptive learning under real-world scenarios

Scenarios:
1. Frustrated beginner learning Python
2. Curious intermediate learning calculus
3. Advanced student in flow state
4. Struggling student with cognitive overload
5. Long conversation with context compression
6. Semantic memory retrieval
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List
import uuid

# Test imports
from core.context_manager import ContextManager, EmbeddingEngine, TokenBudgetManager
from core.adaptive_learning import (
    AdaptiveLearningEngine, 
    PerformanceMetrics,
    DifficultyLevel,
    CognitiveLoadEstimate
)
from core.models import Message, MessageRole, EmotionState, LearningReadiness
from services.emotion.emotion_engine import EmotionEngine
from utils.database import get_database
from motor.motor_asyncio import AsyncIOMotorClient


class TestMetrics:
    """Track test metrics"""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.timings = []
        self.results = []
    
    def add_result(self, test_name: str, passed: bool, duration_ms: float, details: Dict = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        
        self.timings.append(duration_ms)
        self.results.append({
            'test': test_name,
            'passed': passed,
            'duration_ms': duration_ms,
            'details': details or {}
        })
    
    def summary(self) -> Dict:
        return {
            'total_tests': self.tests_run,
            'passed': self.tests_passed,
            'failed': self.tests_failed,
            'success_rate': f"{(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%",
            'avg_duration_ms': sum(self.timings) / len(self.timings) if self.timings else 0,
            'total_duration_ms': sum(self.timings)
        }


class IntelligenceSystemTester:
    """Comprehensive tester for intelligence systems"""
    
    def __init__(self):
        self.metrics = TestMetrics()
        self.db = None
        self.context_manager = None
        self.adaptive_engine = None
        self.emotion_engine = None
        self.test_session_id = str(uuid.uuid4())
        self.test_user_id = "test_user_001"
    
    async def setup(self):
        """Initialize all components"""
        print("\n" + "="*80)
        print("🚀 MASTERX INTELLIGENCE SYSTEM TEST SUITE")
        print("="*80)
        print("\n📋 Setting up test environment...")
        
        # Connect to database
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        self.db = client.masterx
        
        # Initialize components
        self.context_manager = ContextManager(
            db=self.db,
            max_context_tokens=8000,
            short_term_memory_size=20
        )
        
        self.adaptive_engine = AdaptiveLearningEngine(db=self.db)
        self.emotion_engine = EmotionEngine()
        
        print("✅ Test environment ready!\n")
    
    async def cleanup(self):
        """Clean up test data"""
        print("\n🧹 Cleaning up test data...")
        # Remove test messages and sessions
        await self.db.messages.delete_many({'session_id': self.test_session_id})
        await self.db.sessions.delete_many({'_id': self.test_session_id})
        await self.db.user_performance.delete_many({'user_id': self.test_user_id})
        print("✅ Cleanup complete")
    
    # ========================================================================
    # SCENARIO 1: Frustrated Beginner Learning Python
    # ========================================================================
    
    async def test_frustrated_beginner(self):
        """Test: Frustrated beginner struggling with Python loops"""
        print("\n" + "-"*80)
        print("📝 SCENARIO 1: Frustrated Beginner Learning Python")
        print("-"*80)
        
        scenario_start = time.time()
        
        # Simulate frustrated messages
        messages = [
            "I don't understand for loops at all! This is so confusing!",
            "Why do we even need loops? Can't we just repeat code?",
            "I've tried 10 times and my code still doesn't work",
            "Maybe programming isn't for me..."
        ]
        
        for i, msg_text in enumerate(messages):
            start = time.time()
            
            # 1. Analyze emotion
            emotion_result = await self.emotion_engine.analyze_emotion(
                user_id=self.test_user_id,
                text=msg_text,
                context={}
            )
            
            emotion_state = EmotionState(
                primary_emotion=emotion_result.metrics.primary_emotion,
                arousal=emotion_result.metrics.arousal,
                valence=emotion_result.metrics.valence,
                learning_readiness=LearningReadiness(emotion_result.metrics.learning_readiness)
            )
            
            print(f"\n  Message {i+1}: '{msg_text[:50]}...'")
            print(f"  └─ Emotion: {emotion_state.primary_emotion} (arousal={emotion_state.arousal:.2f}, valence={emotion_state.valence:.2f})")
            
            # 2. Add to context
            message = Message(
                id=str(uuid.uuid4()),
                session_id=self.test_session_id,
                user_id=self.test_user_id,
                role=MessageRole.USER,
                content=msg_text,
                timestamp=datetime.utcnow(),
                emotion_state=emotion_state
            )
            
            await self.context_manager.add_message(
                session_id=self.test_session_id,
                message=message,
                generate_embedding=True
            )
            
            # 3. Get recommended difficulty
            performance = PerformanceMetrics(
                accuracy=0.3,  # Low accuracy (struggling)
                response_time_ms=15000,  # Slow (15 seconds)
                help_requests=i + 1,
                retries=i * 2,
                success_streak=0,
                failure_streak=i + 1
            )
            
            difficulty = await self.adaptive_engine.recommend_difficulty(
                user_id=self.test_user_id,
                subject="python_basics",
                emotion_state=emotion_state,
                recent_performance=performance
            )
            
            print(f"  └─ Recommended difficulty: {difficulty.label} ({difficulty.value:.2f})")
            
            # 4. Estimate cognitive load
            cognitive_load = self.adaptive_engine.cognitive_estimator.estimate_load(
                task_complexity=0.5,
                time_on_task_seconds=15.0,
                emotion_state=emotion_state,
                help_requests=i + 1,
                retries=i * 2
            )
            
            print(f"  └─ Cognitive load: {cognitive_load.level} ({cognitive_load.load:.2f})")
            
            elapsed = (time.time() - start) * 1000
            print(f"  └─ Processing time: {elapsed:.0f}ms")
        
        # Test context retrieval
        context = await self.context_manager.get_context(
            session_id=self.test_session_id,
            include_semantic=False
        )
        
        print(f"\n  📊 Context Summary:")
        print(f"  └─ Messages in context: {len(context['recent_messages'])}")
        print(f"  └─ Total tokens: {context['total_tokens']}")
        print(f"  └─ Compressed: {context['compressed']}")
        
        scenario_duration = (time.time() - scenario_start) * 1000
        
        # Validate results
        passed = (
            len(context['recent_messages']) == 4 and
            difficulty.value < 0.4 and  # Should be low for frustrated beginner
            cognitive_load.level in ['high', 'overload']
        )
        
        self.metrics.add_result(
            "Frustrated Beginner Scenario",
            passed,
            scenario_duration,
            {
                'messages_processed': 4,
                'final_difficulty': difficulty.value,
                'cognitive_load': cognitive_load.level,
                'context_tokens': context['total_tokens']
            }
        )
        
        print(f"\n  ✅ Scenario complete in {scenario_duration:.0f}ms")
        return passed
    
    # ========================================================================
    # SCENARIO 2: Curious Intermediate Learning Calculus
    # ========================================================================
    
    async def test_curious_intermediate(self):
        """Test: Curious student exploring calculus concepts"""
        print("\n" + "-"*80)
        print("📝 SCENARIO 2: Curious Intermediate Learning Calculus")
        print("-"*80)
        
        scenario_start = time.time()
        session_id = str(uuid.uuid4())
        
        messages = [
            "Can you explain what a derivative really means?",
            "That's interesting! How does this relate to physics?",
            "I see the pattern now. What about second derivatives?",
            "This is making sense! Can we try a harder problem?"
        ]
        
        for i, msg_text in enumerate(messages):
            start = time.time()
            
            # Analyze emotion
            emotion_result = await self.emotion_engine.analyze_emotion(
                user_id=self.test_user_id,
                text=msg_text,
                context={}
            )
            
            emotion_state = EmotionState(
                primary_emotion=emotion_result.metrics.primary_emotion,
                arousal=emotion_result.metrics.arousal,
                valence=emotion_result.metrics.valence,
                learning_readiness=LearningReadiness(emotion_result.metrics.learning_readiness)
            )
            
            print(f"\n  Message {i+1}: '{msg_text}'")
            print(f"  └─ Emotion: {emotion_state.primary_emotion}")
            
            # Add to context
            message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=self.test_user_id,
                role=MessageRole.USER,
                content=msg_text,
                timestamp=datetime.utcnow(),
                emotion_state=emotion_state
            )
            
            await self.context_manager.add_message(
                session_id=session_id,
                message=message,
                generate_embedding=True
            )
            
            # Simulate improving performance
            performance = PerformanceMetrics(
                accuracy=0.6 + (i * 0.1),  # Improving accuracy
                response_time_ms=5000,
                help_requests=0,
                retries=0,
                success_streak=i + 1,
                failure_streak=0
            )
            
            # Update ability (simulating correct answers)
            new_ability = await self.adaptive_engine.ability_estimator.update_ability(
                user_id=self.test_user_id,
                subject="calculus",
                item_difficulty=0.5 + (i * 0.05),
                result=True
            )
            
            print(f"  └─ Ability updated: {new_ability:.3f}")
            
            # Get difficulty recommendation
            difficulty = await self.adaptive_engine.recommend_difficulty(
                user_id=self.test_user_id,
                subject="calculus",
                emotion_state=emotion_state,
                recent_performance=performance
            )
            
            print(f"  └─ Recommended difficulty: {difficulty.label} ({difficulty.value:.2f})")
            
            elapsed = (time.time() - start) * 1000
            print(f"  └─ Processing time: {elapsed:.0f}ms")
        
        scenario_duration = (time.time() - scenario_start) * 1000
        
        # Validate: ability should increase, difficulty should adapt upward
        passed = new_ability > 0.5
        
        self.metrics.add_result(
            "Curious Intermediate Scenario",
            passed,
            scenario_duration,
            {
                'messages_processed': 4,
                'final_ability': new_ability,
                'final_difficulty': difficulty.value
            }
        )
        
        print(f"\n  ✅ Scenario complete in {scenario_duration:.0f}ms")
        
        # Cleanup this session
        await self.db.messages.delete_many({'session_id': session_id})
        
        return passed
    
    # ========================================================================
    # SCENARIO 3: Flow State Detection
    # ========================================================================
    
    async def test_flow_state_detection(self):
        """Test: Detect when student enters flow state"""
        print("\n" + "-"*80)
        print("📝 SCENARIO 3: Flow State Detection")
        print("-"*80)
        
        scenario_start = time.time()
        
        # Simulate flow state indicators
        print("\n  Simulating optimal learning conditions...")
        
        # Set ability to moderate level
        await self.adaptive_engine.ability_estimator._save_ability(
            user_id=self.test_user_id,
            subject="algorithms",
            ability=0.65
        )
        
        # Create flow-state emotion
        flow_emotion = EmotionState(
            primary_emotion="flow_state",
            arousal=0.7,  # Engaged
            valence=0.8,  # Positive
            learning_readiness=LearningReadiness.HIGH_READINESS
        )
        
        # Create flow-state performance
        flow_performance = PerformanceMetrics(
            accuracy=0.75,  # Sweet spot (65-85%)
            response_time_ms=3000,
            help_requests=0,
            retries=0,
            success_streak=5,
            failure_streak=0
        )
        
        # Calculate optimal difficulty
        optimal_difficulty = self.adaptive_engine.flow_optimizer.calculate_optimal_difficulty(
            ability=0.65,
            current_emotion=flow_emotion
        )
        
        print(f"  └─ Ability: 0.65")
        print(f"  └─ Optimal difficulty: {optimal_difficulty:.2f}")
        
        # Detect flow state
        in_flow = self.adaptive_engine.flow_optimizer.detect_flow_state(
            emotion_state=flow_emotion,
            performance=flow_performance,
            ability=0.65,
            current_difficulty=optimal_difficulty
        )
        
        print(f"  └─ Flow state detected: {in_flow}")
        
        # Get recommendations
        recommendations = self.adaptive_engine.flow_optimizer.get_flow_recommendations(
            ability=0.65,
            current_difficulty=optimal_difficulty,
            emotion_state=flow_emotion
        )
        
        print(f"  └─ Recommendation: {recommendations['suggested_action']}")
        print(f"  └─ Reasoning: {recommendations['reasoning']}")
        
        scenario_duration = (time.time() - scenario_start) * 1000
        
        passed = in_flow and recommendations['suggested_action'] == 'maintain'
        
        self.metrics.add_result(
            "Flow State Detection",
            passed,
            scenario_duration,
            {
                'flow_detected': in_flow,
                'optimal_difficulty': optimal_difficulty,
                'recommendation': recommendations['suggested_action']
            }
        )
        
        print(f"\n  ✅ Scenario complete in {scenario_duration:.0f}ms")
        return passed
    
    # ========================================================================
    # SCENARIO 4: Semantic Memory Retrieval
    # ========================================================================
    
    async def test_semantic_retrieval(self):
        """Test: Semantic search for relevant past messages"""
        print("\n" + "-"*80)
        print("📝 SCENARIO 4: Semantic Memory Retrieval")
        print("-"*80)
        
        scenario_start = time.time()
        session_id = str(uuid.uuid4())
        
        # Add diverse messages
        past_messages = [
            "I learned about recursion in computer science yesterday",
            "The concept of derivatives in calculus is fascinating",
            "Can you help me understand binary search trees?",
            "What's the difference between a stack and a queue?",
            "I'm struggling with understanding pointers in C++",
        ]
        
        print("\n  Adding past messages to memory...")
        for msg_text in past_messages:
            message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=self.test_user_id,
                role=MessageRole.USER,
                content=msg_text,
                timestamp=datetime.utcnow() - timedelta(hours=2)
            )
            
            await self.context_manager.add_message(
                session_id=session_id,
                message=message,
                generate_embedding=True
            )
            print(f"  └─ Added: '{msg_text[:50]}...'")
        
        # Small delay to ensure embeddings are ready
        await asyncio.sleep(0.5)
        
        # Now search for relevant messages
        queries = [
            "explain recursion",
            "data structures",
            "calculus help"
        ]
        
        print("\n  Performing semantic searches...")
        for query in queries:
            start = time.time()
            
            relevant = await self.context_manager.memory_retriever.find_relevant(
                query=query,
                session_id=session_id,
                top_k=2,
                min_similarity=0.3
            )
            
            elapsed = (time.time() - start) * 1000
            
            print(f"\n  Query: '{query}'")
            print(f"  └─ Found {len(relevant)} relevant messages in {elapsed:.0f}ms")
            
            for msg, similarity in relevant:
                print(f"     • [{similarity:.2f}] {msg.content[:60]}...")
        
        scenario_duration = (time.time() - scenario_start) * 1000
        
        passed = len(relevant) > 0
        
        self.metrics.add_result(
            "Semantic Memory Retrieval",
            passed,
            scenario_duration,
            {
                'messages_added': len(past_messages),
                'queries_tested': len(queries),
                'retrieval_working': passed
            }
        )
        
        print(f"\n  ✅ Scenario complete in {scenario_duration:.0f}ms")
        
        # Cleanup
        await self.db.messages.delete_many({'session_id': session_id})
        
        return passed
    
    # ========================================================================
    # SCENARIO 5: Context Compression Under Load
    # ========================================================================
    
    async def test_context_compression(self):
        """Test: Context compression with long conversations"""
        print("\n" + "-"*80)
        print("📝 SCENARIO 5: Context Compression Under Load")
        print("-"*80)
        
        scenario_start = time.time()
        session_id = str(uuid.uuid4())
        
        # Generate 50 messages
        print("\n  Generating 50 messages...")
        message_count = 50
        
        for i in range(message_count):
            message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=self.test_user_id,
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"Message {i+1}: This is test content for compression testing.",
                timestamp=datetime.utcnow() - timedelta(minutes=message_count - i)
            )
            
            await self.context_manager.add_message(
                session_id=session_id,
                message=message,
                generate_embedding=False  # Skip embeddings for speed
            )
        
        print(f"  ✅ Added {message_count} messages")
        
        # Get context (should auto-fit to budget)
        context = await self.context_manager.get_context(
            session_id=session_id,
            include_semantic=False
        )
        
        print(f"\n  Before compression:")
        print(f"  └─ Messages retrieved: {len(context['recent_messages'])}")
        print(f"  └─ Tokens: {context['total_tokens']}")
        print(f"  └─ Compressed: {context['compressed']}")
        
        # Perform compression
        start = time.time()
        removed = await self.context_manager.compress_context(
            session_id=session_id,
            compression_ratio=0.4  # Keep 40%
        )
        compression_time = (time.time() - start) * 1000
        
        print(f"\n  After compression:")
        print(f"  └─ Messages removed: {removed}")
        print(f"  └─ Compression time: {compression_time:.0f}ms")
        
        # Verify compression
        final_count = await self.db.messages.count_documents({'session_id': session_id})
        print(f"  └─ Final message count: {final_count}")
        
        scenario_duration = (time.time() - scenario_start) * 1000
        
        passed = removed > 0 and final_count < message_count
        
        self.metrics.add_result(
            "Context Compression",
            passed,
            scenario_duration,
            {
                'initial_messages': message_count,
                'messages_removed': removed,
                'final_count': final_count,
                'compression_time_ms': compression_time
            }
        )
        
        print(f"\n  ✅ Scenario complete in {scenario_duration:.0f}ms")
        
        # Cleanup
        await self.db.messages.delete_many({'session_id': session_id})
        
        return passed
    
    # ========================================================================
    # SCENARIO 6: Cognitive Overload Detection
    # ========================================================================
    
    async def test_cognitive_overload(self):
        """Test: Detect and respond to cognitive overload"""
        print("\n" + "-"*80)
        print("📝 SCENARIO 6: Cognitive Overload Detection & Response")
        print("-"*80)
        
        scenario_start = time.time()
        
        print("\n  Simulating overwhelmed student...")
        
        # Overload indicators
        overload_emotion = EmotionState(
            primary_emotion="overwhelmed",
            arousal=0.9,  # Very high stress
            valence=0.2,  # Very negative
            learning_readiness=LearningReadiness.NEEDS_BREAK
        )
        
        # Estimate cognitive load
        cognitive_load = self.adaptive_engine.cognitive_estimator.estimate_load(
            task_complexity=0.8,  # Hard task
            time_on_task_seconds=600,  # 10 minutes
            emotion_state=overload_emotion,
            help_requests=7,
            retries=5
        )
        
        print(f"  └─ Cognitive load: {cognitive_load.level} ({cognitive_load.load:.2f})")
        print(f"  └─ Load factors:")
        for factor, value in cognitive_load.factors.items():
            print(f"     • {factor}: {value:.2f}")
        
        # Get difficulty recommendation (should reduce)
        difficulty = await self.adaptive_engine.recommend_difficulty(
            user_id=self.test_user_id,
            subject="advanced_algorithms",
            emotion_state=overload_emotion,
            recent_performance=PerformanceMetrics(
                accuracy=0.2,
                response_time_ms=30000,
                help_requests=7,
                retries=5,
                success_streak=0,
                failure_streak=5
            )
        )
        
        print(f"\n  System response:")
        print(f"  └─ Recommended difficulty: {difficulty.label} ({difficulty.value:.2f})")
        print(f"  └─ Explanation: {difficulty.explanation}")
        
        scenario_duration = (time.time() - scenario_start) * 1000
        
        # Should detect overload and reduce difficulty significantly
        passed = (
            cognitive_load.level == "overload" and
            difficulty.value < 0.4
        )
        
        self.metrics.add_result(
            "Cognitive Overload Detection",
            passed,
            scenario_duration,
            {
                'cognitive_load': cognitive_load.level,
                'load_value': cognitive_load.load,
                'difficulty_reduced': difficulty.value < 0.4
            }
        )
        
        print(f"\n  ✅ Scenario complete in {scenario_duration:.0f}ms")
        return passed
    
    # ========================================================================
    # Performance Benchmarks
    # ========================================================================
    
    async def test_performance_benchmarks(self):
        """Test: Measure performance of key operations"""
        print("\n" + "-"*80)
        print("📝 PERFORMANCE BENCHMARKS")
        print("-"*80)
        
        benchmarks = {}
        
        # 1. Embedding generation
        print("\n  Testing embedding generation...")
        start = time.time()
        for i in range(10):
            await self.context_manager.embedding_engine.embed_text(
                "Test message for embedding benchmark"
            )
        embedding_time = (time.time() - start) * 1000 / 10
        benchmarks['embedding_avg_ms'] = embedding_time
        print(f"  └─ Average: {embedding_time:.1f}ms per embedding")
        
        # 2. Token estimation
        print("\n  Testing token estimation...")
        test_text = "This is a test message for token estimation. " * 20
        start = time.time()
        for i in range(1000):
            tokens = self.context_manager.token_manager.estimate_tokens(test_text)
        token_time = (time.time() - start) * 1000 / 1000
        benchmarks['token_estimation_avg_ms'] = token_time
        print(f"  └─ Average: {token_time:.3f}ms per estimation")
        
        # 3. Ability update
        print("\n  Testing ability updates...")
        start = time.time()
        for i in range(20):
            await self.adaptive_engine.ability_estimator.update_ability(
                user_id=self.test_user_id,
                subject="benchmark_test",
                item_difficulty=0.5,
                result=True
            )
        ability_time = (time.time() - start) * 1000 / 20
        benchmarks['ability_update_avg_ms'] = ability_time
        print(f"  └─ Average: {ability_time:.1f}ms per update")
        
        # 4. Cognitive load estimation
        print("\n  Testing cognitive load estimation...")
        start = time.time()
        test_emotion = EmotionState(
            primary_emotion="neutral",
            arousal=0.5,
            valence=0.5,
            learning_readiness=LearningReadiness.MODERATE_READINESS
        )
        for i in range(1000):
            self.adaptive_engine.cognitive_estimator.estimate_load(
                task_complexity=0.5,
                time_on_task_seconds=10.0,
                emotion_state=test_emotion,
                help_requests=0,
                retries=0
            )
        cognitive_time = (time.time() - start) * 1000 / 1000
        benchmarks['cognitive_load_avg_ms'] = cognitive_time
        print(f"  └─ Average: {cognitive_time:.3f}ms per estimation")
        
        print(f"\n  📊 Performance Summary:")
        print(f"  └─ Embedding: {benchmarks['embedding_avg_ms']:.1f}ms (target: <50ms)")
        print(f"  └─ Token estimation: {benchmarks['token_estimation_avg_ms']:.3f}ms (target: <1ms)")
        print(f"  └─ Ability update: {benchmarks['ability_update_avg_ms']:.1f}ms (target: <100ms)")
        print(f"  └─ Cognitive load: {benchmarks['cognitive_load_avg_ms']:.3f}ms (target: <1ms)")
        
        # All should meet performance targets
        passed = (
            benchmarks['embedding_avg_ms'] < 50 and
            benchmarks['token_estimation_avg_ms'] < 1 and
            benchmarks['ability_update_avg_ms'] < 100 and
            benchmarks['cognitive_load_avg_ms'] < 1
        )
        
        self.metrics.add_result(
            "Performance Benchmarks",
            passed,
            sum(benchmarks.values()),
            benchmarks
        )
        
        print(f"\n  ✅ Benchmarks complete")
        return passed
    
    # ========================================================================
    # Main Test Runner
    # ========================================================================
    
    async def run_all_tests(self):
        """Run all test scenarios"""
        await self.setup()
        
        try:
            # Run all scenarios
            await self.test_frustrated_beginner()
            await self.test_curious_intermediate()
            await self.test_flow_state_detection()
            await self.test_semantic_retrieval()
            await self.test_context_compression()
            await self.test_cognitive_overload()
            await self.test_performance_benchmarks()
            
        finally:
            await self.cleanup()
        
        # Print final summary
        self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive test report"""
        print("\n" + "="*80)
        print("📊 FINAL TEST REPORT")
        print("="*80)
        
        summary = self.metrics.summary()
        
        print(f"\n🎯 Overall Results:")
        print(f"  • Total Tests: {summary['total_tests']}")
        print(f"  • Passed: {summary['passed']} ✅")
        print(f"  • Failed: {summary['failed']} ❌")
        print(f"  • Success Rate: {summary['success_rate']}")
        print(f"  • Average Duration: {summary['avg_duration_ms']:.0f}ms")
        print(f"  • Total Duration: {summary['total_duration_ms']:.0f}ms")
        
        print(f"\n📋 Detailed Results:")
        for result in self.metrics.results:
            status = "✅" if result['passed'] else "❌"
            print(f"\n  {status} {result['test']}")
            print(f"     Duration: {result['duration_ms']:.0f}ms")
            if result['details']:
                for key, value in result['details'].items():
                    print(f"     {key}: {value}")
        
        print("\n" + "="*80)
        if self.metrics.tests_failed == 0:
            print("🎉 ALL TESTS PASSED! Intelligence system fully operational!")
        else:
            print(f"⚠️  {self.metrics.tests_failed} test(s) failed. Review results above.")
        print("="*80 + "\n")


# Run tests
async def main():
    tester = IntelligenceSystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
