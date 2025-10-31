"""
MasterX Emotion Detection Test Suite
====================================

This comprehensive test suite validates the emotion detection system with diverse 
emotional scenarios to identify misclassification issues.

Test Categories:
1. Basic Emotions (Positive/Negative)
2. Learning-Specific Emotions (Confusion, Curiosity, Frustration)
3. Mixed Emotions (Complex emotional states)
4. Edge Cases (Neutral, Ambiguous, Very Short)
5. Contextual Emotions (Same words, different contexts)

Author: MasterX Testing Team
Date: 2025
"""

import asyncio
import sys
import time
from typing import List, Dict, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/app/backend')

from services.emotion.emotion_engine import EmotionEngine, EmotionEngineConfig
from services.emotion.emotion_core import EmotionCategory, LearningReadiness

# ============================================================================
# TEST SCENARIOS
# ============================================================================

# Define comprehensive test scenarios with EXPECTED emotions
TEST_SCENARIOS = [
    # ==== CATEGORY 1: POSITIVE EMOTIONS ====
    {
        "category": "Positive - Joy",
        "text": "I finally understand this! This is amazing!",
        "expected_primary": [EmotionCategory.JOY, EmotionCategory.EXCITEMENT],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL, LearningReadiness.GOOD]
    },
    {
        "category": "Positive - Excitement",
        "text": "Wow! This is so exciting! I can't wait to learn more!",
        "expected_primary": [EmotionCategory.EXCITEMENT, EmotionCategory.JOY],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL, LearningReadiness.GOOD]
    },
    {
        "category": "Positive - Gratitude",
        "text": "Thank you so much for explaining this clearly. I really appreciate it.",
        "expected_primary": [EmotionCategory.GRATITUDE, EmotionCategory.APPROVAL],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL, LearningReadiness.GOOD]
    },
    {
        "category": "Positive - Pride",
        "text": "I solved it on my own! I'm so proud of myself!",
        "expected_primary": [EmotionCategory.PRIDE, EmotionCategory.JOY],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL]
    },
    {
        "category": "Positive - Optimism",
        "text": "I think I can figure this out with a bit more practice.",
        "expected_primary": [EmotionCategory.OPTIMISM, EmotionCategory.APPROVAL],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.GOOD, LearningReadiness.MODERATE]
    },
    
    # ==== CATEGORY 2: NEGATIVE EMOTIONS ====
    {
        "category": "Negative - Frustration",
        "text": "This is so frustrating! I've tried three times and still can't get it right.",
        "expected_primary": [EmotionCategory.ANNOYANCE, EmotionCategory.DISAPPOINTMENT, EmotionCategory.ANGER],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.MODERATE]
    },
    {
        "category": "Negative - Anger",
        "text": "This is ridiculous! Why doesn't this work? This makes no sense!",
        "expected_primary": [EmotionCategory.ANGER, EmotionCategory.ANNOYANCE],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.BLOCKED]
    },
    {
        "category": "Negative - Disappointment",
        "text": "I thought I understood, but I was wrong. I feel disappointed in myself.",
        "expected_primary": [EmotionCategory.DISAPPOINTMENT, EmotionCategory.SADNESS],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.MODERATE]
    },
    {
        "category": "Negative - Fear/Nervousness",
        "text": "I'm really nervous about this test. What if I fail?",
        "expected_primary": [EmotionCategory.NERVOUSNESS, EmotionCategory.FEAR],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.MODERATE]
    },
    {
        "category": "Negative - Sadness",
        "text": "I feel sad because everyone else understands this except me.",
        "expected_primary": [EmotionCategory.SADNESS, EmotionCategory.DISAPPOINTMENT],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW]
    },
    
    # ==== CATEGORY 3: LEARNING-SPECIFIC EMOTIONS ====
    {
        "category": "Learning - Curiosity (High)",
        "text": "That's interesting! How does that work? Can you tell me more?",
        "expected_primary": [EmotionCategory.CURIOSITY, EmotionCategory.EXCITEMENT],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL, LearningReadiness.GOOD]
    },
    {
        "category": "Learning - Confusion (Moderate)",
        "text": "I don't understand this part. Could you explain it differently?",
        "expected_primary": [EmotionCategory.CONFUSION, EmotionCategory.CURIOSITY],
        "expected_valence": "neutral_or_slightly_negative",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.GOOD]
    },
    {
        "category": "Learning - Confusion (High)",
        "text": "I'm completely lost. Nothing makes sense. What does this even mean?",
        "expected_primary": [EmotionCategory.CONFUSION, EmotionCategory.ANNOYANCE],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.BLOCKED]
    },
    {
        "category": "Learning - Realization",
        "text": "Oh! Now I get it! That makes so much sense now!",
        "expected_primary": [EmotionCategory.REALIZATION, EmotionCategory.JOY],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL, LearningReadiness.GOOD]
    },
    {
        "category": "Learning - Admiration",
        "text": "This is a brilliant solution! I never would have thought of that approach.",
        "expected_primary": [EmotionCategory.ADMIRATION, EmotionCategory.APPROVAL],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.GOOD, LearningReadiness.OPTIMAL]
    },
    
    # ==== CATEGORY 4: MIXED EMOTIONS ====
    {
        "category": "Mixed - Confusion + Curiosity",
        "text": "This is confusing, but I'm curious to understand how it works.",
        "expected_primary": [EmotionCategory.CONFUSION, EmotionCategory.CURIOSITY],
        "expected_valence": "neutral",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.GOOD]
    },
    {
        "category": "Mixed - Frustration + Determination",
        "text": "This is frustrating, but I'm not giving up. I'll keep trying.",
        "expected_primary": [EmotionCategory.ANNOYANCE, EmotionCategory.OPTIMISM],
        "expected_valence": "mixed",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.GOOD]
    },
    {
        "category": "Mixed - Nervousness + Excitement",
        "text": "I'm nervous but also excited about learning this new concept.",
        "expected_primary": [EmotionCategory.NERVOUSNESS, EmotionCategory.EXCITEMENT],
        "expected_valence": "mixed",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.GOOD]
    },
    {
        "category": "Mixed - Disappointment + Hope",
        "text": "I'm disappointed with my result, but I hope I can improve next time.",
        "expected_primary": [EmotionCategory.DISAPPOINTMENT, EmotionCategory.OPTIMISM],
        "expected_valence": "mixed",
        "expected_readiness": [LearningReadiness.MODERATE]
    },
    
    # ==== CATEGORY 5: NEUTRAL & AMBIGUOUS ====
    {
        "category": "Neutral - Statement",
        "text": "The formula is F = ma. Newton's second law of motion.",
        "expected_primary": [EmotionCategory.NEUTRAL, EmotionCategory.APPROVAL],
        "expected_valence": "neutral",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.GOOD]
    },
    {
        "category": "Neutral - Question",
        "text": "What is the derivative of x squared?",
        "expected_primary": [EmotionCategory.NEUTRAL, EmotionCategory.CURIOSITY],
        "expected_valence": "neutral",
        "expected_readiness": [LearningReadiness.GOOD, LearningReadiness.MODERATE]
    },
    {
        "category": "Ambiguous - Surprise (Positive)",
        "text": "Wait, what? That actually worked! Surprising!",
        "expected_primary": [EmotionCategory.SURPRISE, EmotionCategory.JOY],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.GOOD, LearningReadiness.OPTIMAL]
    },
    {
        "category": "Ambiguous - Surprise (Negative)",
        "text": "What? That's not right at all. I'm shocked this is wrong.",
        "expected_primary": [EmotionCategory.SURPRISE, EmotionCategory.DISAPPOINTMENT],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.LOW]
    },
    
    # ==== CATEGORY 6: EDGE CASES ====
    {
        "category": "Edge - Very Short (Positive)",
        "text": "Yes!",
        "expected_primary": [EmotionCategory.JOY, EmotionCategory.APPROVAL, EmotionCategory.EXCITEMENT],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.GOOD, LearningReadiness.OPTIMAL]
    },
    {
        "category": "Edge - Very Short (Negative)",
        "text": "No!",
        "expected_primary": [EmotionCategory.DISAPPROVAL, EmotionCategory.ANNOYANCE, EmotionCategory.DISAPPOINTMENT],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.MODERATE, LearningReadiness.LOW]
    },
    {
        "category": "Edge - Very Short (Confused)",
        "text": "Huh?",
        "expected_primary": [EmotionCategory.CONFUSION, EmotionCategory.SURPRISE],
        "expected_valence": "neutral",
        "expected_readiness": [LearningReadiness.MODERATE]
    },
    {
        "category": "Edge - Single Word (Complex)",
        "text": "Help!",
        "expected_primary": [EmotionCategory.FEAR, EmotionCategory.NERVOUSNESS, EmotionCategory.CONFUSION],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.BLOCKED]
    },
    
    # ==== CATEGORY 7: CONTEXTUAL (Same Words, Different Meanings) ====
    {
        "category": "Context - 'I can't' (Giving Up)",
        "text": "I can't do this anymore. It's too hard.",
        "expected_primary": [EmotionCategory.DISAPPOINTMENT, EmotionCategory.SADNESS],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.BLOCKED]
    },
    {
        "category": "Context - 'I can't' (Surprise Positive)",
        "text": "I can't believe I solved it! This is amazing!",
        "expected_primary": [EmotionCategory.JOY, EmotionCategory.SURPRISE],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL]
    },
    {
        "category": "Context - 'Interesting' (Genuine)",
        "text": "This is really interesting! I want to learn more about this topic.",
        "expected_primary": [EmotionCategory.CURIOSITY, EmotionCategory.EXCITEMENT],
        "expected_valence": "positive",
        "expected_readiness": [LearningReadiness.OPTIMAL, LearningReadiness.GOOD]
    },
    {
        "category": "Context - 'Interesting' (Sarcastic/Bored)",
        "text": "Interesting. Another formula to memorize. Great.",
        "expected_primary": [EmotionCategory.ANNOYANCE, EmotionCategory.DISAPPROVAL],
        "expected_valence": "negative",
        "expected_readiness": [LearningReadiness.LOW, LearningReadiness.MODERATE]
    },
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class EmotionTestRunner:
    """Runs comprehensive emotion detection tests"""
    
    def __init__(self):
        self.engine = None
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.misclassifications = []
    
    async def initialize(self):
        """Initialize emotion engine"""
        print("\n" + "="*80)
        print("INITIALIZING EMOTION DETECTION SYSTEM")
        print("="*80)
        
        config = EmotionEngineConfig()
        self.engine = EmotionEngine(config)
        
        print("Loading ML models... (this may take 10-15 seconds)")
        start = time.time()
        await self.engine.initialize()
        elapsed = time.time() - start
        
        print(f"✅ Initialization complete ({elapsed:.2f}s)")
        print(f"   Device: {self.engine.transformer.device}")
        print(f"   Models loaded: Primary + Fallback")
        print()
    
    async def run_tests(self):
        """Run all test scenarios"""
        print("="*80)
        print("STARTING EMOTION DETECTION TESTS")
        print(f"Total scenarios: {len(TEST_SCENARIOS)}")
        print("="*80)
        print()
        
        for idx, scenario in enumerate(TEST_SCENARIOS, 1):
            result = await self._test_scenario(idx, scenario)
            self.results.append(result)
            self.total_tests += 1
            
            if result['passed']:
                self.passed_tests += 1
            else:
                self.failed_tests += 1
                self.misclassifications.append(result)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        self._print_summary()
    
    async def _test_scenario(self, idx: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single scenario"""
        category = scenario['category']
        text = scenario['text']
        expected_primary = scenario['expected_primary']
        expected_valence = scenario['expected_valence']
        expected_readiness = scenario['expected_readiness']
        
        print(f"\n[{idx}/{len(TEST_SCENARIOS)}] Testing: {category}")
        print(f"   Text: \"{text}\"")
        
        try:
            # Analyze emotion
            start = time.time()
            emotion_result = await self.engine.analyze_emotion(
                text=text,
                user_id=f"test_user_{idx}",
                session_id=f"test_session_{idx}"
            )
            elapsed_ms = (time.time() - start) * 1000
            
            # Extract results
            detected_emotion = emotion_result.primary_emotion  # This is EmotionCategory enum
            detected_confidence = emotion_result.primary_confidence
            detected_readiness = emotion_result.learning_readiness  # This is LearningReadiness enum
            detected_valence = emotion_result.pad_dimensions.pleasure
            
            # Check if primary emotion matches expected
            primary_match = detected_emotion in expected_primary
            
            # Check if readiness matches expected
            readiness_match = detected_readiness in expected_readiness
            
            # Check valence direction
            valence_match = self._check_valence(detected_valence, expected_valence)
            
            # Overall pass/fail
            passed = primary_match and readiness_match and valence_match
            
            # Print results
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}")
            print(f"   Detected: {detected_emotion} (confidence: {detected_confidence:.3f})")
            print(f"   Expected: {', '.join([e.value for e in expected_primary])}")
            print(f"   Readiness: {detected_readiness} (expected: {', '.join([r.value for r in expected_readiness])})")
            print(f"   Valence: {detected_valence:.3f} (expected: {expected_valence})")
            print(f"   Time: {elapsed_ms:.1f}ms")
            
            if not passed:
                print(f"   ⚠️  MISCLASSIFICATION DETECTED")
                if not primary_match:
                    print(f"      - Primary emotion mismatch")
                if not readiness_match:
                    print(f"      - Learning readiness mismatch")
                if not valence_match:
                    print(f"      - Valence direction mismatch")
            
            return {
                'passed': passed,
                'scenario': scenario,
                'detected_emotion': str(detected_emotion),  # Convert to string
                'detected_confidence': detected_confidence,
                'detected_readiness': str(detected_readiness),  # Convert to string
                'detected_valence': detected_valence,
                'primary_match': primary_match,
                'readiness_match': readiness_match,
                'valence_match': valence_match,
                'processing_time_ms': elapsed_ms
            }
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'passed': False,
                'scenario': scenario,
                'detected_emotion': 'error',
                'detected_confidence': 0.0,
                'detected_readiness': 'error',
                'detected_valence': 0.0,
                'primary_match': False,
                'readiness_match': False,
                'valence_match': False,
                'processing_time_ms': 0.0,
                'error': str(e)
            }
    
    def _check_valence(self, detected: float, expected: str) -> bool:
        """Check if valence direction matches expectation"""
        if expected == "positive":
            return detected > 0.2
        elif expected == "negative":
            return detected < -0.1
        elif expected == "neutral":
            return -0.2 <= detected <= 0.2
        elif expected == "neutral_or_slightly_negative":
            return detected <= 0.3
        elif expected == "mixed":
            return True  # Mixed emotions can have any valence
        return True
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"\nTotal Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests} ({self.passed_tests/self.total_tests*100:.1f}%)")
        print(f"❌ Failed: {self.failed_tests} ({self.failed_tests/self.total_tests*100:.1f}%)")
        
        if self.misclassifications:
            print(f"\n⚠️  MISCLASSIFICATIONS DETECTED: {len(self.misclassifications)}")
            print("\nDetailed Analysis:")
            
            for i, result in enumerate(self.misclassifications, 1):
                scenario = result['scenario']
                print(f"\n  [{i}] {scenario['category']}")
                print(f"      Text: \"{scenario['text']}\"")
                print(f"      Expected: {', '.join([e.value for e in scenario['expected_primary']])}")
                print(f"      Detected: {result['detected_emotion']} ({result['detected_confidence']:.3f})")
                print(f"      Issues:")
                if not result['primary_match']:
                    print(f"        - Wrong primary emotion")
                if not result['readiness_match']:
                    print(f"        - Wrong learning readiness")
                if not result['valence_match']:
                    print(f"        - Wrong emotional valence")
        else:
            print("\n🎉 ALL TESTS PASSED! No misclassifications detected.")
        
        # Performance statistics
        avg_time = sum(r.get('processing_time_ms', 0) for r in self.results) / len(self.results)
        print(f"\n📊 Performance:")
        print(f"   Average processing time: {avg_time:.1f}ms")
        print(f"   Device: {self.engine.transformer.device}")
        
        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main test execution"""
    runner = EmotionTestRunner()
    
    try:
        await runner.initialize()
        await runner.run_tests()
        
        # Return exit code based on results
        return 0 if runner.failed_tests == 0 else 1
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
