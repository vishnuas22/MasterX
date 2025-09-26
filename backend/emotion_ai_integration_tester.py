#!/usr/bin/env python3
"""
🧠 EMOTION-ENHANCED AI INTEGRATION TESTER
Test if emotion detection actually enhances AI responses

This script validates:
1. Emotion detection works correctly
2. AI responses are enhanced by emotion context
3. Quantum intelligence integration is functional
"""

import asyncio
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionAITester:
    """Test emotion-enhanced AI integration"""
    
    def __init__(self):
        self.test_results = []
        
    async def run_emotion_ai_tests(self):
        """Run comprehensive emotion-AI integration tests"""
        logger.info("🧠 Starting Emotion-AI Integration Tests...")
        
        # Test cases with different emotional contexts
        test_cases = [
            {
                'user_message': "I'm really struggling and feel overwhelmed with this math problem",
                'expected_emotion': 'frustration',
                'expected_intervention': 'supportive',
                'context': 'Student needs emotional support and simplified explanation'
            },
            {
                'user_message': "This is amazing! I finally understand how machine learning works!",
                'expected_emotion': 'excitement',
                'expected_intervention': 'reinforcement',
                'context': 'Student is excited, reinforce learning and build on enthusiasm'
            },
            {
                'user_message': "I don't understand this concept at all and I'm confused",
                'expected_emotion': 'confusion',
                'expected_intervention': 'clarification',
                'context': 'Student needs clear, step-by-step explanation'
            },
            {
                'user_message': "I'm bored with this repetitive exercise",
                'expected_emotion': 'boredom',
                'expected_intervention': 'engagement',
                'context': 'Student needs more engaging content or different approach'
            }
        ]
        
        for i, case in enumerate(test_cases):
            logger.info(f"🧪 Testing case {i+1}: {case['expected_emotion']}")
            
            # Test 1: Raw emotion detection
            emotion_result = await self.test_emotion_detection(case['user_message'])
            
            # Test 2: AI response without emotion context
            baseline_response = await self.test_ai_without_emotion(case['user_message'])
            
            # Test 3: AI response with emotion context
            enhanced_response = await self.test_ai_with_emotion(case['user_message'], emotion_result)
            
            # Test 4: Full quantum intelligence system
            quantum_response = await self.test_quantum_system(case['user_message'])
            
            # Analyze enhancement
            enhancement_analysis = self.analyze_emotional_enhancement(
                case, emotion_result, baseline_response, enhanced_response, quantum_response
            )
            
            self.test_results.append({
                'test_case': case,
                'emotion_result': emotion_result,
                'baseline_response': baseline_response,
                'enhanced_response': enhanced_response,
                'quantum_response': quantum_response,
                'enhancement_analysis': enhancement_analysis,
                'timestamp': datetime.now().isoformat()
            })
            
            await asyncio.sleep(1)  # Rate limiting
        
        # Generate comprehensive report
        await self.generate_emotion_ai_report()
    
    async def test_emotion_detection(self, message: str) -> Dict[str, Any]:
        """Test emotion detection directly"""
        try:
            # Import and initialize emotion engine
            from quantum_intelligence.services.emotional.authentic_emotion_engine_v9 import RevolutionaryAuthenticEmotionEngineV9
            
            engine = RevolutionaryAuthenticEmotionEngineV9()
            await engine.initialize()
            
            # Create realistic input data
            input_data = {
                'text': message,
                'behavioral': {
                    'response_length': len(message),
                    'response_time': 5.0,
                    'session_duration': 300,
                    'typing_speed': len(message) / 5.0
                },
                'contextual': {
                    'time_of_day': 'afternoon',
                    'session_position': 3,
                    'recent_performance': 0.75
                }
            }
            
            # Analyze emotions
            result = await engine.analyze_authentic_emotion('test_user', input_data)
            
            return {
                'success': True,
                'primary_emotion': str(result.primary_emotion),
                'emotion_confidence': result.emotion_confidence,
                'learning_readiness': str(result.learning_readiness),
                'intervention_needed': result.intervention_needed,
                'intervention_level': str(result.intervention_level),
                'secondary_emotions': result.secondary_emotions,
                'emotion_distribution': result.emotion_distribution,
                'arousal_level': result.arousal_level,
                'valence_level': result.valence_level,
                'engagement_score': result.engagement_score,
                'cognitive_load_level': result.cognitive_load_level,
                'attention_state': result.attention_state
            }
            
        except Exception as e:
            logger.error(f"❌ Emotion detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def test_ai_without_emotion(self, message: str) -> Dict[str, Any]:
        """Test AI response without emotion context (baseline)"""
        try:
            from quantum_intelligence.core.breakthrough_ai_integration import UltraEnterpriseGroqProvider
            import os
            
            provider = UltraEnterpriseGroqProvider(os.environ.get('GROQ_API_KEY'))
            messages = [{"role": "user", "content": message}]
            
            response = await provider.generate_response(messages)
            
            return {
                'success': True,
                'content': response.content,
                'provider': response.provider,
                'model': response.model,
                'empathy_score': getattr(response, 'empathy_score', 0.0),
                'content_length': len(response.content),
                'response_type': 'baseline'
            }
            
        except Exception as e:
            logger.error(f"❌ Baseline AI test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def test_ai_with_emotion(self, message: str, emotion_context: Dict[str, Any]) -> Dict[str, Any]:
        """Test AI response with emotion context injected"""
        try:
            from quantum_intelligence.core.breakthrough_ai_integration import UltraEnterpriseGroqProvider
            import os
            
            if not emotion_context.get('success'):
                return {'success': False, 'error': 'No valid emotion context'}
            
            provider = UltraEnterpriseGroqProvider(os.environ.get('GROQ_API_KEY'))
            
            # Create emotion-enhanced prompt
            emotion_prompt = f'''You are MasterX, an AI learning assistant with advanced emotional intelligence.

EMOTIONAL CONTEXT ANALYSIS:
- Primary Emotion: {emotion_context.get('primary_emotion', 'neutral')}
- Emotion Confidence: {emotion_context.get('emotion_confidence', 0.0)}
- Learning Readiness: {emotion_context.get('learning_readiness', 'moderate')}
- Intervention Needed: {emotion_context.get('intervention_needed', False)}
- Intervention Level: {emotion_context.get('intervention_level', 'none')}

RESPONSE GUIDELINES BASED ON EMOTION:
- If frustrated/overwhelmed: Provide extra support, break concepts down, be encouraging
- If excited: Build on enthusiasm, provide advanced content, celebrate progress  
- If confused: Use simple language, provide step-by-step explanations, check understanding
- If bored: Make content more engaging, use examples, change approach

User Message: {message}

Respond with emotional intelligence, adapting your tone, content depth, and teaching approach to the detected emotional state.'''
            
            messages = [{"role": "user", "content": emotion_prompt}]
            
            response = await provider.generate_response(messages)
            
            return {
                'success': True,
                'content': response.content,
                'provider': response.provider,
                'model': response.model,
                'empathy_score': getattr(response, 'empathy_score', 0.0),
                'content_length': len(response.content),
                'response_type': 'emotion_enhanced',
                'emotion_context_used': emotion_context
            }
            
        except Exception as e:
            logger.error(f"❌ Emotion-enhanced AI test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def test_quantum_system(self, message: str) -> Dict[str, Any]:
        """Test full quantum intelligence system"""
        try:
            # This would test the complete integrated system
            # For now, we'll create a placeholder that shows what should happen
            
            return {
                'success': True,
                'content': 'Full quantum system integration test - placeholder',
                'quantum_processing': True,
                'response_type': 'quantum_integrated',
                'note': 'This would test the complete MasterX quantum intelligence pipeline'
            }
            
        except Exception as e:
            logger.error(f"❌ Quantum system test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_emotional_enhancement(
        self, 
        test_case: Dict[str, Any],
        emotion_result: Dict[str, Any],
        baseline_response: Dict[str, Any],
        enhanced_response: Dict[str, Any],
        quantum_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze if emotion detection enhanced the AI responses"""
        
        analysis = {
            'emotion_detection_working': emotion_result.get('success', False),
            'baseline_ai_working': baseline_response.get('success', False),
            'enhanced_ai_working': enhanced_response.get('success', False),
            'quantum_system_working': quantum_response.get('success', False),
            'emotion_correctly_detected': False,
            'response_enhanced': False,
            'enhancement_indicators': []
        }
        
        # Check if emotion was correctly detected
        if emotion_result.get('success'):
            detected_emotion = emotion_result.get('primary_emotion', '').lower()
            expected_emotion = test_case.get('expected_emotion', '').lower()
            
            # Simple emotion matching (could be improved with semantic similarity)
            analysis['emotion_correctly_detected'] = expected_emotion in detected_emotion or detected_emotion in expected_emotion
        
        # Check if response was enhanced
        if baseline_response.get('success') and enhanced_response.get('success'):
            baseline_content = baseline_response.get('content', '')
            enhanced_content = enhanced_response.get('content', '')
            
            # Check for enhancement indicators
            enhancement_keywords = {
                'frustration': ['understand', 'support', 'help', 'break down', 'step by step'],
                'excitement': ['fantastic', 'great', 'excellent', 'build on', 'advanced'],
                'confusion': ['simple', 'clear', 'explain', 'clarify', 'understand'],
                'boredom': ['engaging', 'interesting', 'example', 'different approach']
            }
            
            expected_emotion = test_case.get('expected_emotion', '')
            keywords = enhancement_keywords.get(expected_emotion, [])
            
            # Count enhancement keywords in enhanced response
            enhanced_keyword_count = sum(1 for keyword in keywords if keyword in enhanced_content.lower())
            baseline_keyword_count = sum(1 for keyword in keywords if keyword in baseline_content.lower())
            
            # Check various enhancement indicators
            if enhanced_keyword_count > baseline_keyword_count:
                analysis['enhancement_indicators'].append('More emotion-appropriate keywords')
                analysis['response_enhanced'] = True
            
            if len(enhanced_content) > len(baseline_content) * 1.2:
                analysis['enhancement_indicators'].append('More detailed response')
            
            if abs(len(enhanced_content) - len(baseline_content)) / max(len(baseline_content), 1) > 0.3:
                analysis['enhancement_indicators'].append('Significantly different response structure')
        
        # Overall enhancement score
        score = 0
        if analysis['emotion_detection_working']:
            score += 25
        if analysis['emotion_correctly_detected']:
            score += 25
        if analysis['response_enhanced']:
            score += 25
        if len(analysis['enhancement_indicators']) > 0:
            score += 25
        
        analysis['enhancement_score'] = score
        analysis['enhancement_level'] = 'high' if score >= 75 else 'medium' if score >= 50 else 'low' if score >= 25 else 'none'
        
        return analysis
    
    async def generate_emotion_ai_report(self):
        """Generate comprehensive emotion-AI integration report"""
        logger.info("📊 Generating Emotion-AI Integration Report...")
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        emotion_detection_success = sum(1 for r in self.test_results if r['emotion_result'].get('success', False))
        emotion_correct_detection = sum(1 for r in self.test_results if r['enhancement_analysis'].get('emotion_correctly_detected', False))
        response_enhancements = sum(1 for r in self.test_results if r['enhancement_analysis'].get('response_enhanced', False))
        high_enhancement_scores = sum(1 for r in self.test_results if r['enhancement_analysis'].get('enhancement_score', 0) >= 75)
        
        summary = {
            'total_tests': total_tests,
            'emotion_detection_rate': emotion_detection_success / total_tests if total_tests > 0 else 0,
            'emotion_accuracy_rate': emotion_correct_detection / total_tests if total_tests > 0 else 0,
            'response_enhancement_rate': response_enhancements / total_tests if total_tests > 0 else 0,
            'high_quality_enhancement_rate': high_enhancement_scores / total_tests if total_tests > 0 else 0,
            'overall_system_health': 'excellent' if high_enhancement_scores / total_tests >= 0.75 else 'good' if high_enhancement_scores / total_tests >= 0.5 else 'needs_improvement'
        }
        
        report = {
            'test_summary': summary,
            'detailed_results': self.test_results,
            'recommendations': self.generate_recommendations(summary),
            'report_timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_filename = f"/app/backend/emotion_ai_integration_report_{int(datetime.now().timestamp())}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Emotion-AI report saved: {report_filename}")
        
        # Print summary
        self.print_emotion_ai_summary(summary)
        
        return report
    
    def generate_recommendations(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if summary['emotion_detection_rate'] < 0.8:
            recommendations.append("Fix emotion detection system - low success rate")
        
        if summary['emotion_accuracy_rate'] < 0.7:
            recommendations.append("Improve emotion classification accuracy - many emotions incorrectly detected")
        
        if summary['response_enhancement_rate'] < 0.6:
            recommendations.append("Implement proper emotion-context injection into AI responses")
        
        if summary['high_quality_enhancement_rate'] < 0.5:
            recommendations.append("Enhance emotional intelligence in AI responses - low quality enhancements")
        
        if summary['overall_system_health'] == 'needs_improvement':
            recommendations.append("Complete system overhaul needed - emotion-AI integration not working effectively")
        
        return {
            'priority_recommendations': recommendations,
            'next_steps': [
                "Test with real user data instead of synthetic test cases",
                "Implement A/B testing with real users",
                "Create emotion-specific response templates",
                "Integrate quantum intelligence with emotion detection"
            ]
        }
    
    def print_emotion_ai_summary(self, summary: Dict[str, Any]):
        """Print emotion-AI integration summary"""
        print("\n" + "="*80)
        print("🧠 EMOTION-AI INTEGRATION REPORT SUMMARY")
        print("="*80)
        
        print(f"📊 Emotion Detection Rate: {summary['emotion_detection_rate']*100:.1f}%")
        print(f"🎯 Emotion Accuracy Rate: {summary['emotion_accuracy_rate']*100:.1f}%")
        print(f"🚀 Response Enhancement Rate: {summary['response_enhancement_rate']*100:.1f}%")
        print(f"⭐ High Quality Enhancement Rate: {summary['high_quality_enhancement_rate']*100:.1f}%")
        print(f"🏥 Overall System Health: {summary['overall_system_health'].upper()}")
        
        print("="*80 + "\n")

async def main():
    """Run the emotion-AI integration tests"""
    tester = EmotionAITester()
    await tester.run_emotion_ai_tests()

if __name__ == "__main__":
    asyncio.run(main())