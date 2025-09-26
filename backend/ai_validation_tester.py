#!/usr/bin/env python3
"""
🔬 AI PROVIDER INTEGRATION VALIDATOR & A/B TESTING SYSTEM
Test real AI provider responses vs cached/optimized responses

This script validates:
1. Real AI provider calls are being made
2. Emotion detection enhances responses
3. System isn't defaulting to cached responses
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIProviderValidator:
    """Validates real AI provider integration vs cached responses"""
    
    def __init__(self):
        self.test_results = []
        
    async def run_validation_suite(self):
        """Run comprehensive AI validation tests"""
        logger.info("🔬 Starting AI Provider Validation Suite...")
        
        # Test 1: Direct AI Provider Calls
        await self.test_direct_provider_calls()
        
        # Test 2: Emotion Detection Enhancement
        await self.test_emotion_enhancement()
        
        # Test 3: Response Variation Analysis  
        await self.test_response_variation()
        
        # Test 4: Cache vs Real AI Detection
        await self.test_cache_vs_real_ai()
        
        # Generate comprehensive report
        await self.generate_validation_report()
    
    async def test_direct_provider_calls(self):
        """Test 1: Validate direct AI provider calls"""
        logger.info("🧪 Test 1: Direct AI Provider Calls")
        
        test_messages = [
            "I'm feeling frustrated with learning math concepts",
            "Can you explain quantum mechanics in simple terms?", 
            "I'm excited about this new programming challenge!",
            "I'm confused and need help understanding this topic"
        ]
        
        for i, message in enumerate(test_messages):
            logger.info(f"Testing message {i+1}: {message[:50]}...")
            
            # Test Groq directly
            groq_result = await self.test_groq_direct(message)
            
            # Test Emergent directly
            emergent_result = await self.test_emergent_direct(message)
            
            # Test through MasterX system
            system_result = await self.test_masterx_system(message)
            
            self.test_results.append({
                'test': 'direct_provider_calls',
                'message': message,
                'groq_result': groq_result,
                'emergent_result': emergent_result,
                'system_result': system_result,
                'timestamp': datetime.now().isoformat()
            })
            
            await asyncio.sleep(1)  # Rate limiting
    
    async def test_groq_direct(self, message: str) -> Dict[str, Any]:
        """Test Groq provider directly"""
        try:
            from quantum_intelligence.core.breakthrough_ai_integration import UltraEnterpriseGroqProvider
            import os
            
            groq_key = os.environ.get('GROQ_API_KEY')
            if not groq_key:
                return {'error': 'No GROQ_API_KEY found'}
            
            provider = UltraEnterpriseGroqProvider(groq_key)
            messages = [{"role": "user", "content": message}]
            
            start_time = time.time()
            response = await provider.generate_response(messages)
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'content': response.content,
                'provider': response.provider,
                'model': response.model,
                'response_time': response_time,
                'confidence': response.confidence,
                'empathy_score': response.empathy_score,
                'is_real_ai': len(response.content) > 50 and response.response_time > 0.1  # Basic heuristic
            }
            
        except Exception as e:
            logger.error(f"Groq direct test failed: {e}")
            return {'error': str(e), 'success': False}
    
    async def test_emergent_direct(self, message: str) -> Dict[str, Any]:
        """Test Emergent provider directly"""
        try:
            from quantum_intelligence.core.breakthrough_ai_integration import UltraEnterpriseEmergentProvider
            import os
            
            emergent_key = os.environ.get('EMERGENT_LLM_KEY')
            if not emergent_key:
                return {'error': 'No EMERGENT_LLM_KEY found'}
            
            provider = UltraEnterpriseEmergentProvider(emergent_key, model="gpt-4o-mini", provider_name="openai")
            messages = [{"role": "user", "content": message}]
            
            start_time = time.time()
            response = await provider.generate_response(messages)
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'content': response.content,
                'provider': response.provider,
                'model': response.model,
                'response_time': response_time,
                'confidence': response.confidence,
                'empathy_score': response.empathy_score,
                'is_real_ai': len(response.content) > 50 and response.response_time > 0.1  # Basic heuristic
            }
            
        except Exception as e:
            logger.error(f"Emergent direct test failed: {e}")
            return {'error': str(e), 'success': False}
    
    async def test_masterx_system(self, message: str) -> Dict[str, Any]:
        """Test through MasterX quantum intelligence system"""
        try:
            # This would test the full system pipeline
            # We'll implement this after we examine the integration
            return {
                'success': True,
                'content': 'System test placeholder',
                'note': 'Full system integration test needed'
            }
            
        except Exception as e:
            logger.error(f"MasterX system test failed: {e}")
            return {'error': str(e), 'success': False}
    
    async def test_emotion_enhancement(self):
        """Test 2: Validate emotion detection enhances AI responses"""
        logger.info("🧪 Test 2: Emotion Detection Enhancement")
        
        emotion_test_cases = [
            {
                'message': "I'm really struggling and feeling overwhelmed with this subject",
                'expected_emotion': 'frustration',
                'expected_enhancement': 'empathetic_support'
            },
            {
                'message': "This is amazing! I finally understand this concept completely!",
                'expected_emotion': 'excitement',
                'expected_enhancement': 'positive_reinforcement'
            },
            {
                'message': "I'm confused about this topic and don't know where to start",
                'expected_emotion': 'confusion',
                'expected_enhancement': 'simplified_explanation'
            }
        ]
        
        for case in emotion_test_cases:
            logger.info(f"Testing emotion case: {case['expected_emotion']}")
            
            # Test emotion detection
            emotion_result = await self.test_emotion_detection(case['message'])
            
            # Test response with emotion context
            enhanced_result = await self.test_response_with_emotion(case['message'], emotion_result)
            
            # Test response without emotion context
            baseline_result = await self.test_response_without_emotion(case['message'])
            
            self.test_results.append({
                'test': 'emotion_enhancement',
                'case': case,
                'emotion_result': emotion_result,
                'enhanced_result': enhanced_result,
                'baseline_result': baseline_result,
                'enhancement_detected': self.analyze_enhancement(enhanced_result, baseline_result),
                'timestamp': datetime.now().isoformat()
            })
    
    async def test_emotion_detection(self, message: str) -> Dict[str, Any]:
        """Test emotion detection system directly"""
        try:
            from quantum_intelligence.services.emotional.authentic_emotion_engine_v9 import RevolutionaryAuthenticEmotionEngineV9
            
            engine = RevolutionaryAuthenticEmotionEngineV9()
            await engine.initialize()
            
            input_data = {
                'text': message,
                'behavioral': {
                    'response_length': len(message),
                    'response_time': 5.0,
                    'session_duration': 300
                }
            }
            
            result = await engine.analyze_authentic_emotion('test_user', input_data)
            
            return {
                'success': True,
                'primary_emotion': str(result.primary_emotion),
                'emotion_confidence': result.emotion_confidence,
                'learning_readiness': str(result.learning_readiness),
                'intervention_needed': result.intervention_needed,
                'intervention_level': str(result.intervention_level)
            }
            
        except Exception as e:
            logger.error(f"Emotion detection test failed: {e}")
            return {'error': str(e), 'success': False}
    
    async def test_response_with_emotion(self, message: str, emotion_context: Dict[str, Any]) -> Dict[str, Any]:
        """Test AI response with emotion context"""
        # Implementation would inject emotion context into AI call
        return {'placeholder': True, 'message': 'Implementation needed'}
    
    async def test_response_without_emotion(self, message: str) -> Dict[str, Any]:
        """Test AI response without emotion context"""
        # Implementation would call AI without emotion context
        return {'placeholder': True, 'message': 'Implementation needed'}
    
    def analyze_enhancement(self, enhanced_result: Dict[str, Any], baseline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if emotion detection enhanced the response"""
        # Implementation would compare responses for enhancement indicators
        return {
            'enhanced': False,
            'analysis': 'Analysis implementation needed'
        }
    
    async def test_response_variation(self):
        """Test 3: Response variation analysis"""
        logger.info("🧪 Test 3: Response Variation Analysis")
        
        test_message = "Can you help me understand machine learning?"
        responses = []
        
        # Generate multiple responses to same message
        for i in range(5):
            logger.info(f"Generating response variation {i+1}/5...")
            
            # Test with different providers/approaches
            groq_response = await self.test_groq_direct(test_message)
            emergent_response = await self.test_emergent_direct(test_message)
            
            responses.append({
                'iteration': i+1,
                'groq': groq_response,
                'emergent': emergent_response,
                'timestamp': datetime.now().isoformat()
            })
            
            await asyncio.sleep(2)  # Rate limiting
        
        # Analyze variation
        variation_analysis = self.analyze_response_variation(responses)
        
        self.test_results.append({
            'test': 'response_variation',
            'message': test_message,
            'responses': responses,
            'variation_analysis': variation_analysis
        })
    
    def analyze_response_variation(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze variation in responses"""
        # Check if responses are too similar (indicating caching)
        groq_contents = [r['groq'].get('content', '') for r in responses if r['groq'].get('success')]
        emergent_contents = [r['emergent'].get('content', '') for r in responses if r['emergent'].get('success')]
        
        # Simple similarity check
        groq_unique = len(set(groq_contents))
        emergent_unique = len(set(emergent_contents))
        
        return {
            'groq_responses': len(groq_contents),
            'groq_unique': groq_unique,
            'groq_variation_rate': groq_unique / max(len(groq_contents), 1),
            'emergent_responses': len(emergent_contents),
            'emergent_unique': emergent_unique,
            'emergent_variation_rate': emergent_unique / max(len(emergent_contents), 1),
            'likely_cached': groq_unique == 1 and len(groq_contents) > 1,  # Same response multiple times
            'analysis': 'High variation suggests real AI, low variation suggests caching'
        }
    
    async def test_cache_vs_real_ai(self):
        """Test 4: Detect cached vs real AI responses"""
        logger.info("🧪 Test 4: Cache vs Real AI Detection")
        
        # Test with repeated identical messages
        test_message = "What is the capital of France?"
        
        response_times = []
        contents = []
        
        for i in range(3):
            logger.info(f"Testing repeated message {i+1}/3...")
            
            start_time = time.time()
            result = await self.test_groq_direct(test_message)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            if result.get('success'):
                contents.append(result.get('content', ''))
            
            await asyncio.sleep(1)
        
        # Analyze patterns
        cache_analysis = {
            'response_times': response_times,
            'avg_response_time': sum(response_times) / len(response_times),
            'time_variation': max(response_times) - min(response_times),
            'contents_identical': len(set(contents)) == 1 if contents else False,
            'likely_cached': (max(response_times) - min(response_times)) < 0.1 and len(set(contents)) == 1,
            'analysis': 'Very consistent timing + identical content suggests caching'
        }
        
        self.test_results.append({
            'test': 'cache_vs_real_ai',
            'message': test_message,
            'cache_analysis': cache_analysis
        })
    
    async def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("📊 Generating Validation Report...")
        
        report = {
            'validation_summary': {
                'total_tests': len(self.test_results),
                'test_timestamp': datetime.now().isoformat(),
                'validation_version': '1.0'
            },
            'test_results': self.test_results,
            'conclusions': self.generate_conclusions()
        }
        
        # Save report
        report_filename = f"/app/backend/ai_validation_report_{int(time.time())}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Validation report saved: {report_filename}")
        
        # Print summary
        self.print_validation_summary(report)
    
    def generate_conclusions(self) -> Dict[str, Any]:
        """Generate conclusions from test results"""
        conclusions = {
            'ai_providers_working': False,
            'emotion_detection_working': False,
            'emotion_enhances_responses': False,
            'caching_concerns': False,
            'recommendations': []
        }
        
        # Analyze results
        for result in self.test_results:
            if result.get('test') == 'direct_provider_calls':
                if result.get('groq_result', {}).get('success'):
                    conclusions['ai_providers_working'] = True
                    
            elif result.get('test') == 'emotion_enhancement':
                if result.get('emotion_result', {}).get('success'):
                    conclusions['emotion_detection_working'] = True
                    
            elif result.get('test') == 'cache_vs_real_ai':
                if result.get('cache_analysis', {}).get('likely_cached'):
                    conclusions['caching_concerns'] = True
        
        # Generate recommendations
        if not conclusions['ai_providers_working']:
            conclusions['recommendations'].append('Fix AI provider integrations - providers not responding correctly')
            
        if not conclusions['emotion_detection_working']:
            conclusions['recommendations'].append('Fix emotion detection system - not analyzing emotions properly')
            
        if conclusions['caching_concerns']:
            conclusions['recommendations'].append('Review caching strategy - responses may be overly cached')
            
        if conclusions['ai_providers_working'] and not conclusions['emotion_enhances_responses']:
            conclusions['recommendations'].append('Implement emotion-context injection into AI provider calls')
        
        return conclusions
    
    def print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary to console"""
        print("\n" + "="*80)
        print("🔬 AI PROVIDER VALIDATION REPORT SUMMARY")
        print("="*80)
        
        conclusions = report['conclusions']
        
        print(f"✅ AI Providers Working: {conclusions['ai_providers_working']}")
        print(f"🧠 Emotion Detection Working: {conclusions['emotion_detection_working']}")
        print(f"🎯 Emotion Enhances Responses: {conclusions['emotion_enhances_responses']}")
        print(f"⚠️  Caching Concerns: {conclusions['caching_concerns']}")
        
        print("\n📝 RECOMMENDATIONS:")
        for rec in conclusions['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n📄 Full report: {report.get('report_filename', 'ai_validation_report_*.json')}")
        print("="*80 + "\n")

async def main():
    """Run the AI validation suite"""
    validator = AIProviderValidator()
    await validator.run_validation_suite()

if __name__ == "__main__":
    asyncio.run(main())