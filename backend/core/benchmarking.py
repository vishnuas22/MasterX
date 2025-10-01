"""
Dynamic AI Benchmarking System
Following specifications from 4.DYNAMIC_AI_ROUTING_SYSTEM.md

Continuously tests all providers across task categories to determine best performers.
No hardcoded assumptions - all decisions based on real performance data.

Key Features:
- 6 task categories (coding, math, research, language, empathy, general)
- Quality, speed, and cost scoring
- MongoDB storage for historical analysis
- Automated scheduling (every 1-12 hours)
- Real-time provider performance tracking
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from core.models import BenchmarkResult, BenchmarkTestResult
from core.ai_providers import UniversalProvider, ProviderRegistry
from utils.database import get_database

logger = logging.getLogger(__name__)


# ============================================================================
# BENCHMARK TEST SUITE CONFIGURATION
# ============================================================================

BENCHMARK_CATEGORIES = {
    "coding": {
        "description": "Code generation, debugging, algorithms",
        "tests": [
            {
                "id": "coding_001",
                "prompt": "Explain how the merge sort algorithm works, including its time complexity and why it's efficient.",
                "expected_keywords": ["divide", "conquer", "merge", "O(n log n)", "recursive", "sorted"],
                "min_length": 150,
                "scoring_rubric": {
                    "correctness": 0.4,
                    "clarity": 0.3,
                    "completeness": 0.3
                }
            },
            {
                "id": "coding_002",
                "prompt": """Debug this Python code and explain the error:
```python
for i in range(10):
    print(i)
  print('Done')
```""",
                "expected_keywords": ["indentation", "error", "indent", "spaces", "IndentationError"],
                "correct_answer_contains": "IndentationError",
                "min_length": 50
            },
            {
                "id": "coding_003",
                "prompt": "Write a Python function that reverses a string without using built-in reverse functions.",
                "expected_output_format": "function",
                "must_contain": ["def ", "return"],
                "min_length": 50
            }
        ],
        "weights": {
            "quality": 0.5,
            "speed": 0.3,
            "cost": 0.2
        }
    },
    
    "math": {
        "description": "Mathematical problem solving and explanations",
        "tests": [
            {
                "id": "math_001",
                "prompt": "Solve the equation: 2x + 5 = 15. Show all steps clearly.",
                "expected_answer": "5",
                "expected_keywords": ["subtract", "5", "divide", "2", "x = 5", "x=5"],
                "min_steps": 3,
                "min_length": 50
            },
            {
                "id": "math_002",
                "prompt": "Explain what a derivative is in calculus in simple terms for a beginner.",
                "expected_keywords": ["rate", "change", "slope", "tangent", "instantaneous"],
                "min_length": 100
            },
            {
                "id": "math_003",
                "prompt": "Calculate the area of a circle with radius 7. Show the formula and calculation.",
                "expected_answer": "153",
                "must_contain": ["π", "²", "49"],
                "min_length": 50
            }
        ],
        "weights": {
            "quality": 0.6,
            "speed": 0.3,
            "cost": 0.1
        }
    },
    
    "research": {
        "description": "Deep analysis, research, citations",
        "tests": [
            {
                "id": "research_001",
                "prompt": "Analyze the impact of artificial intelligence on modern education. Provide evidence-based insights.",
                "expected_keywords": ["research", "study", "evidence", "data", "impact", "learning", "technology"],
                "min_length": 300,
                "depth_required": "high"
            },
            {
                "id": "research_002",
                "prompt": "Compare and contrast quantum computing versus classical computing.",
                "expected_keywords": ["quantum", "classical", "qubit", "bit", "difference", "advantage", "superposition"],
                "min_length": 250
            }
        ],
        "weights": {
            "quality": 0.7,
            "speed": 0.2,
            "cost": 0.1
        }
    },
    
    "language": {
        "description": "Grammar, writing, translation, language learning",
        "tests": [
            {
                "id": "language_001",
                "prompt": "Correct this sentence and explain the errors: 'Me and him goes to the store yesterday'",
                "expected_answer": "He and I went",
                "expected_keywords": ["He and I", "went", "subject", "verb", "past tense"],
                "min_length": 50
            },
            {
                "id": "language_002",
                "prompt": "Translate to Spanish: 'I love learning new languages'",
                "expected_answer": "Me encanta aprender nuevos idiomas",
                "language": "spanish",
                "min_length": 20
            }
        ],
        "weights": {
            "quality": 0.5,
            "speed": 0.4,
            "cost": 0.1
        }
    },
    
    "empathy": {
        "description": "Emotional support, encouragement, motivation",
        "tests": [
            {
                "id": "empathy_001",
                "prompt": "I'm really frustrated with this math problem and want to give up.",
                "expected_tone": "supportive",
                "expected_keywords": ["understand", "difficult", "help", "together", "try", "can"],
                "should_avoid": ["easy", "simple", "just do"],
                "min_length": 80
            },
            {
                "id": "empathy_002",
                "prompt": "I failed my test again. I feel like I'm not smart enough.",
                "expected_tone": "encouraging",
                "expected_keywords": ["capable", "growth", "learn", "practice", "progress", "believe"],
                "should_avoid": ["stupid", "dumb", "failure", "give up"],
                "min_length": 80
            }
        ],
        "weights": {
            "quality": 0.8,  # Empathy quality is critical
            "speed": 0.2,
            "cost": 0.0      # Cost doesn't matter for emotional support
        }
    },
    
    "general": {
        "description": "General conversation, Q&A, summaries",
        "tests": [
            {
                "id": "general_001",
                "prompt": "What is photosynthesis? Explain simply.",
                "expected_keywords": ["plants", "light", "energy", "chlorophyll", "oxygen", "glucose"],
                "min_length": 100
            },
            {
                "id": "general_002",
                "prompt": "Summarize the water cycle in 3 sentences.",
                "max_length": 200,
                "expected_keywords": ["evaporation", "condensation", "precipitation", "water"],
                "min_length": 50
            }
        ],
        "weights": {
            "quality": 0.5,
            "speed": 0.4,
            "cost": 0.1
        }
    }
}


# Provider pricing (per 1M tokens) - Update regularly
PROVIDER_PRICING = {
    'openai': {'input': 2.50, 'output': 10.00},
    'anthropic': {'input': 3.00, 'output': 15.00},
    'gemini': {'input': 0.075, 'output': 0.30},
    'groq': {'input': 0.05, 'output': 0.08},
    'emergent': {'input': 2.50, 'output': 10.00},  # Similar to OpenAI
    'together': {'input': 0.20, 'output': 0.80},
    'perplexity': {'input': 1.00, 'output': 1.00},
    'deepseek': {'input': 0.14, 'output': 0.28},
    'mistral': {'input': 0.70, 'output': 2.00},
    'cohere': {'input': 0.50, 'output': 1.50}
}


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

class BenchmarkEngine:
    """
    Continuous benchmarking system for AI providers
    
    Runs automated tests across all providers and categories:
    - Measures quality (0-100), speed (ms), cost ($)
    - Stores results in MongoDB
    - Provides latest benchmarks for smart routing
    """
    
    def __init__(
        self,
        registry: ProviderRegistry,
        universal_provider: UniversalProvider
    ):
        self.registry = registry
        self.universal = universal_provider
        self.db = None
        self.benchmarks_collection = None
        self.is_running = False
        
        logger.info("✅ BenchmarkEngine initialized")
    
    async def initialize_db(self):
        """Initialize database connection"""
        if not self.db:
            self.db = get_database()
            self.benchmarks_collection = self.db['benchmark_results']
            logger.info("✅ BenchmarkEngine database initialized")
    
    async def run_benchmarks(
        self,
        categories: Optional[List[str]] = None,
        providers: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """
        Run benchmarks across all providers and categories
        
        Args:
            categories: List of categories to test (None = all)
            providers: List of providers to test (None = all discovered)
        
        Returns:
            Dict[category][provider] = BenchmarkResult
        """
        
        if self.is_running:
            logger.warning("⚠️ Benchmarks already running, skipping...")
            return {}
        
        self.is_running = True
        await self.initialize_db()
        
        logger.info("🚀 Starting benchmark run...")
        
        # Determine what to test
        test_categories = categories or list(BENCHMARK_CATEGORIES.keys())
        test_providers = providers or list(self.registry.get_all_providers().keys())
        
        logger.info(f"📊 Testing {len(test_providers)} providers across {len(test_categories)} categories")
        
        results = {}
        timestamp = datetime.utcnow()
        
        for category in test_categories:
            results[category] = {}
            category_config = BENCHMARK_CATEGORIES[category]
            
            logger.info(f"  📁 Testing category: {category}")
            
            for provider_name in test_providers:
                logger.info(f"    🤖 Testing provider: {provider_name}")
                
                try:
                    # Run all tests for this provider+category
                    test_results = await self._run_category_tests(
                        provider_name,
                        category,
                        category_config['tests']
                    )
                    
                    # Calculate aggregate scores
                    benchmark_result = self._calculate_scores(
                        provider_name,
                        category,
                        test_results,
                        category_config['weights']
                    )
                    
                    results[category][provider_name] = benchmark_result
                    
                    # Save to database
                    await self._save_benchmark_result(
                        category,
                        provider_name,
                        benchmark_result,
                        timestamp
                    )
                    
                    logger.info(f"      ✅ Score: {benchmark_result.final_score:.1f} (Q:{benchmark_result.quality_score:.1f} S:{benchmark_result.speed_score:.1f} C:{benchmark_result.cost_score:.1f})")
                
                except Exception as e:
                    logger.error(f"      ❌ Error testing {provider_name} on {category}: {e}")
                    continue
        
        self.is_running = False
        logger.info("✅ Benchmark run complete!")
        
        return results
    
    async def _run_category_tests(
        self,
        provider_name: str,
        category: str,
        tests: List[Dict]
    ) -> List[Dict]:
        """Run all tests in a category for one provider"""
        
        results = []
        
        for test in tests:
            try:
                start_time = time.time()
                
                # Generate response
                response = await self.universal.generate(
                    provider_name=provider_name,
                    prompt=test['prompt'],
                    max_tokens=2000
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Evaluate response quality
                quality_score = self._evaluate_quality(response.content, test)
                
                results.append({
                    'test_id': test['id'],
                    'response': response,
                    'quality_score': quality_score,
                    'time_ms': elapsed_ms,
                    'passed': quality_score >= 60  # 60% threshold
                })
                
            except Exception as e:
                logger.error(f"Test {test['id']} failed for {provider_name}: {e}")
                results.append({
                    'test_id': test['id'],
                    'response': None,
                    'quality_score': 0,
                    'time_ms': 0,
                    'passed': False,
                    'error': str(e)
                })
        
        return results
    
    def _evaluate_quality(
        self,
        response_content: str,
        test: Dict
    ) -> float:
        """
        Evaluate response quality (0-100 score)
        
        Checks:
        - Contains expected keywords
        - Correct answer (if applicable)
        - Appropriate length
        - Avoids problematic phrases
        - Tone matches expectations
        """
        
        if not response_content:
            return 0.0
        
        score = 0.0
        content_lower = response_content.lower()
        
        # Check expected keywords (40 points)
        if 'expected_keywords' in test:
            keywords = test['expected_keywords']
            found = sum(1 for kw in keywords if kw.lower() in content_lower)
            score += (found / len(keywords)) * 40
        
        # Check correct answer (30 points)
        if 'expected_answer' in test:
            if test['expected_answer'].lower() in content_lower:
                score += 30
        
        # Check correct answer contains (30 points)
        if 'correct_answer_contains' in test:
            if test['correct_answer_contains'].lower() in content_lower:
                score += 30
        
        # Check minimum length (10 points)
        if 'min_length' in test:
            if len(response_content) >= test['min_length']:
                score += 10
            else:
                score += (len(response_content) / test['min_length']) * 10
        
        # Check maximum length (don't penalize, just note)
        if 'max_length' in test:
            if len(response_content) > test['max_length']:
                score -= 5  # Small penalty for being too verbose
        
        # Check must contain (10 points)
        if 'must_contain' in test:
            phrases = test['must_contain']
            found = sum(1 for phrase in phrases if phrase.lower() in content_lower)
            score += (found / len(phrases)) * 10
        
        # Check should avoid (deduct 20 points if found)
        if 'should_avoid' in test:
            avoid = test['should_avoid']
            found_bad = sum(1 for phrase in avoid if phrase.lower() in content_lower)
            if found_bad > 0:
                score -= 20
        
        # Tone check for empathy (10 points)
        if 'expected_tone' in test:
            if test['expected_tone'] == 'supportive':
                supportive_words = ['understand', 'help', 'together', 'support', 'can do']
                found = sum(1 for word in supportive_words if word in content_lower)
                score += min(found * 2.5, 10)
            
            elif test['expected_tone'] == 'encouraging':
                encouraging_words = ['capable', 'believe', 'progress', 'growth', 'improve']
                found = sum(1 for word in encouraging_words if word in content_lower)
                score += min(found * 2.5, 10)
        
        # Ensure score is between 0 and 100
        return max(0, min(score, 100))
    
    def _calculate_scores(
        self,
        provider_name: str,
        category: str,
        test_results: List[Dict],
        weights: Dict[str, float]
    ) -> BenchmarkResult:
        """Calculate aggregate scores for provider in category"""
        
        # Quality score (average of all test scores)
        quality_scores = [r['quality_score'] for r in test_results if r['quality_score'] is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Speed score (average response time)
        response_times = [r['time_ms'] for r in test_results if r['time_ms'] > 0]
        avg_time_ms = sum(response_times) / len(response_times) if response_times else 99999
        
        # Cost score (estimated cost per request)
        token_counts = [r['response'].tokens_used for r in test_results if r['response']]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_cost = self._estimate_cost(provider_name, avg_tokens)
        
        # Normalize scores (0-100)
        # Speed: faster = higher score (10000ms = baseline)
        speed_score = min(100, (10000 / avg_time_ms) * 100) if avg_time_ms > 0 else 0
        
        # Cost: cheaper = higher score ($0.01 = baseline)
        cost_score = min(100, (0.01 / avg_cost) * 100) if avg_cost > 0 else 100
        
        # Calculate weighted final score
        final_score = (
            weights['quality'] * avg_quality +
            weights['speed'] * speed_score +
            weights['cost'] * cost_score
        )
        
        # Get model name
        provider_config = self.registry.get_provider(provider_name)
        model_name = provider_config.get('model_name', 'unknown') if provider_config else 'unknown'
        
        # Create test results summary
        test_results_summary = []
        for r in test_results:
            test_results_summary.append(BenchmarkTestResult(
                test_id=r['test_id'],
                quality=r['quality_score'],
                time_ms=r['time_ms'],
                passed=r['passed']
            ))
        
        return BenchmarkResult(
            category=category,
            provider=provider_name,
            model_name=model_name,
            timestamp=datetime.utcnow(),
            quality_score=avg_quality,
            speed_score=speed_score,
            cost_score=cost_score,
            final_score=final_score,
            avg_response_time_ms=avg_time_ms,
            avg_cost=avg_cost,
            tests_passed=sum(1 for r in test_results if r['passed']),
            tests_total=len(test_results),
            test_results=test_results_summary
        )
    
    def _estimate_cost(self, provider_name: str, tokens: float) -> float:
        """Estimate cost per request"""
        
        if provider_name not in PROVIDER_PRICING:
            # Default fallback pricing
            return 0.00001 * tokens
        
        pricing = PROVIDER_PRICING[provider_name]
        
        # Assume 50/50 split input/output tokens
        input_tokens = tokens * 0.5
        output_tokens = tokens * 0.5
        
        # Calculate cost (pricing is per 1M tokens)
        cost = (
            (input_tokens * pricing['input'] / 1_000_000) +
            (output_tokens * pricing['output'] / 1_000_000)
        )
        
        return cost
    
    async def _save_benchmark_result(
        self,
        category: str,
        provider: str,
        result: BenchmarkResult,
        timestamp: datetime
    ):
        """Save benchmark result to MongoDB"""
        
        try:
            document = {
                '_id': result.id,
                'category': category,
                'provider': provider,
                'model_name': result.model_name,
                'quality_score': result.quality_score,
                'speed_score': result.speed_score,
                'cost_score': result.cost_score,
                'final_score': result.final_score,
                'avg_response_time_ms': result.avg_response_time_ms,
                'avg_cost': result.avg_cost,
                'tests_passed': result.tests_passed,
                'tests_total': result.tests_total,
                'test_results': [
                    {
                        'test_id': tr.test_id,
                        'quality': tr.quality,
                        'time_ms': tr.time_ms,
                        'passed': tr.passed
                    }
                    for tr in result.test_results
                ],
                'timestamp': timestamp,
                'date': timestamp.date().isoformat()
            }
            
            await self.benchmarks_collection.insert_one(document)
        
        except Exception as e:
            logger.error(f"Error saving benchmark result: {e}")
    
    async def get_latest_benchmarks(
        self,
        category: str,
        max_age_hours: int = 24
    ) -> List[BenchmarkResult]:
        """
        Get latest benchmark results for a category
        
        Returns benchmarks from last max_age_hours, sorted by score (highest first)
        """
        
        await self.initialize_db()
        
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        try:
            cursor = self.benchmarks_collection.find({
                'category': category,
                'timestamp': {'$gte': cutoff}
            }).sort('final_score', -1)
            
            results = []
            async for doc in cursor:
                # Reconstruct test results
                test_results = []
                for tr in doc.get('test_results', []):
                    test_results.append(BenchmarkTestResult(
                        test_id=tr['test_id'],
                        quality=tr['quality'],
                        time_ms=tr['time_ms'],
                        passed=tr['passed']
                    ))
                
                results.append(BenchmarkResult(
                    id=doc['_id'],
                    provider=doc['provider'],
                    model_name=doc['model_name'],
                    category=doc['category'],
                    quality_score=doc['quality_score'],
                    speed_score=doc['speed_score'],
                    cost_score=doc['cost_score'],
                    final_score=doc['final_score'],
                    avg_response_time_ms=doc['avg_response_time_ms'],
                    avg_cost=doc['avg_cost'],
                    tests_passed=doc['tests_passed'],
                    tests_total=doc['tests_total'],
                    timestamp=doc['timestamp'],
                    test_results=test_results
                ))
            
            return results
        
        except Exception as e:
            logger.error(f"Error fetching benchmarks: {e}")
            return []
    
    async def schedule_benchmarks(self, interval_hours: int = 1):
        """
        Run benchmarks periodically in background
        
        Args:
            interval_hours: How often to run benchmarks (default: 1 hour)
        """
        
        logger.info(f"📅 Benchmark scheduler started (interval: {interval_hours} hours)")
        
        while True:
            try:
                await self.run_benchmarks()
                logger.info(f"⏰ Next benchmark run in {interval_hours} hour(s)")
                await asyncio.sleep(interval_hours * 3600)
            
            except Exception as e:
                logger.error(f"❌ Benchmark scheduler error: {e}", exc_info=True)
                # Wait 5 minutes on error before retrying
                await asyncio.sleep(300)
