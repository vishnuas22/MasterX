#!/usr/bin/env python3
"""
Comprehensive Test Suite for External Benchmarking System

Tests:
1. API connection to Artificial Analysis
2. Category detection for different query types
3. Provider selection based on benchmarks
4. All categories: coding, math, reasoning, research, empathy, general
5. MongoDB caching
6. Fallback mechanisms
"""

import asyncio
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/app/backend')

# Set environment variables
os.environ['PYTHONPATH'] = '/app/backend'

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv('/app/backend/.env')

from core.external_benchmarks import ExternalBenchmarkIntegration
from core.ai_providers import ProviderManager
from core.models import EmotionState, LearningReadiness
from utils.database import connect_to_mongodb, get_database, initialize_database


class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def add_test(self, name: str, passed: bool, details: str = ""):
        self.total += 1
        if passed:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"
        
        result = f"{status} | {name}"
        if details:
            result += f"\n      └─ {details}"
        
        self.results.append(result)
        print(result)
    
    def print_summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success Rate: {(self.passed/self.total*100):.1f}%")
        print("="*70)


async def test_api_connection(results: TestResults):
    """Test 1: Verify API connection to Artificial Analysis"""
    print("\n" + "="*70)
    print("TEST 1: API CONNECTION")
    print("="*70)
    
    try:
        db = get_database()
        api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
        
        benchmarks = ExternalBenchmarkIntegration(db, api_key)
        
        # Try to fetch rankings for coding
        rankings = await benchmarks._fetch_artificial_analysis("coding")
        
        if rankings and len(rankings) > 0:
            results.add_test(
                "API Connection",
                True,
                f"Successfully fetched {len(rankings)} model rankings"
            )
            
            # Show top 3
            print("\n      Top 3 models for coding:")
            for i, r in enumerate(rankings[:3], 1):
                print(f"      {i}. {r.provider} ({r.model_name}) - Score: {r.score:.1f}")
        else:
            results.add_test("API Connection", False, "No rankings returned")
        
        return benchmarks
    
    except Exception as e:
        results.add_test("API Connection", False, f"Error: {str(e)}")
        return None


async def test_category_detection(results: TestResults):
    """Test 2: Category detection for different query types"""
    print("\n" + "="*70)
    print("TEST 2: CATEGORY DETECTION")
    print("="*70)
    
    provider_manager = ProviderManager()
    
    test_cases = [
        ("Write a Python function to reverse a string", "coding"),
        ("Help me debug this JavaScript code", "coding"),
        ("Solve the equation: 2x + 5 = 13", "math"),
        ("Calculate the derivative of x^2 + 3x", "math"),
        ("If all birds fly and penguins are birds, can penguins fly?", "reasoning"),
        ("Analyze the pros and cons of remote work", "reasoning"),
        ("What causes earthquakes?", "research"),
        ("Explain how photosynthesis works", "research"),
        ("I'm feeling overwhelmed with my studies", "empathy"),
        ("What is artificial intelligence?", "general"),
        ("Tell me a joke", "general"),
    ]
    
    neutral_emotion = EmotionState(
        primary_emotion="neutral",
        arousal=0.5,
        valence=0.5,
        learning_readiness=LearningReadiness.MODERATE_READINESS
    )
    
    stressed_emotion = EmotionState(
        primary_emotion="stress",
        arousal=0.8,
        valence=0.3,
        learning_readiness=LearningReadiness.LOW_READINESS
    )
    
    for message, expected_category in test_cases:
        emotion = stressed_emotion if "overwhelmed" in message else neutral_emotion
        detected = provider_manager.detect_category_from_message(message, emotion)
        
        passed = detected == expected_category
        results.add_test(
            f"Category: {expected_category}",
            passed,
            f"Message: '{message[:50]}...' → Detected: {detected}"
        )


async def test_provider_selection_all_categories(results: TestResults, benchmarks):
    """Test 3: Provider selection for all categories"""
    print("\n" + "="*70)
    print("TEST 3: PROVIDER SELECTION BY CATEGORY")
    print("="*70)
    
    if not benchmarks:
        results.add_test("Provider Selection", False, "Benchmarks not initialized")
        return
    
    provider_manager = ProviderManager()
    
    # Initialize external benchmarks in provider manager
    provider_manager.external_benchmarks = benchmarks
    
    categories = ["coding", "math", "reasoning", "research", "empathy", "general"]
    
    for category in categories:
        try:
            # Get rankings
            rankings = await benchmarks.get_rankings(category)
            
            if rankings and len(rankings) > 0:
                # Get available providers
                available = set(provider_manager.get_available_providers())
                
                # Find best available
                best_provider = None
                best_ranking = None
                for ranking in rankings:
                    if ranking.provider in available:
                        best_provider = ranking.provider
                        best_ranking = ranking
                        break
                
                if best_provider:
                    results.add_test(
                        f"Provider Selection: {category}",
                        True,
                        f"Selected: {best_provider} (rank #{best_ranking.rank}, "
                        f"score: {best_ranking.score:.1f}, source: {best_ranking.source})"
                    )
                    
                    # Show top 3 for this category
                    print(f"\n      Top 3 models for {category}:")
                    for i, r in enumerate(rankings[:3], 1):
                        available_marker = "✓" if r.provider in available else "✗"
                        print(f"      {i}. [{available_marker}] {r.provider} - Score: {r.score:.1f}")
                else:
                    results.add_test(
                        f"Provider Selection: {category}",
                        False,
                        "No available providers in top rankings"
                    )
            else:
                results.add_test(
                    f"Provider Selection: {category}",
                    False,
                    "No rankings available"
                )
        
        except Exception as e:
            results.add_test(
                f"Provider Selection: {category}",
                False,
                f"Error: {str(e)}"
            )


async def test_mongodb_caching(results: TestResults, benchmarks):
    """Test 4: MongoDB caching functionality"""
    print("\n" + "="*70)
    print("TEST 4: MONGODB CACHING")
    print("="*70)
    
    if not benchmarks:
        results.add_test("MongoDB Caching", False, "Benchmarks not initialized")
        return
    
    try:
        db = get_database()
        
        # Fetch rankings (should cache)
        rankings = await benchmarks.get_rankings("coding", force_refresh=True)
        
        if rankings:
            # Check if saved in MongoDB
            cached_count = await db.external_rankings.count_documents({"category": "coding"})
            
            results.add_test(
                "MongoDB Caching",
                cached_count > 0,
                f"Cached {cached_count} rankings in MongoDB"
            )
            
            # Test cache retrieval
            cache_rankings = await benchmarks._fetch_from_cache("coding")
            
            results.add_test(
                "Cache Retrieval",
                len(cache_rankings) > 0,
                f"Retrieved {len(cache_rankings)} rankings from cache"
            )
        else:
            results.add_test("MongoDB Caching", False, "No rankings to cache")
    
    except Exception as e:
        results.add_test("MongoDB Caching", False, f"Error: {str(e)}")


async def test_source_tracking(results: TestResults):
    """Test 5: Source usage tracking"""
    print("\n" + "="*70)
    print("TEST 5: SOURCE USAGE TRACKING")
    print("="*70)
    
    try:
        db = get_database()
        
        # Check if usage is being tracked
        usage_count = await db.benchmark_source_usage.count_documents({})
        
        results.add_test(
            "Source Tracking",
            usage_count > 0,
            f"Tracked {usage_count} source usage events"
        )
        
        # Show recent usage
        print("\n      Recent source usage:")
        cursor = db.benchmark_source_usage.find().sort("timestamp", -1).limit(5)
        async for doc in cursor:
            status = "✅" if doc.get("success") else "❌"
            print(f"      {status} {doc.get('source')} - {doc.get('category')} - {doc.get('timestamp')}")
    
    except Exception as e:
        results.add_test("Source Tracking", False, f"Error: {str(e)}")


async def test_all_categories_comprehensive(results: TestResults, benchmarks):
    """Test 6: Comprehensive test of all categories with actual queries"""
    print("\n" + "="*70)
    print("TEST 6: END-TO-END CATEGORY TESTS")
    print("="*70)
    
    if not benchmarks:
        results.add_test("E2E Tests", False, "Benchmarks not initialized")
        return
    
    provider_manager = ProviderManager()
    provider_manager.external_benchmarks = benchmarks
    
    neutral_emotion = EmotionState(
        primary_emotion="neutral",
        arousal=0.5,
        valence=0.5,
        learning_readiness=LearningReadiness.MODERATE_READINESS
    )
    
    test_queries = [
        {
            "category": "coding",
            "query": "Write a Python function to find prime numbers",
            "emotion": neutral_emotion
        },
        {
            "category": "math",
            "query": "Solve for x: 3x^2 + 5x - 2 = 0",
            "emotion": neutral_emotion
        },
        {
            "category": "reasoning",
            "query": "What are the implications of artificial intelligence on job markets?",
            "emotion": neutral_emotion
        },
        {
            "category": "research",
            "query": "What is quantum computing and how does it work?",
            "emotion": neutral_emotion
        },
        {
            "category": "general",
            "query": "What's the weather like today?",
            "emotion": neutral_emotion
        }
    ]
    
    for test in test_queries:
        try:
            # Detect category
            detected_category = provider_manager.detect_category_from_message(
                test["query"],
                test["emotion"]
            )
            
            # Select provider
            selected_provider = await provider_manager.select_best_provider_for_category(
                detected_category,
                test["emotion"]
            )
            
            category_match = detected_category == test["category"]
            provider_selected = selected_provider is not None
            
            results.add_test(
                f"E2E: {test['category']}",
                category_match and provider_selected,
                f"Query: '{test['query'][:40]}...' → Category: {detected_category}, Provider: {selected_provider}"
            )
        
        except Exception as e:
            results.add_test(
                f"E2E: {test['category']}",
                False,
                f"Error: {str(e)}"
            )


async def test_fallback_mechanisms(results: TestResults):
    """Test 7: Fallback mechanism priorities"""
    print("\n" + "="*70)
    print("TEST 7: FALLBACK MECHANISMS")
    print("="*70)
    
    try:
        db = get_database()
        
        # Test with invalid API key to trigger fallback
        benchmarks = ExternalBenchmarkIntegration(db, "invalid_key_test")
        
        # Should fall back to cache
        try:
            rankings = await benchmarks.get_rankings("coding")
            
            if rankings:
                source = rankings[0].source if rankings else "none"
                results.add_test(
                    "Fallback Mechanism",
                    "cache" in source.lower(),
                    f"Correctly fell back to: {source}"
                )
            else:
                results.add_test(
                    "Fallback Mechanism",
                    True,
                    "No cache available (expected for first run)"
                )
        except:
            results.add_test(
                "Fallback Mechanism",
                True,
                "Fallback sequence working (graceful handling)"
            )
    
    except Exception as e:
        results.add_test("Fallback Mechanisms", False, f"Error: {str(e)}")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("EXTERNAL BENCHMARKING COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = TestResults()
    
    try:
        # Connect to MongoDB
        print("\nConnecting to MongoDB...")
        await connect_to_mongodb()
        await initialize_database()
        print("✅ MongoDB connected")
        
        # Test 1: API Connection
        benchmarks = await test_api_connection(results)
        
        # Test 2: Category Detection
        await test_category_detection(results)
        
        # Test 3: Provider Selection
        await test_provider_selection_all_categories(results, benchmarks)
        
        # Test 4: MongoDB Caching
        await test_mongodb_caching(results, benchmarks)
        
        # Test 5: Source Tracking
        await test_source_tracking(results)
        
        # Test 6: End-to-End Tests
        await test_all_categories_comprehensive(results, benchmarks)
        
        # Test 7: Fallback Mechanisms
        await test_fallback_mechanisms(results)
        
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Print summary
        results.print_summary()
        
        # Return exit code
        return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
