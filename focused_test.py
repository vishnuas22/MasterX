#!/usr/bin/env python3
import requests
import json
import time
import uuid
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Get backend URL from frontend .env file
def get_backend_url():
    """Get backend URL from frontend .env file or use local URL"""
    # For testing purposes, we'll use the local URL
    # The preview URL might not be accessible during testing
    return "http://localhost:8001"  # Use local URL for testing

BACKEND_URL = get_backend_url()
API_URL = f"{BACKEND_URL}/api"

print(f"Testing backend at: {API_URL}")

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.response = None
        self.duration = 0
    
    def __str__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        error_msg = f"\n   Error: {self.error}" if self.error else ""
        return f"{status} - {self.name} ({self.duration:.2f}s){error_msg}"

def run_test(test_func):
    """Decorator to run a test function and track results"""
    def wrapper(*args, **kwargs):
        test_name = test_func.__name__.replace('_', ' ').title()
        result = TestResult(test_name)
        
        start_time = time.time()
        try:
            response = test_func(*args, **kwargs)
            result.response = response
            result.passed = True
        except Exception as e:
            result.error = str(e)
        finally:
            result.duration = time.time() - start_time
        
        print(result)
        return result
    
    return wrapper

# Test results tracking
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "results": []
}

@run_test
def test_health_endpoint():
    """Test the health check endpoint"""
    response = requests.get(f"{API_URL}/health")
    response.raise_for_status()
    data = response.json()
    
    assert "status" in data, "Response should contain status"
    assert "database" in data, "Response should contain database status"
    assert data["status"] in ["healthy", "degraded"], f"Status should be healthy or degraded, got {data['status']}"
    
    # Verify database connection
    assert data["database"] == "healthy", f"Database should be healthy, got {data['database']}"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Health Endpoint", "passed": True})
    
    return data

@run_test
def test_create_user():
    """Test user creation"""
    user_data = {
        "email": f"test.user.{uuid.uuid4()}@example.com",
        "name": "Test User",
        "learning_preferences": {}
    }
    
    response = requests.post(f"{API_URL}/users", json=user_data)
    response.raise_for_status()
    data = response.json()
    
    assert "id" in data, "Response should contain user ID"
    assert data["email"] == user_data["email"], "Email should match"
    assert data["name"] == user_data["name"], "Name should match"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "User Creation", "passed": True})
    
    return data

@run_test
def test_create_session(user_id):
    """Test session creation"""
    session_data = {
        "user_id": user_id,
        "subject": "Python Programming",
        "learning_objectives": ["Learn basic syntax", "Understand functions"],
        "difficulty_level": "beginner"
    }
    
    response = requests.post(f"{API_URL}/sessions", json=session_data)
    response.raise_for_status()
    data = response.json()
    
    assert "id" in data, "Response should contain session ID"
    assert data["user_id"] == user_id, "User ID should match"
    assert data["subject"] == session_data["subject"], "Subject should match"
    assert data["is_active"] == True, "Session should be active"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Session Creation", "passed": True})
    
    return data

@run_test
def test_create_learning_goal(user_id: str):
    """Test creating a learning goal"""
    goal_data = {
        "title": "Master Python Programming",
        "description": "Become proficient in Python programming language",
        "goal_type": "skill_mastery",
        "target_date": (datetime.now() + timedelta(days=90)).isoformat(),
        "skills_required": ["Python syntax", "Data structures", "OOP concepts"],
        "success_criteria": ["Complete 3 projects", "Pass assessment with 80%"]
    }
    
    response = requests.post(f"{API_URL}/users/{user_id}/goals", json=goal_data)
    response.raise_for_status()
    data = response.json()
    
    assert "goal_id" in data, "Response should contain goal ID"
    assert data["title"] == goal_data["title"], "Goal title should match"
    assert data["user_id"] == user_id, "User ID should match"
    assert data["status"] == "active", "Goal should be active"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Create Learning Goal", "passed": True})
    
    return data

@run_test
def test_get_user_goals(user_id: str):
    """Test getting user's learning goals"""
    response = requests.get(f"{API_URL}/users/{user_id}/goals")
    response.raise_for_status()
    data = response.json()
    
    assert isinstance(data, list), "Response should be a list of goals"
    if data:
        assert "goal_id" in data[0], "Goal should contain ID"
        assert "user_id" in data[0], "Goal should contain user_id"
        assert data[0]["user_id"] == user_id, "User ID should match"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Get User Goals", "passed": True})
    
    return data

@run_test
def test_update_goal_progress(goal_id: str):
    """Test updating goal progress"""
    progress_data = {
        "progress_delta": 10.0,
        "session_context": {
            "session_duration_minutes": 30
        }
    }
    
    response = requests.put(f"{API_URL}/goals/{goal_id}/progress", json=progress_data)
    response.raise_for_status()
    data = response.json()
    
    assert "goal_id" in data, "Response should contain goal ID"
    assert "progress_percentage" in data, "Response should contain progress percentage"
    assert data["progress_percentage"] >= 10.0, "Progress should be updated"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Update Goal Progress", "passed": True})
    
    return data

@run_test
def test_add_learning_memory(user_id: str, goal_id: str = None):
    """Test adding a learning memory"""
    memory_data = {
        "memory_type": "insight",
        "content": "I realized that Python decorators are a powerful way to modify function behavior",
        "context": {
            "subject": "Python Programming",
            "session_id": str(uuid.uuid4())
        },
        "importance": 0.8,
        "related_goals": [goal_id] if goal_id else [],
        "related_concepts": ["decorators", "metaprogramming"]
    }
    
    response = requests.post(f"{API_URL}/users/{user_id}/memories", json=memory_data)
    response.raise_for_status()
    data = response.json()
    
    assert "memory_id" in data, "Response should contain memory ID"
    assert data["user_id"] == user_id, "User ID should match"
    assert data["content"] == memory_data["content"], "Memory content should match"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Add Learning Memory", "passed": True})
    
    return data

@run_test
def test_get_user_memories(user_id: str):
    """Test getting user's learning memories"""
    response = requests.get(f"{API_URL}/users/{user_id}/memories")
    response.raise_for_status()
    data = response.json()
    
    assert isinstance(data, list), "Response should be a list of memories"
    if data:
        assert "memory_id" in data[0], "Memory should contain ID"
        assert "user_id" in data[0], "Memory should contain user_id"
        assert data[0]["user_id"] == user_id, "User ID should match"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Get User Memories", "passed": True})
    
    return data

@run_test
def test_get_personalized_recommendations(user_id: str):
    """Test getting personalized recommendations"""
    response = requests.get(f"{API_URL}/users/{user_id}/recommendations")
    response.raise_for_status()
    data = response.json()
    
    assert "next_actions" in data, "Response should contain next actions"
    assert "skill_gaps" in data, "Response should contain skill gaps"
    assert "optimization_suggestions" in data, "Response should contain optimization suggestions"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Get Personalized Recommendations", "passed": True})
    
    return data

@run_test
def test_get_learning_insights(user_id: str):
    """Test getting learning insights"""
    response = requests.get(f"{API_URL}/users/{user_id}/insights")
    response.raise_for_status()
    data = response.json()
    
    assert isinstance(data, list), "Response should be a list of insights"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Get Learning Insights", "passed": True})
    
    return data

@run_test
def test_learning_dna_analysis(user_id: str):
    """Test learning DNA analysis endpoint"""
    response = requests.get(f"{API_URL}/users/{user_id}/learning-dna")
    response.raise_for_status()
    data = response.json()
    
    assert "learning_style" in data, "Response should contain learning style"
    assert "cognitive_patterns" in data, "Response should contain cognitive patterns"
    assert "preferred_pace" in data, "Response should contain preferred pace"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Learning DNA Analysis", "passed": True})
    
    return data

@run_test
def test_adaptive_parameters(user_id: str):
    """Test adaptive parameters endpoint"""
    response = requests.get(f"{API_URL}/users/{user_id}/adaptive-parameters")
    response.raise_for_status()
    data = response.json()
    
    assert "complexity_level" in data, "Response should contain complexity level"
    assert "explanation_depth" in data, "Response should contain explanation depth"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Adaptive Parameters", "passed": True})
    
    return data

@run_test
def test_mood_analysis(user_id: str):
    """Test mood analysis endpoint"""
    response = requests.get(f"{API_URL}/users/{user_id}/mood-analysis")
    response.raise_for_status()
    data = response.json()
    
    assert "detected_mood" in data, "Response should contain detected mood"
    assert "recommended_pace" in data, "Response should contain recommended pace"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Mood Analysis", "passed": True})
    
    return data

@run_test
def test_personalization_features():
    """Test personalization features endpoint"""
    response = requests.get(f"{API_URL}/personalization/features")
    response.raise_for_status()
    data = response.json()
    
    assert "features" in data, "Response should contain features"
    
    # Verify key personalization features
    features = data["features"]
    assert "learning_dna_profiling" in features, "Should include learning DNA profiling"
    assert "adaptive_content_generation" in features, "Should include adaptive content generation"
    assert "personal_learning_assistant" in features, "Should include personal learning assistant"
    assert "mood_based_adaptation" in features, "Should include mood-based adaptation"
    assert "real_time_personalization" in features, "Should include real-time personalization"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Personalization Features", "passed": True})
    
    return data

@run_test
def test_available_models():
    """Test getting available AI models"""
    response = requests.get(f"{API_URL}/models/available")
    response.raise_for_status()
    data = response.json()
    
    assert "available_models" in data, "Response should list available models"
    assert "model_capabilities" in data, "Response should include model capabilities"
    assert "deepseek-r1" in data["model_capabilities"], "DeepSeek R1 should be in capabilities"
    
    # Verify Groq API key is working
    assert "deepseek-r1" in data["available_models"] or "deepseek-r1-distill-llama-70b" in data["available_models"], "DeepSeek R1 or fallback model should be available with Groq API key"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Available Models", "passed": True})
    
    return data

def run_personal_learning_assistant_tests():
    """Run tests for Personal Learning Assistant endpoints"""
    print("\n========== TESTING PERSONAL LEARNING ASSISTANT ==========\n")
    
    # Create a user for testing
    user_result = test_create_user()
    if not user_result.passed:
        print("❌ User creation failed, skipping dependent tests")
        return False
    
    user_id = user_result.response["id"]
    
    # Create a session for testing
    session_result = test_create_session(user_id)
    if not session_result.passed:
        print("❌ Session creation failed, skipping dependent tests")
        return False
    
    # Test creating a learning goal
    goal_result = test_create_learning_goal(user_id)
    if not goal_result.passed:
        print("❌ Learning goal creation failed, skipping related tests")
        return False
    
    goal_id = goal_result.response["goal_id"]
    
    # Test getting user goals
    goals_result = test_get_user_goals(user_id)
    if not goals_result.passed:
        print("❌ Getting user goals failed")
    
    # Test updating goal progress
    progress_result = test_update_goal_progress(goal_id)
    if not progress_result.passed:
        print("❌ Goal progress update failed")
    
    # Test adding a learning memory
    memory_result = test_add_learning_memory(user_id, goal_id)
    if not memory_result.passed:
        print("❌ Adding learning memory failed")
    
    # Test getting user memories
    memories_result = test_get_user_memories(user_id)
    if not memories_result.passed:
        print("❌ Getting user memories failed")
    
    # Test getting personalized recommendations
    recommendations_result = test_get_personalized_recommendations(user_id)
    if not recommendations_result.passed:
        print("❌ Getting personalized recommendations failed")
    
    # Test getting learning insights
    insights_result = test_get_learning_insights(user_id)
    if not insights_result.passed:
        print("❌ Getting learning insights failed")
    
    # Calculate success rate
    total_tests = 7  # Number of Personal Learning Assistant tests
    passed_tests = sum(1 for result in [
        goal_result.passed, 
        goals_result.passed, 
        progress_result.passed, 
        memory_result.passed, 
        memories_result.passed, 
        recommendations_result.passed, 
        insights_result.passed
    ] if result)
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nPersonal Learning Assistant Tests: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
    
    return success_rate == 100

def run_personalization_engine_tests():
    """Run tests for Personalization Engine endpoints"""
    print("\n========== TESTING PERSONALIZATION ENGINE ==========\n")
    
    # Create a user for testing
    user_result = test_create_user()
    if not user_result.passed:
        print("❌ User creation failed, skipping dependent tests")
        return False
    
    user_id = user_result.response["id"]
    
    # Test learning DNA analysis
    dna_result = test_learning_dna_analysis(user_id)
    if not dna_result.passed:
        print("❌ Learning DNA analysis failed")
    
    # Test adaptive parameters
    params_result = test_adaptive_parameters(user_id)
    if not params_result.passed:
        print("❌ Adaptive parameters failed")
    
    # Test mood analysis
    mood_result = test_mood_analysis(user_id)
    if not mood_result.passed:
        print("❌ Mood analysis failed")
    
    # Test personalization features
    features_result = test_personalization_features()
    if not features_result.passed:
        print("❌ Personalization features failed")
    
    # Calculate success rate
    total_tests = 4  # Number of Personalization Engine tests
    passed_tests = sum(1 for result in [
        dna_result.passed, 
        params_result.passed, 
        mood_result.passed, 
        features_result.passed
    ] if result)
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nPersonalization Engine Tests: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
    
    return success_rate == 100

def run_groq_api_tests():
    """Run tests for Groq API integration"""
    print("\n========== TESTING GROQ API INTEGRATION ==========\n")
    
    # Test available models
    models_result = test_available_models()
    if not models_result.passed:
        print("❌ Available models test failed")
        return False
    
    # Calculate success rate
    total_tests = 1  # Number of Groq API tests
    passed_tests = sum(1 for result in [models_result.passed] if result)
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nGroq API Tests: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
    
    return success_rate == 100

def run_all_tests():
    """Run all tests"""
    print("\n========== MASTERX AI MENTOR SYSTEM BACKEND TESTS ==========\n")
    
    # Test health endpoint
    health_result = test_health_endpoint()
    if not health_result.passed:
        print("❌ Health check failed, skipping other tests")
        return
    
    # Run Personal Learning Assistant tests
    pla_success = run_personal_learning_assistant_tests()
    
    # Run Personalization Engine tests
    pe_success = run_personalization_engine_tests()
    
    # Run Groq API tests
    groq_success = run_groq_api_tests()
    
    # Print overall summary
    print("\n========== TEST SUMMARY ==========\n")
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ({test_results['passed']/test_results['total']*100:.1f}%)")
    print(f"Failed: {test_results['failed']} ({test_results['failed']/test_results['total']*100:.1f}%)")
    
    # Print component status
    print("\nComponent Status:")
    print(f"Personal Learning Assistant: {'✅ WORKING' if pla_success else '❌ FAILING'}")
    print(f"Personalization Engine: {'✅ WORKING' if pe_success else '❌ FAILING'}")
    print(f"Groq API Integration: {'✅ WORKING' if groq_success else '❌ FAILING'}")
    
    print("\nAll tests completed. See results above for details.")

if __name__ == "__main__":
    run_all_tests()