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
    
    print(f"Create Learning Goal Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Get User Goals Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Update Goal Progress Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Add Learning Memory Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Get User Memories Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Get Personalized Recommendations Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Get Learning Insights Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Learning DNA Analysis Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Adaptive Parameters Response: {json.dumps(data, indent=2)}")
    
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
    # First try GET method
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/mood-analysis")
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 405:  # Method Not Allowed
            # Try POST method instead
            response = requests.post(f"{API_URL}/users/{user_id}/mood-analysis", json={
                "recent_messages": [
                    {"sender": "user", "message": "I'm feeling a bit overwhelmed with all this information", "timestamp": datetime.now().isoformat()},
                    {"sender": "user", "message": "Can you explain this more clearly?", "timestamp": datetime.now().isoformat()}
                ]
            })
            response.raise_for_status()
            data = response.json()
        else:
            raise
    
    print(f"Mood Analysis Response: {json.dumps(data, indent=2)}")
    
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
    
    print(f"Personalization Features Response: {json.dumps(data, indent=2)}")
    
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

def run_personal_learning_tests():
    """Run tests for personal learning assistant"""
    print("\n========== TESTING PERSONAL LEARNING ASSISTANT ==========\n")
    
    # Create a user for testing
    user_result = test_create_user()
    if not user_result.passed:
        print("❌ User creation failed, skipping dependent tests")
        return
    
    user_id = user_result.response["id"]
    
    # Test learning goal creation and management
    goal_result = test_create_learning_goal(user_id)
    if goal_result.passed:
        goal_id = goal_result.response.get("goal_id")
        if goal_id:
            test_get_user_goals(user_id)
            test_update_goal_progress(goal_id)
            test_add_learning_memory(user_id, goal_id)
        else:
            print("❌ Goal ID not found in response, skipping dependent tests")
            test_add_learning_memory(user_id)  # Test without goal_id
    else:
        print("❌ Learning goal creation failed, skipping dependent tests")
        test_add_learning_memory(user_id)  # Test without goal_id
    
    # Test memory and recommendation features
    test_get_user_memories(user_id)
    test_get_personalized_recommendations(user_id)
    test_get_learning_insights(user_id)
    
    # Test personalization engine features
    test_learning_dna_analysis(user_id)
    test_adaptive_parameters(user_id)
    test_mood_analysis(user_id)
    test_personalization_features()
    
    print("\n========== TEST SUMMARY ==========\n")
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ({test_results['passed']/test_results['total']*100:.1f}%)")
    print(f"Failed: {test_results['failed']} ({test_results['failed']/test_results['total']*100:.1f}%)")
    print("\nAll tests completed. See results above for details.")

if __name__ == "__main__":
    run_personal_learning_tests()