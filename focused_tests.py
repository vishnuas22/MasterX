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
    
    try:
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
    except Exception as e:
        # Print detailed error for debugging
        print(f"Error in test_create_learning_goal: {str(e)}")
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"Response content: {e.response.content.decode()}")
        
        # Track result
        test_results["total"] += 1
        test_results["failed"] += 1
        test_results["results"].append({"name": "Create Learning Goal", "passed": False})
        
        raise

@run_test
def test_get_user_goals(user_id: str):
    """Test getting user's learning goals"""
    response = requests.get(f"{API_URL}/users/{user_id}/goals")
    response.raise_for_status()
    data = response.json()
    
    assert "goals" in data, "Response should contain goals list"
    assert isinstance(data["goals"], list), "Goals should be a list"
    if data["goals"]:
        assert "goal_id" in data["goals"][0], "Goal should contain ID"
        assert "user_id" in data["goals"][0], "Goal should contain user_id"
        assert data["goals"][0]["user_id"] == user_id, "User ID should match"
    
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
    
    try:
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
    except Exception as e:
        # Print detailed error for debugging
        print(f"Error in test_update_goal_progress: {str(e)}")
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"Response content: {e.response.content.decode()}")
        
        # Track result
        test_results["total"] += 1
        test_results["failed"] += 1
        test_results["results"].append({"name": "Update Goal Progress", "passed": False})
        
        raise

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
    
    assert "memories" in data, "Response should contain memories list"
    assert isinstance(data["memories"], list), "Memories should be a list"
    if data["memories"]:
        assert "memory_id" in data["memories"][0], "Memory should contain ID"
        assert "user_id" in data["memories"][0], "Memory should contain user_id"
        assert data["memories"][0]["user_id"] == user_id, "User ID should match"
    
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
    
    assert "recommendations" in data, "Response should contain recommendations"
    assert "next_actions" in data["recommendations"], "Response should contain next actions"
    assert "skill_gaps" in data["recommendations"], "Response should contain skill gaps"
    assert "optimization_suggestions" in data["recommendations"], "Response should contain optimization suggestions"
    
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
    
    assert "insights" in data, "Response should contain insights list"
    assert isinstance(data["insights"], list), "Insights should be a list"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Get Learning Insights", "passed": True})
    
    return data

@run_test
def test_personalization_engine_endpoints(user_id: str):
    """Test personalization engine endpoints"""
    try:
        # Test learning DNA endpoint
        response = requests.get(f"{API_URL}/users/{user_id}/learning-dna")
        response.raise_for_status()
        dna_data = response.json()
        
        assert "learning_style" in dna_data, "Response should contain learning style"
        assert "cognitive_patterns" in dna_data, "Response should contain cognitive patterns"
        assert "preferred_pace" in dna_data, "Response should contain preferred pace"
        
        # Test adaptive parameters endpoint
        response = requests.get(f"{API_URL}/users/{user_id}/adaptive-parameters")
        response.raise_for_status()
        params_data = response.json()
        
        assert "complexity_level" in params_data, "Response should contain complexity level"
        assert "explanation_depth" in params_data, "Response should contain explanation depth"
        
        # Test mood analysis endpoint
        response = requests.post(f"{API_URL}/users/{user_id}/mood-analysis", json={
            "recent_messages": [
                {"sender": "user", "message": "I'm feeling a bit confused about this topic", "timestamp": datetime.now().isoformat()},
                {"sender": "mentor", "message": "Let me explain it differently", "timestamp": datetime.now().isoformat()}
            ]
        })
        response.raise_for_status()
        mood_data = response.json()
        
        assert "detected_mood" in mood_data, "Response should contain detected mood"
        assert "recommended_pace" in mood_data, "Response should contain recommended pace"
        
        # Track result
        test_results["total"] += 1
        test_results["passed"] += 1
        test_results["results"].append({"name": "Personalization Engine Endpoints", "passed": True})
        
        return {"dna": dna_data, "params": params_data, "mood": mood_data}
    except Exception as e:
        # Print detailed error for debugging
        print(f"Error in test_personalization_engine_endpoints: {str(e)}")
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"Response content: {e.response.content.decode()}")
        
        # Track result
        test_results["total"] += 1
        test_results["failed"] += 1
        test_results["results"].append({"name": "Personalization Engine Endpoints", "passed": False})
        
        raise

@run_test
def test_personalization_features_endpoint():
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
def test_adaptive_ai_chat(session_id: str):
    """Test adaptive AI chat endpoint"""
    chat_data = {
        "session_id": session_id,
        "user_message": "I want to learn about Python classes and inheritance",
        "context": {
            "learning_mode": "adaptive"
        }
    }
    
    response = requests.post(f"{API_URL}/chat/adaptive", json=chat_data)
    response.raise_for_status()
    data = response.json()
    
    assert "response" in data, "Response should contain AI response"
    assert len(data["response"]) > 50, "Response should be substantial"
    assert "response_type" in data, "Response should have a type"
    assert "metadata" in data, "Response should include metadata"
    
    # Track result
    test_results["total"] += 1
    test_results["passed"] += 1
    test_results["results"].append({"name": "Adaptive AI Chat", "passed": True})
    
    return data

def run_personal_learning_assistant_tests():
    """Run focused tests for Personal Learning Assistant"""
    print("\n========== TESTING PERSONAL LEARNING ASSISTANT ==========\n")
    
    # Create a user for testing
    user_result = test_create_user()
    if not user_result.passed:
        print("Failed to create user, skipping Personal Learning Assistant tests")
        return
    
    user_id = user_result.response["id"]
    
    # Test creating a learning goal
    try:
        goal_result = test_create_learning_goal(user_id)
        if goal_result.passed:
            goal_id = goal_result.response.get("goal_id")
            if goal_id:
                print(f"Successfully created goal with ID: {goal_id}")
                
                # Test getting user goals
                goals_result = test_get_user_goals(user_id)
                if goals_result.passed:
                    print("Successfully retrieved user goals")
                else:
                    print("Failed to retrieve user goals")
                
                # Test updating goal progress
                try:
                    progress_result = test_update_goal_progress(goal_id)
                    if progress_result.passed:
                        print("Successfully updated goal progress")
                    else:
                        print("Failed to update goal progress")
                except Exception as e:
                    print(f"Error testing goal progress update: {str(e)}")
            else:
                print("Goal ID not found in response")
        else:
            print("Failed to create learning goal")
            
        # Test adding learning memory
        memory_result = test_add_learning_memory(user_id)
        if memory_result.passed:
            print("Successfully added learning memory")
            
            # Test getting user memories
            memories_result = test_get_user_memories(user_id)
            if memories_result.passed:
                print("Successfully retrieved user memories")
            else:
                print("Failed to retrieve user memories")
        else:
            print("Failed to add learning memory")
        
        # Test getting personalized recommendations
        recommendations_result = test_get_personalized_recommendations(user_id)
        if recommendations_result.passed:
            print("Successfully retrieved personalized recommendations")
        else:
            print("Failed to retrieve personalized recommendations")
        
        # Test getting learning insights
        insights_result = test_get_learning_insights(user_id)
        if insights_result.passed:
            print("Successfully retrieved learning insights")
        else:
            print("Failed to retrieve learning insights")
            
    except Exception as e:
        print(f"Error in Personal Learning Assistant tests: {str(e)}")
    
    print("\nPersonal Learning Assistant tests completed.")

def run_personalization_engine_tests():
    """Run focused tests for Personalization Engine"""
    print("\n========== TESTING PERSONALIZATION ENGINE ==========\n")
    
    # Create a user for testing
    user_result = test_create_user()
    if not user_result.passed:
        print("Failed to create user, skipping Personalization Engine tests")
        return
    
    user_id = user_result.response["id"]
    
    try:
        # Test personalization engine endpoints
        personalization_result = test_personalization_engine_endpoints(user_id)
        if personalization_result.passed:
            print("Successfully tested personalization engine endpoints")
        else:
            print("Failed to test personalization engine endpoints")
        
        # Test personalization features endpoint
        features_result = test_personalization_features_endpoint()
        if features_result.passed:
            print("Successfully tested personalization features endpoint")
        else:
            print("Failed to test personalization features endpoint")
        
        # Test adaptive AI chat
        session_result = test_create_session(user_id)
        if session_result.passed:
            session_id = session_result.response["id"]
            
            chat_result = test_adaptive_ai_chat(session_id)
            if chat_result.passed:
                print("Successfully tested adaptive AI chat")
            else:
                print("Failed to test adaptive AI chat")
        else:
            print("Failed to create session for adaptive AI chat test")
            
    except Exception as e:
        print(f"Error in Personalization Engine tests: {str(e)}")
    
    print("\nPersonalization Engine tests completed.")

if __name__ == "__main__":
    print("\n========== MASTERX AI MENTOR SYSTEM FOCUSED TESTS ==========\n")
    
    # Run Personal Learning Assistant tests
    run_personal_learning_assistant_tests()
    
    # Run Personalization Engine tests
    run_personalization_engine_tests()
    
    # Print test summary
    print("\n========== TEST SUMMARY ==========\n")
    print(f"Total Tests: {test_results['total']}")
    if test_results['total'] > 0:
        print(f"Passed: {test_results['passed']} ({test_results['passed']/test_results['total']*100:.1f}%)")
        print(f"Failed: {test_results['failed']} ({test_results['failed']/test_results['total']*100:.1f}%)")
    else:
        print("No tests were executed.")
    print("\nAll tests completed. See results above for details.")