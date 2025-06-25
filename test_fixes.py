#!/usr/bin/env python3
import requests
import json
import uuid
from datetime import datetime, timedelta

# Get backend URL
BACKEND_URL = "http://localhost:8001"
API_URL = f"{BACKEND_URL}/api"

print(f"Testing backend at: {API_URL}")

def test_personal_learning_assistant():
    """Test Personal Learning Assistant MongoDB ObjectID fixes"""
    print("\n===== Testing Personal Learning Assistant MongoDB ObjectID fixes =====")
    
    # 1. Create a test user
    user_data = {
        "email": f"test.user.{uuid.uuid4()}@example.com",
        "name": "Test User",
        "learning_preferences": {}
    }
    
    print("\n1. Creating test user...")
    response = requests.post(f"{API_URL}/users", json=user_data)
    if response.status_code != 200:
        print(f"❌ Failed to create user: {response.status_code} - {response.text}")
        return
    
    user = response.json()
    user_id = user["id"]
    print(f"✅ User created with ID: {user_id}")
    
    # 2. Create a learning goal for that user
    goal_data = {
        "title": "Master Python Programming",
        "description": "Become proficient in Python programming language",
        "goal_type": "skill_mastery",
        "target_date": (datetime.now() + timedelta(days=90)).isoformat(),
        "skills_required": ["Python syntax", "Data structures", "OOP concepts"],
        "success_criteria": ["Complete 3 projects", "Pass assessment with 80%"]
    }
    
    print("\n2. Creating learning goal...")
    response = requests.post(f"{API_URL}/users/{user_id}/goals", json=goal_data)
    if response.status_code != 200:
        print(f"❌ Failed to create goal: {response.status_code} - {response.text}")
        return
    
    goal_response = response.json()
    print(f"Goal response: {json.dumps(goal_response, indent=2)}")
    
    # Check if goal_id is present in the response
    if "goal" not in goal_response or "goal_id" not in goal_response["goal"]:
        print("❌ goal_id not found in response")
        return
    
    goal_id = goal_response["goal"]["goal_id"]
    print(f"✅ Goal created with ID: {goal_id}")
    
    # 3. Retrieve the goals
    print("\n3. Retrieving goals...")
    response = requests.get(f"{API_URL}/users/{user_id}/goals")
    if response.status_code != 200:
        print(f"❌ Failed to retrieve goals: {response.status_code} - {response.text}")
        return
    
    goals_response = response.json()
    print(f"Goals response: {json.dumps(goals_response, indent=2)}")
    
    # Check if goals are returned and have goal_id
    if "goals" not in goals_response or not goals_response["goals"]:
        print("❌ No goals found in response")
        return
    
    if "goal_id" not in goals_response["goals"][0]:
        print("❌ goal_id not found in retrieved goals")
        return
    
    print("✅ Goals retrieved successfully with goal_id")
    
    # 4. Update goal progress
    progress_data = {
        "progress_delta": 10.0,
        "session_context": {
            "session_duration_minutes": 30
        }
    }
    
    print(f"\n4. Updating goal progress for goal_id: {goal_id}...")
    response = requests.put(f"{API_URL}/goals/{goal_id}/progress", json=progress_data)
    if response.status_code != 200:
        print(f"❌ Failed to update goal progress: {response.status_code} - {response.text}")
        return
    
    progress_response = response.json()
    print(f"Progress update response: {json.dumps(progress_response, indent=2)}")
    
    # Check if goal_id is present in the response
    if "goal" not in progress_response or "goal_id" not in progress_response["goal"]:
        print("❌ goal_id not found in progress update response")
        return
    
    print("✅ Goal progress updated successfully")
    
    # 5. Create learning memory
    memory_data = {
        "memory_type": "insight",
        "content": "I realized that Python decorators are a powerful way to modify function behavior",
        "context": {
            "subject": "Python Programming",
            "session_id": str(uuid.uuid4())
        },
        "importance": 0.8,
        "related_goals": [goal_id],
        "related_concepts": ["decorators", "metaprogramming"]
    }
    
    print("\n5. Creating learning memory...")
    response = requests.post(f"{API_URL}/users/{user_id}/memories", json=memory_data)
    if response.status_code != 200:
        print(f"❌ Failed to create memory: {response.status_code} - {response.text}")
        return
    
    memory_response = response.json()
    print(f"Memory response: {json.dumps(memory_response, indent=2)}")
    
    # Check if memory_id is present in the response
    if "memory" not in memory_response or "memory_id" not in memory_response["memory"]:
        print("❌ memory_id not found in response")
        return
    
    memory_id = memory_response["memory"]["memory_id"]
    print(f"✅ Memory created with ID: {memory_id}")
    
    # 6. Retrieve learning memories
    print("\n6. Retrieving learning memories...")
    response = requests.get(f"{API_URL}/users/{user_id}/memories")
    if response.status_code != 200:
        print(f"❌ Failed to retrieve memories: {response.status_code} - {response.text}")
        return
    
    memories_response = response.json()
    print(f"Memories response: {json.dumps(memories_response, indent=2)}")
    
    # Check if memories are returned and have memory_id
    if "memories" not in memories_response or not memories_response["memories"]:
        print("❌ No memories found in response")
        return
    
    if "memory_id" not in memories_response["memories"][0]:
        print("❌ memory_id not found in retrieved memories")
        return
    
    print("✅ Memories retrieved successfully with memory_id")
    
    print("\n✅ Personal Learning Assistant MongoDB ObjectID fixes test PASSED")

def test_mood_analysis():
    """Test Personalization Engine mood analysis GET method fix"""
    print("\n===== Testing Personalization Engine mood analysis =====")
    
    # Create a test user
    user_data = {
        "email": f"test.user.{uuid.uuid4()}@example.com",
        "name": "Test User",
        "learning_preferences": {}
    }
    
    print("\n1. Creating test user...")
    response = requests.post(f"{API_URL}/users", json=user_data)
    if response.status_code != 200:
        print(f"❌ Failed to create user: {response.status_code} - {response.text}")
        return
    
    user = response.json()
    user_id = user["id"]
    print(f"✅ User created with ID: {user_id}")
    
    # 7. Test GET method for mood analysis
    print("\n7. Testing GET method for mood analysis...")
    response = requests.get(f"{API_URL}/users/{user_id}/mood-analysis")
    if response.status_code != 200:
        print(f"❌ GET mood analysis failed: {response.status_code} - {response.text}")
        return
    
    mood_get_response = response.json()
    print(f"GET mood analysis response: {json.dumps(mood_get_response, indent=2)}")
    
    # Check if mood_analysis is present in the response
    if "mood_analysis" not in mood_get_response:
        print("❌ mood_analysis not found in GET response")
        return
    
    print("✅ GET mood analysis successful")
    
    # 8. Test POST method for mood analysis
    mood_post_data = {
        "session_id": None,
        "context": {
            "current_topic": "Python Programming"
        }
    }
    
    print("\n8. Testing POST method for mood analysis...")
    response = requests.post(f"{API_URL}/users/{user_id}/mood-analysis", json=mood_post_data)
    if response.status_code != 200:
        print(f"❌ POST mood analysis failed: {response.status_code} - {response.text}")
        return
    
    mood_post_response = response.json()
    print(f"POST mood analysis response: {json.dumps(mood_post_response, indent=2)}")
    
    # Check if mood_analysis is present in the response
    if "mood_analysis" not in mood_post_response:
        print("❌ mood_analysis not found in POST response")
        return
    
    print("✅ POST mood analysis successful")
    
    print("\n✅ Personalization Engine mood analysis test PASSED")

def test_personalization_features():
    """Test other Personalization features"""
    print("\n===== Testing other Personalization features =====")
    
    # Create a test user
    user_data = {
        "email": f"test.user.{uuid.uuid4()}@example.com",
        "name": "Test User",
        "learning_preferences": {}
    }
    
    print("\n1. Creating test user...")
    response = requests.post(f"{API_URL}/users", json=user_data)
    if response.status_code != 200:
        print(f"❌ Failed to create user: {response.status_code} - {response.text}")
        return
    
    user = response.json()
    user_id = user["id"]
    print(f"✅ User created with ID: {user_id}")
    
    # 9. Test learning DNA
    print("\n9. Testing learning DNA...")
    response = requests.get(f"{API_URL}/users/{user_id}/learning-dna")
    if response.status_code != 200:
        print(f"❌ Learning DNA request failed: {response.status_code} - {response.text}")
        return
    
    dna_response = response.json()
    print(f"Learning DNA response: {json.dumps(dna_response, indent=2)}")
    
    # Check if learning_style is present in the response
    if "learning_style" not in dna_response:
        print("❌ learning_style not found in response")
        return
    
    print("✅ Learning DNA request successful")
    
    # 10. Test adaptive parameters
    print("\n10. Testing adaptive parameters...")
    response = requests.get(f"{API_URL}/users/{user_id}/adaptive-parameters")
    if response.status_code != 200:
        print(f"❌ Adaptive parameters request failed: {response.status_code} - {response.text}")
        return
    
    params_response = response.json()
    print(f"Adaptive parameters response: {json.dumps(params_response, indent=2)}")
    
    # Check if complexity_level is present in the response
    if "complexity_level" not in params_response:
        print("❌ complexity_level not found in response")
        return
    
    print("✅ Adaptive parameters request successful")
    
    print("\n✅ Other Personalization features test PASSED")

if __name__ == "__main__":
    print("Starting tests for MasterX AI Mentor System fixes...")
    
    # Test Personal Learning Assistant MongoDB ObjectID fixes
    test_personal_learning_assistant()
    
    # Test Personalization Engine mood analysis
    test_mood_analysis()
    
    # Test other Personalization features
    test_personalization_features()
    
    print("\nAll tests completed!")