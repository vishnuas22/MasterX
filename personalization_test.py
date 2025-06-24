#!/usr/bin/env python3
import requests
import json
import time
import uuid
from datetime import datetime, timedelta

# Get backend URL
BACKEND_URL = "http://localhost:8001"
API_URL = f"{BACKEND_URL}/api"

print(f"Testing personalization endpoints at: {API_URL}")

# Create a test user
def create_test_user():
    user_data = {
        "email": f"test.user.{uuid.uuid4()}@example.com",
        "name": "Test User",
        "learning_preferences": {}
    }
    
    response = requests.post(f"{API_URL}/users", json=user_data)
    response.raise_for_status()
    data = response.json()
    
    print(f"✅ Created test user: {data['id']}")
    return data

# Create a test session
def create_test_session(user_id):
    session_data = {
        "user_id": user_id,
        "subject": "Python Programming",
        "learning_objectives": ["Learn basic syntax", "Understand functions"],
        "difficulty_level": "beginner"
    }
    
    response = requests.post(f"{API_URL}/sessions", json=session_data)
    response.raise_for_status()
    data = response.json()
    
    print(f"✅ Created test session: {data['id']}")
    return data

# Test learning DNA endpoint
def test_learning_dna(user_id):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/learning-dna")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Learning DNA endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Learning DNA endpoint failed: {str(e)}")
        return False

# Test adaptive parameters endpoint
def test_adaptive_parameters(user_id):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/adaptive-parameters")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Adaptive parameters endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Adaptive parameters endpoint failed: {str(e)}")
        return False

# Test mood analysis endpoint
def test_mood_analysis(user_id):
    try:
        mood_data = {
            "message": "I'm feeling a bit overwhelmed with all this new information."
        }
        
        response = requests.post(f"{API_URL}/users/{user_id}/mood-analysis", json=mood_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Mood analysis endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Mood analysis endpoint failed: {str(e)}")
        return False

# Test learning goal creation
def test_create_learning_goal(user_id):
    try:
        goal_data = {
            "title": "Master Python Programming",
            "description": "Learn Python programming from basics to advanced concepts",
            "goal_type": "skill_mastery",
            "target_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "skills_required": ["Programming", "Problem Solving"],
            "success_criteria": ["Complete 5 projects", "Pass assessment with 80%"]
        }
        
        response = requests.post(f"{API_URL}/users/{user_id}/goals", json=goal_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Learning goal creation endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.keys())}")
        return data.get("goal", {}).get("goal_id")
    except Exception as e:
        print(f"❌ Learning goal creation endpoint failed: {str(e)}")
        return None

# Test getting learning goals
def test_get_learning_goals(user_id):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/goals")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Get learning goals endpoint: {response.status_code}")
        print(f"   Goals count: {data.get('total_count', 0)}")
        return True
    except Exception as e:
        print(f"❌ Get learning goals endpoint failed: {str(e)}")
        return False

# Test updating goal progress
def test_update_goal_progress(goal_id):
    try:
        if not goal_id:
            print("❌ Cannot update goal progress: No goal ID provided")
            return False
            
        progress_data = {
            "progress_delta": 10.0,
            "session_context": {
                "session_duration_minutes": 30
            }
        }
        
        response = requests.post(f"{API_URL}/goals/{goal_id}/progress", json=progress_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Update goal progress endpoint: {response.status_code}")
        print(f"   New progress: {data.get('goal', {}).get('progress_percentage')}%")
        return True
    except Exception as e:
        print(f"❌ Update goal progress endpoint failed: {str(e)}")
        return False

# Test personalized recommendations
def test_get_personalized_recommendations(user_id):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/recommendations")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Personalized recommendations endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.get('recommendations', {}).keys())}")
        return True
    except Exception as e:
        print(f"❌ Personalized recommendations endpoint failed: {str(e)}")
        return False

# Test learning insights
def test_get_learning_insights(user_id):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/insights")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Learning insights endpoint: {response.status_code}")
        print(f"   Insights count: {data.get('insights_count', 0)}")
        return True
    except Exception as e:
        print(f"❌ Learning insights endpoint failed: {str(e)}")
        return False

# Test learning memory creation
def test_create_learning_memory(user_id):
    try:
        memory_data = {
            "memory_type": "insight",
            "content": "I realized that breaking down problems helps me understand them better",
            "context": {
                "subject": "Problem Solving",
                "session_id": str(uuid.uuid4())
            },
            "importance": 0.8
        }
        
        response = requests.post(f"{API_URL}/users/{user_id}/memories", json=memory_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Learning memory creation endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Learning memory creation endpoint failed: {str(e)}")
        return False

# Test getting learning memories
def test_get_learning_memories(user_id):
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/memories")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Get learning memories endpoint: {response.status_code}")
        print(f"   Memories count: {data.get('total_count', 0)}")
        return True
    except Exception as e:
        print(f"❌ Get learning memories endpoint failed: {str(e)}")
        return False

# Test adaptive chat
def test_adaptive_chat(session_id):
    try:
        chat_data = {
            "session_id": session_id,
            "user_message": "Can you explain machine learning concepts in a way that matches my learning style?"
        }
        
        response = requests.post(f"{API_URL}/chat/adaptive", json=chat_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Adaptive chat endpoint: {response.status_code}")
        print(f"   Response keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Adaptive chat endpoint failed: {str(e)}")
        return False

# Test personalization features
def test_personalization_features():
    try:
        response = requests.get(f"{API_URL}/personalization/features")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Personalization features endpoint: {response.status_code}")
        print(f"   Features: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Personalization features endpoint failed: {str(e)}")
        return False

def run_tests():
    print("\n========== TESTING PERSONALIZATION ENDPOINTS ==========\n")
    
    # Create test user and session
    user = create_test_user()
    user_id = user["id"]
    session = create_test_session(user_id)
    session_id = session["id"]
    
    # Test personalization features
    test_personalization_features()
    
    # Test learning DNA and adaptive parameters
    test_learning_dna(user_id)
    test_adaptive_parameters(user_id)
    test_mood_analysis(user_id)
    
    # Test personal learning assistant
    goal_id = test_create_learning_goal(user_id)
    test_get_learning_goals(user_id)
    test_update_goal_progress(goal_id)
    test_get_personalized_recommendations(user_id)
    test_get_learning_insights(user_id)
    test_create_learning_memory(user_id)
    test_get_learning_memories(user_id)
    
    # Test adaptive chat
    test_adaptive_chat(session_id)
    
    print("\n========== PERSONALIZATION TESTING COMPLETE ==========\n")

if __name__ == "__main__":
    run_tests()