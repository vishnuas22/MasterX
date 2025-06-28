#!/usr/bin/env python3
import requests
import json
import time
import uuid
from datetime import datetime

# Get backend URL
BACKEND_URL = "http://localhost:8001"
API_URL = f"{BACKEND_URL}/api"

print(f"Testing advanced analytics endpoints at: {API_URL}")

def test_record_learning_event():
    """Test recording a learning event"""
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    
    event_data = {
        "user_id": user_id,
        "concept_id": "programming_basics",
        "event_type": "explanation",
        "duration_seconds": 300,
        "performance_score": 0.8,
        "confidence_level": 0.7,
        "session_id": session_id,
        "context": {
            "topic": "Python Programming",
            "difficulty": "beginner"
        }
    }
    
    response = requests.post(f"{API_URL}/analytics/learning-event", json=event_data)
    if response.status_code != 200:
        print(f"❌ FAILED - Record Learning Event: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Record Learning Event")
    print(f"Response: {data}")
    return user_id, session_id

def test_comprehensive_dashboard(user_id):
    """Test comprehensive analytics dashboard"""
    response = requests.get(f"{API_URL}/analytics/{user_id}/comprehensive-dashboard")
    if response.status_code != 200:
        print(f"❌ FAILED - Comprehensive Dashboard: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Comprehensive Dashboard")
    print(f"Response contains: {', '.join(data.keys())}")
    
    # Verify required fields
    required_fields = [
        "knowledge_graph", 
        "competency_heat_map", 
        "learning_velocity", 
        "retention_curves", 
        "learning_path_optimization", 
        "summary"
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        print(f"❌ Missing fields in response: {', '.join(missing_fields)}")
        return False
    
    return True

def test_knowledge_graph(user_id):
    """Test knowledge graph endpoint"""
    response = requests.get(f"{API_URL}/analytics/{user_id}/knowledge-graph")
    if response.status_code != 200:
        print(f"❌ FAILED - Knowledge Graph: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Knowledge Graph")
    print(f"Response contains: {', '.join(data.keys())}")
    return True

def test_competency_heatmap(user_id):
    """Test competency heatmap endpoint"""
    response = requests.get(f"{API_URL}/analytics/{user_id}/competency-heatmap")
    if response.status_code != 200:
        print(f"❌ FAILED - Competency Heatmap: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Competency Heatmap")
    print(f"Response contains: {', '.join(data.keys())}")
    return True

def test_learning_velocity(user_id):
    """Test learning velocity endpoint"""
    response = requests.get(f"{API_URL}/analytics/{user_id}/learning-velocity")
    if response.status_code != 200:
        print(f"❌ FAILED - Learning Velocity: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Learning Velocity")
    print(f"Response contains: {', '.join(data.keys())}")
    return True

def test_retention_curves(user_id):
    """Test retention curves endpoint"""
    response = requests.get(f"{API_URL}/analytics/{user_id}/retention-curves")
    if response.status_code != 200:
        print(f"❌ FAILED - Retention Curves: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Retention Curves")
    print(f"Response contains: {', '.join(data.keys())}")
    return True

def test_learning_path(user_id):
    """Test learning path endpoint"""
    response = requests.get(f"{API_URL}/analytics/{user_id}/learning-path")
    if response.status_code != 200:
        print(f"❌ FAILED - Learning Path: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    data = response.json()
    print(f"✅ PASSED - Learning Path")
    print(f"Response contains: {', '.join(data.keys())}")
    return True

def test_multiple_events(user_id, session_id):
    """Test recording multiple events"""
    event_types = ["question", "explanation", "practice", "assessment"]
    
    for event_type in event_types:
        event_data = {
            "user_id": user_id,
            "concept_id": "programming_basics",
            "event_type": event_type,
            "duration_seconds": 300,
            "performance_score": 0.8,
            "confidence_level": 0.7,
            "session_id": session_id,
            "context": {
                "topic": "Python Programming",
                "difficulty": "beginner",
                "event_type_specific": event_type
            }
        }
        
        response = requests.post(f"{API_URL}/analytics/learning-event", json=event_data)
        if response.status_code != 200:
            print(f"❌ FAILED - Record {event_type} Event: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    print(f"✅ PASSED - Record Multiple Event Types")
    return True

def test_multiple_concepts(user_id, session_id):
    """Test recording events for multiple concepts"""
    concepts = ["programming_basics", "data_structures", "algorithms"]
    
    for concept in concepts:
        event_data = {
            "user_id": user_id,
            "concept_id": concept,
            "event_type": "explanation",
            "duration_seconds": 300,
            "performance_score": 0.8,
            "confidence_level": 0.7,
            "session_id": session_id,
            "context": {
                "topic": "Python Programming",
                "difficulty": "beginner",
                "concept_specific": concept
            }
        }
        
        response = requests.post(f"{API_URL}/analytics/learning-event", json=event_data)
        if response.status_code != 200:
            print(f"❌ FAILED - Record {concept} Event: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    print(f"✅ PASSED - Record Multiple Concepts")
    return True

def run_all_tests():
    """Run all tests for advanced analytics endpoints"""
    print("\n========== TESTING ADVANCED ANALYTICS ENDPOINTS ==========\n")
    
    # Test recording learning events
    result = test_record_learning_event()
    if not result:
        print("❌ Learning event recording failed, skipping dependent tests")
        return
    
    user_id, session_id = result
    
    # Test recording multiple events
    test_multiple_events(user_id, session_id)
    
    # Test recording events for multiple concepts
    test_multiple_concepts(user_id, session_id)
    
    # Test analytics endpoints
    test_knowledge_graph(user_id)
    test_competency_heatmap(user_id)
    test_learning_velocity(user_id)
    test_retention_curves(user_id)
    test_learning_path(user_id)
    
    # Test comprehensive dashboard
    test_comprehensive_dashboard(user_id)
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    run_all_tests()