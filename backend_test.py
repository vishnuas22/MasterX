#!/usr/bin/env python3
import requests
import json
import uuid
import time
import sseclient
from typing import Dict, Any, List, Optional

# Configuration
BACKEND_URL = "http://localhost:8001/api"
TEST_USER_EMAIL = "test@masterx.ai"
TEST_USER_NAME = "Test User"
TEST_SESSION_SUBJECT = "Mathematics"
TEST_SESSION_DIFFICULTY = "intermediate"

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "tests": []
}

def log_test(name: str, passed: bool, response: Optional[requests.Response] = None, error: Optional[str] = None):
    """Log test results"""
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {name}")
    
    result = {
        "name": name,
        "passed": passed,
        "timestamp": time.time()
    }
    
    if response:
        try:
            result["status_code"] = response.status_code
            result["response"] = response.json() if response.headers.get('content-type') == 'application/json' else response.text[:200]
            print(f"  Status Code: {response.status_code}")
            print(f"  Response: {result['response']}")
        except Exception as e:
            result["response"] = "Could not parse response"
            print(f"  Could not parse response: {str(e)}")
            print(f"  Raw response: {response.text[:200]}")
    
    if error:
        result["error"] = error
        print(f"  Error: {error}")
    
    test_results["tests"].append(result)
    
    if passed:
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1

def test_health_check():
    """Test the health check endpoint"""
    try:
        print(f"Testing health check endpoint: {BACKEND_URL}/health")
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        print(f"Response received: Status code {response.status_code}")
        passed = response.status_code == 200 and response.json().get("status") in ["healthy", "degraded"]
        log_test("Health Check Endpoint", passed, response)
        return response.json() if passed else None
    except requests.exceptions.RequestException as e:
        log_test("Health Check Endpoint", False, error=f"Request exception: {str(e)}")
        return None
    except Exception as e:
        log_test("Health Check Endpoint", False, error=str(e))
        return None

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/")
        passed = response.status_code == 200 and "message" in response.json()
        log_test("Root Endpoint", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Root Endpoint", False, error=str(e))
        return None

def create_test_user():
    """Create a test user or get existing one"""
    try:
        # First try to get the user by email
        response = requests.get(f"{BACKEND_URL}/users/email/{TEST_USER_EMAIL}")
        
        if response.status_code == 200:
            log_test("Get User by Email", True, response)
            return response.json()
        
        # If user doesn't exist, create one
        user_data = {
            "email": TEST_USER_EMAIL,
            "name": TEST_USER_NAME,
            "learning_preferences": {
                "preferred_pace": "moderate",
                "learning_style": "visual"
            }
        }
        
        response = requests.post(f"{BACKEND_URL}/users", json=user_data)
        passed = response.status_code == 200 and "id" in response.json()
        log_test("Create User", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Create/Get User", False, error=str(e))
        return None

def test_get_user_by_email():
    """Test getting a user by email"""
    try:
        response = requests.get(f"{BACKEND_URL}/users/email/{TEST_USER_EMAIL}")
        passed = response.status_code == 200 and response.json().get("email") == TEST_USER_EMAIL
        log_test("Get User by Email", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Get User by Email", False, error=str(e))
        return None

def create_test_session(user_id: str):
    """Create a test session"""
    try:
        session_data = {
            "user_id": user_id,
            "subject": TEST_SESSION_SUBJECT,
            "learning_objectives": ["Understand quadratic equations", "Master factorization"],
            "difficulty_level": TEST_SESSION_DIFFICULTY
        }
        
        response = requests.post(f"{BACKEND_URL}/sessions", json=session_data)
        passed = response.status_code == 200 and "id" in response.json()
        log_test("Create Session", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Create Session", False, error=str(e))
        return None

def test_get_user_sessions(user_id: str):
    """Test getting user sessions"""
    try:
        response = requests.get(f"{BACKEND_URL}/users/{user_id}/sessions")
        passed = response.status_code == 200 and isinstance(response.json(), list)
        log_test("Get User Sessions", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Get User Sessions", False, error=str(e))
        return None

def test_basic_chat(session_id: str):
    """Test basic chat functionality"""
    try:
        chat_data = {
            "session_id": session_id,
            "user_message": "Can you explain what a quadratic equation is?"
        }
        
        response = requests.post(f"{BACKEND_URL}/chat", json=chat_data)
        passed = response.status_code == 200 and "response" in response.json()
        log_test("Basic Chat", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Basic Chat", False, error=str(e))
        return None

def test_premium_chat(session_id: str):
    """Test premium chat functionality"""
    try:
        chat_data = {
            "session_id": session_id,
            "user_message": "Can you explain the quadratic formula in detail?",
            "context": {
                "learning_mode": "adaptive"
            }
        }
        
        response = requests.post(f"{BACKEND_URL}/chat/premium", json=chat_data)
        passed = response.status_code == 200 and "response" in response.json()
        log_test("Premium Chat", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Premium Chat", False, error=str(e))
        return None

def test_available_models():
    """Test getting available AI models"""
    try:
        response = requests.get(f"{BACKEND_URL}/models/available")
        passed = response.status_code == 200 and "available_models" in response.json()
        log_test("Available Models", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Available Models", False, error=str(e))
        return None

def test_analytics_comprehensive_dashboard(user_id: str):
    """Test comprehensive analytics dashboard"""
    try:
        response = requests.get(f"{BACKEND_URL}/analytics/{user_id}/comprehensive-dashboard")
        passed = response.status_code == 200 and "knowledge_graph" in response.json()
        log_test("Comprehensive Analytics Dashboard", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Comprehensive Analytics Dashboard", False, error=str(e))
        return None

def test_analytics_knowledge_graph(user_id: str):
    """Test knowledge graph analytics"""
    try:
        response = requests.get(f"{BACKEND_URL}/analytics/{user_id}/knowledge-graph")
        passed = response.status_code == 200
        log_test("Knowledge Graph Analytics", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Knowledge Graph Analytics", False, error=str(e))
        return None

def test_analytics_competency_heatmap(user_id: str):
    """Test competency heatmap analytics"""
    try:
        response = requests.get(f"{BACKEND_URL}/analytics/{user_id}/competency-heatmap")
        passed = response.status_code == 200
        log_test("Competency Heatmap Analytics", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Competency Heatmap Analytics", False, error=str(e))
        return None

def test_analytics_learning_velocity(user_id: str):
    """Test learning velocity analytics"""
    try:
        response = requests.get(f"{BACKEND_URL}/analytics/{user_id}/learning-velocity")
        passed = response.status_code == 200
        log_test("Learning Velocity Analytics", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Learning Velocity Analytics", False, error=str(e))
        return None

def test_analytics_retention_curves(user_id: str):
    """Test retention curves analytics"""
    try:
        response = requests.get(f"{BACKEND_URL}/analytics/{user_id}/retention-curves")
        passed = response.status_code == 200
        log_test("Retention Curves Analytics", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Retention Curves Analytics", False, error=str(e))
        return None

def test_premium_chat_stream(session_id: str):
    """Test premium streaming chat functionality"""
    try:
        chat_data = {
            "session_id": session_id,
            "user_message": "Explain the concept of derivatives in calculus with examples",
            "context": {
                "learning_mode": "adaptive"
            }
        }
        
        # For streaming endpoints, we need to handle SSE (Server-Sent Events)
        # We'll just check if the connection is established and some data is received
        response = requests.post(f"{BACKEND_URL}/chat/premium/stream", json=chat_data, stream=True)
        
        if response.status_code != 200:
            log_test("Premium Chat Streaming", False, response, "Non-200 status code")
            return None
            
        # Check if we receive some data (at least one chunk)
        received_data = False
        try:
            # Read just a bit of data to confirm streaming works
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    received_data = True
                    break
                    
            response.close()  # Close the connection after testing
            passed = received_data
            log_test("Premium Chat Streaming", passed, response)
            return {"streaming": "working"} if passed else None
            
        except Exception as e:
            log_test("Premium Chat Streaming", False, error=f"Streaming error: {str(e)}")
            return None
            
    except Exception as e:
        log_test("Premium Chat Streaming", False, error=str(e))
        return None

def test_context_awareness(user_id: str, session_id: str):
    """Test context awareness functionality"""
    try:
        context_data = {
            "user_id": user_id,
            "session_id": session_id,
            "message": "I'm having trouble understanding this concept. Can you explain it differently?",
            "conversation_context": [
                {"role": "user", "content": "What is a derivative?"},
                {"role": "assistant", "content": "A derivative measures the rate of change of a function with respect to one of its variables."}
            ]
        }
        
        response = requests.post(f"{BACKEND_URL}/context/analyze", json=context_data)
        passed = response.status_code == 200 and "context_state" in response.json()
        log_test("Context Awareness", passed, response)
        return response.json() if passed else None
    except Exception as e:
        log_test("Context Awareness", False, error=str(e))
        return None

def run_all_tests():
    """Run all tests in sequence"""
    print("\n===== MASTERX BACKEND API TESTING =====\n")
    
    # 1. Health Check & Basic Connectivity
    test_health_check()
    test_root_endpoint()
    
    # 2. User & Session Management
    user = create_test_user()
    if not user:
        print("Failed to create/get test user. Aborting further tests.")
        return
    
    user_id = user.get("id")
    test_get_user_by_email()
    
    session = create_test_session(user_id)
    if not session:
        print("Failed to create test session. Aborting further tests.")
        return
    
    session_id = session.get("id")
    test_get_user_sessions(user_id)
    
    # 3. Chat & AI Functionality
    test_basic_chat(session_id)
    test_premium_chat(session_id)
    test_premium_chat_stream(session_id)  # Test streaming functionality
    test_available_models()
    
    # 4. Advanced Analytics Endpoints
    test_analytics_comprehensive_dashboard(user_id)
    test_analytics_knowledge_graph(user_id)
    test_analytics_competency_heatmap(user_id)
    test_analytics_learning_velocity(user_id)
    test_analytics_retention_curves(user_id)
    
    # 5. Context Awareness
    test_context_awareness(user_id, session_id)
    
    # Print summary
    print("\n===== TEST SUMMARY =====")
    print(f"Total Tests: {test_results['passed'] + test_results['failed']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print("========================\n")

if __name__ == "__main__":
    run_all_tests()