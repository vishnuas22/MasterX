#!/usr/bin/env python3
import requests
import json
import uuid
import time
from datetime import datetime

# Use the local backend URL for testing
BACKEND_URL = "http://localhost:8001"

# Add /api prefix for all API calls
API_URL = f"{BACKEND_URL}/api"

print(f"Using API URL: {API_URL}")

# Helper functions
def print_separator(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_response(response, label="Response"):
    print(f"\n{label}:")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response Text: {response.text}")
    print()

# Test functions
def test_health_check():
    print_separator("Testing Health Check Endpoint")
    response = requests.get(f"{API_URL}/health")
    print_response(response)
    assert response.status_code == 200, "Health check failed"
    return response.json()

def create_test_user():
    print_separator("Creating Test User")
    test_user = {
        "email": f"test_user_{uuid.uuid4()}@example.com",
        "name": "Test User",
        "learning_preferences": {}
    }
    response = requests.post(f"{API_URL}/users", json=test_user)
    print_response(response)
    
    if response.status_code == 400 and "already exists" in response.json().get("detail", ""):
        print("User already exists, fetching by email...")
        response = requests.get(f"{API_URL}/users/email/{test_user['email']}")
        print_response(response)
    
    assert response.status_code in [200, 201, 400], "Failed to create or get user"
    return response.json()

def create_test_session(user_id):
    print_separator("Creating Test Session")
    session_data = {
        "user_id": user_id,
        "subject": "AR/VR Testing",
        "learning_objectives": ["Test AR/VR endpoints", "Test gesture control endpoints"]
    }
    response = requests.post(f"{API_URL}/sessions", json=session_data)
    print_response(response)
    assert response.status_code in [200, 201], "Failed to create session"
    return response.json()

def test_arvr_settings(user_id):
    print_separator("Testing AR/VR Settings Endpoints")
    
    # Test GET AR/VR settings
    print("Testing GET /api/users/{user_id}/arvr-settings")
    get_response = requests.get(f"{API_URL}/users/{user_id}/arvr-settings")
    print_response(get_response, "GET Response")
    
    # Test POST AR/VR settings
    print("Testing POST /api/users/{user_id}/arvr-settings")
    arvr_settings = {
        "settings": {
            "vr_enabled": True,
            "ar_enabled": True,
            "3d_mode_enabled": True,
            "render_quality": "ultra",
            "enable_physics": True,
            "enable_shadows": True,
            "enable_lighting": True,
            "fov": 85,
            "auto_rotate": True,
            "rotation_speed": 1.5,
            "zoom_level": 1.2,
            "background_color": "#001122"
        }
    }
    post_response = requests.post(f"{API_URL}/users/{user_id}/arvr-settings", json=arvr_settings)
    print_response(post_response, "POST Response")
    
    # Verify settings were updated by getting them again
    verify_response = requests.get(f"{API_URL}/users/{user_id}/arvr-settings")
    print_response(verify_response, "Verification Response")
    
    return {
        "get_successful": get_response.status_code == 200,
        "post_successful": post_response.status_code == 200,
        "settings_updated": verify_response.status_code == 200 and 
                           verify_response.json().get("arvr_settings", {}).get("vr_enabled") == True
    }

def test_gesture_settings(user_id):
    print_separator("Testing Gesture Control Settings Endpoints")
    
    # Test GET gesture settings
    print("Testing GET /api/users/{user_id}/gesture-settings")
    get_response = requests.get(f"{API_URL}/users/{user_id}/gesture-settings")
    print_response(get_response, "GET Response")
    
    # Test POST gesture settings
    print("Testing POST /api/users/{user_id}/gesture-settings")
    gesture_settings = {
        "settings": {
            "enabled": True,
            "sensitivity": 0.9,
            "gesture_timeout": 1500,
            "enabled_gestures": {
                "scroll": True,
                "navigate": True,
                "voice": True,
                "speed": True,
                "volume": True
            },
            "custom_gestures": [
                {
                    "name": "page_flip",
                    "description": "Flip to next page",
                    "motion_sequence": ["right_swipe"]
                }
            ],
            "camera_permission": "granted"
        }
    }
    post_response = requests.post(f"{API_URL}/users/{user_id}/gesture-settings", json=gesture_settings)
    print_response(post_response, "POST Response")
    
    # Verify settings were updated by getting them again
    verify_response = requests.get(f"{API_URL}/users/{user_id}/gesture-settings")
    print_response(verify_response, "Verification Response")
    
    return {
        "get_successful": get_response.status_code == 200,
        "post_successful": post_response.status_code == 200,
        "settings_updated": verify_response.status_code == 200 and 
                           verify_response.json().get("gesture_settings", {}).get("enabled") == True
    }

def test_session_arvr_state(session_id):
    print_separator("Testing Session AR/VR State Endpoint")
    
    # Test POST session AR/VR state
    print("Testing POST /api/sessions/{session_id}/arvr-state")
    arvr_state = {
        "state": {
            "mode": "vr",
            "enabled": True,
            "settings": {
                "immersion_level": "full",
                "interaction_mode": "hands",
                "environment": "space",
                "avatar_visible": True
            }
        }
    }
    post_response = requests.post(f"{API_URL}/sessions/{session_id}/arvr-state", json=arvr_state)
    print_response(post_response, "POST Response")
    
    return {
        "post_successful": post_response.status_code == 200,
        "state_updated": post_response.status_code == 200 and 
                        post_response.json().get("state", {}).get("mode") == "vr"
    }

def test_learning_dna(user_id):
    print_separator("Testing Learning DNA Endpoint")
    
    # Test GET learning DNA
    print("Testing GET /api/users/{user_id}/learning-dna")
    get_response = requests.get(f"{API_URL}/users/{user_id}/learning-dna")
    print_response(get_response, "GET Response")
    
    return {
        "get_successful": get_response.status_code == 200,
        "has_learning_style": get_response.status_code == 200 and 
                             "learning_style" in get_response.json()
    }

def test_learning_mode(user_id):
    print_separator("Testing Learning Mode Endpoint")
    
    # Test POST learning mode
    print("Testing POST /api/users/{user_id}/learning-mode")
    learning_mode = {
        "preferred_mode": "socratic",
        "preferences": {
            "depth": "advanced",
            "pace": "accelerated",
            "style": "interactive",
            "focus_areas": ["critical_thinking", "problem_solving"]
        }
    }
    post_response = requests.post(f"{API_URL}/users/{user_id}/learning-mode", json=learning_mode)
    print_response(post_response, "POST Response")
    
    return {
        "post_successful": post_response.status_code == 200,
        "mode_updated": post_response.status_code == 200 and 
                       post_response.json().get("preferred_mode") == "socratic"
    }

if __name__ == "__main__":
    print_separator("MasterX AR/VR and Gesture Control API Testing")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test health check
        health_data = test_health_check()
        
        # Create test user
        user = create_test_user()
        user_id = user["id"]
        
        # Create test session
        session = create_test_session(user_id)
        session_id = session["id"]
        
        # Test AR/VR settings
        arvr_settings_results = test_arvr_settings(user_id)
        
        # Test gesture settings
        gesture_settings_results = test_gesture_settings(user_id)
        
        # Test session AR/VR state (this was the fixed endpoint)
        session_arvr_state_results = test_session_arvr_state(session_id)
        
        # Test learning DNA
        learning_dna_results = test_learning_dna(user_id)
        
        # Test learning mode
        learning_mode_results = test_learning_mode(user_id)
        
        # Print summary
        print_separator("Test Summary")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        print(f"AR/VR Settings: GET {'✅' if arvr_settings_results['get_successful'] else '❌'}, POST {'✅' if arvr_settings_results['post_successful'] else '❌'}")
        print(f"Gesture Settings: GET {'✅' if gesture_settings_results['get_successful'] else '❌'}, POST {'✅' if gesture_settings_results['post_successful'] else '❌'}")
        print(f"Session AR/VR State: POST {'✅' if session_arvr_state_results['post_successful'] else '❌'}")
        print(f"Learning DNA: GET {'✅' if learning_dna_results['get_successful'] else '❌'}")
        print(f"Learning Mode: POST {'✅' if learning_mode_results['post_successful'] else '❌'}")
        
        # Print overall results
        print_separator("Overall Test Results")
        print(json.dumps({
            "arvr_settings": arvr_settings_results,
            "gesture_settings": gesture_settings_results,
            "session_arvr_state": session_arvr_state_results,
            "learning_dna": learning_dna_results,
            "learning_mode": learning_mode_results,
            "timestamp": datetime.now().isoformat()
        }, indent=2))
        
        print("\nTests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise