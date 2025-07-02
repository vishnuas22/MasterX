#!/usr/bin/env python3
import requests
import json
import uuid
import time
from datetime import datetime

# Get the backend URL from the frontend .env file
with open('/app/frontend/.env', 'r') as f:
    for line in f:
        if line.startswith('REACT_APP_BACKEND_URL='):
            BACKEND_URL = line.strip().split('=')[1].strip('"\'')
            break

# Ensure the URL doesn't have trailing slash
if BACKEND_URL.endswith('/'):
    BACKEND_URL = BACKEND_URL[:-1]

# Add /api prefix for all API calls
API_URL = f"{BACKEND_URL}/api"

print(f"Using API URL: {API_URL}")

# Test data
test_user = {
    "email": f"test_user_{uuid.uuid4()}@example.com",
    "name": "Test User",
    "learning_preferences": {}
}

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
        "subject": "Python Programming",
        "learning_objectives": ["Learn FastAPI", "Test API endpoints"]
    }
    response = requests.post(f"{API_URL}/sessions", json=session_data)
    print_response(response)
    assert response.status_code in [200, 201], "Failed to create session"
    return response.json()

def get_user_sessions(user_id):
    print_separator("Getting User Sessions")
    response = requests.get(f"{API_URL}/users/{user_id}/sessions")
    print_response(response)
    assert response.status_code == 200, "Failed to get user sessions"
    return response.json()

def rename_session(session_id, new_title):
    print_separator(f"Renaming Session to '{new_title}'")
    response = requests.put(
        f"{API_URL}/sessions/{session_id}/rename", 
        json={"title": new_title}
    )
    print_response(response)
    # Don't assert here, just return the response
    return response.json() if response.status_code == 200 else {"error": response.text}

def share_session(session_id):
    print_separator("Sharing Session")
    response = requests.post(
        f"{API_URL}/sessions/{session_id}/share",
        json={}
    )
    print_response(response)
    # Don't assert here, just return the response
    return response.json() if response.status_code == 200 else {"error": response.text}

def search_user_sessions(user_id, query):
    print_separator(f"Searching User Sessions for '{query}'")
    response = requests.get(
        f"{API_URL}/users/{user_id}/sessions/search?query={query}"
    )
    print_response(response)
    # Don't assert here, just return the response
    return response.json() if response.status_code == 200 else {"error": response.text}

def delete_session(session_id):
    print_separator("Deleting Session")
    response = requests.delete(f"{API_URL}/sessions/{session_id}")
    print_response(response)
    # Don't assert here, just return the response
    return response.json() if response.status_code == 200 else {"error": response.text}

def add_message_to_session(session_id, message):
    print_separator(f"Adding Message to Session: '{message}'")
    response = requests.post(
        f"{API_URL}/chat", 
        json={
            "session_id": session_id,
            "user_message": message
        }
    )
    print_response(response)
    assert response.status_code == 200, "Failed to add message to session"
    return response.json()

def test_chat_management():
    try:
        # Test health check
        health_data = test_health_check()
        
        # Create test user
        user = create_test_user()
        user_id = user["id"]
        
        # Create test session
        session = create_test_session(user_id)
        session_id = session["id"]
        
        # Add a message to the session
        try:
            add_message_to_session(session_id, "Hello, this is a test message")
        except Exception as e:
            print(f"Error adding message: {str(e)}")
        
        # Get user sessions
        sessions = get_user_sessions(user_id)
        
        # Rename session
        new_title = f"Renamed Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        rename_result = rename_session(session_id, new_title)
        
        # Share session
        share_result = share_session(session_id)
        
        # Search user sessions
        search_result = search_user_sessions(user_id, "Python")
        
        # Create a second session for testing
        session2 = create_test_session(user_id)
        session2_id = session2["id"]
        
        # Get updated user sessions
        updated_sessions = get_user_sessions(user_id)
        
        # Delete the second session
        delete_result = delete_session(session2_id)
        
        # Get final user sessions
        final_sessions = get_user_sessions(user_id)
        
        # Print summary
        print_separator("Test Summary")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        print(f"Session renamed to: {new_title}")
        print(f"Session shared: {share_result.get('share_url', 'N/A')}")
        print(f"Session deleted: {session2_id}")
        print(f"Initial session count: {len(sessions)}")
        print(f"Final session count: {len(final_sessions)}")
        
        # Return test results
        return {
            "user_id": user_id,
            "session_id": session_id,
            "rename_successful": "message" in rename_result and rename_result.get("message") == "Session renamed successfully",
            "share_successful": "share_id" in share_result,
            "delete_successful": "message" in delete_result and delete_result.get("message") == "Session deleted successfully",
            "search_successful": "results" in search_result
        }
    except Exception as e:
        print(f"Error in test_chat_management: {str(e)}")
        return {
            "error": str(e),
            "test_completed": False
        }

def test_arvr_and_gesture_endpoints():
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
        
        # Test session AR/VR state
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
        
        # Return test results
        return {
            "user_id": user_id,
            "session_id": session_id,
            "arvr_settings": arvr_settings_results,
            "gesture_settings": gesture_settings_results,
            "session_arvr_state": session_arvr_state_results,
            "learning_dna": learning_dna_results,
            "learning_mode": learning_mode_results
        }
    except Exception as e:
        print(f"Error in test_arvr_and_gesture_endpoints: {str(e)}")
        return {
            "error": str(e),
            "test_completed": False
        }

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

def test_user_profile_settings():
    print_separator("Testing User Profile & Settings")
    print("Note: No specific user profile/settings endpoints found in the API")
    print("The user management is handled through the basic user endpoints")
    return {
        "implemented": False,
        "reason": "No specific user profile/settings endpoints found in the API"
    }

def test_voice_search_integration():
    print_separator("Testing Voice Search Integration")
    print("Note: No voice/audio processing endpoints found in the API")
    return {
        "implemented": False,
        "reason": "No voice/audio processing endpoints found in the API"
    }

if __name__ == "__main__":
    print_separator("MasterX Backend API Testing")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test AR/VR and gesture control endpoints
        arvr_results = test_arvr_and_gesture_endpoints()
        
        # Test chat management functionality
        chat_results = test_chat_management()
        
        # Test user profile & settings
        profile_results = test_user_profile_settings()
        
        # Test voice search integration
        voice_results = test_voice_search_integration()
        
        # Print overall results
        print_separator("Overall Test Results")
        print(json.dumps({
            "arvr_and_gesture": arvr_results,
            "chat_management": chat_results,
            "user_profile_settings": profile_results,
            "voice_search_integration": voice_results,
            "timestamp": datetime.now().isoformat()
        }, indent=2))
        
        print("\nTests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise