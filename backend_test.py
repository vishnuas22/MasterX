#!/usr/bin/env python3
import requests
import json
import uuid
import time
from datetime import datetime

# Use local backend URL for testing
BACKEND_URL = "http://localhost:8001"

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

def test_core_functionality():
    """Test the core functionality required for the onboarding flow"""
    try:
        print_separator("Testing Core Functionality for Onboarding Flow")
        
        # 1. Test health check endpoint
        print("1. Testing Health Check Endpoint")
        health_response = requests.get(f"{API_URL}/health")
        print_response(health_response)
        health_check_success = health_response.status_code == 200
        
        # 2. Test user creation
        print("2. Testing User Creation")
        test_email = f"onboarding_test_{uuid.uuid4()}@example.com"
        test_name = "Onboarding Test User"
        user_data = {
            "email": test_email,
            "name": test_name,
            "learning_preferences": {}
        }
        user_response = requests.post(f"{API_URL}/users", json=user_data)
        print_response(user_response)
        user_creation_success = user_response.status_code in [200, 201]
        
        if user_creation_success:
            user_id = user_response.json()["id"]
        else:
            # If user creation failed, try to get by email
            print("User creation failed, trying to get by email...")
            user_response = requests.get(f"{API_URL}/users/email/{test_email}")
            print_response(user_response)
            user_id = user_response.json()["id"] if user_response.status_code == 200 else None
        
        # 3. Test user retrieval by email
        print("3. Testing User Retrieval by Email")
        email_response = requests.get(f"{API_URL}/users/email/{test_email}")
        print_response(email_response)
        user_retrieval_success = email_response.status_code == 200
        
        # 4. Test session creation
        print("4. Testing Session Creation")
        session_data = {
            "user_id": user_id,
            "subject": "Programming",
            "learning_objectives": ["Learn Python", "Build APIs"]
        }
        session_response = requests.post(f"{API_URL}/sessions", json=session_data)
        print_response(session_response)
        session_creation_success = session_response.status_code in [200, 201]
        
        if session_creation_success:
            session_id = session_response.json()["id"]
        else:
            session_id = None
        
        # 5. Test basic chat functionality
        print("5. Testing Basic Chat Functionality")
        chat_success = False
        if session_id:
            chat_data = {
                "session_id": session_id,
                "user_message": "Hello, I'm testing the chat functionality."
            }
            chat_response = requests.post(f"{API_URL}/chat", json=chat_data)
            print_response(chat_response)
            chat_success = chat_response.status_code == 200
        
        # Print summary
        print_separator("Core Functionality Test Summary")
        print(f"1. Health Check: {'✅' if health_check_success else '❌'}")
        print(f"2. User Creation: {'✅' if user_creation_success else '❌'}")
        print(f"3. User Retrieval by Email: {'✅' if user_retrieval_success else '❌'}")
        print(f"4. Session Creation: {'✅' if session_creation_success else '❌'}")
        print(f"5. Basic Chat: {'✅' if chat_success else '❌'}")
        
        return {
            "health_check": health_check_success,
            "user_creation": user_creation_success,
            "user_retrieval": user_retrieval_success,
            "session_creation": session_creation_success,
            "basic_chat": chat_success,
            "all_passed": all([
                health_check_success,
                user_creation_success,
                user_retrieval_success,
                session_creation_success,
                chat_success
            ])
        }
    except Exception as e:
        print(f"Error in test_core_functionality: {str(e)}")
        return {
            "error": str(e),
            "test_completed": False
        }

def test_premium_chat_streaming():
    """Test the premium chat streaming endpoint"""
    try:
        print_separator("Testing Premium Chat Streaming")
        
        # Create test user
        user = create_test_user()
        user_id = user["id"]
        
        # Create test session
        session = create_test_session(user_id)
        session_id = session["id"]
        
        # Test premium chat streaming
        print("Testing POST /api/chat/premium/stream")
        
        # Prepare request data
        stream_data = {
            "session_id": session_id,
            "user_message": "Explain quantum computing in simple terms",
            "context": {
                "learning_mode": "adaptive"
            }
        }
        
        # Make the request
        try:
            # Using stream=True to handle streaming response
            stream_response = requests.post(
                f"{API_URL}/chat/premium/stream", 
                json=stream_data,
                stream=True,
                timeout=30  # Set a timeout to avoid hanging
            )
            
            print(f"Status Code: {stream_response.status_code}")
            
            # Check if the request was successful
            if stream_response.status_code == 200:
                print("Stream started successfully")
                
                # Read a few chunks to verify streaming works
                chunk_count = 0
                for chunk in stream_response.iter_lines():
                    if chunk:
                        # Process the chunk (remove 'data: ' prefix if present)
                        chunk_str = chunk.decode('utf-8')
                        if chunk_str.startswith('data: '):
                            chunk_str = chunk_str[6:]
                        
                        try:
                            chunk_data = json.loads(chunk_str)
                            chunk_type = chunk_data.get('type', '')
                            
                            if chunk_type == 'chunk':
                                print(f"Received content chunk: {chunk_data.get('content', '')[:30]}...")
                            elif chunk_type == 'complete':
                                print(f"Stream completed with suggestions: {chunk_data.get('suggestions', [])}")
                                break
                            elif chunk_type == 'error':
                                print(f"Stream error: {chunk_data.get('message', '')}")
                                break
                        except json.JSONDecodeError:
                            print(f"Received non-JSON chunk: {chunk_str[:50]}...")
                        
                        chunk_count += 1
                        if chunk_count >= 5:  # Limit to 5 chunks for testing
                            print("Received 5 chunks, stopping stream read...")
                            break
                
                print(f"Successfully read {chunk_count} chunks from the stream")
                streaming_success = True
            else:
                print(f"Stream request failed with status code: {stream_response.status_code}")
                print(f"Response: {stream_response.text}")
                streaming_success = False
                
        except requests.exceptions.RequestException as e:
            print(f"Error making streaming request: {str(e)}")
            streaming_success = False
        
        # Print summary
        print_separator("Premium Chat Streaming Test Summary")
        print(f"Premium Chat Streaming: {'✅' if streaming_success else '❌'}")
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "premium_streaming": streaming_success
        }
    except Exception as e:
        print(f"Error in test_premium_chat_streaming: {str(e)}")
        return {
            "error": str(e),
            "test_completed": False
        }

if __name__ == "__main__":
    print_separator("MasterX Backend API Testing")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test core functionality for onboarding flow
        core_results = test_core_functionality()
        
        # Test premium chat streaming
        premium_streaming_results = test_premium_chat_streaming()
        
        # Test AR/VR and gesture control endpoints
        arvr_results = test_arvr_and_gesture_endpoints()
        
        # Test chat management functionality
        chat_results = test_chat_management()
        
        # Print overall results
        print_separator("Overall Test Results")
        print(json.dumps({
            "core_functionality": core_results,
            "premium_streaming": premium_streaming_results,
            "arvr_and_gesture": arvr_results,
            "chat_management": chat_results,
            "user_profile_settings": test_user_profile_settings(),
            "voice_search_integration": test_voice_search_integration(),
            "timestamp": datetime.now().isoformat()
        }, indent=2))
        
        print("\nTests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise