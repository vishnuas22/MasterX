#!/usr/bin/env python3
import requests
import json
import uuid
import time
import os

# Configuration
BACKEND_URL = "http://localhost:8001/api"
TEST_USER_EMAIL = "test_groq@masterx.ai"
TEST_USER_NAME = "Test Groq User"
TEST_SESSION_SUBJECT = "Mathematics"
TEST_SESSION_DIFFICULTY = "intermediate"

def create_test_user():
    """Create a test user or get existing one"""
    try:
        # First try to get the user by email
        response = requests.get(f"{BACKEND_URL}/users/email/{TEST_USER_EMAIL}")
        
        if response.status_code == 200:
            print("User already exists, using existing user")
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
        if response.status_code == 200 and "id" in response.json():
            print("User created successfully")
            return response.json()
        else:
            print(f"Failed to create user: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error creating/getting user: {str(e)}")
        return None

def create_test_session(user_id):
    """Create a test session"""
    try:
        session_data = {
            "user_id": user_id,
            "subject": TEST_SESSION_SUBJECT,
            "learning_objectives": ["Understand quadratic equations", "Master factorization"],
            "difficulty_level": TEST_SESSION_DIFFICULTY
        }
        
        response = requests.post(f"{BACKEND_URL}/sessions", json=session_data)
        if response.status_code == 200 and "id" in response.json():
            print("Session created successfully")
            return response.json()
        else:
            print(f"Failed to create session: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error creating session: {str(e)}")
        return None

def test_premium_chat(session_id):
    """Test premium chat functionality with Groq API key"""
    try:
        chat_data = {
            "session_id": session_id,
            "user_message": "Can you explain the quadratic formula in detail?",
            "context": {
                "learning_mode": "adaptive"
            }
        }
        
        print("Sending premium chat request...")
        response = requests.post(f"{BACKEND_URL}/chat/premium", json=chat_data)
        
        if response.status_code == 200:
            print("Premium chat request successful")
            result = response.json()
            print(f"Response type: {result.get('response_type', 'unknown')}")
            
            # Check if there's an error in the response
            if result.get('response_type') == 'error':
                print(f"Error in response: {result.get('metadata', {}).get('error', 'unknown error')}")
                return False
            
            # Print a snippet of the response
            response_text = result.get('response', '')
            print(f"Response snippet: {response_text[:100]}...")
            
            return True
        else:
            print(f"Failed to send premium chat request: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error testing premium chat: {str(e)}")
        return False

def test_available_models():
    """Test getting available AI models"""
    try:
        print("Getting available models...")
        response = requests.get(f"{BACKEND_URL}/models/available")
        
        if response.status_code == 200:
            print("Available models request successful")
            result = response.json()
            
            # Check if Groq models are available
            available_models = result.get('available_models', [])
            print(f"Available models: {available_models}")
            
            # Check model capabilities
            model_capabilities = result.get('model_capabilities', {})
            for model_name, capabilities in model_capabilities.items():
                if capabilities.get('provider') == 'groq':
                    print(f"Groq model: {model_name}")
                    print(f"  Available: {capabilities.get('available', False)}")
                    print(f"  Specialties: {capabilities.get('specialties', [])}")
            
            # Check if at least one Groq model is available
            groq_models = [model for model in available_models if 'deepseek' in model.lower()]
            return len(groq_models) > 0
        else:
            print(f"Failed to get available models: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error testing available models: {str(e)}")
        return False

def test_streaming_chat(session_id):
    """Test streaming chat functionality with Groq API key"""
    try:
        chat_data = {
            "session_id": session_id,
            "user_message": "Explain the concept of derivatives in calculus with examples",
            "context": {
                "learning_mode": "adaptive"
            }
        }
        
        print("Sending streaming chat request...")
        response = requests.post(f"{BACKEND_URL}/chat/premium/stream", json=chat_data, stream=True)
        
        if response.status_code == 200:
            print("Streaming chat request successful")
            
            # Read a few chunks to verify streaming is working
            chunk_count = 0
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    chunk_count += 1
                    print(f"Received chunk {chunk_count}")
                    if chunk_count >= 3:  # Just check a few chunks
                        break
            
            response.close()  # Close the connection after testing
            
            return chunk_count > 0
        else:
            print(f"Failed to send streaming chat request: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error testing streaming chat: {str(e)}")
        return False

def run_groq_tests():
    """Run tests specifically for Groq API integration"""
    print("\n===== TESTING GROQ API INTEGRATION =====\n")
    
    # Check if Groq API key is set
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        print("GROQ_API_KEY environment variable not set")
        return False
    
    print(f"Using Groq API key: {groq_api_key[:10]}...")
    
    # Test available models
    models_available = test_available_models()
    if not models_available:
        print("No Groq models available")
        return False
    
    # Create test user and session
    user = create_test_user()
    if not user:
        print("Failed to create/get test user")
        return False
    
    user_id = user.get("id")
    session = create_test_session(user_id)
    if not session:
        print("Failed to create test session")
        return False
    
    session_id = session.get("id")
    
    # Test premium chat
    premium_chat_success = test_premium_chat(session_id)
    
    # Test streaming chat
    streaming_chat_success = test_streaming_chat(session_id)
    
    # Print summary
    print("\n===== GROQ API TEST SUMMARY =====")
    print(f"Models Available: {'✅ PASS' if models_available else '❌ FAIL'}")
    print(f"Premium Chat: {'✅ PASS' if premium_chat_success else '❌ FAIL'}")
    print(f"Streaming Chat: {'✅ PASS' if streaming_chat_success else '❌ FAIL'}")
    print("==================================\n")
    
    return models_available and premium_chat_success and streaming_chat_success

if __name__ == "__main__":
    run_groq_tests()