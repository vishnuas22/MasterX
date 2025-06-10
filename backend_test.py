#!/usr/bin/env python3
import requests
import json
import uuid
import time
import unittest
from datetime import datetime

# Backend URL from frontend/.env
BACKEND_URL = "https://d4d3b5f0-2654-475a-90aa-7c47846cedbb.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class MasterXBackendTests(unittest.TestCase):
    """Comprehensive tests for MasterX AI Mentor System backend"""
    
    def setUp(self):
        """Setup for tests - create test user and session"""
        self.test_user_email = f"test_{uuid.uuid4()}@example.com"
        self.test_user_name = "Test User"
        
        # Create test user
        user_data = {
            "email": self.test_user_email,
            "name": self.test_user_name,
            "learning_preferences": {
                "preferred_style": "visual",
                "pace": "moderate"
            }
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        if response.status_code == 200:
            self.test_user = response.json()
            print(f"Created test user: {self.test_user['id']}")
            print(f"User data: {self.test_user}")
            
            # Verify user exists by getting it by ID
            verify_response = requests.get(f"{API_BASE}/users/{self.test_user['id']}")
            if verify_response.status_code == 200:
                print(f"Verified user exists: {verify_response.json()['id']}")
            else:
                print(f"Failed to verify user: {verify_response.status_code} - {verify_response.text}")
            
            # Create test session
            session_data = {
                "user_id": self.test_user["id"],
                "subject": "Python Programming",
                "learning_objectives": ["Learn basic syntax", "Understand functions"],
                "difficulty_level": "beginner"
            }
            
            response = requests.post(f"{API_BASE}/sessions", json=session_data)
            if response.status_code == 200:
                self.test_session = response.json()
                print(f"Created test session: {self.test_session['id']}")
            else:
                print(f"Failed to create test session: {response.status_code} - {response.text}")
                self.test_session = None
        else:
            print(f"Failed to create test user: {response.status_code} - {response.text}")
            self.test_user = None
            self.test_session = None
    
    def test_01_health_check(self):
        """Test health check endpoint"""
        print("\n=== Testing Health Check Endpoint ===")
        
        # Test root endpoint
        response = requests.get(f"{API_BASE}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        print(f"Root endpoint: {data}")
        
        # Test health endpoint
        response = requests.get(f"{API_BASE}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("database", data)
        print(f"Health endpoint: {data}")
    
    def test_02_user_management(self):
        """Test user management endpoints"""
        print("\n=== Testing User Management ===")
        
        # Create a new user specifically for this test
        user_email = f"user_mgmt_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "User Management Test",
            "learning_preferences": {"preferred_style": "visual"}
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Created user for management test: {user['id']}")
        
        # Test get user by ID
        response = requests.get(f"{API_BASE}/users/{user['id']}")
        self.assertEqual(response.status_code, 200)
        user_data = response.json()
        self.assertEqual(user_data["email"], user_email)
        print(f"Get user by ID: {user_data['id']}")
        
        # Test get user by email
        response = requests.get(f"{API_BASE}/users/email/{user_email}")
        self.assertEqual(response.status_code, 200)
        user_data = response.json()
        self.assertEqual(user_data["id"], user["id"])
        print(f"Get user by email: {user_data['email']}")
        
        # Test user not found
        response = requests.get(f"{API_BASE}/users/{uuid.uuid4()}")
        self.assertEqual(response.status_code, 404)
        print(f"User not found test: {response.status_code}")
        
        # Test duplicate user creation (should fail)
        user_data = {
            "email": user_email,
            "name": "Duplicate User"
        }
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 400)
        print(f"Duplicate user test: {response.status_code}")
    
    def test_03_session_management(self):
        """Test session management endpoints"""
        print("\n=== Testing Session Management ===")
        
        # Create a new user specifically for this test
        user_email = f"session_test_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "Session Test User"
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Created user for session test: {user['id']}")
        
        # Verify user exists
        response = requests.get(f"{API_BASE}/users/{user['id']}")
        self.assertEqual(response.status_code, 200)
        print(f"Verified user exists: {response.json()['id']}")
        
        # Create a session
        session_data = {
            "user_id": user["id"],
            "subject": "Python Programming",
            "learning_objectives": ["Learn basic syntax"],
            "difficulty_level": "beginner"
        }
        
        response = requests.post(f"{API_BASE}/sessions", json=session_data)
        self.assertEqual(response.status_code, 200)
        session = response.json()
        print(f"Created session: {session['id']}")
        
        # Test get session by ID
        response = requests.get(f"{API_BASE}/sessions/{session['id']}")
        self.assertEqual(response.status_code, 200)
        session_data = response.json()
        self.assertEqual(session_data["user_id"], user["id"])
        print(f"Get session by ID: {session_data['id']}")
        
        # Test get user sessions
        response = requests.get(f"{API_BASE}/users/{user['id']}/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()
        self.assertGreaterEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["user_id"], user["id"])
        print(f"Get user sessions: Found {len(sessions)} sessions")
        
        # Test end session
        response = requests.put(f"{API_BASE}/sessions/{session['id']}/end")
        self.assertEqual(response.status_code, 200)
        print(f"End session: {response.json()}")
        
        # Verify session is ended
        response = requests.get(f"{API_BASE}/sessions/{session['id']}")
        self.assertEqual(response.status_code, 200)
        session_data = response.json()
        self.assertFalse(session_data["is_active"])
        print(f"Verify session ended: is_active={session_data['is_active']}")
    
    def test_04_chat_functionality(self):
        """Test AI chat functionality"""
        print("\n=== Testing AI Chat Functionality ===")
        
        # Create a new user and session for this test
        user_email = f"chat_test_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "Chat Test User"
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Created user for chat test: {user['id']}")
        
        # Create a session
        session_data = {
            "user_id": user["id"],
            "subject": "Machine Learning",
            "learning_objectives": ["Understand neural networks"],
            "difficulty_level": "beginner"
        }
        
        response = requests.post(f"{API_BASE}/sessions", json=session_data)
        self.assertEqual(response.status_code, 200)
        session = response.json()
        print(f"Created session for chat test: {session['id']}")
        
        # Test regular chat endpoint
        chat_request = {
            "session_id": session["id"],
            "user_message": "What is a neural network?",
            "context": {
                "user_background": "Beginner in machine learning"
            }
        }
        
        response = requests.post(f"{API_BASE}/chat", json=chat_request)
        self.assertEqual(response.status_code, 200)
        chat_response = response.json()
        self.assertIn("response", chat_response)
        self.assertIn("response_type", chat_response)
        self.assertIn("suggested_actions", chat_response)
        print(f"Chat response type: {chat_response['response_type']}")
        print(f"Chat response length: {len(chat_response['response'])}")
        
        # Test getting session messages
        response = requests.get(f"{API_BASE}/sessions/{session['id']}/messages")
        self.assertEqual(response.status_code, 200)
        messages = response.json()
        self.assertGreaterEqual(len(messages), 2)  # Should have user message and AI response
        print(f"Session messages: Found {len(messages)} messages")
        
        # Test streaming chat endpoint
        chat_request = {
            "session_id": session["id"],
            "user_message": "Explain backpropagation in simple terms",
            "context": {
                "user_background": "Beginner in machine learning"
            }
        }
        
        # For streaming, we need to handle the response differently
        # We'll just check that the endpoint returns a 200 status code
        response = requests.post(f"{API_BASE}/chat/stream", json=chat_request, stream=True)
        self.assertEqual(response.status_code, 200)
        
        # Read a few chunks to verify streaming works
        chunks_received = 0
        for chunk in response.iter_lines():
            if chunk:
                chunks_received += 1
                if chunks_received >= 5:  # Just read a few chunks
                    break
        
        self.assertGreater(chunks_received, 0)
        print(f"Streaming chat: Received {chunks_received} chunks")
    
    def test_05_exercise_generation(self):
        """Test exercise generation functionality"""
        print("\n=== Testing Exercise Generation ===")
        
        # Test exercise generation
        response = requests.post(
            f"{API_BASE}/exercises/generate",
            params={
                "topic": "Python loops",
                "difficulty": "beginner",
                "exercise_type": "multiple_choice"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        exercise_data = response.json()
        self.assertIn("question", exercise_data)
        print(f"Generated exercise: {exercise_data.get('question', '')[:50]}...")
        
        # Test exercise analysis
        analysis_request = {
            "question": "What is the output of: for i in range(3): print(i)",
            "user_answer": "0, 1, 2",
            "correct_answer": "0\n1\n2"
        }
        
        response = requests.post(f"{API_BASE}/exercises/analyze", json=analysis_request)
        self.assertEqual(response.status_code, 200)
        analysis = response.json()
        self.assertIn("feedback", analysis)
        print(f"Exercise analysis: {analysis.get('feedback', '')[:50]}...")
    
    def test_06_learning_path(self):
        """Test learning path generation"""
        print("\n=== Testing Learning Path Generation ===")
        
        # Test learning path generation
        response = requests.post(
            f"{API_BASE}/learning-paths/generate",
            params={
                "subject": "Data Science",
                "user_level": "intermediate",
                "goals": ["Master machine learning", "Learn data visualization"]
            }
        )
        
        self.assertEqual(response.status_code, 200)
        path_data = response.json()
        self.assertIn("learning_path", path_data)
        print(f"Generated learning path: {path_data.get('learning_path', '')[:50]}...")
    
    def test_07_error_handling(self):
        """Test error handling"""
        print("\n=== Testing Error Handling ===")
        
        # Test invalid session ID
        chat_request = {
            "session_id": str(uuid.uuid4()),
            "user_message": "This should fail",
            "context": {}
        }
        
        response = requests.post(f"{API_BASE}/chat", json=chat_request)
        self.assertEqual(response.status_code, 404)
        print(f"Invalid session test: {response.status_code}")
        
        # Test missing required fields
        user_data = {
            "name": "Missing Email User"
            # Missing email field
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertNotEqual(response.status_code, 200)
        print(f"Missing required field test: {response.status_code}")
        
        # Test invalid user ID
        response = requests.get(f"{API_BASE}/users/{uuid.uuid4()}")
        self.assertEqual(response.status_code, 404)
        print(f"Invalid user ID test: {response.status_code}")

if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
