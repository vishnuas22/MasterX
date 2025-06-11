#!/usr/bin/env python3
import requests
import json
import uuid
import time
import unittest
from datetime import datetime
import sseclient

# Backend URL from frontend/.env
BACKEND_URL = "https://d4d3b5f0-2654-475a-90aa-7c47846cedbb.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

# Test data from review request
TEST_USER_ID = "d2fa2390-24ef-4bf9-a288-eb6b70c0b5ab"
TEST_USER_EMAIL = "test@example.com"
TEST_SESSION_ID = "a2c574e3-0414-4fe2-ab8e-74c1cdba2489"

class MasterXBackendTests(unittest.TestCase):
    """Comprehensive tests for MasterX AI Mentor System backend with premium features"""
    
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
                "pace": "moderate",
                "interests": ["AI", "Machine Learning", "Data Science"],
                "goals": ["Master programming", "Build AI applications"]
            }
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        if response.status_code == 200:
            self.test_user = response.json()
            print(f"Created test user: {self.test_user['id']}")
            
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
                "learning_objectives": ["Learn basic syntax", "Understand functions", "Master object-oriented programming"],
                "difficulty_level": "intermediate"
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
            
        # Also try to use the provided test user and session from review request
        self.provided_user_id = TEST_USER_ID
        self.provided_user_email = TEST_USER_EMAIL
        self.provided_session_id = TEST_SESSION_ID
    
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
        
        # Verify API is using the correct model
        self.assertIn("ai_service", data)
        print(f"AI service status: {data.get('ai_service', 'Not reported')}")
    
    def test_02_user_management(self):
        """Test user management endpoints"""
        print("\n=== Testing User Management ===")
        
        # Create a new user specifically for this test
        user_email = f"user_mgmt_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "User Management Test",
            "learning_preferences": {
                "preferred_style": "visual",
                "pace": "fast",
                "interests": ["Quantum Computing", "Blockchain"],
                "background": "Computer Science student"
            }
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
        
        # Test with provided test user ID from review request
        if self.provided_user_id:
            response = requests.get(f"{API_BASE}/users/{self.provided_user_id}")
            print(f"Provided test user lookup status: {response.status_code}")
            if response.status_code == 200:
                print(f"Provided test user exists: {response.json()['id']}")
            
        # Test with provided test user email from review request
        if self.provided_user_email:
            response = requests.get(f"{API_BASE}/users/email/{self.provided_user_email}")
            print(f"Provided test user email lookup status: {response.status_code}")
            if response.status_code == 200:
                print(f"Provided test user email exists: {response.json()['id']}")
    
    def test_03_session_management(self):
        """Test session management endpoints"""
        print("\n=== Testing Session Management ===")
        
        # Create a new user specifically for this test
        user_email = f"session_test_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "Session Test User",
            "learning_preferences": {
                "preferred_style": "kinesthetic",
                "pace": "moderate"
            }
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Created user for session test: {user['id']}")
        
        # Verify user exists
        response = requests.get(f"{API_BASE}/users/{user['id']}")
        self.assertEqual(response.status_code, 200)
        print(f"Verified user exists: {response.json()['id']}")
        
        # Create a session with advanced learning objectives
        session_data = {
            "user_id": user["id"],
            "subject": "Advanced Machine Learning",
            "learning_objectives": [
                "Understand neural network architectures", 
                "Master reinforcement learning algorithms",
                "Implement transformer models"
            ],
            "difficulty_level": "advanced"
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
        
        # Test with provided test session ID from review request
        if self.provided_session_id:
            response = requests.get(f"{API_BASE}/sessions/{self.provided_session_id}")
            print(f"Provided test session lookup status: {response.status_code}")
            if response.status_code == 200:
                print(f"Provided test session exists: {response.json()['id']}")
                
                # Get messages for the provided test session
                response = requests.get(f"{API_BASE}/sessions/{self.provided_session_id}/messages")
                if response.status_code == 200:
                    messages = response.json()
                    print(f"Found {len(messages)} messages in provided test session")
    
    def test_04_enhanced_ai_chat(self):
        """Test enhanced AI chat functionality with DeepSeek R1 70B model"""
        print("\n=== Testing Enhanced AI Chat Functionality ===")
        
        # Create a new user and session for this test
        user_email = f"enhanced_chat_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "Enhanced Chat Test User",
            "learning_preferences": {
                "preferred_style": "visual",
                "pace": "moderate",
                "interests": ["Quantum Computing", "AI Ethics"]
            }
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Created user for enhanced chat test: {user['id']}")
        
        # Create a session with advanced topics
        session_data = {
            "user_id": user["id"],
            "subject": "Quantum Computing",
            "learning_objectives": [
                "Understand quantum superposition", 
                "Learn about quantum algorithms",
                "Explore quantum machine learning"
            ],
            "difficulty_level": "advanced"
        }
        
        response = requests.post(f"{API_BASE}/sessions", json=session_data)
        self.assertEqual(response.status_code, 200)
        session = response.json()
        print(f"Created session for enhanced chat test: {session['id']}")
        
        # Test enhanced chat with complex question
        chat_request = {
            "session_id": session["id"],
            "user_message": "Explain the relationship between quantum computing and machine learning. How can quantum algorithms enhance AI capabilities?",
            "context": {
                "user_background": "Advanced in computer science, beginner in quantum physics"
            }
        }
        
        response = requests.post(f"{API_BASE}/chat", json=chat_request)
        self.assertEqual(response.status_code, 200)
        chat_response = response.json()
        
        # Verify enhanced response structure
        self.assertIn("response", chat_response)
        self.assertIn("response_type", chat_response)
        self.assertIn("suggested_actions", chat_response)
        self.assertIn("concepts_covered", chat_response)
        self.assertIn("metadata", chat_response)
        
        # Verify premium metadata features
        metadata = chat_response.get("metadata", {})
        self.assertIn("model_used", metadata)
        self.assertIn("complexity_score", metadata)
        self.assertIn("engagement_score", metadata)
        self.assertIn("premium_features", metadata)
        
        # Check if DeepSeek model is being used
        model_used = metadata.get("model_used", "")
        self.assertIn("deepseek", model_used.lower())
        
        print(f"Chat response type: {chat_response['response_type']}")
        print(f"Chat response length: {len(chat_response['response'])}")
        print(f"Model used: {model_used}")
        print(f"Complexity score: {metadata.get('complexity_score', 'N/A')}")
        print(f"Concepts covered: {len(chat_response.get('concepts_covered', []))}")
        print(f"Suggested actions: {len(chat_response.get('suggested_actions', []))}")
        
        # Test premium formatting features
        response_text = chat_response.get("response", "")
        premium_formatting_elements = ["##", "###", "💡", "🎯", "📋", "🚀", "Pro Tip"]
        has_premium_formatting = any(element in response_text for element in premium_formatting_elements)
        
        self.assertTrue(has_premium_formatting, "Response should have premium formatting elements")
        print(f"Premium formatting detected: {has_premium_formatting}")
        
        # Test getting session messages
        response = requests.get(f"{API_BASE}/sessions/{session['id']}/messages")
        self.assertEqual(response.status_code, 200)
        messages = response.json()
        self.assertGreaterEqual(len(messages), 2)  # Should have user message and AI response
        print(f"Session messages: Found {len(messages)} messages")
        
        # Test with provided test session if available
        if self.provided_session_id:
            chat_request = {
                "session_id": self.provided_session_id,
                "user_message": "What are the key principles of effective learning that you use as an AI mentor?",
                "context": {
                    "user_background": "Education professional"
                }
            }
            
            response = requests.post(f"{API_BASE}/chat", json=chat_request)
            print(f"Chat with provided test session status: {response.status_code}")
            if response.status_code == 200:
                chat_response = response.json()
                print(f"Chat with provided session successful, response length: {len(chat_response.get('response', ''))}")
    
    def test_05_real_time_streaming(self):
        """Test real-time streaming chat functionality"""
        print("\n=== Testing Real-time Streaming Chat ===")
        
        if not self.test_session:
            print("Skipping streaming test as no test session is available")
            return
        
        # Test streaming chat endpoint with complex query
        chat_request = {
            "session_id": self.test_session["id"],
            "user_message": "Explain the concept of neural networks and backpropagation in detail. Include code examples in Python.",
            "context": {
                "user_background": "Intermediate programmer with basic ML knowledge"
            }
        }
        
        # For streaming, we need to handle the response differently
        response = requests.post(f"{API_BASE}/chat/stream", json=chat_request, stream=True)
        self.assertEqual(response.status_code, 200)
        
        # Read chunks to verify streaming works
        chunks_received = 0
        content_chunks = 0
        complete_signal = False
        
        for chunk in response.iter_lines():
            if chunk:
                chunks_received += 1
                try:
                    # Parse the SSE data
                    if chunk.startswith(b'data: '):
                        data = json.loads(chunk[6:].decode('utf-8'))
                        if data.get('type') == 'chunk':
                            content_chunks += 1
                        elif data.get('type') == 'complete':
                            complete_signal = True
                            # Check for suggestions in completion signal
                            self.assertIn('suggestions', data)
                except Exception as e:
                    print(f"Error parsing chunk: {e}")
                
                # Only read a reasonable number of chunks for testing
                if chunks_received >= 20 or complete_signal:
                    break
        
        self.assertGreater(content_chunks, 0, "Should receive content chunks")
        print(f"Streaming chat: Received {chunks_received} total chunks, {content_chunks} content chunks")
        print(f"Complete signal received: {complete_signal}")
        
        # Test with provided test session if available
        if self.provided_session_id:
            chat_request = {
                "session_id": self.provided_session_id,
                "user_message": "Explain how quantum computing differs from classical computing",
                "context": {}
            }
            
            response = requests.post(f"{API_BASE}/chat/stream", json=chat_request, stream=True)
            print(f"Streaming with provided test session status: {response.status_code}")
            
            if response.status_code == 200:
                chunks = 0
                for chunk in response.iter_lines():
                    if chunk:
                        chunks += 1
                        if chunks >= 5:  # Just check a few chunks
                            break
                print(f"Streaming with provided session successful, received {chunks} chunks")
    
    def test_06_premium_exercise_generation(self):
        """Test premium exercise generation functionality"""
        print("\n=== Testing Premium Exercise Generation ===")
        
        # Test exercise generation with various topics and difficulties
        topics = [
            ("Neural Networks", "advanced"),
            ("Python Data Structures", "intermediate"),
            ("Quantum Computing Basics", "beginner")
        ]
        
        for topic, difficulty in topics:
            print(f"\nGenerating {difficulty} exercise for: {topic}")
            response = requests.post(
                f"{API_BASE}/exercises/generate",
                params={
                    "topic": topic,
                    "difficulty": difficulty,
                    "exercise_type": "multiple_choice"
                }
            )
            
            self.assertEqual(response.status_code, 200)
            exercise_data = response.json()
            
            # Verify premium exercise features
            self.assertIn("question", exercise_data)
            
            # Check for premium features
            premium_features = [
                "options", "correct_answer", "explanation", "concepts",
                "difficulty_score", "pro_tips", "related_topics",
                "real_world_applications", "follow_up_questions"
            ]
            
            present_features = [feature for feature in premium_features if feature in exercise_data]
            print(f"Premium features present: {len(present_features)}/{len(premium_features)}")
            print(f"Exercise difficulty: {difficulty}")
            print(f"Question: {exercise_data.get('question', '')[:100]}...")
            
            # Test exercise analysis with the generated exercise
            if "question" in exercise_data:
                analysis_request = {
                    "question": exercise_data["question"],
                    "user_answer": "This is a test answer that demonstrates understanding of the concept",
                    "correct_answer": exercise_data.get("correct_answer", "The correct answer would be here")
                }
                
                response = requests.post(f"{API_BASE}/exercises/analyze", json=analysis_request)
                self.assertEqual(response.status_code, 200)
                analysis = response.json()
                self.assertIn("feedback", analysis)
                print(f"Exercise analysis feedback: {analysis.get('feedback', '')[:100]}...")
    
    def test_07_premium_learning_paths(self):
        """Test premium learning path generation"""
        print("\n=== Testing Premium Learning Path Generation ===")
        
        # Test learning path generation with various subjects and goals
        test_cases = [
            {
                "subject": "Machine Learning",
                "user_level": "beginner",
                "goals": ["Build a recommendation system", "Understand neural networks"]
            },
            {
                "subject": "Quantum Computing",
                "user_level": "intermediate",
                "goals": ["Implement quantum algorithms", "Understand quantum machine learning"]
            }
        ]
        
        for test_case in test_cases:
            print(f"\nGenerating learning path for: {test_case['subject']} ({test_case['user_level']})")
            response = requests.post(
                f"{API_BASE}/learning-paths/generate",
                params=test_case
            )
            
            self.assertEqual(response.status_code, 200)
            path_data = response.json()
            
            # Verify premium learning path features
            self.assertIn("learning_path", path_data)
            learning_path = path_data.get("learning_path", {})
            
            if isinstance(learning_path, dict):
                # Check for premium features in structured format
                premium_features = [
                    "title", "overview", "milestones", "learning_techniques",
                    "motivation_strategies", "adaptive_features", "premium_resources"
                ]
                
                present_features = [feature for feature in premium_features if feature in learning_path]
                print(f"Premium features present: {len(present_features)}/{len(premium_features)}")
                
                # Check milestones
                milestones = learning_path.get("milestones", [])
                print(f"Number of milestones: {len(milestones)}")
                
                if milestones and isinstance(milestones, list) and len(milestones) > 0:
                    first_milestone = milestones[0]
                    print(f"First milestone: {first_milestone.get('title', 'N/A')}")
                    
                    # Check for spaced repetition
                    has_spaced_repetition = "spaced_repetition" in first_milestone
                    print(f"Has spaced repetition: {has_spaced_repetition}")
            else:
                # Fallback text format
                print(f"Learning path content length: {len(str(learning_path))}")
    
    def test_08_error_handling(self):
        """Test error handling and edge cases"""
        print("\n=== Testing Error Handling and Edge Cases ===")
        
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
        
        # Test extremely long message
        if self.test_session:
            long_message = "Test " * 500  # Very long message
            chat_request = {
                "session_id": self.test_session["id"],
                "user_message": long_message,
                "context": {}
            }
            
            response = requests.post(f"{API_BASE}/chat", json=chat_request)
            print(f"Long message test: {response.status_code}")
            
            # Should either succeed or fail gracefully
            self.assertIn(response.status_code, [200, 400, 413, 422])
        
        # Test invalid exercise parameters
        response = requests.post(
            f"{API_BASE}/exercises/generate",
            params={
                "topic": "Invalid!@#$%^&*()",
                "difficulty": "impossible",
                "exercise_type": "invalid_type"
            }
        )
        
        print(f"Invalid exercise parameters test: {response.status_code}")
        # Should either handle gracefully or return error
        self.assertIn(response.status_code, [200, 400, 422])

if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
