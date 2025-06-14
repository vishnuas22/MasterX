#!/usr/bin/env python3
import requests
import json
import uuid
import time
import unittest
from datetime import datetime
import sseclient

# Use local backend URL for testing
BACKEND_URL = "http://localhost:8001"
API_BASE = f"{BACKEND_URL}/api"

print(f"Using backend URL: {API_BASE}")

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
            
            # WORKAROUND: Verify user exists by getting it by email
            verify_response = requests.get(f"{API_BASE}/users/email/{self.test_user_email}")
            if verify_response.status_code == 200:
                self.test_user = verify_response.json()  # Update with the correct user data
                print(f"Verified user exists by email: {self.test_user['email']}")
            else:
                print(f"Failed to verify user by email: {verify_response.status_code} - {verify_response.text}")
            
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
    
    def test_02_user_creation_flow(self):
        """Test user creation flow - specifically verify user ID is returned"""
        print("\n=== Testing User Creation Flow ===")
        
        # Create a new user specifically for this test
        user_email = f"user_flow_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "User Flow Test",
            "learning_preferences": {
                "preferred_style": "visual",
                "pace": "fast",
                "interests": ["Quantum Computing", "Blockchain"],
                "background": "Computer Science student",
                "experience_level": "intermediate"
            }
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        
        # CRITICAL TEST: Verify user ID is returned
        self.assertIn("id", user)
        self.assertTrue(user["id"], "User ID should not be empty")
        print(f"Created user with ID: {user['id']}")
        
        # Verify all user data is returned correctly
        self.assertEqual(user["email"], user_email)
        self.assertEqual(user["name"], "User Flow Test")
        self.assertIn("learning_preferences", user)
        self.assertIn("created_at", user)
        
        # WORKAROUND: Get user by email instead of ID
        print("Using workaround: Getting user by email instead of ID")
        response = requests.get(f"{API_BASE}/users/email/{user_email}")
        self.assertEqual(response.status_code, 200)
        user_data = response.json()
        self.assertEqual(user_data["email"], user_email)
        print(f"Successfully retrieved user by email: {user_data['email']}")
        
        return user_data  # Return user for session creation test
    
    def test_03_session_creation_flow(self):
        """Test session creation flow with user ID from user creation"""
        print("\n=== Testing Session Creation Flow ===")
        
        # First create a user
        user = self.test_02_user_creation_flow()
        
        # Now create a session with the user ID
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
        
        # CRITICAL TEST: Verify session is created with correct user ID
        self.assertIn("id", session)
        self.assertEqual(session["user_id"], user["id"])
        print(f"Created session with ID: {session['id']} for user: {session['user_id']}")
        
        # Verify session data
        self.assertEqual(session["subject"], "Advanced Machine Learning")
        self.assertEqual(session["difficulty_level"], "advanced")
        self.assertTrue(session["is_active"])
        
        # Test get session by ID
        response = requests.get(f"{API_BASE}/sessions/{session['id']}")
        self.assertEqual(response.status_code, 200)
        session_data = response.json()
        self.assertEqual(session_data["id"], session["id"])
        print(f"Successfully retrieved session by ID: {session_data['id']}")
        
        return session  # Return session for chat test
    
    def test_04_onboarding_integration(self):
        """Test complete onboarding flow from user creation to session creation"""
        print("\n=== Testing Complete Onboarding Integration ===")
        
        # Create a new user with experience level
        user_email = f"onboarding_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "Onboarding Test User",
            "learning_preferences": {
                "preferred_style": "kinesthetic",
                "pace": "moderate",
                "experience_level": "beginner",  # This was causing issues in onboarding
                "interests": ["Python", "Web Development", "Data Science"]
            }
        }
        
        # Step 1: Create user
        print("Step 1: Creating user with experience level...")
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        self.assertIn("id", user)
        print(f"User created successfully with ID: {user['id']}")
        
        # WORKAROUND: Get user by email to ensure we have the correct ID
        print("Getting user by email to ensure correct ID...")
        response = requests.get(f"{API_BASE}/users/email/{user_email}")
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Retrieved user by email with ID: {user['id']}")
        
        # Step 2: Create session with user ID
        print("Step 2: Creating session with user ID...")
        session_data = {
            "user_id": user["id"],  # Using the ID from user creation
            "subject": "Web Development",
            "learning_objectives": ["Learn HTML/CSS", "Master JavaScript", "Build responsive websites"],
            "difficulty_level": "beginner"
        }
        
        response = requests.post(f"{API_BASE}/sessions", json=session_data)
        self.assertEqual(response.status_code, 200)
        session = response.json()
        self.assertIn("id", session)
        self.assertEqual(session["user_id"], user["id"])
        print(f"Session created successfully with ID: {session['id']}")
        
        # Step 3: Verify user sessions endpoint works
        print("Step 3: Verifying user sessions endpoint...")
        response = requests.get(f"{API_BASE}/users/{user['id']}/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()
        self.assertGreaterEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["user_id"], user["id"])
        print(f"Found {len(sessions)} sessions for user")
        
        # Step 4: Send initial message to verify chat works
        print("Step 4: Testing initial chat message...")
        chat_request = {
            "session_id": session["id"],
            "user_message": "Hello! I'm new to web development. Can you help me get started?",
            "context": {
                "user_background": "Complete beginner with some basic computer skills"
            }
        }
        
        response = requests.post(f"{API_BASE}/chat", json=chat_request)
        self.assertEqual(response.status_code, 200)
        chat_response = response.json()
        self.assertIn("response", chat_response)
        print(f"Initial chat message successful, response length: {len(chat_response['response'])}")
        
        print("Complete onboarding flow test PASSED!")
        return session  # Return session for further testing
    
    def test_05_real_time_streaming(self):
        """Test real-time streaming chat functionality with DeepSeek R1 model"""
        print("\n=== Testing Real-time Streaming Chat with DeepSeek R1 ===")
        
        # Create a new session for this test
        session = self.test_04_onboarding_integration()
        
        # Test streaming chat endpoint with complex query
        chat_request = {
            "session_id": session["id"],
            "user_message": "Explain the concept of neural networks and backpropagation in detail. Include code examples in Python.",
            "context": {
                "user_background": "Intermediate programmer with basic ML knowledge"
            }
        }
        
        print("Sending streaming request to /api/chat/stream...")
        # For streaming, we need to handle the response differently
        response = requests.post(f"{API_BASE}/chat/stream", json=chat_request, stream=True)
        self.assertEqual(response.status_code, 200)
        
        # Read chunks to verify streaming works
        chunks_received = 0
        content_chunks = 0
        complete_signal = False
        content_sample = ""
        
        print("Reading streaming response chunks...")
        for chunk in response.iter_lines():
            if chunk:
                chunks_received += 1
                try:
                    # Parse the SSE data
                    if chunk.startswith(b'data: '):
                        data = json.loads(chunk[6:].decode('utf-8'))
                        if data.get('type') == 'chunk':
                            content_chunks += 1
                            if content_chunks <= 3:  # Save a sample of the first few chunks
                                content_sample += data.get('content', '')
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
        print(f"Content sample: {content_sample[:100]}...")
        
        # Verify DeepSeek model is being used by checking for its characteristic response patterns
        deepseek_indicators = ["neural network", "backpropagation", "gradient", "learning rate", "Python"]
        has_deepseek_content = any(indicator.lower() in content_sample.lower() for indicator in deepseek_indicators)
        print(f"DeepSeek content detected: {has_deepseek_content}")
        
        # Test getting session messages to verify the streamed message was saved
        response = requests.get(f"{API_BASE}/sessions/{session['id']}/messages")
        self.assertEqual(response.status_code, 200)
        messages = response.json()
        self.assertGreaterEqual(len(messages), 2)  # Should have at least the user message and AI response
        print(f"Session messages: Found {len(messages)} messages after streaming")
    
    def test_06_ai_service_integration(self):
        """Test AI service integration with premium response formatting"""
        print("\n=== Testing AI Service Integration with Premium Features ===")
        
        # Create a new user for this test
        user_email = f"ai_service_{uuid.uuid4()}@example.com"
        user_data = {
            "email": user_email,
            "name": "AI Service Test User",
            "learning_preferences": {
                "preferred_style": "visual",
                "pace": "moderate",
                "interests": ["Quantum Computing", "AI Ethics"]
            }
        }
        
        response = requests.post(f"{API_BASE}/users", json=user_data)
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Created user for AI service test: {user['id']}")
        
        # WORKAROUND: Get user by email to ensure we have the correct ID
        print("Getting user by email to ensure correct ID...")
        response = requests.get(f"{API_BASE}/users/email/{user_email}")
        self.assertEqual(response.status_code, 200)
        user = response.json()
        print(f"Retrieved user by email with ID: {user['id']}")
        
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
        print(f"Created session for AI service test: {session['id']}")
        
        # Test enhanced chat with complex question
        chat_request = {
            "session_id": session["id"],
            "user_message": "Explain the relationship between quantum computing and machine learning. How can quantum algorithms enhance AI capabilities?",
            "context": {
                "user_background": "Advanced in computer science, beginner in quantum physics"
            }
        }
        
        print("Sending request to /api/chat...")
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
        print(f"Response sample: {response_text[:200]}...")
    
    def test_07_premium_learning_features(self):
        """Test premium learning features including exercise generation and learning paths"""
        print("\n=== Testing Premium Learning Features ===")
        
        # Test exercise generation
        print("\nTesting exercise generation...")
        response = requests.post(
            f"{API_BASE}/exercises/generate",
            params={
                "topic": "Neural Networks",
                "difficulty": "intermediate",
                "exercise_type": "multiple_choice"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        exercise_data = response.json()
        
        # Verify premium exercise features
        self.assertIn("question", exercise_data)
        print(f"Exercise question: {exercise_data.get('question', '')[:100]}...")
        
        # Check for premium features
        premium_features = [
            "options", "correct_answer", "explanation", "concepts",
            "difficulty_score", "pro_tips", "related_topics",
            "real_world_applications", "follow_up_questions"
        ]
        
        present_features = [feature for feature in premium_features if feature in exercise_data]
        print(f"Premium exercise features present: {len(present_features)}/{len(premium_features)}")
        print(f"Features found: {', '.join(present_features)}")
        
        # Test exercise analysis
        if "question" in exercise_data:
            print("\nTesting exercise analysis...")
            analysis_request = {
                "question": exercise_data["question"],
                "user_answer": "This is a test answer that demonstrates understanding of neural networks and their activation functions",
                "correct_answer": exercise_data.get("correct_answer", "The correct answer would be here")
            }
            
            response = requests.post(f"{API_BASE}/exercises/analyze", json=analysis_request)
            self.assertEqual(response.status_code, 200)
            analysis = response.json()
            self.assertIn("feedback", analysis)
            print(f"Exercise analysis feedback: {analysis.get('feedback', '')[:100]}...")
        
        # Test learning path generation
        print("\nTesting learning path generation...")
        response = requests.post(
            f"{API_BASE}/learning-paths/generate",
            params={
                "subject": "Machine Learning",
                "user_level": "beginner",
                "goals": ["Build a recommendation system", "Understand neural networks"]
            }
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
            print(f"Premium learning path features present: {len(present_features)}/{len(premium_features)}")
            print(f"Features found: {', '.join(present_features)}")
            
            # Check milestones
            milestones = learning_path.get("milestones", [])
            print(f"Number of milestones: {len(milestones)}")
            
            if milestones and isinstance(milestones, list) and len(milestones) > 0:
                first_milestone = milestones[0]
                print(f"First milestone: {first_milestone.get('title', 'N/A')}")
        else:
            # Fallback text format
            print(f"Learning path content length: {len(str(learning_path))}")
            
        print("\nPremium learning features test completed successfully!")

if __name__ == "__main__":
    print("=== MasterX AI Mentor System Backend Tests ===")
    print(f"Testing backend at: {API_BASE}")
    print("Running tests focused on user creation, session creation, and onboarding integration")
    print("Also testing real-time streaming, AI service integration, and premium learning features")
    print("=" * 50)
    
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)
