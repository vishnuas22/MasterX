#!/usr/bin/env python3
import requests
import json
import uuid
import unittest
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent / 'backend' / '.env')

# Get backend URL from environment or use local default
BACKEND_URL = "http://localhost:8001"
API_BASE = f"{BACKEND_URL}/api"

print(f"Using backend URL: {API_BASE}")

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
        """Test health check endpoints"""
        print("\n=== Testing Health Check Endpoints ===")
        
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
        
        return True
    
    def test_02_database_connection(self):
        """Test database connection through user and session operations"""
        print("\n=== Testing Database Connection ===")
        
        if not self.test_user or not self.test_session:
            self.skipTest("Test user or session not available")
        
        # Test getting user by ID
        response = requests.get(f"{API_BASE}/users/{self.test_user['id']}")
        self.assertEqual(response.status_code, 200)
        user_data = response.json()
        self.assertEqual(user_data["id"], self.test_user["id"])
        print(f"Retrieved user by ID: {user_data['id']}")
        
        # Test getting session by ID
        response = requests.get(f"{API_BASE}/sessions/{self.test_session['id']}")
        self.assertEqual(response.status_code, 200)
        session_data = response.json()
        self.assertEqual(session_data["id"], self.test_session["id"])
        print(f"Retrieved session by ID: {session_data['id']}")
        
        # Test getting user sessions
        response = requests.get(f"{API_BASE}/users/{self.test_user['id']}/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()
        self.assertGreaterEqual(len(sessions), 1)
        print(f"Retrieved {len(sessions)} sessions for user")
        
        return True
    
    def test_03_ai_service_integration(self):
        """Test AI service integration with Groq API"""
        print("\n=== Testing AI Service Integration with Groq API ===")
        
        if not self.test_user or not self.test_session:
            self.skipTest("Test user or session not available")
        
        # Test basic chat endpoint
        chat_request = {
            "session_id": self.test_session["id"],
            "user_message": "What is Python programming?",
            "context": {
                "user_background": "Beginner programmer"
            }
        }
        
        response = requests.post(f"{API_BASE}/chat", json=chat_request)
        self.assertEqual(response.status_code, 200)
        chat_response = response.json()
        self.assertIn("response", chat_response)
        self.assertIn("response_type", chat_response)
        print(f"Chat response type: {chat_response['response_type']}")
        print(f"Chat response length: {len(chat_response['response'])}")
        
        # Test model management endpoint
        response = requests.get(f"{API_BASE}/models/available")
        self.assertEqual(response.status_code, 200)
        models_data = response.json()
        self.assertIn("available_models", models_data)
        self.assertIn("model_capabilities", models_data)
        print(f"Available models: {models_data['available_models']}")
        
        # Check if DeepSeek R1 model is available
        deepseek_available = "deepseek-r1" in models_data["available_models"]
        print(f"DeepSeek R1 model available: {deepseek_available}")
        
        return True
    
    def test_04_metacognitive_training(self):
        """Test metacognitive training endpoint"""
        print("\n=== Testing Metacognitive Training Endpoint ===")
        
        if not self.test_user:
            self.skipTest("Test user not available")
        
        # Start metacognitive session
        metacognitive_request = {
            "strategy": "self_questioning",
            "topic": "Neural Networks",
            "level": "intermediate"
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/metacognitive/start",
            json=metacognitive_request,
            params={"user_id": self.test_user["id"]}
        )
        
        self.assertEqual(response.status_code, 200)
        session = response.json()
        
        # Verify session structure
        self.assertIn("session_id", session)
        self.assertIn("strategy", session)
        self.assertIn("topic", session)
        self.assertIn("level", session)
        self.assertIn("initial_prompt", session)
        
        print(f"Created metacognitive session: {session['session_id']}")
        print(f"Strategy: {session['strategy']}")
        print(f"Initial prompt length: {len(session['initial_prompt'])}")
        
        # Test responding to metacognitive session
        response_data = {
            "user_response": "I already know that neural networks are composed of layers of neurons, with input, hidden, and output layers. I'm uncertain about how backpropagation works in detail and how to optimize hyperparameters. I should be asking myself how each component contributes to the learning process and how I can visualize the information flow."
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/metacognitive/{session['session_id']}/respond",
            json=response_data
        )
        
        self.assertEqual(response.status_code, 200)
        feedback = response.json()
        
        # Verify feedback structure
        self.assertIn("feedback", feedback)
        self.assertIn("analysis", feedback)
        self.assertIn("session_progress", feedback)
        
        print(f"Metacognitive feedback received, length: {len(feedback['feedback'])}")
        
        return session
    
    def test_05_memory_palace_creation(self):
        """Test memory palace creation endpoint"""
        print("\n=== Testing Memory Palace Creation Endpoint ===")
        
        if not self.test_user:
            self.skipTest("Test user not available")
        
        # Create memory palace
        palace_request = {
            "name": "Neural Networks Palace",
            "palace_type": "library",
            "topic": "Neural Networks",
            "information_items": [
                "Neurons are the basic units of neural networks",
                "Activation functions determine the output of neurons",
                "Backpropagation is used to update weights during training",
                "Gradient descent minimizes the loss function",
                "Convolutional layers are specialized for image processing"
            ]
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/memory-palace/create",
            json=palace_request,
            params={"user_id": self.test_user["id"]}
        )
        
        self.assertEqual(response.status_code, 200)
        palace = response.json()
        
        # Verify palace structure
        self.assertIn("palace_id", palace)
        self.assertIn("name", palace)
        self.assertIn("palace_type", palace)
        self.assertIn("description", palace)
        self.assertIn("rooms", palace)
        self.assertIn("pathways", palace)
        self.assertIn("information_nodes", palace)
        
        print(f"Created memory palace: {palace['name']} with ID: {palace['palace_id']}")
        print(f"Palace type: {palace['palace_type']}")
        print(f"Number of rooms: {len(palace['rooms'])}")
        print(f"Number of information nodes: {len(palace['information_nodes'])}")
        
        # Test memory palace practice
        response = requests.post(
            f"{API_BASE}/learning-psychology/memory-palace/{palace['palace_id']}/practice",
            params={"practice_type": "recall"}
        )
        
        self.assertEqual(response.status_code, 200)
        practice = response.json()
        
        # Verify practice structure
        self.assertIn("type", practice)
        self.assertIn("instructions", practice)
        self.assertIn("questions", practice)
        
        print(f"Memory palace practice generated with {len(practice['questions'])} questions")
        
        return palace
    
    def test_06_elaborative_questions(self):
        """Test elaborative questions generation endpoint"""
        print("\n=== Testing Elaborative Questions Generation Endpoint ===")
        
        # Generate elaborative questions
        question_request = {
            "topic": "Neural Networks",
            "subject_area": "Machine Learning",
            "difficulty_level": "intermediate",
            "question_count": 3
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/elaborative-questions/generate",
            json=question_request
        )
        
        self.assertEqual(response.status_code, 200)
        questions_data = response.json()
        
        # Verify questions structure
        self.assertIn("questions", questions_data)
        questions = questions_data["questions"]
        self.assertGreaterEqual(len(questions), 1)
        
        # Check first question
        first_question = questions[0]
        self.assertIn("question_id", first_question)
        self.assertIn("question_type", first_question)
        self.assertIn("content", first_question)
        self.assertIn("difficulty_level", first_question)
        self.assertIn("subject_area", first_question)
        self.assertIn("evaluation_criteria", first_question)
        
        print(f"Generated {len(questions)} elaborative questions")
        print(f"First question: {first_question['content']}")
        print(f"Question type: {first_question['question_type']}")
        
        # Test question evaluation
        if len(questions) > 0:
            evaluation_request = {
                "user_response": "Neural networks learn through backpropagation, which is a method of calculating gradients for each weight in the network. This works by applying the chain rule of calculus to compute how much each weight contributes to the final error. By adjusting weights in the direction that reduces error, the network gradually improves its predictions. This process is fundamental to deep learning because it allows networks to automatically learn complex patterns from data without explicit programming."
            }
            
            response = requests.post(
                f"{API_BASE}/learning-psychology/elaborative-questions/{first_question['question_id']}/evaluate",
                json=evaluation_request
            )
            
            self.assertEqual(response.status_code, 200)
            evaluation = response.json()
            
            # Verify evaluation structure
            self.assertIn("accuracy_score", evaluation)
            self.assertIn("depth_score", evaluation)
            self.assertIn("overall_score", evaluation)
            
            print(f"Response evaluation - Overall score: {evaluation['overall_score']}")
        
        return questions
    
    def test_07_transfer_learning(self):
        """Test transfer learning scenario creation endpoint"""
        print("\n=== Testing Transfer Learning Scenario Creation Endpoint ===")
        
        # Create transfer scenario
        scenario_request = {
            "source_domain": "Neural Networks",
            "target_domain": "Economics",
            "key_concepts": ["Neurons", "Weights", "Activation", "Backpropagation", "Gradient Descent"],
            "transfer_type": "analogical"
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/transfer-learning/create-scenario",
            json=scenario_request
        )
        
        self.assertEqual(response.status_code, 200)
        scenario = response.json()
        
        # Verify scenario structure
        self.assertIn("scenario_id", scenario)
        self.assertIn("source_domain", scenario)
        self.assertIn("target_domain", scenario)
        self.assertIn("transfer_type", scenario)
        self.assertIn("scenario_description", scenario)
        self.assertIn("key_concepts", scenario)
        self.assertIn("analogy_mapping", scenario)
        self.assertIn("exercises", scenario)
        
        print(f"Created transfer scenario: {scenario['scenario_id']}")
        print(f"Transfer type: {scenario['transfer_type']}")
        print(f"Source domain: {scenario['source_domain']}")
        print(f"Target domain: {scenario['target_domain']}")
        print(f"Number of analogy mappings: {len(scenario['analogy_mapping'])}")
        print(f"Number of exercises: {len(scenario['exercises'])}")
        
        return scenario
    
    def test_08_premium_model_management(self):
        """Test premium model management endpoints"""
        print("\n=== Testing Premium Model Management Endpoints ===")
        
        # Test available models endpoint
        response = requests.get(f"{API_BASE}/models/available")
        self.assertEqual(response.status_code, 200)
        models_data = response.json()
        
        # Verify models data structure
        self.assertIn("available_models", models_data)
        self.assertIn("total_calls", models_data)
        self.assertIn("model_capabilities", models_data)
        
        # Check if DeepSeek R1 model is available
        deepseek_available = "deepseek-r1" in models_data["available_models"]
        print(f"Available models: {models_data['available_models']}")
        print(f"DeepSeek R1 model available: {deepseek_available}")
        
        # Test model analytics endpoint
        response = requests.get(f"{API_BASE}/analytics/models")
        self.assertEqual(response.status_code, 200)
        analytics_data = response.json()
        
        # Verify analytics data structure
        self.assertIn("available_models", analytics_data)
        self.assertIn("usage_stats", analytics_data)
        self.assertIn("total_calls", analytics_data)
        
        print(f"Model analytics: {analytics_data}")
        
        return models_data
    
    def test_09_premium_chat_endpoints(self):
        """Test premium chat endpoints"""
        print("\n=== Testing Premium Chat Endpoints ===")
        
        if not self.test_user or not self.test_session:
            self.skipTest("Test user or session not available")
        
        # Test premium chat endpoint
        chat_request = {
            "session_id": self.test_session["id"],
            "user_message": "Explain neural networks in detail",
            "context": {
                "learning_mode": "socratic",
                "user_background": "Intermediate programmer"
            }
        }
        
        response = requests.post(f"{API_BASE}/chat/premium", json=chat_request)
        self.assertEqual(response.status_code, 200)
        premium_response = response.json()
        
        # Verify premium response structure
        self.assertIn("response", premium_response)
        self.assertIn("response_type", premium_response)
        self.assertIn("suggested_actions", premium_response)
        self.assertIn("metadata", premium_response)
        
        # Check if learning mode is applied
        self.assertEqual(premium_response["response_type"], "premium_socratic")
        
        print(f"Premium chat response type: {premium_response['response_type']}")
        print(f"Premium chat response length: {len(premium_response['response'])}")
        
        # Test premium chat with context awareness
        context_chat_request = {
            "session_id": self.test_session["id"],
            "user_message": "How do neural networks learn?",
            "context": {
                "learning_mode": "adaptive",
                "user_background": "Intermediate programmer with some ML knowledge"
            }
        }
        
        response = requests.post(f"{API_BASE}/chat/premium-context", json=context_chat_request)
        self.assertEqual(response.status_code, 200)
        context_response = response.json()
        
        # Verify context-aware response
        self.assertIn("response", context_response)
        self.assertIn("metadata", context_response)
        
        # Check for context awareness metadata
        metadata = context_response["metadata"]
        self.assertIn("context_awareness", metadata)
        
        print(f"Premium context-aware chat response length: {len(context_response['response'])}")
        print(f"Context awareness metadata: {metadata['context_awareness']}")
        
        return premium_response
    
    def test_10_streaming_endpoints(self):
        """Test streaming chat endpoints"""
        print("\n=== Testing Streaming Chat Endpoints ===")
        
        if not self.test_user or not self.test_session:
            self.skipTest("Test user or session not available")
        
        # Test basic streaming endpoint
        chat_request = {
            "session_id": self.test_session["id"],
            "user_message": "Explain object-oriented programming briefly",
            "context": {
                "user_background": "Beginner programmer"
            }
        }
        
        print("Testing basic streaming endpoint...")
        response = requests.post(f"{API_BASE}/chat/stream", json=chat_request, stream=True)
        self.assertEqual(response.status_code, 200)
        
        # Read a few chunks to verify streaming works
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
                except Exception as e:
                    print(f"Error parsing chunk: {e}")
                
                # Only read a reasonable number of chunks for testing
                if chunks_received >= 10 or complete_signal:
                    break
        
        self.assertGreater(content_chunks, 0, "Should receive content chunks")
        print(f"Basic streaming: Received {chunks_received} total chunks, {content_chunks} content chunks")
        print(f"Complete signal received: {complete_signal}")
        
        # Test premium streaming endpoint
        premium_chat_request = {
            "session_id": self.test_session["id"],
            "user_message": "What are the principles of clean code?",
            "context": {
                "learning_mode": "adaptive",
                "user_background": "Intermediate programmer"
            }
        }
        
        print("Testing premium streaming endpoint...")
        response = requests.post(f"{API_BASE}/chat/premium/stream", json=premium_chat_request, stream=True)
        self.assertEqual(response.status_code, 200)
        
        # Read a few chunks to verify premium streaming works
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
                except Exception as e:
                    print(f"Error parsing chunk: {e}")
                
                # Only read a reasonable number of chunks for testing
                if chunks_received >= 10 or complete_signal:
                    break
        
        self.assertGreater(content_chunks, 0, "Should receive content chunks")
        print(f"Premium streaming: Received {chunks_received} total chunks, {content_chunks} content chunks")
        print(f"Complete signal received: {complete_signal}")
        
        return True
    
    def test_11_live_learning_sessions(self):
        """Test live learning sessions endpoints"""
        print("\n=== Testing Live Learning Sessions Endpoints ===")
        
        if not self.test_user:
            self.skipTest("Test user not available")
        
        # Create live session
        session_request = {
            "user_id": self.test_user["id"],
            "session_type": "voice_interaction",
            "title": "Python Programming Live Session",
            "duration_minutes": 60,
            "features": {
                "voice_enabled": True,
                "screen_sharing": False,
                "whiteboard": True
            }
        }
        
        response = requests.post(f"{API_BASE}/live-sessions/create", json=session_request)
        self.assertEqual(response.status_code, 200)
        live_session = response.json()
        
        # Verify live session structure
        self.assertIn("id", live_session)
        self.assertIn("user_id", live_session)
        self.assertIn("session_type", live_session)
        self.assertIn("title", live_session)
        self.assertIn("features", live_session)
        
        print(f"Created live session: {live_session['id']}")
        print(f"Session type: {live_session['session_type']}")
        print(f"Session title: {live_session['title']}")
        
        # Test voice interaction endpoint
        voice_request = {
            "user_id": self.test_user["id"],
            "audio_data": "Mock audio data for testing"  # In production, this would be actual audio data
        }
        
        response = requests.post(f"{API_BASE}/live-sessions/{live_session['id']}/voice", json=voice_request)
        self.assertEqual(response.status_code, 200)
        voice_result = response.json()
        
        print(f"Voice interaction result: {voice_result}")
        
        # Test session status endpoint
        response = requests.get(f"{API_BASE}/live-sessions/{live_session['id']}/status")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        
        # Verify status structure
        self.assertIn("session_id", status)
        self.assertIn("status", status)
        self.assertIn("duration", status)
        
        print(f"Session status: {status['status']}")
        
        # Test ending the session
        response = requests.post(f"{API_BASE}/live-sessions/{live_session['id']}/end")
        self.assertEqual(response.status_code, 200)
        end_result = response.json()
        
        print(f"Session end result: {end_result}")
        
        return live_session
    
    def test_12_advanced_context_awareness(self):
        """Test advanced context awareness endpoints"""
        print("\n=== Testing Advanced Context Awareness Endpoints ===")
        
        if not self.test_user or not self.test_session:
            self.skipTest("Test user or session not available")
        
        # Test context analysis endpoint
        context_request = {
            "user_id": self.test_user["id"],
            "session_id": self.test_session["id"],
            "message": "I'm finding neural networks quite challenging to understand, especially backpropagation. Could you explain it in a simpler way?",
            "conversation_context": [
                {"role": "user", "content": "What are neural networks?"},
                {"role": "assistant", "content": "Neural networks are computational models inspired by the human brain..."},
                {"role": "user", "content": "How do they learn?"},
                {"role": "assistant", "content": "They learn through a process called backpropagation..."}
            ]
        }
        
        response = requests.post(f"{API_BASE}/context/analyze", json=context_request)
        self.assertEqual(response.status_code, 200)
        context_analysis = response.json()
        
        # Verify context analysis structure
        self.assertIn("context_state", context_analysis)
        self.assertIn("recommendations", context_analysis)
        self.assertIn("emotional_insights", context_analysis)
        
        # Check context state and recommendations
        context_state = context_analysis["context_state"]
        recommendations = context_analysis["recommendations"]
        emotional_insights = context_analysis["emotional_insights"]
        
        print(f"Context analysis - Emotional state: {emotional_insights['state']}")
        print(f"Recommended response complexity: {recommendations['response_complexity']}")
        print(f"Recommended explanation depth: {recommendations['explanation_depth']}")
        
        # Test memory insights endpoint
        response = requests.get(f"{API_BASE}/context/{self.test_user['id']}/memory")
        self.assertEqual(response.status_code, 200)
        memory_insights = response.json()
        
        print(f"Memory insights: {memory_insights}")
        
        return context_analysis

if __name__ == "__main__":
    print("=== MasterX AI Mentor System Backend Tests ===")
    print(f"Testing backend at: {API_BASE}")
    print("Running tests for:")
    print("1. Health Check and Database Connection")
    print("2. Learning Psychology Features")
    print("3. AI Service Integration")
    print("4. Premium Model Management")
    print("5. Live Learning Sessions")
    print("=" * 50)
    
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)