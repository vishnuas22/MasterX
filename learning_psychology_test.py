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

class LearningPsychologyTests(unittest.TestCase):
    """Tests for Advanced Learning Psychology API endpoints"""
    
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
    
    def test_01_get_learning_psychology_features(self):
        """Test GET /api/learning-psychology/features endpoint"""
        print("\n=== Testing GET /api/learning-psychology/features Endpoint ===")
        
        response = requests.get(f"{API_BASE}/learning-psychology/features")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("features", data)
        self.assertIn("ai_capabilities", data)
        
        # Check features
        features = data["features"]
        self.assertIn("metacognitive_training", features)
        self.assertIn("memory_palace_builder", features)
        self.assertIn("elaborative_interrogation", features)
        self.assertIn("transfer_learning", features)
        
        # Check AI capabilities
        ai_capabilities = data["ai_capabilities"]
        self.assertIn("model", ai_capabilities)
        self.assertIn("real_time_feedback", ai_capabilities)
        
        print(f"Available features: {list(features.keys())}")
        print(f"AI model: {ai_capabilities.get('model', 'Not specified')}")
        
        # Check specific strategies
        metacognitive_strategies = features["metacognitive_training"]["strategies"]
        self.assertGreaterEqual(len(metacognitive_strategies), 1)
        print(f"Metacognitive strategies: {metacognitive_strategies}")
        
        return data
    
    def test_02_start_metacognitive_session(self):
        """Test POST /api/learning-psychology/metacognitive/start endpoint"""
        print("\n=== Testing POST /api/learning-psychology/metacognitive/start Endpoint ===")
        
        if not self.test_user:
            self.skipTest("Test user not available")
        
        # Create metacognitive session request
        session_request = {
            "strategy": "self_questioning",
            "topic": "Neural Networks",
            "level": "intermediate"
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/metacognitive/start",
            json=session_request,
            params={"user_id": self.test_user["id"]}
        )
        
        self.assertEqual(response.status_code, 200)
        session_data = response.json()
        
        # Verify session data
        self.assertIn("session_id", session_data)
        self.assertIn("strategy", session_data)
        self.assertIn("topic", session_data)
        self.assertIn("level", session_data)
        self.assertIn("initial_prompt", session_data)
        self.assertIn("created_at", session_data)
        
        # Check values
        self.assertEqual(session_data["strategy"], "self_questioning")
        self.assertEqual(session_data["topic"], "Neural Networks")
        self.assertEqual(session_data["level"], "intermediate")
        self.assertTrue(len(session_data["initial_prompt"]) > 0)
        
        print(f"Created metacognitive session: {session_data['session_id']}")
        print(f"Strategy: {session_data['strategy']}")
        print(f"Initial prompt sample: {session_data['initial_prompt'][:100]}...")
        
        return session_data
    
    def test_03_create_memory_palace(self):
        """Test POST /api/learning-psychology/memory-palace/create endpoint"""
        print("\n=== Testing POST /api/learning-psychology/memory-palace/create Endpoint ===")
        
        if not self.test_user:
            self.skipTest("Test user not available")
        
        # Create memory palace request
        palace_request = {
            "name": "Python Concepts Palace",
            "palace_type": "library",
            "topic": "Python Programming",
            "information_items": [
                "Variables and data types",
                "Control flow statements",
                "Functions and parameters",
                "Object-oriented programming",
                "Exception handling"
            ]
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/memory-palace/create",
            json=palace_request,
            params={"user_id": self.test_user["id"]}
        )
        
        self.assertEqual(response.status_code, 200)
        palace_data = response.json()
        
        # Verify palace data
        self.assertIn("palace_id", palace_data)
        self.assertIn("name", palace_data)
        self.assertIn("palace_type", palace_data)
        self.assertIn("description", palace_data)
        self.assertIn("rooms", palace_data)
        self.assertIn("pathways", palace_data)
        self.assertIn("information_nodes", palace_data)
        
        # Check values
        self.assertEqual(palace_data["name"], "Python Concepts Palace")
        self.assertEqual(palace_data["palace_type"], "library")
        self.assertTrue(len(palace_data["rooms"]) > 0)
        self.assertTrue(len(palace_data["information_nodes"]) > 0)
        
        print(f"Created memory palace: {palace_data['palace_id']}")
        print(f"Palace type: {palace_data['palace_type']}")
        print(f"Number of rooms: {len(palace_data['rooms'])}")
        print(f"Number of information nodes: {len(palace_data['information_nodes'])}")
        
        # Check if information items are mapped to nodes
        information_items = palace_request["information_items"]
        nodes = palace_data["information_nodes"]
        
        for item in information_items:
            found = False
            for node in nodes:
                if item in node.get("information", ""):
                    found = True
                    break
            self.assertTrue(found, f"Information item '{item}' not found in any node")
        
        return palace_data
    
    def test_04_generate_elaborative_questions(self):
        """Test POST /api/learning-psychology/elaborative-questions/generate endpoint"""
        print("\n=== Testing POST /api/learning-psychology/elaborative-questions/generate Endpoint ===")
        
        # Create elaborative questions request
        questions_request = {
            "topic": "Machine Learning Algorithms",
            "subject_area": "Data Science",
            "difficulty_level": "intermediate",
            "question_count": 3
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/elaborative-questions/generate",
            json=questions_request
        )
        
        self.assertEqual(response.status_code, 200)
        questions_data = response.json()
        
        # Verify questions data
        self.assertIn("questions", questions_data)
        questions = questions_data["questions"]
        
        # Check if we got the requested number of questions
        self.assertGreaterEqual(len(questions), 1)
        
        # Check question structure
        first_question = questions[0]
        self.assertIn("question_id", first_question)
        self.assertIn("question_type", first_question)
        self.assertIn("content", first_question)
        self.assertIn("difficulty_level", first_question)
        self.assertIn("subject_area", first_question)
        self.assertIn("expected_answer_type", first_question)
        self.assertIn("evaluation_criteria", first_question)
        
        print(f"Generated {len(questions)} elaborative questions")
        for i, question in enumerate(questions):
            print(f"Question {i+1} ({question['question_type']}): {question['content'][:100]}...")
        
        return questions_data
    
    def test_05_create_transfer_scenario(self):
        """Test POST /api/learning-psychology/transfer-learning/create-scenario endpoint"""
        print("\n=== Testing POST /api/learning-psychology/transfer-learning/create-scenario Endpoint ===")
        
        # Create transfer scenario request
        scenario_request = {
            "source_domain": "Mathematics",
            "target_domain": "Computer Programming",
            "key_concepts": ["Functions", "Variables", "Iteration", "Conditionals"],
            "transfer_type": "analogical"
        }
        
        response = requests.post(
            f"{API_BASE}/learning-psychology/transfer-learning/create-scenario",
            json=scenario_request
        )
        
        self.assertEqual(response.status_code, 200)
        scenario_data = response.json()
        
        # Verify scenario data
        self.assertIn("scenario_id", scenario_data)
        self.assertIn("source_domain", scenario_data)
        self.assertIn("target_domain", scenario_data)
        self.assertIn("transfer_type", scenario_data)
        self.assertIn("scenario_description", scenario_data)
        self.assertIn("key_concepts", scenario_data)
        self.assertIn("analogy_mapping", scenario_data)
        self.assertIn("exercises", scenario_data)
        
        # Check values
        self.assertEqual(scenario_data["source_domain"], "Mathematics")
        self.assertEqual(scenario_data["target_domain"], "Computer Programming")
        self.assertEqual(scenario_data["transfer_type"], "analogical")
        
        # Check if key concepts are included
        for concept in scenario_request["key_concepts"]:
            self.assertIn(concept, scenario_data["key_concepts"])
        
        # Check if analogy mapping exists
        self.assertTrue(len(scenario_data["analogy_mapping"]) > 0)
        
        # Check if exercises exist
        self.assertTrue(len(scenario_data["exercises"]) > 0)
        
        print(f"Created transfer scenario: {scenario_data['scenario_id']}")
        print(f"Transfer type: {scenario_data['transfer_type']}")
        print(f"Scenario description: {scenario_data['scenario_description'][:100]}...")
        print(f"Number of exercises: {len(scenario_data['exercises'])}")
        
        return scenario_data
    
    def test_06_comprehensive_learning_psychology_flow(self):
        """Test comprehensive learning psychology flow"""
        print("\n=== Testing Comprehensive Learning Psychology Flow ===")
        
        if not self.test_user:
            self.skipTest("Test user not available")
        
        # Step 1: Get available features
        print("\nStep 1: Getting available features...")
        features_data = self.test_01_get_learning_psychology_features()
        
        # Step 2: Start metacognitive session
        print("\nStep 2: Starting metacognitive session...")
        metacognitive_data = self.test_02_start_metacognitive_session()
        session_id = metacognitive_data["session_id"]
        
        # Step 3: Create memory palace
        print("\nStep 3: Creating memory palace...")
        palace_data = self.test_03_create_memory_palace()
        
        # Step 4: Generate elaborative questions
        print("\nStep 4: Generating elaborative questions...")
        questions_data = self.test_04_generate_elaborative_questions()
        
        # Step 5: Create transfer scenario
        print("\nStep 5: Creating transfer scenario...")
        scenario_data = self.test_05_create_transfer_scenario()
        
        # Step 6: Get user progress
        print("\nStep 6: Getting user progress...")
        response = requests.get(f"{API_BASE}/learning-psychology/progress/{self.test_user['id']}")
        self.assertEqual(response.status_code, 200)
        progress_data = response.json()
        
        print(f"User progress: {progress_data}")
        
        print("\nComprehensive learning psychology flow test completed successfully!")

if __name__ == "__main__":
    unittest.main()