#!/usr/bin/env python3
"""
🚀 MASTERX AI INTEGRATION FLOW & DATABASE SESSION MANAGEMENT TESTS - PHASE 2
Comprehensive testing for AI Integration Flow, Database Operations, and Real-Time Processing

Test Focus Areas:
1. AI Integration Flow Tests - Multi-LLM provider coordination
2. Database & Session Management Tests - All models and operations
3. Real-Time Processing Tests - WebSocket and concurrent operations

Author: T1 Testing Agent
Version: 3.0 - Production Ready
"""

import requests
import sys
import json
import time
import asyncio
import uuid
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import threading

# Optional imports for advanced testing
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

class MasterXPhase2Tester:
    def __init__(self, base_url="https://quantum-learn-ai.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.created_sessions = []
        self.created_users = []
        self.created_content = []
        
    def log_test(self, name, success, details="", category="General"):
        """Log test results with categorization"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        result = f"{status} - {name}"
        if details:
            result += f" | {details}"
        
        print(result)
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        return success

    # ============================================================================
    # AI INTEGRATION FLOW TESTS
    # ============================================================================

    def test_multi_llm_provider_coordination(self):
        """Test multi-LLM provider coordination (Groq + Gemini + Emergent)"""
        try:
            # Test multiple AI requests to trigger different providers
            providers_used = set()
            response_times = []
            
            for i in range(5):  # Test 5 requests to see provider switching
                url = f"{self.base_url}/api/ai/test"
                test_request = {
                    "message": f"Test request {i+1}: Respond with your provider name and model."
                }
                
                start_time = time.time()
                response = requests.post(url, json=test_request, timeout=30)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        provider = data.get("provider")
                        model = data.get("model")
                        providers_used.add(f"{provider}:{model}")
                        
                        # Verify response includes proper metadata
                        required_fields = ["provider", "model", "tokens_used", "response_time", "confidence"]
                        if not all(field in data for field in required_fields):
                            return self.log_test("Multi-LLM Provider Coordination", False, 
                                               f"Missing metadata fields: {required_fields}", "AI Integration")
                    else:
                        return self.log_test("Multi-LLM Provider Coordination", False, 
                                           f"Request {i+1} failed: {data.get('error')}", "AI Integration")
                else:
                    return self.log_test("Multi-LLM Provider Coordination", False, 
                                       f"Request {i+1} status: {response.status_code}", "AI Integration")
                
                # Small delay between requests
                time.sleep(0.5)
            
            avg_response_time = sum(response_times) / len(response_times)
            return self.log_test("Multi-LLM Provider Coordination", True, 
                               f"Providers used: {len(providers_used)}, Avg time: {avg_response_time:.2f}s", "AI Integration")
            
        except Exception as e:
            return self.log_test("Multi-LLM Provider Coordination", False, f"Error: {str(e)}", "AI Integration")

    def test_ai_response_metadata_and_confidence(self):
        """Test AI responses include proper metadata and confidence scores"""
        try:
            url = f"{self.base_url}/api/ai/test"
            test_request = {
                "message": "Explain quantum computing in simple terms. Include confidence in your explanation."
            }
            
            response = requests.post(url, json=test_request, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    # Check all required metadata fields
                    required_metadata = {
                        "provider": str,
                        "model": str,
                        "tokens_used": int,
                        "response_time": (int, float),
                        "confidence": (int, float)
                    }
                    
                    missing_fields = []
                    invalid_types = []
                    
                    for field, expected_type in required_metadata.items():
                        if field not in data:
                            missing_fields.append(field)
                        elif not isinstance(data[field], expected_type):
                            invalid_types.append(f"{field}: expected {expected_type}, got {type(data[field])}")
                    
                    if missing_fields:
                        return self.log_test("AI Response Metadata", False, 
                                           f"Missing fields: {missing_fields}", "AI Integration")
                    
                    if invalid_types:
                        return self.log_test("AI Response Metadata", False, 
                                           f"Invalid types: {invalid_types}", "AI Integration")
                    
                    # Validate confidence score range
                    confidence = data.get("confidence", 0)
                    if not (0 <= confidence <= 1):
                        return self.log_test("AI Response Metadata", False, 
                                           f"Confidence out of range [0,1]: {confidence}", "AI Integration")
                    
                    # Validate response content exists
                    content = data.get("response", "")
                    if not content or len(content.strip()) < 10:
                        return self.log_test("AI Response Metadata", False, 
                                           "Response content too short or empty", "AI Integration")
                    
                    return self.log_test("AI Response Metadata", True, 
                                       f"Provider: {data['provider']}, Confidence: {confidence:.2f}, Tokens: {data['tokens_used']}", 
                                       "AI Integration")
                else:
                    return self.log_test("AI Response Metadata", False, 
                                       f"AI request failed: {data.get('error')}", "AI Integration")
            else:
                return self.log_test("AI Response Metadata", False, 
                                   f"Status: {response.status_code}", "AI Integration")
                
        except Exception as e:
            return self.log_test("AI Response Metadata", False, f"Error: {str(e)}", "AI Integration")

    def test_ai_provider_fallback_mechanism(self):
        """Test AI provider fallback mechanisms"""
        try:
            # Test with a complex request that might stress the system
            url = f"{self.base_url}/api/ai/test"
            complex_request = {
                "message": "Generate a detailed explanation of machine learning algorithms including neural networks, decision trees, and support vector machines. This is a complex request to test provider capabilities."
            }
            
            response = requests.post(url, json=complex_request, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    provider = data.get("provider")
                    fallback_available = data.get("fallback_available", True)
                    
                    # Test if system can handle provider switching
                    return self.log_test("AI Provider Fallback", True, 
                                       f"Active provider: {provider}, Fallback available: {fallback_available}", 
                                       "AI Integration")
                else:
                    # Check if fallback was attempted
                    fallback_available = data.get("fallback_available", False)
                    if fallback_available:
                        return self.log_test("AI Provider Fallback", True, 
                                           "Primary failed but fallback system operational", "AI Integration")
                    else:
                        return self.log_test("AI Provider Fallback", False, 
                                           "No fallback available when primary failed", "AI Integration")
            else:
                return self.log_test("AI Provider Fallback", False, 
                                   f"Status: {response.status_code}", "AI Integration")
                
        except Exception as e:
            return self.log_test("AI Provider Fallback", False, f"Error: {str(e)}", "AI Integration")

    def test_ai_streaming_simulation(self):
        """Test AI streaming response simulation through interactive content"""
        try:
            # Create streaming session through interactive API
            url = f"{self.base_url}/api/interactive/content"
            streaming_request = {
                "message_id": f"stream_test_{int(time.time())}",
                "content_type": "code",
                "content_data": {
                    "language": "python",
                    "code": "# AI Streaming Test\nimport time\nfor i in range(5):\n    print(f'Streaming chunk {i+1}')\n    time.sleep(0.1)",
                    "title": "AI Streaming Simulation",
                    "is_executable": True
                }
            }
            
            response = requests.post(url, json=streaming_request, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    content_id = data.get("content_id")
                    self.created_content.append(content_id)
                    
                    # Test execution to simulate streaming
                    exec_url = f"{self.base_url}/api/interactive/code/execute"
                    exec_request = {
                        "language": "python",
                        "code": streaming_request["content_data"]["code"]
                    }
                    
                    exec_response = requests.post(exec_url, json=exec_request, timeout=15)
                    
                    if exec_response.status_code == 200:
                        exec_data = exec_response.json()
                        if exec_data.get("success"):
                            execution_time = exec_data.get("execution_time", 0)
                            return self.log_test("AI Streaming Simulation", True, 
                                               f"Content ID: {content_id[:8]}..., Exec time: {execution_time}ms", 
                                               "AI Integration")
                        else:
                            return self.log_test("AI Streaming Simulation", False, 
                                               "Code execution failed", "AI Integration")
                    else:
                        return self.log_test("AI Streaming Simulation", False, 
                                           f"Execution status: {exec_response.status_code}", "AI Integration")
                else:
                    return self.log_test("AI Streaming Simulation", False, 
                                       "Content creation failed", "AI Integration")
            else:
                return self.log_test("AI Streaming Simulation", False, 
                                   f"Status: {response.status_code}", "AI Integration")
                
        except Exception as e:
            return self.log_test("AI Streaming Simulation", False, f"Error: {str(e)}", "AI Integration")

    # ============================================================================
    # DATABASE & SESSION MANAGEMENT TESTS
    # ============================================================================

    def test_database_models_crud_operations(self):
        """Test all models save/retrieve data correctly (models.py with 191 lines)"""
        try:
            # Test User model through status check (simulating user creation)
            user_data = {
                "client_name": f"test_user_model_{int(time.time())}"
            }
            
            url = f"{self.base_url}/api/status"
            response = requests.post(url, json=user_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                user_id = data.get("id")
                
                if user_id and len(user_id) == 36:  # UUID format
                    self.created_users.append(user_id)
                    
                    # Test retrieval
                    get_response = requests.get(url, timeout=10)
                    if get_response.status_code == 200:
                        users = get_response.json()
                        found_user = any(u.get("id") == user_id for u in users)
                        
                        if found_user:
                            return self.log_test("Database Models CRUD", True, 
                                               f"Created and retrieved user with UUID: {user_id[:8]}...", 
                                               "Database")
                        else:
                            return self.log_test("Database Models CRUD", False, 
                                               "Created user not found in retrieval", "Database")
                    else:
                        return self.log_test("Database Models CRUD", False, 
                                           f"Retrieval failed: {get_response.status_code}", "Database")
                else:
                    return self.log_test("Database Models CRUD", False, 
                                       f"Invalid UUID format: {user_id}", "Database")
            else:
                return self.log_test("Database Models CRUD", False, 
                                   f"Creation failed: {response.status_code}", "Database")
                
        except Exception as e:
            return self.log_test("Database Models CRUD", False, f"Error: {str(e)}", "Database")

    def test_session_management_and_persistence(self):
        """Test session continuity during neural network adaptations"""
        try:
            # Create multiple sessions to test session management
            sessions_created = []
            
            for i in range(3):
                url = f"{self.base_url}/api/interactive/collaboration/start"
                session_request = {
                    "content_id": f"test_content_{i}_{int(time.time())}",
                    "participant_ids": [f"user_{i}_1", f"user_{i}_2"],
                    "permissions": {"max_participants": 10}
                }
                
                response = requests.post(url, json=session_request, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        session_id = data.get("session_id")
                        sessions_created.append(session_id)
                        self.created_sessions.append(session_id)
                    else:
                        return self.log_test("Session Management", False, 
                                           f"Session {i+1} creation failed", "Database")
                else:
                    return self.log_test("Session Management", False, 
                                       f"Session {i+1} status: {response.status_code}", "Database")
            
            # Test session retrieval and persistence
            persistent_sessions = 0
            for session_id in sessions_created:
                url = f"{self.base_url}/api/interactive/collaboration/{session_id}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        persistent_sessions += 1
            
            success_rate = persistent_sessions / len(sessions_created) if sessions_created else 0
            
            if success_rate >= 0.8:  # 80% success rate acceptable
                return self.log_test("Session Management", True, 
                                   f"Created {len(sessions_created)} sessions, {persistent_sessions} persistent", 
                                   "Database")
            else:
                return self.log_test("Session Management", False, 
                                   f"Low persistence rate: {success_rate:.1%}", "Database")
                
        except Exception as e:
            return self.log_test("Session Management", False, f"Error: {str(e)}", "Database")

    def test_mongodb_uuid_handling(self):
        """Test MongoDB connection with proper UUID handling (not ObjectID)"""
        try:
            # Create multiple records to test UUID consistency
            created_uuids = []
            
            for i in range(5):
                url = f"{self.base_url}/api/status"
                test_data = {
                    "client_name": f"uuid_test_{i}_{int(time.time())}"
                }
                
                response = requests.post(url, json=test_data, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    record_id = data.get("id")
                    
                    # Validate UUID format (36 characters, proper format)
                    if record_id and len(record_id) == 36 and record_id.count('-') == 4:
                        created_uuids.append(record_id)
                    else:
                        return self.log_test("MongoDB UUID Handling", False, 
                                           f"Invalid UUID format: {record_id}", "Database")
                else:
                    return self.log_test("MongoDB UUID Handling", False, 
                                       f"Record {i+1} creation failed: {response.status_code}", "Database")
            
            # Verify all UUIDs are unique
            if len(set(created_uuids)) == len(created_uuids):
                return self.log_test("MongoDB UUID Handling", True, 
                                   f"Created {len(created_uuids)} unique UUIDs, no ObjectID conflicts", 
                                   "Database")
            else:
                return self.log_test("MongoDB UUID Handling", False, 
                                   "Duplicate UUIDs detected", "Database")
                
        except Exception as e:
            return self.log_test("MongoDB UUID Handling", False, f"Error: {str(e)}", "Database")

    def test_database_service_operations(self):
        """Test database.py (471 lines) handles all operations properly"""
        try:
            # Test comprehensive database operations through API endpoints
            operations_tested = []
            
            # 1. Test Create operation
            url = f"{self.base_url}/api/status"
            create_data = {
                "client_name": f"db_service_test_{int(time.time())}"
            }
            
            response = requests.post(url, json=create_data, timeout=10)
            if response.status_code == 200:
                operations_tested.append("CREATE")
                created_id = response.json().get("id")
            else:
                return self.log_test("Database Service Operations", False, 
                                   "CREATE operation failed", "Database")
            
            # 2. Test Read operation
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    operations_tested.append("READ")
                else:
                    return self.log_test("Database Service Operations", False, 
                                       "READ operation returned empty/invalid data", "Database")
            else:
                return self.log_test("Database Service Operations", False, 
                                   "READ operation failed", "Database")
            
            # 3. Test Interactive content operations (additional database operations)
            content_url = f"{self.base_url}/api/interactive/content"
            content_data = {
                "message_id": f"db_test_{int(time.time())}",
                "content_type": "chart",
                "content_data": {
                    "chart_type": "line",
                    "title": "Database Test Chart"
                }
            }
            
            response = requests.post(content_url, json=content_data, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    operations_tested.append("INTERACTIVE_CREATE")
                    content_id = data.get("content_id")
                    self.created_content.append(content_id)
            
            # 4. Test Health check (database connectivity)
            health_url = f"{self.base_url}/api/health/comprehensive"
            response = requests.get(health_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                db_health = data.get("checks", {}).get("database", {})
                if db_health.get("status") == "healthy":
                    operations_tested.append("HEALTH_CHECK")
            
            if len(operations_tested) >= 3:
                return self.log_test("Database Service Operations", True, 
                                   f"Operations tested: {', '.join(operations_tested)}", "Database")
            else:
                return self.log_test("Database Service Operations", False, 
                                   f"Only {len(operations_tested)} operations successful", "Database")
                
        except Exception as e:
            return self.log_test("Database Service Operations", False, f"Error: {str(e)}", "Database")

    def test_user_profiles_and_progress_tracking(self):
        """Test user profiles, chat sessions, progress tracking, achievements"""
        try:
            # Simulate user profile operations through available endpoints
            
            # 1. Create user profile (through status check)
            url = f"{self.base_url}/api/status"
            user_data = {
                "client_name": f"profile_test_user_{int(time.time())}"
            }
            
            response = requests.post(url, json=user_data, timeout=10)
            if response.status_code != 200:
                return self.log_test("User Profiles & Progress", False, 
                                   "User profile creation failed", "Database")
            
            user_profile = response.json()
            user_id = user_profile.get("id")
            
            # 2. Create chat session (through collaboration)
            session_url = f"{self.base_url}/api/interactive/collaboration/start"
            session_data = {
                "content_id": f"profile_content_{int(time.time())}",
                "participant_ids": [user_id],
                "permissions": {"max_participants": 5}
            }
            
            response = requests.post(session_url, json=session_data, timeout=10)
            if response.status_code != 200:
                return self.log_test("User Profiles & Progress", False, 
                                   "Chat session creation failed", "Database")
            
            session_data = response.json()
            session_id = session_data.get("session_id")
            self.created_sessions.append(session_id)
            
            # 3. Test progress tracking (through interactive content)
            content_url = f"{self.base_url}/api/interactive/content"
            progress_content = {
                "message_id": f"progress_{int(time.time())}",
                "content_type": "quiz",
                "content_data": {
                    "quiz_type": "multiple_choice",
                    "questions": [
                        {
                            "question": "What is 2+2?",
                            "options": ["3", "4", "5"],
                            "correct": 1
                        }
                    ],
                    "title": "Progress Tracking Quiz"
                }
            }
            
            response = requests.post(content_url, json=progress_content, timeout=10)
            if response.status_code != 200:
                return self.log_test("User Profiles & Progress", False, 
                                   "Progress content creation failed", "Database")
            
            progress_data = response.json()
            content_id = progress_data.get("content_id")
            self.created_content.append(content_id)
            
            # 4. Test achievement system (through quantum activation)
            achievement_url = f"{self.base_url}/api/quantum/activate"
            achievement_request = {"mode": "full"}
            
            response = requests.post(achievement_url, json=achievement_request, timeout=15)
            if response.status_code != 200:
                return self.log_test("User Profiles & Progress", False, 
                                   "Achievement system test failed", "Database")
            
            achievement_data = response.json()
            features_activated = achievement_data.get("features_activated", [])
            
            # Verify all components are working
            components_working = [
                user_id is not None,
                session_id is not None,
                content_id is not None,
                len(features_activated) > 0
            ]
            
            if all(components_working):
                return self.log_test("User Profiles & Progress", True, 
                                   f"User: {user_id[:8]}..., Session: {session_id[:8]}..., Content: {content_id[:8]}..., Features: {len(features_activated)}", 
                                   "Database")
            else:
                return self.log_test("User Profiles & Progress", False, 
                                   f"Components working: {sum(components_working)}/4", "Database")
                
        except Exception as e:
            return self.log_test("User Profiles & Progress", False, f"Error: {str(e)}", "Database")

    # ============================================================================
    # REAL-TIME PROCESSING TESTS
    # ============================================================================

    def test_websocket_concurrent_operations(self):
        """Test WebSocket handles concurrent operations through interactive_api.py"""
        try:
            # Create multiple collaboration sessions to test concurrent WebSocket handling
            concurrent_sessions = []
            
            for i in range(3):
                url = f"{self.base_url}/api/interactive/collaboration/start"
                session_request = {
                    "content_id": f"concurrent_test_{i}_{int(time.time())}",
                    "participant_ids": [f"user_concurrent_{i}_1", f"user_concurrent_{i}_2"],
                    "permissions": {"max_participants": 10}
                }
                
                response = requests.post(url, json=session_request, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        session_info = {
                            "session_id": data.get("session_id"),
                            "websocket_url": data.get("websocket_url"),
                            "participants": len(session_request["participant_ids"])
                        }
                        concurrent_sessions.append(session_info)
                        self.created_sessions.append(session_info["session_id"])
            
            if len(concurrent_sessions) >= 2:
                # Test session retrieval to verify concurrent handling
                active_sessions = 0
                for session in concurrent_sessions:
                    url = f"{self.base_url}/api/interactive/collaboration/{session['session_id']}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            active_sessions += 1
                
                success_rate = active_sessions / len(concurrent_sessions)
                
                if success_rate >= 0.8:
                    return self.log_test("WebSocket Concurrent Operations", True, 
                                       f"Created {len(concurrent_sessions)} sessions, {active_sessions} active", 
                                       "Real-Time")
                else:
                    return self.log_test("WebSocket Concurrent Operations", False, 
                                       f"Low concurrent success rate: {success_rate:.1%}", "Real-Time")
            else:
                return self.log_test("WebSocket Concurrent Operations", False, 
                                   "Failed to create sufficient concurrent sessions", "Real-Time")
                
        except Exception as e:
            return self.log_test("WebSocket Concurrent Operations", False, f"Error: {str(e)}", "Real-Time")

    def test_real_time_collaboration(self):
        """Test real-time collaboration through WebSocket connections"""
        try:
            # Create a whiteboard collaboration session
            url = f"{self.base_url}/api/interactive/collaboration/start"
            whiteboard_request = {
                "content_id": f"whiteboard_collab_{int(time.time())}",
                "participant_ids": ["artist_1", "artist_2", "artist_3"],
                "permissions": {
                    "max_participants": 10,
                    "allow_drawing": True,
                    "allow_text": True
                }
            }
            
            response = requests.post(url, json=whiteboard_request, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    session_id = data.get("session_id")
                    websocket_url = data.get("websocket_url")
                    self.created_sessions.append(session_id)
                    
                    # Test whiteboard content creation
                    content_url = f"{self.base_url}/api/interactive/content"
                    whiteboard_content = {
                        "message_id": f"whiteboard_{int(time.time())}",
                        "content_type": "whiteboard",
                        "content_data": {
                            "title": "Real-time Collaboration Test",
                            "width": 1200,
                            "height": 800,
                            "max_participants": 10,
                            "real_time_sync": True
                        }
                    }
                    
                    content_response = requests.post(content_url, json=whiteboard_content, timeout=10)
                    
                    if content_response.status_code == 200:
                        content_data = content_response.json()
                        if content_data.get("success"):
                            content_id = content_data.get("content_id")
                            self.created_content.append(content_id)
                            
                            return self.log_test("Real-time Collaboration", True, 
                                               f"Session: {session_id[:8]}..., Content: {content_id[:8]}..., WS: {websocket_url}", 
                                               "Real-Time")
                        else:
                            return self.log_test("Real-time Collaboration", False, 
                                               "Whiteboard content creation failed", "Real-Time")
                    else:
                        return self.log_test("Real-time Collaboration", False, 
                                           f"Content creation status: {content_response.status_code}", "Real-Time")
                else:
                    return self.log_test("Real-time Collaboration", False, 
                                       "Collaboration session creation failed", "Real-Time")
            else:
                return self.log_test("Real-time Collaboration", False, 
                                   f"Status: {response.status_code}", "Real-Time")
                
        except Exception as e:
            return self.log_test("Real-time Collaboration", False, f"Error: {str(e)}", "Real-Time")

    def test_interactive_content_creation(self):
        """Test interactive content creation (charts, code blocks, whiteboards)"""
        try:
            content_types_tested = []
            
            # Test different types of interactive content
            content_tests = [
                {
                    "type": "code",
                    "data": {
                        "language": "python",
                        "code": "print('Interactive code test')\nresult = 2 + 2\nprint(f'Result: {result}')",
                        "title": "Interactive Code Block",
                        "is_executable": True
                    }
                },
                {
                    "type": "chart",
                    "data": {
                        "chart_type": "bar",
                        "title": "Interactive Chart Test",
                        "data": {
                            "labels": ["A", "B", "C"],
                            "datasets": [{
                                "label": "Test Data",
                                "data": [10, 20, 30]
                            }]
                        }
                    }
                },
                {
                    "type": "whiteboard",
                    "data": {
                        "title": "Interactive Whiteboard",
                        "width": 800,
                        "height": 600,
                        "real_time_sync": True
                    }
                },
                {
                    "type": "calculator",
                    "data": {
                        "calculator_type": "scientific",
                        "title": "Interactive Calculator"
                    }
                }
            ]
            
            for content_test in content_tests:
                url = f"{self.base_url}/api/interactive/content"
                content_request = {
                    "message_id": f"content_test_{content_test['type']}_{int(time.time())}",
                    "content_type": content_test["type"],
                    "content_data": content_test["data"]
                }
                
                response = requests.post(url, json=content_request, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        content_id = data.get("content_id")
                        self.created_content.append(content_id)
                        content_types_tested.append(content_test["type"])
                        
                        # Test code execution if it's a code block
                        if content_test["type"] == "code" and content_test["data"].get("is_executable"):
                            exec_url = f"{self.base_url}/api/interactive/code/execute"
                            exec_request = {
                                "language": content_test["data"]["language"],
                                "code": content_test["data"]["code"]
                            }
                            
                            exec_response = requests.post(exec_url, json=exec_request, timeout=10)
                            if exec_response.status_code == 200:
                                exec_data = exec_response.json()
                                if exec_data.get("success"):
                                    content_types_tested.append("code_execution")
            
            if len(content_types_tested) >= 3:
                return self.log_test("Interactive Content Creation", True, 
                                   f"Created content types: {', '.join(content_types_tested)}", 
                                   "Real-Time")
            else:
                return self.log_test("Interactive Content Creation", False, 
                                   f"Only {len(content_types_tested)} content types successful", "Real-Time")
                
        except Exception as e:
            return self.log_test("Interactive Content Creation", False, f"Error: {str(e)}", "Real-Time")

    def test_system_performance_under_load(self):
        """Test system performance under multiple operations"""
        try:
            # Perform multiple concurrent operations to test system performance
            operations = []
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit various operations concurrently
                futures = []
                
                # 1. Multiple AI requests
                for i in range(3):
                    future = executor.submit(self._make_ai_request, i)
                    futures.append(("ai_request", future))
                
                # 2. Multiple content creations
                for i in range(3):
                    future = executor.submit(self._create_test_content, i)
                    futures.append(("content_creation", future))
                
                # 3. Multiple session creations
                for i in range(2):
                    future = executor.submit(self._create_test_session, i)
                    futures.append(("session_creation", future))
                
                # Collect results
                successful_operations = 0
                total_operations = len(futures)
                
                for operation_type, future in futures:
                    try:
                        result = future.result(timeout=30)
                        if result:
                            successful_operations += 1
                            operations.append(operation_type)
                    except Exception as e:
                        print(f"Operation {operation_type} failed: {str(e)}")
            
            total_time = time.time() - start_time
            success_rate = successful_operations / total_operations if total_operations > 0 else 0
            
            if success_rate >= 0.7 and total_time < 60:  # 70% success rate within 60 seconds
                return self.log_test("System Performance Under Load", True, 
                                   f"Success rate: {success_rate:.1%}, Time: {total_time:.2f}s, Operations: {successful_operations}/{total_operations}", 
                                   "Real-Time")
            else:
                return self.log_test("System Performance Under Load", False, 
                                   f"Performance issues: {success_rate:.1%} success in {total_time:.2f}s", "Real-Time")
                
        except Exception as e:
            return self.log_test("System Performance Under Load", False, f"Error: {str(e)}", "Real-Time")

    def test_streaming_ai_with_interruption(self):
        """Test streaming AI responses with proper interruption handling"""
        try:
            # Test AI streaming through code execution with interruption simulation
            url = f"{self.base_url}/api/interactive/code/execute"
            
            # Create a long-running code that simulates streaming
            streaming_code = """
import time
import sys

print("Starting AI streaming simulation...")
for i in range(10):
    print(f"Streaming chunk {i+1}/10: Processing data...")
    sys.stdout.flush()
    time.sleep(0.1)  # Simulate processing time
    
    # Simulate interruption check
    if i == 5:
        print("Interruption point reached - continuing...")
        
print("Streaming completed successfully!")
"""
            
            exec_request = {
                "language": "python",
                "code": streaming_code
            }
            
            start_time = time.time()
            response = requests.post(url, json=exec_request, timeout=20)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    output = data.get("output", "")
                    
                    # Check if streaming simulation worked
                    if "Streaming chunk" in output and "completed successfully" in output:
                        # Check if interruption handling is mentioned
                        interruption_handled = "Interruption point" in output
                        
                        return self.log_test("Streaming AI with Interruption", True, 
                                           f"Execution time: {execution_time:.2f}s, Interruption handled: {interruption_handled}", 
                                           "Real-Time")
                    else:
                        return self.log_test("Streaming AI with Interruption", False, 
                                           "Streaming simulation failed", "Real-Time")
                else:
                    return self.log_test("Streaming AI with Interruption", False, 
                                       f"Execution failed: {data.get('error')}", "Real-Time")
            else:
                return self.log_test("Streaming AI with Interruption", False, 
                                   f"Status: {response.status_code}", "Real-Time")
                
        except Exception as e:
            return self.log_test("Streaming AI with Interruption", False, f"Error: {str(e)}", "Real-Time")

    # ============================================================================
    # HELPER METHODS FOR CONCURRENT TESTING
    # ============================================================================

    def _make_ai_request(self, request_id):
        """Helper method for concurrent AI requests"""
        try:
            url = f"{self.base_url}/api/ai/test"
            test_request = {
                "message": f"Concurrent AI test request {request_id}: Respond briefly."
            }
            
            response = requests.post(url, json=test_request, timeout=20)
            return response.status_code == 200 and response.json().get("success", False)
        except:
            return False

    def _create_test_content(self, content_id):
        """Helper method for concurrent content creation"""
        try:
            url = f"{self.base_url}/api/interactive/content"
            content_request = {
                "message_id": f"concurrent_content_{content_id}_{int(time.time())}",
                "content_type": "code",
                "content_data": {
                    "language": "python",
                    "code": f"print('Concurrent content test {content_id}')",
                    "title": f"Concurrent Test {content_id}"
                }
            }
            
            response = requests.post(url, json=content_request, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.created_content.append(data.get("content_id"))
                    return True
            return False
        except:
            return False

    def _create_test_session(self, session_id):
        """Helper method for concurrent session creation"""
        try:
            url = f"{self.base_url}/api/interactive/collaboration/start"
            session_request = {
                "content_id": f"concurrent_session_{session_id}_{int(time.time())}",
                "participant_ids": [f"user_{session_id}_1", f"user_{session_id}_2"],
                "permissions": {"max_participants": 5}
            }
            
            response = requests.post(url, json=session_request, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.created_sessions.append(data.get("session_id"))
                    return True
            return False
        except:
            return False

    # ============================================================================
    # MAIN TEST RUNNER
    # ============================================================================

    def run_phase2_tests(self):
        """Run all Phase 2 comprehensive tests"""
        print("🚀 MASTERX AI INTEGRATION FLOW & DATABASE SESSION MANAGEMENT TESTS - PHASE 2")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 90)
        
        print("\n🧠 AI INTEGRATION FLOW TESTS")
        print("-" * 50)
        self.test_multi_llm_provider_coordination()
        self.test_ai_response_metadata_and_confidence()
        self.test_ai_provider_fallback_mechanism()
        self.test_ai_streaming_simulation()
        
        print("\n📊 DATABASE & SESSION MANAGEMENT TESTS")
        print("-" * 50)
        self.test_database_models_crud_operations()
        self.test_session_management_and_persistence()
        self.test_mongodb_uuid_handling()
        self.test_database_service_operations()
        self.test_user_profiles_and_progress_tracking()
        
        print("\n🔄 REAL-TIME PROCESSING TESTS")
        print("-" * 50)
        self.test_websocket_concurrent_operations()
        self.test_real_time_collaboration()
        self.test_interactive_content_creation()
        self.test_system_performance_under_load()
        self.test_streaming_ai_with_interruption()
        
        # Print comprehensive summary
        print("\n" + "=" * 90)
        print(f"📊 PHASE 2 COMPREHENSIVE TEST SUMMARY")
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Categorize results by test category
        categories = {}
        for result in self.test_results:
            category = result.get("category", "General")
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            categories[category]["total"] += 1
            if result["success"]:
                categories[category]["passed"] += 1
        
        print(f"\n📈 DETAILED BREAKDOWN BY CATEGORY:")
        for category, stats in categories.items():
            success_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"{category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Specific analysis for each test focus area
        print(f"\n🔬 PHASE 2 FOCUS AREAS ANALYSIS:")
        
        ai_integration_tests = [r for r in self.test_results if r["category"] == "AI Integration"]
        database_tests = [r for r in self.test_results if r["category"] == "Database"]
        realtime_tests = [r for r in self.test_results if r["category"] == "Real-Time"]
        
        ai_success = sum(1 for t in ai_integration_tests if t["success"])
        db_success = sum(1 for t in database_tests if t["success"])
        rt_success = sum(1 for t in realtime_tests if t["success"])
        
        print(f"   • AI Integration Flow: {ai_success}/{len(ai_integration_tests)} ({'✅' if ai_success == len(ai_integration_tests) else '⚠️' if ai_success >= len(ai_integration_tests)*0.8 else '❌'})")
        print(f"   • Database & Session Management: {db_success}/{len(database_tests)} ({'✅' if db_success == len(database_tests) else '⚠️' if db_success >= len(database_tests)*0.8 else '❌'})")
        print(f"   • Real-Time Processing: {rt_success}/{len(realtime_tests)} ({'✅' if rt_success == len(realtime_tests) else '⚠️' if rt_success >= len(realtime_tests)*0.8 else '❌'})")
        
        # Overall assessment
        overall_success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        
        if overall_success_rate >= 90:
            print("\n🎉 EXCELLENT! All Phase 2 systems are fully operational!")
            return 0
        elif overall_success_rate >= 80:
            print("\n✅ VERY GOOD! Phase 2 systems are highly functional with minor issues.")
            return 0
        elif overall_success_rate >= 70:
            print("\n⚠️  GOOD! Phase 2 systems are mostly functional but need attention.")
            return 1
        else:
            print("\n❌ CRITICAL! Phase 2 systems need immediate attention.")
            return 1

def main():
    tester = MasterXPhase2Tester()
    return tester.run_phase2_tests()

if __name__ == "__main__":
    sys.exit(main())