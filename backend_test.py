#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE MASTERX QUANTUM INTELLIGENCE SYSTEM TESTING
Ultra-Enterprise V6.0 API Testing Suite

Testing the entire 130+ file MasterX backend system through API endpoints
to validate quantum intelligence, personalization, and learning effectiveness.
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

class MasterXQuantumTester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Test user profile for Sarah (16, first-time learner, visual learning, nervous about algebra)
        self.test_user = {
            "user_id": "sarah_test_16",
            "profile": {
                "age": 16,
                "learning_style": "visual",
                "subject_anxiety": ["algebra"],
                "experience_level": "beginner",
                "preferences": {
                    "explanation_style": "step_by_step",
                    "examples": "visual_diagrams",
                    "encouragement": "high"
                }
            }
        }
        
        print(f"🚀 Initializing MasterX Quantum Intelligence Tester")
        print(f"📡 API Base URL: {self.api_base}")
        print(f"👤 Test User: Sarah (16, visual learner, algebra anxiety)")
        print("=" * 80)

    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test result with comprehensive details"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            "test_name": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results.append(result)
        
        print(f"{status} | {test_name}")
        if details.get("response_time_ms"):
            print(f"    ⏱️  Response Time: {details['response_time_ms']:.2f}ms")
        if details.get("status_code"):
            print(f"    📊 Status Code: {details['status_code']}")
        if not success and details.get("error"):
            print(f"    🚨 Error: {details['error']}")
        print()

    def test_health_endpoint(self) -> bool:
        """Test system health endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.api_base}/health")
            response_time = (time.time() - start_time) * 1000
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_data": response.json() if success else response.text[:200]
            }
            
            self.log_test_result("System Health Check", success, details)
            return success
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            details = {
                "error": str(e),
                "response_time_ms": response_time
            }
            self.log_test_result("System Health Check", False, details)
            return False

    def test_quantum_system_status(self) -> bool:
        """Test quantum intelligence system status"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.api_base}/quantum/system/status")
            response_time = (time.time() - start_time) * 1000
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "response_data": response.json() if success else response.text[:200]
            }
            
            self.log_test_result("Quantum System Status", success, details)
            return success
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            details = {
                "error": str(e),
                "response_time_ms": response_time
            }
            self.log_test_result("Quantum System Status", False, details)
            return False

    def test_prometheus_metrics(self) -> bool:
        """Test Prometheus metrics endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.api_base}/metrics/prometheus")
            response_time = (time.time() - start_time) * 1000
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "content_type": response.headers.get("content-type", ""),
                "response_size": len(response.text)
            }
            
            self.log_test_result("Prometheus Metrics", success, details)
            return success
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            details = {
                "error": str(e),
                "response_time_ms": response_time
            }
            self.log_test_result("Prometheus Metrics", False, details)
            return False

    def test_user_profile_retrieval(self) -> bool:
        """Test user profile retrieval"""
        start_time = time.time()
        user_id = self.test_user["user_id"]
        
        try:
            response = self.session.get(f"{self.api_base}/quantum/user/{user_id}/profile")
            response_time = (time.time() - start_time) * 1000
            
            success = response.status_code in [200, 404]  # 404 is acceptable for new user
            details = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "user_id": user_id
            }
            
            if response.status_code == 200:
                details["profile_data"] = response.json()
            elif response.status_code == 404:
                details["note"] = "User profile not found (expected for new user)"
            else:
                details["error_response"] = response.text[:200]
            
            self.log_test_result("User Profile Retrieval", success, details)
            return success
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            details = {
                "error": str(e),
                "response_time_ms": response_time,
                "user_id": user_id
            }
            self.log_test_result("User Profile Retrieval", False, details)
            return False

    def test_quantum_message_processing(self, message: str, task_type: str = "general") -> Dict[str, Any]:
        """Test quantum message processing with comprehensive analysis"""
        start_time = time.time()
        
        payload = {
            "user_id": self.test_user["user_id"],
            "message": message,
            "session_id": f"test_session_{int(time.time())}",
            "task_type": task_type,
            "priority": "balanced",
            "initial_context": self.test_user["profile"],
            "enable_caching": True,
            "max_response_time_ms": 30000,  # 30 second timeout for AI processing
            "enable_streaming": False
        }
        
        try:
            response = self.session.post(
                f"{self.api_base}/quantum/message",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response_time = (time.time() - start_time) * 1000
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "message": message[:50] + "..." if len(message) > 50 else message,
                "task_type": task_type
            }
            
            if success:
                try:
                    response_data = response.json()
                    details["response_structure"] = {
                        "has_response": "response" in response_data,
                        "has_conversation": "conversation" in response_data,
                        "has_analytics": "analytics" in response_data,
                        "has_quantum_metrics": "quantum_metrics" in response_data,
                        "has_performance": "performance" in response_data,
                        "has_recommendations": "recommendations" in response_data
                    }
                    
                    if "performance" in response_data:
                        perf = response_data["performance"]
                        details["performance_metrics"] = {
                            "processing_time_ms": perf.get("total_processing_time_ms"),
                            "target_achieved": perf.get("target_achieved"),
                            "ultra_target_achieved": perf.get("ultra_target_achieved"),
                            "performance_tier": perf.get("performance_tier")
                        }
                    
                    details["ai_response_preview"] = str(response_data.get("response", {}))[:100]
                    
                except json.JSONDecodeError:
                    details["response_data"] = response.text[:200]
            else:
                try:
                    error_data = response.json()
                    details["error_details"] = error_data
                except:
                    details["error_response"] = response.text[:200]
            
            test_name = f"Quantum Message Processing ({task_type})"
            self.log_test_result(test_name, success, details)
            
            return {
                "success": success,
                "response_time_ms": response_time,
                "details": details,
                "response_data": response.json() if success else None
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            details = {
                "error": str(e),
                "response_time_ms": response_time,
                "message": message[:50] + "..." if len(message) > 50 else message,
                "task_type": task_type
            }
            
            test_name = f"Quantum Message Processing ({task_type})"
            self.log_test_result(test_name, False, details)
            
            return {
                "success": False,
                "response_time_ms": response_time,
                "details": details,
                "error": str(e)
            }

    def run_learning_journey_simulation(self):
        """Simulate Sarah's learning journey through various scenarios"""
        print("🎓 LEARNING JOURNEY SIMULATION - Sarah's Algebra Adventure")
        print("=" * 60)
        
        learning_scenarios = [
            {
                "message": "Hi! I'm Sarah and I'm 16. I'm really nervous about starting algebra. Can you help me understand what algebra is in a simple way?",
                "task_type": "emotional_support",
                "description": "Initial introduction with anxiety"
            },
            {
                "message": "Can you show me a visual example of what a variable is? I learn better with pictures and diagrams.",
                "task_type": "beginner_concepts", 
                "description": "Visual learning request"
            },
            {
                "message": "I still don't understand why we use letters like 'x' in math. It's confusing me.",
                "task_type": "complex_explanation",
                "description": "Conceptual confusion"
            },
            {
                "message": "Can you give me a really easy algebra problem to try? Something that won't make me feel stupid.",
                "task_type": "personalized_learning",
                "description": "Confidence building request"
            },
            {
                "message": "I think I'm getting it! Can you give me another problem that's just a little bit harder?",
                "task_type": "personalized_learning", 
                "description": "Progressive difficulty"
            }
        ]
        
        journey_results = []
        
        for i, scenario in enumerate(learning_scenarios, 1):
            print(f"📚 Scenario {i}: {scenario['description']}")
            print(f"💬 Message: {scenario['message'][:60]}...")
            
            result = self.test_quantum_message_processing(
                scenario["message"], 
                scenario["task_type"]
            )
            
            journey_results.append({
                "scenario": scenario,
                "result": result
            })
            
            # Brief pause between requests to simulate real usage
            time.sleep(1)
        
        return journey_results

    def test_basic_frontend_endpoint(self) -> bool:
        """Test the basic endpoint that frontend calls"""
        start_time = time.time()
        
        try:
            # Test the root API endpoint that frontend calls
            response = self.session.get(f"{self.api_base}/")
            response_time = (time.time() - start_time) * 1000
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "endpoint": "/api/",
                "note": "Frontend integration endpoint"
            }
            
            if success:
                try:
                    details["response_data"] = response.json()
                except:
                    details["response_text"] = response.text[:200]
            else:
                details["error_response"] = response.text[:200]
            
            self.log_test_result("Frontend Integration Endpoint", success, details)
            return success
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            details = {
                "error": str(e),
                "response_time_ms": response_time,
                "endpoint": "/api/"
            }
            self.log_test_result("Frontend Integration Endpoint", False, details)
            return False

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Categorize results
        passed_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        # Calculate performance metrics
        response_times = [r["details"].get("response_time_ms", 0) for r in self.test_results if r["details"].get("response_time_ms")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # System health analysis
        health_tests = [r for r in self.test_results if "health" in r["test_name"].lower() or "status" in r["test_name"].lower()]
        core_functionality_tests = [r for r in self.test_results if "quantum message" in r["test_name"].lower()]
        
        report = {
            "test_summary": {
                "total_tests": self.tests_run,
                "passed": self.tests_passed,
                "failed": self.tests_run - self.tests_passed,
                "success_rate": (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0,
                "avg_response_time_ms": avg_response_time
            },
            "system_health": {
                "health_checks_passed": len([r for r in health_tests if r["success"]]),
                "health_checks_total": len(health_tests),
                "core_functionality_working": len([r for r in core_functionality_tests if r["success"]]) > 0
            },
            "detailed_results": {
                "passed_tests": passed_tests,
                "failed_tests": failed_tests
            },
            "recommendations": [],
            "working_components": [],
            "failing_components": []
        }
        
        # Generate recommendations based on results
        if len(failed_tests) > len(passed_tests):
            report["recommendations"].append("CRITICAL: More than 50% of functionality is broken - requires immediate attention")
        
        for test in failed_tests:
            if "quantum message" in test["test_name"].lower():
                report["recommendations"].append("Core quantum intelligence processing is failing - check method name mismatches")
            elif "health" in test["test_name"].lower():
                report["recommendations"].append("System health endpoints failing - check service initialization")
        
        # Identify working vs failing components
        for test in passed_tests:
            report["working_components"].append(test["test_name"])
        
        for test in failed_tests:
            report["failing_components"].append({
                "component": test["test_name"],
                "error": test["details"].get("error", "Unknown error")
            })
        
        return report

    def run_comprehensive_test_suite(self):
        """Run the complete test suite"""
        print("🧪 STARTING COMPREHENSIVE MASTERX QUANTUM INTELLIGENCE TESTING")
        print("=" * 80)
        
        # Phase 1: System Validation
        print("📋 PHASE 1: SYSTEM VALIDATION")
        print("-" * 40)
        
        self.test_health_endpoint()
        self.test_quantum_system_status()
        self.test_prometheus_metrics()
        self.test_basic_frontend_endpoint()
        
        # Phase 2: Component Testing
        print("📋 PHASE 2: COMPONENT TESTING")
        print("-" * 40)
        
        self.test_user_profile_retrieval()
        
        # Phase 3: Core Functionality Testing
        print("📋 PHASE 3: CORE FUNCTIONALITY TESTING")
        print("-" * 40)
        
        # Test basic quantum message processing
        basic_test = self.test_quantum_message_processing(
            "Hello, can you help me learn?", 
            "general"
        )
        
        # Phase 4: Learning Journey Simulation (only if basic functionality works)
        if basic_test["success"]:
            print("📋 PHASE 4: LEARNING JOURNEY SIMULATION")
            print("-" * 40)
            self.run_learning_journey_simulation()
        else:
            print("⚠️  PHASE 4 SKIPPED: Core functionality not working")
        
        # Generate and display comprehensive report
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        report = self.generate_comprehensive_report()
        
        # Display summary
        summary = report["test_summary"]
        print(f"📈 Test Results: {summary['passed']}/{summary['total_tests']} passed ({summary['success_rate']:.1f}%)")
        print(f"⏱️  Average Response Time: {summary['avg_response_time_ms']:.2f}ms")
        print()
        
        # Display working components
        if report["working_components"]:
            print("✅ WORKING COMPONENTS:")
            for component in report["working_components"]:
                print(f"   • {component}")
            print()
        
        # Display failing components
        if report["failing_components"]:
            print("❌ FAILING COMPONENTS:")
            for component in report["failing_components"]:
                print(f"   • {component['component']}")
                if component.get("error"):
                    print(f"     Error: {component['error']}")
            print()
        
        # Display recommendations
        if report["recommendations"]:
            print("🔧 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
            print()
        
        return report

def main():
    """Main test execution"""
    print("🚀 MasterX Quantum Intelligence System Testing")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tester = MasterXQuantumTester()
    
    try:
        report = tester.run_comprehensive_test_suite()
        
        # Determine exit code based on results
        if report["test_summary"]["success_rate"] < 50:
            print("🚨 CRITICAL: Less than 50% of tests passed - system needs major fixes")
            return 1
        elif report["test_summary"]["passed"] == report["test_summary"]["total_tests"]:
            print("🎉 SUCCESS: All tests passed!")
            return 0
        else:
            print("⚠️  PARTIAL SUCCESS: Some issues found but system partially functional")
            return 0
            
    except Exception as e:
        print(f"💥 TESTING FRAMEWORK ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())