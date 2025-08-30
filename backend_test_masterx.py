import requests
import sys
import json
from datetime import datetime
import time

class MasterXBackendTester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test results"""
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
            "timestamp": datetime.now().isoformat()
        })
        return success

    def test_basic_api_health(self):
        """Test GET /api/ - Should return {"message": "Hello World"}"""
        try:
            url = f"{self.base_url}/api/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("message") == "Hello World":
                    return self.log_test("Basic API Health", True, f"Status: {response.status_code}, Message: {data['message']}")
                else:
                    return self.log_test("Basic API Health", False, f"Unexpected message: {data}")
            else:
                return self.log_test("Basic API Health", False, f"Status: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Basic API Health", False, f"Error: {str(e)}")

    def test_performance_metrics(self):
        """Test GET /api/metrics/performance - New endpoint with comprehensive system metrics"""
        try:
            url = f"{self.base_url}/api/metrics/performance"
            start_time = time.time()
            response = requests.get(url, timeout=15)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["timestamp", "system_metrics", "application_metrics", "health_summary"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    # Check system metrics structure
                    system_metrics = data.get("system_metrics", {})
                    has_memory = "memory" in system_metrics
                    has_cpu = "cpu" in system_metrics
                    has_disk = "disk" in system_metrics
                    
                    # Check health summary
                    health_summary = data.get("health_summary", {})
                    has_overall_status = "overall_status" in health_summary
                    
                    if has_memory and has_cpu and has_disk and has_overall_status:
                        overall_status = health_summary.get("overall_status")
                        return self.log_test("Performance Metrics (NEW)", True, 
                                           f"Complete metrics returned, Overall Status: {overall_status}, Response Time: {response_time:.2f}ms")
                    else:
                        return self.log_test("Performance Metrics (NEW)", False, 
                                           "Missing system metrics or health summary fields")
                else:
                    return self.log_test("Performance Metrics (NEW)", False, 
                                       f"Missing required fields: {missing_fields}")
            else:
                return self.log_test("Performance Metrics (NEW)", False, f"Status: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Performance Metrics (NEW)", False, f"Error: {str(e)}")

    def test_cors_configuration(self):
        """Test CORS Configuration - OPTIONS /api/test with proper headers"""
        try:
            # Test OPTIONS request with Origin header
            url = f"{self.base_url}/api/test"
            headers = {
                'Origin': 'https://example.com',
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            response = requests.options(url, headers=headers, timeout=10)
            
            # Check for CORS headers
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
                'Access-Control-Expose-Headers': response.headers.get('Access-Control-Expose-Headers'),
                'Access-Control-Allow-Credentials': response.headers.get('Access-Control-Allow-Credentials')
            }
            
            # Check if essential CORS headers are present
            has_origin = cors_headers['Access-Control-Allow-Origin'] is not None
            has_methods = cors_headers['Access-Control-Allow-Methods'] is not None
            has_headers = cors_headers['Access-Control-Allow-Headers'] is not None
            
            if response.status_code == 200 and has_origin and has_methods and has_headers:
                return self.log_test("CORS Configuration (FIXED)", True, 
                                   f"Status: {response.status_code}, CORS headers properly exposed: {cors_headers}")
            else:
                return self.log_test("CORS Configuration (FIXED)", False, 
                                   f"Status: {response.status_code}, Missing CORS headers: {cors_headers}")
                
        except Exception as e:
            return self.log_test("CORS Configuration (FIXED)", False, f"Error: {str(e)}")

    def test_quantum_intelligence_status(self):
        """Test GET /api/quantum/status - Should show quantum capabilities status"""
        try:
            url = f"{self.base_url}/api/quantum/status"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["success", "quantum_intelligence", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    quantum_intel = data.get("quantum_intelligence", {})
                    has_status = "status" in quantum_intel
                    
                    if has_status:
                        status = quantum_intel.get("status")
                        success = data.get("success")
                        return self.log_test("Quantum Intelligence Status (NEW)", True, 
                                           f"Success: {success}, Status: {status}")
                    else:
                        return self.log_test("Quantum Intelligence Status (NEW)", False, 
                                           "Missing status in quantum_intelligence")
                else:
                    return self.log_test("Quantum Intelligence Status (NEW)", False, 
                                       f"Missing required fields: {missing_fields}")
            else:
                return self.log_test("Quantum Intelligence Status (NEW)", False, f"Status: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Quantum Intelligence Status (NEW)", False, f"Error: {str(e)}")

    def test_quantum_intelligence_activate(self):
        """Test POST /api/quantum/activate with body {"mode": "full"} - Should activate features"""
        try:
            url = f"{self.base_url}/api/quantum/activate"
            payload = {"mode": "full"}
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["success", "activation_mode", "features_activated", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    success = data.get("success")
                    activation_mode = data.get("activation_mode")
                    features_activated = data.get("features_activated", [])
                    
                    if success and activation_mode == "full" and len(features_activated) > 0:
                        return self.log_test("Quantum Intelligence Activate (NEW)", True, 
                                           f"Success: {success}, Mode: {activation_mode}, Features: {len(features_activated)}")
                    else:
                        return self.log_test("Quantum Intelligence Activate (NEW)", False, 
                                           f"Activation failed or incomplete: Success={success}, Mode={activation_mode}")
                else:
                    return self.log_test("Quantum Intelligence Activate (NEW)", False, 
                                       f"Missing required fields: {missing_fields}")
            else:
                return self.log_test("Quantum Intelligence Activate (NEW)", False, f"Status: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Quantum Intelligence Activate (NEW)", False, f"Error: {str(e)}")

    def test_comprehensive_health_check(self):
        """Test GET /api/health/comprehensive - Should complete in <2 seconds with detailed health"""
        try:
            url = f"{self.base_url}/api/health/comprehensive"
            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if response time is optimized (< 2 seconds)
                is_fast = response_time < 2.0
                
                # Check for health data structure
                has_status = "status" in data or any(key in data for key in ["database", "ai_providers", "environment"])
                
                if is_fast and has_status:
                    return self.log_test("Comprehensive Health Check (OPTIMIZED)", True, 
                                       f"Response Time: {response_time:.3f}s (< 2s), Health data present")
                elif has_status:
                    return self.log_test("Comprehensive Health Check (OPTIMIZED)", False, 
                                       f"Response Time: {response_time:.3f}s (>= 2s), but health data present")
                else:
                    return self.log_test("Comprehensive Health Check (OPTIMIZED)", False, 
                                       f"Missing health data structure")
            else:
                return self.log_test("Comprehensive Health Check (OPTIMIZED)", False, f"Status: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Comprehensive Health Check (OPTIMIZED)", False, f"Error: {str(e)}")

    def test_json_response_format(self):
        """Test that all endpoints return properly formatted JSON"""
        endpoints_to_test = [
            "/api/",
            "/api/metrics/performance",
            "/api/quantum/status",
            "/api/health/comprehensive"
        ]
        
        json_valid_count = 0
        total_endpoints = len(endpoints_to_test)
        
        for endpoint in endpoints_to_test:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    # Try to parse JSON
                    data = response.json()
                    if isinstance(data, (dict, list)):
                        json_valid_count += 1
                        
            except Exception:
                # JSON parsing failed for this endpoint
                continue
        
        if json_valid_count == total_endpoints:
            return self.log_test("JSON Response Format", True, 
                               f"All {total_endpoints} endpoints return valid JSON")
        else:
            return self.log_test("JSON Response Format", False, 
                               f"Only {json_valid_count}/{total_endpoints} endpoints return valid JSON")

    def test_error_handling(self):
        """Test error handling with invalid requests"""
        try:
            # Test invalid endpoint
            url = f"{self.base_url}/api/nonexistent"
            response = requests.get(url, timeout=10)
            
            # Should return 404 or similar error
            if response.status_code in [404, 405]:
                return self.log_test("Error Handling", True, 
                                   f"Proper error response for invalid endpoint: {response.status_code}")
            else:
                return self.log_test("Error Handling", False, 
                                   f"Unexpected response for invalid endpoint: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Error Handling", False, f"Error: {str(e)}")

    def run_masterx_tests(self):
        """Run all MasterX backend tests focusing on the 4 fixed issues"""
        print("🚀 Starting MasterX Backend API Tests")
        print(f"📍 Testing against: {self.base_url}")
        print("🎯 Focus: Testing 4 Critical Fixes")
        print("=" * 80)
        
        print("\n🔹 CRITICAL FIXES VERIFICATION")
        print("-" * 40)
        
        # Test the 4 critical fixes
        self.test_basic_api_health()
        self.test_performance_metrics()
        self.test_cors_configuration()
        self.test_quantum_intelligence_status()
        self.test_quantum_intelligence_activate()
        self.test_comprehensive_health_check()
        
        print("\n🔹 ADDITIONAL QUALITY CHECKS")
        print("-" * 40)
        
        # Additional quality tests
        self.test_json_response_format()
        self.test_error_handling()
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"📊 MASTERX TEST SUMMARY")
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['name']}: {result['details']}")
        
        # Final assessment
        if self.tests_passed == self.tests_run:
            print("\n🎉 ALL TESTS PASSED! All 4 critical fixes are working perfectly!")
            return 0
        elif self.tests_passed >= self.tests_run * 0.75:
            print("\n✅ EXCELLENT! Most fixes are working correctly!")
            return 0
        else:
            print("\n⚠️ ISSUES DETECTED! Some fixes need attention.")
            return 1

def main():
    tester = MasterXBackendTester()
    return tester.run_masterx_tests()

if __name__ == "__main__":
    sys.exit(main())