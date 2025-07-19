// MasterX Chat Functionality Test Script
// This script tests the complete chat workflow from frontend to backend

const testChatAPI = async () => {
  console.log('🧪 Testing MasterX Chat API Functionality...\n');

  const baseURL = 'http://localhost:8000/api/v1';
  
  // Test cases for different providers and task types
  const testCases = [
    {
      name: 'Groq Coding Test',
      payload: {
        message: 'Write a simple hello world function in Python',
        task_type: 'coding',
        provider: 'groq'
      }
    },
    {
      name: 'Gemini Reasoning Test',
      payload: {
        message: 'Explain the concept of recursion in programming',
        task_type: 'reasoning',
        provider: 'gemini'
      }
    },
    {
      name: 'Auto-Selection Test',
      payload: {
        message: 'What is machine learning?',
        task_type: 'general',
        provider: ''
      }
    },
    {
      name: 'Creative Task Test',
      payload: {
        message: 'Write a short poem about artificial intelligence',
        task_type: 'creative',
        provider: 'gemini'
      }
    }
  ];

  let passedTests = 0;
  let totalTests = testCases.length;

  for (const testCase of testCases) {
    try {
      console.log(`🔄 Running: ${testCase.name}`);
      
      const startTime = Date.now();
      const response = await fetch(`${baseURL}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(testCase.payload),
      });

      const endTime = Date.now();
      const responseTime = endTime - startTime;

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // Validate response structure
      if (!data.success || !data.response || !data.message_id) {
        throw new Error('Invalid response structure');
      }

      console.log(`✅ ${testCase.name} - PASSED`);
      console.log(`   Response Time: ${responseTime}ms`);
      console.log(`   Provider Used: ${data.provider_used || 'auto-selected'}`);
      console.log(`   Response Length: ${data.response.length} characters`);
      console.log(`   Session ID: ${data.session_id}`);
      console.log('');

      passedTests++;

    } catch (error) {
      console.log(`❌ ${testCase.name} - FAILED`);
      console.log(`   Error: ${error.message}`);
      console.log('');
    }
  }

  // Test health endpoint
  try {
    console.log('🔄 Testing Health Endpoint...');
    const healthResponse = await fetch(`${baseURL.replace('/api/v1', '')}/health`);
    const healthData = await healthResponse.json();
    
    if (healthData.status === 'healthy') {
      console.log('✅ Health Check - PASSED');
      console.log(`   Backend Status: ${healthData.status}`);
      console.log(`   Timestamp: ${healthData.timestamp}`);
    } else {
      throw new Error('Backend not healthy');
    }
  } catch (error) {
    console.log('❌ Health Check - FAILED');
    console.log(`   Error: ${error.message}`);
  }

  console.log('\n📊 Test Results Summary:');
  console.log(`   Passed: ${passedTests}/${totalTests} chat tests`);
  console.log(`   Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
  
  if (passedTests === totalTests) {
    console.log('🎉 All tests passed! MasterX Chat API is fully functional.');
  } else {
    console.log('⚠️  Some tests failed. Please check the errors above.');
  }
};

// Run the tests
testChatAPI().catch(console.error);
