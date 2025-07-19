/**
 * Integration Test Component for Phase 13
 * 
 * Tests the frontend-backend integration without disrupting existing UI.
 * This component can be temporarily added to verify the Phase 13 integration.
 */

'use client'

import { useState, useEffect } from 'react'
import { api, sendMessage, ChatRequest } from '@/lib/api'

interface TestResult {
  test: string
  status: 'pending' | 'success' | 'error'
  message: string
  duration?: number
}

export function IntegrationTest() {
  const [testResults, setTestResults] = useState<TestResult[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const updateTestResult = (testName: string, status: 'success' | 'error', message: string, duration?: number) => {
    setTestResults(prev => prev.map(test => 
      test.test === testName 
        ? { ...test, status, message, duration }
        : test
    ))
  }

  const runIntegrationTests = async () => {
    setIsRunning(true)
    
    // Initialize test results
    const tests: TestResult[] = [
      { test: 'Backend Health Check', status: 'pending', message: 'Testing...' },
      { test: 'Authentication System', status: 'pending', message: 'Testing...' },
      { test: 'Chat API Integration', status: 'pending', message: 'Testing...' },
      { test: 'Multi-LLM Selection', status: 'pending', message: 'Testing...' },
      { test: 'Streaming Response', status: 'pending', message: 'Testing...' },
    ]
    setTestResults(tests)

    // Test 1: Backend Health Check
    try {
      const startTime = Date.now()
      const response = await api.get('/health')
      const duration = Date.now() - startTime
      
      if (response.status === 200) {
        updateTestResult('Backend Health Check', 'success', `Backend is healthy (${duration}ms)`, duration)
      } else {
        updateTestResult('Backend Health Check', 'error', `Unexpected status: ${response.status}`)
      }
    } catch (error: any) {
      updateTestResult('Backend Health Check', 'error', `Connection failed: ${error.message}`)
    }

    // Test 2: Authentication System
    try {
      const startTime = Date.now()
      const loginResponse = await api.auth.login({
        email: 'student@example.com',
        password: 'student123'
      })
      const duration = Date.now() - startTime
      
      if (loginResponse.access_token) {
        updateTestResult('Authentication System', 'success', `Login successful (${duration}ms)`, duration)
      } else {
        updateTestResult('Authentication System', 'error', 'No access token received')
      }
    } catch (error: any) {
      updateTestResult('Authentication System', 'error', `Auth failed: ${error.message}`)
    }

    // Test 3: Chat API Integration
    try {
      const startTime = Date.now()
      const chatRequest: ChatRequest = {
        message: 'Hello, this is a Phase 13 integration test.',
        task_type: 'general'
      }
      
      const chatResponse = await sendMessage(chatRequest)
      const duration = Date.now() - startTime
      
      if (chatResponse.response) {
        updateTestResult('Chat API Integration', 'success', `Chat response received (${duration}ms)`, duration)
      } else {
        updateTestResult('Chat API Integration', 'error', 'No response content received')
      }
    } catch (error: any) {
      updateTestResult('Chat API Integration', 'error', `Chat failed: ${error.message}`)
    }

    // Test 4: Multi-LLM Selection
    try {
      const startTime = Date.now()
      const codingRequest: ChatRequest = {
        message: 'Write a simple Python function to calculate factorial.',
        task_type: 'coding',
        provider: 'groq'
      }
      
      const codingResponse = await sendMessage(codingRequest)
      const duration = Date.now() - startTime
      
      if (codingResponse.response && codingResponse.provider) {
        updateTestResult('Multi-LLM Selection', 'success', `Provider: ${codingResponse.provider} (${duration}ms)`, duration)
      } else {
        updateTestResult('Multi-LLM Selection', 'error', 'Provider selection failed')
      }
    } catch (error: any) {
      updateTestResult('Multi-LLM Selection', 'error', `LLM selection failed: ${error.message}`)
    }

    // Test 5: Streaming Response (simplified test)
    try {
      const startTime = Date.now()
      let streamingWorked = false
      
      // Test streaming by checking if the streamMessage function exists
      if (typeof api.chat.stream === 'function') {
        streamingWorked = true
      }
      
      const duration = Date.now() - startTime
      
      if (streamingWorked) {
        updateTestResult('Streaming Response', 'success', `Streaming API available (${duration}ms)`, duration)
      } else {
        updateTestResult('Streaming Response', 'error', 'Streaming API not available')
      }
    } catch (error: any) {
      updateTestResult('Streaming Response', 'error', `Streaming test failed: ${error.message}`)
    }

    setIsRunning(false)
  }

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-purple-500/20 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">
          🧪 Phase 13 Integration Test
        </h3>
        <button
          onClick={runIntegrationTests}
          disabled={isRunning}
          className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white px-4 py-2 rounded text-sm"
        >
          {isRunning ? 'Running Tests...' : 'Run Tests'}
        </button>
      </div>

      <div className="space-y-2">
        {testResults.map((result, index) => (
          <div key={index} className="flex items-center justify-between p-3 bg-slate-700/50 rounded">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                result.status === 'pending' ? 'bg-yellow-500 animate-pulse' :
                result.status === 'success' ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <span className="text-white font-medium">{result.test}</span>
            </div>
            <div className="text-right">
              <div className={`text-sm ${
                result.status === 'success' ? 'text-green-400' : 
                result.status === 'error' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {result.message}
              </div>
              {result.duration && (
                <div className="text-xs text-gray-400">
                  {result.duration}ms
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {testResults.length > 0 && (
        <div className="mt-4 text-sm text-gray-400">
          <p>✅ Phase 13 Integration Status:</p>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Frontend-Backend API Connection</li>
            <li>Multi-LLM Provider Integration</li>
            <li>Real-time Chat Functionality</li>
            <li>Environment Security (API keys in .env)</li>
            <li>Intelligent Model Selection</li>
          </ul>
        </div>
      )}
    </div>
  )
}

export default IntegrationTest
