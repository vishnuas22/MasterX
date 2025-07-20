import axios, { AxiosResponse } from 'axios'

// Environment configuration
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001'
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || `${BACKEND_URL}/api`

// Create axios instance with enhanced configuration
export const apiService = axios.create({
  baseURL: API_BASE,
  timeout: 30000, // Increased timeout for LLM responses
  headers: {
    'Content-Type': 'application/json',
  },
})

// Authentication token management
let authToken: string | null = null

export const setAuthToken = (token: string) => {
  authToken = token
  // Authentication disabled for development
  // apiService.defaults.headers.common['Authorization'] = `Bearer ${token}`
  // Store in localStorage for persistence
  if (typeof window !== 'undefined') {
    localStorage.setItem('masterx_auth_token', token)
  }
}

export const clearAuthToken = () => {
  authToken = null
  delete apiService.defaults.headers.common['Authorization']
  if (typeof window !== 'undefined') {
    localStorage.removeItem('masterx_auth_token')
  }
}

// Initialize auth token from localStorage
if (typeof window !== 'undefined') {
  const storedToken = localStorage.getItem('masterx_auth_token')
  if (storedToken) {
    setAuthToken(storedToken)
  }
}

// Request interceptor
apiService.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiService.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface LoginRequest {
  email: string
  password: string
  remember_me?: boolean
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  user_info: {
    user_id: string
    email: string
    name: string
    role: string
  }
}

export interface ChatRequest {
  message: string
  session_id?: string
  message_type?: 'text' | 'image' | 'audio' | 'video' | 'file' | 'code'
  context?: Record<string, any>
  stream?: boolean
  task_type?: 'reasoning' | 'coding' | 'creative' | 'fast' | 'multimodal' | 'general'
  provider?: 'groq' | 'gemini' | 'openai' | 'anthropic'
}

export interface ChatResponse {
  session_id: string
  message_id: string
  response: string
  suggestions: string[]
  learning_insights?: Record<string, any>
  personalization_data?: Record<string, any>
  provider?: string
  model?: string
  task_type?: string
  metadata?: {
    response_time?: number
    tokens_used?: number
    task_optimization?: string
    [key: string]: any
  }
}

export interface LearningGoal {
  goal_id: string
  user_id: string
  title: string
  description: string
  subject: string
  target_skills: string[]
  difficulty_level: number
  estimated_duration_hours: number
  status: 'not_started' | 'in_progress' | 'completed' | 'paused'
  progress_percentage: number
  created_at: string
  target_completion_date?: string
}

export interface AnalyticsData {
  user_id: string
  predictions: Array<{
    prediction_type: string
    predicted_outcome: Record<string, any>
    confidence_score: number
    risk_level: string
    recommendations: string[]
  }>
  learning_analytics: Record<string, any>
  performance_insights: Record<string, any>
}

// ============================================================================
// ENHANCED API METHODS
// ============================================================================

// Authentication API
export const login = async (credentials: LoginRequest): Promise<LoginResponse> => {
  const response = await apiService.post('/auth/login', credentials)
  const loginData = response.data

  // Set auth token for future requests
  setAuthToken(loginData.access_token)

  return loginData
}

export const logout = async () => {
  clearAuthToken()
  // Could add server-side logout call here if needed
}

export const refreshToken = async (refreshToken: string) => {
  const response = await apiService.post('/auth/refresh', {
    refresh_token: refreshToken
  })
  const tokenData = response.data
  setAuthToken(tokenData.access_token)
  return tokenData
}

// Chat API
export const sendMessage = async (chatRequest: ChatRequest): Promise<ChatResponse> => {
  const response = await apiService.post('/chat/message', chatRequest)
  return response.data
}

export const streamMessage = async (
  chatRequest: ChatRequest,
  onChunk: (chunk: any) => void,
  onComplete: () => void,
  onError: (error: any) => void
) => {
  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Authorization disabled for development
        // 'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify({ ...chatRequest, stream: true }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body reader available')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        onComplete()
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            onChunk(data)
          } catch (e) {
            console.warn('Failed to parse SSE data:', line)
          }
        }
      }
    }
  } catch (error) {
    onError(error)
  }
}

export const getChatSessions = async () => {
  const response = await apiService.get('/chat/sessions')
  return response.data
}

export const getChatSession = async (sessionId: string) => {
  const response = await apiService.get(`/chat/sessions/${sessionId}`)
  return response.data
}

export const deleteChatSession = async (sessionId: string) => {
  const response = await apiService.delete(`/chat/sessions/${sessionId}`)
  return response.data
}

// Learning API
export const createLearningGoal = async (goalData: Partial<LearningGoal>) => {
  const response = await apiService.post('/learning/goals', goalData)
  return response.data
}

export const getLearningGoals = async (): Promise<LearningGoal[]> => {
  const response = await apiService.get('/learning/goals')
  return response.data
}

export const updateLearningGoal = async (goalId: string, updates: Partial<LearningGoal>) => {
  const response = await apiService.put(`/learning/goals/${goalId}`, updates)
  return response.data
}

export const recordLearningSession = async (sessionData: any) => {
  const response = await apiService.post('/learning/sessions', sessionData)
  return response.data
}

export const getLearningProgress = async (timeRange: string = 'week') => {
  const response = await apiService.post('/learning/progress', { time_range: timeRange })
  return response.data
}

// Analytics API
export const getAnalytics = async (predictionType: string = 'learning_outcome'): Promise<AnalyticsData> => {
  const response = await apiService.post('/analytics/predict', {
    prediction_type: predictionType,
    time_horizon: 'medium_term',
    include_interventions: true
  })
  return response.data
}

export const getAnalyticsDashboard = async () => {
  const response = await apiService.get('/analytics/dashboard')
  return response.data
}

// Content Generation API
export const generateContent = async (contentRequest: any) => {
  const response = await apiService.post('/content/generate', contentRequest)
  return response.data
}

export const getContentLibrary = async (subject?: string, contentType?: string) => {
  const params = new URLSearchParams()
  if (subject) params.append('subject', subject)
  if (contentType) params.append('content_type', contentType)

  const response = await apiService.get(`/content/library?${params.toString()}`)
  return response.data
}

// Assessment API
export const createAssessment = async (assessmentRequest: any) => {
  const response = await apiService.post('/assessment/create', assessmentRequest)
  return response.data
}

export const submitAssessment = async (assessmentId: string, answers: any) => {
  const response = await apiService.post(`/assessment/submit/${assessmentId}`, answers)
  return response.data
}

export const getAssessmentResult = async (assessmentId: string) => {
  const response = await apiService.get(`/assessment/results/${assessmentId}`)
  return response.data
}

// Personalization API
export const getLearningDNAProfile = async () => {
  const response = await apiService.get('/personalization/profile')
  return response.data
}

export const updatePersonalization = async (personalizationData: any) => {
  const response = await apiService.post('/personalization/update', personalizationData)
  return response.data
}

// Legacy API object (for backward compatibility)
export const api = {
  // Basic endpoints
  get: (url: string) => apiService.get(url),
  post: (url: string, data?: any) => apiService.post(url, data),
  put: (url: string, data?: any) => apiService.put(url, data),
  delete: (url: string) => apiService.delete(url),

  // Health check
  healthCheck: () => apiService.get('/health'),

  // Enhanced chat endpoints
  chat: {
    send: sendMessage,
    stream: streamMessage,
    getSessions: getChatSessions,
    getSession: getChatSession,
    deleteSession: deleteChatSession,
  },

  // Authentication
  auth: {
    login,
    logout,
    refreshToken,
  },

  // Learning
  learning: {
    createGoal: createLearningGoal,
    getGoals: getLearningGoals,
    updateGoal: updateLearningGoal,
    recordSession: recordLearningSession,
    getProgress: getLearningProgress,
  },

  // Analytics
  analytics: {
    get: getAnalytics,
    getDashboard: getAnalyticsDashboard,
  },

  // Content
  content: {
    generate: generateContent,
    getLibrary: getContentLibrary,
  },

  // Assessment
  assessment: {
    create: createAssessment,
    submit: submitAssessment,
    getResult: getAssessmentResult,
  },

  // Personalization
  personalization: {
    getProfile: getLearningDNAProfile,
    update: updatePersonalization,
  },
}