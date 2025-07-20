/**
 * Comprehensive API Services for MasterX Quantum Intelligence Platform
 * 
 * Complete integration layer with backend APIs including authentication,
 * chat, learning, analytics, and real-time features.
 */

import { apiService, setAuthToken, clearAuthToken } from './api'

// ============================================================================
// TYPE DEFINITIONS (matching backend models)
// ============================================================================

export interface UserProfile {
  user_id: string
  email: string
  name: string
  role: 'admin' | 'teacher' | 'student'
  created_at: string
  last_login?: string
  preferences: Record<string, any>
}

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
  user_info: UserProfile
}

export interface ChatMessage {
  message_id: string
  content: string
  message_type: 'user' | 'assistant' | 'system'
  timestamp: string
  metadata?: Record<string, any>
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
  response: string
  session_id: string
  message_id: string
  provider_used: string
  processing_time: number
  metadata: Record<string, any>
  learning_insights?: Record<string, any>
}

export interface ChatSession {
  session_id: string
  user_id: string
  created_at: string
  last_activity: string
  messages: ChatMessage[]
  context: Record<string, any>
  learning_insights: Record<string, any>
}

export interface LearningGoal {
  goal_id: string
  user_id: string
  title: string
  description: string
  status: 'active' | 'completed' | 'paused' | 'archived'
  target_date?: string
  progress_percentage: number
  created_at: string
  updated_at: string
}

export interface LearningSession {
  session_id: string
  user_id: string
  goal_id?: string
  session_type: string
  duration_minutes: number
  achievements: string[]
  insights: Record<string, any>
  created_at: string
}

export interface SystemMetrics {
  cpu_usage: number
  memory_usage: number
  network_activity: number
  active_users: number
  total_sessions: number
  response_time: number
  timestamp: string
}

export interface DashboardData {
  user_stats: {
    total_users: number
    active_users: number
    new_users_today: number
  }
  session_stats: {
    total_sessions: number
    active_sessions: number
    average_duration: number
  }
  system_health: {
    status: 'healthy' | 'warning' | 'error'
    uptime: number
    last_check: string
  }
  recent_activity: Array<{
    type: string
    description: string
    timestamp: string
  }>
}

// ============================================================================
// AUTHENTICATION SERVICE
// ============================================================================

export class AuthService {
  static async login(credentials: LoginRequest): Promise<LoginResponse> {
    try {
      const response = await apiService.post<LoginResponse>('/auth/login', credentials)
      
      // Store token
      setAuthToken(response.data.access_token)
      
      return response.data
    } catch (error: any) {
      console.error('Login failed:', error)
      throw new Error(error.response?.data?.detail || 'Login failed')
    }
  }

  static async logout(): Promise<void> {
    try {
      await apiService.post('/auth/logout')
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      clearAuthToken()
    }
  }

  static async refreshToken(): Promise<LoginResponse> {
    try {
      const response = await apiService.post<LoginResponse>('/auth/refresh')
      setAuthToken(response.data.access_token)
      return response.data
    } catch (error: any) {
      console.error('Token refresh failed:', error)
      clearAuthToken()
      throw new Error('Session expired')
    }
  }

  static async getCurrentUser(): Promise<UserProfile> {
    try {
      const response = await apiService.get<UserProfile>('/auth/me')
      return response.data
    } catch (error: any) {
      console.error('Get current user failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get user info')
    }
  }
}

// ============================================================================
// CHAT SERVICE
// ============================================================================

export class ChatService {
  static async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await apiService.post<ChatResponse>('/chat/message', request)
      return response.data
    } catch (error: any) {
      console.error('Send message failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to send message')
    }
  }

  static async createSession(userId?: string): Promise<{ session_id: string }> {
    try {
      const response = await apiService.post<{ session_id: string }>('/chat/sessions', {
        user_id: userId
      })
      return response.data
    } catch (error: any) {
      console.error('Create session failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to create session')
    }
  }

  static async getSession(sessionId: string): Promise<ChatSession> {
    try {
      const response = await apiService.get<ChatSession>(`/chat/sessions/${sessionId}`)
      return response.data
    } catch (error: any) {
      console.error('Get session failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get session')
    }
  }

  static async getSessions(): Promise<ChatSession[]> {
    try {
      const response = await apiService.get<ChatSession[]>('/chat/sessions')
      return response.data
    } catch (error: any) {
      console.error('Get sessions failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get sessions')
    }
  }

  static createStreamingConnection(request: ChatRequest): EventSource {
    // For development, we'll use a simple approach without authentication
    // Note: EventSource doesn't support POST, so we'll need to modify this approach

    // Create a simple URL with basic parameters  
    const baseUrl = `${process.env.REACT_APP_BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001'}/api`
    const url = `${baseUrl}/chat/stream`

    // For now, let's create a basic EventSource connection
    // In production, this would need proper authentication headers
    const eventSource = new EventSource(url, {
      withCredentials: false  // Disabled for development
    })

    return eventSource
  }
}

// ============================================================================
// LEARNING SERVICE
// ============================================================================

export class LearningService {
  static async getGoals(): Promise<LearningGoal[]> {
    try {
      const response = await apiService.get<LearningGoal[]>('/learning/goals')
      return response.data
    } catch (error: any) {
      console.error('Get goals failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get learning goals')
    }
  }

  static async createGoal(goal: Partial<LearningGoal>): Promise<LearningGoal> {
    try {
      const response = await apiService.post<LearningGoal>('/learning/goals', goal)
      return response.data
    } catch (error: any) {
      console.error('Create goal failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to create learning goal')
    }
  }

  static async updateGoal(goalId: string, updates: Partial<LearningGoal>): Promise<LearningGoal> {
    try {
      const response = await apiService.put<LearningGoal>(`/learning/goals/${goalId}`, updates)
      return response.data
    } catch (error: any) {
      console.error('Update goal failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to update learning goal')
    }
  }

  static async getSessions(): Promise<LearningSession[]> {
    try {
      const response = await apiService.get<LearningSession[]>('/learning/sessions')
      return response.data
    } catch (error: any) {
      console.error('Get sessions failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get learning sessions')
    }
  }
}

// ============================================================================
// ANALYTICS SERVICE
// ============================================================================

export class AnalyticsService {
  static async getDashboardData(): Promise<DashboardData> {
    try {
      const response = await apiService.get<DashboardData>('/analytics/dashboard')
      return response.data
    } catch (error: any) {
      console.error('Get dashboard data failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get dashboard data')
    }
  }

  static async getSystemMetrics(): Promise<SystemMetrics> {
    try {
      const response = await apiService.get<SystemMetrics>('/analytics/metrics')
      return response.data
    } catch (error: any) {
      console.error('Get system metrics failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get system metrics')
    }
  }

  static async getUserAnalytics(userId: string): Promise<any> {
    try {
      const response = await apiService.get(`/analytics/users/${userId}`)
      return response.data
    } catch (error: any) {
      console.error('Get user analytics failed:', error)
      throw new Error(error.response?.data?.detail || 'Failed to get user analytics')
    }
  }
}

// ============================================================================
// SYSTEM SERVICE
// ============================================================================

export class SystemService {
  static async getHealth(): Promise<any> {
    try {
      const response = await apiService.get('/health')
      return response.data
    } catch (error: any) {
      console.error('Health check failed:', error)
      throw new Error('System health check failed')
    }
  }

  static async getStatus(): Promise<any> {
    try {
      const response = await apiService.get('/')
      return response.data
    } catch (error: any) {
      console.error('Status check failed:', error)
      throw new Error('System status check failed')
    }
  }
}

// ============================================================================
// REAL-TIME DATA MANAGER
// ============================================================================

export class RealTimeDataManager {
  private static instance: RealTimeDataManager
  private eventSource: EventSource | null = null
  private subscribers: Map<string, Set<(data: any) => void>> = new Map()

  static getInstance(): RealTimeDataManager {
    if (!RealTimeDataManager.instance) {
      RealTimeDataManager.instance = new RealTimeDataManager()
    }
    return RealTimeDataManager.instance
  }

  subscribe(event: string, callback: (data: any) => void): () => void {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, new Set())
    }
    this.subscribers.get(event)!.add(callback)

    // Return unsubscribe function
    return () => {
      this.subscribers.get(event)?.delete(callback)
    }
  }

  private emit(event: string, data: any): void {
    this.subscribers.get(event)?.forEach(callback => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in event callback:', error)
      }
    })
  }

  startRealTimeUpdates(): void {
    if (this.eventSource) {
      return // Already connected
    }

    const token = localStorage.getItem('masterx_auth_token')
    if (!token) {
      console.warn('No auth token available for real-time updates')
      return
    }

    try {
      this.eventSource = new EventSource(`${apiService.defaults.baseURL}/streaming/events`, {
        withCredentials: true
      })

      this.eventSource.onopen = () => {
        console.log('Real-time connection established')
        this.emit('connection', { status: 'connected' })
      }

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.emit(data.type || 'message', data)
        } catch (error) {
          console.error('Error parsing real-time data:', error)
        }
      }

      this.eventSource.onerror = (error) => {
        console.error('Real-time connection error:', error)
        this.emit('connection', { status: 'error', error })
      }

    } catch (error) {
      console.error('Failed to start real-time updates:', error)
    }
  }

  stopRealTimeUpdates(): void {
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
      this.emit('connection', { status: 'disconnected' })
    }
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

export const formatError = (error: any): string => {
  if (error.response?.data?.detail) {
    return error.response.data.detail
  }
  if (error.message) {
    return error.message
  }
  return 'An unexpected error occurred'
}

export const isAuthenticated = (): boolean => {
  return !!localStorage.getItem('masterx_auth_token')
}

export const getStoredUser = (): UserProfile | null => {
  try {
    const userData = localStorage.getItem('masterx_user')
    return userData ? JSON.parse(userData) : null
  } catch {
    return null
  }
}

// Export all services as a single object for convenience
export const API = {
  Auth: AuthService,
  Chat: ChatService,
  Learning: LearningService,
  Analytics: AnalyticsService,
  System: SystemService,
  RealTime: RealTimeDataManager.getInstance()
}
