/**
 * Comprehensive Session Management for MasterX Quantum Intelligence Platform
 * 
 * Handles user sessions, state persistence, real-time updates, and data synchronization
 */

import { API, UserProfile, ChatSession, LearningGoal, SystemMetrics } from './api-services'

// ============================================================================
// SESSION STORAGE KEYS
// ============================================================================

const STORAGE_KEYS = {
  AUTH_TOKEN: 'masterx_auth_token',
  REFRESH_TOKEN: 'masterx_refresh_token',
  USER_PROFILE: 'masterx_user',
  ACTIVE_SESSION: 'masterx_active_session',
  CHAT_SESSIONS: 'masterx_chat_sessions',
  LEARNING_GOALS: 'masterx_learning_goals',
  USER_PREFERENCES: 'masterx_preferences',
  SYSTEM_METRICS: 'masterx_system_metrics',
  LAST_ACTIVITY: 'masterx_last_activity',
  SESSION_HISTORY: 'masterx_session_history'
} as const

// ============================================================================
// SESSION STATE INTERFACE
// ============================================================================

export interface SessionState {
  isAuthenticated: boolean
  user: UserProfile | null
  activeSession: ChatSession | null
  chatSessions: ChatSession[]
  learningGoals: LearningGoal[]
  systemMetrics: SystemMetrics | null
  preferences: UserPreferences
  lastActivity: string
  connectionStatus: 'connected' | 'disconnected' | 'reconnecting'
}

export interface UserPreferences {
  theme: 'dark' | 'light' | 'auto'
  language: string
  notifications: {
    email: boolean
    push: boolean
    desktop: boolean
  }
  ai: {
    defaultProvider: 'auto' | 'groq' | 'gemini'
    defaultTaskType: 'general' | 'reasoning' | 'coding' | 'creative'
    streamingEnabled: boolean
  }
  dashboard: {
    layout: 'default' | 'compact' | 'detailed'
    refreshInterval: number
    showSystemMetrics: boolean
  }
}

// ============================================================================
// SESSION MANAGER CLASS
// ============================================================================

export class SessionManager {
  private static instance: SessionManager
  private state: SessionState
  private listeners: Set<(state: SessionState) => void> = new Set()
  private activityTimer: NodeJS.Timeout | null = null
  private syncTimer: NodeJS.Timeout | null = null

  private constructor() {
    this.state = this.getInitialState()
    this.startActivityTracking()
    this.startPeriodicSync()
  }

  static getInstance(): SessionManager {
    if (!SessionManager.instance) {
      SessionManager.instance = new SessionManager()
    }
    return SessionManager.instance
  }

  // ============================================================================
  // STATE MANAGEMENT
  // ============================================================================

  private getInitialState(): SessionState {
    const defaultPreferences: UserPreferences = {
      theme: 'dark',
      language: 'en',
      notifications: {
        email: true,
        push: true,
        desktop: true
      },
      ai: {
        defaultProvider: 'auto',
        defaultTaskType: 'general',
        streamingEnabled: true
      },
      dashboard: {
        layout: 'default',
        refreshInterval: 30000,
        showSystemMetrics: true
      }
    }

    return {
      isAuthenticated: !!this.getStoredData(STORAGE_KEYS.AUTH_TOKEN),
      user: this.getStoredData(STORAGE_KEYS.USER_PROFILE),
      activeSession: this.getStoredData(STORAGE_KEYS.ACTIVE_SESSION),
      chatSessions: this.getStoredData(STORAGE_KEYS.CHAT_SESSIONS) || [],
      learningGoals: this.getStoredData(STORAGE_KEYS.LEARNING_GOALS) || [],
      systemMetrics: this.getStoredData(STORAGE_KEYS.SYSTEM_METRICS),
      preferences: { ...defaultPreferences, ...this.getStoredData(STORAGE_KEYS.USER_PREFERENCES) },
      lastActivity: this.getStoredData(STORAGE_KEYS.LAST_ACTIVITY) || new Date().toISOString(),
      connectionStatus: 'disconnected'
    }
  }

  getState(): SessionState {
    return { ...this.state }
  }

  subscribe(listener: (state: SessionState) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private updateState(updates: Partial<SessionState>): void {
    this.state = { ...this.state, ...updates }
    this.persistState()
    this.notifyListeners()
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => {
      try {
        listener(this.getState())
      } catch (error) {
        console.error('Error in session state listener:', error)
      }
    })
  }

  // ============================================================================
  // AUTHENTICATION MANAGEMENT
  // ============================================================================

  async login(email: string, password: string, rememberMe: boolean = false): Promise<void> {
    try {
      const response = await API.Auth.login({ email, password, remember_me: rememberMe })
      
      this.setStoredData(STORAGE_KEYS.AUTH_TOKEN, response.access_token)
      this.setStoredData(STORAGE_KEYS.REFRESH_TOKEN, response.refresh_token)
      this.setStoredData(STORAGE_KEYS.USER_PROFILE, response.user_info)
      
      this.updateState({
        isAuthenticated: true,
        user: response.user_info,
        connectionStatus: 'connected'
      })

      // Start real-time updates
      API.RealTime.startRealTimeUpdates()
      
      // Load user data
      await this.loadUserData()
      
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }

  async logout(): Promise<void> {
    try {
      await API.Auth.logout()
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      this.clearSession()
    }
  }

  private clearSession(): void {
    // Clear stored data
    Object.values(STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key)
    })

    // Stop real-time updates
    API.RealTime.stopRealTimeUpdates()

    // Reset state
    this.updateState({
      isAuthenticated: false,
      user: null,
      activeSession: null,
      chatSessions: [],
      learningGoals: [],
      systemMetrics: null,
      connectionStatus: 'disconnected'
    })
  }

  async refreshSession(): Promise<void> {
    try {
      const response = await API.Auth.refreshToken()
      this.setStoredData(STORAGE_KEYS.AUTH_TOKEN, response.access_token)
      this.setStoredData(STORAGE_KEYS.USER_PROFILE, response.user_info)
      
      this.updateState({
        isAuthenticated: true,
        user: response.user_info
      })
    } catch (error) {
      console.error('Session refresh failed:', error)
      this.clearSession()
      throw error
    }
  }

  // ============================================================================
  // DATA MANAGEMENT
  // ============================================================================

  async loadUserData(): Promise<void> {
    try {
      // Load chat sessions
      const chatSessions = await API.Chat.getSessions()
      this.updateState({ chatSessions })
      this.setStoredData(STORAGE_KEYS.CHAT_SESSIONS, chatSessions)

      // Load learning goals
      const learningGoals = await API.Learning.getGoals()
      this.updateState({ learningGoals })
      this.setStoredData(STORAGE_KEYS.LEARNING_GOALS, learningGoals)

      // Load system metrics
      const systemMetrics = await API.Analytics.getSystemMetrics()
      this.updateState({ systemMetrics })
      this.setStoredData(STORAGE_KEYS.SYSTEM_METRICS, systemMetrics)

    } catch (error) {
      console.error('Failed to load user data:', error)
    }
  }

  async createChatSession(): Promise<ChatSession> {
    try {
      const newSession = await API.Chat.createSession()
      const session = await API.Chat.getSession(newSession.session_id)
      
      const updatedSessions = [session, ...this.state.chatSessions]
      this.updateState({ 
        chatSessions: updatedSessions,
        activeSession: session
      })
      
      this.setStoredData(STORAGE_KEYS.CHAT_SESSIONS, updatedSessions)
      this.setStoredData(STORAGE_KEYS.ACTIVE_SESSION, session)
      
      return session
    } catch (error) {
      console.error('Failed to create chat session:', error)
      throw error
    }
  }

  setActiveSession(session: ChatSession): void {
    this.updateState({ activeSession: session })
    this.setStoredData(STORAGE_KEYS.ACTIVE_SESSION, session)
  }

  updatePreferences(preferences: Partial<UserPreferences>): void {
    const updatedPreferences = { ...this.state.preferences, ...preferences }
    this.updateState({ preferences: updatedPreferences })
    this.setStoredData(STORAGE_KEYS.USER_PREFERENCES, updatedPreferences)
  }

  // ============================================================================
  // ACTIVITY TRACKING
  // ============================================================================

  private startActivityTracking(): void {
    // Check if we're in browser environment
    if (typeof window === 'undefined') return

    this.updateActivity()

    // Track user activity
    const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart']
    events.forEach(event => {
      document.addEventListener(event, this.updateActivity.bind(this), true)
    })
  }

  private updateActivity(): void {
    const now = new Date().toISOString()
    this.updateState({ lastActivity: now })
    this.setStoredData(STORAGE_KEYS.LAST_ACTIVITY, now)

    // Reset activity timer
    if (this.activityTimer) {
      clearTimeout(this.activityTimer)
    }

    // Auto-logout after 24 hours of inactivity
    this.activityTimer = setTimeout(() => {
      if (this.state.isAuthenticated) {
        console.log('Auto-logout due to inactivity')
        this.logout()
      }
    }, 24 * 60 * 60 * 1000) // 24 hours
  }

  // ============================================================================
  // PERIODIC SYNC
  // ============================================================================

  private startPeriodicSync(): void {
    this.syncTimer = setInterval(async () => {
      if (this.state.isAuthenticated) {
        try {
          await this.loadUserData()
        } catch (error) {
          console.error('Periodic sync failed:', error)
        }
      }
    }, this.state.preferences.dashboard.refreshInterval)
  }

  // ============================================================================
  // STORAGE UTILITIES
  // ============================================================================

  private getStoredData<T>(key: string): T | null {
    try {
      // Check if we're in browser environment
      if (typeof window === 'undefined') return null

      const data = localStorage.getItem(key)
      return data ? JSON.parse(data) : null
    } catch {
      return null
    }
  }

  private setStoredData(key: string, data: any): void {
    try {
      // Check if we're in browser environment
      if (typeof window === 'undefined') return

      localStorage.setItem(key, JSON.stringify(data))
    } catch (error) {
      console.error('Failed to store data:', error)
    }
  }

  private persistState(): void {
    // Persist critical state data
    if (this.state.user) {
      this.setStoredData(STORAGE_KEYS.USER_PROFILE, this.state.user)
    }
    if (this.state.activeSession) {
      this.setStoredData(STORAGE_KEYS.ACTIVE_SESSION, this.state.activeSession)
    }
    this.setStoredData(STORAGE_KEYS.CHAT_SESSIONS, this.state.chatSessions)
    this.setStoredData(STORAGE_KEYS.LEARNING_GOALS, this.state.learningGoals)
    this.setStoredData(STORAGE_KEYS.USER_PREFERENCES, this.state.preferences)
    this.setStoredData(STORAGE_KEYS.LAST_ACTIVITY, this.state.lastActivity)
  }

  // ============================================================================
  // CLEANUP
  // ============================================================================

  destroy(): void {
    if (this.activityTimer) {
      clearTimeout(this.activityTimer)
    }
    if (this.syncTimer) {
      clearInterval(this.syncTimer)
    }
    this.listeners.clear()
    API.RealTime.stopRealTimeUpdates()
  }
}

// Export singleton instance
export const sessionManager = SessionManager.getInstance()
