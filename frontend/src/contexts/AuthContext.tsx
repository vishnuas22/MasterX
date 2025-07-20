/**
 * Authentication Context for MasterX Quantum Intelligence Platform
 * 
 * Provides authentication state management, user session handling,
 * and secure token management across the application.
 */

'use client'

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { login as apiLogin, logout as apiLogout, LoginRequest, LoginResponse } from '../lib/api'

interface User {
  user_id: string
  email: string
  name: string
  role: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (credentials: LoginRequest) => Promise<void>
  logout: () => void
  refreshAuth: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  // Development mode: Mock user and token
  const [user, setUser] = useState<User | null>({
    user_id: 'dev_user_001',
    name: 'Developer',
    email: 'dev@masterx.ai',
    role: 'student'
  })
  const [token, setToken] = useState<string | null>('dev_token_123')
  const [isLoading, setIsLoading] = useState(false)  // Set to false for development

  // Development mode: Always authenticated
  const isAuthenticated = !!user && !!token

  // Initialize auth state from localStorage
  useEffect(() => {
    const initializeAuth = () => {
      try {
        const storedToken = localStorage.getItem('masterx_auth_token')
        const storedUser = localStorage.getItem('masterx_user')

        if (storedToken && storedUser) {
          setToken(storedToken)
          setUser(JSON.parse(storedUser))
        }
      } catch (error) {
        console.error('Failed to initialize auth from localStorage:', error)
        // Clear potentially corrupted data
        localStorage.removeItem('masterx_auth_token')
        localStorage.removeItem('masterx_user')
      } finally {
        setIsLoading(false)
      }
    }

    initializeAuth()
  }, [])

  const login = async (credentials: LoginRequest): Promise<void> => {
    try {
      setIsLoading(true)
      
      const response: LoginResponse = await apiLogin(credentials)
      
      const userData: User = {
        user_id: response.user_info.user_id,
        email: response.user_info.email,
        name: response.user_info.name,
        role: response.user_info.role,
      }

      // Update state
      setToken(response.access_token)
      setUser(userData)

      // Store in localStorage
      localStorage.setItem('masterx_auth_token', response.access_token)
      localStorage.setItem('masterx_user', JSON.stringify(userData))
      
      // Store refresh token if provided
      if (response.refresh_token) {
        localStorage.setItem('masterx_refresh_token', response.refresh_token)
      }

      console.log('Login successful:', userData.name)
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    try {
      // Call API logout
      apiLogout()

      // Clear state
      setUser(null)
      setToken(null)

      // Clear localStorage
      localStorage.removeItem('masterx_auth_token')
      localStorage.removeItem('masterx_user')
      localStorage.removeItem('masterx_refresh_token')

      console.log('Logout successful')
    } catch (error) {
      console.error('Logout error:', error)
    }
  }

  const refreshAuth = () => {
    // Force re-initialization from localStorage
    const storedToken = localStorage.getItem('masterx_auth_token')
    const storedUser = localStorage.getItem('masterx_user')

    if (storedToken && storedUser) {
      try {
        setToken(storedToken)
        setUser(JSON.parse(storedUser))
      } catch (error) {
        console.error('Failed to refresh auth:', error)
        logout()
      }
    } else {
      logout()
    }
  }

  // Auto-logout on token expiration (optional enhancement)
  useEffect(() => {
    if (token) {
      try {
        // Validate token format first
        if (!token.includes('.') || token.split('.').length !== 3) {
          console.warn('Invalid token format, clearing token')
          logout()
          return
        }

        // Decode JWT to check expiration (basic implementation)
        const payload = JSON.parse(atob(token.split('.')[1]))

        // Check if payload has required fields
        if (!payload.exp) {
          console.warn('Token missing expiration, treating as valid for session')
          return
        }

        const expirationTime = payload.exp * 1000 // Convert to milliseconds
        const currentTime = Date.now()

        if (expirationTime <= currentTime) {
          console.log('Token expired, logging out')
          logout()
        } else {
          // Set timeout to auto-logout when token expires
          const timeUntilExpiration = expirationTime - currentTime
          const timeoutId = setTimeout(() => {
            console.log('Token expired, auto-logout')
            logout()
          }, timeUntilExpiration)

          return () => clearTimeout(timeoutId)
        }
      } catch (error) {
        console.warn('Failed to decode token, clearing invalid token:', error)
        // If token is invalid, clear it silently
        localStorage.removeItem('masterx_auth_token')
        localStorage.removeItem('masterx_user')
        setToken(null)
        setUser(null)
      }
    }
  }, [token])

  const value: AuthContextType = {
    user,
    token,
    isAuthenticated,
    isLoading,
    login,
    logout,
    refreshAuth,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext
