/**
 * WebSocket Hook for MasterX Quantum Intelligence Platform
 * 
 * Provides real-time communication capabilities for live chat,
 * learning sessions, and collaborative features.
 */

import { useEffect, useRef, useState, useCallback } from 'react'

interface WebSocketMessage {
  type: string
  data?: any
  timestamp?: string
}

interface UseWebSocketOptions {
  url: string
  token?: string
  reconnectAttempts?: number
  reconnectInterval?: number
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  onMessage?: (message: WebSocketMessage) => void
}

interface WebSocketState {
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  lastMessage: WebSocketMessage | null
}

export const useWebSocket = (options: UseWebSocketOptions) => {
  const {
    url,
    token,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    onOpen,
    onClose,
    onError,
    onMessage,
  } = options

  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastMessage: null,
  })

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectCountRef = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setState(prev => ({ ...prev, isConnecting: true, error: null }))

    try {
      const wsUrl = token ? `${url}?token=${token}` : url
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
        }))
        reconnectCountRef.current = 0
        onOpen?.()
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
        }))
        onClose?.()

        // Attempt to reconnect
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++
          console.log(`Attempting to reconnect (${reconnectCountRef.current}/${reconnectAttempts})`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        } else {
          setState(prev => ({
            ...prev,
            error: 'Max reconnection attempts reached',
          }))
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setState(prev => ({
          ...prev,
          error: 'WebSocket connection error',
          isConnecting: false,
        }))
        onError?.(error)
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setState(prev => ({ ...prev, lastMessage: message }))
          onMessage?.(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setState(prev => ({
        ...prev,
        error: 'Failed to create WebSocket connection',
        isConnecting: false,
      }))
    }
  }, [url, token, reconnectAttempts, reconnectInterval, onOpen, onClose, onError, onMessage])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setState({
      isConnected: false,
      isConnecting: false,
      error: null,
      lastMessage: null,
    })
  }, [])

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
      return true
    } else {
      console.warn('WebSocket is not connected')
      return false
    }
  }, [])

  const sendChatMessage = useCallback((message: string, messageId?: string) => {
    return sendMessage({
      type: 'chat',
      data: {
        message,
        message_id: messageId,
        timestamp: new Date().toISOString(),
      },
    })
  }, [sendMessage])

  const joinLearningSession = useCallback((sessionId: string) => {
    return sendMessage({
      type: 'join_session',
      data: { session_id: sessionId },
    })
  }, [sendMessage])

  const leaveLearningSession = useCallback((sessionId: string) => {
    return sendMessage({
      type: 'leave_session',
      data: { session_id: sessionId },
    })
  }, [sendMessage])

  const ping = useCallback(() => {
    return sendMessage({ type: 'ping' })
  }, [sendMessage])

  // Auto-connect on mount
  useEffect(() => {
    connect()

    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Ping interval to keep connection alive
  useEffect(() => {
    if (state.isConnected) {
      const pingInterval = setInterval(() => {
        ping()
      }, 30000) // Ping every 30 seconds

      return () => clearInterval(pingInterval)
    }
  }, [state.isConnected, ping])

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    sendChatMessage,
    joinLearningSession,
    leaveLearningSession,
    ping,
  }
}

export default useWebSocket
