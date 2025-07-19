/**
 * Server-Sent Events Hook for MasterX Quantum Intelligence Platform
 * 
 * Provides real-time streaming capabilities for live updates,
 * progress tracking, and notifications.
 */

import { useEffect, useRef, useState, useCallback } from 'react'

interface SSEEvent {
  type: string
  data: any
  timestamp?: string
}

interface UseSSEOptions {
  url: string
  token?: string
  eventTypes?: string[]
  onEvent?: (event: SSEEvent) => void
  onError?: (error: Event) => void
  onOpen?: () => void
  onClose?: () => void
  autoReconnect?: boolean
  reconnectInterval?: number
}

interface SSEState {
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  lastEvent: SSEEvent | null
}

export const useServerSentEvents = (options: UseSSEOptions) => {
  const {
    url,
    token,
    eventTypes = ['notification', 'learning_update', 'progress_update'],
    onEvent,
    onError,
    onOpen,
    onClose,
    autoReconnect = true,
    reconnectInterval = 5000,
  } = options

  const [state, setState] = useState<SSEState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastEvent: null,
  })

  const eventSourceRef = useRef<EventSource | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (eventSourceRef.current?.readyState === EventSource.OPEN) {
      return
    }

    setState(prev => ({ ...prev, isConnecting: true, error: null }))

    try {
      // Build URL with query parameters
      const urlParams = new URLSearchParams()
      if (eventTypes.length > 0) {
        urlParams.append('event_types', eventTypes.join(','))
      }

      const sseUrl = `${url}?${urlParams.toString()}`
      
      // Create EventSource with authorization header (if token provided)
      const eventSource = new EventSource(sseUrl)

      // Note: EventSource doesn't support custom headers directly
      // The token should be passed as a query parameter or handled server-side
      // For now, we'll include it in the URL if provided
      if (token) {
        // This would need server-side support for token in query params
        console.log('Token authentication for SSE needs server-side implementation')
      }

      eventSource.onopen = () => {
        console.log('SSE connected')
        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
        }))
        onOpen?.()
      }

      eventSource.onerror = (error) => {
        console.error('SSE error:', error)
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
          error: 'SSE connection error',
        }))
        onError?.(error)

        // Auto-reconnect if enabled
        if (autoReconnect) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      eventSource.onmessage = (event) => {
        try {
          const eventData: SSEEvent = JSON.parse(event.data)
          setState(prev => ({ ...prev, lastEvent: eventData }))
          onEvent?.(eventData)
        } catch (error) {
          console.error('Failed to parse SSE event:', error)
        }
      }

      // Handle specific event types
      eventTypes.forEach(eventType => {
        eventSource.addEventListener(eventType, (event) => {
          try {
            const eventData: SSEEvent = {
              type: eventType,
              data: JSON.parse(event.data),
              timestamp: new Date().toISOString(),
            }
            setState(prev => ({ ...prev, lastEvent: eventData }))
            onEvent?.(eventData)
          } catch (error) {
            console.error(`Failed to parse ${eventType} event:`, error)
          }
        })
      })

      eventSourceRef.current = eventSource
    } catch (error) {
      console.error('Failed to create SSE connection:', error)
      setState(prev => ({
        ...prev,
        error: 'Failed to create SSE connection',
        isConnecting: false,
      }))
    }
  }, [url, token, eventTypes, onEvent, onError, onOpen, autoReconnect, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }

    setState({
      isConnected: false,
      isConnecting: false,
      error: null,
      lastEvent: null,
    })

    onClose?.()
  }, [onClose])

  // Auto-connect on mount
  useEffect(() => {
    connect()

    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    ...state,
    connect,
    disconnect,
  }
}

// Specialized hooks for different types of SSE streams

export const useProgressStream = (token?: string) => {
  const sseUrl = `${process.env.NEXT_PUBLIC_SSE_URL}/progress`
  
  return useServerSentEvents({
    url: sseUrl,
    token,
    eventTypes: ['progress_update', 'achievement', 'milestone'],
  })
}

export const useNotificationStream = (token?: string) => {
  const sseUrl = `${process.env.NEXT_PUBLIC_SSE_URL}/notifications`
  
  return useServerSentEvents({
    url: sseUrl,
    token,
    eventTypes: ['notification', 'alert', 'reminder'],
  })
}

export const useEventStream = (eventTypes: string[], token?: string) => {
  const sseUrl = `${process.env.NEXT_PUBLIC_SSE_URL}/events`
  
  return useServerSentEvents({
    url: sseUrl,
    token,
    eventTypes,
  })
}

export default useServerSentEvents
