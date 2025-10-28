/**
 * useWebSocket Hook - React hook for WebSocket connection
 * 
 * Features:
 * - Auto-connect on mount
 * - Auto-disconnect on unmount
 * - Connection state tracking
 * - Event listener management
 * - Subscribe/unsubscribe pattern
 * 
 * Usage:
 * ```tsx
 * const { isConnected, subscribe } = useWebSocket();
 * 
 * useEffect(() => {
 *   if (!isConnected) return;
 *   
 *   const unsubscribe = subscribe('emotion_update', (data) => {
 *     console.log('Emotion:', data);
 *   });
 *   
 *   return unsubscribe;
 * }, [isConnected]);
 * ```
 */

import { useEffect, useState, useCallback } from 'react';
import socketClient from '@/services/websocket/socket.client';
import { 
  initializeSocketHandlers, 
  cleanupSocketHandlers 
} from '@/services/websocket/socket.handlers';

export interface UseWebSocketReturn {
  /**
   * Connection status
   */
  isConnected: boolean;
  
  /**
   * Emit event to server
   */
  emit: (event: string, data: any) => void;
  
  /**
   * Subscribe to event (returns unsubscribe function)
   */
  subscribe: (event: string, callback: (data: any) => void) => () => void;
  
  /**
   * Manually reconnect
   */
  reconnect: () => void;
  
  /**
   * Manually disconnect
   */
  disconnect: () => void;
}

export const useWebSocket = (): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect on mount
    socketClient.connect();
    
    // Initialize event handlers
    initializeSocketHandlers();

    // Listen to connection state
    const checkConnection = () => {
      setIsConnected(socketClient.isConnected());
    };

    const interval = setInterval(checkConnection, 1000);
    checkConnection(); // Initial check

    // Cleanup on unmount
    return () => {
      clearInterval(interval);
      cleanupSocketHandlers();
      socketClient.disconnect();
    };
  }, []);

  /**
   * Subscribe to event (returns unsubscribe function)
   */
  const subscribe = useCallback((event: string, callback: (data: any) => void): (() => void) => {
    socketClient.on(event, callback);
    
    // Return unsubscribe function
    return () => {
      socketClient.off(event, callback);
    };
  }, []);

  /**
   * Emit event to server
   */
  const emit = useCallback((event: string, data: any) => {
    socketClient.emit(event, data);
  }, []);

  /**
   * Manually reconnect
   */
  const reconnect = useCallback(() => {
    socketClient.reconnect();
  }, []);

  /**
   * Manually disconnect
   */
  const disconnect = useCallback(() => {
    socketClient.disconnect();
  }, []);

  return {
    isConnected,
    emit,
    subscribe,
    reconnect,
    disconnect,
  };
};
