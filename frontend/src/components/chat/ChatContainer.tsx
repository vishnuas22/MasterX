/**
 * ChatContainer Component - Main Chat Interface
 * 
 * WCAG 2.1 AA Compliant:
 * - Landmark <main> element
 * - Keyboard navigation (Tab, Arrow keys)
 * - Screen reader announcements for new messages
 * - Focus management (auto-focus on input after message)
 * 
 * Performance:
 * - Virtual scrolling for large message lists (>100 messages)
 * - Lazy loading of message history
 * - Optimistic UI updates (instant message display)
 * - Debounced typing indicators
 * 
 * Backend Integration:
 * - POST /api/v1/chat - Send message and get AI response
 * - WebSocket connection for real-time emotion updates
 * - Session persistence in MongoDB
 * - Automatic reconnection on network issues
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useChatStore } from '@/store/chatStore';
import { useEmotionStore } from '@/store/emotionStore';
import { useAuthStore } from '@/store/authStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { joinSession, leaveSession } from '@/services/websocket/socket.handlers';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { EmotionIndicator } from './EmotionIndicator';
import { TypingIndicator } from './TypingIndicator';
import { VoiceButton } from './VoiceButton';
import { cn } from '@/utils/cn';
import { toast } from '@/components/ui/Toast';
import { AlertCircle, Wifi, WifiOff } from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

export interface ChatContainerProps {
  /**
   * Session ID to load (optional, creates new if not provided)
   */
  sessionId?: string;
  
  /**
   * Initial topic for new session
   * @default "general"
   */
  initialTopic?: string;
  
  /**
   * Show emotion indicator
   * @default true
   */
  showEmotion?: boolean;
  
  /**
   * Enable voice interaction
   * @default true
   */
  enableVoice?: boolean;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

/**
 * Connection status for real-time updates
 */
type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const ChatContainer: React.FC<ChatContainerProps> = ({
  sessionId: propSessionId,
  initialTopic = 'general',
  showEmotion = true,
  enableVoice = true,
  className
}) => {
  // ============================================================================
  // STATE & REFS
  // ============================================================================
  
  // Store hooks
  const { user } = useAuthStore();
  const {
    messages,
    isLoading,
    error,
    sessionId: storeSessionId,
    sendMessage: storeSendMessage,
    loadHistory,
    clearError,
    setTyping
  } = useChatStore();
  
  const {
    currentEmotion,
    isAnalyzing
  } = useEmotionStore();
  
  // Local state
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const messageEndRef = useRef<HTMLDivElement>(null);
  
  // Navigation
  const navigate = useNavigate();
  const { sessionId: urlSessionId } = useParams<{ sessionId: string }>();
  
  // Determine active session ID
  const activeSessionId = propSessionId || urlSessionId || storeSessionId;
  
  // ============================================================================
  // WEBSOCKET CONNECTION - Real-time updates
  // ============================================================================
  
  const { isConnected, subscribe, emit: sendEvent } = useWebSocket();
  
  // Update connection status based on WebSocket state
  useEffect(() => {
    setConnectionStatus(isConnected ? 'connected' : 'disconnected');
  }, [isConnected]);
  
  // Join/leave session for real-time updates
  useEffect(() => {
    if (!activeSessionId) return;
    
    // ✅ Try to join if WebSocket connected, but don't block if it fails
    if (isConnected) {
      try {
        joinSession(activeSessionId);
        console.log('✓ Joined chat session:', activeSessionId);
      } catch (err) {
        console.warn('⚠️ Failed to join WebSocket session:', err);
        // Non-blocking - HTTP chat will still work
      }
    }
    
    // Leave session on unmount or session change
    return () => {
      if (isConnected) {
        try {
          leaveSession(activeSessionId);
          console.log('✓ Left chat session:', activeSessionId);
        } catch (err) {
          console.warn('⚠️ Failed to leave WebSocket session:', err);
          // Non-blocking
        }
      }
    };
  }, [isConnected, activeSessionId]);
  
  // Subscribe to real-time events (emotion updates, typing indicators are handled in socket.handlers.ts)
  useEffect(() => {
    if (!isConnected) return;
    
    // Subscribe to session-specific updates
    const unsubscribe = subscribe('session_update', (data: any) => {
      console.log('Session update:', data);
      // Additional session-specific logic can be added here
    });
    
    return unsubscribe;
  }, [isConnected, subscribe]);
  
  
  // ============================================================================
  // SESSION INITIALIZATION
  // ============================================================================
  
  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }
    
    const initializeSession = async () => {
      try {
        if (activeSessionId) {
          // Load existing session messages
          await loadHistory(activeSessionId);
        }
        // If no session ID, a new one will be created when first message is sent
        
        setIsInitialized(true);
      } catch (err) {
        console.error('Failed to initialize session:', err);
        toast.error('Session Error', {
          description: 'Failed to load chat session. Please try again.'
        });
      }
    };
    
    initializeSession();
  }, [activeSessionId, user, navigate, loadHistory]);
  
  // ============================================================================
  // AUTO-SCROLL TO BOTTOM (Optimized to prevent excessive re-renders)
  // ============================================================================
  
  const scrollToBottom = useCallback(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);
  
  useEffect(() => {
    // Only scroll when new messages are added, not on every render
    if (messages.length > 0) {
      // Use requestAnimationFrame to prevent layout thrashing
      requestAnimationFrame(() => {
        scrollToBottom();
      });
    }
  }, [messages.length, scrollToBottom]); // Only trigger on message count change
  
  // ============================================================================
  // MESSAGE SENDING HANDLER
  // ============================================================================
  
  const handleSendMessage = useCallback(async (content: string) => {
    if (!content.trim() || !user) return;
    
    try {
      await storeSendMessage(content.trim(), user.id);
      
      // WebSocket event for real-time updates (other tabs/devices)
      if (storeSessionId && isConnected) {
        try {
          sendEvent({
            type: 'message_sent',
            sessionId: storeSessionId,
            userId: user.id
          });
        } catch (wsErr) {
          console.warn('WebSocket notification failed (non-critical):', wsErr);
          // Non-blocking - message already sent via HTTP
        }
      }
      
    } catch (err: any) {
      console.error('Failed to send message:', err);
      
      // Determine specific error message
      let errorTitle = 'Send Failed';
      let errorMessage = 'Failed to send message. Please try again.';
      
      if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
        errorTitle = 'Request Timeout';
        errorMessage = 'Request timed out. Check your internet connection and try again.';
      } else if (err.response?.status === 401) {
        errorTitle = 'Authentication Required';
        errorMessage = 'Your session expired. Please log in again.';
      } else if (err.response?.status === 429) {
        errorTitle = 'Too Many Requests';
        errorMessage = 'You\'re sending messages too quickly. Please wait a moment.';
      } else if (err.response?.status === 500) {
        errorTitle = 'Server Error';
        errorMessage = 'Something went wrong on our end. Please try again in a moment.';
      } else if (err.response?.status === 404) {
        errorTitle = 'Not Found';
        errorMessage = 'The chat endpoint is not available. Please contact support.';
      } else if (!navigator.onLine) {
        errorTitle = 'No Connection';
        errorMessage = 'You appear to be offline. Check your network connection.';
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      toast.error(errorTitle, {
        description: errorMessage
      });
    }
  }, [user, storeSendMessage, storeSessionId, isConnected, sendEvent]);
  
  // ============================================================================
  // ERROR HANDLING
  // ============================================================================
  
  useEffect(() => {
    if (error) {
      toast.error('Error', {
        description: error
      });
    }
  }, [error]);
  
  // ============================================================================
  // LOADING STATE
  // ============================================================================
  
  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-full bg-bg-primary">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 border-4 border-accent-primary border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-text-secondary">Loading chat session...</p>
        </div>
      </div>
    );
  }
  
  // ============================================================================
  // RENDER
  // ============================================================================
  
  return (
    <div
      ref={containerRef}
      className={cn(
        'flex flex-col h-full bg-bg-primary',
        className
      )}
      role="main"
      aria-label="Chat interface"
    >
      {/* Connection Status Bar */}
      {connectionStatus !== 'connected' && (
        <div
          className={cn(
            'px-4 py-2 text-sm flex items-center gap-2 border-b border-white/10',
            connectionStatus === 'connecting' && 'bg-bg-secondary text-text-secondary',
            connectionStatus === 'disconnected' && 'bg-accent-warning/20 text-accent-warning',
            connectionStatus === 'error' && 'bg-accent-error/20 text-accent-error'
          )}
          role="status"
          aria-live="polite"
        >
          {connectionStatus === 'connecting' && (
            <>
              <Wifi className="w-4 h-4 animate-pulse" />
              <span>Connecting...</span>
            </>
          )}
          {connectionStatus === 'disconnected' && (
            <>
              <WifiOff className="w-4 h-4" />
              <span>Disconnected - Attempting to reconnect...</span>
            </>
          )}
          {connectionStatus === 'error' && (
            <>
              <AlertCircle className="w-4 h-4" />
              <span>Connection error - Some features may be limited</span>
            </>
          )}
        </div>
      )}
      
      {/* Emotion Indicator (Floating) */}
      {showEmotion && currentEmotion && (
        <div className="absolute top-4 right-4 z-10">
          <EmotionIndicator
            emotion={currentEmotion}
            isAnalyzing={isAnalyzing}
            compact
          />
        </div>
      )}
      
      {/* Message List */}
      <div className="flex-1 overflow-hidden">
        <MessageList
          messages={messages}
          isLoading={isLoading}
          currentUserId={user?.id}
        />
        
        {/* Typing Indicator */}
        {isLoading && (
          <div className="px-4 py-2">
            <TypingIndicator />
          </div>
        )}
        
        {/* Auto-scroll anchor */}
        <div ref={messageEndRef} />
      </div>
      
      {/* Message Input Area */}
      <div className="border-t border-white/10 bg-bg-secondary">
        <div className="container max-w-4xl mx-auto p-4">
          <div className="flex items-end gap-3">
            {/* Voice Button (if enabled) */}
            {enableVoice && (
              <VoiceButton
                onTranscription={handleSendMessage}
                disabled={isLoading || !isConnected}
              />
            )}
            
            {/* Text Input */}
            <div className="flex-1">
              <MessageInput
                onSend={handleSendMessage}
                disabled={isLoading}
                placeholder={
                  isLoading
                    ? 'AI is responding...'
                    : 'Ask me anything...'
                }
              />
              
              {/* Connection Warning (non-blocking) */}
              {!isConnected && (
                <div className="mt-1 text-xs text-accent-warning flex items-center gap-1">
                  <WifiOff className="w-3 h-3" />
                  <span>Real-time updates unavailable. Messages will still send.</span>
                </div>
              )}
            </div>
          </div>
          
          {/* Status indicators */}
          <div className="mt-2 flex items-center justify-between text-xs text-text-tertiary">
            <div className="flex items-center gap-4">
              <span
                className={cn(
                  'flex items-center gap-1',
                  isConnected ? 'text-accent-success' : 'text-accent-error'
                )}
              >
                <span className={cn(
                  'w-2 h-2 rounded-full',
                  isConnected ? 'bg-accent-success animate-pulse' : 'bg-accent-error'
                )} />
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
              
              {storeSessionId && (
                <span>
                  Session: {storeSessionId.slice(0, 8)}...
                </span>
              )}
            </div>
            
            <div>
              <kbd className="px-2 py-1 text-xs bg-bg-tertiary rounded">Enter</kbd>
              <span className="ml-1">to send</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

ChatContainer.displayName = 'ChatContainer';

export default ChatContainer;
