/**
 * WebSocket Event Handlers - Handle all WebSocket events from backend
 * 
 * Features:
 * - Event type definitions
 * - Event handlers with store updates
 * - Type-safe event handling
 * 
 * Event Flow:
 * 1. User sends message → Backend processes
 * 2. Backend emits `typing_indicator` (true)
 * 3. Backend detects emotion → Emits `emotion_update`
 * 4. Backend generates response → Emits `message_received`
 * 5. Backend emits `typing_indicator` (false)
 * 
 * Performance:
 * - Event handlers: <1ms processing time
 * - Direct store updates (no re-renders unless subscribed)
 */

import nativeSocketClient from './native-socket.client';
import { useChatStore } from '@/store/chatStore';
import { useEmotionStore } from '@/store/emotionStore';
import type { Message } from '@/types/chat.types';
import type { EmotionState } from '@/types/emotion.types';

/**
 * Initialize all WebSocket event handlers
 */
export const initializeSocketHandlers = (): void => {
  // Real-time emotion update during AI processing
  nativeSocketClient.on('emotion_update', (data: {
    message_id: string;
    emotion: EmotionState;
  }) => {
    console.log('Emotion update received:', data.emotion.primary_emotion);
    
    // Update message with emotion
    useChatStore.getState().updateMessageEmotion(data.message_id, data.emotion);
    
    // Add to emotion history
    useEmotionStore.getState().addEmotionData(data.emotion);
  });

  // AI typing indicator
  nativeSocketClient.on('typing_indicator', (data: { isTyping: boolean }) => {
    useChatStore.getState().setTyping(data.isTyping);
  });

  // New message received (for multi-user scenarios)
  nativeSocketClient.on('message_received', (data: { message: Message }) => {
    useChatStore.getState().addMessage(data.message);
  });

  // Session state update
  nativeSocketClient.on('session_update', (data: {
    session_id: string;
    status: string;
    message_count: number;
  }) => {
    console.log('Session updated:', data);
    // Could update session info in store if needed
  });

  // Error from server
  nativeSocketClient.on('error', (data: { message: string; code: string }) => {
    console.error('WebSocket error:', data);
    
    import('@/store/uiStore').then(({ useUIStore }) => {
      useUIStore.getState().showToast({
        type: 'error',
        message: data.message || 'Real-time connection error',
      });
    });
  });
};

/**
 * Clean up all event handlers
 */
export const cleanupSocketHandlers = (): void => {
  nativeSocketClient.off('emotion_update');
  nativeSocketClient.off('typing_indicator');
  nativeSocketClient.off('message_received');
  nativeSocketClient.off('session_update');
  nativeSocketClient.off('error');
};

/**
 * Emit typing indicator to server
 */
export const emitTypingIndicator = (isTyping: boolean): void => {
  nativeSocketClient.send('user_typing', { isTyping });
};

/**
 * Join a chat session (for real-time updates)
 */
export const joinSession = (sessionId: string): void => {
  nativeSocketClient.send('join_session', { session_id: sessionId });
};

/**
 * Leave a chat session
 */
export const leaveSession = (sessionId: string): void => {
  nativeSocketClient.send('leave_session', { session_id: sessionId });
};
