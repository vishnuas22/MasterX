/**
 * Chat Operations Hook
 * 
 * Provides simplified chat operations with automatic emotion tracking,
 * error handling, and user feedback via toast notifications.
 * 
 * @example
 * const { sendMessage, messages, isTyping, currentEmotion } = useChat();
 * await sendMessage('What is photosynthesis?');
 */

import { useChatStore } from '@/store/chatStore';
import { useEmotionStore } from '@/store/emotionStore';
import { useAuthStore } from '@/store/authStore';
import { useUIStore } from '@/store/uiStore';

export const useChat = () => {
  const { user } = useAuthStore();
  const { showToast } = useUIStore();
  const {
    messages,
    isTyping,
    isLoading,
    currentEmotion,
    sessionId,
    error,
    sendMessage: storeSendMessage,
    clearMessages,
  } = useChatStore();
  
  const { addEmotionData } = useEmotionStore();

  /**
   * Send message with automatic emotion tracking
   * Validates input and provides user feedback
   */
  const sendMessage = async (content: string) => {
    if (!content.trim()) {
      showToast({
        type: 'warning',
        message: 'Please enter a message',
      });
      return;
    }

    if (!user) {
      showToast({
        type: 'error',
        message: 'Please login to send messages',
      });
      return;
    }

    try {
      await storeSendMessage(content, user.id);
      
      // Add emotion to emotion store for tracking
      if (currentEmotion) {
        addEmotionData(currentEmotion);
      }
    } catch (error: any) {
      showToast({
        type: 'error',
        message: error.message || 'Failed to send message',
      });
    }
  };

  /**
   * Start new conversation
   * Clears current messages and provides feedback
   */
  const startNewConversation = () => {
    clearMessages();
    showToast({
      type: 'info',
      message: 'Started new conversation',
    });
  };

  return {
    // State
    messages,
    isTyping,
    isLoading,
    currentEmotion,
    sessionId,
    error,
    
    // Actions
    sendMessage,
    startNewConversation,
  };
};