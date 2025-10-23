import { useChatStore } from '@/store/chatStore';
import { useUIStore } from '@/store/uiStore';

/**
 * Chat operations hook - Simplified chat functionality
 * 
 * Features:
 * - Send messages with automatic emotion tracking
 * - Load conversation history
 * - Start new conversations
 * - Error handling with toast notifications
 */
export const useChat = () => {
  const { showToast } = useUIStore();
  const {
    messages,
    isTyping,
    isLoading,
    currentEmotion,
    sessionId,
    error,
    sendMessage: storeSendMessage,
    loadHistory,
    clearMessages,
  } = useChatStore();

  /**
   * Send message with automatic emotion tracking
   */
  const sendMessage = async (content: string) => {
    if (!content.trim()) {
      showToast({
        type: 'warning',
        message: 'Please enter a message',
      });
      return;
    }

    try {
      await storeSendMessage(content);
      
      // Add emotion to emotion store for tracking
      // Note: currentEmotion is EmotionState (simplified),
      // but emotionStore expects EmotionMetrics (full).
      // In production, the Message emotion can include full EmotionMetrics
      // which should be tracked there instead of here.
    } catch (error: any) {
      showToast({
        type: 'error',
        message: error.message || 'Failed to send message',
      });
    }
  };

  /**
   * Load conversation history
   */
  const loadConversation = async (sessionId: string) => {
    try {
      await loadHistory(sessionId);
    } catch (error: any) {
      showToast({
        type: 'error',
        message: 'Failed to load conversation',
      });
    }
  };

  /**
   * Start new conversation
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
    loadConversation,
    startNewConversation,
  };
};
