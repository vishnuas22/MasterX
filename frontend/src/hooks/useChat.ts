// **Purpose:** Simplified chat operations

// **What This File Contributes:**
// 1. Send message action
// 2. Message list access
// 3. Typing indicators
// 4. Real-time emotion updates

// **Implementation:**
// ```typescript
import { useChatStore } from '@store/chatStore';
import { useEmotionStore } from '@store/emotionStore';
import { useAuthStore } from '@store/authStore';
import { useUIStore } from '@store/uiStore';

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
    loadHistory,
    clearMessages,
  } = useChatStore();
  
  const { addEmotionData } = useEmotionStore();

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


// **Benefits:**
// 1. Single hook for chat operations
// 2. Automatic emotion tracking
// 3. Error handling included
// 4. Type-safe

// **Performance:**
// - Zustand optimized re-renders
// - Only components using chat data re-render

// **Connected Files:**
// - ← `store/chatStore.ts`
// - ← `store/emotionStore.ts`
// - ← `store/authStore.ts`
// - → `components/chat/MessageInput.tsx`
// - → `pages/MainApp.tsx`