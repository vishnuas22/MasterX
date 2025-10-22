import { create } from 'zustand';
import { chatAPI } from '@/services/api/chat.api';
import type { Message, ChatRequest, ChatResponse } from '@/types/chat.types';
import { MessageRole } from '@/types/chat.types';
import type { EmotionState } from '@/types/emotion.types';

interface ChatState {
  // State
  messages: Message[];
  isTyping: boolean;
  isLoading: boolean;
  currentEmotion: EmotionState | null;
  sessionId: string | null;
  error: string | null;
  
  // Actions
  sendMessage: (content: string) => Promise<void>;
  addMessage: (message: Message) => void;
  updateMessageEmotion: (messageId: string, emotion: EmotionState) => void;
  clearMessages: () => void;
  loadHistory: (sessionId: string) => Promise<void>;
  setTyping: (isTyping: boolean) => void;
  setCurrentEmotion: (emotion: EmotionState | null) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  // Initial state
  messages: [],
  isTyping: false,
  isLoading: false,
  currentEmotion: null,
  sessionId: null,
  error: null,
  
  // Send message action
  sendMessage: async (content: string) => {
    const { sessionId } = get();
    
    // Optimistic update: Add user message immediately
    const userMessage: Message = {
      id: `temp-${Date.now()}`,
      session_id: sessionId || '',
      user_id: 'current_user',
      role: MessageRole.USER,
      content,
      timestamp: new Date().toISOString(),
      emotion: null,
    };
    
    set((state) => ({
      messages: [...state.messages, userMessage],
      isLoading: true,
      isTyping: true,
      error: null,
    }));
    
    try {
      // Call backend API
      const request: ChatRequest = {
        message: content,
        user_id: 'current_user', // Get from authStore
        session_id: sessionId || undefined,
      };
      
      const response: ChatResponse = await chatAPI.sendMessage(request);
      
      // Replace temp message with actual message
      set((state) => ({
        messages: state.messages.map((msg) =>
          msg.id === userMessage.id
            ? { ...msg, session_id: response.session_id, emotion: response.emotion_state }
            : msg
        ),
      }));
      
      // Add AI response
      const aiMessage: Message = {
        id: `ai-${Date.now()}`,
        session_id: response.session_id,
        user_id: 'current_user',
        role: MessageRole.ASSISTANT,
        content: response.message,
        timestamp: response.timestamp,
        emotion: response.emotion_state,
        provider: response.provider_used,
        responseTime: response.response_time_ms,
      };
      
      set((state) => ({
        messages: [...state.messages, aiMessage],
        isLoading: false,
        isTyping: false,
        currentEmotion: response.emotion_state,
        sessionId: response.session_id,
      }));
    } catch (error: any) {
      // Remove optimistic message on error
      set((state) => ({
        messages: state.messages.filter((msg) => msg.id !== userMessage.id),
        isLoading: false,
        isTyping: false,
        error: error.message || 'Failed to send message',
      }));
      throw error;
    }
  },
  
  // Add message (for WebSocket updates)
  addMessage: (message) => {
    set((state) => ({
      messages: [...state.messages, message],
    }));
  },
  
  // Update emotion for existing message
  updateMessageEmotion: (messageId, emotion) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === messageId ? { ...msg, emotion } : msg
      ),
      currentEmotion: emotion,
    }));
  },
  
  // Clear all messages
  clearMessages: () => {
    set({
      messages: [],
      currentEmotion: null,
      sessionId: null,
      error: null,
    });
  },
  
  // Load message history
  loadHistory: async (sessionId: string) => {
    set({ isLoading: true, error: null });
    try {
      const messages = await chatAPI.getHistory(sessionId);
      set({
        messages,
        sessionId,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.message,
        isLoading: false,
      });
    }
  },
  
  // Set typing indicator
  setTyping: (isTyping) => set({ isTyping }),
  
  // Set current emotion
  setCurrentEmotion: (emotion) => set({ currentEmotion: emotion }),
}));
