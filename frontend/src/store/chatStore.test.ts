/**
 * chatStore.test.ts - Unit Tests for Chat Store
 * 
 * Purpose: Test chat message management and real-time updates
 * 
 * Coverage:
 * - Send message flow
 * - Optimistic UI updates
 * - Message history loading
 * - Emotion updates
 * - Typing indicators
 * - Error handling
 * 
 * Following AGENTS_FRONTEND.md:
 * - Test coverage > 80%
 * - Isolated tests
 * - Mock API calls
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useChatStore } from '@/store/chatStore';
import { chatAPI } from '@/services/api/chat.api';
import { MessageRole } from '@/types/chat.types';
import type { ChatResponse } from '@/types/chat.types';

// Mock the chat API
vi.mock('@/services/api/chat.api', () => ({
  chatAPI: {
    sendMessage: vi.fn(),
    getChatHistory: vi.fn(),
  },
}));

describe('chatStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useChatStore.setState({
      messages: [],
      isTyping: false,
      isLoading: false,
      currentEmotion: null,
      sessionId: null,
      error: null,
    });
    
    // Clear all mocks
    vi.clearAllMocks();
  });

  // ============================================================================
  // INITIAL STATE
  // ============================================================================

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const state = useChatStore.getState();
      
      expect(state.messages).toEqual([]);
      expect(state.isTyping).toBe(false);
      expect(state.isLoading).toBe(false);
      expect(state.currentEmotion).toBeNull();
      expect(state.sessionId).toBeNull();
      expect(state.error).toBeNull();
    });
  });

  // ============================================================================
  // SEND MESSAGE
  // ============================================================================

  describe('sendMessage', () => {
    const userId = 'user-1';
    const messageContent = 'Hello, how are you?';
    
    const mockResponse: ChatResponse = {
      user_message: {
        id: 'msg-user-1',
        session_id: 'session-1',
        user_id: userId,
        role: MessageRole.USER,
        content: messageContent,
        timestamp: '2024-01-01T00:00:00Z',
        emotion_state: null,
      },
      assistant_message: {
        id: 'msg-assistant-1',
        session_id: 'session-1',
        user_id: userId,
        role: MessageRole.ASSISTANT,
        content: 'I am doing well, thank you!',
        timestamp: '2024-01-01T00:00:01Z',
        emotion_state: null,
      },
      session_id: 'session-1',
      emotion: {
        primary_emotion: 'joy',
        confidence: 0.85,
        pad: {
          pleasure: 0.7,
          arousal: 0.5,
          dominance: 0.6,
        },
        categories: {
          joy: 0.85,
          excitement: 0.6,
        },
        learning_readiness: 0.8,
        cognitive_load: 0.4,
        flow_state: 0.7,
        timestamp: '2024-01-01T00:00:00Z',
      },
    };

    it('should add optimistic user message immediately', async () => {
      vi.mocked(chatAPI.sendMessage).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(mockResponse), 100))
      );

      const { sendMessage } = useChatStore.getState();
      const sendPromise = sendMessage(messageContent, userId);

      // Check optimistic update
      const stateBeforeResponse = useChatStore.getState();
      expect(stateBeforeResponse.messages.length).toBe(1);
      expect(stateBeforeResponse.messages[0].content).toBe(messageContent);
      expect(stateBeforeResponse.messages[0].role).toBe(MessageRole.USER);
      expect(stateBeforeResponse.isLoading).toBe(true);
      expect(stateBeforeResponse.isTyping).toBe(true);

      await sendPromise;
    });

    it('should replace temp message with confirmed messages from backend', async () => {
      vi.mocked(chatAPI.sendMessage).mockResolvedValue(mockResponse);

      const { sendMessage } = useChatStore.getState();
      await sendMessage(messageContent, userId);

      const state = useChatStore.getState();
      expect(state.messages.length).toBe(2);
      expect(state.messages[0].id).toBe('msg-user-1');
      expect(state.messages[1].id).toBe('msg-assistant-1');
      expect(state.messages[1].content).toBe('I am doing well, thank you!');
    });

    it('should update session ID', async () => {
      vi.mocked(chatAPI.sendMessage).mockResolvedValue(mockResponse);

      const { sendMessage } = useChatStore.getState();
      await sendMessage(messageContent, userId);

      const state = useChatStore.getState();
      expect(state.sessionId).toBe('session-1');
    });

    it('should update current emotion', async () => {
      vi.mocked(chatAPI.sendMessage).mockResolvedValue(mockResponse);

      const { sendMessage } = useChatStore.getState();
      await sendMessage(messageContent, userId);

      const state = useChatStore.getState();
      expect(state.currentEmotion).toBeDefined();
      expect(state.currentEmotion?.primary_emotion).toBe('joy');
      expect(state.currentEmotion?.confidence).toBe(0.85);
    });

    it('should reset loading and typing states after response', async () => {
      vi.mocked(chatAPI.sendMessage).mockResolvedValue(mockResponse);

      const { sendMessage } = useChatStore.getState();
      await sendMessage(messageContent, userId);

      const state = useChatStore.getState();
      expect(state.isLoading).toBe(false);
      expect(state.isTyping).toBe(false);
    });

    it('should handle send message errors', async () => {
      const errorMessage = 'Network error';
      vi.mocked(chatAPI.sendMessage).mockRejectedValue(new Error(errorMessage));

      const { sendMessage } = useChatStore.getState();
      await sendMessage(messageContent, userId);

      const state = useChatStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
      expect(state.isTyping).toBe(false);
      // Optimistic message should still be there
      expect(state.messages.length).toBe(1);
    });
  });

  // ============================================================================
  // ADD MESSAGE
  // ============================================================================

  describe('addMessage', () => {
    it('should add a message to the list', () => {
      const message = {
        id: 'msg-1',
        session_id: 'session-1',
        user_id: 'user-1',
        role: MessageRole.USER,
        content: 'Test message',
        timestamp: new Date().toISOString(),
        emotion_state: null,
      };

      const { addMessage } = useChatStore.getState();
      addMessage(message);

      const state = useChatStore.getState();
      expect(state.messages).toHaveLength(1);
      expect(state.messages[0]).toEqual(message);
    });

    it('should append messages to existing list', () => {
      const message1 = {
        id: 'msg-1',
        session_id: 'session-1',
        user_id: 'user-1',
        role: MessageRole.USER,
        content: 'First message',
        timestamp: new Date().toISOString(),
        emotion_state: null,
      };

      const message2 = {
        id: 'msg-2',
        session_id: 'session-1',
        user_id: 'user-1',
        role: MessageRole.ASSISTANT,
        content: 'Second message',
        timestamp: new Date().toISOString(),
        emotion_state: null,
      };

      const { addMessage } = useChatStore.getState();
      addMessage(message1);
      addMessage(message2);

      const state = useChatStore.getState();
      expect(state.messages).toHaveLength(2);
      expect(state.messages[0].id).toBe('msg-1');
      expect(state.messages[1].id).toBe('msg-2');
    });
  });

  // ============================================================================
  // UPDATE MESSAGE EMOTION
  // ============================================================================

  describe('updateMessageEmotion', () => {
    beforeEach(() => {
      useChatStore.setState({
        messages: [
          {
            id: 'msg-1',
            session_id: 'session-1',
            user_id: 'user-1',
            role: MessageRole.USER,
            content: 'Test message',
            timestamp: new Date().toISOString(),
            emotion_state: null,
          },
        ],
      });
    });

    it('should update emotion for specific message', () => {
      const emotion = {
        primary_emotion: 'joy',
        confidence: 0.9,
        pad: {
          pleasure: 0.8,
          arousal: 0.6,
          dominance: 0.7,
        },
        categories: {
          joy: 0.9,
        },
        learning_readiness: 0.85,
        cognitive_load: 0.3,
        flow_state: 0.75,
        timestamp: new Date().toISOString(),
      };

      const { updateMessageEmotion } = useChatStore.getState();
      updateMessageEmotion('msg-1', emotion);

      const state = useChatStore.getState();
      expect(state.messages[0].emotion_state).toEqual(emotion);
    });

    it('should not update emotion for non-existent message', () => {
      const emotion = {
        primary_emotion: 'joy',
        confidence: 0.9,
        pad: {
          pleasure: 0.8,
          arousal: 0.6,
          dominance: 0.7,
        },
        categories: {
          joy: 0.9,
        },
        learning_readiness: 0.85,
        cognitive_load: 0.3,
        flow_state: 0.75,
        timestamp: new Date().toISOString(),
      };

      const { updateMessageEmotion } = useChatStore.getState();
      updateMessageEmotion('non-existent-id', emotion);

      const state = useChatStore.getState();
      expect(state.messages[0].emotion_state).toBeNull();
    });
  });

  // ============================================================================
  // LOAD HISTORY
  // ============================================================================

  describe('loadHistory', () => {
    const mockHistory = {
      messages: [
        {
          id: 'msg-1',
          session_id: 'session-1',
          user_id: 'user-1',
          role: MessageRole.USER,
          content: 'Previous message',
          timestamp: '2024-01-01T00:00:00Z',
          emotion_state: null,
        },
        {
          id: 'msg-2',
          session_id: 'session-1',
          user_id: 'user-1',
          role: MessageRole.ASSISTANT,
          content: 'Previous response',
          timestamp: '2024-01-01T00:00:01Z',
          emotion_state: null,
        },
      ],
      session_id: 'session-1',
    };

    it('should load message history', async () => {
      vi.mocked(chatAPI.getChatHistory).mockResolvedValue(mockHistory);

      const { loadHistory } = useChatStore.getState();
      await loadHistory('session-1');

      const state = useChatStore.getState();
      expect(state.messages).toEqual(mockHistory.messages);
      expect(state.sessionId).toBe('session-1');
    });

    it('should handle history loading errors', async () => {
      const errorMessage = 'Failed to load history';
      vi.mocked(chatAPI.getChatHistory).mockRejectedValue(new Error(errorMessage));

      const { loadHistory } = useChatStore.getState();
      await loadHistory('session-1');

      const state = useChatStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.messages).toEqual([]);
    });
  });

  // ============================================================================
  // TYPING INDICATOR
  // ============================================================================

  describe('setTyping', () => {
    it('should set typing state to true', () => {
      const { setTyping } = useChatStore.getState();
      setTyping(true);

      const state = useChatStore.getState();
      expect(state.isTyping).toBe(true);
    });

    it('should set typing state to false', () => {
      useChatStore.setState({ isTyping: true });

      const { setTyping } = useChatStore.getState();
      setTyping(false);

      const state = useChatStore.getState();
      expect(state.isTyping).toBe(false);
    });
  });

  // ============================================================================
  // CURRENT EMOTION
  // ============================================================================

  describe('setCurrentEmotion', () => {
    it('should set current emotion', () => {
      const emotion = {
        primary_emotion: 'joy',
        confidence: 0.9,
        pad: {
          pleasure: 0.8,
          arousal: 0.6,
          dominance: 0.7,
        },
        categories: {
          joy: 0.9,
        },
        learning_readiness: 0.85,
        cognitive_load: 0.3,
        flow_state: 0.75,
        timestamp: new Date().toISOString(),
      };

      const { setCurrentEmotion } = useChatStore.getState();
      setCurrentEmotion(emotion);

      const state = useChatStore.getState();
      expect(state.currentEmotion).toEqual(emotion);
    });

    it('should clear current emotion', () => {
      useChatStore.setState({
        currentEmotion: {
          primary_emotion: 'joy',
          confidence: 0.9,
          pad: {
            pleasure: 0.8,
            arousal: 0.6,
            dominance: 0.7,
          },
          categories: {
            joy: 0.9,
          },
          learning_readiness: 0.85,
          cognitive_load: 0.3,
          flow_state: 0.75,
          timestamp: new Date().toISOString(),
        },
      });

      const { setCurrentEmotion } = useChatStore.getState();
      setCurrentEmotion(null);

      const state = useChatStore.getState();
      expect(state.currentEmotion).toBeNull();
    });
  });

  // ============================================================================
  // CLEAR MESSAGES
  // ============================================================================

  describe('clearMessages', () => {
    beforeEach(() => {
      useChatStore.setState({
        messages: [
          {
            id: 'msg-1',
            session_id: 'session-1',
            user_id: 'user-1',
            role: MessageRole.USER,
            content: 'Test',
            timestamp: new Date().toISOString(),
            emotion_state: null,
          },
        ],
        sessionId: 'session-1',
      });
    });

    it('should clear all messages', () => {
      const { clearMessages } = useChatStore.getState();
      clearMessages();

      const state = useChatStore.getState();
      expect(state.messages).toEqual([]);
      expect(state.sessionId).toBeNull();
    });
  });

  // ============================================================================
  // CLEAR ERROR
  // ============================================================================

  describe('clearError', () => {
    it('should clear error state', () => {
      useChatStore.setState({ error: 'Some error' });

      const { clearError } = useChatStore.getState();
      clearError();

      const state = useChatStore.getState();
      expect(state.error).toBeNull();
    });
  });
});
