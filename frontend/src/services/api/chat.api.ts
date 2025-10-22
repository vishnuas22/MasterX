import apiClient from './client';
import type { ChatRequest, ChatResponse, Message } from '@/types/chat.types';

export const chatAPI = {
  /**
   * Send a chat message
   * Backend: POST /api/v1/chat
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const { data } = await apiClient.post<ChatResponse>('/api/v1/chat', request);
    return data;
  },
  
  /**
   * Get conversation history
   * Backend: GET /api/v1/chat/history/:sessionId
   */
  getHistory: async (sessionId: string): Promise<Message[]> => {
    const { data } = await apiClient.get<Message[]>(
      `/api/v1/chat/history/${sessionId}`
    );
    return data;
  },
  
  /**
   * Delete conversation
   * Backend: DELETE /api/v1/chat/session/:sessionId
   */
  deleteSession: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/api/v1/chat/session/${sessionId}`);
  },
};
