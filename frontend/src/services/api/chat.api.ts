// **Purpose:** Chat-related API calls

// **What This File Contributes:**
// 1. Send message to backend
// 2. Get conversation history
// 3. Real-time typing indicators

// **Implementation:**
// ```typescript
import apiClient from './client';
import type { ChatRequest, ChatResponse, Message } from '@types/chat.types';

export const chatAPI = {
  /**
   * Send a chat message
   * Backend: POST /api/v1/chat
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const { data } = await apiClient.post<ChatResponse>('/api/v1/chat', request, {
      retry: 2, // Retry twice on failure
    });
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


// **Key Features:**
// 1. **Type-safe:** Full TypeScript types
// 2. **Error handling:** Automatic via interceptors
// 3. **Retry:** Configured per-endpoint

// **Connected Files:**
// - ← `services/api/client.ts` (axios instance)
// - ← `types/chat.types.ts` (type definitions)
// - → `store/chatStore.ts` (uses these functions)

// **Integration with Backend:**
// ```
// POST   /api/v1/chat                  ← sendMessage()
// GET    /api/v1/chat/history/:id      ← getHistory()
// DELETE /api/v1/chat/session/:id      ← deleteSession()