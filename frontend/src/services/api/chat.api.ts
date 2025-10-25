/**
 * Chat API Service
 * 
 * Handles all chat-related API calls:
 * - Send messages to backend with emotion detection
 * - Process AI responses with context and adaptation
 * - Real-time learning interactions
 * 
 * Backend Integration:
 * POST /api/v1/chat - Main learning interaction endpoint
 * 
 * The chat endpoint handles:
 * - Emotion detection (frustration, confusion, etc.)
 * - Context management (conversation memory)
 * - Adaptive learning (difficulty adjustment)
 * - Provider selection (best AI model)
 * - Cost tracking
 * 
 * @module services/api/chat.api
 */

import apiClient from './client';
import type { ChatRequest, ChatResponse } from '../../types/chat.types';

/**
 * Chat API endpoints
 */
export const chatAPI = {
  /**
   * Send a chat message
   * 
   * Processes user message through the full MasterX pipeline:
   * 1. Emotion detection (identifies frustration, confusion, etc.)
   * 2. Context retrieval (loads relevant conversation history)
   * 3. Adaptive learning (adjusts difficulty to student's ability)
   * 4. Provider selection (chooses best AI model for task)
   * 5. Response generation (personalized, emotion-aware response)
   * 
   * Session Handling:
   * - If session_id provided: Continues existing conversation
   * - If no session_id: Creates new session automatically
   * 
   * @param request - Chat request with message and context
   * @param request.user_id - User identifier (required)
   * @param request.message - User's message text (required)
   * @param request.session_id - Optional: existing session ID
   * @param request.context - Optional: additional context (subject, etc.)
   * @returns Chat response with AI message and metadata
   * 
   * Response includes:
   * - message: AI-generated response text
   * - emotion_state: Detected emotions and learning readiness
   * - session_id: Session identifier for follow-up messages
   * - provider_used: Which AI model was used (gemini, claude, etc.)
   * - response_time_ms: Processing time
   * - cost: Cost in USD for this interaction
   * - context_retrieved: Relevant conversation history used
   * - ability_info: Student's current ability level
   * 
   * @throws 404 - Session not found (if invalid session_id)
   * @throws 500 - Chat processing failed
   * 
   * @example
   * ```typescript
   * const response = await chatAPI.sendMessage({
   *   user_id: 'user-123',
   *   message: "I'm frustrated with this calculus problem",
   *   session_id: 'session-abc', // optional
   *   context: { subject: 'calculus' } // optional
   * });
   * 
   * // Response includes emotion detection
   * console.log(response.emotion_state.primary_emotion); // "frustration"
   * console.log(response.emotion_state.learning_readiness); // "STRUGGLING"
   * 
   * // AI response is personalized based on emotion
   * console.log(response.message); // Encouraging, supportive response
   * ```
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const { data } = await apiClient.post<ChatResponse>(
      '/api/v1/chat',
      request,
      {
        timeout: 30000, // 30 seconds (AI processing can take time)
      }
    );
    return data;
  },
};