/**
 * Chat & Messaging Types
 * 
 * Matches backend models.py:
 * - Message (lines 203-218)
 * - ChatRequest (lines 374-379)
 * - ChatResponse (lines 382-403)
 * - ContextInfo (lines 334-339)
 * - AbilityInfo (lines 342-347)
 */

import type { EmotionState, EmotionMetrics } from './emotion.types';

// ============================================================================
// MESSAGE TYPES
// ============================================================================

export enum MessageRole {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system',
}

export interface Message {
  id: string; // UUID
  session_id: string;
  user_id: string;
  role: MessageRole;
  content: string;
  timestamp: string; // ISO 8601
  
  // Emotion data (for user messages)
  emotion_state?: EmotionState | null;
  
  // AI response metadata (for assistant messages)
  provider_used?: string;
  response_time_ms?: number; // milliseconds
  tokens_used?: number;
  cost?: number;
  
  // Optional fields
  embedding?: number[];
  quality_rating?: number; // 1-5
  
  // Legacy aliases for backward compatibility
  emotion?: EmotionState | EmotionMetrics | null;
  provider?: string;
  responseTime?: number;
}

// ============================================================================
// CHAT API TYPES
// ============================================================================

export interface ChatRequest {
  user_id: string;
  session_id?: string;
  message: string;
  context?: Record<string, unknown>;
}

export interface ContextInfo {
  recent_messages_count: number;
  relevant_messages_count: number;
  has_context: boolean;
  retrieval_time_ms?: number;
}

export interface AbilityInfo {
  ability_level: number; // 0.0 - 1.0
  recommended_difficulty: number; // 0.0 - 1.0
  cognitive_load: number; // 0.0 - 1.0
  flow_state_score?: number;
}

export interface ChatResponse {
  session_id: string;
  message: string;
  emotion_state?: EmotionState | null;
  provider_used: string;
  response_time_ms: number;
  timestamp: string; // ISO 8601
  
  // Enhanced metadata (Phase 2-4)
  category_detected?: string;
  tokens_used?: number;
  cost?: number;
  
  // Phase 3 metadata
  context_retrieved?: ContextInfo;
  ability_info?: AbilityInfo;
  ability_updated?: boolean;
  
  // Phase 4 metadata
  cached?: boolean;
  processing_breakdown?: Record<string, number>;
}

/**
 * Chat History Response
 * 
 * Response from GET /api/v1/chat/history/{session_id}
 * Returns all messages from a conversation session
 */
export interface ChatHistoryResponse {
  session_id: string;
  messages: Message[];
  total_messages: number;
  session_started: string; // ISO 8601
  total_cost: number;
}

// ============================================================================
// REAL-TIME TYPES (WebSocket)
// ============================================================================

export interface TypingIndicator {
  user_id: string;
  session_id: string;
  is_typing: boolean;
  timestamp: string;
}

export interface MessageUpdate {
  message_id: string;
  session_id: string;
  updates: Partial<Message>;
  timestamp: string;
}

export interface EmotionUpdate {
  session_id: string;
  emotion: string;
  confidence: number;
  timestamp: string;
}

// ============================================================================
// CHAT UI STATE
// ============================================================================

export interface ChatUIState {
  isTyping: boolean;
  isLoading: boolean;
  error: string | null;
  scrollToBottom: boolean;
}

export interface MessageGroup {
  date: string; // YYYY-MM-DD
  messages: Message[];
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

export const isMessage = (obj: unknown): obj is Message => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    'role' in obj &&
    'content' in obj
  );
};

export const isChatResponse = (obj: unknown): obj is ChatResponse => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'session_id' in obj &&
    'message' in obj &&
    'provider_used' in obj
  );
};

// ============================================================================
// HELPER TYPES
// ============================================================================

export type MessageWithEmotion = Message & {
  emotion: EmotionMetrics;
};

export type OptimisticMessage = Omit<Message, 'id'> & {
  id: string;
  optimistic: boolean;
};
