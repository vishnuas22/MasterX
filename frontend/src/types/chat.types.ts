/**
 * Chat & Messaging Types
 * 
 * Matches backend models.py:
 * - Message (lines 203-218)
 * - ChatRequest (lines 374-379)
 * - ChatResponse (lines 382-403)
 * - ContextInfo (lines 334-339)
 * - AbilityInfo (lines 342-347)
 * 
 * @module types/chat
 */

import type { EmotionState, EmotionMetrics } from './emotion.types';

// ============================================================================
// MESSAGE TYPES
// ============================================================================

/**
 * Message role enum matching backend
 */
export enum MessageRole {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system',
}

/**
 * Core message structure matching backend Message model
 * Contains all metadata for a single chat message
 */
export interface Message {
  id: string; // UUID
  session_id: string;
  user_id: string;
  role: MessageRole;
  content: string;
  timestamp: string; // ISO 8601
  emotion?: EmotionState | EmotionMetrics | null;
  provider?: string; // AI provider used (groq, emergent, gemini)
  responseTime?: number; // milliseconds
  tokens_used?: number;
  cost?: number;
  embedding?: number[]; // Vector embedding for semantic search
  quality_rating?: number; // 1-5 user rating
}

// ============================================================================
// CHAT API TYPES
// ============================================================================

/**
 * Request payload for sending a chat message
 * Matches backend ChatRequest model
 */
export interface ChatRequest {
  user_id: string;
  session_id?: string;
  message: string;
  context?: Record<string, unknown>;
}

/**
 * Context information from context manager
 * Shows how much conversational context was retrieved
 */
export interface ContextInfo {
  recent_messages_count: number;
  relevant_messages_count: number;
  has_context: boolean;
  retrieval_time_ms?: number;
}

/**
 * Ability information from adaptive learning system
 * IRT-based ability estimation and recommendations
 */
export interface AbilityInfo {
  ability_level: number; // 0.0 - 1.0
  recommended_difficulty: number; // 0.0 - 1.0
  cognitive_load: number; // 0.0 - 1.0
  flow_state_score?: number; // 0.0 - 1.0
}

/**
 * Response payload from chat endpoint
 * Matches backend ChatResponse model
 */
export interface ChatResponse {
  session_id: string;
  message: string;
  emotion_state?: EmotionState | null;
  provider_used: string;
  response_time_ms: number;
  timestamp: string; // ISO 8601
  
  // Enhanced metadata (Phase 2-4)
  category_detected?: string; // Task category (coding, math, reasoning, etc.)
  tokens_used?: number;
  cost?: number;
  
  // Phase 3 metadata (Context + Adaptive Learning)
  context_retrieved?: ContextInfo;
  ability_info?: AbilityInfo;
  ability_updated?: boolean;
  
  // Phase 4 metadata (Optimization)
  cached?: boolean;
  processing_breakdown?: Record<string, number>; // Time breakdown by component
}

// ============================================================================
// REAL-TIME TYPES (WebSocket)
// ============================================================================

/**
 * Typing indicator for showing "User is typing..." or "AI is thinking..."
 */
export interface TypingIndicator {
  user_id: string;
  session_id: string;
  is_typing: boolean;
  timestamp: string;
}

/**
 * Message update event (e.g., emotion added after analysis)
 */
export interface MessageUpdate {
  message_id: string;
  session_id: string;
  updates: Partial<Message>;
  timestamp: string;
}

/**
 * Real-time emotion update during message processing
 */
export interface EmotionUpdate {
  session_id: string;
  emotion: string;
  confidence: number;
  timestamp: string;
}

// ============================================================================
// CHAT UI STATE
// ============================================================================

/**
 * UI state for chat interface
 * Manages loading, error, and interaction states
 */
export interface ChatUIState {
  isTyping: boolean; // AI is generating response
  isLoading: boolean; // Waiting for API response
  error: string | null;
  scrollToBottom: boolean; // Auto-scroll flag
}

/**
 * Messages grouped by date for UI display
 * Used for "Today", "Yesterday" headers
 */
export interface MessageGroup {
  date: string; // YYYY-MM-DD
  messages: Message[];
}

/**
 * Chat session information
 */
export interface ChatSession {
  session_id: string;
  user_id: string;
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
  message_count: number;
  title?: string; // Optional session title
  is_active: boolean;
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

/**
 * Type guard to check if object is a Message
 * @param obj - Object to check
 * @returns True if object matches Message interface
 */
export const isMessage = (obj: unknown): obj is Message => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    'role' in obj &&
    'content' in obj &&
    'timestamp' in obj
  );
};

/**
 * Type guard to check if object is a ChatResponse
 * @param obj - Object to check
 * @returns True if object matches ChatResponse interface
 */
export const isChatResponse = (obj: unknown): obj is ChatResponse => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'session_id' in obj &&
    'message' in obj &&
    'provider_used' in obj &&
    'response_time_ms' in obj
  );
};

/**
 * Type guard to check if object is a ChatRequest
 * @param obj - Object to check
 * @returns True if object matches ChatRequest interface
 */
export const isChatRequest = (obj: unknown): obj is ChatRequest => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'user_id' in obj &&
    'message' in obj
  );
};

// ============================================================================
// HELPER TYPES
// ============================================================================

/**
 * Message with guaranteed emotion data
 * Used when emotion is required (not optional)
 */
export type MessageWithEmotion = Message & {
  emotion: EmotionMetrics;
};

/**
 * Optimistic message for immediate UI update
 * Has temporary ID until confirmed by backend
 */
export type OptimisticMessage = Omit<Message, 'id'> & {
  id: string; // Temporary client-side ID
  optimistic: boolean; // Flag to identify optimistic messages
};

/**
 * Message with partial updates (for editing/patching)
 */
export type MessagePatch = Pick<Message, 'id'> & Partial<Omit<Message, 'id'>>;

// ============================================================================
// STREAMING TYPES
// ============================================================================

/**
 * Streaming response chunk for real-time AI response
 * Used when AI response is streamed token-by-token
 */
export interface StreamChunk {
  session_id: string;
  message_id: string;
  chunk: string; // Text chunk
  is_final: boolean; // Last chunk in stream
  timestamp: string;
}

/**
 * Stream event types
 */
export enum StreamEvent {
  START = 'stream_start',
  CHUNK = 'stream_chunk',
  END = 'stream_end',
  ERROR = 'stream_error',
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Create optimistic message for immediate UI update
 * @param content - Message content
 * @param userId - User ID
 * @param sessionId - Session ID
 * @returns Optimistic message object
 */
export const createOptimisticMessage = (
  content: string,
  userId: string,
  sessionId: string
): OptimisticMessage => {
  return {
    id: `optimistic-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    session_id: sessionId,
    user_id: userId,
    role: MessageRole.USER,
    content,
    timestamp: new Date().toISOString(),
    optimistic: true,
  };
};

/**
 * Group messages by date for UI display
 * @param messages - Array of messages
 * @returns Messages grouped by date
 */
export const groupMessagesByDate = (messages: Message[]): MessageGroup[] => {
  const groups = new Map<string, Message[]>();
  
  messages.forEach(message => {
    const date = message.timestamp.split('T')[0]; // Get YYYY-MM-DD
    if (!groups.has(date)) {
      groups.set(date, []);
    }
    groups.get(date)!.push(message);
  });
  
  return Array.from(groups.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, msgs]) => ({
      date,
      messages: msgs.sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      ),
    }));
};

/**
 * Format date header for message groups
 * @param date - Date string (YYYY-MM-DD)
 * @returns Formatted string like "Today", "Yesterday", or "Monday, Jan 15"
 */
export const formatDateHeader = (date: string): string => {
  const messageDate = new Date(date);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  // Reset hours for accurate date comparison
  today.setHours(0, 0, 0, 0);
  yesterday.setHours(0, 0, 0, 0);
  messageDate.setHours(0, 0, 0, 0);
  
  if (messageDate.getTime() === today.getTime()) {
    return 'Today';
  } else if (messageDate.getTime() === yesterday.getTime()) {
    return 'Yesterday';
  } else {
    // Format as "Monday, Jan 15"
    return messageDate.toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'short',
      day: 'numeric',
    });
  }
};

/**
 * Check if message is from user
 * @param message - Message to check
 * @returns True if message is from user
 */
export const isUserMessage = (message: Message): boolean => {
  return message.role === MessageRole.USER;
};

/**
 * Check if message is from AI assistant
 * @param message - Message to check
 * @returns True if message is from assistant
 */
export const isAssistantMessage = (message: Message): boolean => {
  return message.role === MessageRole.ASSISTANT;
};

/**
 * Calculate total cost of messages
 * @param messages - Array of messages
 * @returns Total cost in dollars
 */
export const calculateTotalCost = (messages: Message[]): number => {
  return messages.reduce((total, msg) => total + (msg.cost || 0), 0);
};

/**
 * Calculate average response time
 * @param messages - Array of messages
 * @returns Average response time in milliseconds
 */
export const calculateAvgResponseTime = (messages: Message[]): number => {
  const responseTimes = messages
    .filter(msg => msg.responseTime !== undefined)
    .map(msg => msg.responseTime!);
  
  if (responseTimes.length === 0) return 0;
  
  return responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
};
