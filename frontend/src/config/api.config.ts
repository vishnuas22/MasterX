/**
 * API Configuration
 * 
 * Centralized API endpoint definitions
 * Matches backend server.py routes exactly
 */

import { BACKEND_URL } from './constants';

// Base API URL
export const API_BASE = `${BACKEND_URL}/api`;

// ============================================================================
// HEALTH & SYSTEM ENDPOINTS
// ============================================================================

export const HEALTH_ENDPOINTS = {
  BASIC: `${API_BASE}/health`,
  DETAILED: `${API_BASE}/health/detailed`,
  MODEL_STATUS: `${API_BASE}/v1/system/model-status`,
} as const;

// ============================================================================
// AUTHENTICATION ENDPOINTS
// ============================================================================

export const AUTH_ENDPOINTS = {
  LOGIN: `${API_BASE}/auth/login`,
  REGISTER: `${API_BASE}/auth/register`,
  REFRESH: `${API_BASE}/auth/refresh`,
  LOGOUT: `${API_BASE}/auth/logout`,
  ME: `${API_BASE}/auth/me`,
  VERIFY_EMAIL: `${API_BASE}/auth/verify-email`,
  RESET_PASSWORD: `${API_BASE}/auth/reset-password`,
  CHANGE_PASSWORD: `${API_BASE}/auth/change-password`,
} as const;

// ============================================================================
// CHAT ENDPOINTS
// ============================================================================

export const CHAT_ENDPOINTS = {
  SEND_MESSAGE: `${API_BASE}/v1/chat`,
  GET_HISTORY: (sessionId: string) => `${API_BASE}/v1/chat/history/${sessionId}`,
  GET_SESSION: (sessionId: string) => `${API_BASE}/v1/chat/session/${sessionId}`,
  LIST_SESSIONS: `${API_BASE}/v1/chat/sessions`,
  DELETE_SESSION: (sessionId: string) => `${API_BASE}/v1/chat/session/${sessionId}`,
} as const;

// ============================================================================
// ANALYTICS ENDPOINTS
// ============================================================================

export const ANALYTICS_ENDPOINTS = {
  DASHBOARD: (userId: string) => `${API_BASE}/v1/analytics/dashboard/${userId}`,
  PERFORMANCE: (userId: string) => `${API_BASE}/v1/analytics/performance/${userId}`,
  EMOTION_TRENDS: (userId: string) => `${API_BASE}/v1/analytics/emotion-trends/${userId}`,
  TOPIC_MASTERY: (userId: string) => `${API_BASE}/v1/analytics/topic-mastery/${userId}`,
  LEARNING_VELOCITY: (userId: string) => `${API_BASE}/v1/analytics/learning-velocity/${userId}`,
} as const;

// ============================================================================
// GAMIFICATION ENDPOINTS
// ============================================================================

export const GAMIFICATION_ENDPOINTS = {
  STATS: (userId: string) => `${API_BASE}/v1/gamification/stats/${userId}`,
  LEADERBOARD: `${API_BASE}/v1/gamification/leaderboard`,
  ACHIEVEMENTS: `${API_BASE}/v1/gamification/achievements`,
  RECORD_ACTIVITY: `${API_BASE}/v1/gamification/record-activity`,
  UNLOCK_ACHIEVEMENT: `${API_BASE}/v1/gamification/unlock-achievement`,
} as const;

// ============================================================================
// PERSONALIZATION ENDPOINTS
// ============================================================================

export const PERSONALIZATION_ENDPOINTS = {
  PROFILE: (userId: string) => `${API_BASE}/v1/personalization/profile/${userId}`,
  RECOMMENDATIONS: (userId: string) => `${API_BASE}/v1/personalization/recommendations/${userId}`,
  LEARNING_PATH: (userId: string, topic: string) => 
    `${API_BASE}/v1/personalization/learning-path/${userId}/${topic}`,
  UPDATE_PREFERENCES: `${API_BASE}/v1/personalization/preferences`,
} as const;

// ============================================================================
// CONTENT DELIVERY ENDPOINTS
// ============================================================================

export const CONTENT_ENDPOINTS = {
  NEXT_CONTENT: (userId: string) => `${API_BASE}/v1/content/next/${userId}`,
  SEQUENCE: (userId: string, topic: string) => `${API_BASE}/v1/content/sequence/${userId}/${topic}`,
  SEARCH: `${API_BASE}/v1/content/search`,
  RECOMMEND: `${API_BASE}/v1/content/recommend`,
} as const;

// ============================================================================
// VOICE INTERACTION ENDPOINTS
// ============================================================================

export const VOICE_ENDPOINTS = {
  TRANSCRIBE: `${API_BASE}/v1/voice/transcribe`,
  SYNTHESIZE: `${API_BASE}/v1/voice/synthesize`,
  ASSESS_PRONUNCIATION: `${API_BASE}/v1/voice/assess-pronunciation`,
  VOICE_CHAT: `${API_BASE}/v1/voice/chat`,
} as const;

// ============================================================================
// SPACED REPETITION ENDPOINTS
// ============================================================================

export const SPACED_REPETITION_ENDPOINTS = {
  DUE_CARDS: (userId: string) => `${API_BASE}/v1/spaced-repetition/due-cards/${userId}`,
  CREATE_CARD: `${API_BASE}/v1/spaced-repetition/create-card`,
  REVIEW_CARD: `${API_BASE}/v1/spaced-repetition/review-card`,
  STATS: (userId: string) => `${API_BASE}/v1/spaced-repetition/stats/${userId}`,
  DELETE_CARD: (cardId: string) => `${API_BASE}/v1/spaced-repetition/card/${cardId}`,
} as const;

// ============================================================================
// COLLABORATION ENDPOINTS
// ============================================================================

export const COLLABORATION_ENDPOINTS = {
  FIND_PEERS: `${API_BASE}/v1/collaboration/find-peers`,
  CREATE_SESSION: `${API_BASE}/v1/collaboration/create-session`,
  MATCH_AND_CREATE: `${API_BASE}/v1/collaboration/match-and-create`,
  JOIN: `${API_BASE}/v1/collaboration/join`,
  LEAVE: `${API_BASE}/v1/collaboration/leave`,
  SEND_MESSAGE: `${API_BASE}/v1/collaboration/send-message`,
  SESSIONS: `${API_BASE}/v1/collaboration/sessions`,
  SESSION_ANALYTICS: (sessionId: string) => `${API_BASE}/v1/collaboration/session/${sessionId}/analytics`,
  SESSION_DYNAMICS: (sessionId: string) => `${API_BASE}/v1/collaboration/session/${sessionId}/dynamics`,
} as const;

// ============================================================================
// ADMIN ENDPOINTS
// ============================================================================

export const ADMIN_ENDPOINTS = {
  COSTS: `${API_BASE}/v1/admin/costs`,
  PERFORMANCE: `${API_BASE}/v1/admin/performance`,
  CACHE: `${API_BASE}/v1/admin/cache`,
  PRODUCTION_READINESS: `${API_BASE}/v1/admin/production-readiness`,
  SYSTEM_STATUS: `${API_BASE}/v1/admin/system/status`,
  PROVIDERS: `${API_BASE}/v1/providers`,
} as const;

// ============================================================================
// BUDGET ENDPOINTS
// ============================================================================

export const BUDGET_ENDPOINTS = {
  STATUS: `${API_BASE}/v1/budget/status`,
  HISTORY: `${API_BASE}/v1/budget/history`,
} as const;

// ============================================================================
// WEBSOCKET ENDPOINTS
// ============================================================================

export const WS_ENDPOINTS = {
  CHAT: '/ws/chat',
  EMOTION: '/ws/emotion',
  COLLABORATION: '/ws/collaboration',
} as const;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export const buildEndpoint = (base: string, params?: Record<string, string>): string => {
  if (!params) return base;
  
  const query = new URLSearchParams(params).toString();
  return query ? `${base}?${query}` : base;
};

export const isValidEndpoint = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};
