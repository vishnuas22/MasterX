// **Purpose:** Centralized constants and configuration values

// **What This File Contributes:**
// 1. App-wide constants
// 2. API timeouts
// 3. Cache durations
// 4. UI configuration

// **Implementation:**
// ```typescript
// /**
//  * Application Constants
//  * 
//  * Following AGENTS_FRONTEND.md:
//  * - No magic numbers in code
//  * - Centralized configuration
//  * - Type-safe constants
//  */

// ============================================================================
// APPLICATION INFO
// ============================================================================

export const APP_NAME = 'MasterX';
export const APP_VERSION = '1.0.0';
export const APP_DESCRIPTION = 'AI-Powered Adaptive Learning Platform';

// ============================================================================
// API CONFIGURATION
// ============================================================================

export const API_CONFIG = {
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
  MAX_RETRY_DELAY: 30000, // 30 seconds
} as const;

export const CACHE_DURATION = {
  USER_PROFILE: 5 * 60 * 1000, // 5 minutes
  ANALYTICS: 10 * 60 * 1000, // 10 minutes
  LEADERBOARD: 2 * 60 * 1000, // 2 minutes
  ACHIEVEMENTS: 30 * 60 * 1000, // 30 minutes
} as const;

// ============================================================================
// WEBSOCKET CONFIGURATION
// ============================================================================

export const WEBSOCKET_CONFIG = {
  RECONNECT_ATTEMPTS: 5,
  RECONNECT_DELAY: 1000,
  HEARTBEAT_INTERVAL: 30000, // 30 seconds
} as const;

// ============================================================================
// UI CONFIGURATION
// ============================================================================

export const UI_CONFIG = {
  TOAST_DURATION: 3000, // 3 seconds
  ANIMATION_DURATION: 250, // 250ms
  DEBOUNCE_DELAY: 500, // 500ms
  SCROLL_THRESHOLD: 100, // pixels
  MAX_MESSAGE_LENGTH: 10000,
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
} as const;

export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

// ============================================================================
// EMOTION CONFIGURATION
// ============================================================================

export const EMOTION_CONFIG = {
  HISTORY_LIMIT: 100, // Keep last 100 emotions
  UPDATE_INTERVAL: 1000, // Update every 1 second
  VISUALIZATION_POINTS: 50, // Chart data points
} as const;

// ============================================================================
// CHAT CONFIGURATION
// ============================================================================

export const CHAT_CONFIG = {
  MAX_MESSAGES_DISPLAY: 50,
  MESSAGE_BATCH_SIZE: 20,
  TYPING_INDICATOR_DELAY: 500,
  AUTO_SCROLL_THRESHOLD: 100,
} as const;

// ============================================================================
// STORAGE KEYS
// ============================================================================

export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_PROFILE: 'user_profile',
  THEME: 'theme',
  LANGUAGE: 'language',
  UI_PREFERENCES: 'ui_preferences',
} as const;

// ============================================================================
// ROUTES
// ============================================================================

export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  SIGNUP: '/signup',
  ONBOARDING: '/onboarding',
  APP: '/app',
  DASHBOARD: '/app/dashboard',
  ANALYTICS: '/app/analytics',
  SETTINGS: '/app/settings',
  PROFILE: '/app/profile',
} as const;

// ============================================================================
// ERROR MESSAGES
// ============================================================================

export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  AUTH_FAILED: 'Authentication failed. Please try again.',
  SESSION_EXPIRED: 'Session expired. Please log in again.',
  INVALID_INPUT: 'Invalid input. Please check your data.',
  SERVER_ERROR: 'Server error. Please try again later.',
  UNKNOWN_ERROR: 'An unknown error occurred.',
} as const;

// ============================================================================
// REGEX PATTERNS
// ============================================================================

export const REGEX_PATTERNS = {
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PASSWORD: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/, // Min 8 chars, 1 uppercase, 1 lowercase, 1 number
  URL: /^https?:\/\/.+/,
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type RouteKey = keyof typeof ROUTES;
export type StorageKey = keyof typeof STORAGE_KEYS;
// ```

// // **Key Features:**
// // 1. **Centralized:** All constants in one place
// // 2. **Type-safe:** `as const` for literal types
// // 3. **Organized:** Grouped by functionality
// // 4. **No magic numbers:** Named constants throughout
// // 5. **Easily maintainable:** Single source of truth

// // **Connected Files:**
// // - → All components import from here
// // - → No hardcoded values in codebase
// // - → Type-safe constant usage