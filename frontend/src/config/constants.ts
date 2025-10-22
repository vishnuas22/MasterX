/**
 * Application Constants
 * 
 * Centralized constants for the application
 * Following AGENTS_FRONTEND.md: Zero hardcoded values in components
 */

// ============================================================================
// APP METADATA
// ============================================================================

export const APP_NAME = 'MasterX';
export const APP_DESCRIPTION = 'AI-Powered Adaptive Learning Platform';
export const APP_VERSION = import.meta.env.VITE_APP_VERSION || '1.0.0';

// ============================================================================
// API CONFIGURATION
// ============================================================================

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001';

export const API_TIMEOUT = 30000; // 30 seconds
export const API_RETRY_COUNT = 1;
export const API_RETRY_DELAY = 1000; // 1 second

// ============================================================================
// STORAGE KEYS
// ============================================================================

export const STORAGE_KEYS = {
  AUTH_TOKEN: 'jwt_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_PROFILE: 'user_profile',
  THEME: 'theme',
  LANGUAGE: 'language',
  ONBOARDING_COMPLETE: 'onboarding_complete',
  LAST_SESSION_ID: 'last_session_id',
} as const;

// ============================================================================
// ROUTE PATHS
// ============================================================================

export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  SIGNUP: '/signup',
  ONBOARDING: '/onboarding',
  APP: '/app',
  DASHBOARD: '/app/dashboard',
  SETTINGS: '/app/settings',
  PROFILE: '/app/profile',
} as const;

// ============================================================================
// FEATURE FLAGS
// ============================================================================

export const FEATURES = {
  VOICE_ENABLED: import.meta.env.VITE_ENABLE_VOICE === 'true',
  ANALYTICS_ENABLED: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  GAMIFICATION_ENABLED: import.meta.env.VITE_ENABLE_GAMIFICATION === 'true',
  DARK_MODE_ONLY: false,
} as const;

// ============================================================================
// UI CONSTANTS
// ============================================================================

export const UI = {
  MAX_MESSAGE_LENGTH: 10000,
  TOAST_DURATION: 3000, // milliseconds
  TOAST_MAX_VISIBLE: 3,
  MODAL_ANIMATION_DURATION: 250, // milliseconds
  SIDEBAR_WIDTH: 280, // pixels
  HEADER_HEIGHT: 64, // pixels
  MESSAGE_LOAD_COUNT: 50,
  INFINITE_SCROLL_THRESHOLD: 0.8,
} as const;

// ============================================================================
// EMOTION CONSTANTS
// ============================================================================

export const EMOTION = {
  UPDATE_INTERVAL: 100, // milliseconds
  HISTORY_LIMIT: 100, // number of emotion data points to keep
  CHART_DATA_POINTS: 20,
} as const;

// ============================================================================
// ANALYTICS CONSTANTS
// ============================================================================

export const ANALYTICS = {
  SESSION_TIMEOUT: 30 * 60 * 1000, // 30 minutes
  BATCH_SIZE: 10,
  FLUSH_INTERVAL: 5000, // 5 seconds
} as const;

// ============================================================================
// PERFORMANCE CONSTANTS
// ============================================================================

export const PERFORMANCE = {
  LCP_TARGET: 2500, // milliseconds
  FID_TARGET: 100, // milliseconds
  CLS_TARGET: 0.1,
  IMAGE_LAZY_THRESHOLD: '200px',
  DEBOUNCE_DELAY: 300, // milliseconds
  THROTTLE_DELAY: 500, // milliseconds
} as const;

// ============================================================================
// VALIDATION RULES
// ============================================================================

export const VALIDATION = {
  EMAIL_REGEX: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PASSWORD_MIN_LENGTH: 8,
  PASSWORD_REGEX: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
  USERNAME_MIN_LENGTH: 3,
  USERNAME_MAX_LENGTH: 30,
  NAME_MAX_LENGTH: 100,
} as const;

// ============================================================================
// ERROR MESSAGES
// ============================================================================

export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  AUTH_REQUIRED: 'Please log in to continue.',
  SESSION_EXPIRED: 'Your session has expired. Please log in again.',
  INVALID_CREDENTIALS: 'Invalid email or password.',
  EMAIL_REQUIRED: 'Email is required.',
  PASSWORD_REQUIRED: 'Password is required.',
  PASSWORD_TOO_SHORT: `Password must be at least ${VALIDATION.PASSWORD_MIN_LENGTH} characters.`,
  PASSWORD_WEAK: 'Password must contain uppercase, lowercase, number, and special character.',
  INVALID_EMAIL: 'Please enter a valid email address.',
  GENERIC_ERROR: 'Something went wrong. Please try again.',
  RATE_LIMITED: 'Too many requests. Please slow down.',
} as const;

// ============================================================================
// SUCCESS MESSAGES
// ============================================================================

export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: 'Welcome back!',
  SIGNUP_SUCCESS: 'Account created successfully!',
  PROFILE_UPDATED: 'Profile updated successfully.',
  PASSWORD_CHANGED: 'Password changed successfully.',
  MESSAGE_SENT: 'Message sent.',
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type RoutePath = typeof ROUTES[keyof typeof ROUTES];
export type StorageKey = typeof STORAGE_KEYS[keyof typeof STORAGE_KEYS];
export type FeatureFlag = typeof FEATURES[keyof typeof FEATURES];
