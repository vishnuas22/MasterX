/**
 * Feature Flags Configuration
 * 
 * Centralized feature toggles for gradual rollout
 * Can be controlled via environment variables
 */

import { FEATURES } from './constants';

// ============================================================================
// FEATURE FLAGS
// ============================================================================

export interface FeatureFlags {
  // Core features
  chat: boolean;
  emotion: boolean;
  voiceInput: boolean;
  voiceOutput: boolean;
  
  // Advanced features
  analytics: boolean;
  gamification: boolean;
  collaboration: boolean;
  spacedRepetition: boolean;
  
  // UI features
  darkMode: boolean;
  lightMode: boolean;
  animations: boolean;
  
  // Experimental features
  aiProviderSelection: boolean;
  advancedAnalytics: boolean;
  customThemes: boolean;
  offlineMode: boolean;
  
  // Admin features
  adminPanel: boolean;
  costTracking: boolean;
}

export const featureFlags: FeatureFlags = {
  // Core features (always enabled in production)
  chat: true,
  emotion: true,
  voiceInput: FEATURES.VOICE_ENABLED,
  voiceOutput: FEATURES.VOICE_ENABLED,
  
  // Advanced features
  analytics: FEATURES.ANALYTICS_ENABLED,
  gamification: FEATURES.GAMIFICATION_ENABLED,
  collaboration: true,
  spacedRepetition: true,
  
  // UI features
  darkMode: true,
  lightMode: !FEATURES.DARK_MODE_ONLY,
  animations: true,
  
  // Experimental features (disabled by default)
  aiProviderSelection: false,
  advancedAnalytics: false,
  customThemes: false,
  offlineMode: false,
  
  // Admin features (disabled by default)
  adminPanel: false,
  costTracking: false,
};

// ============================================================================
// FEATURE FLAG HELPERS
// ============================================================================

export const isFeatureEnabled = (feature: keyof FeatureFlags): boolean => {
  return featureFlags[feature] || false;
};

export const enableFeature = (feature: keyof FeatureFlags): void => {
  featureFlags[feature] = true;
};

export const disableFeature = (feature: keyof FeatureFlags): void => {
  featureFlags[feature] = false;
};

export const toggleFeature = (feature: keyof FeatureFlags): void => {
  featureFlags[feature] = !featureFlags[feature];
};

// ============================================================================
// FEATURE GROUPS
// ============================================================================

export const FEATURE_GROUPS = {
  CORE: ['chat', 'emotion'] as const,
  VOICE: ['voiceInput', 'voiceOutput'] as const,
  LEARNING: ['analytics', 'gamification', 'spacedRepetition'] as const,
  SOCIAL: ['collaboration'] as const,
  UI: ['darkMode', 'lightMode', 'animations'] as const,
  EXPERIMENTAL: ['aiProviderSelection', 'advancedAnalytics', 'customThemes', 'offlineMode'] as const,
  ADMIN: ['adminPanel', 'costTracking'] as const,
} as const;

export const areAllFeaturesEnabled = (features: readonly (keyof FeatureFlags)[]): boolean => {
  return features.every(feature => isFeatureEnabled(feature));
};

export const areSomeFeaturesEnabled = (features: readonly (keyof FeatureFlags)[]): boolean => {
  return features.some(feature => isFeatureEnabled(feature));
};

// ============================================================================
// FEATURE DESCRIPTIONS
// ============================================================================

export const FEATURE_DESCRIPTIONS: Record<keyof FeatureFlags, string> = {
  chat: 'AI-powered chat interface',
  emotion: 'Real-time emotion detection',
  voiceInput: 'Voice-to-text input',
  voiceOutput: 'Text-to-speech output',
  analytics: 'Learning analytics dashboard',
  gamification: 'XP, levels, and achievements',
  collaboration: 'Peer learning sessions',
  spacedRepetition: 'SM-2 algorithm flashcards',
  darkMode: 'Dark theme',
  lightMode: 'Light theme',
  animations: 'UI animations and transitions',
  aiProviderSelection: 'Choose AI provider manually',
  advancedAnalytics: 'Detailed performance metrics',
  customThemes: 'Create custom color themes',
  offlineMode: 'Work without internet',
  adminPanel: 'Admin dashboard access',
  costTracking: 'View API cost breakdown',
};

export default featureFlags;
