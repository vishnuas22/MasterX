/**
 * User & Authentication Types
 * 
 * Matches backend models.py exactly:
 * - UserDocument (lines 107-136)
 * - RegisterRequest (lines 409-414)
 * - LoginRequest (lines 417-420)
 * - TokenResponse (lines 423-429)
 * - UserResponse (lines 436-445)
 */

// ============================================================================
// ENUMS
// ============================================================================

export enum SubscriptionTier {
  FREE = 'free',
  PRO = 'pro',
  PREMIUM = 'premium',
}

export enum LearningStyle {
  VISUAL = 'visual',
  AUDITORY = 'auditory',
  KINESTHETIC = 'kinesthetic',
  READING_WRITING = 'reading_writing',
}

// ============================================================================
// USER PROFILE TYPES
// ============================================================================

export interface LearningPreferences {
  preferred_subjects: string[];
  learning_style: LearningStyle;
  difficulty_preference: 'easy' | 'medium' | 'hard' | 'adaptive';
  session_length_minutes?: number;
  notifications_enabled?: boolean;
}

export interface EmotionalProfile {
  baseline_engagement: number; // 0.0 - 1.0
  frustration_threshold: number; // 0.0 - 1.0
  celebration_responsiveness: number; // 0.0 - 1.0
}

export interface User {
  id: string; // UUID
  email: string;
  name: string;
  created_at: string; // ISO 8601
  learning_preferences: LearningPreferences;
  emotional_profile: EmotionalProfile;
  subscription_tier: SubscriptionTier;
  total_sessions: number;
  last_active: string; // ISO 8601
}

export interface UserDocument extends User {
  is_active: boolean;
  is_verified: boolean;
  failed_login_attempts?: number;
  locked_until?: string | null;
}

// ============================================================================
// AUTHENTICATION TYPES
// ============================================================================

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface SignupData {
  email: string;
  password: string;
  name: string;
  preferences?: Partial<LearningPreferences>;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';
  expires_in: number;
  user: {
    id: string;
    email: string;
    name: string;
  };
}

export interface TokenVerifyResponse {
  user_id: string;
  email: string;
  exp: number;
}

export interface UserProfileResponse {
  id: string;
  email: string;
  name: string;
  subscription_tier: SubscriptionTier;
  total_sessions: number;
  created_at: string;
  last_active: string;
}

// ============================================================================
// SESSION TYPES
// ============================================================================

export enum SessionStatus {
  ACTIVE = 'active',
  COMPLETED = 'completed',
  ABANDONED = 'abandoned',
}

export interface LearningSession {
  id: string;
  user_id: string;
  started_at: string;
  ended_at?: string | null;
  current_topic?: string;
  assigned_provider?: string;
  total_messages: number;
  total_tokens: number;
  total_cost: number;
  avg_response_time_ms: number;
  emotion_trajectory: string[];
  performance_score: number;
  status: SessionStatus;
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

export const isUser = (obj: unknown): obj is User => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    'email' in obj &&
    'name' in obj
  );
};

export const isLoginResponse = (obj: unknown): obj is LoginResponse => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'access_token' in obj &&
    'refresh_token' in obj &&
    'user' in obj
  );
};

// ============================================================================
// HELPER TYPES
// ============================================================================

export type PartialUser = Partial<User>;
export type UserUpdatePayload = Pick<User, 'name'> & {
  learning_preferences?: Partial<LearningPreferences>;
};
