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
  // Additional fields from backend UserDocument
  is_active: boolean;
  is_verified: boolean;
  failed_login_attempts?: number;
  locked_until?: string | null; // ISO 8601
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
  expires_in: number; // seconds
  user: {
    id: string;
    email: string;
    name: string;
  };
}

/**
 * Registration Response
 * Can be either immediate login OR email verification required
 */
export interface RegistrationResponse {
  // Email verification flow
  message?: string;
  email?: string;
  requires_verification?: boolean;
  
  // Immediate login flow (if email verification disabled)
  access_token?: string;
  refresh_token?: string;
  token_type?: 'Bearer';
  expires_in?: number;
  user?: {
    id: string;
    email: string;
    name: string;
  };
}

export interface TokenVerifyResponse {
  user_id: string;
  email: string;
  exp: number; // Unix timestamp
}

/**
 * Email Verification Response
 * Response from email verification endpoint
 * Includes tokens for immediate login after successful verification
 */
export interface EmailVerificationResponse {
  message: string;
  user: {
    id: string;
    email: string;
    name: string;
    is_verified: boolean;
  };
  // Tokens for immediate login (only present on successful verification)
  access_token?: string;
  refresh_token?: string;
  token_type?: 'Bearer';
  expires_in?: number;
}

/**
 * Resend Verification Response
 * Response from resend verification email endpoint
 */
export interface ResendVerificationResponse {
  message: string;
  email: string;
}

/**
 * User API Response from Backend
 * Matches backend UserResponse model exactly (models.py lines 436-445)
 */
export interface UserApiResponse {
  id: string;
  email: string;
  name: string;
  subscription_tier: string;
  total_sessions: number;
  created_at: string;
  last_active: string;
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

/**
 * Adapter: Convert Backend UserApiResponse to Frontend User Type
 * 
 * Backend returns minimal user data without nested objects.
 * Frontend needs full User type with learning_preferences and emotional_profile.
 * 
 * @param apiUser - User data from backend API
 * @returns Complete User object with default preferences
 */
export const adaptUserApiResponse = (apiUser: UserApiResponse): User => {
  return {
    id: apiUser.id,
    email: apiUser.email,
    name: apiUser.name,
    created_at: apiUser.created_at,
    subscription_tier: (apiUser.subscription_tier as SubscriptionTier) || SubscriptionTier.FREE,
    total_sessions: apiUser.total_sessions,
    last_active: apiUser.last_active,
    // Default learning preferences (will be updated from backend later)
    learning_preferences: {
      preferred_subjects: [],
      learning_style: LearningStyle.VISUAL,
      difficulty_preference: 'adaptive',
      session_length_minutes: 30,
      notifications_enabled: true,
    },
    // Default emotional profile (will be updated from backend later)
    emotional_profile: {
      baseline_engagement: 0.5,
      frustration_threshold: 0.7,
      celebration_responsiveness: 0.5,
    },
  };
};

// ============================================================================
// SESSION TYPES
// ============================================================================

export enum SessionStatus {
  ACTIVE = 'active',
  COMPLETED = 'completed',
  ABANDONED = 'abandoned',
}

export interface LearningSession {
  id: string; // UUID
  user_id: string;
  started_at: string; // ISO 8601
  ended_at?: string | null;
  current_topic?: string;
  assigned_provider?: string;
  total_messages: number;
  total_tokens: number;
  total_cost: number;
  avg_response_time_ms: number;
  emotion_trajectory: string[];
  performance_score: number; // 0.0 - 1.0
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
