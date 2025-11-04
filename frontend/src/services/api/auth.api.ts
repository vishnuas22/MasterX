/**
 * Authentication API Service
 * 
 * Handles all authentication-related API calls:
 * - User registration with email verification
 * - User login with JWT token generation
 * - Token refresh for session management
 * - User logout with token invalidation
 * - Current user profile retrieval
 * 
 * Security Features:
 * - JWT token-based authentication (access + refresh tokens)
 * - Password hashing with Bcrypt (12 rounds)
 * - Rate limiting (10 attempts/min per IP)
 * - Account lockout (5 failed attempts → 15 min lock)
 * - Token blacklisting on logout
 * 
 * Backend Integration:
 * POST   /api/auth/register  - Register new user
 * POST   /api/auth/login     - User login
 * POST   /api/auth/refresh   - Refresh access token
 * POST   /api/auth/logout    - Invalidate token
 * GET    /api/auth/me        - Get current user
 * 
 * Future Enhancements (Not Yet Implemented):
 * - Password reset functionality
 * - Profile update endpoint
 * - Email verification
 * - Two-factor authentication
 * 
 * @module services/api/auth.api
 */

import apiClient from './client';
import type { 
  User, 
  LoginCredentials, 
  SignupData, 
  LoginResponse,
  UserApiResponse,
  EmailVerificationResponse,
  ResendVerificationResponse
} from '../../types/user.types';

/**
 * Authentication API endpoints
 */
export const authAPI = {
  /**
   * Register new user
   * 
   * Creates a new user account with email and password.
   * Returns JWT tokens for immediate login (no email verification required).
   * 
   * Password Requirements:
   * - Minimum 8 characters
   * - At least one uppercase letter
   * - At least one lowercase letter
   * - At least one number
   * - At least one special character
   * 
   * @param data - Signup form data (email, password, name)
   * @returns Login response with tokens and user data
   * @throws 400 - Invalid email format
   * @throws 400 - Password does not meet requirements
   * @throws 409 - Email already registered
   * @throws 500 - Registration failed
   * 
   * @example
   * ```typescript
   * const response = await authAPI.signup({
   *   email: 'john@example.com',
   *   password: 'SecurePass123!',
   *   name: 'John Doe'
   * });
   * 
   * // Store tokens
   * localStorage.setItem('access_token', response.access_token);
   * localStorage.setItem('refresh_token', response.refresh_token);
   * 
   * // Access user data
   * console.log(response.user.name); // "John Doe"
   * ```
   */
  signup: async (data: SignupData): Promise<LoginResponse> => {
    const { data: response } = await apiClient.post<LoginResponse>(
      '/api/auth/register',
      {
        email: data.email,
        password: data.password,
        name: data.name,
      }
    );
    return response;
  },

  /**
   * Login user
   * 
   * Authenticates user with email and password.
   * Returns JWT tokens (access token + refresh token) on success.
   * 
   * Security:
   * - Rate limited: 10 attempts per minute per IP
   * - Account locked after 5 failed attempts (15 minutes)
   * - Secure password comparison with timing-attack protection
   * 
   * @param credentials - Login credentials (email, password)
   * @returns Login response with tokens and user data
   * @throws 401 - Invalid email or password
   * @throws 423 - Account temporarily locked (too many failed attempts)
   * @throws 429 - Too many login attempts (rate limit exceeded)
   * @throws 500 - Login failed (server error)
   * 
   * @example
   * ```typescript
   * try {
   *   const response = await authAPI.login({
   *     email: 'john@example.com',
   *     password: 'SecurePass123!'
   *   });
   *   
   *   console.log('Login successful!');
   *   console.log('Access token:', response.access_token);
   * } catch (error) {
   *   if (error.response?.status === 423) {
   *     console.error('Account locked. Try again in 15 minutes.');
   *   }
   * }
   * ```
   */
  login: async (credentials: LoginCredentials): Promise<LoginResponse> => {
    const { data: response } = await apiClient.post<LoginResponse>(
      '/api/auth/login',
      credentials
    );
    return response;
  },

  /**
   * Refresh access token
   * 
   * Uses refresh token to obtain a new access token.
   * Called automatically when access token expires (30 minutes).
   * 
   * Token Lifecycle:
   * - Access token: Valid for 30 minutes (matches backend ACCESS_TOKEN_EXPIRE_MINUTES)
   * - Refresh token: Valid for 7 days (matches backend REFRESH_TOKEN_EXPIRE_DAYS)
   * - Auto-refresh: Triggered 5 minutes before expiration
   * 
   * @param refreshToken - Valid refresh token
   * @returns New tokens and updated user data
   * @throws 401 - Invalid or expired refresh token
   * @throws 500 - Token refresh failed
   * 
   * @example
   * ```typescript
   * // Called automatically by authStore
   * const response = await authAPI.refresh(storedRefreshToken);
   * 
   * // Update stored tokens
   * localStorage.setItem('access_token', response.access_token);
   * localStorage.setItem('refresh_token', response.refresh_token);
   * ```
   */
  refresh: async (refreshToken: string): Promise<LoginResponse> => {
    const { data: response } = await apiClient.post<LoginResponse>(
      '/api/auth/refresh',
      {
        refresh_token: refreshToken,
      }
    );
    return response;
  },

  /**
   * Get current user information
   * 
   * Returns profile information for authenticated user.
   * Requires valid access token in Authorization header.
   * 
   * Backend returns UserResponse (id, email, name, subscription_tier, etc.)
   * Frontend must adapt this to full User type with nested objects.
   * 
   * Automatically called by authStore to:
   * - Verify token validity on app start
   * - Restore user session from localStorage
   * - Refresh user data after updates
   * 
   * @returns UserApiResponse from backend (to be adapted by authStore)
   * @throws 401 - Unauthorized (invalid or expired token)
   * @throws 404 - User not found (account deleted)
   * @throws 500 - Server error
   * 
   * @example
   * ```typescript
   * // Get current user (token from headers automatically)
   * const apiUser = await authAPI.getCurrentUser();
   * 
   * // Adapt to frontend User type
   * const user = adaptUserApiResponse(apiUser);
   * 
   * console.log(user.id);     // "user-123"
   * console.log(user.email);  // "john@example.com"
   * console.log(user.name);   // "John Doe"
   * ```
   */
  getCurrentUser: async (): Promise<UserApiResponse> => {
    const { data } = await apiClient.get<UserApiResponse>('/api/auth/me');
    return data;
  },

  /**
   * Logout user
   * 
   * Invalidates current access token by adding it to blacklist.
   * Token cannot be used again after logout.
   * 
   * Also clears:
   * - Access token from memory
   * - Refresh token from localStorage
   * - User data from authStore
   * 
   * @returns Success message
   * @throws 500 - Logout failed
   * 
   * @example
   * ```typescript
   * await authAPI.logout();
   * 
   * // Clean up local storage
   * localStorage.removeItem('access_token');
   * localStorage.removeItem('refresh_token');
   * 
   * // Redirect to login
   * window.location.href = '/login';
   * ```
   */
  logout: async (): Promise<{ message: string }> => {
    const { data } = await apiClient.post<{ message: string }>('/api/auth/logout');
    return data;
  },

  /**
   * Update user profile
   * 
   * Updates the authenticated user's profile information.
   * Only provided fields will be updated (partial updates supported).
   * 
   * Updatable Fields:
   * - name: string (1-100 characters)
   * - learning_preferences: LearningPreferences object
   * - emotional_profile: EmotionalProfile object
   * 
   * Backend Endpoint: PATCH /api/auth/profile
   * Auth Required: Yes (JWT token)
   * 
   * @param updates - Partial user data to update
   * @returns Updated user profile
   * @throws 400 - Invalid input data
   * @throws 401 - Unauthorized (invalid or missing token)
   * @throws 404 - User not found
   * @throws 500 - Update failed
   * 
   * @example
   * ```typescript
   * const updatedUser = await authAPI.updateProfile({
   *   name: 'Jane Smith',
   *   learning_preferences: {
   *     preferred_subjects: ['math', 'science'],
   *     learning_style: 'visual',
   *     difficulty_preference: 'adaptive'
   *   }
   * });
   * ```
   */
  updateProfile: async (updates: Partial<User>): Promise<UserApiResponse> => {
    // Build request payload matching backend UpdateProfileRequest model
    const payload: {
      name?: string;
      learning_preferences?: typeof updates.learning_preferences;
      emotional_profile?: typeof updates.emotional_profile;
    } = {};

    // Only include fields that are actually being updated
    if (updates.name !== undefined) {
      payload.name = updates.name;
    }
    if (updates.learning_preferences !== undefined) {
      payload.learning_preferences = updates.learning_preferences;
    }
    if (updates.emotional_profile !== undefined) {
      payload.emotional_profile = updates.emotional_profile;
    }

    const { data } = await apiClient.patch<UserApiResponse>(
      '/api/auth/profile',
      payload
    );
    
    return data;
  },

  /**
   * Request password reset
   * 
   * Generates a reset token and saves it to the user's account.
   * Token is valid for 1 hour.
   * 
   * Security: Always returns success (even if email doesn't exist)
   * to prevent email enumeration attacks.
   * 
   * Backend: POST /api/auth/password-reset-request
   * 
   * @param email - User's email address
   * @returns Success message
   * 
   * @example
   * ```typescript
   * const result = await authAPI.requestPasswordReset('user@example.com');
   * // result: { message: "If the email exists, a password reset link has been sent" }
   * ```
   */
  requestPasswordReset: async (email: string): Promise<{ message: string; note?: string }> => {
    const { data } = await apiClient.post<{ message: string; note?: string }>(
      '/api/auth/password-reset-request',
      { email: email.toLowerCase().trim() }
    );
    
    return data;
  },

  /**
   * Reset password with token
   * 
   * Validates the reset token and updates the user's password.
   * Token must be valid and not expired.
   * 
   * Backend: POST /api/auth/password-reset-confirm
   * 
   * @param token - Reset token from email link
   * @param newPassword - New password (min 8 characters)
   * @returns Success message
   * @throws 400 - Invalid or expired token
   * @throws 500 - Server error
   * 
   * @example
   * ```typescript
   * const result = await authAPI.resetPassword('token-from-email', 'NewPass123!');
   * // result: { message: "Password reset successful. You can now login with your new password." }
   * ```
   */
  resetPassword: async (
    token: string,
    newPassword: string
  ): Promise<{ message: string }> => {
    const { data } = await apiClient.post<{ message: string }>(
      '/api/auth/password-reset-confirm',
      {
        token: token.trim(),
        new_password: newPassword
      }
    );
    
    return data;
  },

  /**
   * Verify email with token
   * 
   * Verifies user's email address using the token from the verification email.
   * Marks the user as verified in the database.
   * 
   * Backend: GET /api/auth/verify-email/{token}
   * Auth Required: No (token is in URL)
   * 
   * @param token - Email verification token from URL
   * @returns Verification response with user data
   * @throws 400 - Invalid or expired token
   * @throws 404 - User not found
   * @throws 500 - Server error
   * 
   * @example
   * ```typescript
   * const result = await authAPI.verifyEmail('token-from-email');
   * // result: { message: "Email verified successfully!", user: {...} }
   * ```
   */
  verifyEmail: async (token: string): Promise<EmailVerificationResponse> => {
    const { data } = await apiClient.get<EmailVerificationResponse>(
      `/api/auth/verify-email/${token.trim()}`
    );
    
    return data;
  },

  /**
   * Resend verification email
   * 
   * Generates a new verification token and sends a new verification email
   * to the currently logged-in user.
   * 
   * Backend: POST /api/auth/resend-verification
   * Auth Required: Yes (JWT token)
   * 
   * @returns Success message with email address
   * @throws 400 - Email already verified
   * @throws 401 - Not authenticated
   * @throws 500 - Server error
   * 
   * @example
   * ```typescript
   * const result = await authAPI.resendVerification();
   * // result: { message: "Verification email sent successfully...", email: "user@example.com" }
   * ```
   */
  resendVerification: async (): Promise<ResendVerificationResponse> => {
    const { data } = await apiClient.post<ResendVerificationResponse>(
      '/api/auth/resend-verification'
    );
    
    return data;
  },
};