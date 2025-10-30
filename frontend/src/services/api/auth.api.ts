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
  UserApiResponse 
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
   * NOTE: This endpoint is not yet implemented in the backend.
   * 
   * Future implementation will support:
   * - PATCH /api/auth/profile
   * - Update name, avatar, preferences
   * - Returns updated user object
   * 
   * @throws 501 - Not Implemented
   */
  updateProfile: async (_userId: string, _updates: Partial<User>): Promise<User> => {
    throw new Error(
      'Profile update endpoint not yet implemented. ' +
      'Future feature: PATCH /api/auth/profile'
    );
  },

  /**
   * Request password reset
   * 
   * NOTE: This endpoint is not yet implemented in the backend.
   * 
   * Future implementation will:
   * - Send password reset email with secure token
   * - Token valid for 1 hour
   * - POST /api/auth/password-reset
   * 
   * @throws 501 - Not Implemented
   */
  requestPasswordReset: async (_email: string): Promise<{ message: string }> => {
    throw new Error(
      'Password reset not yet implemented. ' +
      'Future feature: POST /api/auth/password-reset'
    );
  },

  /**
   * Reset password with token
   * 
   * NOTE: This endpoint is not yet implemented in the backend.
   * 
   * Future implementation will:
   * - Verify reset token
   * - Update password
   * - Invalidate all existing tokens
   * - POST /api/auth/password-reset/confirm
   * 
   * @throws 501 - Not Implemented
   */
  resetPassword: async (
    _token: string,
    _newPassword: string
  ): Promise<{ message: string }> => {
    throw new Error(
      'Password reset confirmation not yet implemented. ' +
      'Future feature: POST /api/auth/password-reset/confirm'
    );
  },
};