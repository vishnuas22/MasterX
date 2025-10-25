/**
 * Authentication API Service
 * 
 * Handles all authentication-related API calls:
 * - User registration
 * - User login
 * - Token refresh
 * - Token verification
 * - Profile updates
 * - Logout (token invalidation)
 * 
 * Backend Integration:
 * POST   /api/auth/register  - Register new user
 * POST   /api/auth/login     - User login
 * POST   /api/auth/refresh   - Refresh access token
 * POST   /api/auth/logout    - Invalidate token
 * GET    /api/auth/me        - Get current user
 * 
 * @module services/api/auth.api
 */

import apiClient from './client';
import type { 
  User, 
  LoginCredentials, 
  SignupData, 
  LoginResponse 
} from '../../types/user.types';

/**
 * Authentication API endpoints
 */
export const authAPI = {
  /**
   * Register new user
   * 
   * Creates a new user account with email and password.
   * Returns JWT tokens for immediate login.
   * 
   * @param data - Signup form data (email, password, name)
   * @returns Login response with tokens and user data
   * @throws 400 - Invalid email or password
   * @throws 400 - Email already registered
   * @throws 500 - Registration failed
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
   * Returns JWT tokens on success.
   * 
   * Rate limited: 10 attempts per minute per IP
   * Account locked after 5 failed attempts (15 minutes)
   * 
   * @param credentials - Login credentials (email, password)
   * @returns Login response with tokens and user data
   * @throws 401 - Invalid email or password
   * @throws 423 - Account temporarily locked
   * @throws 429 - Too many login attempts
   * @throws 500 - Login failed
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
   * Uses refresh token to get new access token.
   * Called automatically when access token expires.
   * 
   * @param refreshToken - Valid refresh token
   * @returns New tokens and user data
   * @throws 401 - Invalid refresh token
   * @throws 500 - Token refresh failed
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
   * Requires valid access token in headers.
   * 
   * @returns User profile data
   * @throws 401 - Unauthorized (invalid or expired token)
   * @throws 404 - User not found
   * @throws 500 - Server error
   */
  getCurrentUser: async (): Promise<User> => {
    const { data } = await apiClient.get<User>('/api/auth/me');
    return data;
  },

  /**
   * Logout user
   * 
   * Invalidates current access token.
   * Token is added to blacklist.
   * 
   * @returns Success message
   * @throws 500 - Logout failed
   */
  logout: async (): Promise<{ message: string }> => {
    const { data } = await apiClient.post<{ message: string }>('/api/auth/logout');
    return data;
  },
};