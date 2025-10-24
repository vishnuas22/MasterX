**Purpose:** Handle all authentication-related API calls

// **What This File Contributes:**
// 1. User registration
// 2. User login
// 3. Token verification
// 4. Password reset
// 5. Profile updates
// 6. Logout (token invalidation)

// **Implementation:**

import apiClient from './client';
import type { 
  User, 
  LoginCredentials, 
  SignupData, 
  LoginResponse,
  TokenVerifyResponse 
} from '@types/user.types';

export const authAPI = {
  /**
   * Register new user
   * POST /api/v1/auth/register
   */
  signup: async (data: SignupData): Promise<LoginResponse> => {
    const { data: response } = await apiClient.post<LoginResponse>(
      '/api/v1/auth/register',
      {
        email: data.email,
        password: data.password,
        name: data.name,
        preferences: data.preferences || {},
      }
    );
    return response;
  },

  /**
   * Login user
   * POST /api/v1/auth/login
   */
  login: async (credentials: LoginCredentials): Promise<LoginResponse> => {
    const { data: response } = await apiClient.post<LoginResponse>(
      '/api/v1/auth/login',
      credentials
    );
    return response;
  },

  /**
   * Verify JWT token and get user data
   * GET /api/v1/auth/me
   */
  verifyToken: async (token: string): Promise<User> => {
    const { data } = await apiClient.get<User>('/api/v1/auth/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return data;
  },

  /**
   * Update user profile
   * PATCH /api/v1/auth/profile
   */
  updateProfile: async (userId: string, updates: Partial<User>): Promise<User> => {
    const { data } = await apiClient.patch<User>(
      `/api/v1/auth/profile`,
      updates
    );
    return data;
  },

  /**
   * Request password reset
   * POST /api/v1/auth/password-reset
   */
  requestPasswordReset: async (email: string): Promise<{ message: string }> => {
    const { data } = await apiClient.post('/api/v1/auth/password-reset', {
      email,
    });
    return data;
  },

  /**
   * Reset password with token
   * POST /api/v1/auth/password-reset/confirm
   */
  resetPassword: async (token: string, newPassword: string): Promise<{ message: string }> => {
    const { data } = await apiClient.post('/api/v1/auth/password-reset/confirm', {
      token,
      new_password: newPassword,
    });
    return data;
  },

  /**
   * Logout user (invalidate token)
   * POST /api/v1/auth/logout
   */
  logout: async (): Promise<void> => {
    await apiClient.post('/api/v1/auth/logout');
  },
};


// **Performance Considerations:**
// - Token verification cached in authStore (avoids repeated calls)
// - Login/signup responses include user data (no extra fetch needed)
// - Logout clears all client-side data

// **Connected Files:**
// - ← `services/api/client.ts` (axios instance)
// - ← `types/user.types.ts` (type definitions)
// - → `store/authStore.ts` (uses these API calls)
// - → `pages/Login.tsx`, `pages/Signup.tsx` (authentication UI)

// **Backend Integration:**
// ```
// POST   /api/v1/auth/register              ← signup()
// POST   /api/v1/auth/login                 ← login()
// GET    /api/v1/auth/me                    ← verifyToken()
// PATCH  /api/v1/auth/profile               ← updateProfile()
// POST   /api/v1/auth/password-reset        ← requestPasswordReset()
// POST   /api/v1/auth/password-reset/confirm ← resetPassword()
// POST   /api/v1/auth/logout                ← logout()
// ```

// **Error Handling:**
// - 401: Invalid credentials → Clear error message
// - 409: Email already exists → "Email already registered"
// - 422: Validation error → Show field-specific errors
// - 500: Server error → "Please try again later"