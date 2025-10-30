/**
 * Authentication Store - JWT Token Management & User State
 * 
 * **Purpose:** Manage user authentication, JWT tokens, profile data
 * 
 * **Features:**
 * 1. Centralized auth state (logged in/out)
 * 2. Token management (access + refresh tokens)
 * 3. Automatic token refresh
 * 4. User profile caching
 * 5. Login/logout/signup flows
 * 6. Secure token storage
 * 
 * **Security:**
 * - JWT tokens stored securely in localStorage
 * - Auto token refresh before expiration
 * - Rate limiting awareness
 * - Account lock detection
 * 
 * **Performance:**
 * - Zustand optimized re-renders
 * - LocalStorage persistence
 * - Minimal bundle size
 * 
 * **Backend Integration:**
 * - POST /api/auth/login - Email/password authentication
 * - POST /api/auth/register - User registration
 * - POST /api/auth/refresh - Token refresh
 * - POST /api/auth/logout - Token invalidation
 * - GET /api/auth/me - Get current user
 * 
 * @module store/authStore
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authAPI } from '@/services/api/auth.api';
import type { User, LoginCredentials, SignupData } from '@/types/user.types';
import { adaptUserApiResponse } from '@/types/user.types';

// ============================================================================
// TYPES
// ============================================================================

interface AuthState {
  // State
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  lastRefreshTime: number | null;
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  signup: (data: SignupData) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
  refreshAccessToken: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<void>;
  clearError: () => void;
}

// ============================================================================
// TOKEN MANAGEMENT
// ============================================================================

/**
 * Check if token is expired or about to expire (within 5 minutes)
 */
const isTokenExpiringSoon = (token: string | null): boolean => {
  if (!token) return true;
  
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const expirationTime = payload.exp * 1000; // Convert to milliseconds
    const currentTime = Date.now();
    const fiveMinutes = 5 * 60 * 1000;
    
    return (expirationTime - currentTime) < fiveMinutes;
  } catch {
    return true; // If we can't parse, treat as expired
  }
};

// ============================================================================
// STORE
// ============================================================================

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // Initial state
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      lastRefreshTime: null,
      
      // -----------------------------------------------------------------------
      // LOGIN
      // -----------------------------------------------------------------------
      
      /**
       * Login with email and password
       * 
       * - Authenticates user with backend
       * - Stores access and refresh tokens
       * - Fetches user profile
       * - Sets authenticated state
       * 
       * @throws Error if login fails (invalid credentials, account locked, rate limited)
       */
      login: async (credentials) => {
        set({ isLoading: true, error: null });
        
        try {
          console.log('🔐 Starting login process...');
          
          // Step 1: Authenticate with backend
          const response = await authAPI.login(credentials);
          console.log('✓ Login API call successful, tokens received');
          
          // Step 2: CRITICAL - Set tokens in state FIRST
          // This ensures the API interceptor has access when we call getCurrentUser
          set({
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
            lastRefreshTime: Date.now(),
          });
          console.log('✓ Tokens set in Zustand state');
          
          // Step 3: Store in localStorage for interceptor fallback and persistence
          localStorage.setItem('jwt_token', response.access_token);
          localStorage.setItem('refresh_token', response.refresh_token);
          console.log('✓ Tokens stored in localStorage');
          
          // Step 4: Small delay to ensure Zustand state propagates to interceptors
          // This prevents race condition where getCurrentUser fires before token is available
          await new Promise(resolve => setTimeout(resolve, 10));
          
          // Step 5: Fetch user profile with token attached
          console.log('→ Fetching user profile...');
          const apiUser = await authAPI.getCurrentUser();
          console.log('✓ User profile fetched:', apiUser);
          
          // Step 6: Adapt backend response to frontend User type
          const user = adaptUserApiResponse(apiUser);
          console.log('✓ User data adapted for frontend');
          
          // Step 7: Update state with user info
          set({
            user,
            isLoading: false,
            error: null,
          });
          
          console.log('✅ Login complete! User:', user.name);
        } catch (error: any) {
          console.error('❌ Login failed:', error);
          
          // Clear any stored tokens on error
          localStorage.removeItem('jwt_token');
          localStorage.removeItem('refresh_token');
          
          // Parse error message for better UX
          let errorMessage = 'Login failed. Please try again.';
          
          if (error.response?.status === 401) {
            errorMessage = 'Invalid email or password';
          } else if (error.response?.status === 423) {
            errorMessage = 'Account temporarily locked. Please try again in 15 minutes.';
          } else if (error.response?.status === 429) {
            errorMessage = 'Too many login attempts. Please try again later.';
          } else if (error.message) {
            errorMessage = error.message;
          }
          
          set({
            error: errorMessage,
            isLoading: false,
            isAuthenticated: false,
            accessToken: null,
            refreshToken: null,
          });
          
          throw new Error(errorMessage);
        }
      },
      
      // -----------------------------------------------------------------------
      // SIGNUP
      // -----------------------------------------------------------------------
      
      /**
       * Register new user
       * 
       * - Creates new account
       * - Automatically logs in user
       * - Stores tokens
       * - Fetches user profile
       * 
       * @throws Error if signup fails (email exists, invalid data, rate limited)
       */
      signup: async (data) => {
        set({ isLoading: true, error: null });
        
        try {
          console.log('📝 Starting signup process...');
          
          // Step 1: Register with backend
          const response = await authAPI.signup(data);
          console.log('✓ Signup API call successful, tokens received');
          
          // Step 2: CRITICAL - Set tokens in state FIRST
          set({
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
            lastRefreshTime: Date.now(),
          });
          console.log('✓ Tokens set in Zustand state');
          
          // Step 3: Store tokens in localStorage for backup and interceptor fallback
          localStorage.setItem('jwt_token', response.access_token);
          localStorage.setItem('refresh_token', response.refresh_token);
          console.log('✓ Tokens stored in localStorage');
          
          // Step 4: Small delay to ensure state propagates
          await new Promise(resolve => setTimeout(resolve, 10));
          
          // Step 5: Fetch user profile with token attached
          console.log('→ Fetching user profile...');
          const apiUser = await authAPI.getCurrentUser();
          console.log('✓ User profile fetched:', apiUser);
          
          // Step 6: Adapt backend response to frontend User type
          const user = adaptUserApiResponse(apiUser);
          console.log('✓ User data adapted for frontend');
          
          // Step 7: Update state with user info
          set({
            user,
            isLoading: false,
            error: null,
          });
          
          console.log('✅ Signup complete! Welcome,', user.name);
        } catch (error: any) {
          console.error('❌ Signup failed:', error);
          
          // Clear tokens on error
          localStorage.removeItem('jwt_token');
          localStorage.removeItem('refresh_token');
          
          // Parse error message
          let errorMessage = 'Signup failed. Please try again.';
          
          if (error.response?.status === 400) {
            if (error.response.data?.detail?.includes('email')) {
              errorMessage = 'This email is already registered';
            } else {
              errorMessage = 'Invalid signup data. Please check your information.';
            }
          } else if (error.response?.status === 429) {
            errorMessage = 'Too many signup attempts. Please try again later.';
          } else if (error.message) {
            errorMessage = error.message;
          }
          
          set({
            error: errorMessage,
            isLoading: false,
            isAuthenticated: false,
            accessToken: null,
            refreshToken: null,
          });
          
          throw new Error(errorMessage);
        }
      },
      
      // -----------------------------------------------------------------------
      // LOGOUT
      // -----------------------------------------------------------------------
      
      /**
       * Logout user
       * 
       * - Invalidates tokens on backend
       * - Clears local storage
       * - Resets auth state
       */
      logout: async () => {
        try {
          // Call backend logout to invalidate token
          await authAPI.logout();
        } catch {
          // Continue with logout even if backend call fails
        }
        
        // Clear storage
        localStorage.removeItem('jwt_token');
        localStorage.removeItem('refresh_token');
        
        // Reset state
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
          lastRefreshTime: null,
        });
      },
      
      // -----------------------------------------------------------------------
      // CHECK AUTH
      // -----------------------------------------------------------------------
      
      /**
       * Check authentication on app load
       * 
       * - Verifies stored tokens
       * - Fetches user profile if valid
       * - Refreshes token if needed
       * - Logs out if invalid
       */
      checkAuth: async () => {
        const token = localStorage.getItem('jwt_token');
        const refreshToken = localStorage.getItem('refresh_token');
        
        if (!token) {
          set({ isAuthenticated: false });
          return;
        }
        
        // Check if token is expiring soon
        if (isTokenExpiringSoon(token) && refreshToken) {
          try {
            await get().refreshAccessToken();
            return;
          } catch {
            // If refresh fails, try to get user with current token
          }
        }
        
        try {
          // Verify token by fetching user
          const apiUser = await authAPI.getCurrentUser();
          
          // Adapt to frontend User type
          const user = adaptUserApiResponse(apiUser);
          
          set({
            user,
            accessToken: token,
            refreshToken,
            isAuthenticated: true,
          });
        } catch (error) {
          // Token invalid or expired, logout
          await get().logout();
        }
      },
      
      // -----------------------------------------------------------------------
      // REFRESH TOKEN
      // -----------------------------------------------------------------------
      
      /**
       * Refresh access token using refresh token
       * 
       * - Called automatically when token expires
       * - Updates access token
       * - Maintains user session
       * 
       * @throws Error if refresh fails (user must login again)
       */
      refreshAccessToken: async () => {
        const { refreshToken } = get();
        
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }
        
        try {
          const response = await authAPI.refresh(refreshToken);
          
          // Update tokens
          localStorage.setItem('jwt_token', response.access_token);
          localStorage.setItem('refresh_token', response.refresh_token);
          
          set({
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            lastRefreshTime: Date.now(),
          });
        } catch (error) {
          // Refresh failed, logout user
          await get().logout();
          throw new Error('Session expired. Please login again.');
        }
      },
      
      // -----------------------------------------------------------------------
      // UPDATE PROFILE
      // -----------------------------------------------------------------------
      
      /**
       * Update user profile
       * 
       * Note: Backend profile update endpoint not yet implemented
       * Currently updates local state only
       * 
       * @param updates - Partial user data to update
       */
      updateProfile: async (updates) => {
        const { user } = get();
        if (!user) return;
        
        try {
          // TODO: Implement backend profile update when available
          // const updatedUser = await authAPI.updateProfile(user.id, updates);
          
          // For now, update local state
          set({ 
            user: { ...user, ...updates },
            error: null,
          });
        } catch (error: any) {
          set({ error: error.message || 'Failed to update profile' });
          throw error;
        }
      },
      
      // -----------------------------------------------------------------------
      // CLEAR ERROR
      // -----------------------------------------------------------------------
      
      /**
       * Clear error state
       */
      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      // Only persist these fields (don't persist loading/error states)
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        lastRefreshTime: state.lastRefreshTime,
      }),
    }
  )
);

// ============================================================================
// EXPORTS
// ============================================================================

export default useAuthStore;