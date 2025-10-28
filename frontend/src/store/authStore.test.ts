/**
 * authStore.test.ts - Unit Tests for Authentication Store
 * 
 * Purpose: Test authentication state management and token handling
 * 
 * Coverage:
 * - Login flow
 * - Signup flow
 * - Logout flow
 * - Token refresh
 * - Error handling
 * - Profile updates
 * 
 * Following AGENTS_FRONTEND.md:
 * - Test coverage > 80%
 * - Isolated tests
 * - Mock API calls
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useAuthStore } from '@/store/authStore';
import { authAPI } from '@/services/api/auth.api';
import type { LoginCredentials, SignupData } from '@/types/user.types';

// Mock the auth API
vi.mock('@/services/api/auth.api', () => ({
  authAPI: {
    login: vi.fn(),
    signup: vi.fn(),
    logout: vi.fn(),
    refresh: vi.fn(),
    getCurrentUser: vi.fn(),
  },
}));

describe('authStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useAuthStore.setState({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      lastRefreshTime: null,
    });
    
    // Clear all mocks
    vi.clearAllMocks();
    
    // Clear localStorage
    localStorage.clear();
  });

  // ============================================================================
  // INITIAL STATE
  // ============================================================================

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const state = useAuthStore.getState();
      
      expect(state.user).toBeNull();
      expect(state.accessToken).toBeNull();
      expect(state.refreshToken).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
      expect(state.lastRefreshTime).toBeNull();
    });
  });

  // ============================================================================
  // LOGIN
  // ============================================================================

  describe('login', () => {
    const mockCredentials: LoginCredentials = {
      email: 'test@example.com',
      password: 'password123',
    };

    // Backend returns access_token and refresh_token (snake_case)
    const mockLoginResponse = {
      access_token: 'access-token-123',
      refresh_token: 'refresh-token-123',
      token_type: 'Bearer'
    };

    // User is fetched separately via getCurrentUser
    const mockUser = {
      id: 'user-1',
      email: 'test@example.com',
      name: 'Test User',
      avatar: null,
      learning_goals: ['Math'],
      preferences: {
        theme: 'dark',
        difficulty: 'intermediate',
        voice_enabled: false,
      },
    };

    it('should successfully login with valid credentials', async () => {
      vi.mocked(authAPI.login).mockResolvedValue(mockLoginResponse);
      vi.mocked(authAPI.getCurrentUser).mockResolvedValue(mockUser);

      const { login } = useAuthStore.getState();
      await login(mockCredentials);

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.accessToken).toBe(mockLoginResponse.access_token);
      expect(state.refreshToken).toBe(mockLoginResponse.refresh_token);
      expect(state.isAuthenticated).toBe(true);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should set loading state during login', async () => {
      vi.mocked(authAPI.login).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(mockLoginResponse), 100))
      );
      vi.mocked(authAPI.getCurrentUser).mockResolvedValue(mockUser);

      const { login } = useAuthStore.getState();
      const loginPromise = login(mockCredentials);

      // Check loading state immediately
      const loadingState = useAuthStore.getState();
      expect(loadingState.isLoading).toBe(true);

      await loginPromise;

      // Check final state
      const finalState = useAuthStore.getState();
      expect(finalState.isLoading).toBe(false);
    });

    it('should handle login errors', async () => {
      const errorMessage = 'Invalid credentials';
      vi.mocked(authAPI.login).mockRejectedValue(new Error(errorMessage));

      const { login } = useAuthStore.getState();
      await expect(login(mockCredentials)).rejects.toThrow();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.error).toBe(errorMessage);  // Exact match
      expect(state.isLoading).toBe(false);
    });

    it('should store tokens in localStorage', async () => {
      vi.mocked(authAPI.login).mockResolvedValue(mockLoginResponse);
      vi.mocked(authAPI.getCurrentUser).mockResolvedValue(mockUser);

      const { login } = useAuthStore.getState();
      await login(mockCredentials);

      // Real implementation uses jwt_token and refresh_token keys
      expect(localStorage.getItem('jwt_token')).toBe(mockLoginResponse.access_token);
      expect(localStorage.getItem('refresh_token')).toBe(mockLoginResponse.refresh_token);
    });
  });

  // ============================================================================
  // SIGNUP
  // ============================================================================

  describe('signup', () => {
    const mockSignupData: SignupData = {
      email: 'newuser@example.com',
      password: 'password123',
      name: 'New User',
      learning_goals: ['Science'],
    };

    const mockSignupResponse = {
      access_token: 'new-access-token',
      refresh_token: 'new-refresh-token',
      token_type: 'Bearer'
    };

    const mockNewUser = {
      id: 'user-2',
      email: 'newuser@example.com',
      name: 'New User',
      avatar: null,
      learning_goals: ['Science'],
      preferences: {
        theme: 'dark',
        difficulty: 'beginner',
        voice_enabled: false,
      },
    };

    it('should successfully signup with valid data', async () => {
      vi.mocked(authAPI.signup).mockResolvedValue(mockSignupResponse);
      vi.mocked(authAPI.getCurrentUser).mockResolvedValue(mockNewUser);

      const { signup } = useAuthStore.getState();
      await signup(mockSignupData);

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockNewUser);
      expect(state.accessToken).toBe(mockSignupResponse.access_token);
      expect(state.isAuthenticated).toBe(true);
    });

    it('should handle signup errors', async () => {
      const errorMessage = 'Email already exists';
      vi.mocked(authAPI.signup).mockRejectedValue(new Error(errorMessage));

      const { signup } = useAuthStore.getState();
      await expect(signup(mockSignupData)).rejects.toThrow();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.error).toBe(errorMessage);  // Exact match
    });
  });

  // ============================================================================
  // LOGOUT
  // ============================================================================

  describe('logout', () => {
    beforeEach(() => {
      // Set up authenticated state
      useAuthStore.setState({
        user: {
          id: 'user-1',
          email: 'test@example.com',
          name: 'Test User',
          avatar: null,
          learning_goals: [],
          preferences: {
            theme: 'dark',
            difficulty: 'intermediate',
            voice_enabled: false,
          },
        },
        accessToken: 'token-123',
        refreshToken: 'refresh-123',
        isAuthenticated: true,
      });
      
      localStorage.setItem('jwt_token', 'token-123');
      localStorage.setItem('refresh_token', 'refresh-123');
    });

    it('should successfully logout', async () => {
      vi.mocked(authAPI.logout).mockResolvedValue(undefined);

      const { logout } = useAuthStore.getState();
      await logout();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.accessToken).toBeNull();
      expect(state.refreshToken).toBeNull();
      expect(state.isAuthenticated).toBe(false);
    });

    it('should clear localStorage on logout', async () => {
      vi.mocked(authAPI.logout).mockResolvedValue(undefined);

      const { logout } = useAuthStore.getState();
      await logout();

      expect(localStorage.getItem('jwt_token')).toBeNull();
      expect(localStorage.getItem('refresh_token')).toBeNull();
    });

    it('should logout even if API call fails', async () => {
      vi.mocked(authAPI.logout).mockRejectedValue(new Error('Network error'));

      const { logout } = useAuthStore.getState();
      await logout();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
    });
  });

  // ============================================================================
  // TOKEN REFRESH
  // ============================================================================

  describe('refreshAccessToken', () => {
    beforeEach(() => {
      useAuthStore.setState({
        refreshToken: 'refresh-token-123',
        isAuthenticated: true,
      });
    });

    it('should refresh access token successfully', async () => {
      const mockNewTokens = {
        access_token: 'new-access-token-456',
        refresh_token: 'new-refresh-token-789',
        token_type: 'Bearer'
      };
      
      vi.mocked(authAPI.refresh).mockResolvedValue(mockNewTokens);

      const { refreshAccessToken } = useAuthStore.getState();
      await refreshAccessToken();

      const state = useAuthStore.getState();
      expect(state.accessToken).toBe(mockNewTokens.access_token);
      expect(state.refreshToken).toBe(mockNewTokens.refresh_token);
      expect(state.lastRefreshTime).toBeGreaterThan(0);
    });

    it('should handle refresh token errors', async () => {
      vi.mocked(authAPI.refresh).mockRejectedValue(new Error('Invalid refresh token'));

      const { refreshAccessToken } = useAuthStore.getState();
      await expect(refreshAccessToken()).rejects.toThrow();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
    });
  });

  // ============================================================================
  // CHECK AUTH
  // ============================================================================

  describe('checkAuth', () => {
    it('should verify existing token and load user', async () => {
      const mockUser = {
        id: 'user-1',
        email: 'test@example.com',
        name: 'Test User',
        avatar: null,
        learning_goals: [],
        preferences: {
          theme: 'dark',
          difficulty: 'intermediate',
          voice_enabled: false,
        },
      };

      localStorage.setItem('jwt_token', 'valid-token');
      vi.mocked(authAPI.getCurrentUser).mockResolvedValue(mockUser);

      const { checkAuth } = useAuthStore.getState();
      await checkAuth();

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.isAuthenticated).toBe(true);
    });

    it('should clear auth if token is invalid', async () => {
      localStorage.setItem('jwt_token', 'invalid-token');
      vi.mocked(authAPI.getCurrentUser).mockRejectedValue(new Error('Unauthorized'));

      const { checkAuth } = useAuthStore.getState();
      await checkAuth();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(localStorage.getItem('jwt_token')).toBeNull();
    });
  });

  // ============================================================================
  // UPDATE PROFILE
  // ============================================================================

  describe('updateProfile', () => {
    beforeEach(() => {
      useAuthStore.setState({
        user: {
          id: 'user-1',
          email: 'test@example.com',
          name: 'Test User',
          avatar: null,
          learning_goals: ['Math'],
          preferences: {
            theme: 'dark',
            difficulty: 'intermediate',
            voice_enabled: false,
          },
        },
        isAuthenticated: true,
      });
    });

    it('should update user profile locally', async () => {
      const updates = { name: 'Updated Name' };
      
      // Profile update not implemented on backend yet, updates locally
      const { updateProfile } = useAuthStore.getState();
      await updateProfile(updates);

      const state = useAuthStore.getState();
      expect(state.user?.name).toBe('Updated Name');
      expect(state.error).toBeNull();
    });

    it.skip('should handle update profile errors', async () => {
      // Skip until backend implements PATCH /api/auth/profile
    });
  });

  // ============================================================================
  // ERROR HANDLING
  // ============================================================================

  describe('clearError', () => {
    it('should clear error state', () => {
      useAuthStore.setState({ error: 'Some error' });

      const { clearError } = useAuthStore.getState();
      clearError();

      const state = useAuthStore.getState();
      expect(state.error).toBeNull();
    });
  });
});
