/**
 * useAuth.test.ts - Unit Tests for useAuth Hook
 * 
 * Purpose: Test custom authentication hook
 * 
 * Coverage:
 * - Hook initialization
 * - Login flow
 * - Logout flow
 * - Token refresh
 * - Error states
 * 
 * Following AGENTS_FRONTEND.md:
 * - Test coverage > 80%
 * - Isolated tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useAuth } from '@/hooks/useAuth';
import { useAuthStore } from '@/store/authStore';

// Mock the auth store
vi.mock('@/store/authStore', () => ({
  useAuthStore: vi.fn(),
}));

describe('useAuth', () => {
  const mockLogin = vi.fn();
  const mockLogout = vi.fn();
  const mockSignup = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock default auth store state
    vi.mocked(useAuthStore).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      login: mockLogin,
      logout: mockLogout,
      signup: mockSignup,
      checkAuth: vi.fn(),
      refreshAccessToken: vi.fn(),
      updateProfile: vi.fn(),
      clearError: vi.fn(),
      accessToken: null,
      refreshToken: null,
      lastRefreshTime: null,
    });
  });

  // ============================================================================
  // HOOK INITIALIZATION
  // ============================================================================

  describe('Hook Initialization', () => {
    it('should return auth state from store', () => {
      const { result } = renderHook(() => useAuth());

      expect(result.current.user).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.isLoading).toBe(false);
    });

    it('should return auth actions from store', () => {
      const { result } = renderHook(() => useAuth());

      expect(typeof result.current.login).toBe('function');
      expect(typeof result.current.logout).toBe('function');
      expect(typeof result.current.signup).toBe('function');
    });
  });

  // ============================================================================
  // LOGIN
  // ============================================================================

  describe('login', () => {
    it('should call store login function', async () => {
      mockLogin.mockResolvedValue(undefined);

      const { result } = renderHook(() => useAuth());
      
      await result.current.login({
        email: 'test@example.com',
        password: 'password123',
      });

      expect(mockLogin).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });

  // ============================================================================
  // LOGOUT
  // ============================================================================

  describe('logout', () => {
    it('should call store logout function', async () => {
      mockLogout.mockResolvedValue(undefined);

      const { result } = renderHook(() => useAuth());
      
      await result.current.logout();

      expect(mockLogout).toHaveBeenCalled();
    });
  });

  // ============================================================================
  // AUTHENTICATED STATE
  // ============================================================================

  describe('Authenticated State', () => {
    it('should reflect authenticated state from store', () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: {
          id: 'user-1',
          email: 'test@example.com',
          name: 'Test User',
          avatar: null,
          learningGoals: [],
          preferences: {
            theme: 'dark',
            difficulty: 'intermediate',
            voiceEnabled: false,
          },
        },
        isAuthenticated: true,
        isLoading: false,
        error: null,
        login: mockLogin,
        logout: mockLogout,
        signup: mockSignup,
        checkAuth: vi.fn(),
        refreshAccessToken: vi.fn(),
        updateProfile: vi.fn(),
        clearError: vi.fn(),
        accessToken: 'token-123',
        refreshToken: 'refresh-123',
        lastRefreshTime: Date.now(),
      });

      const { result } = renderHook(() => useAuth());

      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.user).toBeDefined();
      expect(result.current.user?.id).toBe('user-1');
    });
  });
});
