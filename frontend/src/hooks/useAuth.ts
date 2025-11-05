/**
 * Authentication Hook - Comprehensive Auth Operations
 * 
 * **Purpose:** Simplified authentication operations with automatic navigation,
 * toast notifications, and comprehensive error handling.
 * 
 * **Features:**
 * 1. Login/Signup/Logout operations
 * 2. Automatic navigation after success
 * 3. Toast notifications for all operations
 * 4. Comprehensive error handling
 * 5. Auth state management
 * 6. Token refresh handling
 * 
 * **Usage:**
 * ```tsx
 * const { login, signup, logout, isAuthenticated, user, isLoading } = useAuth();
 * 
 * // Login
 * await login({ 
 *   email: 'user@example.com', 
 *   password: 'password' 
 * });
 * 
 * // Signup
 * await signup({ 
 *   email: 'new@example.com', 
 *   password: 'password',
 *   name: 'John Doe'
 * });
 * 
 * // Logout
 * logout();
 * ```
 * 
 * **Error Handling:**
 * - Invalid credentials (401)
 * - Account locked (423)
 * - Rate limiting (429)
 * - Email already exists (400)
 * - Network errors
 * 
 * @module hooks/useAuth
 */

import { useCallback } from 'react';
import { useAuthStore } from '@/store/authStore';
import { useNavigate } from 'react-router-dom';
import { useUIStore } from '@/store/uiStore';
import type { LoginCredentials, SignupData, User } from '@/types/user.types';

// ============================================================================
// TYPES
// ============================================================================

interface UseAuthReturn {
  // State
  user: ReturnType<typeof useAuthStore>['user'];
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<boolean>;
  signup: (data: SignupData) => Promise<boolean>;
  logout: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<void>;
  clearError: () => void;
}

// ============================================================================
// HOOK
// ============================================================================

export const useAuth = (): UseAuthReturn => {
  const navigate = useNavigate();
  const { showToast } = useUIStore();
  const {
    user,
    isAuthenticated,
    isLoading,
    error,
    login: storeLogin,
    signup: storeSignup,
    logout: storeLogout,
    updateProfile: storeUpdateProfile,
    clearError,
  } = useAuthStore();

  // -------------------------------------------------------------------------
  // LOGIN
  // -------------------------------------------------------------------------
  
  /**
   * Login with email and password
   * 
   * - Calls backend authentication
   * - Shows success/error toast
   * - Navigates to /app on success
   * - Returns boolean for success/failure
   * 
   * @param credentials - Email and password
   * @returns Promise<boolean> - true if successful, false if failed
   * 
   * @example
   * const success = await login({ 
   *   email: 'user@example.com', 
   *   password: 'SecurePass123!' 
   * });
   * if (success) {
   *   // Handle successful login
   * }
   */
  const login = useCallback(async (credentials: LoginCredentials): Promise<boolean> => {
    try {
      await storeLogin(credentials);
      
      // Show success message
      showToast({
        type: 'success',
        message: `Welcome back${user?.name ? `, ${user.name}` : ''}!`,
        duration: 3000,
      });
      
      // Navigate to main app
      navigate('/app');
      
      return true;
    } catch (error: any) {
      // Error already set in store, just show toast
      showToast({
        type: 'error',
        message: error.message || 'Login failed. Please try again.',
        duration: 5000,
      });
      
      return false;
    }
  }, [storeLogin, showToast, navigate, user]);

  // -------------------------------------------------------------------------
  // SIGNUP
  // -------------------------------------------------------------------------
  
  /**
   * Register new user
   * 
   * - Creates new account
   * - Automatically logs in
   * - Shows success/error toast
   * - Navigates to /onboarding on success
   * - Returns boolean for success/failure
   * 
   * @param data - Signup form data (email, password, name)
   * @returns Promise<boolean> - true if successful, false if failed
   * 
   * @example
   * const success = await signup({ 
   *   email: 'new@example.com', 
   *   password: 'SecurePass123!',
   *   name: 'John Doe'
   * });
   * if (success) {
   *   // Handle successful signup
   * }
   */
  const signup = useCallback(async (data: SignupData): Promise<boolean> => {
    try {
      await storeSignup(data);
      
      // Check if email verification is required
      const pendingEmail = localStorage.getItem('pending_verification_email');
      if (pendingEmail) {
        // Show info message about email verification
        showToast({
          type: 'info',
          message: `Registration successful! Please check ${pendingEmail} to verify your account.`,
          duration: 6000,
        });
        
        // Navigate to email sent page
        navigate('/email-sent', { state: { email: pendingEmail } });
        
        return true;
      }
      
      // If no verification required, show success and go to onboarding
      showToast({
        type: 'success',
        message: `Welcome to MasterX, ${data.name}! 🎉`,
        duration: 4000,
      });
      
      // Navigate to onboarding
      navigate('/onboarding');
      
      return true;
    } catch (error: any) {
      // Error already set in store, just show toast
      showToast({
        type: 'error',
        message: error.message || 'Signup failed. Please try again.',
        duration: 5000,
      });
      
      return false;
    }
  }, [storeSignup, showToast, navigate]);

  // -------------------------------------------------------------------------
  // LOGOUT
  // -------------------------------------------------------------------------
  
  /**
   * Logout current user
   * 
   * - Invalidates tokens
   * - Clears auth state
   * - Shows info toast
   * - Navigates to landing page
   * 
   * @example
   * await logout();
   */
  const logout = useCallback(async (): Promise<void> => {
    try {
      await storeLogout();
      
      // Show logout message
      showToast({
        type: 'info',
        message: 'You have been logged out successfully',
        duration: 3000,
      });
      
      // Navigate to landing page
      navigate('/');
    } catch (error: any) {
      // Even if backend logout fails, we still want to clear local state
      // So we don't show error toast here
      console.error('Logout error:', error);
      
      // Still navigate to landing
      navigate('/');
    }
  }, [storeLogout, showToast, navigate]);

  // -------------------------------------------------------------------------
  // UPDATE PROFILE
  // -------------------------------------------------------------------------
  
  /**
   * Update user profile
   * 
   * - Calls backend API to update profile
   * - Updates local state with new data
   * - Shows success/error toast
   * 
   * @param updates - Partial user data to update
   */
  const updateProfile = async (updates: Partial<User>) => {
    try {
      await storeUpdateProfile(updates);
      showToast('Profile updated successfully', 'success');
    } catch (err) {
      const errorMessage = (err as any)?.response?.data?.detail || 
                          (err as Error).message || 
                          'Failed to update profile';
      showToast(errorMessage, 'error');
      throw err;
    }
  };

  // -------------------------------------------------------------------------
  // RETURN
  // -------------------------------------------------------------------------
  
  return {
    // State
    user,
    isAuthenticated,
    isLoading,
    error,
    
    // Actions
    login,
    signup,
    logout,
    updateProfile,
    clearError,
  };
};

// ============================================================================
// EXPORTS
// ============================================================================

export default useAuth;