import { useAuthStore } from '@/store/authStore';
import { useNavigate } from 'react-router-dom';
import { useUIStore } from '@/store/uiStore';
import type { LoginCredentials, SignupData } from '@/types/user.types';

/**
 * Authentication hook - Simplified auth operations with navigation
 * 
 * Features:
 * - Login/logout/signup with automatic navigation
 * - Toast notifications for all actions
 * - Error handling
 * - Type-safe operations
 */
export const useAuth = () => {
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
    clearError,
  } = useAuthStore();

  /**
   * Login with credentials
   * Automatically navigates to /app on success
   */
  const login = async (credentials: LoginCredentials) => {
    try {
      await storeLogin(credentials);
      showToast({
        type: 'success',
        message: 'Welcome back!',
      });
      navigate('/app');
    } catch (error: any) {
      showToast({
        type: 'error',
        message: error.message || 'Login failed',
      });
    }
  };

  /**
   * Signup new user
   * Automatically navigates to /onboarding on success
   */
  const signup = async (data: SignupData) => {
    try {
      await storeSignup(data);
      showToast({
        type: 'success',
        message: 'Account created successfully!',
      });
      navigate('/onboarding');
    } catch (error: any) {
      showToast({
        type: 'error',
        message: error.message || 'Signup failed',
      });
    }
  };

  /**
   * Logout user
   * Clears all auth state and navigates to home
   */
  const logout = () => {
    storeLogout();
    showToast({
      type: 'info',
      message: 'Logged out successfully',
    });
    navigate('/');
  };

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
    clearError,
  };
};
