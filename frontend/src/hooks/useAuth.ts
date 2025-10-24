// **Purpose:** Simplified authentication operations

// **What This File Contributes:**
// 1. Login/logout/signup actions
// 2. Auth state access
// 3. Loading/error states
// 4. Auto-redirect logic

// **Implementation:**
// ```typescript
import { useAuthStore } from '@store/authStore';
import { useNavigate } from 'react-router-dom';
import { useUIStore } from '@store/uiStore';
import type { LoginCredentials, SignupData } from '@types/user.types';

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


// **Benefits:**
// 1. Single hook for all auth operations
// 2. Automatic navigation after login/signup
// 3. Toast notifications included
// 4. Type-safe operations

// **Performance:**
// - No additional overhead (thin wrapper)
// - Zustand ensures minimal re-renders

// **Connected Files:**
// - ← `store/authStore.ts`
// - ← `store/uiStore.ts`
// - → `pages/Login.tsx`, `pages/Signup.tsx`
// - → `components/layout/Header.tsx`