import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authAPI } from '@/services/api/auth.api';
import type { User, LoginCredentials, SignupData } from '@/types/user.types';

interface AuthState {
  // State
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  signup: (data: SignupData) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(  persist(
    (set, get) => ({
      // Initial state
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      
      // Login action
      login: async (credentials) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authAPI.login(credentials);
          // Convert minimal user to full User type
          const fullUser: User = {
            id: response.user.id,
            email: response.user.email,
            name: response.user.name,
            created_at: new Date().toISOString(),
            learning_preferences: {
              preferred_subjects: [],
              learning_style: 'visual' as any,
              difficulty_preference: 'adaptive',
            },
            emotional_profile: {
              baseline_engagement: 0.5,
              frustration_threshold: 0.7,
              celebration_responsiveness: 0.8,
            },
            subscription_tier: 'free' as any,
            total_sessions: 0,
            last_active: new Date().toISOString(),
          };
          set({
            user: fullUser,
            token: response.access_token,
            isAuthenticated: true,
            isLoading: false,
          });
          // Store token in localStorage for API requests
          localStorage.setItem('jwt_token', response.access_token);
        } catch (error: any) {
          set({
            error: error.message || 'Login failed',
            isLoading: false,
          });
          throw error;
        }
      },
      
      // Signup action
      signup: async (data) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authAPI.signup(data);
          // Convert minimal user to full User type
          const fullUser: User = {
            id: response.user.id,
            email: response.user.email,
            name: response.user.name,
            created_at: new Date().toISOString(),
            learning_preferences: {
              preferred_subjects: [],
              learning_style: 'visual' as any,
              difficulty_preference: 'adaptive',
            },
            emotional_profile: {
              baseline_engagement: 0.5,
              frustration_threshold: 0.7,
              celebration_responsiveness: 0.8,
            },
            subscription_tier: 'free' as any,
            total_sessions: 0,
            last_active: new Date().toISOString(),
          };
          set({
            user: fullUser,
            token: response.access_token,
            isAuthenticated: true,
            isLoading: false,
          });
          localStorage.setItem('jwt_token', response.access_token);
        } catch (error: any) {
          set({
            error: error.message || 'Signup failed',
            isLoading: false,
          });
          throw error;
        }
      },
      
      // Logout action
      logout: () => {
        localStorage.removeItem('jwt_token');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
      },
      
      // Check auth on app load
      checkAuth: async () => {
        const token = localStorage.getItem('jwt_token');
        if (!token) {
          set({ isAuthenticated: false });
          return;
        }
        
        try {
          const user = await authAPI.verifyToken(token);
          set({
            user,
            token,
            isAuthenticated: true,
          });
        } catch (error) {
          // Token invalid or expired
          get().logout();
        }
      },
      
      // Update user profile
      updateProfile: async (updates) => {
        const { user } = get();
        if (!user) return;
        
        try {
          const updatedUser = await authAPI.updateProfile(user.id, updates);
          set({ user: updatedUser });
        } catch (error: any) {
          set({ error: error.message });
          throw error;
        }
      },
      
      // Clear error
      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage', // LocalStorage key
      partialize: (state) => ({
        // Only persist these fields
        token: state.token,
        user: state.user,
      }),
    }
  )
);
