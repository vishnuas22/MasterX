import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Theme = 'dark' | 'light';
type Modal = 'dashboard' | 'analytics' | 'settings' | 'profile' | null;

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
}

interface UIState {
  // State
  theme: Theme;
  activeModal: Modal;
  isSidebarOpen: boolean;
  toasts: Toast[];
  isPageLoading: boolean;
  
  // Actions
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  initializeTheme: () => void;
  openModal: (modal: Modal) => void;
  closeModal: () => void;
  toggleSidebar: () => void;
  showToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  setPageLoading: (isLoading: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // Initial state
      theme: 'dark', // Default to dark mode
      activeModal: null,
      isSidebarOpen: false,
      toasts: [],
      isPageLoading: false,
      
      // Set theme
      setTheme: (theme) => {
        set({ theme });
        document.documentElement.classList.toggle('dark', theme === 'dark');
      },
      
      // Toggle theme
      toggleTheme: () => {
        const newTheme = get().theme === 'dark' ? 'light' : 'dark';
        get().setTheme(newTheme);
      },
      
      // Initialize theme (respects system preference)
      initializeTheme: () => {
        const saved = localStorage.getItem('theme');
        if (saved) {
          get().setTheme(saved as Theme);
        } else {
          // Use system preference
          const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          get().setTheme(prefersDark ? 'dark' : 'light');
        }
      },
      
      // Open modal
      openModal: (modal) => set({ activeModal: modal }),
      
      // Close modal
      closeModal: () => set({ activeModal: null }),
      
      // Toggle sidebar
      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      
      // Show toast notification
      showToast: (toast) => {
        const id = `toast-${Date.now()}`;
        const newToast = { ...toast, id };
        
        set((state) => ({
          toasts: [...state.toasts, newToast],
        }));
        
        // Auto-remove after duration
        setTimeout(() => {
          get().removeToast(id);
        }, toast.duration || 3000);
      },
      
      // Remove toast
      removeToast: (id) => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      },
      
      // Set page loading
      setPageLoading: (isLoading) => set({ isPageLoading: isLoading }),
    }),
    {
      name: 'ui-storage',
      partialize: (state) => ({
        theme: state.theme,
        // Don't persist modals, toasts, etc.
      }),
    }
  )
);
