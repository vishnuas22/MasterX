/**
 * Toast Notification System
 * 
 * WCAG 2.1 AA Compliant:
 * - role="status" for dynamic content
 * - Sufficient contrast (icons + colors)
 * - Screen reader announcements
 * 
 * Performance:
 * - CSS animations (60fps)
 * - Portal rendering (no layout shift)
 * - Automatic cleanup
 */

import React from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';
import { create } from 'zustand';

// ============================================================================
// TYPES
// ============================================================================

export type ToastVariant = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  variant: ToastVariant;
  title: string;
  description?: string;
  duration?: number; // milliseconds, 0 = no auto-dismiss
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface ToastOptions {
  variant?: ToastVariant;
  description?: string;
  duration?: number;
  action?: Toast['action'];
}

// ============================================================================
// STORE (Zustand)
// ============================================================================

interface ToastStore {
  toasts: Toast[];
  addToast: (title: string, options?: ToastOptions) => string;
  removeToast: (id: string) => void;
  clearAll: () => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  
  addToast: (title, options = {}) => {
    const id = `toast-${Date.now()}-${Math.random()}`;
    const toast: Toast = {
      id,
      title,
      variant: options.variant || 'info',
      description: options.description,
      duration: options.duration !== undefined ? options.duration : 5000,
      action: options.action,
    };

    set((state) => ({
      // Limit to 3 toasts max (UX research-backed)
      toasts: [...state.toasts.slice(-2), toast],
    }));

    // Auto-dismiss if duration > 0
    if (toast.duration && toast.duration > 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      }, toast.duration);
    }

    return id;
  },
  
  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },
  
  clearAll: () => {
    set({ toasts: [] });
  },
}));

// ============================================================================
// HELPER FUNCTIONS (Programmatic API)
// ============================================================================

export const toast = {
  success: (title: string, options?: Omit<ToastOptions, 'variant'>) =>
    useToastStore.getState().addToast(title, { ...options, variant: 'success' }),
  
  error: (title: string, options?: Omit<ToastOptions, 'variant'>) =>
    useToastStore.getState().addToast(title, { ...options, variant: 'error' }),
  
  warning: (title: string, options?: Omit<ToastOptions, 'variant'>) =>
    useToastStore.getState().addToast(title, { ...options, variant: 'warning' }),
  
  info: (title: string, options?: Omit<ToastOptions, 'variant'>) =>
    useToastStore.getState().addToast(title, { ...options, variant: 'info' }),
  
  custom: (title: string, options?: ToastOptions) =>
    useToastStore.getState().addToast(title, options),
  
  dismiss: (id: string) => useToastStore.getState().removeToast(id),
  
  dismissAll: () => useToastStore.getState().clearAll(),
};

// ============================================================================
// VARIANT STYLES
// ============================================================================

const variantStyles: Record<ToastVariant, {
  bg: string;
  border: string;
  iconColor: string;
}> = {
  success: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-green-200 dark:border-green-800',
    iconColor: 'text-green-600 dark:text-green-400',
  },
  error: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    border: 'border-red-200 dark:border-red-800',
    iconColor: 'text-red-600 dark:text-red-400',
  },
  warning: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    border: 'border-yellow-200 dark:border-yellow-800',
    iconColor: 'text-yellow-600 dark:text-yellow-400',
  },
  info: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-200 dark:border-blue-800',
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
};

// ============================================================================
// ICONS
// ============================================================================

const CheckCircleIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
  </svg>
);

const AlertCircleIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
  </svg>
);

const AlertTriangleIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
  </svg>
);

const InfoIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
  </svg>
);

const XIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
  </svg>
);

const iconMap: Record<ToastVariant, React.FC> = {
  success: CheckCircleIcon,
  error: AlertCircleIcon,
  warning: AlertTriangleIcon,
  info: InfoIcon,
};

// ============================================================================
// SINGLE TOAST COMPONENT
// ============================================================================

const ToastItem = React.memo<{ toast: Toast }>(({ toast }) => {
  const removeToast = useToastStore((state) => state.removeToast);
  const style = variantStyles[toast.variant];
  const Icon = iconMap[toast.variant];

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className={clsx(
        'relative flex items-start gap-3 p-4 rounded-lg shadow-lg border backdrop-blur-xl',
        'min-w-[320px] max-w-[420px]',
        'animate-slideInRight',
        style.bg,
        style.border,
        'hover:shadow-xl transition-shadow cursor-pointer'
      )}
      onClick={() => removeToast(toast.id)}
    >
      {/* Icon */}
      <div className={clsx('flex-shrink-0 mt-0.5', style.iconColor)}>
        <Icon />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{toast.title}</p>
        {toast.description && (
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{toast.description}</p>
        )}
        {toast.action && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              toast.action!.onClick();
              removeToast(toast.id);
            }}
            className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline mt-2"
          >
            {toast.action.label}
          </button>
        )}
      </div>

      {/* Close button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          removeToast(toast.id);
        }}
        className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        aria-label="Dismiss notification"
      >
        <XIcon />
      </button>
    </div>
  );
});

ToastItem.displayName = 'ToastItem';

// ============================================================================
// TOAST CONTAINER (Portal)
// ============================================================================

export const ToastContainer = React.memo(() => {
  const toasts = useToastStore((state) => state.toasts);
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return createPortal(
    <div
      className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      aria-label="Notifications"
      role="region"
    >
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} />
        </div>
      ))}
    </div>,
    document.body
  );
});

ToastContainer.displayName = 'ToastContainer';

export default ToastContainer;
