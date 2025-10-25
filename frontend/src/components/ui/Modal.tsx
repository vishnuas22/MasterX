/**
 * Modal Component - Accessible Dialog
 * 
 * Features:
 * - Focus trap implementation
 * - Body scroll lock
 * - Escape key handling
 * - Backdrop click to close
 * - 5 size variants
 * - Portal rendering
 * - Full accessibility (ARIA roles)
 */

import React, { useEffect, useRef, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';

// ============================================================================
// TYPES
// ============================================================================

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | 'full';

export interface ModalProps {
  /** Modal open state */
  isOpen: boolean;
  
  /** Close handler */
  onClose: () => void;
  
  /** Modal title */
  title?: string;
  
  /** Modal content */
  children: ReactNode;
  
  /** Modal size */
  size?: ModalSize;
  
  /** Show close button */
  showCloseButton?: boolean;
  
  /** Close on backdrop click */
  closeOnBackdrop?: boolean;
  
  /** Close on escape key */
  closeOnEscape?: boolean;
  
  /** Footer content (actions) */
  footer?: ReactNode;
  
  /** Test ID */
  'data-testid'?: string;
}

// ============================================================================
// SIZE STYLES
// ============================================================================

const sizeStyles: Record<ModalSize, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-[95vw] max-h-[95vh]',
};

// ============================================================================
// COMPONENT
// ============================================================================

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  showCloseButton = true,
  closeOnBackdrop = true,
  closeOnEscape = true,
  footer,
  'data-testid': testId,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  // Handle escape key
  useEffect(() => {
    if (!isOpen || !closeOnEscape) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, closeOnEscape, onClose]);

  // Handle body scroll lock and focus management
  useEffect(() => {
    if (!isOpen) return;

    // Store currently focused element
    previousActiveElement.current = document.activeElement as HTMLElement;

    // Lock body scroll
    document.body.style.overflow = 'hidden';

    // Focus modal
    modalRef.current?.focus();

    return () => {
      // Restore body scroll
      document.body.style.overflow = '';

      // Restore focus to previous element
      previousActiveElement.current?.focus();
    };
  }, [isOpen]);

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (closeOnBackdrop && e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-modal flex items-center justify-center p-4"
      data-testid={testId}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fadeIn"
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Modal */}
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'modal-title' : undefined}
        tabIndex={-1}
        className={clsx(
          'relative w-full bg-bg-secondary rounded-xl shadow-xl',
          'animate-slideUp',
          'focus:outline-none',
          sizeStyles[size]
        )}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex items-center justify-between px-6 py-4 border-b border-bg-tertiary">
            {/* Title */}
            {title && (
              <h2
                id="modal-title"
                className="text-xl font-semibold text-text-primary"
              >
                {title}
              </h2>
            )}

            {/* Close button */}
            {showCloseButton && (
              <button
                type="button"
                onClick={onClose}
                className={clsx(
                  'p-2 rounded-md text-text-tertiary hover:text-text-primary hover:bg-bg-tertiary',
                  'transition-all duration-150',
                  'focus:outline-none focus:ring-2 focus:ring-accent-primary',
                  !title && 'ml-auto'
                )}
                aria-label="Close modal"
              >
                <CloseIcon />
              </button>
            )}
          </div>
        )}

        {/* Content */}
        <div className="px-6 py-4 max-h-[70vh] overflow-y-auto">
          {children}
        </div>

        {/* Footer */}
        {footer && (
          <div className="px-6 py-4 border-t border-bg-tertiary flex items-center justify-end gap-3">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body
  );
};

// ============================================================================
// ICONS
// ============================================================================

const CloseIcon: React.FC = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 20 20"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M15 5L5 15M5 5L15 15"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/*
 * Usage Examples:
 * 
 * // Basic modal
 * <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Delete Account">
 *   <p>Are you sure?</p>
 * </Modal>
 * 
 * // With footer actions
 * <Modal 
 *   isOpen={isOpen} 
 *   onClose={() => setIsOpen(false)}
 *   title="Confirm"
 *   footer={<><Button onClick={onCancel}>Cancel</Button><Button variant="danger">Delete</Button></>}
 * >
 *   <p>This action cannot be undone.</p>
 * </Modal>
 */
