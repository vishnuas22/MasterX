/**
 * Button Component - Universal, Accessible Button
 * 
 * Features:
 * - 4 variants: primary, secondary, ghost, danger
 * - 3 sizes: sm (32px), md (44px WCAG), lg (52px)
 * - Loading states with spinner
 * - Icon support (left/right)
 * - Full accessibility (WCAG 2.1 AA)
 * - Type-safe props
 */

import React from 'react';
import { clsx } from 'clsx';
import type { ButtonHTMLAttributes, ReactNode } from 'react';

// ============================================================================
// TYPES
// ============================================================================

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Button variant style */
  variant?: ButtonVariant;
  
  /** Button size */
  size?: ButtonSize;
  
  /** Loading state (shows spinner, disables interaction) */
  loading?: boolean;
  
  /** Full width button */
  fullWidth?: boolean;
  
  /** Icon to display (left side) */
  leftIcon?: ReactNode;
  
  /** Icon to display (right side) */
  rightIcon?: ReactNode;
  
  /** Children content */
  children: ReactNode;
  
  /** Test ID for automation */
  'data-testid'?: string;
}

// ============================================================================
// VARIANT STYLES
// ============================================================================

const variantStyles: Record<ButtonVariant, string> = {
  primary: clsx(
    'bg-accent-primary text-white',
    'hover:opacity-90',
    'active:opacity-80',
    'disabled:opacity-50 disabled:cursor-not-allowed'
  ),
  secondary: clsx(
    'bg-bg-secondary text-text-primary border border-bg-tertiary',
    'hover:bg-bg-tertiary',
    'active:bg-bg-tertiary/80',
    'disabled:opacity-50 disabled:cursor-not-allowed'
  ),
  ghost: clsx(
    'bg-transparent text-text-primary',
    'hover:bg-bg-secondary',
    'active:bg-bg-tertiary',
    'disabled:opacity-50 disabled:cursor-not-allowed'
  ),
  danger: clsx(
    'bg-accent-error text-white',
    'hover:opacity-90',
    'active:opacity-80',
    'disabled:opacity-50 disabled:cursor-not-allowed'
  ),
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm min-h-[32px]',
  md: 'px-4 py-2 text-base min-h-[44px]', // WCAG touch target
  lg: 'px-6 py-3 text-lg min-h-[52px]',
};

// ============================================================================
// COMPONENT
// ============================================================================

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      fullWidth = false,
      leftIcon,
      rightIcon,
      children,
      className,
      disabled,
      type = 'button',
      'data-testid': testId,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        type={type}
        disabled={isDisabled}
        data-testid={testId}
        className={clsx(
          // Base styles
          'inline-flex items-center justify-center gap-2',
          'font-medium rounded-md',
          'transition-all duration-150',
          'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary',
          
          // Variant styles
          variantStyles[variant],
          
          // Size styles
          sizeStyles[size],
          
          // Full width
          fullWidth && 'w-full',
          
          // Custom className
          className
        )}
        {...props}
      >
        {/* Left icon or loading spinner */}
        {loading ? (
          <LoadingSpinner size={size} />
        ) : (
          leftIcon && <span className="flex-shrink-0">{leftIcon}</span>
        )}
        
        {/* Button text */}
        <span className={clsx(loading && 'opacity-0')}>{children}</span>
        
        {/* Right icon */}
        {!loading && rightIcon && (
          <span className="flex-shrink-0">{rightIcon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// ============================================================================
// LOADING SPINNER
// ============================================================================

const LoadingSpinner: React.FC<{ size: ButtonSize }> = ({ size }) => {
  const spinnerSize = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  }[size];

  return (
    <svg
      className={clsx(spinnerSize, 'animate-spin')}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
};

/*
 * Usage Examples:
 * 
 * // Primary button
 * <Button variant="primary" onClick={handleClick}>Save Changes</Button>
 * 
 * // With icon
 * <Button variant="primary" leftIcon={<SaveIcon />}>Save</Button>
 * 
 * // Loading state
 * <Button variant="primary" loading>Saving...</Button>
 * 
 * // Full width
 * <Button variant="primary" fullWidth>Continue</Button>
 */
