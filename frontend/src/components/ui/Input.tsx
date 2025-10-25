/**
 * Input Component - Accessible Form Input
 * 
 * Features:
 * - Text, email, password types
 * - Error, success, default states
 * - Left/right icons
 * - Character count display
 * - Helper text and labels
 * - Full accessibility (WCAG 2.1 AA)
 */

import React, { forwardRef, InputHTMLAttributes, ReactNode, useState } from 'react';
import { clsx } from 'clsx';

// ============================================================================
// TYPES
// ============================================================================

export type InputSize = 'sm' | 'md' | 'lg';
export type InputState = 'default' | 'error' | 'success';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Input label */
  label?: string;
  
  /** Helper text below input */
  helperText?: string;
  
  /** Error message (also sets error state) */
  error?: string;
  
  /** Success state */
  success?: boolean;
  
  /** Input size */
  size?: InputSize;
  
  /** Left icon */
  leftIcon?: ReactNode;
  
  /** Right icon */
  rightIcon?: ReactNode;
  
  /** Show character count */
  showCount?: boolean;
  
  /** Max length for count */
  maxLength?: number;
  
  /** Full width input */
  fullWidth?: boolean;
  
  /** Test ID */
  'data-testid'?: string;
}

// ============================================================================
// STYLES
// ============================================================================

const sizeStyles: Record<InputSize, string> = {
  sm: 'px-3 py-1.5 text-sm min-h-[32px]',
  md: 'px-4 py-2 text-base min-h-[44px]', // WCAG touch target
  lg: 'px-4 py-3 text-lg min-h-[52px]',
};

const stateStyles: Record<InputState, string> = {
  default: clsx(
    'border-bg-tertiary',
    'focus:border-accent-primary focus:ring-2 focus:ring-accent-primary/20'
  ),
  error: clsx(
    'border-accent-error',
    'focus:border-accent-error focus:ring-2 focus:ring-accent-error/20'
  ),
  success: clsx(
    'border-accent-success',
    'focus:border-accent-success focus:ring-2 focus:ring-accent-success/20'
  ),
};

// ============================================================================
// COMPONENT
// ============================================================================

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      helperText,
      error,
      success,
      size = 'md',
      leftIcon,
      rightIcon,
      showCount,
      maxLength,
      fullWidth = false,
      className,
      id,
      'data-testid': testId,
      value,
      onChange,
      ...props
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = useState('');
    const inputValue = value !== undefined ? value : internalValue;
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    
    // Determine state
    const state: InputState = error ? 'error' : success ? 'success' : 'default';
    
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (value === undefined) {
        setInternalValue(e.target.value);
      }
      onChange?.(e);
    };
    
    const charCount = typeof inputValue === 'string' ? inputValue.length : 0;

    return (
      <div className={clsx('flex flex-col gap-1.5', fullWidth && 'w-full')}>
        {/* Label */}
        {label && (
          <label
            htmlFor={inputId}
            className="text-sm font-medium text-text-primary"
          >
            {label}
            {props.required && <span className="text-accent-error ml-1">*</span>}
          </label>
        )}
        
        {/* Input wrapper */}
        <div className="relative">
          {/* Left icon */}
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none">
              {leftIcon}
            </div>
          )}
          
          {/* Input field */}
          <input
            ref={ref}
            id={inputId}
            data-testid={testId}
            value={inputValue}
            onChange={handleChange}
            maxLength={maxLength}
            className={clsx(
              // Base styles
              'w-full rounded-md border bg-bg-secondary text-text-primary',
              'placeholder:text-text-tertiary',
              'transition-all duration-150',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              
              // Size
              sizeStyles[size],
              
              // State
              stateStyles[state],
              
              // Icon padding
              leftIcon && 'pl-10',
              rightIcon && 'pr-10',
              
              // Custom className
              className
            )}
            {...props}
          />
          
          {/* Right icon */}
          {rightIcon && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none">
              {rightIcon}
            </div>
          )}
        </div>
        
        {/* Helper text / Error / Character count */}
        <div className="flex items-center justify-between gap-2 text-xs">
          {/* Left side: helper text or error */}
          <div className={clsx(
            error && 'text-accent-error',
            success && 'text-accent-success',
            !error && !success && 'text-text-tertiary'
          )}>
            {error || helperText}
          </div>
          
          {/* Right side: character count */}
          {showCount && maxLength && (
            <div className={clsx(
              'text-text-tertiary',
              charCount >= maxLength && 'text-accent-warning'
            )}>
              {charCount}/{maxLength}
            </div>
          )}
        </div>
      </div>
    );
  }
);

Input.displayName = 'Input';

/*
 * Usage Examples:
 * 
 * // Basic input
 * <Input label="Email" placeholder="Enter your email" type="email" />
 * 
 * // With error
 * <Input label="Password" type="password" error="Password must be at least 8 characters" />
 * 
 * // With icon and character count
 * <Input label="Username" leftIcon={<UserIcon />} showCount maxLength={20} />
 */
