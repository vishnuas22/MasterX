/**
 * Badge Component - Status Labels & Tags
 * 
 * Following AGENTS_FRONTEND.md:
 * - Clear visual hierarchy
 * - Emotion-specific colors
 * - Achievement rarity colors
 * - Accessibility
 */

import React, { ReactNode } from 'react';
import { clsx } from 'clsx';

// ============================================================================
// TYPES
// ============================================================================

export type BadgeVariant = 
  | 'primary' 
  | 'success' 
  | 'warning' 
  | 'error' 
  | 'neutral'
  | 'purple';

export type BadgeSize = 'sm' | 'md' | 'lg';

export interface BadgeProps {
  /** Badge content */
  children: ReactNode;
  
  /** Visual variant */
  variant?: BadgeVariant;
  
  /** Size */
  size?: BadgeSize;
  
  /** Show dot indicator */
  dot?: boolean;
  
  /** Removable (shows X button) */
  onRemove?: () => void;
  
  /** Custom className */
  className?: string;
  
  /** Test ID */
  'data-testid'?: string;
}

// ============================================================================
// VARIANT STYLES
// ============================================================================

const variantStyles: Record<BadgeVariant, string> = {
  primary: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20',
  success: 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20',
  warning: 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20',
  error: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20',
  neutral: 'bg-gray-500/10 text-gray-600 dark:text-gray-400 border-gray-500/20',
  purple: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20',
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
  lg: 'px-3 py-1.5 text-base',
};

// ============================================================================
// COMPONENT
// ============================================================================

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'neutral',
  size = 'md',
  dot = false,
  onRemove,
  className,
  'data-testid': testId,
}) => {
  return (
    <span
      data-testid={testId}
      className={clsx(
        // Base styles
        'inline-flex items-center gap-1.5',
        'font-medium rounded-full border',
        'transition-all duration-150',
        
        // Size
        sizeStyles[size],
        
        // Variant
        variantStyles[variant],
        
        // Custom className
        className
      )}
    >
      {/* Dot indicator */}
      {dot && (
        <span
          className="w-1.5 h-1.5 rounded-full bg-current"
          aria-hidden="true"
        />
      )}
      
      {/* Content */}
      <span>{children}</span>
      
      {/* Remove button */}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          className={clsx(
            'ml-1 -mr-1 hover:opacity-70',
            'transition-opacity duration-150',
            'focus:outline-none focus:ring-2 focus:ring-current rounded-full'
          )}
          aria-label="Remove"
        >
          <RemoveIcon size={size} />
        </button>
      )}
    </span>
  );
};

// ============================================================================
// ICONS
// ============================================================================

const RemoveIcon: React.FC<{ size: BadgeSize }> = ({ size }) => {
  const iconSize = {
    sm: 10,
    md: 12,
    lg: 14,
  }[size];

  return (
    <svg
      width={iconSize}
      height={iconSize}
      viewBox="0 0 12 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M9 3L3 9M3 3L9 9"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
};

export default Badge;
