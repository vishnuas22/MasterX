/**
 * Card Component - Versatile Container
 * 
 * Following AGENTS_FRONTEND.md:
 * - Glass morphism (Apple Liquid Glass)
 * - Smooth transitions
 * - Hover states
 * - Semantic HTML
 */

import React, { ReactNode } from 'react';
import { clsx } from 'clsx';

// ============================================================================
// TYPES
// ============================================================================

export type CardVariant = 'glass' | 'solid' | 'bordered' | 'elevated';
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

export interface CardProps {
  /** Card content */
  children: ReactNode;
  
  /** Visual variant */
  variant?: CardVariant;
  
  /** Padding size */
  padding?: CardPadding;
  
  /** Header content */
  header?: ReactNode;
  
  /** Footer content */
  footer?: ReactNode;
  
  /** Click handler (makes card interactive) */
  onClick?: () => void;
  
  /** Hover effect */
  hoverable?: boolean;
  
  /** Custom className */
  className?: string;
  
  /** Test ID */
  'data-testid'?: string;
}

// ============================================================================
// VARIANT STYLES
// ============================================================================

const variantStyles: Record<CardVariant, string> = {
  glass: clsx(
    'bg-white/10 dark:bg-gray-900/10 backdrop-blur-[40px]',
    'border border-white/20 dark:border-gray-700/20',
    'shadow-lg'
  ),
  solid: clsx(
    'bg-white dark:bg-gray-800',
    'border border-gray-200 dark:border-gray-700'
  ),
  bordered: clsx(
    'bg-transparent',
    'border border-gray-200 dark:border-gray-700'
  ),
  elevated: clsx(
    'bg-white dark:bg-gray-800',
    'shadow-lg'
  ),
};

const paddingStyles: Record<CardPadding, string> = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
};

// ============================================================================
// COMPONENT
// ============================================================================

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      children,
      variant = 'solid',
      padding = 'md',
      header,
      footer,
      onClick,
      hoverable = false,
      className,
      'data-testid': testId,
    },
    ref
  ) => {
    const isClickable = !!onClick;
    const Component = isClickable ? 'button' : 'div';

    return (
      <Component
        ref={ref as any}
        onClick={onClick}
        data-testid={testId}
        className={clsx(
          // Base styles
          'rounded-xl overflow-hidden',
          'transition-all duration-250',
          
          // Variant
          variantStyles[variant],
          
          // Hover effect
          (hoverable || isClickable) && clsx(
            'hover:scale-[1.02] hover:shadow-xl',
            'active:scale-[0.98]'
          ),
          
          // Clickable styles
          isClickable && clsx(
            'cursor-pointer text-left w-full',
            'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
          ),
          
          // Custom className
          className
        )}
      >
        {/* Header */}
        {header && (
          <div className={clsx(
            'border-b border-gray-200 dark:border-gray-700',
            padding !== 'none' && paddingStyles[padding]
          )}>
            {header}
          </div>
        )}

        {/* Content */}
        <div className={padding !== 'none' ? paddingStyles[padding] : ''}>
          {children}
        </div>

        {/* Footer */}
        {footer && (
          <div className={clsx(
            'border-t border-gray-200 dark:border-gray-700',
            padding !== 'none' && paddingStyles[padding]
          )}>
            {footer}
          </div>
        )}
      </Component>
    );
  }
);

Card.displayName = 'Card';

export default Card;
