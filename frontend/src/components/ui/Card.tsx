/**
 * Card Component - Versatile Container
 * 
 * Features:
 * - 4 variants: glass, solid, bordered, elevated
 * - Flexible padding options
 * - Header and footer sections
 * - Click handler support
 * - Apple-style glass morphism
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
    'bg-glass-bg backdrop-blur-[40px]',
    'border border-glass-border',
    'shadow-glass'
  ),
  solid: clsx(
    'bg-bg-secondary',
    'border border-bg-tertiary'
  ),
  bordered: clsx(
    'bg-transparent',
    'border border-bg-tertiary'
  ),
  elevated: clsx(
    'bg-bg-secondary',
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

    // Shared className logic
    const cardClassName = clsx(
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
        'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary'
      ),
      
      // Custom className
      className
    );

    // Shared content
    const cardContent = (
      <>
        {/* Header */}
        {header && (
          <div className={clsx(
            'border-b border-bg-tertiary',
            padding !== 'none' && paddingStyles[padding]
          )}>
            {header}
          </div>
        )}
        
        {/* Body */}
        <div className={padding !== 'none' ? paddingStyles[padding] : undefined}>
          {children}
        </div>
        
        {/* Footer */}
        {footer && (
          <div className={clsx(
            'border-t border-bg-tertiary',
            padding !== 'none' && paddingStyles[padding]
          )}>
            {footer}
          </div>
        )}
      </>
    );

    // Render button if clickable
    if (isClickable) {
      return (
        <button
          ref={ref as React.Ref<HTMLButtonElement>}
          onClick={onClick}
          data-testid={testId}
          className={cardClassName}
        >
          {cardContent}
        </button>
      );
    }

    // Render div if not clickable
    return (
      <div
        ref={ref}
        data-testid={testId}
        className={cardClassName}
      >
        {cardContent}
      </div>
    );
  }
);

Card.displayName = 'Card';

/*
 * Usage Examples:
 * 
 * // Basic card
 * <Card><h3>Title</h3><p>Content</p></Card>
 * 
 * // Glass morphism
 * <Card variant="glass"><EmotionIndicator /></Card>
 * 
 * // Clickable card
 * <Card hoverable onClick={() => navigate('/session')}><h3>Session</h3></Card>
 * 
 * // With header and footer
 * <Card header={<h3>Achievement</h3>} footer={<Button>Claim</Button>}>Content</Card>
 */
