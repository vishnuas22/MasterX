// **Purpose:** Versatile card container with glass morphism and variants

// **What This File Contributes:**
// 1. Multiple visual variants (glass, solid, bordered)
// 2. Hover effects
// 3. Padding variations
// 4. Header and footer sections
// 5. Click handler support
// 6. Apple-style glass morphism

// **Implementation:**
// ```typescript
// /**
//  * Card Component - Versatile Container
//  * 
//  * Following AGENTS_FRONTEND.md:
//  * - Glass morphism (Apple Liquid Glass)
//  * - Smooth transitions
//  * - Hover states
//  * - Semantic HTML
//  */

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
            'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary'
          ),
          
          // Custom className
          className
        )}
      >
        {/* Header */}
        {header && (
          <div className={clsx(
            'border-b border-bg-tertiary',
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
            'border-t border-bg-tertiary',
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

// ============================================================================
// USAGE EXAMPLES
// ============================================================================

/*
// Basic solid card
<Card>
  <h3>Card Title</h3>
  <p>Card content goes here.</p>
</Card>

// Glass morphism card
<Card variant="glass">
  <div className="flex items-center gap-4">
    <EmotionIcon emotion="joy" />
    <div>
      <h4>Joyful State</h4>
      <p>You're in an optimal learning zone!</p>
    </div>
  </div>
</Card>

// Clickable card with hover
<Card
  variant="solid"
  hoverable
  onClick={() => navigate('/session/123')}
>
  <h3>Session #123</h3>
  <p>Math: Calculus</p>
</Card>

// Card with header and footer
<Card
  header={<h3 className="text-lg font-semibold">Achievement Unlocked!</h3>}
  footer={<Button fullWidth>Claim Reward</Button>}
>
  <div className="text-center py-4">
    <span className="text-4xl">🏆</span>
    <p>Week Warrior - 7 day streak!</p>
  </div>
</Card>

// Analytics card (elevated)
<Card variant="elevated" padding="lg">
  <div className="space-y-4">
    <h3 className="text-xl font-semibold">Learning Stats</h3>
    <div className="grid grid-cols-2 gap-4">
      <StatItem label="Sessions" value="42" />
      <StatItem label="Hours" value="12.5" />
    </div>
  </div>
</Card>
*/
// ```

// **Key Features:**
// 1. **4 variants:** Glass, solid, bordered, elevated
// 2. **Flexible padding:** None to large (4 levels)
// 3. **Header/footer:** Optional sections with dividers
// 4. **Interactive:** Click handlers with hover effects
// 5. **Glass morphism:** Apple-style translucent effect
// 6. **Semantic HTML:** Button when clickable, div otherwise

// **Performance:**
// - CSS-only animations
// - GPU-accelerated transforms
// - No JavaScript for hover
// - ~2KB gzipped

// **Connected Files:**
// - → Session cards, achievement cards, analytics widgets
// - → Dashboard, profile pages
// - ← `theme.config.ts` (glass effect)
// - → Emotion detail views
