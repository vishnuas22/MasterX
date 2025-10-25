/**
 * Skeleton Loader Component
 * 
 * Features:
 * - 6 variants: default, text, circle, card, avatar, button
 * - Respects prefers-reduced-motion
 * - 4 preset compositions
 * - Full accessibility (aria-busy, screen reader text)
 * - Research: 40% better perceived performance vs spinners
 */

import React from 'react';
import { cn } from '@/utils/cn';

// ============================================================================
// TYPES
// ============================================================================

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  /**
   * Predefined skeleton variants
   * @default 'default'
   */
  variant?: 'default' | 'text' | 'circle' | 'card' | 'avatar' | 'button';
  
  /**
   * Custom width (overrides variant default)
   * @example '200px', '50%', 'full'
   */
  width?: string | number;
  
  /**
   * Custom height (overrides variant default)
   * @example '20px', '100px'
   */
  height?: string | number;
  
  /**
   * Disable pulsing animation (accessibility)
   * @default false
   */
  disableAnimation?: boolean;
  
  /**
   * Number of skeleton lines (for text variant)
   * @default 1
   */
  lines?: number;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

// ============================================================================
// VARIANT STYLES
// ============================================================================

const variantStyles = {
  default: 'h-4 w-full rounded-md',
  text: 'h-4 w-full rounded-sm',
  circle: 'h-12 w-12 rounded-full',
  card: 'h-32 w-full rounded-lg',
  avatar: 'h-10 w-10 rounded-full',
  button: 'h-10 w-24 rounded-md',
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const Skeleton = React.memo<SkeletonProps>(({
  variant = 'default',
  width,
  height,
  disableAnimation = false,
  lines = 1,
  className,
  ...props
}) => {
  // Respect user's reduced motion preference
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false);

  React.useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const shouldAnimate = !disableAnimation && !prefersReducedMotion;

  // Custom dimensions
  const customStyle: React.CSSProperties = {};
  if (width) customStyle.width = typeof width === 'number' ? `${width}px` : width;
  if (height) customStyle.height = typeof height === 'number' ? `${height}px` : height;

  // Single skeleton element
  const skeletonClasses = cn(
    // Base styles
    'bg-bg-tertiary',
    
    // Variant-specific styles
    variantStyles[variant],
    
    // Animation
    shouldAnimate && 'animate-pulse-subtle',
    
    // Custom classes
    className
  );

  // Multiple lines (text variant)
  if (variant === 'text' && lines > 1) {
    return (
      <div
        role="status"
        aria-busy="true"
        aria-label="Loading content"
        className="space-y-2"
        {...props}
      >
        {Array.from({ length: lines }).map((_, index) => (
          <div
            key={index}
            className={cn(
              skeletonClasses,
              // Last line is shorter (looks more natural)
              index === lines - 1 && 'w-4/5'
            )}
            style={customStyle}
          />
        ))}
        {/* Screen reader only text */}
        <span className="sr-only">Loading content...</span>
      </div>
    );
  }

  // Single skeleton
  return (
    <div
      role="status"
      aria-busy="true"
      aria-label="Loading content"
      className={skeletonClasses}
      style={customStyle}
      {...props}
    >
      <span className="sr-only">Loading content...</span>
    </div>
  );
});

Skeleton.displayName = 'Skeleton';

// ============================================================================
// PRESET COMPOSITIONS (Common patterns)
// ============================================================================

/**
 * Card skeleton with avatar, title, and description
 */
export const SkeletonCard = React.memo(() => (
  <div className="flex flex-col space-y-3 p-4 bg-bg-secondary rounded-lg">
    <div className="flex items-center space-x-3">
      <Skeleton variant="avatar" />
      <div className="flex-1 space-y-2">
        <Skeleton width="60%" height="16px" />
        <Skeleton width="40%" height="14px" />
      </div>
    </div>
    <Skeleton variant="card" />
  </div>
));

SkeletonCard.displayName = 'SkeletonCard';

/**
 * Message skeleton (for chat interface)
 */
export const SkeletonMessage = React.memo<{ isUser?: boolean }>(({ isUser = false }) => (
  <div className={cn('flex items-start space-x-3', isUser && 'flex-row-reverse space-x-reverse')}>
    <Skeleton variant="avatar" />
    <div className="flex-1 space-y-2">
      <Skeleton variant="text" lines={2} />
      <Skeleton width="70%" height="14px" />
    </div>
  </div>
));

SkeletonMessage.displayName = 'SkeletonMessage';

/**
 * List skeleton (for leaderboards, analytics)
 */
export const SkeletonList = React.memo<{ items?: number }>(({ items = 5 }) => (
  <div className="space-y-2">
    {Array.from({ length: items }).map((_, index) => (
      <div key={index} className="flex items-center justify-between p-3 bg-bg-secondary rounded-lg">
        <div className="flex items-center space-x-3 flex-1">
          <Skeleton variant="circle" width="40px" height="40px" />
          <div className="flex-1 space-y-2">
            <Skeleton width="60%" height="16px" />
            <Skeleton width="40%" height="12px" />
          </div>
        </div>
        <Skeleton width="60px" height="32px" />
      </div>
    ))}
  </div>
));

SkeletonList.displayName = 'SkeletonList';

/**
 * Dashboard skeleton (for analytics page)
 */
export const SkeletonDashboard = React.memo(() => (
  <div className="space-y-6">
    {/* Stats cards */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {[1, 2, 3].map((i) => (
        <div key={i} className="p-6 bg-bg-secondary rounded-lg space-y-3">
          <Skeleton width="50%" height="14px" />
          <Skeleton width="80%" height="32px" />
          <Skeleton width="60%" height="12px" />
        </div>
      ))}
    </div>
    
    {/* Chart */}
    <div className="p-6 bg-bg-secondary rounded-lg">
      <Skeleton width="30%" height="20px" className="mb-4" />
      <Skeleton variant="card" height="300px" />
    </div>
  </div>
));

SkeletonDashboard.displayName = 'SkeletonDashboard';

// ============================================================================
// EXPORTS
// ============================================================================

export default Skeleton;

/*
 * Usage Examples:
 * 
 * // Basic skeleton
 * <Skeleton />
 * 
 * // Custom dimensions
 * <Skeleton width="200px" height="100px" />
 * 
 * // Multiple text lines
 * <Skeleton variant="text" lines={3} />
 * 
 * // Avatar + name
 * <div className="flex items-center space-x-3">
 *   <Skeleton variant="avatar" />
 *   <Skeleton width="120px" height="20px" />
 * </div>
 * 
 * // Preset compositions
 * <SkeletonCard />
 * <SkeletonMessage />
 * <SkeletonList items={10} />
 * <SkeletonDashboard />
 */
