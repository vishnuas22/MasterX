/**
 * Badge Component - Status Labels & Tags
 * 
 * Features:
 * - 7 variants including emotion and rarity
 * - 3 sizes
 * - Dot indicator option
 * - Removable badges
 * - Emotion and gamification color support
 */

import React, { ReactNode } from 'react';
import { clsx } from 'clsx';
import { emotionColorMap, achievementRarityColors } from '@/config/theme.config';

// ============================================================================
// TYPES
// ============================================================================

export type BadgeVariant = 
  | 'primary' 
  | 'success' 
  | 'warning' 
  | 'error' 
  | 'neutral'
  | 'purple'
  | 'emotion'
  | 'rarity';

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
  
  /** Emotion name (for emotion variant) */
  emotion?: string;
  
  /** Achievement rarity (for rarity variant) */
  rarity?: 'common' | 'rare' | 'epic' | 'legendary';
  
  /** Custom className */
  className?: string;
  
  /** Test ID */
  'data-testid'?: string;
}

// ============================================================================
// VARIANT STYLES
// ============================================================================

const variantStyles: Record<BadgeVariant, string> = {
  primary: 'bg-accent-primary/10 text-accent-primary border-accent-primary/20',
  success: 'bg-accent-success/10 text-accent-success border-accent-success/20',
  warning: 'bg-accent-warning/10 text-accent-warning border-accent-warning/20',
  error: 'bg-accent-error/10 text-accent-error border-accent-error/20',
  neutral: 'bg-bg-tertiary text-text-secondary border-bg-tertiary',
  purple: 'bg-accent-purple/10 text-accent-purple border-accent-purple/20',
  emotion: '', // Dynamic based on emotion prop
  rarity: '', // Dynamic based on rarity prop
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
  emotion,
  rarity,
  className,
  'data-testid': testId,
}) => {
  // Determine dynamic styles for emotion/rarity variants
  let dynamicStyle = '';
  
  if (variant === 'emotion' && emotion) {
    const emotionColor = emotionColorMap[emotion.toLowerCase()] || emotionColorMap.neutral;
    dynamicStyle = `border-[${emotionColor}]/20`;
    // Note: For production, use CSS variables or predefined classes
  }
  
  if (variant === 'rarity' && rarity) {
    const rarityColor = achievementRarityColors[rarity];
    dynamicStyle = `border-[${rarityColor}]/20`;
  }

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
        variant !== 'emotion' && variant !== 'rarity' 
          ? variantStyles[variant]
          : dynamicStyle,
        
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

// ============================================================================
// HELPER COMPONENTS
// ============================================================================

/**
 * Emotion Badge - Pre-configured for emotion display
 */
export const EmotionBadge: React.FC<{
  emotion: string;
  confidence?: number;
  size?: BadgeSize;
}> = ({ emotion, confidence, size = 'md' }) => {
  const displayText = confidence
    ? `${emotion} (${(confidence * 100).toFixed(0)}%)`
    : emotion;

  return (
    <Badge
      variant="emotion"
      emotion={emotion}
      size={size}
      dot
    >
      {displayText}
    </Badge>
  );
};

/**
 * Achievement Badge - Pre-configured for achievements
 */
export const AchievementBadge: React.FC<{
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  name: string;
  size?: BadgeSize;
}> = ({ rarity, name, size = 'md' }) => {
  return (
    <Badge
      variant="rarity"
      rarity={rarity}
      size={size}
    >
      {name}
    </Badge>
  );
};

/*
 * Usage Examples:
 * 
 * // Basic badges
 * <Badge variant="success">Active</Badge>
 * <Badge variant="error">Failed</Badge>
 * 
 * // With dot
 * <Badge variant="primary" dot>Online</Badge>
 * 
 * // Removable
 * <Badge onRemove={() => removeTag('math')}>Math</Badge>
 * 
 * // Emotion badge
 * <EmotionBadge emotion="joy" confidence={0.87} />
 * 
 * // Achievement badge
 * <AchievementBadge rarity="legendary" name="Century Club" />
 */
