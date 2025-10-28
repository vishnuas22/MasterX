/**
 * StatsCard Component - Metric Display Card
 * 
 * Reusable component for displaying key statistics
 * with trend indicators and comparisons
 * 
 * WCAG 2.1 AA Compliant:
 * - Semantic HTML with proper ARIA labels
 * - Color contrast ratio ≥ 4.5:1
 * - Keyboard accessible
 * - Screen reader friendly
 * 
 * Performance:
 * - Pure component (React.memo)
 * - No unnecessary re-renders
 * - Lightweight DOM structure
 */

import React from 'react';
import { Card } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';
import { cn } from '@/utils/cn';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus,
  type LucideIcon 
} from 'lucide-react';

export interface StatsCardProps {
  /**
   * Card title
   */
  title: string;
  
  /**
   * Main statistic value
   */
  value: string | number;
  
  /**
   * Optional subtitle or description
   */
  subtitle?: string;
  
  /**
   * Icon component
   */
  icon?: LucideIcon;
  
  /**
   * Trend direction
   */
  trend?: 'up' | 'down' | 'neutral';
  
  /**
   * Trend value (e.g., "+12%")
   */
  trendValue?: string;
  
  /**
   * Comparison period (e.g., "vs last week")
   */
  trendLabel?: string;
  
  /**
   * Loading state
   */
  isLoading?: boolean;
  
  /**
   * Color theme
   */
  color?: 'blue' | 'green' | 'orange' | 'purple' | 'red';
  
  className?: string;
}

const COLOR_VARIANTS = {
  blue: {
    bg: 'bg-blue-500/10',
    text: 'text-blue-500',
    icon: 'text-blue-500'
  },
  green: {
    bg: 'bg-green-500/10',
    text: 'text-green-500',
    icon: 'text-green-500'
  },
  orange: {
    bg: 'bg-orange-500/10',
    text: 'text-orange-500',
    icon: 'text-orange-500'
  },
  purple: {
    bg: 'bg-purple-500/10',
    text: 'text-purple-500',
    icon: 'text-purple-500'
  },
  red: {
    bg: 'bg-red-500/10',
    text: 'text-red-500',
    icon: 'text-red-500'
  }
};

const TREND_CONFIG = {
  up: {
    icon: TrendingUp,
    color: 'text-green-500',
    ariaLabel: 'increasing'
  },
  down: {
    icon: TrendingDown,
    color: 'text-red-500',
    ariaLabel: 'decreasing'
  },
  neutral: {
    icon: Minus,
    color: 'text-gray-500',
    ariaLabel: 'stable'
  }
};

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  trendLabel,
  isLoading = false,
  color = 'blue',
  className
}) => {
  const colorVariant = COLOR_VARIANTS[color];
  const trendConfig = trend ? TREND_CONFIG[trend] : null;
  const TrendIcon = trendConfig?.icon;

  if (isLoading) {
    return (
      <Card className={cn('p-6', className)}>
        <div className="space-y-3">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-3 w-20" />
        </div>
      </Card>
    );
  }

  return (
    <Card 
      className={cn('p-6 space-y-3', className)}
      role="article"
      aria-label={`${title}: ${value}`}
      data-testid="stats-card"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-400">
          {title}
        </h3>
        {Icon && (
          <div className={cn('p-2 rounded-lg', colorVariant.bg)}>
            <Icon className={cn('w-5 h-5', colorVariant.icon)} />
          </div>
        )}
      </div>

      {/* Value */}
      <div>
        <div className={cn('text-3xl font-bold', colorVariant.text)}>
          {value}
        </div>
        {subtitle && (
          <div className="text-sm text-gray-500 mt-1">
            {subtitle}
          </div>
        )}
      </div>

      {/* Trend */}
      {trend && trendConfig && (
        <div className="flex items-center gap-2">
          {TrendIcon && (
            <TrendIcon 
              className={cn('w-4 h-4', trendConfig.color)}
              aria-label={trendConfig.ariaLabel}
            />
          )}
          {trendValue && (
            <span className={cn('text-sm font-medium', trendConfig.color)}>
              {trendValue}
            </span>
          )}
          {trendLabel && (
            <span className="text-sm text-gray-500">
              {trendLabel}
            </span>
          )}
        </div>
      )}
    </Card>
  );
};

// Memoize component to prevent unnecessary re-renders
export default React.memo(StatsCard);

/**
 * Usage Example:
 * 
 * <StatsCard
 *   title="Total Sessions"
 *   value="142"
 *   subtitle="This month"
 *   icon={BookOpen}
 *   trend="up"
 *   trendValue="+12%"
 *   trendLabel="vs last month"
 *   color="blue"
 * />
 */
