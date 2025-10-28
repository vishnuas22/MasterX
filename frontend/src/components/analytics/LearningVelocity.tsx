/**
 * LearningVelocity Component - Learning Pace Visualization
 * 
 * Displays learning velocity (questions/hour) with trend indicators
 * and personalized recommendations
 * 
 * WCAG 2.1 AA Compliant:
 * - Semantic HTML structure
 * - ARIA labels for visual elements
 * - Color + text indicators (not color alone)
 * - Keyboard accessible
 * 
 * Performance:
 * - Pure component (memoized)
 * - Smooth SVG animations (CSS-based)
 * - No heavy computations
 * 
 * Backend Integration:
 * - Velocity Tracker: From adaptive learning engine
 * - Trend Analysis: Time series velocity data
 * - GET /api/v1/analytics/velocity/:userId
 */

import React from 'react';
import { Card } from '@/components/ui/Card';
import { cn } from '@/utils/cn';

export interface LearningVelocityProps {
  /**
   * Current velocity (questions per hour)
   */
  currentVelocity: number;
  
  /**
   * Average velocity
   */
  averageVelocity: number;
  
  /**
   * Velocity trend
   */
  trend: 'accelerating' | 'steady' | 'slowing';
  
  /**
   * Loading state
   */
  isLoading?: boolean;
  
  className?: string;
}

const TREND_CONFIG = {
  accelerating: { 
    color: 'text-green-500', 
    emoji: '📈', 
    label: 'Accelerating',
    description: 'You\'re learning faster!',
    bgColor: 'bg-green-500/10'
  },
  steady: { 
    color: 'text-blue-500', 
    emoji: '➡️', 
    label: 'Steady Pace',
    description: 'Consistent learning pace',
    bgColor: 'bg-blue-500/10'
  },
  slowing: { 
    color: 'text-orange-500', 
    emoji: '📉', 
    label: 'Slowing Down',
    description: 'Consider taking a break',
    bgColor: 'bg-orange-500/10'
  }
};

export const LearningVelocity: React.FC<LearningVelocityProps> = ({
  currentVelocity,
  averageVelocity,
  trend,
  isLoading = false,
  className
}) => {
  // Calculate percentage for progress ring (cap at 100%)
  const percentage = Math.min((currentVelocity / (averageVelocity * 2)) * 100, 100);
  const circumference = 2 * Math.PI * 45; // radius = 45
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const config = TREND_CONFIG[trend];

  if (isLoading) {
    return (
      <Card className={cn('p-6 animate-pulse', className)}>
        <div className="space-y-4">
          <div className="h-6 bg-gray-700 rounded w-1/2"></div>
          <div className="h-32 bg-gray-800 rounded"></div>
          <div className="h-4 bg-gray-700 rounded w-1/3"></div>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      className={cn('p-6', className)}
      data-testid="learning-velocity"
      role="article"
      aria-label={`Learning velocity: ${currentVelocity.toFixed(1)} questions per hour, ${config.label}`}
    >
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white">Learning Velocity</h3>
            <p className="text-sm text-gray-400">Questions per hour</p>
          </div>
          <div 
            className={cn('text-4xl p-3 rounded-lg', config.bgColor)}
            role="img"
            aria-label={config.label}
          >
            {config.emoji}
          </div>
        </div>

        {/* Velocity Gauge */}
        <div className="relative flex items-center justify-center">
          {/* Center Value */}
          <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
            <div className="flex items-baseline">
              <span className="text-5xl font-bold text-white">
                {currentVelocity.toFixed(1)}
              </span>
              <span className="ml-2 text-lg text-gray-400">/hr</span>
            </div>
            <div className="text-sm text-gray-500 mt-1">
              current pace
            </div>
          </div>
          
          {/* SVG Progress Ring */}
          <svg 
            className="w-48 h-48 -rotate-90 transform"
            viewBox="0 0 100 100"
            aria-hidden="true"
          >
            {/* Background Circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="#374151"
              strokeWidth="8"
            />
            {/* Progress Circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="#3B82F6"
              strokeWidth="8"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
              style={{
                filter: 'drop-shadow(0 0 8px rgba(59, 130, 246, 0.5))'
              }}
            />
          </svg>
        </div>

        {/* Trend Indicator */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className={cn('text-sm font-medium', config.color)}>
                {config.label}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {config.description}
              </div>
            </div>
          </div>

          {/* Average Comparison */}
          <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
            <span className="text-sm text-gray-400">Your Average</span>
            <span className="text-sm font-semibold text-white">
              {averageVelocity.toFixed(1)}/hr
            </span>
          </div>

          {/* Performance Badge */}
          {currentVelocity > averageVelocity * 1.2 && (
            <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="text-lg">🏆</span>
                <div className="flex-1">
                  <div className="text-sm font-medium text-green-400">
                    Outstanding Performance!
                  </div>
                  <div className="text-xs text-gray-400">
                    {((currentVelocity / averageVelocity - 1) * 100).toFixed(0)}% above your average
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentVelocity < averageVelocity * 0.7 && (
            <div className="p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="text-lg">💡</span>
                <div className="flex-1">
                  <div className="text-sm font-medium text-orange-400">
                    Consider a Break
                  </div>
                  <div className="text-xs text-gray-400">
                    Rest helps consolidate learning
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default React.memo(LearningVelocity);

/**
 * Usage Example:
 * 
 * <LearningVelocity
 *   currentVelocity={12.5}
 *   averageVelocity={10.2}
 *   trend="accelerating"
 * />
 */
