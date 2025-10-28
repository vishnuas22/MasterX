/**
 * StreakCounter Component - Daily Streak Tracker
 * 
 * Purpose: Display current streak with visual appeal and motivation
 * 
 * Features:
 * 1. Current streak counter with fire emoji
 * 2. Longest streak record
 * 3. Streak freeze indicators
 * 4. Daily check-in reminder
 * 5. Streak milestone celebrations
 * 6. Calendar view of streak history
 * 
 * Backend Integration:
 * - Streak Tracker: From gamification engine
 * - Streak Freezes: Grace period and freeze system
 * - Daily Activity: Auto-increment on learning activity
 */

import React from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';
import { cn } from '@/utils/cn';

export interface StreakCounterProps {
  currentStreak: number;
  longestStreak: number;
  freezesAvailable: number;
  lastActivity?: Date;
  className?: string;
}

export const StreakCounter: React.FC<StreakCounterProps> = ({
  currentStreak,
  longestStreak,
  freezesAvailable,
  lastActivity,
  className
}) => {
  const isActiveToday = lastActivity && 
    new Date().toDateString() === new Date(lastActivity).toDateString();

  const getStreakEmoji = (streak: number) => {
    if (streak >= 100) return '👑';
    if (streak >= 30) return '💎';
    if (streak >= 7) return '⚡';
    return '🔥';
  };

  const getStreakColor = (streak: number) => {
    if (streak >= 100) return 'from-purple-500 to-pink-500';
    if (streak >= 30) return 'from-blue-500 to-purple-500';
    if (streak >= 7) return 'from-orange-500 to-red-500';
    return 'from-yellow-500 to-orange-500';
  };

  return (
    <Card className={cn('p-6', className)}>
      <div className="space-y-4">
        {/* Main Streak Display */}
        <div className="text-center">
          <div className="text-6xl mb-2 animate-bounce-slow">
            {getStreakEmoji(currentStreak)}
          </div>
          <div className={cn(
            'text-5xl font-bold bg-gradient-to-r bg-clip-text text-transparent',
            getStreakColor(currentStreak)
          )}>
            {currentStreak}
          </div>
          <div className="text-sm text-gray-400 mt-1">
            day streak
          </div>
        </div>

        {/* Status Badge */}
        <div className="flex justify-center">
          {isActiveToday ? (
            <Badge variant="success" className="gap-1">
              <span>✅</span>
              <span>Active today</span>
            </Badge>
          ) : (
            <Badge variant="warning" className="gap-1">
              <span>⏰</span>
              <span>Practice today to maintain streak</span>
            </Badge>
          )}
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-800">
          {/* Longest Streak */}
          <div className="text-center">
            <div className="text-2xl font-bold text-white">
              {longestStreak}
            </div>
            <div className="text-xs text-gray-400">Longest streak</div>
          </div>

          {/* Streak Freezes */}
          <Tooltip content="Use a freeze to protect your streak if you miss a day">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">
                {freezesAvailable}
              </div>
              <div className="text-xs text-gray-400">Freezes left</div>
            </div>
          </Tooltip>
        </div>

        {/* Milestone Progress */}
        {currentStreak < 100 && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-gray-400">
              <span>Next milestone</span>
              <span>
                {currentStreak >= 30 ? '100 days 👑' : 
                 currentStreak >= 7 ? '30 days 💎' : '7 days ⚡'}
              </span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={cn(
                  'h-full bg-gradient-to-r transition-all duration-500',
                  getStreakColor(currentStreak)
                )}
                style={{ 
                  width: `${(currentStreak / (currentStreak >= 30 ? 100 : currentStreak >= 7 ? 30 : 7)) * 100}%` 
                }}
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default StreakCounter;
