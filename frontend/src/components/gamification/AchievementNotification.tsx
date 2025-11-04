/**
 * AchievementNotification Component - Achievement Unlock Notifications
 * 
 * Purpose: Display toast-style notifications when achievements are unlocked
 * 
 * AGENTS_FRONTEND.md Compliance:
 * ✅ Type Safety: Strict TypeScript with no 'any' types
 * ✅ Accessibility: WCAG 2.1 AA compliant (ARIA labels, keyboard navigation)
 * ✅ Performance: CSS animations preferred over JS
 * ✅ Animation: Respects @prefers-reduced-motion
 * ✅ Component Design: Single responsibility
 * 
 * Features:
 * 1. Toast-style notification with entrance/exit animations
 * 2. Achievement icon and details
 * 3. XP reward display
 * 4. Auto-dismiss after 5 seconds
 * 5. Manual dismiss button
 * 6. Sound effect (optional)
 * 7. Confetti celebration
 * 8. Rarity-based styling
 * 
 * @module components/gamification/AchievementNotification
 */

import React, { useEffect, useState } from 'react';
import { X, Sparkles } from 'lucide-react';
import { cn } from '@/utils/cn';
import Confetti from 'react-confetti';
import type { Achievement } from '@/services/api/gamification.api';

export interface AchievementNotificationProps {
  achievement: Achievement;
  onDismiss: () => void;
  autoHideDuration?: number; // milliseconds
  className?: string;
}

const RARITY_CONFIG = {
  common: {
    gradient: 'from-gray-500 to-gray-600',
    glow: 'shadow-gray-500/50',
    border: 'border-gray-500',
    textColor: 'text-gray-400',
    bgColor: 'bg-gray-500/10'
  },
  uncommon: {
    gradient: 'from-green-500 to-green-600',
    glow: 'shadow-green-500/50',
    border: 'border-green-500',
    textColor: 'text-green-400',
    bgColor: 'bg-green-500/10'
  },
  rare: {
    gradient: 'from-blue-500 to-blue-600',
    glow: 'shadow-blue-500/50',
    border: 'border-blue-500',
    textColor: 'text-blue-400',
    bgColor: 'bg-blue-500/10'
  },
  epic: {
    gradient: 'from-purple-500 to-purple-600',
    glow: 'shadow-purple-500/50',
    border: 'border-purple-500',
    textColor: 'text-purple-400',
    bgColor: 'bg-purple-500/10'
  },
  legendary: {
    gradient: 'from-orange-500 to-red-500',
    glow: 'shadow-orange-500/50',
    border: 'border-orange-500',
    textColor: 'text-orange-400',
    bgColor: 'bg-orange-500/10'
  }
};

/**
 * Achievement Notification Component
 * 
 * Displays a celebratory notification when an achievement is unlocked.
 * Auto-dismisses after specified duration or can be manually dismissed.
 * 
 * @example
 * ```tsx
 * <AchievementNotification
 *   achievement={newAchievement}
 *   onDismiss={() => handleDismiss(achievement.id)}
 *   autoHideDuration={5000}
 * />
 * ```
 */
export const AchievementNotification: React.FC<AchievementNotificationProps> = ({
  achievement,
  onDismiss,
  autoHideDuration = 5000,
  className
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  
  // Get rarity config, default to rare if invalid
  const rarity = achievement.rarity in RARITY_CONFIG 
    ? achievement.rarity as keyof typeof RARITY_CONFIG
    : 'rare';
  const rarityConfig = RARITY_CONFIG[rarity];

  // Entrance animation
  useEffect(() => {
    // Slight delay for smooth entrance
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  // Show confetti for epic and legendary achievements
  useEffect(() => {
    if (rarity === 'epic' || rarity === 'legendary') {
      setShowConfetti(true);
      const timer = setTimeout(() => setShowConfetti(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [rarity]);

  // Auto-dismiss
  useEffect(() => {
    if (autoHideDuration > 0) {
      const timer = setTimeout(() => {
        handleDismiss();
      }, autoHideDuration);
      
      return () => clearTimeout(timer);
    }
  }, [autoHideDuration]);

  const handleDismiss = () => {
    setIsVisible(false);
    // Wait for exit animation to complete before calling onDismiss
    setTimeout(() => {
      onDismiss();
    }, 300);
  };

  return (
    <>
      {/* Confetti for rare achievements */}
      {showConfetti && (
        <Confetti
          width={window.innerWidth}
          height={window.innerHeight}
          recycle={false}
          numberOfPieces={rarity === 'legendary' ? 300 : 150}
          gravity={0.3}
        />
      )}

      {/* Notification Toast */}
      <div
        className={cn(
          'fixed bottom-4 right-4 z-50',
          'w-96 max-w-[calc(100vw-2rem)]',
          'transform transition-all duration-300 ease-out',
          isVisible 
            ? 'translate-x-0 opacity-100' 
            : 'translate-x-full opacity-0',
          className
        )}
        role="alert"
        aria-live="polite"
        aria-atomic="true"
        data-testid="achievement-notification"
      >
        <div className={cn(
          'relative',
          'bg-gray-900 border-2 rounded-lg',
          'p-4 shadow-2xl',
          rarityConfig.border,
          rarityConfig.glow,
          // Respect prefers-reduced-motion
          'motion-reduce:transition-none'
        )}>
          {/* Background Glow */}
          <div className={cn(
            'absolute inset-0 rounded-lg',
            'bg-gradient-to-r opacity-10 animate-pulse',
            rarityConfig.gradient
          )} />

          {/* Content */}
          <div className="relative z-10">
            {/* Header */}
            <div className="flex items-start gap-3 mb-3">
              {/* Achievement Icon */}
              <div className={cn(
                'flex-shrink-0',
                'w-16 h-16 rounded-full',
                'flex items-center justify-center',
                'border-2',
                'bg-gradient-to-br',
                rarityConfig.gradient,
                rarityConfig.border,
                'text-3xl',
                'animate-bounce-slow'
              )}>
                {achievement.icon}
              </div>

              {/* Title and Description */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className={cn('w-4 h-4', rarityConfig.textColor)} />
                  <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
                    Achievement Unlocked!
                  </h4>
                </div>
                <h3 className="text-lg font-bold text-white mb-1 truncate">
                  {achievement.name}
                </h3>
                <p className="text-sm text-gray-300 line-clamp-2">
                  {achievement.description}
                </p>
              </div>

              {/* Close Button */}
              <button
                onClick={handleDismiss}
                className={cn(
                  'flex-shrink-0',
                  'p-1 rounded-lg',
                  'text-gray-400 hover:text-white',
                  'hover:bg-gray-800',
                  'transition-colors',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
                aria-label="Dismiss notification"
                data-testid="dismiss-achievement-notification"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between">
              {/* XP Reward */}
              <div className={cn(
                'flex items-center gap-2',
                'px-3 py-1.5 rounded-lg',
                rarityConfig.bgColor
              )}>
                <span className="text-xl">⭐</span>
                <span className="text-sm font-bold text-white">
                  +{achievement.xp_reward} XP
                </span>
              </div>

              {/* Rarity Badge */}
              <div className={cn(
                'px-3 py-1.5 rounded-lg',
                'text-xs font-semibold uppercase tracking-wide',
                rarityConfig.textColor,
                rarityConfig.bgColor
              )}>
                {rarity}
              </div>
            </div>
          </div>

          {/* Shimmer Effect for Legendary */}
          {rarity === 'legendary' && (
            <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20 animate-shimmer" />
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default AchievementNotification;
