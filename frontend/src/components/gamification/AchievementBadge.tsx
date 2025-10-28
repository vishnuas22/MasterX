/**
 * AchievementBadge Component - Achievement Display
 * 
 * Purpose: Display unlocked achievements with animations
 * 
 * Features:
 * 1. Achievement badge with icon and metadata
 * 2. Unlock animations
 * 3. Rarity indicators (common, rare, epic, legendary)
 * 4. Progress towards locked achievements
 * 5. Share functionality
 * 6. Achievement details modal
 * 
 * Backend Integration:
 * - Achievement Engine: 17 achievements from backend
 * - Unlock System: Real-time achievement detection
 * - XP Rewards: Experience points awarded
 */

import React, { useState } from 'react';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/utils/cn';

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  xpReward: number;
  unlockedAt?: Date;
  progress?: number; // 0-1 for locked achievements
}

export interface AchievementBadgeProps {
  achievement: Achievement;
  size?: 'sm' | 'md' | 'lg';
  showDetails?: boolean;
  className?: string;
}

const RARITY_CONFIG = {
  common: {
    gradient: 'from-gray-500 to-gray-600',
    glow: 'shadow-gray-500/50',
    border: 'border-gray-500'
  },
  rare: {
    gradient: 'from-blue-500 to-blue-600',
    glow: 'shadow-blue-500/50',
    border: 'border-blue-500'
  },
  epic: {
    gradient: 'from-purple-500 to-purple-600',
    glow: 'shadow-purple-500/50',
    border: 'border-purple-500'
  },
  legendary: {
    gradient: 'from-orange-500 to-red-500',
    glow: 'shadow-orange-500/50',
    border: 'border-orange-500'
  }
};

export const AchievementBadge: React.FC<AchievementBadgeProps> = ({
  achievement,
  size = 'md',
  showDetails = false,
  className
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const rarityConfig = RARITY_CONFIG[achievement.rarity];
  const isUnlocked = !!achievement.unlockedAt;

  const sizeClasses = {
    sm: 'w-16 h-16 text-2xl',
    md: 'w-24 h-24 text-4xl',
    lg: 'w-32 h-32 text-5xl'
  };

  return (
    <>
      <button
        onClick={() => setIsModalOpen(true)}
        className={cn(
          'relative group',
          'transition-transform duration-200',
          'hover:scale-110 focus:scale-110',
          'focus:outline-none',
          className
        )}
        aria-label={`${achievement.name}: ${achievement.description}`}
      >
        {/* Badge Container */}
        <div className={cn(
          'relative rounded-full',
          'border-4',
          sizeClasses[size],
          'flex items-center justify-center',
          isUnlocked 
            ? cn('bg-gradient-to-br', rarityConfig.gradient, rarityConfig.border)
            : 'bg-gray-800 border-gray-700',
          isUnlocked && rarityConfig.glow
        )}>
          {/* Icon */}
          <span className={cn(
            'relative z-10',
            !isUnlocked && 'grayscale opacity-30'
          )}>
            {achievement.icon}
          </span>

          {/* Lock Overlay */}
          {!isUnlocked && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full">
              <svg className="w-1/2 h-1/2 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
            </div>
          )}

          {/* Progress Ring (for locked) */}
          {!isUnlocked && achievement.progress !== undefined && (
            <svg className="absolute inset-0 w-full h-full -rotate-90">
              <circle
                cx="50%"
                cy="50%"
                r="48%"
                fill="none"
                stroke="#374151"
                strokeWidth="4"
              />
              <circle
                cx="50%"
                cy="50%"
                r="48%"
                fill="none"
                stroke="#3B82F6"
                strokeWidth="4"
                strokeDasharray={`${achievement.progress * 314} 314`}
                strokeLinecap="round"
              />
            </svg>
          )}

          {/* Glow Effect (for unlocked) */}
          {isUnlocked && (
            <div className={cn(
              'absolute inset-0 rounded-full animate-pulse',
              'bg-gradient-to-br',
              rarityConfig.gradient,
              'opacity-20 blur-xl'
            )} />
          )}
        </div>

        {/* XP Badge */}
        {showDetails && isUnlocked && (
          <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
            <Badge variant="neutral" className="text-xs font-bold">
              +{achievement.xpReward} XP
            </Badge>
          </div>
        )}
      </button>

      {/* Details Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={achievement.name}
      >
        <div className="space-y-4">
          {/* Large Icon */}
          <div className="flex justify-center">
            <div className={cn(
              'w-32 h-32 rounded-full flex items-center justify-center',
              'border-4',
              isUnlocked 
                ? cn('bg-gradient-to-br', rarityConfig.gradient, rarityConfig.border)
                : 'bg-gray-800 border-gray-700'
            )}>
              <span className="text-6xl">{achievement.icon}</span>
            </div>
          </div>

          {/* Description */}
          <div className="text-center">
            <p className="text-gray-300">{achievement.description}</p>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 p-4 bg-gray-800 rounded-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {achievement.xpReward}
              </div>
              <div className="text-xs text-gray-400">XP Reward</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold capitalize">
                <span className={cn(
                  achievement.rarity === 'legendary' && 'text-orange-500',
                  achievement.rarity === 'epic' && 'text-purple-500',
                  achievement.rarity === 'rare' && 'text-blue-500',
                  achievement.rarity === 'common' && 'text-gray-500'
                )}>
                  {achievement.rarity}
                </span>
              </div>
              <div className="text-xs text-gray-400">Rarity</div>
            </div>
          </div>

          {/* Unlock Date */}
          {isUnlocked && achievement.unlockedAt && (
            <div className="text-center text-sm text-gray-400">
              Unlocked on {new Date(achievement.unlockedAt).toLocaleDateString()}
            </div>
          )}

          {/* Progress */}
          {!isUnlocked && achievement.progress !== undefined && (
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Progress</span>
                <span className="text-white">{(achievement.progress * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${achievement.progress * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </Modal>
    </>
  );
};

export default AchievementBadge;
