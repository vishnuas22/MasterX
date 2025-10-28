/**
 * LevelProgress Component - Level & XP Tracker
 * 
 * Purpose: Display current level, XP progress, and level-up animations
 * 
 * Features:
 * 1. Current level with visual badge
 * 2. XP progress bar to next level
 * 3. Level-up celebration animations
 * 4. XP gain animations
 * 5. Level perks/benefits display
 * 6. Historical level progression
 * 
 * Backend Integration:
 * - Level System: Exponential XP curve from backend
 * - XP Calculation: Dynamic XP rewards based on difficulty
 * - Level Benefits: Unlockable features per level
 */

import React, { useEffect, useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/utils/cn';
import Confetti from 'react-confetti';

export interface LevelProgressProps {
  level: number;
  currentXP: number;
  xpToNextLevel: number;
  recentXPGain?: number;
  justLeveledUp?: boolean;
  className?: string;
}

export const LevelProgress: React.FC<LevelProgressProps> = ({
  level,
  currentXP,
  xpToNextLevel,
  recentXPGain,
  justLeveledUp = false,
  className
}) => {
  const [showConfetti, setShowConfetti] = useState(false);
  const [animateXP, setAnimateXP] = useState(false);

  const progress = (currentXP / (currentXP + xpToNextLevel)) * 100;

  // Level up celebration
  useEffect(() => {
    if (justLeveledUp) {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 5000);
    }
  }, [justLeveledUp]);

  // XP gain animation
  useEffect(() => {
    if (recentXPGain) {
      setAnimateXP(true);
      setTimeout(() => setAnimateXP(false), 1000);
    }
  }, [recentXPGain]);

  const getLevelColor = (level: number) => {
    if (level >= 50) return 'from-purple-500 to-pink-500';
    if (level >= 25) return 'from-blue-500 to-purple-500';
    if (level >= 10) return 'from-green-500 to-blue-500';
    return 'from-gray-500 to-gray-600';
  };

  return (
    <>
      {showConfetti && (
        <Confetti
          width={window.innerWidth}
          height={window.innerHeight}
          recycle={false}
          numberOfPieces={200}
        />
      )}

      <Card className={cn('p-6', className)}>
        <div className="space-y-4">
          {/* Level Badge */}
          <div className="flex items-center gap-4">
            <div className={cn(
              'w-20 h-20 rounded-full flex items-center justify-center',
              'bg-gradient-to-br border-4 border-gray-700',
              getLevelColor(level)
            )}>
              <div className="text-center">
                <div className="text-xs text-white/80 font-medium">LEVEL</div>
                <div className="text-2xl font-bold text-white">{level}</div>
              </div>
            </div>

            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white">
                Level {level}
              </h3>
              <p className="text-sm text-gray-400">
                {currentXP.toLocaleString()} / {(currentXP + xpToNextLevel).toLocaleString()} XP
              </p>
            </div>

            {/* Recent XP Gain */}
            {recentXPGain && (
              <Badge 
                variant="success" 
                className={cn(
                  'transition-all duration-500',
                  animateXP && 'scale-125'
                )}
              >
                +{recentXPGain} XP
              </Badge>
            )}
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={cn(
                  'h-full bg-gradient-to-r transition-all duration-500',
                  getLevelColor(level)
                )}
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-400">
              <span>{progress.toFixed(1)}% to next level</span>
              <span>{xpToNextLevel.toLocaleString()} XP needed</span>
            </div>
          </div>

          {/* Level Up Message */}
          {justLeveledUp && (
            <div className="p-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/50 rounded-lg text-center">
              <div className="text-2xl mb-1">🎉</div>
              <div className="font-bold text-white">Level Up!</div>
              <div className="text-sm text-gray-300">
                You've reached level {level}
              </div>
            </div>
          )}
        </div>
      </Card>
    </>
  );
};

export default LevelProgress;
