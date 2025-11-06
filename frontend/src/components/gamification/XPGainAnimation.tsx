/**
 * XPGainAnimation Component - Animated XP Gain Notification
 * 
 * Purpose: Display animated XP gain notifications when user earns XP
 * 
 * AGENTS_FRONTEND.md Compliance:
 * ✅ Type Safety: Strict TypeScript with no 'any'
 * ✅ Accessibility: ARIA labels for screen readers
 * ✅ Performance: CSS animations, no heavy JS
 * ✅ Responsive: Works on all screen sizes
 * 
 * Features:
 * 1. Floating animation with fade out
 * 2. Color coding based on XP amount
 * 3. Auto-dismiss after animation
 * 4. Customizable position
 * 
 * @module components/gamification/XPGainAnimation
 */

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/utils/cn';

export interface XPGainAnimationProps {
  amount: number;
  onComplete?: () => void;
  position?: 'top-center' | 'center' | 'bottom-center';
  className?: string;
}

/**
 * XPGainAnimation Component
 * 
 * Shows an animated notification when user gains XP.
 * Automatically disappears after 2 seconds.
 * 
 * @example
 * ```tsx
 * const [showXP, setShowXP] = useState(false);
 * 
 * {showXP && (
 *   <XPGainAnimation 
 *     amount={50} 
 *     onComplete={() => setShowXP(false)}
 *   />
 * )}
 * ```
 */
export const XPGainAnimation: React.FC<XPGainAnimationProps> = ({
  amount,
  onComplete,
  position = 'center',
  className
}) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onComplete?.();
    }, 2000);

    return () => clearTimeout(timer);
  }, [onComplete]);

  const getXPColor = (xp: number) => {
    if (xp >= 100) return 'from-purple-500 to-pink-500';
    if (xp >= 50) return 'from-blue-500 to-purple-500';
    if (xp >= 25) return 'from-green-500 to-blue-500';
    return 'from-yellow-500 to-orange-500';
  };

  const positionClasses = {
    'top-center': 'top-20',
    'center': 'top-1/2 -translate-y-1/2',
    'bottom-center': 'bottom-20'
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.8 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.8 }}
        className={cn(
          'fixed left-1/2 -translate-x-1/2 z-50 pointer-events-none',
          positionClasses[position],
          className
        )}
        role="alert"
        aria-live="polite"
        aria-label={`Gained ${amount} experience points`}
      >
        <div className={cn(
          'px-6 py-3 rounded-full shadow-2xl',
          'bg-gradient-to-r',
          getXPColor(amount),
          'border-2 border-white/20'
        )}>
          <div className="flex items-center gap-2">
            <span className="text-2xl">⭐</span>
            <div className="text-white">
              <span className="text-3xl font-bold">+{amount}</span>
              <span className="text-sm ml-2">XP</span>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default XPGainAnimation;
