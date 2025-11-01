/**
 * Achievements Page - Gamification & Progress Tracking
 * 
 * WCAG 2.1 AA Compliant:
 * - Alt text for badge images
 * - Keyboard navigation
 * - Progress bar labels
 * 
 * Backend Integration:
 * - GET /api/v1/gamification/achievements
 * - GET /api/v1/gamification/stats
 * - GET /api/v1/gamification/leaderboard
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Trophy, Star, Award, Target, Zap, Gift } from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

export interface AchievementsProps {
  onClose: () => void;
}

// ============================================================================
// COMPONENT
// ============================================================================

export const Achievements: React.FC<AchievementsProps> = ({ onClose }) => {
  // Mock achievements data
  const mockAchievements = [
    { id: 1, name: 'First Steps', description: 'Complete your first learning session', icon: '👋', locked: false },
    { id: 2, name: 'Quick Learner', description: 'Complete 5 sessions in one day', icon: '⚡', locked: true },
    { id: 3, name: 'Curious Mind', description: 'Ask 100 questions', icon: '🤔', locked: true },
    { id: 4, name: 'Emotion Master', description: 'Experience 10 different emotions', icon: '😊', locked: true },
    { id: 5, name: 'Week Warrior', description: 'Maintain a 7-day learning streak', icon: '🔥', locked: true },
    { id: 6, name: 'Night Owl', description: 'Complete a session after midnight', icon: '🦉', locked: true },
  ];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        data-testid="achievements-modal"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-6xl h-[85vh] bg-dark-800 rounded-2xl shadow-2xl overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-white/10">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                <Trophy className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">
                  Achievements & Progress
                </h2>
                <p className="text-sm text-gray-400">
                  Your learning milestones and rewards
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-dark-700 rounded-lg transition"
              aria-label="Close achievements"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto h-[calc(85vh-88px)]">
            {/* Stats Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-dark-700 rounded-xl p-6 text-center">
                <Trophy className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                <p className="text-2xl font-bold text-white">0</p>
                <p className="text-sm text-gray-400">Achievements Unlocked</p>
              </div>
              
              <div className="bg-dark-700 rounded-xl p-6 text-center">
                <Star className="w-12 h-12 text-yellow-400 mx-auto mb-3" />
                <p className="text-2xl font-bold text-white">0</p>
                <p className="text-sm text-gray-400">Total Points</p>
              </div>
              
              <div className="bg-dark-700 rounded-xl p-6 text-center">
                <Award className="w-12 h-12 text-green-400 mx-auto mb-3" />
                <p className="text-2xl font-bold text-white">-</p>
                <p className="text-sm text-gray-400">Current Rank</p>
              </div>
            </div>

            {/* Achievements Grid */}
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-4">
                <Target className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">
                  Available Achievements
                </h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {mockAchievements.map((achievement) => (
                  <motion.div
                    key={achievement.id}
                    whileHover={{ scale: 1.02 }}
                    className={`
                      bg-dark-700 rounded-xl p-4 border-2 transition-all
                      ${achievement.locked 
                        ? 'border-dark-600 opacity-60' 
                        : 'border-purple-500/30 shadow-lg shadow-purple-500/10'
                      }
                    `}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`
                        w-16 h-16 rounded-full flex items-center justify-center text-3xl
                        ${achievement.locked ? 'bg-dark-600 grayscale' : 'bg-purple-500/20'}
                      `}>
                        {achievement.locked ? '🔒' : achievement.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-white mb-1 truncate">
                          {achievement.name}
                        </h4>
                        <p className="text-xs text-gray-400 line-clamp-2">
                          {achievement.description}
                        </p>
                        {achievement.locked && (
                          <div className="mt-2">
                            <div className="h-1.5 bg-dark-600 rounded-full overflow-hidden">
                              <div className="h-full bg-purple-500 w-0" />
                            </div>
                            <p className="text-xs text-gray-500 mt-1">Locked</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-dark-700 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-white">
                  Recent Activity
                </h3>
              </div>
              <div className="h-32 flex items-center justify-center border-2 border-dashed border-dark-600 rounded-lg">
                <div className="text-center">
                  <div className="text-3xl mb-2">🎯</div>
                  <p className="text-gray-400">No recent achievements</p>
                  <p className="text-xs text-gray-500 mt-1">Start learning to unlock rewards!</p>
                </div>
              </div>
            </div>

            {/* Info Card */}
            <div className="mt-6 bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <Gift className="w-5 h-5 text-purple-400 mt-0.5" />
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-purple-400 mb-1">
                    Unlock Achievements
                  </h4>
                  <p className="text-xs text-gray-400">
                    Complete learning sessions, maintain streaks, and explore different topics to unlock exclusive badges and rewards!
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default Achievements;
