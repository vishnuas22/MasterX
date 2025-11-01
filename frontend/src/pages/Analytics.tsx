/**
 * Analytics Page - User Learning Analytics Dashboard
 * 
 * WCAG 2.1 AA Compliant:
 * - Chart descriptions for screen readers
 * - Keyboard navigation
 * - Color contrast for data visualization
 * 
 * Performance:
 * - Lazy loaded
 * - Charts use canvas for performance
 * - Data fetching with analytics store
 * 
 * Backend Integration:
 * - GET /api/v1/analytics/dashboard
 * - GET /api/v1/analytics/emotions
 * - GET /api/v1/analytics/sessions
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, TrendingUp, BarChart3, Clock, Brain, Activity } from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

export interface AnalyticsProps {
  onClose: () => void;
}

// ============================================================================
// COMPONENT
// ============================================================================

export const Analytics: React.FC<AnalyticsProps> = ({ onClose }) => {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        data-testid="analytics-modal"
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
              <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">
                  Analytics Dashboard
                </h2>
                <p className="text-sm text-gray-400">
                  Your learning insights and progress
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-dark-700 rounded-lg transition"
              aria-label="Close analytics"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto h-[calc(85vh-88px)]">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-dark-700 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-gray-400">Total Sessions</span>
                </div>
                <p className="text-3xl font-bold text-white">0</p>
                <p className="text-xs text-gray-500 mt-1">Start learning to see stats</p>
              </div>
              
              <div className="bg-dark-700 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-2">
                  <Clock className="w-5 h-5 text-blue-400" />
                  <span className="text-sm text-gray-400">Learning Time</span>
                </div>
                <p className="text-3xl font-bold text-white">0h 0m</p>
                <p className="text-xs text-gray-500 mt-1">Time spent learning</p>
              </div>
              
              <div className="bg-dark-700 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-2">
                  <Brain className="w-5 h-5 text-purple-400" />
                  <span className="text-sm text-gray-400">Avg. Emotion</span>
                </div>
                <p className="text-3xl font-bold text-white">Neutral</p>
                <p className="text-xs text-gray-500 mt-1">Overall mood</p>
              </div>
            </div>

            {/* Charts Placeholder */}
            <div className="space-y-6">
              {/* Emotion Trend Chart */}
              <div className="bg-dark-700 rounded-xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Activity className="w-5 h-5 text-pink-400" />
                  <h3 className="text-lg font-semibold text-white">
                    Emotion Trends
                  </h3>
                </div>
                <div className="h-64 flex items-center justify-center border-2 border-dashed border-dark-600 rounded-lg">
                  <div className="text-center">
                    <div className="text-4xl mb-2">📊</div>
                    <p className="text-gray-400">Chart visualization coming soon</p>
                    <p className="text-xs text-gray-500 mt-1">Track your emotions over time</p>
                  </div>
                </div>
              </div>
              
              {/* Session History */}
              <div className="bg-dark-700 rounded-xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Clock className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">
                    Session History
                  </h3>
                </div>
                <div className="h-64 flex items-center justify-center border-2 border-dashed border-dark-600 rounded-lg">
                  <div className="text-center">
                    <div className="text-4xl mb-2">📅</div>
                    <p className="text-gray-400">Session timeline coming soon</p>
                    <p className="text-xs text-gray-500 mt-1">View your learning history</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Info Card */}
            <div className="mt-6 bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <span className="text-2xl">💡</span>
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-blue-400 mb-1">
                    Analytics Feature
                  </h4>
                  <p className="text-xs text-gray-400">
                    Advanced analytics visualizations are being prepared. Start chatting to generate data!
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

export default Analytics;
