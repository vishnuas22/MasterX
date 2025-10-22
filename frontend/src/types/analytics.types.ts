/**
 * Analytics & Gamification Types
 * 
 * Matches backend analytics and gamification systems:
 * - services/analytics.py
 * - services/gamification.py
 * - services/spaced_repetition.py
 * 
 * @module types/analytics
 */

// ============================================================================
// PERFORMANCE METRICS
// ============================================================================

/**
 * User performance metrics over a time range
 * Provides aggregate statistics on learning activity
 */
export interface PerformanceMetrics {
  user_id: string;
  time_range: string; // e.g., "7d", "30d", "90d"
  total_sessions: number;
  total_time_minutes: number;
  avg_session_duration_minutes: number;
  total_questions_answered: number;
  accuracy_rate: number; // 0.0 - 1.0
  improvement_rate: number; // percentage change
  streak_days: number;
  timestamp: string; // ISO 8601
}

/**
 * Emotion trends over time
 * Tracks emotional patterns during learning
 */
export interface EmotionTrend {
  date: string; // YYYY-MM-DD
  dominant_emotion: string;
  positive_ratio: number; // 0.0 - 1.0
  negative_ratio: number; // 0.0 - 1.0
  neutral_ratio: number; // 0.0 - 1.0
  avg_learning_readiness: number; // 0.0 - 1.0
  avg_cognitive_load: number; // 0.0 - 1.0
}

/**
 * Mastery level for a specific topic
 * IRT-based difficulty and progress tracking
 */
export interface TopicMastery {
  topic: string;
  mastery_level: number; // 0.0 - 1.0
  questions_attempted: number;
  questions_correct: number;
  avg_difficulty: number; // 0.0 - 1.0
  last_practiced: string; // ISO 8601
  next_review: string; // ISO 8601
}

/**
 * Learning velocity metrics
 * Tracks pace of learning progress
 */
export interface LearningVelocity {
  user_id: string;
  current_velocity: number; // topics per week
  velocity_trend: 'increasing' | 'stable' | 'decreasing';
  predicted_completion_days: number;
  milestones_achieved: number;
  timestamp: string; // ISO 8601
}

/**
 * Session statistics and patterns
 * Identifies optimal learning times and habits
 */
export interface SessionStats {
  user_id: string;
  total_sessions: number;
  avg_duration_minutes: number;
  longest_session_minutes: number;
  favorite_time_of_day: string; // e.g., "morning", "afternoon", "evening"
  most_productive_day: string; // e.g., "Monday", "Tuesday"
  total_cost_usd: number;
}

// ============================================================================
// GAMIFICATION TYPES
// ============================================================================

/**
 * Complete gamification profile
 * Includes XP, level, ELO rating, and achievements
 */
export interface GamificationProfile {
  user_id: string;
  level: number;
  xp: number;
  xp_to_next_level: number;
  elo_rating: number; // Chess-style rating system
  current_streak: number; // consecutive days
  longest_streak: number;
  total_sessions: number;
  total_questions: number;
  total_time_minutes: number;
  achievements_unlocked: string[];
  badges: string[];
  rank: number; // Global rank
}

/**
 * Achievement definition
 * Unlockable rewards for milestones
 */
export interface Achievement {
  id: string;
  name: string;
  description: string;
  type: 'milestone' | 'streak' | 'mastery' | 'social' | 'special';
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  xp_reward: number;
  icon: string; // emoji or icon name
  criteria: Record<string, unknown>;
  unlocked_at?: string; // ISO 8601 (if unlocked)
}

/**
 * Leaderboard entry
 * Ranks users by various metrics
 */
export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  user_name: string;
  metric_value: number; // XP, ELO, streak, etc.
  avatar_url?: string;
  is_current_user: boolean;
}

/**
 * Activity record for gamification system
 * Tracks individual learning events
 */
export interface ActivityRecord {
  user_id: string;
  session_id: string;
  question_difficulty: number; // 0.0 - 1.0
  success: boolean;
  time_spent_seconds: number;
  timestamp: string; // ISO 8601
}

// ============================================================================
// SPACED REPETITION TYPES
// ============================================================================

/**
 * Spaced repetition flashcard
 * SM-2+ algorithm for optimal review scheduling
 */
export interface SpacedRepetitionCard {
  id: string;
  user_id: string;
  topic: string;
  content: Record<string, unknown>; // Flexible card content
  difficulty: 'easy' | 'medium' | 'hard';
  interval_days: number;
  repetitions: number;
  ease_factor: number; // SM-2 ease factor
  next_review_date: string; // ISO 8601
  last_reviewed_date?: string; // ISO 8601
  created_at: string; // ISO 8601
}

/**
 * Review result after card practice
 * Updates scheduling parameters
 */
export interface ReviewResult {
  card_id: string;
  quality: number; // 0-5 (SM-2 quality rating)
  new_interval_days: number;
  new_ease_factor: number;
  next_review_date: string; // ISO 8601
}

// ============================================================================
// DASHBOARD TYPES
// ============================================================================

/**
 * Complete dashboard data
 * Aggregates all user analytics for main view
 */
export interface DashboardData {
  user: {
    name: string;
    level: number;
    xp: number;
    streak: number;
  };
  today: {
    sessions: number;
    time_minutes: number;
    questions: number;
    accuracy: number; // 0.0 - 1.0
  };
  week: {
    sessions: number;
    time_minutes: number;
    improvement: number; // percentage
  };
  insights: string[]; // AI-generated insights
  recent_achievements: Achievement[];
  upcoming_reviews: number;
}

/**
 * Chart data point for time series
 */
export interface ChartDataPoint {
  date: string; // YYYY-MM-DD
  value: number;
  label?: string;
}

/**
 * Progress snapshot for a specific metric
 */
export interface ProgressSnapshot {
  metric_name: string;
  current_value: number;
  previous_value: number;
  change_percentage: number;
  trend: 'up' | 'down' | 'stable';
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

/**
 * Type guard to check if object is PerformanceMetrics
 * @param obj - Object to check
 * @returns True if object matches PerformanceMetrics interface
 */
export const isPerformanceMetrics = (obj: unknown): obj is PerformanceMetrics => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'user_id' in obj &&
    'total_sessions' in obj &&
    'accuracy_rate' in obj
  );
};

/**
 * Type guard to check if object is GamificationProfile
 * @param obj - Object to check
 * @returns True if object matches GamificationProfile interface
 */
export const isGamificationProfile = (obj: unknown): obj is GamificationProfile => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'level' in obj &&
    'xp' in obj &&
    'elo_rating' in obj
  );
};

/**
 * Type guard to check if object is Achievement
 * @param obj - Object to check
 * @returns True if object matches Achievement interface
 */
export const isAchievement = (obj: unknown): obj is Achievement => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    'name' in obj &&
    'type' in obj &&
    'rarity' in obj
  );
};

/**
 * Type guard to check if object is SpacedRepetitionCard
 * @param obj - Object to check
 * @returns True if object matches SpacedRepetitionCard interface
 */
export const isSpacedRepetitionCard = (obj: unknown): obj is SpacedRepetitionCard => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    'topic' in obj &&
    'ease_factor' in obj &&
    'next_review_date' in obj
  );
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Calculate XP progress percentage to next level
 * @param currentXP - Current XP amount
 * @param xpToNext - XP required for next level
 * @returns Progress percentage (0-100)
 */
export const calculateXPProgress = (currentXP: number, xpToNext: number): number => {
  if (xpToNext === 0) return 100;
  return Math.min(100, Math.max(0, (currentXP / xpToNext) * 100));
};

/**
 * Get achievement rarity color
 * @param rarity - Achievement rarity
 * @returns CSS color class
 */
export const getAchievementRarityColor = (rarity: Achievement['rarity']): string => {
  const colors = {
    common: '#8E8E93', // Gray
    rare: '#64D2FF', // Blue
    epic: '#BF5AF2', // Purple
    legendary: '#FFD60A', // Gold
  };
  return colors[rarity];
};

/**
 * Format streak days for display
 * @param days - Number of consecutive days
 * @returns Formatted string with emoji
 */
export const formatStreak = (days: number): string => {
  if (days === 0) return 'No streak';
  if (days === 1) return '1 day 🔥';
  return `${days} days 🔥`;
};

/**
 * Calculate accuracy color based on rate
 * @param accuracyRate - Accuracy rate (0.0 - 1.0)
 * @returns CSS color for visualization
 */
export const getAccuracyColor = (accuracyRate: number): string => {
  if (accuracyRate >= 0.9) return '#30D158'; // Green - Excellent
  if (accuracyRate >= 0.7) return '#FFD60A'; // Yellow - Good
  if (accuracyRate >= 0.5) return '#FF9F0A'; // Orange - Fair
  return '#FF453A'; // Red - Needs improvement
};

/**
 * Format time duration for display
 * @param minutes - Total minutes
 * @returns Human-readable string
 */
export const formatDuration = (minutes: number): string => {
  if (minutes < 60) {
    return `${Math.round(minutes)}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = Math.round(minutes % 60);
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
};

/**
 * Get trend emoji
 * @param trend - Trend direction
 * @returns Emoji representing trend
 */
export const getTrendEmoji = (trend: 'up' | 'down' | 'stable'): string => {
  const emojis = {
    up: '📈',
    down: '📉',
    stable: '➡️',
  };
  return emojis[trend];
};

/**
 * Calculate level from total XP
 * Uses exponential curve: XP = 100 * (level ^ 1.5)
 * @param xp - Total XP
 * @returns Level number
 */
export const calculateLevel = (xp: number): number => {
  return Math.floor(Math.pow(xp / 100, 1 / 1.5));
};

/**
 * Calculate XP required for specific level
 * @param level - Target level
 * @returns XP required
 */
export const calculateXPForLevel = (level: number): number => {
  return Math.floor(100 * Math.pow(level, 1.5));
};

/**
 * Get next review date color based on urgency
 * @param nextReviewDate - ISO 8601 date string
 * @returns CSS color
 */
export const getReviewUrgencyColor = (nextReviewDate: string): string => {
  const now = new Date();
  const reviewDate = new Date(nextReviewDate);
  const daysUntilReview = Math.ceil((reviewDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
  
  if (daysUntilReview <= 0) return '#FF453A'; // Overdue - Red
  if (daysUntilReview <= 1) return '#FF9F0A'; // Due soon - Orange
  if (daysUntilReview <= 3) return '#FFD60A'; // Coming up - Yellow
  return '#8E8E93'; // Not urgent - Gray
};

/**
 * Format review date for display
 * @param nextReviewDate - ISO 8601 date string
 * @returns Human-readable string
 */
export const formatReviewDate = (nextReviewDate: string): string => {
  const now = new Date();
  const reviewDate = new Date(nextReviewDate);
  const daysUntilReview = Math.ceil((reviewDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
  
  if (daysUntilReview < 0) return 'Overdue';
  if (daysUntilReview === 0) return 'Today';
  if (daysUntilReview === 1) return 'Tomorrow';
  if (daysUntilReview <= 7) return `In ${daysUntilReview} days`;
  if (daysUntilReview <= 30) return `In ${Math.floor(daysUntilReview / 7)} weeks`;
  return `In ${Math.floor(daysUntilReview / 30)} months`;
};
