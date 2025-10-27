/**
 * Gamification API Service
 * 
 * Handles gamification and engagement features:
 * - User statistics (level, XP, streaks, ELO rating)
 * - Achievement tracking and unlocking
 * - Global leaderboards (ELO, XP, streaks)
 * - Activity recording for XP and progression
 * 
 * Backend Integration:
 * GET  /api/v1/gamification/stats/{user_id}     - User gamification stats
 * POST /api/v1/gamification/record-activity     - Record learning activity
 * GET  /api/v1/gamification/leaderboard         - Global leaderboard
 * GET  /api/v1/gamification/achievements        - All available achievements
 * 
 * Caching Strategy (via React Query):
 * - User stats: 5 minutes cache
 * - Leaderboard: 2 minutes cache (dynamic data)
 * - Achievements: 30 minutes cache (static data)
 * 
 * @module services/api/gamification.api
 */

import apiClient from './client';

/**
 * User Gamification Statistics
 */
export interface GamificationStats {
  user_id: string;
  level: number;
  xp: number;
  xp_to_next_level: number;
  elo_rating: number;
  current_streak: number;
  longest_streak: number;
  total_sessions: number;
  total_questions: number;
  total_time_minutes: number;
  achievements_unlocked: number;
  badges: string[];
  rank: number;
}

/**
 * Achievement Definition
 */
export interface Achievement {
  id: string;
  name: string;
  description: string;
  type: string;
  rarity: string;
  xp_reward: number;
  icon: string;
  criteria: Record<string, unknown>;
}

/**
 * Leaderboard Entry
 */
export interface LeaderboardEntry {
  user_id: string;
  username: string;
  rank: number;
  score: number;
  level: number;
  avatar?: string;
}

/**
 * Activity Record Request
 */
export interface ActivityRecord {
  user_id: string;
  session_id: string;
  question_difficulty: number;
  success: boolean;
  time_spent_seconds: number;
}

/**
 * Activity Recording Result
 */
export interface ActivityResult {
  xp_gained: number;
  new_level?: number;
  level_up: boolean;
  achievements_unlocked: Achievement[];
  new_badges: string[];
  streak_updated: boolean;
  current_streak: number;
}

/**
 * Leaderboard Response
 */
export interface LeaderboardResponse {
  leaderboard: LeaderboardEntry[];
  metric: string;
  count: number;
}

/**
 * Achievements Response
 */
export interface AchievementsResponse {
  achievements: Achievement[];
  count: number;
}

/**
 * Gamification API endpoints
 */
export const gamificationAPI = {
  /**
   * Get user gamification statistics
   * 
   * Returns comprehensive gamification data including:
   * - Current level and XP progress
   * - ELO rating for competitive matching
   * - Streak information (current and longest)
   * - Total sessions, questions, and time spent
   * - Achievements unlocked count and badges
   * - Global rank
   * 
   * @param userId - User identifier
   * @returns User gamification statistics
   * @throws 404 - User not found
   * @throws 500 - Failed to fetch stats
   * 
   * @example
   * ```typescript
   * const stats = await gamificationAPI.getStats('user-123');
   * console.log(stats.level); // 5
   * console.log(stats.xp); // 1250
   * console.log(stats.current_streak); // 7 days
   * console.log(stats.rank); // 42 (global rank)
   * ```
   */
  getStats: async (userId: string): Promise<GamificationStats> => {
    const { data } = await apiClient.get<GamificationStats>(
      `/api/v1/gamification/stats/${userId}`
    );
    return data;
  },

  /**
   * Record learning activity
   * 
   * Records user activity for gamification rewards:
   * - Calculates XP gain based on difficulty and success
   * - Checks for level up
   * - Unlocks achievements if criteria met
   * - Updates streaks
   * - Awards badges for milestones
   * 
   * XP Calculation:
   * - Base XP = difficulty * 10
   * - Success bonus = +50%
   * - Streak multiplier = 1 + (streak_days * 0.1)
   * 
   * @param activity - Activity details (difficulty, success, time)
   * @returns Activity result with XP gained and rewards
   * @throws 400 - Invalid activity data
   * @throws 500 - Failed to record activity
   * 
   * @example
   * ```typescript
   * const result = await gamificationAPI.recordActivity({
   *   user_id: 'user-123',
   *   session_id: 'session-abc',
   *   question_difficulty: 0.7,
   *   success: true,
   *   time_spent_seconds: 120
   * });
   * 
   * if (result.level_up) {
   *   console.log(`Level up! New level: ${result.new_level}`);
   * }
   * 
   * if (result.achievements_unlocked.length > 0) {
   *   console.log('New achievements:', result.achievements_unlocked);
   * }
   * ```
   */
  recordActivity: async (activity: ActivityRecord): Promise<ActivityResult> => {
    const { data } = await apiClient.post<ActivityResult>(
      '/api/v1/gamification/record-activity',
      activity
    );
    return data;
  },

  /**
   * Get global leaderboard
   * 
   * Returns top users ranked by specified metric.
   * 
   * Available metrics:
   * - elo_rating (default): Competitive skill rating
   * - xp: Total experience points
   * - streak: Current learning streak
   * - total_time: Total learning time
   * 
   * @param limit - Max entries to return (default: 100)
   * @param metric - Ranking metric (default: 'elo_rating')
   * @returns Leaderboard with ranked users
   * @throws 500 - Failed to fetch leaderboard
   * 
   * @example
   * ```typescript
   * // Get top 50 by ELO rating
   * const { leaderboard } = await gamificationAPI.getLeaderboard(50, 'elo_rating');
   * 
   * leaderboard.forEach((entry, index) => {
   *   console.log(`#${entry.rank} ${entry.username} - ${entry.score}`);
   * });
   * ```
   */
  getLeaderboard: async (
    limit: number = 100,
    metric: string = 'elo_rating'
  ): Promise<LeaderboardResponse> => {
    const { data } = await apiClient.get<LeaderboardResponse>(
      '/api/v1/gamification/leaderboard',
      {
        params: { limit, metric },
      }
    );
    return data;
  },

  /**
   * Get all available achievements
   * 
   * Returns complete list of all achievements that can be unlocked.
   * 
   * Achievement types:
   * - milestone: Reach specific milestones (100 questions, etc.)
   * - streak: Maintain learning streaks
   * - mastery: Master specific topics
   * - speed: Complete tasks quickly
   * - perfect: Perfect scores
   * 
   * Rarity levels: common, uncommon, rare, epic, legendary
   * 
   * @returns All achievements with details
   * @throws 500 - Failed to fetch achievements
   * 
   * @example
   * ```typescript
   * const { achievements } = await gamificationAPI.getAchievements();
   * 
   * const legendaryAchievements = achievements.filter(
   *   a => a.rarity === 'legendary'
   * );
   * 
   * console.log(`${legendaryAchievements.length} legendary achievements`);
   * ```
   */
  getAchievements: async (): Promise<AchievementsResponse> => {
    const { data } = await apiClient.get<AchievementsResponse>(
      '/api/v1/gamification/achievements'
    );
    return data;
  },
};