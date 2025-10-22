import apiClient from './client';

// These types will be defined properly when we have the full type definitions
type GamificationProfile = any;
type Achievement = any;
type LeaderboardEntry = any;
type ActivityRecord = any;

export const gamificationAPI = {
  /**
   * Get user gamification profile
   * GET /api/v1/gamification/profile/:userId
   */
  getProfile: async (userId: string): Promise<GamificationProfile> => {
    const { data } = await apiClient.get<GamificationProfile>(
      `/api/v1/gamification/profile/${userId}`
    );
    return data;
  },

  /**
   * Record learning activity
   * POST /api/v1/gamification/record-activity
   */
  recordActivity: async (activity: ActivityRecord): Promise<{
    xp_gained: number;
    achievements_unlocked: Achievement[];
    level_up: boolean;
  }> => {
    const { data } = await apiClient.post(
      '/api/v1/gamification/record-activity',
      activity
    );
    return data;
  },

  /**
   * Get leaderboard
   * GET /api/v1/gamification/leaderboard
   */
  getLeaderboard: async (
    timeRange: 'daily' | 'weekly' | 'monthly' | 'all-time' = 'weekly'
  ): Promise<LeaderboardEntry[]> => {
    const { data } = await apiClient.get<LeaderboardEntry[]>(
      '/api/v1/gamification/leaderboard',
      {
        params: { time_range: timeRange },
      }
    );
    return data;
  },

  /**
   * Get user achievements
   * GET /api/v1/gamification/achievements/:userId
   */
  getAchievements: async (userId: string): Promise<Achievement[]> => {
    const { data } = await apiClient.get<Achievement[]>(
      `/api/v1/gamification/achievements/${userId}`
    );
    return data;
  },

  /**
   * Get streak information
   * GET /api/v1/gamification/streak/:userId
   */
  getStreak: async (userId: string): Promise<{
    current_streak: number;
    longest_streak: number;
    last_activity: string;
  }> => {
    const { data } = await apiClient.get(
      `/api/v1/gamification/streak/${userId}`
    );
    return data;
  },
};
