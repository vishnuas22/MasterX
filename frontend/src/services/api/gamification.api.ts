// **Purpose:** Fetch gamification data (achievements, leaderboard, etc.)

// **What This File Contributes:**
// 1. User gamification profile
// 2. Achievement tracking
// 3. Leaderboard data
// 4. Streak information
// 5. Level progression

// **Implementation:**
// ```typescript
import apiClient from './client';
import type { 
  GamificationProfile,
  Achievement,
  LeaderboardEntry,
  ActivityRecord
} from '@types/api.types';

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


// **Performance:**
// - Profile data cached for 5 minutes
// - Leaderboard cached for 2 minutes (dynamic data)
// - Achievements cached until unlocked (long TTL)

// **Connected Files:**
// - ← `services/api/client.ts`
// - → `components/gamification/*`
// - → `store/chatStore.ts` (records activity after each interaction)

// **Backend Integration:**
// ```
// GET  /api/v1/gamification/profile/:userId        ← getProfile()
// POST /api/v1/gamification/record-activity        ← recordActivity()
// GET  /api/v1/gamification/leaderboard            ← getLeaderboard()
// GET  /api/v1/gamification/achievements/:userId   ← getAchievements()
// GET  /api/v1/gamification/streak/:userId         ← getStreak()