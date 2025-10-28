/**
 * useGamification Hook
 * 
 * Manages gamification features including:
 * - User stats (level, XP, streaks)
 * - Achievements
 * - Leaderboards
 * - Activity recording
 * 
 * Uses React Query for caching and automatic refetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  gamificationAPI, 
  type GamificationStats,
  type Achievement,
  type LeaderboardResponse,
  type ActivityRecord,
  type ActivityResult
} from '@/services/api/gamification.api';

/**
 * Gamification query keys for cache management
 */
export const GAMIFICATION_KEYS = {
  stats: (userId: string) => ['gamification', 'stats', userId],
  leaderboard: (metric: string, limit: number) => ['gamification', 'leaderboard', metric, limit],
  achievements: () => ['gamification', 'achievements'],
};

/**
 * Hook to get user gamification statistics
 */
export const useGamificationStats = (userId: string) => {
  return useQuery({
    queryKey: GAMIFICATION_KEYS.stats(userId),
    queryFn: () => gamificationAPI.getStats(userId),
    staleTime: 5 * 60 * 1000, // 5 minutes
    enabled: !!userId,
  });
};

/**
 * Hook to get leaderboard
 */
export const useLeaderboard = (metric: string = 'elo_rating', limit: number = 100) => {
  return useQuery({
    queryKey: GAMIFICATION_KEYS.leaderboard(metric, limit),
    queryFn: () => gamificationAPI.getLeaderboard(limit, metric),
    staleTime: 2 * 60 * 1000, // 2 minutes (dynamic data)
  });
};

/**
 * Hook to get all achievements
 */
export const useAchievements = () => {
  return useQuery({
    queryKey: GAMIFICATION_KEYS.achievements(),
    queryFn: () => gamificationAPI.getAchievements(),
    staleTime: 30 * 60 * 1000, // 30 minutes (mostly static)
  });
};

/**
 * Hook to record activity and get rewards
 */
export const useRecordActivity = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (activity: ActivityRecord) => gamificationAPI.recordActivity(activity),
    onSuccess: (result: ActivityResult, variables: ActivityRecord) => {
      // Invalidate user stats to refetch updated data
      queryClient.invalidateQueries({ 
        queryKey: GAMIFICATION_KEYS.stats(variables.user_id) 
      });

      // If there's a level up or achievements, also invalidate leaderboard
      if (result.level_up || result.achievements_unlocked.length > 0) {
        queryClient.invalidateQueries({ 
          queryKey: ['gamification', 'leaderboard'] 
        });
      }
    },
  });
};

/**
 * Combined hook for all gamification data
 */
export const useGamification = (userId: string) => {
  const stats = useGamificationStats(userId);
  const achievements = useAchievements();
  const recordActivity = useRecordActivity();

  return {
    // User stats
    stats: stats.data,
    isLoadingStats: stats.isLoading,
    statsError: stats.error,
    refetchStats: stats.refetch,

    // Achievements
    achievements: achievements.data?.achievements || [],
    isLoadingAchievements: achievements.isLoading,
    achievementsError: achievements.error,

    // Activity recording
    recordActivity: recordActivity.mutate,
    isRecordingActivity: recordActivity.isPending,
    activityResult: recordActivity.data,
  };
};

export default useGamification;
