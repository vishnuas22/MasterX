import apiClient from './client';

// These types will be defined properly when we have the full type definitions
type PerformanceMetrics = any;
type EmotionTrend = any;
type TopicMastery = any;
type LearningVelocity = any;
type SessionStats = any;

export const analyticsAPI = {
  /**
   * Get user performance metrics
   * GET /api/v1/analytics/performance/:userId
   */
  getPerformance: async (userId: string, timeRange?: string): Promise<PerformanceMetrics> => {
    const { data } = await apiClient.get<PerformanceMetrics>(
      `/api/v1/analytics/performance/${userId}`,
      {
        params: { time_range: timeRange || '7d' },
      }
    );
    return data;
  },

  /**
   * Get emotion trends
   * GET /api/v1/analytics/emotions/:userId
   */
  getEmotionTrends: async (userId: string, days: number = 7): Promise<EmotionTrend[]> => {
    const { data } = await apiClient.get<EmotionTrend[]>(
      `/api/v1/analytics/emotions/${userId}`,
      {
        params: { days },
      }
    );
    return data;
  },

  /**
   * Get topic mastery
   * GET /api/v1/analytics/topics/:userId
   */
  getTopicMastery: async (userId: string): Promise<TopicMastery[]> => {
    const { data } = await apiClient.get<TopicMastery[]>(
      `/api/v1/analytics/topics/${userId}`
    );
    return data;
  },

  /**
   * Get learning velocity
   * GET /api/v1/analytics/velocity/:userId
   */
  getLearningVelocity: async (userId: string): Promise<LearningVelocity> => {
    const { data } = await apiClient.get<LearningVelocity>(
      `/api/v1/analytics/velocity/${userId}`
    );
    return data;
  },

  /**
   * Get session statistics
   * GET /api/v1/analytics/sessions/:userId
   */
  getSessionStats: async (userId: string): Promise<SessionStats> => {
    const { data } = await apiClient.get<SessionStats>(
      `/api/v1/analytics/sessions/${userId}`
    );
    return data;
  },

  /**
   * Get insights (AI-generated)
   * GET /api/v1/analytics/insights/:userId
   */
  getInsights: async (userId: string): Promise<{ insights: string[] }> => {
    const { data } = await apiClient.get<{ insights: string[] }>(
      `/api/v1/analytics/insights/${userId}`
    );
    return data;
  },
};
