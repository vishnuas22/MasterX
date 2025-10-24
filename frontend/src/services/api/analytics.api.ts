// **Purpose:** Fetch user analytics and learning progress data

// **What This File Contributes:**
// 1. Learning performance metrics
// 2. Emotion trends over time
// 3. Topic mastery data
// 4. Learning velocity
// 5. Session statistics

// **Implementation:**
// ```typescript
import apiClient from './client';
import type { 
  PerformanceMetrics,
  EmotionTrend,
  TopicMastery,
  LearningVelocity,
  SessionStats
} from '@types/api.types';

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


// **Caching Strategy:**
// - Performance data: Cache 5 minutes (React Query)
// - Emotion trends: Cache 10 minutes
// - Topic mastery: Cache 15 minutes
// - Insights: Cache 30 minutes

// **Performance:**
// - All requests use React Query automatic caching
// - Parallel requests when loading dashboard
// - Stale-while-revalidate pattern

// **Connected Files:**
// - ← `services/api/client.ts`
// - ← `types/api.types.ts`
// - → `store/analyticsStore.ts`
// - → `components/analytics/*` (charts, stats)
// - → `pages/Dashboard.tsx`

// **Backend Integration:**
// ```
// GET /api/v1/analytics/performance/:userId  ← getPerformance()
// GET /api/v1/analytics/emotions/:userId     ← getEmotionTrends()
// GET /api/v1/analytics/topics/:userId       ← getTopicMastery()
// GET /api/v1/analytics/velocity/:userId     ← getLearningVelocity()
// GET /api/v1/analytics/sessions/:userId     ← getSessionStats()
// GET /api/v1/analytics/insights/:userId     ← getInsights()