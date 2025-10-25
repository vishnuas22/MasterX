/**
 * Analytics API Service
 * 
 * Handles analytics and learning progress data:
 * - Real-time dashboard metrics
 * - Performance analysis over time
 * - Learning insights and trends
 * 
 * Backend Integration:
 * GET /api/v1/analytics/dashboard/{user_id}    - Real-time dashboard metrics
 * GET /api/v1/analytics/performance/{user_id}  - Performance analysis
 * 
 * Caching Strategy (via React Query):
 * - Dashboard data: 5 minutes cache
 * - Performance data: 10 minutes cache
 * - Stale-while-revalidate pattern
 * 
 * @module services/api/analytics.api
 */

import apiClient from './client';

/**
 * Dashboard Metrics Response
 */
export interface DashboardMetrics {
  total_interactions: number;
  total_sessions: number;
  average_session_duration: number;
  total_cost: number;
  primary_emotions: Record<string, number>;
  learning_velocity: number;
  ability_estimates: Record<string, number>;
  recent_achievements: Array<{
    id: string;
    title: string;
    earned_at: string;
  }>;
}

/**
 * Performance Analysis Response
 */
export interface PerformanceAnalysis {
  user_id: string;
  time_period: {
    start_date: string;
    end_date: string;
    days: number;
  };
  learning_metrics: {
    total_interactions: number;
    average_daily_interactions: number;
    total_learning_time_minutes: number;
    topics_explored: string[];
  };
  emotion_analysis: {
    primary_emotions: Record<string, number>;
    emotion_trajectory: Array<{
      date: string;
      emotion: string;
      count: number;
    }>;
    learning_readiness_distribution: Record<string, number>;
  };
  performance_trends: {
    ability_growth: Record<string, number>;
    difficulty_progression: Array<{
      date: string;
      difficulty: number;
    }>;
    success_rate: number;
  };
  insights: string[];
}

/**
 * Analytics API endpoints
 */
export const analyticsAPI = {
  /**
   * Get real-time analytics dashboard
   * 
   * Provides comprehensive dashboard metrics including:
   * - Total interactions and sessions
   * - Primary emotions detected
   * - Learning velocity (progress rate)
   * - Ability estimates by subject
   * - Recent achievements
   * 
   * @param userId - User identifier
   * @returns Dashboard metrics
   * @throws 500 - Failed to fetch dashboard data
   * 
   * @example
   * ```typescript
   * const metrics = await analyticsAPI.getDashboard('user-123');
   * console.log(metrics.total_interactions); // 45
   * console.log(metrics.primary_emotions); // { "curiosity": 20, "frustration": 10 }
   * ```
   */
  getDashboard: async (userId: string): Promise<DashboardMetrics> => {
    const { data } = await apiClient.get<DashboardMetrics>(
      `/api/v1/analytics/dashboard/${userId}`
    );
    return data;
  },

  /**
   * Get comprehensive performance analysis
   * 
   * Analyzes user performance over a time period:
   * - Learning metrics (interactions, time, topics)
   * - Emotion analysis (primary emotions, trajectory)
   * - Performance trends (ability growth, success rate)
   * - AI-generated insights
   * 
   * @param userId - User identifier
   * @param daysBack - Number of days to analyze (default: 30)
   * @returns Performance analysis with trends and insights
   * @throws 500 - Failed to fetch performance data
   * 
   * @example
   * ```typescript
   * // Get last 7 days of performance
   * const analysis = await analyticsAPI.getPerformance('user-123', 7);
   * 
   * console.log(analysis.learning_metrics.total_interactions); // 25
   * console.log(analysis.performance_trends.success_rate); // 0.85
   * console.log(analysis.insights); // ["Great progress in calculus!", ...]
   * ```
   */
  getPerformance: async (
    userId: string,
    daysBack: number = 30
  ): Promise<PerformanceAnalysis> => {
    const { data } = await apiClient.get<PerformanceAnalysis>(
      `/api/v1/analytics/performance/${userId}`,
      {
        params: { days_back: daysBack },
      }
    );
    return data;
  },
};