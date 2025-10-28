/**
 * Gamification Components Index
 * 
 * Exports all gamification-related components:
 * - AchievementBadge: Display achievement badges with unlock animations
 * - StreakCounter: Track daily learning streaks
 * - LevelProgress: Show user level and XP progress
 * - Leaderboard: Display competitive rankings
 */

export { AchievementBadge } from './AchievementBadge';
export type { Achievement, AchievementBadgeProps } from './AchievementBadge';

export { StreakCounter } from './StreakCounter';
export type { StreakCounterProps } from './StreakCounter';

export { LevelProgress } from './LevelProgress';
export type { LevelProgressProps } from './LevelProgress';

export { Leaderboard } from './Leaderboard';
export type { LeaderboardEntry, LeaderboardProps } from './Leaderboard';
