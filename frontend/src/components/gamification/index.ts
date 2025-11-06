/**
 * Gamification Components Index
 * 
 * Exports all gamification-related components:
 * - AchievementBadge: Display achievement badges with unlock animations
 * - AchievementNotification: Toast notifications for unlocked achievements
 * - AchievementNotificationManager: Global notification manager
 * - StreakCounter: Track daily learning streaks
 * - LevelProgress: Show user level and XP progress
 * - Leaderboard: Display competitive rankings
 */

export { AchievementBadge } from './AchievementBadge';
export type { Achievement, AchievementBadgeProps } from './AchievementBadge';

export { AchievementNotification } from './AchievementNotification';
export type { AchievementNotificationProps } from './AchievementNotification';

export { AchievementNotificationManager } from './AchievementNotificationManager';

export { StreakCounter } from './StreakCounter';
export type { StreakCounterProps } from './StreakCounter';

export { LevelProgress } from './LevelProgress';
export type { LevelProgressProps } from './LevelProgress';

export { Leaderboard } from './Leaderboard';
export type { LeaderboardEntry, LeaderboardProps } from './Leaderboard';

export { XPGainAnimation } from './XPGainAnimation';
export type { XPGainAnimationProps } from './XPGainAnimation';

export { AchievementGallery } from './AchievementGallery';
export type { AchievementGalleryProps } from './AchievementGallery';
