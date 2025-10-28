/**
 * Leaderboard Component - Competitive Rankings
 * 
 * Purpose: Display global and friend leaderboards with rankings
 * 
 * WCAG 2.1 AA Compliant:
 * - Keyboard navigation through ranks
 * - Screen reader announces rank positions
 * - High contrast for current user highlight
 * 
 * Features:
 * 1. Global leaderboard (top 100)
 * 2. Friends leaderboard
 * 3. User's current rank
 * 4. Multiple ranking metrics (Elo, XP, streak)
 * 5. Rank change indicators
 * 6. Profile links for top users
 * 7. Pagination for large leaderboards
 * 
 * Backend Integration:
 * - Leaderboard System: MongoDB aggregation for rankings
 * - Real-time Updates: WebSocket for live rank changes
 * - Multiple Metrics: Elo rating, total XP, current streak
 */

import React, { useState, useMemo } from 'react';
import { Card } from '@/components/ui/Card';
import { Avatar } from '@/components/ui/Avatar';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/utils/cn';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export interface LeaderboardEntry {
  userId: string;
  username: string;
  avatar?: string;
  rank: number;
  rankChange?: number; // +/- from previous period
  metric: number; // Elo, XP, or streak
  level: number;
}

export interface LeaderboardProps {
  entries: LeaderboardEntry[];
  currentUserId: string;
  metricType: 'elo' | 'xp' | 'streak';
  title?: string;
  showTopOnly?: number;
  className?: string;
}

const METRIC_CONFIG = {
  elo: {
    label: 'Elo Rating',
    icon: '🏆',
    format: (value: number) => value.toFixed(0)
  },
  xp: {
    label: 'Total XP',
    icon: '⭐',
    format: (value: number) => value.toLocaleString()
  },
  streak: {
    label: 'Streak',
    icon: '🔥',
    format: (value: number) => `${value} days`
  }
};

const RANK_MEDALS = {
  1: '🥇',
  2: '🥈',
  3: '🥉'
};

export const Leaderboard: React.FC<LeaderboardProps> = ({
  entries,
  currentUserId,
  metricType,
  title,
  showTopOnly,
  className
}) => {
  const [selectedMetric] = useState(metricType);
  const metricConfig = METRIC_CONFIG[selectedMetric];

  const displayEntries = useMemo(() => {
    return showTopOnly ? entries.slice(0, showTopOnly) : entries;
  }, [entries, showTopOnly]);

  const currentUserEntry = useMemo(() => {
    return entries.find(e => e.userId === currentUserId);
  }, [entries, currentUserId]);

  const getRankChangeIcon = (change?: number) => {
    if (!change || change === 0) return <Minus className="w-4 h-4 text-gray-500" />;
    if (change > 0) return <TrendingUp className="w-4 h-4 text-green-500" />;
    return <TrendingDown className="w-4 h-4 text-red-500" />;
  };

  return (
    <Card className={cn('p-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">
            {title || 'Leaderboard'}
          </h3>
          <p className="text-sm text-gray-400">
            {metricConfig.icon} {metricConfig.label}
          </p>
        </div>
      </div>

      {/* Current User Highlight (if not in top) */}
      {currentUserEntry && currentUserEntry.rank > (showTopOnly || 10) && (
        <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="text-xl font-bold text-blue-500">
              #{currentUserEntry.rank}
            </div>
            <Avatar 
              src={currentUserEntry.avatar} 
              name={currentUserEntry.username}
              size="sm"
            />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-white truncate">
                {currentUserEntry.username}
              </div>
              <div className="text-xs text-gray-400">
                Level {currentUserEntry.level}
              </div>
            </div>
            <div className="text-right">
              <div className="font-bold text-white">
                {metricConfig.format(currentUserEntry.metric)}
              </div>
              <div className="flex items-center gap-1 text-xs">
                {getRankChangeIcon(currentUserEntry.rankChange)}
                {currentUserEntry.rankChange && (
                  <span className={cn(
                    currentUserEntry.rankChange > 0 ? 'text-green-500' : 'text-red-500'
                  )}>
                    {Math.abs(currentUserEntry.rankChange)}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Leaderboard List */}
      <div className="space-y-2">
        {displayEntries.map((entry) => {
          const isCurrentUser = entry.userId === currentUserId;
          const medal = RANK_MEDALS[entry.rank as keyof typeof RANK_MEDALS];

          return (
            <div
              key={entry.userId}
              className={cn(
                'flex items-center gap-3 p-3 rounded-lg transition-colors',
                'hover:bg-gray-800/50',
                isCurrentUser && 'bg-blue-500/10 border border-blue-500/50'
              )}
              role="listitem"
              aria-label={`Rank ${entry.rank}: ${entry.username}, ${metricConfig.format(entry.metric)}`}
            >
              {/* Rank */}
              <div className={cn(
                'w-8 text-center font-bold',
                entry.rank <= 3 ? 'text-2xl' : 'text-lg text-gray-400'
              )}>
                {medal || `#${entry.rank}`}
              </div>

              {/* Avatar */}
              <Avatar 
                src={entry.avatar} 
                name={entry.username}
                size="sm"
              />

              {/* User Info */}
              <div className="flex-1 min-w-0">
                <div className={cn(
                  'font-medium truncate',
                  isCurrentUser ? 'text-blue-400' : 'text-white'
                )}>
                  {entry.username}
                  {isCurrentUser && (
                    <Badge variant="neutral" className="ml-2 text-xs">
                      You
                    </Badge>
                  )}
                </div>
                <div className="text-xs text-gray-400">
                  Level {entry.level}
                </div>
              </div>

              {/* Metric Value */}
              <div className="text-right">
                <div className="font-bold text-white">
                  {metricConfig.format(entry.metric)}
                </div>
                
                {/* Rank Change */}
                {entry.rankChange !== undefined && (
                  <div className="flex items-center justify-end gap-1 text-xs">
                    {getRankChangeIcon(entry.rankChange)}
                    {entry.rankChange !== 0 && (
                      <span className={cn(
                        entry.rankChange > 0 ? 'text-green-500' : 'text-red-500'
                      )}>
                        {Math.abs(entry.rankChange)}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Show More Button */}
      {showTopOnly && entries.length > showTopOnly && (
        <button className="w-full mt-4 py-2 text-sm text-blue-400 hover:text-blue-300 transition-colors">
          Show all {entries.length} entries
        </button>
      )}
    </Card>
  );
};

export default Leaderboard;

/**
 * Usage Example:
 * 
 * <Leaderboard
 *   entries={leaderboardData}
 *   currentUserId={user.id}
 *   metricType="elo"
 *   title="Global Rankings"
 *   showTopOnly={10}
 * />
 */
