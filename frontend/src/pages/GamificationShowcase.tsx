/**
 * Gamification Showcase - GROUP 13 Gamification Components Demo
 * 
 * Purpose: Visual testing and demonstration of gamification components
 * 
 * Components tested:
 * - AchievementBadge (4 rarities, unlock states, progress)
 * - StreakCounter (active/inactive, milestones, freezes)
 * - LevelProgress (XP progress, level up, animations)
 * - Leaderboard (rankings, current user, rank changes)
 * 
 * Following AGENTS_FRONTEND.md:
 * - Type-safe TypeScript
 * - Accessible (WCAG 2.1 AA)
 * - Responsive design
 * - Performance optimized
 */

import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { 
  AchievementBadge, 
  StreakCounter, 
  LevelProgress, 
  Leaderboard 
} from '@/components/gamification';
import type { Achievement } from '@/components/gamification/AchievementBadge';
import type { LeaderboardEntry } from '@/components/gamification/Leaderboard';

export default function GamificationShowcase() {
  const [justLeveledUp, setJustLeveledUp] = useState(false);
  const [recentXP, setRecentXP] = useState<number | undefined>(undefined);

  // Sample achievements data
  const sampleAchievements: Achievement[] = [
    {
      id: '1',
      name: 'First Steps',
      description: 'Complete your first learning session',
      icon: '🎯',
      rarity: 'common',
      xpReward: 10,
      unlockedAt: new Date('2025-10-15'),
    },
    {
      id: '2',
      name: 'Week Warrior',
      description: 'Maintain a 7-day streak',
      icon: '⚡',
      rarity: 'rare',
      xpReward: 50,
      unlockedAt: new Date('2025-10-20'),
    },
    {
      id: '3',
      name: 'Perfect Score',
      description: 'Get 100% on a difficult quiz',
      icon: '💎',
      rarity: 'epic',
      xpReward: 100,
      unlockedAt: new Date('2025-10-25'),
    },
    {
      id: '4',
      name: 'Master Mind',
      description: 'Reach level 50',
      icon: '👑',
      rarity: 'legendary',
      xpReward: 500,
    },
    {
      id: '5',
      name: 'Speed Demon',
      description: 'Complete 10 questions in under 5 minutes',
      icon: '🚀',
      rarity: 'rare',
      xpReward: 75,
      progress: 0.6, // 60% progress
    },
  ];

  // Sample leaderboard data
  const sampleLeaderboard: LeaderboardEntry[] = [
    {
      userId: '1',
      username: 'AlexCoder',
      rank: 1,
      metric: 2450,
      level: 25,
      avatar: undefined,
      rankChange: 2,
    },
    {
      userId: '2',
      username: 'SarahMath',
      rank: 2,
      metric: 2380,
      level: 24,
      avatar: undefined,
      rankChange: -1,
    },
    {
      userId: '3',
      username: 'MikeScience',
      rank: 3,
      metric: 2310,
      level: 23,
      avatar: undefined,
      rankChange: 1,
    },
    {
      userId: 'current',
      username: 'You',
      rank: 15,
      metric: 1850,
      level: 18,
      avatar: undefined,
      rankChange: 3,
    },
  ];

  // Simulate level up
  const handleLevelUp = () => {
    setJustLeveledUp(true);
    setTimeout(() => setJustLeveledUp(false), 5000);
  };

  // Simulate XP gain
  const handleXPGain = () => {
    setRecentXP(25);
    setTimeout(() => setRecentXP(undefined), 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            🎮 Gamification Components Showcase
          </h1>
          <p className="text-gray-400 text-lg">
            GROUP 13: Achievement Badges, Streaks, Level Progress & Leaderboards
          </p>
        </div>

        {/* Achievement Badges Section */}
        <section>
          <Card className="p-6">
            <h2 className="text-2xl font-semibold text-white mb-6">
              🏆 Achievement Badges
            </h2>
            
            <div className="space-y-6">
              {/* Unlocked Achievements */}
              <div>
                <h3 className="text-lg text-gray-300 mb-4">Unlocked Achievements</h3>
                <div className="flex flex-wrap gap-6">
                  {sampleAchievements.filter(a => a.unlockedAt).map((achievement) => (
                    <AchievementBadge
                      key={achievement.id}
                      achievement={achievement}
                      size="md"
                      showDetails
                    />
                  ))}
                </div>
              </div>

              {/* Locked Achievements */}
              <div>
                <h3 className="text-lg text-gray-300 mb-4">Locked Achievements (with progress)</h3>
                <div className="flex flex-wrap gap-6">
                  {sampleAchievements.filter(a => !a.unlockedAt).map((achievement) => (
                    <AchievementBadge
                      key={achievement.id}
                      achievement={achievement}
                      size="md"
                      showDetails
                    />
                  ))}
                </div>
              </div>

              {/* Different Sizes */}
              <div>
                <h3 className="text-lg text-gray-300 mb-4">Different Sizes</h3>
                <div className="flex items-end gap-6">
                  <AchievementBadge
                    achievement={sampleAchievements[0]}
                    size="sm"
                  />
                  <AchievementBadge
                    achievement={sampleAchievements[0]}
                    size="md"
                  />
                  <AchievementBadge
                    achievement={sampleAchievements[0]}
                    size="lg"
                  />
                </div>
              </div>
            </div>
          </Card>
        </section>

        {/* Streak Counter & Level Progress Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Streak Counter */}
          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">
              🔥 Streak Counter
            </h2>
            <StreakCounter
              currentStreak={15}
              longestStreak={30}
              freezesAvailable={2}
              lastActivity={new Date()}
            />
          </section>

          {/* Level Progress */}
          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">
              ⭐ Level Progress
            </h2>
            <LevelProgress
              level={18}
              currentXP={1850}
              xpToNextLevel={650}
              recentXPGain={recentXP}
              justLeveledUp={justLeveledUp}
            />
            <div className="mt-4 flex gap-4">
              <Button onClick={handleXPGain} variant="primary" size="sm">
                Simulate +25 XP
              </Button>
              <Button onClick={handleLevelUp} variant="success" size="sm">
                Simulate Level Up
              </Button>
            </div>
          </section>
        </div>

        {/* Leaderboard Section */}
        <section>
          <h2 className="text-2xl font-semibold text-white mb-4">
            🏅 Leaderboard
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* ELO Leaderboard */}
            <Leaderboard
              entries={sampleLeaderboard}
              currentUserId="current"
              metricType="elo"
              title="Top Players (ELO)"
              showTopOnly={10}
            />

            {/* XP Leaderboard */}
            <Leaderboard
              entries={sampleLeaderboard.map((e, i) => ({
                ...e,
                metric: e.metric * 10, // Convert ELO to XP scale
              }))}
              currentUserId="current"
              metricType="xp"
              title="Top Learners (XP)"
              showTopOnly={10}
            />

            {/* Streak Leaderboard */}
            <Leaderboard
              entries={sampleLeaderboard.map((e, i) => ({
                ...e,
                metric: Math.floor(e.metric / 100), // Convert to streak days
              }))}
              currentUserId="current"
              metricType="streak"
              title="Top Streaks"
              showTopOnly={10}
            />
          </div>
        </section>

        {/* Component Details */}
        <Card className="p-6 bg-gray-800/50">
          <h2 className="text-2xl font-semibold text-white mb-4">
            📋 Component Details
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-gray-300">
            <div>
              <h3 className="font-semibold text-white mb-2">AchievementBadge</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>4 rarity levels with unique colors</li>
                <li>Unlock animations with glow effects</li>
                <li>Progress rings for locked achievements</li>
                <li>Detailed modal with metadata</li>
                <li>3 size variants (sm, md, lg)</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-white mb-2">StreakCounter</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>Dynamic emoji based on streak length</li>
                <li>Active/inactive daily status</li>
                <li>Streak freeze system</li>
                <li>Milestone progress tracking</li>
                <li>Gradient colors for achievements</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-white mb-2">LevelProgress</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>XP progress bar with percentage</li>
                <li>Level-up confetti celebration</li>
                <li>Recent XP gain animations</li>
                <li>Dynamic gradient colors by level</li>
                <li>Clean level badge design</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-white mb-2">Leaderboard</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>3 ranking metrics (ELO, XP, Streak)</li>
                <li>Medal icons for top 3</li>
                <li>Rank change indicators (↑↓)</li>
                <li>Current user highlighting</li>
                <li>WCAG 2.1 AA compliant</li>
              </ul>
            </div>
          </div>
        </Card>

        {/* Integration Notes */}
        <Card className="p-6 bg-gradient-to-br from-blue-900/30 to-purple-900/30 border-blue-500/30">
          <h2 className="text-2xl font-semibold text-white mb-4">
            🔗 Backend Integration
          </h2>
          <div className="space-y-4 text-gray-300">
            <p>
              All components are ready for backend integration via the <code className="text-blue-400 bg-gray-800 px-2 py-1 rounded">gamification.api.ts</code> service:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li>
                <code className="text-blue-400">GET /api/v1/gamification/stats/{'{userId}'}</code> - User stats
              </li>
              <li>
                <code className="text-blue-400">GET /api/v1/gamification/achievements</code> - All achievements
              </li>
              <li>
                <code className="text-blue-400">GET /api/v1/gamification/leaderboard</code> - Rankings
              </li>
              <li>
                <code className="text-blue-400">POST /api/v1/gamification/record-activity</code> - Record XP
              </li>
            </ul>
            <p className="mt-4">
              Use the <code className="text-blue-400 bg-gray-800 px-2 py-1 rounded">useGamification</code> hook for easy data fetching with React Query caching.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
