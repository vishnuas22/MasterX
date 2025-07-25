'use client'

import { useState, useEffect } from 'react'
import { Brain, Target, BookOpen, Award, TrendingUp, Users, Zap, Clock } from 'lucide-react'

interface LearningStats {
  totalSessions: number
  hoursLearned: number
  conceptsMastered: number
  currentStreak: number
  level: number
  xp: number
  nextLevelXp: number
}

interface LearningMode {
  name: string
  description: string
  usageCount: number
  efficiency: number
  icon: any
}

export function LearningDashboard() {
  const [stats, setStats] = useState<LearningStats>({
    totalSessions: 47,
    hoursLearned: 126,
    conceptsMastered: 89,
    currentStreak: 12,
    level: 8,
    xp: 2340,
    nextLevelXp: 3000
  })

  const [learningModes] = useState<LearningMode[]>([
    {
      name: 'Adaptive Quantum',
      description: 'AI-driven adaptive learning',
      usageCount: 23,
      efficiency: 0.92,
      icon: Brain
    },
    {
      name: 'Socratic Discovery',
      description: 'Question-based learning',
      usageCount: 18,
      efficiency: 0.87,
      icon: Target
    },
    {
      name: 'Debug Mastery',
      description: 'Knowledge gap identification',
      usageCount: 15,
      efficiency: 0.89,
      icon: Zap
    },
    {
      name: 'Creative Synthesis',
      description: 'Creative learning approaches',
      usageCount: 11,
      efficiency: 0.85,
      icon: BookOpen
    }
  ])

  const xpProgress = (stats.xp / stats.nextLevelXp) * 100

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Enhanced Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold quantum-text-glow mb-4">Learning Dashboard</h1>
        <div className="quantum-shimmer h-1 w-32 mx-auto mb-4 rounded-full"></div>
        <p className="text-gray-400">Track your progress with the Quantum Intelligence Engine</p>
      </div>

      {/* Enhanced Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Clock}
          title="Total Sessions"
          value={stats.totalSessions.toString()}
          subtitle="Learning sessions completed"
          color="purple"
        />
        <StatCard
          icon={TrendingUp}
          title="Hours Learned"
          value={stats.hoursLearned.toString()}
          subtitle="Total learning time"
          color="blue"
        />
        <StatCard
          icon={Target}
          title="Concepts Mastered"
          value={stats.conceptsMastered.toString()}
          subtitle="Knowledge milestones reached"
          color="green"
        />
        <StatCard
          icon={Award}
          title="Current Streak"
          value={`${stats.currentStreak} days`}
          subtitle="Consecutive learning days"
          color="orange"
        />
      </div>

      {/* Enhanced Level Progress */}
      <div className="glass-morph p-8 rounded-xl border border-purple-500/20 interactive-card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center quantum-glow">
              <span className="text-2xl font-bold">{stats.level}</span>
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-purple-300">Level {stats.level}</h3>
              <p className="text-gray-400">Quantum Learner</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xl font-semibold">{stats.xp} / {stats.nextLevelXp} XP</p>
            <p className="text-sm text-gray-400">{stats.nextLevelXp - stats.xp} XP to next level</p>
          </div>
        </div>
        <div className="relative">
          <div className="w-full bg-slate-700 rounded-full h-4 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-purple-500 via-blue-500 to-purple-400 h-4 rounded-full transition-all duration-1000 quantum-glow relative"
              style={{ width: `${xpProgress}%` }}
            >
              <div className="absolute inset-0 quantum-shimmer rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Learning Modes Analysis */}
      <div className="glass-morph p-8 rounded-xl border border-purple-500/20 interactive-card">
        <h3 className="text-2xl font-semibold mb-6 text-purple-300 quantum-text-glow">Quantum Learning Modes Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {learningModes.map((mode, index) => (
            <div key={index} className="glass-morph p-6 rounded-lg border border-purple-500/10 interactive-card"
                 style={{ animationDelay: `${index * 0.1}s` }}>
              <div className="flex items-center space-x-4 mb-4">
                <div className="p-3 rounded-full bg-purple-500/20">
                  <mode.icon className="h-6 w-6 text-purple-400 quantum-pulse" />
                </div>
                <div>
                  <h4 className="font-semibold text-lg">{mode.name}</h4>
                  <p className="text-sm text-gray-400">{mode.description}</p>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Usage Count</span>
                  <span className="font-semibold text-purple-300">{mode.usageCount}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Efficiency</span>
                  <span className="font-semibold text-green-400">{Math.round(mode.efficiency * 100)}%</span>
                </div>
                <div className="relative">
                  <div className="w-full bg-slate-600 rounded-full h-3 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full transition-all duration-1000 quantum-glow"
                      style={{ width: `${mode.efficiency * 100}%`, animationDelay: `${index * 0.2}s` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Recent Activities */}
      <div className="glass-morph p-8 rounded-xl border border-purple-500/20 interactive-card">
        <h3 className="text-2xl font-semibold mb-6 text-purple-300 quantum-text-glow">Recent Learning Activities</h3>
        <div className="space-y-4">
          {recentActivities.map((activity, index) => (
            <div key={index} className="flex items-center space-x-4 p-4 glass-morph rounded-lg interactive-card border border-purple-500/10"
                 style={{ animationDelay: `${index * 0.1}s` }}>
              <div className="p-2 rounded-full bg-purple-500/20">
                <activity.icon className="h-5 w-5 text-purple-400 quantum-pulse" />
              </div>
              <div className="flex-1">
                <p className="font-medium text-white">{activity.title}</p>
                <p className="text-sm text-gray-400">{activity.description}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-purple-300 quantum-text-glow">+{activity.xp} XP</p>
                <p className="text-xs text-gray-400">{activity.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Achievement Showcase */}
      <div className="glass-morph p-8 rounded-xl border border-purple-500/20 interactive-card">
        <h3 className="text-2xl font-semibold mb-6 text-purple-300 quantum-text-glow">Recent Achievements</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {achievements.map((achievement, index) => (
            <div key={index} className="glass-morph p-6 rounded-lg border border-purple-500/30 text-center interactive-card"
                 style={{ animationDelay: `${index * 0.15}s` }}>
              <div className="p-4 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 w-16 h-16 mx-auto mb-4 flex items-center justify-center quantum-glow">
                <achievement.icon className="h-8 w-8 text-white quantum-pulse" />
              </div>
              <h4 className="font-semibold mb-2 text-lg">{achievement.title}</h4>
              <p className="text-sm text-gray-400 mb-3">{achievement.description}</p>
              <p className="text-xs text-purple-300">{achievement.date}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon: Icon, title, value, subtitle, color }: {
  icon: any
  title: string
  value: string
  subtitle: string
  color: 'purple' | 'blue' | 'green' | 'orange'
}) {
  const colorClasses = {
    purple: 'text-purple-400 from-purple-900/50 to-purple-800/30 border-purple-500/30',
    blue: 'text-blue-400 from-blue-900/50 to-blue-800/30 border-blue-500/30',
    green: 'text-green-400 from-green-900/50 to-green-800/30 border-green-500/30',
    orange: 'text-orange-400 from-orange-900/50 to-orange-800/30 border-orange-500/30'
  }

  return (
    <div className={`glass-morph p-6 rounded-xl border interactive-card ${colorClasses[color]}`}>
      <div className="flex items-center space-x-4 mb-4">
        <div className="p-3 rounded-full bg-gradient-to-r from-current/20 to-current/10">
          <Icon className="h-8 w-8 quantum-pulse" />
        </div>
        <div>
          <h3 className="font-semibold text-white text-sm opacity-80">{title}</h3>
          <p className="text-3xl font-bold text-white quantum-text-glow">{value}</p>
        </div>
      </div>
      <p className="text-sm text-gray-400 leading-relaxed">{subtitle}</p>
    </div>
  )
}

const recentActivities = [
  {
    icon: Brain,
    title: 'Quantum Mode Session',
    description: 'Completed adaptive learning session on Machine Learning',
    xp: 150,
    time: '2 hours ago',
  },
  {
    icon: Target,
    title: 'Socratic Discovery',
    description: 'Explored neural network concepts through guided questioning',
    xp: 120,
    time: '5 hours ago',
  },
  {
    icon: BookOpen,
    title: 'Memory Palace Created',
    description: 'Built memory palace for Data Structures concepts',
    xp: 100,
    time: '1 day ago',
  },
  {
    icon: Users,
    title: 'Group Study Session',
    description: 'Participated in collaborative learning session',
    xp: 80,
    time: '2 days ago',
  },
]

const achievements = [
  {
    icon: Award,
    title: 'Quantum Explorer',
    description: 'Completed 50 AI sessions',
    date: '3 days ago',
  },
  {
    icon: TrendingUp,
    title: 'Learning Streak',
    description: '10-day learning streak',
    date: '1 week ago',
  },
  {
    icon: Brain,
    title: 'Memory Master',
    description: 'Created 5 memory palaces',
    date: '2 weeks ago',
  },
]