'use client'

import { useState, useEffect, useRef } from 'react'
import { Brain, Target, BookOpen, Award, TrendingUp, Users, Zap, Clock, Sparkles, Activity, Star, Trophy, Flame, Crown } from 'lucide-react'

interface LearningStats {
  totalSessions: number
  hoursLearned: number
  conceptsMastered: number
  currentStreak: number
  level: number
  xp: number
  nextLevelXp: number
  quantumEfficiency: number
  neuralConnections: number
}

interface LearningMode {
  id: string
  name: string
  description: string
  usageCount: number
  efficiency: number
  icon: any
  color: string
  gradient: string
  lastUsed: string
}

export function QuantumLearningDashboard() {
  const [stats, setStats] = useState<LearningStats>({
    totalSessions: 47,
    hoursLearned: 126,
    conceptsMastered: 89,
    currentStreak: 12,
    level: 8,
    xp: 2340,
    nextLevelXp: 3000,
    quantumEfficiency: 94.7,
    neuralConnections: 1247
  })

  const [learningModes] = useState<LearningMode[]>([
    {
      id: 'adaptive-quantum',
      name: 'Adaptive Quantum',
      description: 'AI-driven adaptive learning',
      usageCount: 23,
      efficiency: 0.92,
      icon: Brain,
      color: 'purple',
      gradient: 'from-purple-500 to-violet-600',
      lastUsed: '2 hours ago'
    },
    {
      id: 'socratic-discovery',
      name: 'Socratic Discovery',
      description: 'Question-based learning',
      usageCount: 18,
      efficiency: 0.87,
      icon: Target,
      color: 'cyan',
      gradient: 'from-cyan-500 to-blue-600',
      lastUsed: '5 hours ago'
    },
    {
      id: 'debug-mastery',
      name: 'Debug Mastery',
      description: 'Knowledge gap identification',
      usageCount: 15,
      efficiency: 0.89,
      icon: Zap,
      color: 'amber',
      gradient: 'from-amber-500 to-orange-600',
      lastUsed: '1 day ago'
    },
    {
      id: 'creative-synthesis',
      name: 'Creative Synthesis',
      description: 'Creative learning approaches',
      usageCount: 11,
      efficiency: 0.85,
      icon: Sparkles,
      color: 'emerald',
      gradient: 'from-emerald-500 to-green-600',
      lastUsed: '2 days ago'
    }
  ])

  const [particles, setParticles] = useState<Array<{id: number, x: number, y: number, delay: number}>>([])
  const dashboardRef = useRef<HTMLDivElement>(null)

  // Create quantum particles
  useEffect(() => {
    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 5
    }))
    setParticles(newParticles)
  }, [])

  const xpProgress = (stats.xp / stats.nextLevelXp) * 100

  return (
    <div ref={dashboardRef} className="max-w-7xl mx-auto p-6 space-y-8 relative">
      {/* Quantum Particle Background */}
      <div className="quantum-particles">
        {particles.map((particle) => (
          <div
            key={particle.id}
            className="particle"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              animationDelay: `${particle.delay}s`
            }}
          />
        ))}
      </div>

      {/* Revolutionary Header with 3D Effect */}
      <div className="text-center mb-12 relative">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 via-transparent to-cyan-500/20 blur-3xl"></div>
        <div className="relative">
          <h1 className="quantum-display quantum-glow mb-6">
            Quantum Intelligence Dashboard
          </h1>
          <div className="w-64 h-1 mx-auto mb-6 rounded-full overflow-hidden relative">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 via-cyan-500 to-emerald-500 animate-holographic-shift"></div>
          </div>
          <p className="learning-body text-gray-300">
            Real-time monitoring and analytics for your AI platform
          </p>
        </div>
      </div>

      {/* Revolutionary Stats Grid with 3D Cards */}
      <div className="quantum-grid">
        <QuantumStatCard
          icon={Activity}
          title="Active Users"
          value="1,215"
          change="+12.5%"
          color="purple"
          description="Connected learners"
          particles={true}
        />
        <QuantumStatCard
          icon={Clock}
          title="Total Sessions"
          value="9,042"
          change="+8.7%"
          color="cyan"
          description="Learning interactions"
          particles={true}
        />
        <QuantumStatCard
          icon={Zap}
          title="Response Time"
          value="79.87ms"
          change="-5.3%"
          color="emerald"
          description="AI processing speed"
          particles={true}
        />
        <QuantumStatCard
          icon={Brain}
          title="Neural Efficiency"
          value="94.7%"
          change="+2.3%"
          color="amber"
          description="AI optimization level"
          particles={true}
        />
      </div>

      {/* Revolutionary System Performance with Real-time Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="quantum-card">
          <h3 className="intelligence-title text-purple-300 mb-6">System Performance</h3>
          <div className="space-y-6">
            <QuantumProgressBar
              label="CPU Usage"
              value={58.4}
              color="purple"
              animated={true}
            />
            <QuantumProgressBar
              label="Memory Usage"
              value={65.6}
              color="cyan"
              animated={true}
            />
            <QuantumProgressBar
              label="Network Activity"
              value={95.2}
              color="emerald"
              animated={true}
            />
          </div>
        </div>

        <div className="quantum-card">
          <h3 className="intelligence-title text-cyan-300 mb-6">Neural Networks</h3>
          <div className="space-y-6">
            <QuantumProgressBar
              label="Primary Neural Core"
              value={87}
              color="purple"
              status="Active"
            />
            <QuantumProgressBar
              label="Quantum Processor"
              value={72}
              color="cyan"
              status="Optimal"
            />
            <QuantumProgressBar
              label="Learning Engine"
              value={76}
              color="emerald"
              status="Training"
            />
            <QuantumProgressBar
              label="Memory Matrix"
              value={64}
              color="amber"
              status="Stable"
            />
          </div>
        </div>
      </div>

      {/* Revolutionary Learning Modes with Advanced Visualizations */}
      <div className="quantum-card">
        <h3 className="intelligence-title text-purple-300 mb-8">Quantum Learning Modes Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {learningModes.map((mode, index) => (
            <QuantumLearningModeCard key={mode.id} mode={mode} index={index} />
          ))}
        </div>
      </div>

      {/* Advanced Real-time Activity Feed */}
      <div className="quantum-card">
        <h3 className="intelligence-title text-emerald-300 mb-8">Real-time Learning Activities</h3>
        <div className="space-y-4">
          {recentActivities.map((activity, index) => (
            <QuantumActivityCard key={index} activity={activity} index={index} />
          ))}
        </div>
      </div>

      {/* Revolutionary Achievement Showcase */}
      <div className="quantum-card">
        <h3 className="intelligence-title text-amber-300 mb-8">Recent Achievements</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {achievements.map((achievement, index) => (
            <QuantumAchievementCard key={index} achievement={achievement} index={index} />
          ))}
        </div>
      </div>
    </div>
  )
}

// Revolutionary Stat Card Component
function QuantumStatCard({ 
  icon: Icon, 
  title, 
  value, 
  change, 
  color, 
  description, 
  particles = false 
}: {
  icon: any
  title: string
  value: string
  change: string
  color: string
  description: string
  particles?: boolean
}) {
  const colorClasses = {
    purple: 'border-purple-500/30 from-purple-900/20 to-purple-800/10 text-purple-400',
    cyan: 'border-cyan-500/30 from-cyan-900/20 to-cyan-800/10 text-cyan-400',
    emerald: 'border-emerald-500/30 from-emerald-900/20 to-emerald-800/10 text-emerald-400',
    amber: 'border-amber-500/30 from-amber-900/20 to-amber-800/10 text-amber-400'
  }

  return (
    <div className={`quantum-card relative overflow-hidden group ${colorClasses[color]}`}>
      {particles && (
        <div className="absolute top-0 right-0 w-20 h-20 opacity-20">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className={`absolute w-1 h-1 bg-current rounded-full animate-particle-float`}
              style={{
                left: `${Math.random() * 80}%`,
                top: `${Math.random() * 80}%`,
                animationDelay: `${i * 0.2}s`
              }}
            />
          ))}
        </div>
      )}
      
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className={`p-3 rounded-xl bg-gradient-to-r ${colorClasses[color].split(' ')[1]} border border-current/20`}>
            <Icon className="h-6 w-6 animate-quantum-pulse" />
          </div>
          <span className={`text-sm font-semibold px-2 py-1 rounded-full ${
            change.startsWith('+') ? 'text-emerald-400 bg-emerald-400/10' : 'text-red-400 bg-red-400/10'
          }`}>
            {change}
          </span>
        </div>
        
        <div className="space-y-2">
          <p className="precision-small text-gray-300">{title}</p>
          <p className="neural-heading font-bold text-white quantum-glow">{value}</p>
          <p className="data-micro text-gray-400">{description}</p>
        </div>
      </div>
    </div>
  )
}

// Advanced Progress Bar Component
function QuantumProgressBar({ 
  label, 
  value, 
  color, 
  animated = false, 
  status 
}: {
  label: string
  value: number
  color: string
  animated?: boolean
  status?: string
}) {
  const colorClasses = {
    purple: 'from-purple-500 to-violet-600',
    cyan: 'from-cyan-500 to-blue-600',
    emerald: 'from-emerald-500 to-green-600',
    amber: 'from-amber-500 to-orange-600'
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="precision-small text-gray-300">{label}</span>
        <div className="flex items-center space-x-2">
          {status && (
            <span className="data-micro px-2 py-1 rounded-full bg-emerald-500/20 text-emerald-400">
              {status}
            </span>
          )}
          <span className="precision-small font-semibold text-white">{value}%</span>
        </div>
      </div>
      <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden relative">
        <div 
          className={`h-full bg-gradient-to-r ${colorClasses[color]} rounded-full transition-all duration-1000 relative overflow-hidden`}
          style={{ width: `${value}%` }}
        >
          {animated && (
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-data-flow"></div>
          )}
        </div>
      </div>
    </div>
  )
}

// Enhanced Learning Mode Card
function QuantumLearningModeCard({ mode, index }: { mode: LearningMode; index: number }) {
  return (
    <div 
      className="quantum-card interactive-card group"
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-3 rounded-xl bg-gradient-to-r ${mode.gradient} shadow-lg group-hover:shadow-xl transition-all duration-300`}>
            <mode.icon className="h-6 w-6 text-white animate-quantum-pulse" />
          </div>
          <div>
            <h4 className="intelligence-title text-white">{mode.name}</h4>
            <p className="precision-small text-gray-400">{mode.description}</p>
          </div>
        </div>
        <span className="data-micro text-gray-500">{mode.lastUsed}</span>
      </div>
      
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="precision-small text-gray-300">Usage Count</span>
          <span className="font-semibold text-purple-300">{mode.usageCount}</span>
        </div>
        
        <QuantumProgressBar
          label="Efficiency"
          value={Math.round(mode.efficiency * 100)}
          color={mode.color}
          animated={true}
        />
      </div>
    </div>
  )
}

// Enhanced Activity Card
function QuantumActivityCard({ activity, index }: { activity: any; index: number }) {
  return (
    <div 
      className="quantum-card interactive-card group border-l-4 border-purple-500/50"
      style={{ animationDelay: `${index * 0.05}s` }}
    >
      <div className="flex items-center space-x-4">
        <div className="p-3 rounded-xl bg-gradient-to-r from-purple-500/20 to-cyan-500/20">
          <activity.icon className="h-5 w-5 text-purple-400 animate-quantum-pulse" />
        </div>
        <div className="flex-1">
          <h4 className="precision-small font-semibold text-white">{activity.title}</h4>
          <p className="data-micro text-gray-400">{activity.description}</p>
        </div>
        <div className="text-right">
          <p className="precision-small font-bold text-emerald-400">+{activity.xp} XP</p>
          <p className="data-micro text-gray-500">{activity.time}</p>
        </div>
      </div>
    </div>
  )
}

// Revolutionary Achievement Card
function QuantumAchievementCard({ achievement, index }: { achievement: any; index: number }) {
  return (
    <div 
      className="quantum-card interactive-card text-center group relative overflow-hidden"
      style={{ animationDelay: `${index * 0.15}s` }}
    >
      <div className="absolute top-0 right-0 w-full h-full bg-gradient-to-br from-amber-500/5 via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
      
      <div className="relative z-10">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-r from-amber-500 to-orange-600 flex items-center justify-center shadow-xl group-hover:shadow-2xl transition-all duration-300 animate-quantum-pulse">
          <achievement.icon className="h-8 w-8 text-white" />
        </div>
        
        <h4 className="intelligence-title text-white mb-2">{achievement.title}</h4>
        <p className="precision-small text-gray-400 mb-3">{achievement.description}</p>
        <p className="data-micro text-amber-300">{achievement.date}</p>
      </div>
    </div>
  )
}

// Sample data
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
    icon: Crown,
    title: 'Quantum Master',
    description: 'Achieved 95%+ efficiency across all learning modes',
    date: '2 hours ago',
  },
  {
    icon: Flame,
    title: 'Learning Streak',
    description: 'Maintained 15-day consecutive learning streak',
    date: '1 day ago',
  },
  {
    icon: Trophy,
    title: 'Memory Architect',
    description: 'Created 10 advanced memory palaces',
    date: '1 week ago',
  },
]