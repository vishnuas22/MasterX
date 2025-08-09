'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  User, 
  Edit3, 
  Camera, 
  Mail, 
  Phone, 
  MapPin, 
  Calendar, 
  Award, 
  TrendingUp, 
  Brain, 
  Zap, 
  Sparkles,
  Settings,
  Crown,
  Star,
  Target,
  Clock,
  MessageSquare,
  BarChart3,
  Shield
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface MasterXProfileSectionProps {
  className?: string
}

const achievements = [
  { id: 1, title: 'Quantum Explorer', description: 'Completed 100 AI conversations', icon: Brain, color: 'from-purple-500 to-cyan-500' },
  { id: 2, title: 'Neural Pioneer', description: 'Used advanced AI features', icon: Zap, color: 'from-cyan-500 to-emerald-500' },
  { id: 3, title: 'Intelligence Master', description: 'Achieved expert level', icon: Crown, color: 'from-emerald-500 to-yellow-500' },
  { id: 4, title: 'Code Wizard', description: 'Generated 1000+ lines of code', icon: Sparkles, color: 'from-yellow-500 to-orange-500' },
]

const stats = [
  { label: 'Conversations', value: '1,247', icon: MessageSquare, change: '+12%' },
  { label: 'AI Interactions', value: '5,832', icon: Brain, change: '+8%' },
  { label: 'Code Generated', value: '15.2K', icon: Zap, change: '+24%' },
  { label: 'Hours Saved', value: '342', icon: Clock, change: '+18%' },
]

export function MasterXProfileSection({ className = '' }: MasterXProfileSectionProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [profileData, setProfileData] = useState({
    name: 'Quantum User',
    email: 'user@masterx.ai',
    title: 'AI Researcher',
    location: 'San Francisco, CA',
    joinDate: 'January 2024',
    bio: 'Passionate about quantum intelligence and the future of AI. Building the next generation of intelligent systems.'
  })

  const handleSave = () => {
    setIsEditing(false)
    // Save profile data logic here
  }

  return (
    <div className={cn("h-full overflow-y-auto p-6", className)}>
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Profile Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-morph-premium rounded-3xl p-8 border border-purple-500/30"
        >
          <div className="flex flex-col lg:flex-row items-start lg:items-center space-y-6 lg:space-y-0 lg:space-x-8">
            {/* Avatar */}
            <div className="relative">
              <motion.div 
                className="w-32 h-32 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-full flex items-center justify-center text-4xl font-bold text-white"
                whileHover={{ scale: 1.05 }}
              >
                {profileData.name.split(' ').map(n => n[0]).join('')}
              </motion.div>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="absolute bottom-2 right-2 w-10 h-10 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-full flex items-center justify-center text-white shadow-lg"
              >
                <Camera className="h-5 w-5" />
              </motion.button>
            </div>

            {/* Profile Info */}
            <div className="flex-1">
              <div className="flex items-start justify-between mb-4">
                <div>
                  {isEditing ? (
                    <input
                      type="text"
                      value={profileData.name}
                      onChange={(e) => setProfileData(prev => ({ ...prev, name: e.target.value }))}
                      className="text-3xl font-bold text-plasma-white bg-transparent border-b border-purple-500/50 focus:border-cyan-400 focus:outline-none"
                    />
                  ) : (
                    <h1 className="text-3xl font-bold text-plasma-white">{profileData.name}</h1>
                  )}
                  <div className="flex items-center space-x-2 mt-2">
                    <Crown className="h-5 w-5 text-yellow-400" />
                    <span className="text-lg text-cyan-400 font-medium">Premium Member</span>
                  </div>
                </div>
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => isEditing ? handleSave() : setIsEditing(true)}
                  className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl text-white hover:from-purple-700 hover:to-cyan-700 transition-all"
                >
                  <Edit3 className="h-4 w-4" />
                  <span>{isEditing ? 'Save' : 'Edit'}</span>
                </motion.button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-plasma-white/70">
                <div className="flex items-center space-x-2">
                  <Mail className="h-4 w-4 text-cyan-400" />
                  <span>{profileData.email}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <MapPin className="h-4 w-4 text-cyan-400" />
                  <span>{profileData.location}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Calendar className="h-4 w-4 text-cyan-400" />
                  <span>Joined {profileData.joinDate}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Target className="h-4 w-4 text-cyan-400" />
                  <span>{profileData.title}</span>
                </div>
              </div>

              {isEditing ? (
                <textarea
                  value={profileData.bio}
                  onChange={(e) => setProfileData(prev => ({ ...prev, bio: e.target.value }))}
                  className="w-full mt-4 p-3 glass-morph rounded-xl text-plasma-white bg-transparent border border-purple-500/20 focus:border-cyan-400 focus:outline-none resize-none"
                  rows={3}
                />
              ) : (
                <p className="mt-4 text-plasma-white/80 leading-relaxed">{profileData.bio}</p>
              )}
            </div>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-morph-premium rounded-2xl p-6 border border-purple-500/20 hover:border-purple-500/40 transition-all"
            >
              <div className="flex items-center justify-between mb-4">
                <stat.icon className="h-8 w-8 text-cyan-400" />
                <span className="text-sm text-emerald-400 font-medium">{stat.change}</span>
              </div>
              <div className="text-2xl font-bold text-plasma-white mb-1">{stat.value}</div>
              <div className="text-sm text-plasma-white/60">{stat.label}</div>
            </motion.div>
          ))}
        </div>

        {/* Achievements */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-morph-premium rounded-3xl p-8 border border-purple-500/30"
        >
          <div className="flex items-center space-x-3 mb-6">
            <Award className="h-6 w-6 text-yellow-400" />
            <h2 className="text-2xl font-bold text-plasma-white">Achievements</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {achievements.map((achievement, index) => (
              <motion.div
                key={achievement.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className="flex items-center space-x-4 p-4 glass-morph rounded-xl hover:bg-purple-500/10 transition-all"
              >
                <div className={`w-12 h-12 bg-gradient-to-r ${achievement.color} rounded-xl flex items-center justify-center`}>
                  <achievement.icon className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-plasma-white">{achievement.title}</h3>
                  <p className="text-sm text-plasma-white/60">{achievement.description}</p>
                </div>
                <Star className="h-5 w-5 text-yellow-400 fill-current" />
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Activity Chart Placeholder */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-morph-premium rounded-3xl p-8 border border-purple-500/30"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <BarChart3 className="h-6 w-6 text-cyan-400" />
              <h2 className="text-2xl font-bold text-plasma-white">Activity Overview</h2>
            </div>
            <div className="flex items-center space-x-2 text-sm text-plasma-white/60">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              <span>Last 30 days</span>
            </div>
          </div>
          
          <div className="h-64 flex items-center justify-center">
            <div className="text-center">
              <motion.div 
                className="w-16 h-16 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-2xl flex items-center justify-center mx-auto mb-4"
                animate={{ 
                  boxShadow: [
                    "0 0 20px rgba(168, 85, 247, 0.3)",
                    "0 0 40px rgba(168, 85, 247, 0.6)",
                    "0 0 20px rgba(168, 85, 247, 0.3)"
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <BarChart3 className="w-8 h-8 text-white" />
              </motion.div>
              <h3 className="text-lg font-semibold text-plasma-white mb-2">Advanced Analytics</h3>
              <p className="text-plasma-white/60">Detailed activity charts and insights coming soon</p>
            </div>
          </div>
        </motion.div>

        {/* Security Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="glass-morph-premium rounded-3xl p-8 border border-purple-500/30"
        >
          <div className="flex items-center space-x-3 mb-6">
            <Shield className="h-6 w-6 text-emerald-400" />
            <h2 className="text-2xl font-bold text-plasma-white">Security Status</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3 p-4 glass-morph rounded-xl">
              <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
              <span className="text-plasma-white">Account Secure</span>
            </div>
            <div className="flex items-center space-x-3 p-4 glass-morph rounded-xl">
              <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
              <span className="text-plasma-white">2FA Enabled</span>
            </div>
            <div className="flex items-center space-x-3 p-4 glass-morph rounded-xl">
              <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
              <span className="text-plasma-white">Data Encrypted</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
