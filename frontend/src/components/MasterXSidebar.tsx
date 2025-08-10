'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  MessageSquare, 
  BarChart3, 
  Settings, 
  User, 
  History, 
  Zap, 
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Plus,
  Search,
  Clock,
  Star,
  Archive,
  Trash2
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface MasterXSidebarProps {
  isOpen: boolean
  onToggle: () => void
  activeView: string
  onViewChange: (view: string) => void
  className?: string
}

const navigationItems = [
  {
    id: 'chat',
    label: 'Quantum Chat',
    icon: MessageSquare,
    description: 'AI-powered conversations',
    gradient: 'from-purple-500 to-cyan-500'
  },
  {
    id: 'history',
    label: 'Chat History',
    icon: History,
    description: 'Previous conversations',
    gradient: 'from-cyan-500 to-emerald-500'
  },
  {
    id: 'analytics',
    label: 'Intelligence Analytics',
    icon: BarChart3,
    description: 'Performance insights',
    gradient: 'from-emerald-500 to-yellow-500'
  },
  {
    id: 'tools',
    label: 'AI Tools',
    icon: Zap,
    description: 'Advanced utilities',
    gradient: 'from-yellow-500 to-orange-500'
  },
  {
    id: 'profile',
    label: 'Neural Profile',
    icon: User,
    description: 'Account settings',
    gradient: 'from-orange-500 to-red-500'
  },
  {
    id: 'settings',
    label: 'Quantum Settings',
    icon: Settings,
    description: 'System configuration',
    gradient: 'from-red-500 to-purple-500'
  }
]

const recentChats = [
  { id: '1', title: 'Quantum Computing Explained', time: '2 hours ago', starred: true },
  { id: '2', title: 'Machine Learning Algorithms', time: '1 day ago', starred: false },
  { id: '3', title: 'Neural Network Architecture', time: '2 days ago', starred: true },
  { id: '4', title: 'AI Ethics Discussion', time: '3 days ago', starred: false },
  { id: '5', title: 'Deep Learning Fundamentals', time: '1 week ago', starred: false }
]

export function MasterXSidebar({ 
  isOpen, 
  onToggle, 
  activeView, 
  onViewChange, 
  className = '' 
}: MasterXSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)

  const filteredChats = recentChats.filter(chat =>
    chat.title.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <>
      {/* Mobile Backdrop */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
            onClick={onToggle}
          />
        )}
      </AnimatePresence>

      {/* Sidebar Container */}
      <motion.div
        initial={false}
        animate={{ 
          width: isOpen ? 320 : 0,
          opacity: isOpen ? 1 : 0
        }}
        transition={{ 
          duration: 0.4, 
          ease: [0.4, 0, 0.2, 1] 
        }}
        className={cn(
          "fixed left-0 top-0 h-full z-50 overflow-hidden",
          "lg:relative lg:z-auto",
          className
        )}
      >
        <div className="w-80 h-full glass-morph-premium border-r border-purple-500/30 flex flex-col">
          {/* Modern Premium Header */}
          <div className="p-6 border-b border-primary/10">
            <div className="flex items-center justify-between mb-8">
              <motion.div
                className="flex items-center space-x-4"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
              >
                <motion.div
                  className="relative w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center shadow-lg"
                  whileHover={{
                    scale: 1.05,
                    boxShadow: "0 20px 40px rgba(168, 85, 247, 0.3)"
                  }}
                  whileTap={{ scale: 0.95 }}
                  animate={{
                    boxShadow: [
                      "0 8px 24px rgba(168, 85, 247, 0.2)",
                      "0 12px 32px rgba(6, 182, 212, 0.25)",
                      "0 8px 24px rgba(168, 85, 247, 0.2)"
                    ]
                  }}
                  transition={{
                    boxShadow: { duration: 4, repeat: Infinity, ease: "easeInOut" }
                  }}
                >
                  <Brain className="h-5 w-5 text-white" />
                </motion.div>
                <div>
                  <motion.h1
                    className="text-display-lg font-bold text-primary"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                  >
                    MasterX
                  </motion.h1>
                  <motion.p
                    className="text-caption text-tertiary"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 }}
                  >
                    AI Platform
                  </motion.p>
                </div>
              </motion.div>

              <motion.button
                onClick={onToggle}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="p-2 rounded-lg glass-modern hover:bg-white/10 transition-all duration-200 lg:hidden"
              >
                <ChevronLeft className="h-4 w-4 text-secondary" />
              </motion.button>
            </div>

            {/* Modern New Chat Button */}
            <motion.button
              whileHover={{
                scale: 1.02,
                y: -2
              }}
              whileTap={{ scale: 0.98 }}
              className="w-full glass-premium rounded-xl p-4 text-white font-medium transition-all duration-300 group"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <div className="flex items-center justify-center space-x-3">
                <Plus className="h-4 w-4" />
                <span className="text-body-base font-semibold">New Chat</span>
                <Sparkles className="h-4 w-4 opacity-70 group-hover:opacity-100 transition-opacity" />
              </div>
            </motion.button>
          </div>

          {/* Modern Navigation */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 space-y-2">
              <div className="mb-6">
                <h3 className="text-caption text-quaternary mb-4">
                  Navigation
                </h3>
              </div>

              {navigationItems.map((item, index) => {
                const isActive = activeView === item.id
                const isHovered = hoveredItem === item.id

                return (
                  <motion.button
                    key={item.id}
                    onClick={() => onViewChange(item.id)}
                    onMouseEnter={() => setHoveredItem(item.id)}
                    onMouseLeave={() => setHoveredItem(null)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: index * 0.05 }}
                    whileHover={{
                      x: 4,
                      transition: { duration: 0.2 }
                    }}
                    whileTap={{ scale: 0.98 }}
                    className={cn(
                      "w-full flex items-center space-x-3 p-3 rounded-lg transition-all duration-200 group text-left",
                      isActive
                        ? "glass-premium text-primary"
                        : "hover:glass-modern text-secondary hover:text-primary"
                    )}
                  >
                    <div className={cn(
                      "w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200",
                      isActive
                        ? "bg-gradient-primary text-white shadow-md"
                        : "bg-surface-tertiary text-tertiary group-hover:bg-surface-elevated group-hover:text-secondary"
                    )}>
                      <item.icon className="h-4 w-4" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className={cn(
                        "text-body-sm font-medium truncate",
                        isActive ? "text-primary" : "text-secondary group-hover:text-primary"
                      )}>
                        {item.label}
                      </div>
                      <div className={cn(
                        "text-caption truncate",
                        isActive ? "text-secondary" : "text-quaternary group-hover:text-tertiary"
                      )}>
                        {item.description}
                      </div>
                    </div>

                    {isActive && (
                      <motion.div
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="w-1.5 h-1.5 bg-accent-cyan-400 rounded-full"
                      />
                    )}
                  </motion.button>
                )
              })}
            </div>

            {/* Recent Chats Section */}
            <div className="p-4 border-t border-primary/10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-caption text-quaternary">Recent</h3>
                <button className="text-caption text-accent hover:text-accent-secondary transition-colors">
                  View All
                </button>
              </div>

              <div className="space-y-1">
                {filteredChats.slice(0, 5).map((chat, index) => (
                  <motion.button
                    key={chat.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    whileHover={{ x: 2 }}
                    className="w-full flex items-center space-x-3 p-2 rounded-lg hover:glass-modern transition-all duration-200 group text-left"
                  >
                    <div className="w-6 h-6 rounded-md bg-surface-tertiary flex items-center justify-center group-hover:bg-surface-elevated transition-colors">
                      <MessageSquare className="h-3 w-3 text-tertiary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-body-sm font-medium text-secondary truncate group-hover:text-primary transition-colors">
                        {chat.title}
                      </div>
                      <div className="text-caption text-quaternary">
                        {chat.time}
                      </div>
                    </div>
                    {chat.starred && (
                      <Star className="h-3 w-3 text-accent-cyan-400 fill-current" />
                    )}
                  </motion.button>
                ))}
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-purple-500/20">
            <div className="flex items-center justify-between text-xs text-plasma-white/40">
              <span>Quantum v3.0</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span>Online</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Toggle Button (when sidebar is closed) */}
      {!isOpen && (
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={onToggle}
          className="fixed top-4 left-4 z-50 p-3 glass-morph rounded-xl hover:bg-purple-500/20 transition-all duration-200 lg:hidden"
        >
          <ChevronRight className="h-5 w-5 text-plasma-white/70" />
        </motion.button>
      )}
    </>
  )
}
