/**
 * Sidebar Component for MasterX Quantum Intelligence Platform
 * 
 * Advanced navigation sidebar with quantum-themed design,
 * session management, and intelligent features.
 */

'use client'

import { useState, useEffect } from 'react'
import { 
  MessageCircle, 
  Brain, 
  BarChart3, 
  BookOpen, 
  Target, 
  Settings, 
  Plus, 
  X,
  History,
  Zap,
  Sparkles,
  Code,
  Palette,
  Eye,
  ChevronRight,
  Trash2
} from 'lucide-react'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  activeView: string
  onViewChange: (view: string) => void
  systemMetrics: any
  sessionState: any
}

interface ChatSession {
  id: string
  title: string
  timestamp: string
  messageCount: number
  taskType: string
}

export function Sidebar({ isOpen, onClose, activeView, onViewChange, systemMetrics, sessionState }: SidebarProps) {
  const [activeSection, setActiveSection] = useState('chat')
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([
    {
      id: '1',
      title: 'Python Development Help',
      timestamp: '2 hours ago',
      messageCount: 15,
      taskType: 'coding'
    },
    {
      id: '2',
      title: 'Machine Learning Concepts',
      timestamp: '1 day ago',
      messageCount: 8,
      taskType: 'reasoning'
    },
    {
      id: '3',
      title: 'Creative Writing Project',
      timestamp: '2 days ago',
      messageCount: 12,
      taskType: 'creative'
    }
  ])

  const navigationItems = [
    {
      id: 'chat',
      label: 'Quantum Chat',
      icon: MessageCircle,
      description: 'AI-powered conversations',
      badge: '3'
    },
    {
      id: 'analytics',
      label: 'Neural Analytics',
      icon: BarChart3,
      description: 'Learning insights',
      badge: null
    },
    {
      id: 'learning',
      label: 'Learning Paths',
      icon: BookOpen,
      description: 'Personalized courses',
      badge: '2'
    },
    {
      id: 'goals',
      label: 'Learning Goals',
      icon: Target,
      description: 'Track progress',
      badge: null
    },
    {
      id: 'settings',
      label: 'Quantum Settings',
      icon: Settings,
      description: 'Platform configuration',
      badge: null
    }
  ]

  const taskTypeIcons = {
    coding: Code,
    reasoning: Brain,
    creative: Palette,
    fast: Zap,
    multimodal: Eye,
    general: Sparkles
  }

  const getTaskTypeColor = (taskType: string) => {
    const colors = {
      coding: 'text-green-400',
      reasoning: 'text-blue-400',
      creative: 'text-pink-400',
      fast: 'text-yellow-400',
      multimodal: 'text-purple-400',
      general: 'text-cyan-400'
    }
    return colors[taskType as keyof typeof colors] || 'text-gray-400'
  }

  const handleNewChat = () => {
    // Logic to create new chat session
    console.log('Creating new chat session...')
  }

  const handleDeleteSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(session => session.id !== sessionId))
  }

  return (
    <>
      {/* Sidebar */}
      <div className={`
        fixed lg:relative inset-y-0 left-0 z-50 w-80 
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="h-full glass-morph border-r border-purple-500/20 flex flex-col">
          {/* Sidebar Header */}
          <div className="p-6 border-b border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Brain className="h-6 w-6 text-purple-400 animate-quantum-pulse" />
                <div>
                  <h2 className="text-lg font-semibold text-white">Navigation</h2>
                  <p className="text-xs text-purple-300">Quantum Intelligence Hub</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="lg:hidden p-2 rounded-lg hover:bg-purple-500/20 transition-all duration-200"
              >
                <X className="h-5 w-5 text-purple-300" />
              </button>
            </div>
          </div>

          {/* Navigation Menu */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 space-y-2">
              {navigationItems.map((item) => {
                const Icon = item.icon
                const isActive = activeView === item.id
                
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveSection(item.id)
                      onViewChange(item.id)
                    }}
                    className={`
                      w-full flex items-center space-x-3 p-3 rounded-lg transition-all duration-200
                      ${isActive 
                        ? 'bg-purple-500/20 border border-purple-500/30 text-white' 
                        : 'hover:bg-purple-500/10 text-purple-200'
                      }
                    `}
                  >
                    <Icon className={`h-5 w-5 ${isActive ? 'text-purple-300' : 'text-purple-400'}`} />
                    <div className="flex-1 text-left">
                      <p className="font-medium">{item.label}</p>
                      <p className="text-xs opacity-70">{item.description}</p>
                    </div>
                    {item.badge && (
                      <span className="bg-purple-500 text-white text-xs px-2 py-1 rounded-full">
                        {item.badge}
                      </span>
                    )}
                    <ChevronRight className={`h-4 w-4 transition-transform ${isActive ? 'rotate-90' : ''}`} />
                  </button>
                )
              })}
            </div>

            {/* Chat Sessions Section */}
            {activeView === 'chat' && (
              <div className="p-4 border-t border-purple-500/20">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-white">Recent Sessions</h3>
                  <button
                    onClick={handleNewChat}
                    className="p-2 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 transition-all duration-200"
                  >
                    <Plus className="h-4 w-4 text-purple-300" />
                  </button>
                </div>

                <div className="space-y-2">
                  {chatSessions.map((session) => {
                    const TaskIcon = taskTypeIcons[session.taskType as keyof typeof taskTypeIcons] || Sparkles
                    
                    return (
                      <div
                        key={session.id}
                        className="group p-3 rounded-lg glass-morph hover:bg-purple-500/10 transition-all duration-200 cursor-pointer"
                      >
                        <div className="flex items-start space-x-3">
                          <TaskIcon className={`h-4 w-4 mt-1 ${getTaskTypeColor(session.taskType)}`} />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white truncate">
                              {session.title}
                            </p>
                            <div className="flex items-center space-x-2 mt-1">
                              <span className="text-xs text-purple-300">{session.timestamp}</span>
                              <span className="text-xs text-gray-400">•</span>
                              <span className="text-xs text-gray-400">{session.messageCount} messages</span>
                            </div>
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteSession(session.id)
                            }}
                            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 transition-all duration-200"
                          >
                            <Trash2 className="h-3 w-3 text-red-400" />
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {chatSessions.length === 0 && (
                  <div className="text-center py-8">
                    <MessageCircle className="h-12 w-12 text-purple-400/50 mx-auto mb-3" />
                    <p className="text-purple-300 text-sm">No chat sessions yet</p>
                    <p className="text-purple-400 text-xs">Start a conversation to begin</p>
                  </div>
                )}
              </div>
            )}

            {/* Analytics Section */}
            {activeSection === 'analytics' && (
              <div className="p-4 border-t border-purple-500/20">
                <h3 className="text-sm font-semibold text-white mb-4">Neural Analytics</h3>
                <div className="space-y-3">
                  <div className="p-3 rounded-lg glass-morph">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-purple-300">Learning Progress</span>
                      <span className="text-sm font-medium text-white">68%</span>
                    </div>
                    <div className="w-full bg-slate-700/50 rounded-full h-2 mt-2">
                      <div className="bg-gradient-to-r from-purple-500 to-cyan-500 h-2 rounded-full" style={{ width: '68%' }} />
                    </div>
                  </div>
                  
                  <div className="p-3 rounded-lg glass-morph">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-purple-300">Neural Efficiency</span>
                      <span className="text-sm font-medium text-green-400">92%</span>
                    </div>
                    <div className="w-full bg-slate-700/50 rounded-full h-2 mt-2">
                      <div className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full" style={{ width: '92%' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-purple-500/20">
            <div className="text-center">
              <p className="text-xs text-purple-400">Phase 13: Multi-LLM Integration</p>
              <p className="text-xs text-gray-500 mt-1">Quantum Intelligence Engine v1.0</p>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default Sidebar
