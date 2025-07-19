'use client'

import { useState, useEffect } from 'react'
import {
  Brain,
  Menu,
  Bell,
  Settings,
  User,
  Zap,
  Activity,
  Wifi,
  MessageSquare,
  BarChart3,
  Sparkles,
  TrendingUp,
  Users,
  Clock,
  Target,
  Cpu,
  Database,
  Globe,
  Shield,
  ChevronRight,
  Play,
  Pause,
  RotateCcw,
  Filter,
  Download,
  Share,
  Plus,
  ArrowUpRight,
  Layers,
  Network,
  Atom
} from 'lucide-react'
import { API, SystemMetrics, DashboardData, UserProfile } from '@/lib/api-services'
import { InteractiveChat } from '@/components/interactive-chat'
import { sessionManager, SessionState } from '@/lib/session-manager'

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [activeView, setActiveView] = useState('dashboard')
  const [systemMetrics, setSystemMetrics] = useState({
    cpuUsage: 45,
    memoryUsage: 62,
    networkActivity: 78,
    activeUsers: 1247,
    totalSessions: 8934,
    responseTime: 89
  })
  const [sessionState, setSessionState] = useState<SessionState>(sessionManager.getState())

  // Initialize session manager and data
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Subscribe to session state changes
        const unsubscribeSession = sessionManager.subscribe((newState) => {
          setSessionState(newState)

          // Update system metrics if available
          if (newState.systemMetrics) {
            setSystemMetrics({
              cpuUsage: newState.systemMetrics.cpu_usage,
              memoryUsage: newState.systemMetrics.memory_usage,
              networkActivity: newState.systemMetrics.network_activity,
              activeUsers: newState.systemMetrics.active_users,
              totalSessions: newState.systemMetrics.total_sessions,
              responseTime: newState.systemMetrics.response_time
            })
          }
        })

        // Check if already authenticated
        if (sessionState.isAuthenticated) {
          // Load user data
          await sessionManager.loadUserData()
        } else {
          // Skip auto-login for development - just continue with demo mode
          console.log('Running in demo mode without authentication')
        }

        return () => {
          unsubscribeSession()
        }
      } catch (error) {
        console.error('App initialization error:', error)
      } finally {
        setTimeout(() => setIsLoading(false), 2000)
      }
    }

    initializeApp()

    // Fallback: Update metrics with mock data if real-time fails
    const metricsInterval = setInterval(() => {
      if (!sessionState.systemMetrics) {
        setSystemMetrics(prev => ({
          cpuUsage: Math.max(20, Math.min(90, prev.cpuUsage + (Math.random() - 0.5) * 10)),
          memoryUsage: Math.max(30, Math.min(95, prev.memoryUsage + (Math.random() - 0.5) * 8)),
          networkActivity: Math.max(10, Math.min(100, prev.networkActivity + (Math.random() - 0.5) * 15)),
          activeUsers: prev.activeUsers + Math.floor((Math.random() - 0.5) * 20),
          totalSessions: prev.totalSessions + Math.floor(Math.random() * 5),
          responseTime: Math.max(50, Math.min(200, prev.responseTime + (Math.random() - 0.5) * 20))
        }))
      }
    }, 5000)

    return () => {
      clearInterval(metricsInterval)
    }
  }, [])

  // Get current user for display
  const user = sessionState.user || {
    name: 'Demo User',
    email: 'demo@masterx.ai',
    role: 'student'
  }

  if (isLoading) {
    return <LoadingScreen />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Quantum Background Effects */}
      <QuantumBackground />

      {/* Main Application Layout */}
      <div className="relative z-10 flex h-screen">
        {/* Sidebar */}
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          activeView={activeView}
          onViewChange={setActiveView}
          systemMetrics={systemMetrics}
          sessionState={sessionState}
        />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <Header user={user} onMenuClick={() => setSidebarOpen(true)} />

          {/* Main Content Area */}
          <main className="flex-1 overflow-auto">
            {activeView === 'dashboard' && (
              <div className="p-6 space-y-6">
                <DashboardContent systemMetrics={systemMetrics} />
              </div>
            )}
            {activeView === 'chat' && (
              <div className="h-full">
                <InteractiveChat />
              </div>
            )}
            {activeView === 'analytics' && (
              <div className="p-6">
                <AnalyticsView systemMetrics={systemMetrics} />
              </div>
            )}
            {activeView === 'users' && (
              <div className="p-6">
                <UsersView />
              </div>
            )}
            {activeView === 'sessions' && (
              <div className="p-6">
                <SessionsView />
              </div>
            )}
            {activeView === 'goals' && (
              <div className="p-6">
                <LearningGoalsView />
              </div>
            )}
            {(activeView === 'neural' || activeView === 'quantum') && (
              <div className="p-6">
                <SystemView viewType={activeView} systemMetrics={systemMetrics} />
              </div>
            )}
          </main>

          {/* Status Bar */}
          <StatusBar systemMetrics={systemMetrics} />
        </div>
      </div>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  )
}

// ============================================================================
// LOADING SCREEN COMPONENT
// ============================================================================

function LoadingScreen() {
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState(0)

  const stages = [
    'Initializing Quantum Intelligence Engine...',
    'Connecting to Neural Networks...',
    'Calibrating Multi-LLM Integration...',
    'Preparing Enterprise Dashboard...'
  ]

  useEffect(() => {
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev < 100) {
          return prev + Math.random() * 15
        }
        return 100
      })
    }, 150)

    const stageInterval = setInterval(() => {
      setStage(prev => (prev < stages.length - 1 ? prev + 1 : prev))
    }, 500)

    return () => {
      clearInterval(progressInterval)
      clearInterval(stageInterval)
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-3/4 left-3/4 w-64 h-64 bg-amber-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Neural Network Grid */}
      <div className="absolute inset-0 opacity-10">
        <div className="grid grid-cols-12 grid-rows-8 h-full w-full">
          {Array.from({ length: 96 }).map((_, i) => (
            <div
              key={i}
              className="border border-purple-500/20 animate-pulse"
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
      </div>

      {/* Main Loading Content */}
      <div className="relative z-10 text-center max-w-md mx-auto px-6">
        {/* Logo and Brand */}
        <div className="mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="relative">
              <Brain className="h-16 w-16 text-purple-400 animate-quantum-pulse" />
              <div className="absolute inset-0 h-16 w-16 border-2 border-purple-400/30 rounded-full animate-spin" />
              <div className="absolute inset-2 h-12 w-12 border border-cyan-400/30 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '3s' }} />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2 quantum-glow">
            MasterX
          </h1>
          <p className="text-purple-200 text-lg">
            Quantum Intelligence Platform
          </p>
        </div>

        {/* Loading Stage Indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-center mb-4">
            <Zap className="h-6 w-6 text-purple-400 mr-3 animate-spin" />
            <span className="text-white font-medium">
              {stages[stage]}
            </span>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-slate-700/50 rounded-full h-2 mb-4 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full transition-all duration-300 ease-out relative"
              style={{ width: `${Math.min(progress, 100)}%` }}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse" />
            </div>
          </div>

          {/* Progress Percentage */}
          <div className="text-purple-300 text-sm font-mono">
            {Math.round(Math.min(progress, 100))}% Complete
          </div>
        </div>

        {/* Phase Information */}
        <div className="mt-8 pt-6 border-t border-purple-500/20">
          <p className="text-gray-400 text-xs">
            Enterprise Quantum Intelligence Dashboard
          </p>
          <p className="text-gray-500 text-xs mt-1">
            Powered by Advanced Neural Networks
          </p>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// QUANTUM BACKGROUND COMPONENT
// ============================================================================

function QuantumBackground() {
  const [particles, setParticles] = useState<Array<{id: number, x: number, y: number, delay: number}>>([])

  useEffect(() => {
    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 5
    }))
    setParticles(newParticles)
  }, [])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Cosmic Background Gradients */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-purple-900/30 via-transparent to-cyan-900/30" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        <div className="absolute top-3/4 left-3/4 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }} />
      </div>

      {/* Neural Network Grid */}
      <div className="absolute inset-0 opacity-5">
        <div className="grid grid-cols-20 grid-rows-12 h-full w-full">
          {Array.from({ length: 240 }).map((_, i) => (
            <div
              key={i}
              className="border border-purple-400/20 animate-pulse"
              style={{
                animationDelay: `${(i * 0.05) % 3}s`,
                animationDuration: `${2 + (i % 3)}s`
              }}
            />
          ))}
        </div>
      </div>

      {/* Floating Quantum Particles */}
      <div className="absolute inset-0">
        {particles.map((particle) => (
          <div
            key={particle.id}
            className="absolute w-1 h-1 bg-purple-400/40 rounded-full animate-quantum-float"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              animationDelay: `${particle.delay}s`,
              animationDuration: `${4 + (particle.id % 3)}s`
            }}
          />
        ))}
      </div>

      {/* Data Stream Lines */}
      <div className="absolute inset-0">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="absolute h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent animate-data-stream"
            style={{
              top: `${10 + i * 12}%`,
              left: '0%',
              right: '0%',
              animationDelay: `${i * 0.5}s`,
              animationDuration: '6s'
            }}
          />
        ))}
      </div>

      {/* Neural Connections */}
      <svg className="absolute inset-0 w-full h-full opacity-10">
        <defs>
          <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.3" />
            <stop offset="50%" stopColor="#06B6D4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0.3" />
          </linearGradient>
        </defs>
        {Array.from({ length: 12 }).map((_, i) => (
          <line
            key={i}
            x1={`${Math.random() * 100}%`}
            y1={`${Math.random() * 100}%`}
            x2={`${Math.random() * 100}%`}
            y2={`${Math.random() * 100}%`}
            stroke="url(#neuralGradient)"
            strokeWidth="1"
            className="animate-pulse"
            style={{ animationDelay: `${i * 0.3}s` }}
          />
        ))}
      </svg>
    </div>
  )
}

// ============================================================================
// HEADER COMPONENT
// ============================================================================

interface HeaderProps {
  user: any
  onMenuClick: () => void
}

function Header({ user, onMenuClick }: HeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [notifications] = useState(3)
  const [systemStatus] = useState<'optimal' | 'warning' | 'error'>('optimal')

  const getStatusColor = () => {
    switch (systemStatus) {
      case 'optimal': return 'text-green-400'
      case 'warning': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      default: return 'text-green-400'
    }
  }

  return (
    <header className="glass-morph border-b border-purple-500/20 px-6 py-4 relative z-20">
      <div className="flex items-center justify-between">
        {/* Left Section - Logo and Menu */}
        <div className="flex items-center space-x-4">
          {/* Mobile Menu Button */}
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200"
          >
            <Menu className="h-5 w-5 text-purple-300" />
          </button>

          {/* Logo and Brand */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Brain className="h-8 w-8 text-purple-400 animate-quantum-pulse" />
              <div className="absolute inset-0 h-8 w-8 border border-purple-400/30 rounded-full animate-spin" style={{ animationDuration: '8s' }} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white quantum-glow">
                MasterX
              </h1>
              <p className="text-xs text-purple-300 font-mono">
                Quantum Intelligence
              </p>
            </div>
          </div>
        </div>

        {/* Center Section - System Status */}
        <div className="hidden md:flex items-center space-x-6">
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <Wifi className="h-4 w-4 text-green-400" />
            <span className="text-sm font-medium text-green-400">Connected</span>
          </div>

          {/* System Status */}
          <div className="flex items-center space-x-2">
            <Activity className={`h-4 w-4 ${getStatusColor()}`} />
            <span className={`text-sm font-medium ${getStatusColor()}`}>
              All Systems Optimal
            </span>
          </div>

          {/* Neural Activity Indicator */}
          <div className="flex items-center space-x-2">
            <Zap className="h-4 w-4 text-cyan-400 animate-pulse" />
            <span className="text-sm font-medium text-cyan-400">Neural Active</span>
          </div>
        </div>

        {/* Right Section - User Controls */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <button className="relative p-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200">
            <Bell className="h-5 w-5 text-purple-300" />
            {notifications > 0 && (
              <span className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center animate-pulse">
                {notifications}
              </span>
            )}
          </button>

          {/* Settings */}
          <button className="p-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200">
            <Settings className="h-5 w-5 text-purple-300" />
          </button>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-3 p-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200"
            >
              <div className="flex items-center space-x-2">
                <div className="h-8 w-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                  <User className="h-4 w-4 text-white" />
                </div>
                <div className="hidden sm:block text-left">
                  <p className="text-sm font-medium text-white">{user?.name}</p>
                  <p className="text-xs text-purple-300 capitalize">{user?.role}</p>
                </div>
              </div>
            </button>

            {/* User Dropdown Menu */}
            {showUserMenu && (
              <div className="absolute right-0 top-full mt-2 w-64 glass-morph border border-purple-500/30 rounded-lg shadow-xl z-50">
                <div className="p-4 border-b border-purple-500/20">
                  <div className="flex items-center space-x-3">
                    <div className="h-12 w-12 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                      <User className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <p className="font-medium text-white">{user?.name}</p>
                      <p className="text-sm text-purple-300">{user?.email}</p>
                      <p className="text-xs text-purple-400 capitalize">{user?.role}</p>
                    </div>
                  </div>
                </div>

                <div className="p-2">
                  <button className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-purple-500/20 transition-all duration-200 text-left">
                    <User className="h-4 w-4 text-purple-300" />
                    <span className="text-white">Profile Settings</span>
                  </button>

                  <button className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-purple-500/20 transition-all duration-200 text-left">
                    <Settings className="h-4 w-4 text-purple-300" />
                    <span className="text-white">Preferences</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Click outside to close user menu */}
      {showUserMenu && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowUserMenu(false)}
        />
      )}
    </header>
  )
}

// ============================================================================
// SIDEBAR COMPONENT
// ============================================================================

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  activeView: string
  onViewChange: (view: string) => void
  systemMetrics: any
  sessionState: SessionState
}

function Sidebar({ isOpen, onClose, activeView, onViewChange, systemMetrics, sessionState }: SidebarProps) {
  const [activeItem, setActiveItem] = useState(activeView)

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3, badge: null },
    { id: 'chat', label: 'AI Chat', icon: MessageSquare, badge: '2' },
    { id: 'analytics', label: 'Analytics', icon: TrendingUp, badge: null },
    { id: 'users', label: 'Users', icon: Users, badge: null },
    { id: 'sessions', label: 'Sessions', icon: Clock, badge: '12' },
    { id: 'goals', label: 'Learning Goals', icon: Target, badge: null },
    { id: 'neural', label: 'Neural Networks', icon: Network, badge: null },
    { id: 'quantum', label: 'Quantum Engine', icon: Atom, badge: null },
  ]

  const systemItems = [
    { id: 'database', label: 'Database', icon: Database, status: 'online' },
    { id: 'api', label: 'API Gateway', icon: Globe, status: 'online' },
    { id: 'security', label: 'Security', icon: Shield, status: 'secure' },
    { id: 'compute', label: 'Compute', icon: Cpu, status: 'optimal' },
  ]

  return (
    <>
      {/* Sidebar */}
      <div className={`
        fixed lg:relative inset-y-0 left-0 z-50 w-80
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="flex flex-col h-full glass-morph border-r border-purple-500/20">
          {/* Sidebar Header */}
          <div className="p-6 border-b border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <Layers className="h-6 w-6 text-purple-400" />
                  <div className="absolute inset-0 h-6 w-6 border border-purple-400/30 rounded animate-spin" style={{ animationDuration: '10s' }} />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-white">Control Center</h2>
                  <p className="text-xs text-purple-300">Quantum Intelligence</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="lg:hidden p-1 rounded-lg hover:bg-purple-500/20 transition-colors"
              >
                <ChevronRight className="h-5 w-5 text-purple-300" />
              </button>
            </div>
          </div>

          {/* Navigation Menu */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Main Navigation */}
            <div>
              <h3 className="text-xs font-semibold text-purple-300 uppercase tracking-wider mb-3">
                Navigation
              </h3>
              <nav className="space-y-1">
                {menuItems.map((item) => {
                  const Icon = item.icon
                  const isActive = activeItem === item.id

                  return (
                    <button
                      key={item.id}
                      onClick={() => {
                        setActiveItem(item.id)
                        onViewChange(item.id)
                        // Close sidebar on mobile after selection
                        if (window.innerWidth < 1024) {
                          onClose()
                        }
                      }}
                      className={`
                        w-full flex items-center justify-between p-3 rounded-lg
                        transition-all duration-200 group
                        ${isActive
                          ? 'bg-purple-500/20 border border-purple-500/30 text-white'
                          : 'hover:bg-purple-500/10 text-purple-200 hover:text-white'
                        }
                      `}
                    >
                      <div className="flex items-center space-x-3">
                        <Icon className={`h-5 w-5 ${isActive ? 'text-purple-400' : 'text-purple-300'}`} />
                        <span className="font-medium">{item.label}</span>
                      </div>
                      {item.badge && (
                        <span className="px-2 py-1 text-xs bg-purple-500 text-white rounded-full">
                          {item.badge}
                        </span>
                      )}
                    </button>
                  )
                })}
              </nav>
            </div>

            {/* System Status */}
            <div>
              <h3 className="text-xs font-semibold text-purple-300 uppercase tracking-wider mb-3">
                System Status
              </h3>
              <div className="space-y-2">
                {systemItems.map((item) => {
                  const Icon = item.icon
                  const getStatusColor = () => {
                    switch (item.status) {
                      case 'online': return 'text-green-400'
                      case 'secure': return 'text-blue-400'
                      case 'optimal': return 'text-cyan-400'
                      default: return 'text-gray-400'
                    }
                  }

                  return (
                    <div
                      key={item.id}
                      className="flex items-center justify-between p-2 rounded-lg hover:bg-purple-500/10 transition-colors"
                    >
                      <div className="flex items-center space-x-3">
                        <Icon className="h-4 w-4 text-purple-300" />
                        <span className="text-sm text-purple-200">{item.label}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className={`w-2 h-2 rounded-full ${getStatusColor().replace('text-', 'bg-')} animate-pulse`} />
                        <span className={`text-xs font-medium ${getStatusColor()}`}>
                          {item.status}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Quick Actions */}
            <div>
              <h3 className="text-xs font-semibold text-purple-300 uppercase tracking-wider mb-3">
                Quick Actions
              </h3>
              <div className="space-y-2">
                <button
                  onClick={async () => {
                    try {
                      await sessionManager.createChatSession()
                      onViewChange('chat')
                      if (window.innerWidth < 1024) onClose()
                    } catch (error) {
                      console.error('Failed to create new session:', error)
                    }
                  }}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border border-purple-500/30 hover:from-purple-500/30 hover:to-cyan-500/30 transition-all duration-200 group"
                >
                  <Plus className="h-4 w-4 text-purple-300 group-hover:text-white" />
                  <span className="text-sm font-medium text-purple-200 group-hover:text-white">New Session</span>
                </button>

                <button
                  onClick={() => {
                    // Export dashboard data as JSON
                    const data = {
                      systemMetrics,
                      userInfo: sessionState.user,
                      chatSessions: sessionState.chatSessions,
                      learningGoals: sessionState.learningGoals,
                      exportDate: new Date().toISOString()
                    }
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = `masterx-data-${new Date().toISOString().split('T')[0]}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-purple-500/10 transition-colors group"
                >
                  <Download className="h-4 w-4 text-purple-300 group-hover:text-white" />
                  <span className="text-sm font-medium text-purple-200 group-hover:text-white">Export Data</span>
                </button>

                <button
                  onClick={() => {
                    // Share current dashboard URL
                    if (navigator.share) {
                      navigator.share({
                        title: 'MasterX Quantum Intelligence Platform',
                        text: 'Check out my progress on MasterX AI!',
                        url: window.location.href
                      })
                    } else {
                      // Fallback: copy to clipboard
                      navigator.clipboard.writeText(window.location.href)
                      alert('Dashboard URL copied to clipboard!')
                    }
                  }}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-purple-500/10 transition-colors group"
                >
                  <Share className="h-4 w-4 text-purple-300 group-hover:text-white" />
                  <span className="text-sm font-medium text-purple-200 group-hover:text-white">Share Report</span>
                </button>
              </div>
            </div>
          </div>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-purple-500/20">
            <div className="flex items-center space-x-3 p-3 rounded-lg bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/20">
              <div className="h-10 w-10 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-white">Quantum Mode</p>
                <p className="text-xs text-purple-300">Enhanced Processing</p>
              </div>
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

// ============================================================================
// DASHBOARD CONTENT COMPONENT
// ============================================================================

interface DashboardContentProps {
  systemMetrics: any
}

function DashboardContent({ systemMetrics }: DashboardContentProps) {
  const [activeTimeRange, setActiveTimeRange] = useState('24h')
  const [isPlaying, setIsPlaying] = useState(true)

  const timeRanges = [
    { id: '1h', label: '1H' },
    { id: '24h', label: '24H' },
    { id: '7d', label: '7D' },
    { id: '30d', label: '30D' }
  ]

  const metricCards = [
    {
      title: 'AI Conversations',
      value: '2,847',
      change: '+24.3%',
      trend: 'up',
      icon: MessageSquare,
      color: 'from-purple-500 to-purple-600'
    },
    {
      title: 'Learning Sessions',
      value: '1,293',
      change: '+18.7%',
      trend: 'up',
      icon: Target,
      color: 'from-cyan-500 to-cyan-600'
    },
    {
      title: 'AI Response Time',
      value: `${systemMetrics.responseTime}ms`,
      change: '-12.4%',
      trend: 'down',
      icon: Zap,
      color: 'from-green-500 to-green-600'
    },
    {
      title: 'Model Accuracy',
      value: '96.8%',
      change: '+3.2%',
      trend: 'up',
      icon: Brain,
      color: 'from-amber-500 to-amber-600'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white quantum-glow">
            Quantum Intelligence Dashboard
          </h1>
          <p className="text-purple-300 mt-1">
            Real-time monitoring and analytics for your AI platform
          </p>
        </div>

        <div className="flex items-center space-x-4">
          {/* Time Range Selector */}
          <div className="flex items-center space-x-1 glass-morph p-1 rounded-lg">
            {timeRanges.map((range) => (
              <button
                key={range.id}
                onClick={() => setActiveTimeRange(range.id)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-all duration-200 ${
                  activeTimeRange === range.id
                    ? 'bg-purple-500 text-white'
                    : 'text-purple-300 hover:text-white hover:bg-purple-500/20'
                }`}
              >
                {range.label}
              </button>
            ))}
          </div>

          {/* Control Buttons */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200"
            >
              {isPlaying ? (
                <Pause className="h-4 w-4 text-purple-300" />
              ) : (
                <Play className="h-4 w-4 text-purple-300" />
              )}
            </button>

            <button className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200">
              <RotateCcw className="h-4 w-4 text-purple-300" />
            </button>

            <button className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200">
              <Filter className="h-4 w-4 text-purple-300" />
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metricCards.map((metric) => {
          const Icon = metric.icon
          return (
            <div
              key={metric.title}
              className="glass-morph p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300 group"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg bg-gradient-to-r ${metric.color} bg-opacity-20`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <div className={`flex items-center space-x-1 text-sm font-medium ${
                  metric.trend === 'up' ? 'text-green-400' : 'text-red-400'
                }`}>
                  <ArrowUpRight className={`h-4 w-4 ${metric.trend === 'down' ? 'rotate-180' : ''}`} />
                  <span>{metric.change}</span>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-white mb-1 group-hover:text-purple-300 transition-colors">
                  {metric.value}
                </h3>
                <p className="text-purple-300 text-sm">
                  {metric.title}
                </p>
              </div>
            </div>
          )
        })}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* AI Model Performance Chart */}
        <div className="lg:col-span-2 glass-morph p-6 rounded-xl border border-purple-500/20">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">AI Model Performance</h3>
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-purple-500 rounded-full" />
                <span className="text-sm text-purple-300">Groq</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-cyan-500 rounded-full" />
                <span className="text-sm text-purple-300">Gemini</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-sm text-purple-300">Auto-Select</span>
              </div>
            </div>
          </div>

          {/* AI Performance Metrics */}
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-purple-300">Groq Performance</span>
                <span className="text-sm font-medium text-white">94.2%</span>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-2">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-purple-400 rounded-full transition-all duration-500 relative"
                  style={{ width: '94.2%' }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse rounded-full" />
                </div>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-purple-300">Gemini Performance</span>
                <span className="text-sm font-medium text-white">96.8%</span>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-2">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full transition-all duration-500 relative"
                  style={{ width: '96.8%' }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse rounded-full" />
                </div>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-purple-300">Auto-Selection Accuracy</span>
                <span className="text-sm font-medium text-white">98.1%</span>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-2">
                <div
                  className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500 relative"
                  style={{ width: '98.1%' }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse rounded-full" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* AI Model Status */}
        <div className="glass-morph p-6 rounded-xl border border-purple-500/20">
          <h3 className="text-xl font-semibold text-white mb-6">AI Models</h3>

          <div className="space-y-4">
            {[
              { name: 'Groq Llama 3.3 70B', status: 'Online', load: 94, color: 'from-orange-500 to-orange-400' },
              { name: 'Gemini 2.0 Flash', status: 'Online', load: 97, color: 'from-blue-500 to-blue-400' },
              { name: 'DeepSeek R1 Distill', status: 'Online', load: 89, color: 'from-purple-500 to-purple-400' },
              { name: 'Auto-Selection Engine', status: 'Active', load: 98, color: 'from-green-500 to-green-400' }
            ].map((model) => (
              <div key={model.name} className="p-4 rounded-lg bg-slate-800/30 border border-purple-500/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-white">{model.name}</span>
                  <span className="text-xs text-green-400">{model.status}</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex-1 bg-slate-700/50 rounded-full h-1.5">
                    <div
                      className={`h-full bg-gradient-to-r ${model.color} rounded-full transition-all duration-500`}
                      style={{ width: `${model.load}%` }}
                    />
                  </div>
                  <span className="text-xs text-purple-300 font-mono">{model.load}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// STATUS BAR COMPONENT
// ============================================================================

interface StatusBarProps {
  systemMetrics: any
}

function StatusBar({ systemMetrics }: StatusBarProps) {
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  return (
    <footer className="glass-morph border-t border-purple-500/20 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left Section - System Info */}
        <div className="flex items-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-green-400 font-medium">System Online</span>
          </div>

          <div className="flex items-center space-x-2">
            <Cpu className="h-4 w-4 text-purple-300" />
            <span className="text-purple-300">CPU: {systemMetrics.cpuUsage}%</span>
          </div>

          <div className="flex items-center space-x-2">
            <Database className="h-4 w-4 text-cyan-300" />
            <span className="text-cyan-300">Memory: {systemMetrics.memoryUsage}%</span>
          </div>

          <div className="flex items-center space-x-2">
            <Activity className="h-4 w-4 text-amber-300" />
            <span className="text-amber-300">Response: {systemMetrics.responseTime}ms</span>
          </div>
        </div>

        {/* Center Section - Active Users */}
        <div className="hidden md:flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <Users className="h-4 w-4 text-purple-300" />
            <span className="text-purple-300">
              {systemMetrics.activeUsers.toLocaleString()} Active Users
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <Globe className="h-4 w-4 text-cyan-300" />
            <span className="text-cyan-300">Global Network</span>
          </div>
        </div>

        {/* Right Section - Time and Version */}
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <Clock className="h-4 w-4 text-purple-300" />
            <span className="text-purple-300 font-mono">
              {currentTime.toLocaleTimeString()}
            </span>
          </div>

          <div className="text-purple-400 font-mono text-xs">
            v2.1.0-quantum
          </div>
        </div>
      </div>
    </footer>
  )
}

// ============================================================================
// ADDITIONAL VIEW COMPONENTS
// ============================================================================

function AnalyticsView({ systemMetrics }: { systemMetrics: any }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white quantum-glow">Analytics Dashboard</h1>
        <div className="flex items-center space-x-4">
          <button className="px-4 py-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200">
            <Download className="h-4 w-4 text-purple-300 mr-2 inline" />
            Export Report
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-morph p-6 rounded-xl border border-purple-500/20">
          <h3 className="text-xl font-semibold text-white mb-4">Performance Trends</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-purple-300">CPU Efficiency</span>
              <span className="text-green-400 font-semibold">+12.5%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Response Time</span>
              <span className="text-green-400 font-semibold">-8.3%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">User Satisfaction</span>
              <span className="text-green-400 font-semibold">+15.7%</span>
            </div>
          </div>
        </div>

        <div className="glass-morph p-6 rounded-xl border border-purple-500/20">
          <h3 className="text-xl font-semibold text-white mb-4">AI Model Performance</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Groq (Fast)</span>
              <span className="text-cyan-400 font-semibold">94.2%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Gemini (Reasoning)</span>
              <span className="text-cyan-400 font-semibold">96.8%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Auto-Selection</span>
              <span className="text-green-400 font-semibold">98.1%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function UsersView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white quantum-glow">User Management</h1>
        <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg hover:from-purple-600 hover:to-cyan-600 transition-all duration-200">
          <Plus className="h-4 w-4 text-white mr-2 inline" />
          Add User
        </button>
      </div>

      <div className="glass-morph p-6 rounded-xl border border-purple-500/20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-white mb-2">1,247</div>
            <div className="text-purple-300">Total Users</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">892</div>
            <div className="text-purple-300">Active Users</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-cyan-400 mb-2">45</div>
            <div className="text-purple-300">New Today</div>
          </div>
        </div>

        <div className="space-y-4">
          {[
            { name: 'Dr. Sarah Chen', email: 'sarah.chen@masterx.ai', role: 'Researcher', status: 'Active' },
            { name: 'Alex Rodriguez', email: 'alex.r@masterx.ai', role: 'Student', status: 'Active' },
            { name: 'Prof. Michael Kim', email: 'michael.kim@masterx.ai', role: 'Teacher', status: 'Active' }
          ].map((user, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-slate-800/30 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className="h-10 w-10 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                  <User className="h-5 w-5 text-white" />
                </div>
                <div>
                  <div className="font-medium text-white">{user.name}</div>
                  <div className="text-sm text-purple-300">{user.email}</div>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm">{user.role}</span>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm">{user.status}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function SessionsView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white quantum-glow">Learning Sessions</h1>
        <div className="flex items-center space-x-4">
          <button className="px-4 py-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200">
            <Filter className="h-4 w-4 text-purple-300 mr-2 inline" />
            Filter
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {[
          { id: 'session_001', user: 'Sarah Chen', duration: '45 min', topic: 'Quantum Computing', status: 'Completed' },
          { id: 'session_002', user: 'Alex Rodriguez', duration: '32 min', topic: 'Machine Learning', status: 'Active' },
          { id: 'session_003', user: 'Michael Kim', duration: '28 min', topic: 'Data Science', status: 'Completed' }
        ].map((session, index) => (
          <div key={index} className="glass-morph p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-purple-300 font-mono">{session.id}</span>
              <span className={`px-2 py-1 rounded-full text-xs ${
                session.status === 'Active'
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-blue-500/20 text-blue-400'
              }`}>
                {session.status}
              </span>
            </div>

            <h3 className="text-lg font-semibold text-white mb-2">{session.topic}</h3>
            <p className="text-purple-300 mb-4">{session.user}</p>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-purple-400" />
                <span className="text-sm text-purple-300">{session.duration}</span>
              </div>
              <button className="text-cyan-400 hover:text-cyan-300 transition-colors">
                View Details
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function LearningGoalsView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white quantum-glow">Learning Goals</h1>
        <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg hover:from-purple-600 hover:to-cyan-600 transition-all duration-200">
          <Target className="h-4 w-4 text-white mr-2 inline" />
          New Goal
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[
          { title: 'Master Quantum Computing', progress: 75, deadline: '2024-08-15', status: 'On Track' },
          { title: 'Advanced Machine Learning', progress: 45, deadline: '2024-09-01', status: 'In Progress' },
          { title: 'Data Science Fundamentals', progress: 90, deadline: '2024-07-30', status: 'Almost Complete' }
        ].map((goal, index) => (
          <div key={index} className="glass-morph p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">{goal.title}</h3>
              <span className={`px-3 py-1 rounded-full text-xs ${
                goal.status === 'On Track'
                  ? 'bg-green-500/20 text-green-400'
                  : goal.status === 'Almost Complete'
                    ? 'bg-cyan-500/20 text-cyan-400'
                    : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {goal.status}
              </span>
            </div>

            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-purple-300">Progress</span>
                <span className="text-sm font-medium text-white">{goal.progress}%</span>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-2">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full transition-all duration-500"
                  style={{ width: `${goal.progress}%` }}
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-purple-400" />
                <span className="text-sm text-purple-300">Due: {goal.deadline}</span>
              </div>
              <button className="text-cyan-400 hover:text-cyan-300 transition-colors">
                Edit Goal
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function SystemView({ viewType, systemMetrics }: { viewType: string, systemMetrics: any }) {
  const isNeural = viewType === 'neural'

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white quantum-glow">
          {isNeural ? 'Neural Networks' : 'Quantum Engine'}
        </h1>
        <div className="flex items-center space-x-4">
          <button className="px-4 py-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200">
            <Settings className="h-4 w-4 text-purple-300 mr-2 inline" />
            Configure
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-morph p-6 rounded-xl border border-purple-500/20">
          <h3 className="text-xl font-semibold text-white mb-6">
            {isNeural ? 'Neural Network Status' : 'Quantum Processing Units'}
          </h3>

          <div className="space-y-4">
            {[
              { name: isNeural ? 'Primary Neural Core' : 'Quantum Core Alpha', status: 'Optimal', load: 87, temp: '42°C' },
              { name: isNeural ? 'Learning Engine' : 'Quantum Core Beta', status: 'Active', load: 92, temp: '38°C' },
              { name: isNeural ? 'Memory Matrix' : 'Quantum Entanglement', status: 'Stable', load: 76, temp: '35°C' },
              { name: isNeural ? 'Pattern Recognition' : 'Superposition Engine', status: 'Training', load: 68, temp: '41°C' }
            ].map((unit, index) => (
              <div key={index} className="p-4 rounded-lg bg-slate-800/30 border border-purple-500/10">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-white">{unit.name}</span>
                  <div className="flex items-center space-x-4">
                    <span className="text-xs text-green-400">{unit.status}</span>
                    <span className="text-xs text-purple-300">{unit.temp}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full transition-all duration-500"
                      style={{ width: `${unit.load}%` }}
                    />
                  </div>
                  <span className="text-xs text-purple-300 font-mono w-12">{unit.load}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-morph p-6 rounded-xl border border-purple-500/20">
          <h3 className="text-xl font-semibold text-white mb-6">System Health</h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Overall Status</span>
              <span className="text-green-400 font-semibold">Excellent</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Uptime</span>
              <span className="text-cyan-400 font-semibold">99.97%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Efficiency</span>
              <span className="text-green-400 font-semibold">94.2%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-purple-300">Error Rate</span>
              <span className="text-green-400 font-semibold">0.03%</span>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-purple-500/20">
            <button className="w-full px-4 py-2 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg hover:from-purple-600 hover:to-cyan-600 transition-all duration-200">
              Run Diagnostics
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}