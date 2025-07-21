'use client'

/**
 * Home Page - MasterX Quantum Intelligence Platform
 * 
 * Revolutionary Ultra-Premium Interface with Quantum Intelligence Engine
 * Billion-dollar caliber design with advanced visualizations and interactions
 */

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { QuantumLearningDashboard } from '@/components/quantum-enhanced/QuantumLearningDashboard'
import { QuantumChatInterface } from '@/components/quantum-enhanced/QuantumChatInterface'
import { sessionManager, SessionState } from '@/lib/session-manager'
import { Brain, Loader2, Zap } from 'lucide-react'

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

  // Initialize app
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
          await sessionManager.loadUserData()
        } else {
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
                <QuantumLearningDashboard />
              </div>
            )}
            {activeView === 'chat' && (
              <div className="h-full">
                <QuantumChatInterface />
              </div>
            )}
            {activeView === 'analytics' && (
              <div className="p-6">
                <div className="text-center text-gray-500">Analytics View - Coming Soon</div>
              </div>
            )}
            {activeView === 'learning' && (
              <div className="p-6">
                <div className="text-center text-gray-500">Learning Paths View - Coming Soon</div>
              </div>
            )}
            {activeView === 'goals' && (
              <div className="p-6">
                <div className="text-center text-gray-500">Learning Goals View - Coming Soon</div>
              </div>
            )}
            {activeView === 'settings' && (
              <div className="p-6">
                <div className="text-center text-gray-500">Settings View - Coming Soon</div>
              </div>
            )}
          </main>

          {/* Status Bar */}
          <div className="glass-morph border-t border-purple-500/20 px-6 py-3">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <Brain className="h-3 w-3 text-cyan-400" />
                  <span className="text-purple-300">CPU:</span>
                  <span className="text-white font-mono">{systemMetrics.cpuUsage}%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Zap className="h-3 w-3 text-purple-400" />
                  <span className="text-purple-300">Memory:</span>
                  <span className="text-white font-mono">{systemMetrics.memoryUsage}%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Brain className="h-3 w-3 text-green-400" />
                  <span className="text-purple-300">Active:</span>
                  <span className="text-white font-mono">{systemMetrics.activeUsers}</span>
                </div>
              </div>
              <div className="flex items-center space-x-2 px-2 py-1 rounded bg-purple-500/20">
                <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-purple-300">Quantum Online</span>
              </div>
            </div>
          </div>
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
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </div>

      <div className="relative z-10 text-center max-w-md mx-auto px-6">
        <div className="mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="relative">
              <Brain className="h-16 w-16 text-purple-400 animate-quantum-pulse" />
              <div className="absolute inset-0 h-16 w-16 border-2 border-purple-400/30 rounded-full animate-spin" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2 quantum-glow">
            MasterX
          </h1>
          <p className="text-purple-200 text-lg">
            Quantum Intelligence Platform
          </p>
        </div>

        <div className="mb-8">
          <div className="flex items-center justify-center mb-4">
            <Loader2 className="h-6 w-6 text-purple-400 mr-3 animate-spin" />
            <span className="text-white font-medium">
              {stages[stage]}
            </span>
          </div>

          <div className="w-full bg-slate-700/50 rounded-full h-2 mb-4 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full transition-all duration-300 ease-out relative"
              style={{ width: `${Math.min(progress, 100)}%` }}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse" />
            </div>
          </div>

          <div className="text-purple-300 text-sm font-mono">
            {Math.round(Math.min(progress, 100))}% Complete
          </div>
        </div>
      </div>
    </div>
  )
}

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
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-purple-900/30 via-transparent to-cyan-900/30" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

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
    </div>
  )
}