/**
 * Status Bar Component for MasterX Quantum Intelligence Platform
 * 
 * Real-time status monitoring with system metrics,
 * LLM provider status, and performance indicators.
 */

'use client'

import { useState, useEffect } from 'react'
import { 
  Activity, 
  Zap, 
  Brain, 
  Wifi, 
  Server, 
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle
} from 'lucide-react'

interface SystemMetrics {
  responseTime: number
  tokensProcessed: number
  activeConnections: number
  uptime: string
  memoryUsage: number
  cpuUsage: number
}

interface ProviderStatus {
  name: string
  status: 'online' | 'degraded' | 'offline'
  responseTime: number
  requests: number
}

export function StatusBar() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    responseTime: 1.2,
    tokensProcessed: 15420,
    activeConnections: 847,
    uptime: '99.9%',
    memoryUsage: 68,
    cpuUsage: 34
  })

  const [providers, setProviders] = useState<ProviderStatus[]>([
    { name: 'Groq', status: 'online', responseTime: 0.8, requests: 1240 },
    { name: 'Gemini', status: 'online', responseTime: 1.1, requests: 890 },
    { name: 'OpenAI', status: 'degraded', responseTime: 2.3, requests: 450 },
    { name: 'Claude', status: 'offline', responseTime: 0, requests: 0 }
  ])

  const [currentTime, setCurrentTime] = useState<Date | null>(null)

  // Update current time - initialize on client to prevent hydration mismatch
  useEffect(() => {
    setCurrentTime(new Date())

    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  // Simulate real-time metrics updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        responseTime: Math.max(0.5, prev.responseTime + (Math.random() - 0.5) * 0.3),
        tokensProcessed: prev.tokensProcessed + Math.floor(Math.random() * 50),
        activeConnections: Math.max(800, prev.activeConnections + Math.floor((Math.random() - 0.5) * 20)),
        memoryUsage: Math.max(50, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 5)),
        cpuUsage: Math.max(20, Math.min(80, prev.cpuUsage + (Math.random() - 0.5) * 10))
      }))

      setProviders(prev => prev.map(provider => ({
        ...provider,
        responseTime: provider.status === 'online' 
          ? Math.max(0.3, provider.responseTime + (Math.random() - 0.5) * 0.2)
          : provider.responseTime,
        requests: provider.status === 'online' 
          ? provider.requests + Math.floor(Math.random() * 10)
          : provider.requests
      })))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-400'
      case 'degraded': return 'text-yellow-400'
      case 'offline': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return CheckCircle
      case 'degraded': return AlertTriangle
      case 'offline': return AlertTriangle
      default: return Activity
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}k`
    }
    return num.toString()
  }

  return (
    <div className="glass-morph border-t border-purple-500/20 px-6 py-3">
      <div className="flex items-center justify-between text-xs">
        {/* Left Section - System Metrics */}
        <div className="flex items-center space-x-6">
          {/* Response Time */}
          <div className="flex items-center space-x-2">
            <Zap className="h-3 w-3 text-cyan-400" />
            <span className="text-purple-300">Response:</span>
            <span className="text-white font-mono">
              {metrics.responseTime.toFixed(1)}s
            </span>
          </div>

          {/* Tokens Processed */}
          <div className="flex items-center space-x-2">
            <Brain className="h-3 w-3 text-purple-400" />
            <span className="text-purple-300">Tokens:</span>
            <span className="text-white font-mono">
              {formatNumber(metrics.tokensProcessed)}
            </span>
          </div>

          {/* Active Connections */}
          <div className="flex items-center space-x-2">
            <Wifi className="h-3 w-3 text-green-400" />
            <span className="text-purple-300">Active:</span>
            <span className="text-white font-mono">
              {formatNumber(metrics.activeConnections)}
            </span>
          </div>

          {/* System Uptime */}
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-3 w-3 text-green-400" />
            <span className="text-purple-300">Uptime:</span>
            <span className="text-green-400 font-mono">
              {metrics.uptime}
            </span>
          </div>
        </div>

        {/* Center Section - LLM Provider Status */}
        <div className="hidden lg:flex items-center space-x-4">
          <span className="text-purple-300">Providers:</span>
          {providers.map((provider) => {
            const StatusIcon = getStatusIcon(provider.status)
            return (
              <div key={provider.name} className="flex items-center space-x-1">
                <StatusIcon className={`h-3 w-3 ${getStatusColor(provider.status)}`} />
                <span className="text-white">{provider.name}</span>
                {provider.status === 'online' && (
                  <span className="text-gray-400 font-mono">
                    ({provider.responseTime.toFixed(1)}s)
                  </span>
                )}
              </div>
            )
          })}
        </div>

        {/* Right Section - System Resources & Time */}
        <div className="flex items-center space-x-6">
          {/* Memory Usage */}
          <div className="hidden md:flex items-center space-x-2">
            <Server className="h-3 w-3 text-blue-400" />
            <span className="text-purple-300">Memory:</span>
            <div className="flex items-center space-x-1">
              <div className="w-12 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-300 ${
                    metrics.memoryUsage > 80 ? 'bg-red-400' : 
                    metrics.memoryUsage > 60 ? 'bg-yellow-400' : 'bg-green-400'
                  }`}
                  style={{ width: `${metrics.memoryUsage}%` }}
                />
              </div>
              <span className="text-white font-mono text-xs">
                {metrics.memoryUsage.toFixed(0)}%
              </span>
            </div>
          </div>

          {/* CPU Usage */}
          <div className="hidden md:flex items-center space-x-2">
            <Activity className="h-3 w-3 text-orange-400" />
            <span className="text-purple-300">CPU:</span>
            <div className="flex items-center space-x-1">
              <div className="w-12 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-300 ${
                    metrics.cpuUsage > 70 ? 'bg-red-400' : 
                    metrics.cpuUsage > 50 ? 'bg-yellow-400' : 'bg-green-400'
                  }`}
                  style={{ width: `${metrics.cpuUsage}%` }}
                />
              </div>
              <span className="text-white font-mono text-xs">
                {metrics.cpuUsage.toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Current Time */}
          <div className="flex items-center space-x-2">
            <Clock className="h-3 w-3 text-purple-400" />
            <span className="text-white font-mono">
              {currentTime ? currentTime.toLocaleTimeString() : '--:--:--'}
            </span>
          </div>

          {/* Phase Indicator */}
          <div className="flex items-center space-x-2 px-2 py-1 rounded bg-purple-500/20">
            <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-purple-300">Phase 13</span>
          </div>
        </div>
      </div>

      {/* Mobile Simplified View */}
      <div className="lg:hidden mt-2 pt-2 border-t border-purple-500/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <Zap className="h-3 w-3 text-cyan-400" />
              <span className="text-white font-mono">{metrics.responseTime.toFixed(1)}s</span>
            </div>
            <div className="flex items-center space-x-1">
              <Brain className="h-3 w-3 text-purple-400" />
              <span className="text-white font-mono">{formatNumber(metrics.tokensProcessed)}</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-purple-300">
              {providers.filter(p => p.status === 'online').length}/{providers.length} Online
            </span>
            <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatusBar
