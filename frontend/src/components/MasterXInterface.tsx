'use client'

/**
 * MasterX Main Interface Component
 * 
 * Migrated from React App.js to Next.js component while preserving
 * all existing UI/UX, styling, and functionality exactly as-is.
 */

import { useEffect, useState } from "react"
import { useRouter } from 'next/navigation'
import axios from "axios"

// Preserve exact same backend URL logic
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8001"

interface SystemHealth {
  platform?: string
  version?: string
  [key: string]: any
}

export default function MasterXInterface() {
  // Next.js router for navigation
  const router = useRouter()
  
  // Keep exact same state management
  const [connectionStatus, setConnectionStatus] = useState("connecting")
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)

  // Keep exact same system health check logic
  const checkSystemHealth = async () => {
    try {
      setConnectionStatus("connecting")
      const response = await axios.get(`${BACKEND_URL}/health`)
      console.log("🚀 MasterX System Health:", response.data)
      setSystemHealth(response.data)
      setConnectionStatus("connected")
    } catch (e) {
      console.error("⚠️ MasterX System Connection Error:", e)
      setConnectionStatus("disconnected")
    }
  }

  // Keep exact same useEffect logic
  useEffect(() => {
    checkSystemHealth()
    // Check health every 30 seconds
    const interval = setInterval(checkSystemHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  // Keep exact same status color logic
  const getStatusColor = () => {
    switch (connectionStatus) {
      case "connected": return "#10b981"
      case "connecting": return "#f59e0b"
      default: return "#ef4444"
    }
  }

  // Next.js navigation (updated from React Router)
  const handleLaunchInterface = () => {
    router.push('/chat')
  }

  // Keep EXACT same JSX and styling - no changes
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Quantum Background Effects - Keep exact same */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-cyan-600/20"></div>
        {/* Animated particles - Keep exact same */}
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-pulse"
            style={{
              left: `${(i * 7 + 10) % 90 + 5}%`,
              top: `${(i * 11 + 15) % 80 + 10}%`,
              animationDelay: `${(i * 0.3) % 3}s`,
              animationDuration: `${2 + (i % 3)}s`
            }}
          ></div>
        ))}
      </div>

      {/* Connection Status - Keep exact same */}
      <div
        className="fixed top-6 right-6 px-4 py-2 rounded-full text-sm font-semibold text-white z-50 backdrop-blur-sm"
        style={{ backgroundColor: getStatusColor() }}
      >
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
          <span className="capitalize">{connectionStatus}</span>
        </div>
      </div>

      {/* Main Content - Keep exact same */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6">
        {/* MasterX Logo & Branding - Keep exact same */}
        <div className="text-center mb-12">
          <div className="mb-6">
            <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-r from-cyan-500 to-purple-600 mb-6 shadow-2xl">
              <div className="text-4xl font-bold text-white">MX</div>
            </div>
          </div>

          <h1 className="text-6xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            MasterX
          </h1>

          <h2 className="text-2xl text-gray-300 mb-2">
            Quantum Intelligence Platform
          </h2>

          <p className="text-gray-400 max-w-2xl mx-auto leading-relaxed">
            Advanced AI-powered learning and intelligence system with multi-LLM integration,
            quantum-inspired algorithms, and enterprise-grade performance.
          </p>
        </div>

        {/* System Status - Keep exact same */}
        {systemHealth && (
          <div className="bg-black/20 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/50 mb-8 max-w-md w-full">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <div className="w-3 h-3 rounded-full bg-green-400 mr-3 animate-pulse"></div>
              System Status
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-gray-300">
                <span>Platform:</span>
                <span className="text-cyan-400 font-medium">{systemHealth.platform || "MasterX"}</span>
              </div>
              <div className="flex justify-between text-gray-300">
                <span>Version:</span>
                <span className="text-purple-400 font-medium">{systemHealth.version || "1.0.0"}</span>
              </div>
              <div className="flex justify-between text-gray-300">
                <span>Status:</span>
                <span className="text-green-400 font-medium">Operational</span>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons - Updated navigation only */}
        <div className="flex flex-col sm:flex-row gap-4">
          <button
            onClick={handleLaunchInterface}
            className="px-8 py-4 bg-gradient-to-r from-cyan-600 to-purple-600 text-white font-semibold rounded-xl hover:from-cyan-500 hover:to-purple-500 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            Launch Intelligence Interface
          </button>

          <button
            onClick={checkSystemHealth}
            className="px-8 py-4 bg-gray-800/50 text-gray-300 font-semibold rounded-xl hover:bg-gray-700/50 transition-all duration-300 border border-gray-600/50 hover:border-gray-500/50"
          >
            Refresh System Status
          </button>
        </div>

        {/* Footer - Keep exact same */}
        <div className="mt-16 text-center text-gray-500 text-sm">
          <p>© 2025 MasterX Quantum Intelligence Platform</p>
          <p className="mt-1">Enterprise-Grade AI Solutions</p>
        </div>
      </div>
    </div>
  )
}