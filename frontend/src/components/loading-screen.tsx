/**
 * Loading Screen Component for MasterX Quantum Intelligence Platform
 * 
 * Sophisticated loading experience with quantum-themed animations
 * and progressive loading indicators.
 */

'use client'

import { useEffect, useState } from 'react'
import { Brain, Zap, Sparkles, Loader2 } from 'lucide-react'

export function LoadingScreen() {
  const [loadingStage, setLoadingStage] = useState(0)
  const [progress, setProgress] = useState(0)
  const [isClient, setIsClient] = useState(false)

  const loadingStages = [
    { text: 'Initializing Quantum Intelligence Engine...', icon: Brain },
    { text: 'Connecting to Neural Networks...', icon: Zap },
    { text: 'Calibrating Multi-LLM Integration...', icon: Sparkles },
    { text: 'Preparing Adaptive Learning Systems...', icon: Loader2 },
  ]

  // Client-side detection to prevent hydration mismatch
  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    const stageInterval = setInterval(() => {
      setLoadingStage(prev => {
        if (prev < loadingStages.length - 1) {
          return prev + 1
        }
        return prev
      })
    }, 1000)

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev < 100) {
          return prev + Math.random() * 15
        }
        return 100
      })
    }, 200)

    return () => {
      clearInterval(stageInterval)
      clearInterval(progressInterval)
    }
  }, [])

  const CurrentIcon = loadingStages[loadingStage]?.icon || Brain

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
            <CurrentIcon className="h-6 w-6 text-purple-400 mr-3 animate-spin" />
            <span className="text-white font-medium">
              {loadingStages[loadingStage]?.text || 'Loading...'}
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

        {/* Loading Stages List */}
        <div className="space-y-2">
          {loadingStages.map((stage, index) => {
            const Icon = stage.icon
            const isActive = index === loadingStage
            const isComplete = index < loadingStage
            
            return (
              <div
                key={index}
                className={`flex items-center text-sm transition-all duration-300 ${
                  isActive 
                    ? 'text-purple-300 scale-105' 
                    : isComplete 
                      ? 'text-green-400' 
                      : 'text-gray-500'
                }`}
              >
                <Icon className={`h-4 w-4 mr-2 ${
                  isActive ? 'animate-spin' : isComplete ? 'animate-pulse' : ''
                }`} />
                <span>{stage.text}</span>
                {isComplete && (
                  <span className="ml-auto text-green-400">✓</span>
                )}
              </div>
            )
          })}
        </div>

        {/* Phase Information */}
        <div className="mt-8 pt-6 border-t border-purple-500/20">
          <p className="text-gray-400 text-xs">
            Phase 13: Frontend Integration & Multi-LLM Enhancement
          </p>
          <p className="text-gray-500 text-xs mt-1">
            Powered by Quantum Intelligence Engine
          </p>
        </div>
      </div>

      {/* Floating Particles - Client-side only to prevent hydration mismatch */}
      {isClient && (
        <div className="absolute inset-0 pointer-events-none">
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-purple-400/30 rounded-full animate-pulse"
              style={{
                left: `${(i * 7 + 10) % 90 + 5}%`,
                top: `${(i * 11 + 15) % 80 + 10}%`,
                animationDelay: `${(i * 0.3) % 3}s`,
                animationDuration: `${2 + (i % 3)}s`
              }}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default LoadingScreen
