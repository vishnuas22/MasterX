'use client'

import { useState } from 'react'
import { Brain, BarChart3, MessageSquare, Users, Target, Settings, Menu, X, Home, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  activeView: string
  onViewChange: (view: string) => void
  systemMetrics: any
  sessionState: any
}

export function Sidebar({ isOpen, onClose, activeView, onViewChange, systemMetrics, sessionState }: SidebarProps) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home, color: 'purple' },
    { id: 'chat', label: 'AI Chat', icon: MessageSquare, color: 'cyan' },
    { id: 'analytics', label: 'Analytics', icon: BarChart3, color: 'emerald' },
    { id: 'learning', label: 'Learning Paths', icon: Target, color: 'amber' },
    { id: 'goals', label: 'Learning Goals', icon: Target, color: 'pink' },
    { id: 'settings', label: 'Settings', icon: Settings, color: 'gray' },
  ]

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div className={cn(
        "fixed left-0 top-0 h-full w-80 glass-morph-premium border-r border-purple-500/30 transform transition-transform duration-300 z-50",
        isOpen ? "translate-x-0" : "-translate-x-full",
        "lg:translate-x-0 lg:static lg:z-auto"
      )}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-6 border-b border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 flex items-center justify-center animate-quantum-pulse">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="intelligence-title text-white quantum-glow">MasterX</h1>
                  <p className="data-micro text-purple-400">Quantum Intelligence</p>
                </div>
              </div>
              <button 
                onClick={onClose}
                className="lg:hidden p-2 rounded-lg hover:bg-purple-500/20 transition-colors"
              >
                <X className="h-5 w-5 text-gray-400" />
              </button>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2">
            {menuItems.map((item) => {
              const isActive = activeView === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    onViewChange(item.id)
                    onClose()
                  }}
                  className={cn(
                    "w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300",
                    isActive 
                      ? "bg-purple-500/20 border border-purple-500/30 text-white" 
                      : "hover:bg-purple-500/10 text-gray-300 hover:text-white"
                  )}
                >
                  <item.icon className={cn(
                    "h-5 w-5",
                    isActive ? "text-purple-400 animate-quantum-pulse" : "text-gray-400"
                  )} />
                  <span className="precision-small font-medium">{item.label}</span>
                  {isActive && (
                    <div className="ml-auto w-2 h-2 bg-purple-400 rounded-full animate-quantum-pulse" />
                  )}
                </button>
              )
            })}
          </nav>

          {/* System Status */}
          <div className="p-4 border-t border-purple-500/20">
            <div className="space-y-3">
              <h3 className="precision-small text-gray-400 mb-2">System Status</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="data-micro text-gray-500">CPU</span>
                  <span className="data-micro text-purple-300 font-mono">{systemMetrics.cpuUsage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="data-micro text-gray-500">Memory</span>
                  <span className="data-micro text-cyan-300 font-mono">{systemMetrics.memoryUsage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="data-micro text-gray-500">Active Users</span>
                  <span className="data-micro text-emerald-300 font-mono">{systemMetrics.activeUsers}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}