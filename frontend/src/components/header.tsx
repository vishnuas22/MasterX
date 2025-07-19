/**
 * Header Component for MasterX Quantum Intelligence Platform
 * 
 * Sophisticated header with user management, system status,
 * and quantum-themed design elements.
 */

'use client'

import { useState, useEffect } from 'react'
import { 
  Brain, 
  Menu, 
  Bell, 
  Settings, 
  User, 
  LogOut, 
  Zap, 
  Activity,
  Wifi,
  WifiOff
} from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

interface HeaderProps {
  user: any
  onMenuClick: () => void
}

export function Header({ user, onMenuClick }: HeaderProps) {
  const { logout } = useAuth()
  const [isOnline, setIsOnline] = useState(true)
  const [notifications, setNotifications] = useState(3)
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [systemStatus, setSystemStatus] = useState<'optimal' | 'warning' | 'error'>('optimal')

  // Monitor connection status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Simulate system status monitoring
  useEffect(() => {
    const interval = setInterval(() => {
      const statuses: Array<'optimal' | 'warning' | 'error'> = ['optimal', 'optimal', 'optimal', 'warning']
      setSystemStatus(statuses[Math.floor(Math.random() * statuses.length)])
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = () => {
    switch (systemStatus) {
      case 'optimal': return 'text-green-400'
      case 'warning': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      default: return 'text-green-400'
    }
  }

  const getStatusText = () => {
    switch (systemStatus) {
      case 'optimal': return 'All Systems Optimal'
      case 'warning': return 'Minor Issues Detected'
      case 'error': return 'System Error'
      default: return 'All Systems Optimal'
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
            {isOnline ? (
              <Wifi className="h-4 w-4 text-green-400" />
            ) : (
              <WifiOff className="h-4 w-4 text-red-400" />
            )}
            <span className={`text-sm font-medium ${isOnline ? 'text-green-400' : 'text-red-400'}`}>
              {isOnline ? 'Connected' : 'Offline'}
            </span>
          </div>

          {/* System Status */}
          <div className="flex items-center space-x-2">
            <Activity className={`h-4 w-4 ${getStatusColor()}`} />
            <span className={`text-sm font-medium ${getStatusColor()}`}>
              {getStatusText()}
            </span>
          </div>

          {/* Neural Activity Indicator */}
          <div className="flex items-center space-x-2">
            <Zap className="h-4 w-4 text-cyan-400 animate-pulse" />
            <span className="text-sm font-medium text-cyan-400">
              Neural Active
            </span>
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
                      <p className="text-xs text-purple-400 capitalize">{user?.role} Account</p>
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
                  
                  <div className="border-t border-purple-500/20 my-2" />
                  
                  <button
                    onClick={() => {
                      setShowUserMenu(false)
                      logout()
                    }}
                    className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-red-500/20 transition-all duration-200 text-left text-red-400"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile System Status Bar */}
      <div className="md:hidden mt-3 pt-3 border-t border-purple-500/20">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              {isOnline ? (
                <Wifi className="h-3 w-3 text-green-400" />
              ) : (
                <WifiOff className="h-3 w-3 text-red-400" />
              )}
              <span className={isOnline ? 'text-green-400' : 'text-red-400'}>
                {isOnline ? 'Online' : 'Offline'}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Activity className={`h-3 w-3 ${getStatusColor()}`} />
              <span className={getStatusColor()}>
                {systemStatus === 'optimal' ? 'Optimal' : systemStatus === 'warning' ? 'Warning' : 'Error'}
              </span>
            </div>
          </div>
          
          <div className="flex items-center space-x-1">
            <Zap className="h-3 w-3 text-cyan-400 animate-pulse" />
            <span className="text-cyan-400">Neural Active</span>
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

export default Header
