'use client'

import { Bell, Search, Menu, User, Settings, LogOut } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

interface HeaderProps {
  user: any
  onMenuClick: () => void
}

export function Header({ user, onMenuClick }: HeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false)

  return (
    <header className="glass-morph-premium border-b border-purple-500/20 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left Section */}
        <div className="flex items-center space-x-4">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-lg hover:bg-purple-500/20 transition-colors"
          >
            <Menu className="h-5 w-5 text-gray-400" />
          </button>

          <div className="hidden lg:block">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search or ask AI..."
                className="w-96 pl-10 pr-4 py-2 bg-gray-800/50 border border-purple-500/30 rounded-xl text-white placeholder-gray-400 focus-quantum"
              />
            </div>
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <button className="p-2 rounded-lg hover:bg-purple-500/20 transition-colors relative">
            <Bell className="h-5 w-5 text-gray-400" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-quantum-pulse" />
          </button>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-3 p-2 rounded-xl hover:bg-purple-500/20 transition-colors"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-cyan-500 flex items-center justify-center">
                <User className="h-4 w-4 text-white" />
              </div>
              <div className="text-left hidden sm:block">
                <p className="precision-small text-white font-medium">{user.name}</p>
                <p className="data-micro text-gray-400 capitalize">{user.role}</p>
              </div>
            </button>

            {/* User Dropdown */}
            {showUserMenu && (
              <div className="absolute right-0 top-12 w-48 glass-morph border border-purple-500/30 rounded-xl py-2 z-50">
                <button className="w-full px-4 py-2 text-left precision-small text-gray-300 hover:bg-purple-500/20 hover:text-white transition-colors flex items-center space-x-2">
                  <Settings className="h-4 w-4" />
                  <span>Settings</span>
                </button>
                <button className="w-full px-4 py-2 text-left precision-small text-gray-300 hover:bg-purple-500/20 hover:text-white transition-colors flex items-center space-x-2">
                  <LogOut className="h-4 w-4" />
                  <span>Sign out</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}