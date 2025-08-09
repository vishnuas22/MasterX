'use client'

import { motion } from 'framer-motion'
import { 
  Menu, 
  Bell, 
  Search, 
  User, 
  Zap,
  Wifi,
  Shield,
  Activity
} from 'lucide-react'

interface PremiumHeaderProps {
  setSidebarOpen: (open: boolean) => void
  sidebarOpen: boolean
  activeView: string
}

const viewTitles = {
  chat: 'Quantum Chat Interface',
  analytics: 'Performance Analytics',
  files: 'File Management',
  settings: 'System Settings'
}

export default function PremiumHeader({ setSidebarOpen, sidebarOpen, activeView }: PremiumHeaderProps) {
  return (
    <motion.header 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="sticky top-0 z-40 bg-black/20 backdrop-blur-xl border-b border-white/10"
    >
      <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        {/* Left Section */}
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors lg:hidden"
          >
            <Menu className="w-5 h-5" />
          </button>
          
          <div className="hidden lg:block">
            <h1 className="text-xl font-semibold text-white">
              {viewTitles[activeView as keyof typeof viewTitles] || 'MasterX Platform'}
            </h1>
            <p className="text-sm text-gray-400">
              Enterprise AI Collaboration Platform
            </p>
          </div>
        </div>

        {/* Center Section - Search */}
        <div className="hidden md:flex flex-1 max-w-lg mx-8">
          <div className="relative w-full">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search conversations, files, or commands..."
              className="
                w-full pl-10 pr-4 py-2 
                bg-white/5 border border-white/10 rounded-xl
                text-white placeholder-gray-400
                focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50
                transition-all duration-200
              "
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center space-x-3">
          {/* Status Indicators */}
          <div className="hidden sm:flex items-center space-x-2">
            <motion.div 
              className="flex items-center space-x-1 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-lg"
              whileHover={{ scale: 1.05 }}
            >
              <Wifi className="w-3 h-3 text-green-400" />
              <span className="text-xs text-green-400 font-medium">Online</span>
            </motion.div>
            
            <motion.div 
              className="flex items-center space-x-1 px-3 py-1.5 bg-purple-500/10 border border-purple-500/20 rounded-lg"
              whileHover={{ scale: 1.05 }}
            >
              <Zap className="w-3 h-3 text-purple-400" />
              <span className="text-xs text-purple-400 font-medium">Quantum</span>
            </motion.div>
          </div>

          {/* Notifications */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="relative p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <Bell className="w-5 h-5" />
            <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></div>
          </motion.button>

          {/* User Menu */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex items-center space-x-3 px-3 py-2 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors cursor-pointer"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <div className="hidden sm:block">
              <div className="text-sm font-medium text-white">Admin User</div>
              <div className="text-xs text-gray-400">Enterprise</div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Mobile Search */}
      <div className="md:hidden px-4 pb-3">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search..."
            className="
              w-full pl-10 pr-4 py-2 
              bg-white/5 border border-white/10 rounded-xl
              text-white placeholder-gray-400
              focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50
              transition-all duration-200
            "
          />
        </div>
      </div>
    </motion.header>
  )
}
