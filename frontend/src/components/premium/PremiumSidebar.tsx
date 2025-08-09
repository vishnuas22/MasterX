'use client'

import { Fragment } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MessageSquare, 
  BarChart3, 
  FileText, 
  Settings, 
  Zap,
  Brain,
  Sparkles,
  X
} from 'lucide-react'

interface PremiumSidebarProps {
  open: boolean
  setOpen: (open: boolean) => void
  activeView: string
  setActiveView: (view: string) => void
}

const navigation = [
  { 
    name: 'Quantum Chat', 
    id: 'chat', 
    icon: MessageSquare,
    description: 'AI-powered conversations'
  },
  { 
    name: 'Analytics', 
    id: 'analytics', 
    icon: BarChart3,
    description: 'Performance insights'
  },
  { 
    name: 'Files', 
    id: 'files', 
    icon: FileText,
    description: 'Document management'
  },
  { 
    name: 'Settings', 
    id: 'settings', 
    icon: Settings,
    description: 'System configuration'
  },
]

export default function PremiumSidebar({ open, setOpen, activeView, setActiveView }: PremiumSidebarProps) {
  return (
    <>
      {/* Desktop Sidebar */}
      <motion.div
        initial={false}
        animate={{ 
          x: open ? 0 : -320,
          opacity: open ? 1 : 0
        }}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 30 
        }}
        className="fixed inset-y-0 left-0 z-50 w-80 bg-black/20 backdrop-blur-xl border-r border-white/10 hidden lg:block"
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-white/10">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">MasterX</h1>
              <p className="text-xs text-purple-300">Quantum Intelligence</p>
            </div>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const isActive = activeView === item.id
            return (
              <motion.button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`
                  w-full flex items-center px-4 py-3 text-left rounded-xl transition-all duration-200
                  ${isActive 
                    ? 'bg-gradient-to-r from-purple-500/20 to-cyan-500/20 text-white border border-purple-500/30' 
                    : 'text-gray-300 hover:text-white hover:bg-white/5'
                  }
                `}
              >
                <div className={`
                  p-2 rounded-lg mr-3 transition-colors
                  ${isActive 
                    ? 'bg-gradient-to-br from-purple-500 to-cyan-500 text-white' 
                    : 'bg-gray-700/50 text-gray-400'
                  }
                `}>
                  <item.icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs text-gray-400">{item.description}</div>
                </div>
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="w-2 h-2 bg-gradient-to-br from-purple-400 to-cyan-400 rounded-full"
                  />
                )}
              </motion.button>
            )
          })}
        </nav>

        {/* Status Panel */}
        <div className="p-4 border-t border-white/10">
          <div className="bg-gradient-to-br from-purple-500/10 to-cyan-500/10 rounded-xl p-4 border border-purple-500/20">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-white">Quantum Engine</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400">Active</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Performance</span>
                <span className="text-purple-300">98.7%</span>
              </div>
              <div className="w-full bg-gray-700/50 rounded-full h-1.5">
                <motion.div 
                  className="bg-gradient-to-r from-purple-500 to-cyan-500 h-1.5 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: '98.7%' }}
                  transition={{ duration: 2, ease: "easeOut" }}
                />
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed inset-y-0 left-0 z-50 w-80 bg-black/90 backdrop-blur-xl border-r border-white/10 lg:hidden"
          >
            {/* Mobile content - same as desktop but with backdrop */}
            <div className="flex items-center justify-between h-16 px-6 border-b border-white/10">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">MasterX</h1>
                  <p className="text-xs text-purple-300">Quantum Intelligence</p>
                </div>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <nav className="flex-1 px-4 py-6 space-y-2">
              {navigation.map((item) => {
                const isActive = activeView === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveView(item.id)
                      setOpen(false) // Close mobile sidebar on selection
                    }}
                    className={`
                      w-full flex items-center px-4 py-3 text-left rounded-xl transition-all duration-200
                      ${isActive 
                        ? 'bg-gradient-to-r from-purple-500/20 to-cyan-500/20 text-white border border-purple-500/30' 
                        : 'text-gray-300 hover:text-white hover:bg-white/5'
                      }
                    `}
                  >
                    <div className={`
                      p-2 rounded-lg mr-3 transition-colors
                      ${isActive 
                        ? 'bg-gradient-to-br from-purple-500 to-cyan-500 text-white' 
                        : 'bg-gray-700/50 text-gray-400'
                      }
                    `}>
                      <item.icon className="w-5 h-5" />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">{item.name}</div>
                      <div className="text-xs text-gray-400">{item.description}</div>
                    </div>
                  </button>
                )
              })}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
