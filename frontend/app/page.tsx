'use client'

/**
 * MasterX Quantum Intelligence Platform
 * Ultra-Premium Enterprise Interface
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Sparkles, Zap } from 'lucide-react'
import { MasterXSidebar } from '@/components/MasterXSidebar'
import { MasterXChatInterface } from '@/components/MasterXChatInterface'
import { MasterXSettingsPanel } from '@/components/MasterXSettingsPanel'
import { MasterXProfileSection } from '@/components/MasterXProfileSection'

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [activeView, setActiveView] = useState('chat')
  const [isLoaded, setIsLoaded] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)

  useEffect(() => {
    // Quantum loading sequence
    const timer = setTimeout(() => {
      setIsLoaded(true)
    }, 1200)

    return () => clearTimeout(timer)
  }, [])

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen)
  }

  const handleViewChange = (view: string) => {
    if (view === 'settings') {
      setSettingsOpen(true)
    } else {
      setActiveView(view)
    }
  }

  const renderActiveView = () => {
    switch (activeView) {
      case 'chat':
        return <MasterXChatInterface />
      case 'history':
        return <PlaceholderView title="Chat History" description="Conversation history management coming soon" icon={Brain} />
      case 'analytics':
        return <PlaceholderView title="Intelligence Analytics" description="Advanced analytics dashboard coming soon" icon={Sparkles} />
      case 'tools':
        return <PlaceholderView title="AI Tools" description="Quantum AI utilities coming soon" icon={Zap} />
      case 'profile':
        return <MasterXProfileSection />
      default:
        return <MasterXChatInterface />
    }
  }

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-neural-gray to-quantum-dark flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          {/* Quantum Loading Animation */}
          <motion.div className="relative mb-8">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="w-20 h-20 border-4 border-transparent bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full mx-auto"
              style={{
                background: 'conic-gradient(from 0deg, #a855f7, #06b6d4, #10b981, #f59e0b, #a855f7)',
                padding: '2px'
              }}
            >
              <div className="w-full h-full bg-quantum-dark rounded-full flex items-center justify-center">
                <Brain className="w-8 h-8 text-purple-400 animate-pulse" />
              </div>
            </motion.div>

            {/* Quantum Particles */}
            {[...Array(6)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-cyan-400 rounded-full"
                style={{
                  top: '50%',
                  left: '50%',
                  transformOrigin: '0 0'
                }}
                animate={{
                  rotate: [0, 360],
                  scale: [0.5, 1, 0.5],
                  opacity: [0.3, 1, 0.3]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: i * 0.3,
                  ease: "easeInOut"
                }}
              />
            ))}
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent mb-3"
          >
            MasterX
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="text-plasma-white/70 text-lg"
          >
            Quantum Intelligence Initializing...
          </motion.p>

          {/* Loading Progress */}
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: '100%' }}
            transition={{ duration: 1, delay: 0.7 }}
            className="w-48 h-1 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full mx-auto mt-6"
          />
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-neural-gray to-quantum-dark overflow-hidden">
      {/* Quantum Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {/* Primary Quantum Field */}
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.1, 0.3, 0.1],
            rotate: [0, 180, 360]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-purple-500/20 via-cyan-500/20 to-purple-500/20 rounded-full blur-3xl"
        />

        {/* Secondary Quantum Field */}
        <motion.div
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.1, 0.25, 0.1],
            rotate: [360, 180, 0]
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 10
          }}
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-cyan-500/20 via-emerald-500/20 to-cyan-500/20 rounded-full blur-3xl"
        />

        {/* Quantum Particles */}
        {[...Array(12)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              left: `${(i * 8 + 10) % 90 + 5}%`,
              top: `${(i * 13 + 15) % 80 + 10}%`,
            }}
            animate={{
              y: [0, -20, 0],
              opacity: [0.3, 1, 0.3],
              scale: [0.5, 1, 0.5]
            }}
            transition={{
              duration: 3 + (i % 3),
              repeat: Infinity,
              delay: i * 0.2,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>

      {/* Main Layout */}
      <div className="relative z-10 flex h-screen">
        {/* MasterX Sidebar */}
        <MasterXSidebar
          isOpen={sidebarOpen}
          onToggle={handleSidebarToggle}
          activeView={activeView}
          onViewChange={handleViewChange}
        />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="h-16 glass-morph border-b border-purple-500/20 flex items-center justify-between px-6">
            <button
              onClick={handleSidebarToggle}
              className="p-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200 lg:hidden"
            >
              <Brain className="h-5 w-5 text-purple-400" />
            </button>
            <h2 className="text-lg font-semibold text-plasma-white capitalize">
              {activeView === 'chat' ? 'Quantum Intelligence' : activeView.replace(/([A-Z])/g, ' $1').trim()}
            </h2>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-plasma-white/60">Online</span>
            </div>
          </div>

          {/* Content Area */}
          <div className="flex-1">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeView}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="h-full"
              >
                {renderActiveView()}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      <MasterXSettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}

// Placeholder View Component
interface PlaceholderViewProps {
  title: string
  description: string
  icon: React.ComponentType<{ className?: string }>
}

function PlaceholderView({ title, description, icon: Icon }: PlaceholderViewProps) {
  return (
    <div className="h-full flex items-center justify-center p-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center max-w-md"
      >
        <motion.div
          className="w-20 h-20 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-3xl flex items-center justify-center mx-auto mb-6"
          animate={{
            boxShadow: [
              "0 0 20px rgba(168, 85, 247, 0.3)",
              "0 0 40px rgba(168, 85, 247, 0.6)",
              "0 0 20px rgba(168, 85, 247, 0.3)"
            ]
          }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <Icon className="w-10 h-10 text-white" />
        </motion.div>

        <h3 className="text-2xl font-bold text-plasma-white mb-4">
          {title}
        </h3>
        <p className="text-plasma-white/70 leading-relaxed">
          {description}
        </p>

        <motion.div
          className="mt-8 p-4 glass-morph rounded-xl"
          whileHover={{ scale: 1.02 }}
        >
          <p className="text-sm text-plasma-white/60">
            This premium component is under development with enterprise-grade quality standards.
          </p>
        </motion.div>
      </motion.div>
    </div>
  )
}
