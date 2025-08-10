'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { 
  MessageSquare, 
  Plus, 
  Settings, 
  History,
  Menu,
  X
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface ModernSidebarProps {
  isOpen: boolean
  onToggle: () => void
  activeView: string
  onViewChange: (view: string) => void
  className?: string
}

const navigationItems = [
  {
    id: 'chat',
    label: 'Chat',
    icon: MessageSquare
  },
  {
    id: 'history',
    label: 'History',
    icon: History
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings
  }
]

const recentChats = [
  { id: '1', title: 'Python debugging help', time: '2m ago' },
  { id: '2', title: 'React component design', time: '1h ago' },
  { id: '3', title: 'Database optimization', time: '3h ago' },
  { id: '4', title: 'API integration guide', time: '1d ago' }
]

export default function ModernSidebar({ 
  isOpen, 
  onToggle, 
  activeView, 
  onViewChange, 
  className 
}: ModernSidebarProps) {
  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{
          x: isOpen ? 0 : -320,
          transition: { duration: 0.3, ease: "easeInOut" }
        }}
        className={cn(
          "fixed left-0 top-0 h-full w-80 bg-white border-r border-gray-200 z-50 lg:relative lg:translate-x-0 lg:z-auto",
          "flex flex-col",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center">
              <MessageSquare className="h-4 w-4 text-white" />
            </div>
            <span className="font-semibold text-gray-900">MasterX</span>
          </div>
          
          <button
            onClick={onToggle}
            className="p-2 hover:bg-gray-100 rounded-lg lg:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <button className="w-full flex items-center justify-center space-x-2 bg-black text-white rounded-lg py-3 px-4 hover:bg-gray-800 transition-colors">
            <Plus className="h-4 w-4" />
            <span className="font-medium">New chat</span>
          </button>
        </div>

        {/* Navigation */}
        <div className="px-4 pb-4">
          <nav className="space-y-1">
            {navigationItems.map((item) => {
              const isActive = activeView === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => onViewChange(item.id)}
                  className={cn(
                    "w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors",
                    isActive 
                      ? "bg-gray-100 text-gray-900" 
                      : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  <span className="font-medium">{item.label}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Recent Chats */}
        <div className="flex-1 px-4 pb-4">
          <div className="mb-3">
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider">
              Recent
            </h3>
          </div>
          
          <div className="space-y-1">
            {recentChats.map((chat) => (
              <button
                key={chat.id}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-left hover:bg-gray-50 transition-colors group"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900 truncate">
                    {chat.title}
                  </div>
                  <div className="text-xs text-gray-500">
                    {chat.time}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-100">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
              <span className="text-sm font-medium text-gray-600">U</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-900">User</div>
              <div className="text-xs text-gray-500">Free plan</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Mobile Menu Button */}
      <button
        onClick={onToggle}
        className="fixed top-4 left-4 z-50 p-2 bg-white border border-gray-200 rounded-lg shadow-sm lg:hidden"
      >
        <Menu className="h-4 w-4" />
      </button>
    </>
  )
}
