'use client'

import { useState } from 'react'
import ModernSidebar from '@/components/ModernSidebar'
import ModernChatInterface from '@/components/ModernChatInterface'

export default function ModernPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeView, setActiveView] = useState('chat')

  const renderMainContent = () => {
    switch (activeView) {
      case 'chat':
        return <ModernChatInterface />
      case 'history':
        return <ModernChatInterface />
      case 'settings':
        return <ModernChatInterface />
      default:
        return <ModernChatInterface />
    }
  }

  return (
    <div className="h-screen bg-gray-50 overflow-hidden">
      <div className="flex h-full">
        <ModernSidebar
          isOpen={sidebarOpen}
          onToggle={() => setSidebarOpen(!sidebarOpen)}
          activeView={activeView}
          onViewChange={setActiveView}
        />
        
        <main className="flex-1 relative overflow-hidden">
          {renderMainContent()}
        </main>
      </div>
    </div>
  )
}
