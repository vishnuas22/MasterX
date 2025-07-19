'use client'

/**
 * Chat Page Component
 * 
 * Next.js page for /chat route using the existing interactive chat component
 */

import { InteractiveChat } from '@/components/interactive-chat'

export default function ChatPage() {
  return (
    <div className="h-screen">
      <InteractiveChat />
    </div>
  )
}