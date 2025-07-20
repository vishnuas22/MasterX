'use client'

/**
 * Chat Page Component
 * 
 * Next.js page for /chat route using the interactive chat component
 * This now redirects to the main interface with chat view active
 */

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function ChatPage() {
  const router = useRouter()
  
  useEffect(() => {
    // Redirect to main interface with chat view
    router.replace('/?view=chat')
  }, [router])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-4"></div>
        <p className="text-purple-300">Opening MasterX Chat Interface...</p>
      </div>
    </div>
  )
}