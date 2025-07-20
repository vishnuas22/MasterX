'use client'

/**
 * Dashboard Page - Redirect to Main Interface
 * 
 * This route redirects to the main interface since the sophisticated
 * dashboard is now the default route at '/'
 */

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function DashboardPage() {
  const router = useRouter()
  
  useEffect(() => {
    // Redirect to main interface
    router.replace('/')
  }, [router])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-4"></div>
        <p className="text-purple-300">Redirecting to MasterX Dashboard...</p>
      </div>
    </div>
  )
}