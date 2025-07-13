'use client'

import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { apiService } from '@/lib/api'

export function ConnectivityCheck() {
  const [status, setStatus] = useState<'checking' | 'connected' | 'error'>('checking')
  const [message, setMessage] = useState('')

  useEffect(() => {
    checkBackendConnection()
  }, [])

  const checkBackendConnection = async () => {
    try {
      setStatus('checking')
      const response = await apiService.get('/')
      if (response.data?.message) {
        setStatus('connected')
        setMessage('Connected to Quantum Intelligence Engine')
      }
    } catch (error) {
      setStatus('error')
      setMessage('Failed to connect to backend')
      console.error('Backend connection error:', error)
    }
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className={`
        flex items-center space-x-2 px-4 py-2 rounded-lg backdrop-blur-lg border transition-all
        ${status === 'connected' 
          ? 'bg-green-900/50 border-green-500/50 text-green-300' 
          : status === 'error' 
          ? 'bg-red-900/50 border-red-500/50 text-red-300'
          : 'bg-blue-900/50 border-blue-500/50 text-blue-300'
        }
      `}>
        {status === 'checking' && <Loader2 className="h-4 w-4 animate-spin" />}
        {status === 'connected' && <CheckCircle className="h-4 w-4" />}
        {status === 'error' && <XCircle className="h-4 w-4" />}
        <span className="text-sm font-medium">{message}</span>
        {status === 'error' && (
          <button
            onClick={checkBackendConnection}
            className="text-xs underline hover:no-underline"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  )
}