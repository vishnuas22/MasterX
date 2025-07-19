/**
 * Login Form Component for MasterX Quantum Intelligence Platform
 * 
 * Provides authentication interface with test accounts and
 * integration with the Phase 12 backend authentication system.
 */

'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, LogIn, Brain, Zap, Shield } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

export function LoginForm() {
  const { login, isLoading } = useAuth()
  const [email, setEmail] = useState('student@example.com')
  const [password, setPassword] = useState('student123')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsSubmitting(true)

    try {
      await login({
        email,
        password,
        remember_me: true
      })
    } catch (err: any) {
      setError(err.message || 'Login failed. Please check your credentials.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleTestAccount = (accountType: 'student' | 'teacher' | 'admin') => {
    const accounts = {
      student: { email: 'student@example.com', password: 'student123' },
      teacher: { email: 'teacher@example.com', password: 'teacher123' },
      admin: { email: 'admin@masterx.ai', password: 'admin123' }
    }
    
    const account = accounts[accountType]
    setEmail(account.email)
    setPassword(account.password)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-purple-400 mx-auto mb-4" />
          <p className="text-gray-300">Initializing authentication...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Brain className="h-12 w-12 text-purple-400 mr-3" />
            <h1 className="text-4xl font-bold text-white">MasterX</h1>
          </div>
          <p className="text-gray-300">Quantum Intelligence Platform</p>
          <p className="text-sm text-purple-400 mt-2">Phase 13: Frontend Integration & Multi-LLM Enhancement</p>
        </div>

        {/* Login Card */}
        <Card className="glass-morph border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-center text-white flex items-center justify-center">
              <LogIn className="h-5 w-5 mr-2" />
              Login to Continue
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <Alert className="border-red-500/20 bg-red-500/10">
                  <AlertDescription className="text-red-400">
                    {error}
                  </AlertDescription>
                </Alert>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Email
                </label>
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-slate-700/50 border-slate-600 text-white"
                  placeholder="Enter your email"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Password
                </label>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-slate-700/50 border-slate-600 text-white"
                  placeholder="Enter your password"
                  required
                />
              </div>

              <Button
                type="submit"
                disabled={isSubmitting}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Signing In...
                  </>
                ) : (
                  <>
                    <LogIn className="h-4 w-4 mr-2" />
                    Sign In
                  </>
                )}
              </Button>
            </form>

            {/* Test Accounts */}
            <div className="mt-6 pt-6 border-t border-slate-600">
              <p className="text-sm text-gray-400 mb-3 text-center">Test Accounts:</p>
              <div className="grid grid-cols-3 gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => handleTestAccount('student')}
                  className="text-xs border-slate-600 text-gray-300 hover:bg-slate-700"
                >
                  Student
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => handleTestAccount('teacher')}
                  className="text-xs border-slate-600 text-gray-300 hover:bg-slate-700"
                >
                  Teacher
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => handleTestAccount('admin')}
                  className="text-xs border-slate-600 text-gray-300 hover:bg-slate-700"
                >
                  Admin
                </Button>
              </div>
            </div>

            {/* Features */}
            <div className="mt-6 pt-6 border-t border-slate-600">
              <p className="text-sm text-gray-400 mb-3 text-center">Platform Features:</p>
              <div className="space-y-2 text-xs text-gray-500">
                <div className="flex items-center">
                  <Brain className="h-3 w-3 mr-2 text-purple-400" />
                  <span>Intelligent Model Selection</span>
                </div>
                <div className="flex items-center">
                  <Zap className="h-3 w-3 mr-2 text-purple-400" />
                  <span>Real-time Streaming Responses</span>
                </div>
                <div className="flex items-center">
                  <Shield className="h-3 w-3 mr-2 text-purple-400" />
                  <span>Multi-LLM Integration (Groq, Gemini, OpenAI, Claude)</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center mt-6 text-sm text-gray-500">
          <p>MasterX Quantum Intelligence Platform</p>
          <p>Phase 13: Frontend Integration & Multi-LLM Enhancement</p>
        </div>
      </div>
    </div>
  )
}

export default LoginForm
