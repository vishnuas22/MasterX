'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Brain, Zap, Target, Sparkles, MessageCircle } from 'lucide-react'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  metadata?: {
    learningMode?: string
    concepts?: string[]
    confidence?: number
    intelligence_level?: string
    engagement_prediction?: number
    knowledge_gaps?: string[]
    next_concepts?: string[]
  }
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI mentor powered by the Quantum Intelligence Engine. I can help you learn using advanced modes like Socratic questioning, debug mastery, and creative synthesis. What would you like to explore today?',
      sender: 'ai',
      timestamp: new Date(),
      metadata: {
        learningMode: 'adaptive_quantum',
        concepts: ['introduction', 'quantum_intelligence'],
        confidence: 0.95
      }
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>('')
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    initializeSession()
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const initializeSession = async () => {
    try {
      setConnectionStatus('connecting')
      const response = await api.chat.createSession()
      if (response.data && response.data.session_id) {
        setSessionId(response.data.session_id)
        setConnectionStatus('connected')
        console.log('✅ Session created:', response.data.session_id)
      } else {
        // Fallback session ID
        const fallbackSessionId = `session_${Date.now()}`
        setSessionId(fallbackSessionId)
        setConnectionStatus('connected')
        console.log('⚠️ Using fallback session ID:', fallbackSessionId)
      }
    } catch (error) {
      console.error('❌ Failed to initialize session:', error)
      // Use fallback session ID
      const fallbackSessionId = `session_${Date.now()}`
      setSessionId(fallbackSessionId)
      setConnectionStatus('error')
    }
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      // Call real backend API
      const response = await api.chat.send(inputMessage, sessionId)
      
      if (response.data) {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: response.data.response,
          sender: 'ai',
          timestamp: new Date(),
          metadata: {
            learningMode: response.data.metadata?.learning_mode || 'adaptive_quantum',
            concepts: response.data.metadata?.concepts || [],
            confidence: response.data.metadata?.confidence || 0.85,
            intelligence_level: response.data.metadata?.intelligence_level,
            engagement_prediction: response.data.metadata?.engagement_prediction,
            knowledge_gaps: response.data.metadata?.knowledge_gaps,
            next_concepts: response.data.metadata?.next_concepts
          }
        }
        setMessages(prev => [...prev, aiMessage])
        
        // Update session ID if it changed
        if (response.data.session_id && response.data.session_id !== sessionId) {
          setSessionId(response.data.session_id)
        }
      }
    } catch (error) {
      console.error('❌ Failed to send message:', error)
      
      // Fallback AI response
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I apologize, but I'm experiencing some technical difficulties connecting to the Quantum Intelligence Engine. Please try again in a moment.",
        sender: 'ai',
        timestamp: new Date(),
        metadata: {
          learningMode: 'fallback',
          concepts: ['error_handling'],
          confidence: 0.5
        }
      }
      setMessages(prev => [...prev, fallbackMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="max-w-6xl mx-auto p-6 h-screen flex flex-col">
      {/* Enhanced Header */}
      <div className="mb-6 glass-morph p-6 rounded-xl border border-purple-500/20">
        <div className="flex items-center space-x-3 mb-2">
          <Brain className="h-8 w-8 text-purple-400 quantum-pulse" />
          <h1 className="text-3xl font-bold quantum-text-glow">AI Learning Mentor</h1>
          <Sparkles className="h-6 w-6 text-purple-300 quantum-float" />
          {/* Connection Status Indicator */}
          <div className={`ml-auto flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium ${
            connectionStatus === 'connected' 
              ? 'bg-green-900/50 text-green-300 border border-green-500/30' 
              : connectionStatus === 'error' 
              ? 'bg-red-900/50 text-red-300 border border-red-500/30'
              : 'bg-blue-900/50 text-blue-300 border border-blue-500/30'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-400' : 
              connectionStatus === 'error' ? 'bg-red-400' : 'bg-blue-400 animate-pulse'
            }`}></div>
            <span>{connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'error' ? 'Error' : 'Connecting'}</span>
          </div>
        </div>
        <p className="text-gray-400">Powered by Quantum Intelligence Engine</p>
        {sessionId && (
          <div className="flex items-center space-x-2 mt-2">
            <MessageCircle className="h-4 w-4 text-purple-400" />
            <p className="text-xs text-purple-400">Session: {sessionId.substring(0, 8)}...</p>
          </div>
        )}
      </div>

      {/* Enhanced Messages Container */}
      <div className="flex-1 overflow-y-auto glass-morph rounded-xl p-6 mb-6 border border-purple-500/20">
        <div className="space-y-6">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isLoading && <LoadingIndicator />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Enhanced Input Area */}
      <div className="glass-morph rounded-xl p-6 border border-purple-500/20">
        <div className="flex space-x-4">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about learning, or let me know what you'd like to explore..."
            className="flex-1 bg-slate-700/50 text-white placeholder-gray-400 border border-slate-600 rounded-lg px-4 py-3 focus:border-purple-500 focus:outline-none resize-none focus-quantum transition-all"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-semibold quantum-glow interactive-button transition-all flex items-center space-x-2"
          >
            <Send className="h-5 w-5" />
            <span>Send</span>
          </button>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  return (
    <div className={cn(
      'flex space-x-4 transition-all duration-500 opacity-0 animate-[fadeIn_0.5s_ease-in-out_forwards]',
      message.sender === 'user' ? 'justify-end' : 'justify-start'
    )}>
      {message.sender === 'ai' && (
        <div className="flex-shrink-0">
          <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center quantum-glow">
            <Brain className="h-6 w-6 text-white quantum-pulse" />
          </div>
        </div>
      )}
      
      <div className={cn(
        'max-w-3xl p-6 rounded-xl transition-all duration-300 interactive-card',
        message.sender === 'user' 
          ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white ml-auto quantum-glow' 
          : 'glass-morph text-gray-100 border border-purple-500/20'
      )}>
        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        
        {message.metadata && message.sender === 'ai' && (
          <div className="mt-4 pt-4 border-t border-purple-500/20">
            <div className="flex flex-wrap gap-2 text-xs">
              {message.metadata.learningMode && (
                <span className="glass-morph text-purple-300 px-3 py-1 rounded-full flex items-center space-x-1 border border-purple-500/30">
                  <Zap className="h-3 w-3" />
                  <span>{message.metadata.learningMode.replace('_', ' ')}</span>
                </span>
              )}
              {message.metadata.intelligence_level && (
                <span className="glass-morph text-cyan-300 px-3 py-1 rounded-full flex items-center space-x-1 border border-cyan-500/30">
                  <Brain className="h-3 w-3" />
                  <span>{message.metadata.intelligence_level}</span>
                </span>
              )}
              {message.metadata.concepts && message.metadata.concepts.length > 0 && (
                <span className="glass-morph text-blue-300 px-3 py-1 rounded-full flex items-center space-x-1 border border-blue-500/30">
                  <Target className="h-3 w-3" />
                  <span>{message.metadata.concepts.slice(0, 3).join(', ')}</span>
                </span>
              )}
              {message.metadata.confidence && (
                <span className="glass-morph text-green-300 px-3 py-1 rounded-full border border-green-500/30">
                  {Math.round(message.metadata.confidence * 100)}% confidence
                </span>
              )}
              {message.metadata.engagement_prediction && (
                <span className="glass-morph text-yellow-300 px-3 py-1 rounded-full border border-yellow-500/30">
                  {Math.round(message.metadata.engagement_prediction * 100)}% engagement
                </span>
              )}
            </div>
            
            {/* Advanced Metadata */}
            {(message.metadata.knowledge_gaps && message.metadata.knowledge_gaps.length > 0) && (
              <div className="mt-3 p-3 glass-morph rounded-lg border border-red-500/20">
                <p className="text-xs font-semibold text-red-300 mb-1">Knowledge Gaps Identified:</p>
                <p className="text-xs text-red-200">{message.metadata.knowledge_gaps.slice(0, 2).join(', ')}</p>
              </div>
            )}
            
            {(message.metadata.next_concepts && message.metadata.next_concepts.length > 0) && (
              <div className="mt-3 p-3 glass-morph rounded-lg border border-emerald-500/20">
                <p className="text-xs font-semibold text-emerald-300 mb-1">Next Recommended Concepts:</p>
                <p className="text-xs text-emerald-200">{message.metadata.next_concepts.slice(0, 3).join(', ')}</p>
              </div>
            )}
          </div>
        )}
        
        <p className="text-xs text-gray-400 mt-3 opacity-70">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>

      {message.sender === 'user' && (
        <div className="flex-shrink-0">
          <div className="w-10 h-10 rounded-full bg-gradient-to-r from-slate-600 to-slate-700 flex items-center justify-center border border-purple-500/30">
            <User className="h-6 w-6 text-white" />
          </div>
        </div>
      )}
    </div>
  )
}

function LoadingIndicator() {
  return (
    <div className="flex space-x-4 justify-start">
      <div className="flex-shrink-0">
        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center quantum-glow">
          <Brain className="h-6 w-6 text-white quantum-pulse" />
        </div>
      </div>
      <div className="glass-morph p-6 rounded-xl border border-purple-500/20 interactive-card">
        <div className="flex items-center space-x-3">
          <Loader2 className="h-5 w-5 animate-spin text-purple-400" />
          <span className="text-gray-300">AI is thinking...</span>
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    </div>
  )
}