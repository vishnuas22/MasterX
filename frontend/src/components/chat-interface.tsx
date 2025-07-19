'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Bot, User, Loader2, Brain, Zap, Target, Sparkles, MessageCircle, Settings, Code, Palette, Eye } from 'lucide-react'
import { api, sendMessage, streamMessage, ChatRequest, ChatResponse } from '@/lib/api'
import { cn } from '@/lib/utils'
import { useAuth } from '@/contexts/AuthContext'
import useWebSocket from '@/hooks/useWebSocket'
import { QuantumDropdown, TASK_TYPE_OPTIONS, PROVIDER_OPTIONS } from '@/components/ui/quantum-dropdown'

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  provider?: string
  model?: string
  task_type?: string
  suggestions?: string[]
  metadata?: {
    learningMode?: string
    concepts?: string[]
    confidence?: number
    intelligence_level?: string
    engagement_prediction?: number
    knowledge_gaps?: string[]
    next_concepts?: string[]
    response_time?: number
    tokens_used?: number
    task_optimization?: string
  }
}

export function ChatInterface() {
  const { user, token, isAuthenticated } = useAuth()

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI mentor powered by the MasterX Quantum Intelligence Engine with multi-LLM integration. I can intelligently select the best AI model for your specific task - whether it\'s reasoning, coding, creative writing, or quick responses. What would you like to explore today?',
      sender: 'ai',
      timestamp: new Date(),
      provider: 'system',
      task_type: 'general',
      metadata: {
        learningMode: 'adaptive_quantum',
        concepts: ['introduction', 'quantum_intelligence', 'multi_llm'],
        confidence: 0.95
      }
    }
  ])

  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>('')
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting')

  // Enhanced features
  const [taskType, setTaskType] = useState<string>('general')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [streamingEnabled, setStreamingEnabled] = useState(true)
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)

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

      // Authentication check removed for development
      // if (!isAuthenticated) {
      //   setConnectionStatus('error')
      //   return
      // }

      // Session will be created automatically on first message
      // For now, use a fallback session ID
      const fallbackSessionId = `session_${Date.now()}`
      setSessionId(fallbackSessionId)
      setConnectionStatus('connected')
      console.log('✅ Session initialized:', fallbackSessionId)
    } catch (error) {
      console.error('❌ Failed to initialize session:', error)
      setConnectionStatus('error')
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return  // Removed authentication check for development

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
      task_type: taskType
    }

    setMessages(prev => [...prev, userMessage])
    const currentMessage = inputMessage
    setInputMessage('')
    setIsLoading(true)

    try {
      const chatRequest: ChatRequest = {
        message: currentMessage,
        session_id: sessionId || undefined,
        message_type: 'text',
        task_type: taskType as 'reasoning' | 'coding' | 'creative' | 'fast' | 'multimodal' | 'general',
        provider: (selectedProvider as 'groq' | 'gemini' | 'openai' | 'anthropic') || undefined,
        stream: streamingEnabled
      }

      if (streamingEnabled) {
        // Handle streaming response
        setIsStreaming(true)
        setCurrentStreamingMessage('')

        const streamingMessageId = (Date.now() + 1).toString()

        // Add placeholder message for streaming
        const placeholderMessage: Message = {
          id: streamingMessageId,
          content: '',
          sender: 'ai',
          timestamp: new Date(),
          task_type: taskType
        }
        setMessages(prev => [...prev, placeholderMessage])

        await streamMessage(
          chatRequest,
          (chunk) => {
            if (chunk.content) {
              setCurrentStreamingMessage(prev => prev + chunk.content)
              // Update the placeholder message
              setMessages(prev => prev.map(msg =>
                msg.id === streamingMessageId
                  ? { ...msg, content: msg.content + chunk.content }
                  : msg
              ))
            }
          },
          () => {
            setIsStreaming(false)
            setCurrentStreamingMessage('')
          },
          (error) => {
            console.error('Streaming error:', error)
            setIsStreaming(false)
            setCurrentStreamingMessage('')
          }
        )
      } else {
        // Handle regular response using direct fetch
        const response = await fetch('http://localhost:8000/api/v1/chat/message', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(chatRequest),
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()

        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: data.response,
          sender: 'ai',
          timestamp: new Date(),
          provider: data.provider_used || selectedProvider,
          model: data.model,
          task_type: data.task_type || taskType,
          suggestions: data.suggestions || [],
          metadata: {
            learningMode: 'adaptive_quantum',
            concepts: [],
            confidence: 0.85,
            response_time: data.processing_time,
            tokens_used: data.metadata?.tokens_used,
            task_optimization: data.metadata?.task_optimization
          }
        }
        setMessages(prev => [...prev, aiMessage])

        // Update session ID if it changed
        if (data.session_id && data.session_id !== sessionId) {
          setSessionId(data.session_id)
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
      handleSendMessage()
    }
  }

  return (
    <div className="h-full flex flex-col p-6">
      {/* Session Info Bar */}
      <div className="glass-morph rounded-xl p-4 mb-6 border border-purple-500/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-purple-400 animate-quantum-pulse" />
              <span className="text-white font-medium">Quantum Intelligence Session</span>
            </div>
            {sessionId && (
              <div className="flex items-center space-x-2">
                <MessageCircle className="h-4 w-4 text-purple-400" />
                <span className="text-xs text-purple-400 font-mono">ID: {sessionId.substring(0, 8)}...</span>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 text-sm ${
              connectionStatus === 'connected' ? 'text-green-400' :
              connectionStatus === 'error' ? 'text-red-400' : 'text-yellow-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-400' :
                connectionStatus === 'error' ? 'bg-red-400' : 'bg-blue-400 animate-pulse'
              }`}></div>
              <span>{connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'error' ? 'Error' : 'Connecting'}</span>
            </div>
          </div>
        </div>
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

      {/* Enhanced Controls */}
      <div className="glass-morph rounded-xl p-6 mb-4 border border-purple-500/20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <QuantumDropdown
            label="Task Type"
            options={TASK_TYPE_OPTIONS}
            value={taskType}
            onChange={setTaskType}
            placeholder="Select task type..."
          />

          <QuantumDropdown
            label="AI Provider"
            options={PROVIDER_OPTIONS}
            value={selectedProvider}
            onChange={setSelectedProvider}
            placeholder="Select provider..."
          />

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-purple-300">Streaming:</label>
              <input
                type="checkbox"
                checked={streamingEnabled}
                onChange={(e) => setStreamingEnabled(e.target.checked)}
                className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded focus:ring-purple-500"
              />
            </div>

            {isAuthenticated && user && (
              <div className="flex items-center space-x-2">
                <span className="text-purple-400 border border-purple-400 px-3 py-2 rounded-lg text-sm font-medium">
                  {user.name} ({user.role})
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Input Area */}
      <div className="glass-morph rounded-xl p-6 border border-purple-500/20">
        <div className="flex space-x-4">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask me anything about learning, or let me know what you'd like to explore..."
            className="flex-1 bg-slate-700/50 text-white placeholder-gray-400 border border-slate-600 rounded-lg px-4 py-3 focus:border-purple-500 focus:outline-none resize-none focus-quantum transition-all"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}  // Removed authentication check
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