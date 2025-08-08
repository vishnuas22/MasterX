'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Bot, User, Loader2, Brain, Zap, Target, Sparkles, MessageCircle, Settings, Mic, MicOff, Camera, Paperclip, MoreHorizontal } from 'lucide-react'
import { sendMessage, streamMessage, ChatRequest, ChatResponse } from '@/lib/api'
import { cn } from '@/lib/utils'
import { useAuth } from '@/contexts/AuthContext'
import useWebSocket from '@/hooks/useWebSocket'
import { useVoiceInput } from '@/hooks/useVoiceInput'

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

const TASK_TYPE_OPTIONS = [
  { value: 'general', label: 'General Chat', icon: MessageCircle, color: 'purple' },
  { value: 'reasoning', label: 'Deep Reasoning', icon: Brain, color: 'blue' },
  { value: 'coding', label: 'Code Assistant', icon: Target, color: 'green' },
  { value: 'creative', label: 'Creative Writing', icon: Sparkles, color: 'pink' },
  { value: 'fast', label: 'Quick Response', icon: Zap, color: 'yellow' },
  { value: 'multimodal', label: 'Multimodal', icon: Camera, color: 'cyan' },
]

const PROVIDER_OPTIONS = [
  { value: '', label: 'Auto Select', color: 'purple' },
  { value: 'groq', label: 'Groq (DeepSeek)', color: 'blue' },
  { value: 'gemini', label: 'Google Gemini', color: 'green' },
  { value: 'openai', label: 'OpenAI GPT', color: 'orange' },
  { value: 'anthropic', label: 'Anthropic Claude', color: 'red' },
]

export function QuantumChatInterface() {
  const { user, token, isAuthenticated } = useAuth()

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI mentor powered by the MasterX Quantum Intelligence Engine. I can intelligently select the best AI model for your specific task and provide revolutionary learning experiences. What would you like to explore today?',
      sender: 'ai',
      timestamp: new Date(),
      provider: 'system',
      task_type: 'general',
      metadata: {
        learningMode: 'adaptive_quantum',
        concepts: ['introduction', 'quantum_intelligence', 'multi_llm'],
        confidence: 0.95,
        intelligence_level: 'ENHANCED'
      }
    }
  ])

  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>('')
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connected')

  // Enhanced features
  const [taskType, setTaskType] = useState<string>('general')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [streamingEnabled, setStreamingEnabled] = useState(true)
  const [isStreaming, setIsStreaming] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Voice input functionality
  const voiceInput = useVoiceInput({
    continuous: false,
    interimResults: true,
    language: 'en-US',
    onTranscript: (transcript, isFinal) => {
      if (isFinal) {
        setInputMessage(prev => prev + transcript + ' ')
        inputRef.current?.focus()
      }
    },
    onError: (error) => {
      console.error('Voice input error:', error)
      setConnectionStatus('error')
    },
    onStart: () => {
      console.log('Voice recording started')
    },
    onEnd: () => {
      console.log('Voice recording ended')
    }
  })

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
    if (!inputMessage.trim() || isLoading) return

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
    setConnectionStatus('connecting')

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
        await handleStreamingResponse(chatRequest, userMessage.id)
      } else {
        await handleRegularResponse(chatRequest)
      }

      setConnectionStatus('connected')
    } catch (error) {
      console.error('❌ Failed to send message:', error)
      setConnectionStatus('error')
      addFallbackMessage(error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleStreamingResponse = async (request: ChatRequest, messageId: string) => {
    setIsStreaming(true)
    const streamingMessageId = (Date.now() + 1).toString()

    const placeholderMessage: Message = {
      id: streamingMessageId,
      content: '',
      sender: 'ai',
      timestamp: new Date(),
      task_type: taskType
    }
    setMessages(prev => [...prev, placeholderMessage])

    try {
      await streamMessage(
        request,
        (chunk) => {
          if (chunk.content) {
            setMessages(prev => prev.map(msg =>
              msg.id === streamingMessageId
                ? { ...msg, content: msg.content + chunk.content }
                : msg
            ))
          }
        },
        () => {
          setIsStreaming(false)
        },
        (error) => {
          console.error('Streaming error:', error)
          setIsStreaming(false)
        }
      )
    } catch (error) {
      console.error('Streaming failed:', error)
      setIsStreaming(false)
    }
  }

  const handleRegularResponse = async (request: ChatRequest) => {
    try {
      const response = await sendMessage(request)
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.response,
        sender: 'ai',
        timestamp: new Date(),
        provider: selectedProvider,
        task_type: taskType,
        metadata: {
          learningMode: 'adaptive_quantum',
          concepts: [],
          confidence: 0.85,
          response_time: response.metadata?.response_time,
          tokens_used: response.metadata?.tokens_used,
          task_optimization: response.metadata?.task_optimization
        }
      }
      setMessages(prev => [...prev, aiMessage])

      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(response.session_id)
      }
    } catch (error) {
      console.error('Regular response failed:', error)
      addFallbackMessage()
    }
  }

  const addFallbackMessage = (error?: any) => {
    const fallbackMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: "I apologize, but I'm experiencing some technical difficulties. Please check your connection and try again.",
      sender: 'ai',
      timestamp: new Date(),
      metadata: {
        learningMode: 'fallback',
        concepts: ['error_handling'],
        confidence: 0.5,
        errorDetails: error?.message || 'Connection error'
      }
    }
    setMessages(prev => [...prev, fallbackMessage])
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="h-full flex flex-col relative overflow-hidden">
      {/* Quantum Background */}
      <div className="absolute inset-0 quantum-particles opacity-30">
        {[...Array(30)].map((_, i) => (
          <div
            key={i}
            className="particle"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`
            }}
          />
        ))}
      </div>

      {/* Enhanced Session Info Bar */}
      <div className="relative z-10 glass-morph-premium rounded-xl p-4 m-6 mb-4 border border-purple-500/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-cyan-600 flex items-center justify-center animate-quantum-pulse">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <div>
                <h3 className="intelligence-title text-white">Quantum Intelligence Session</h3>
                <p className="data-micro text-purple-400">Advanced AI Multi-Model Integration</p>
              </div>
            </div>
            {sessionId && (
              <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-purple-500/20 border border-purple-500/30">
                <MessageCircle className="h-3 w-3 text-purple-400" />
                <span className="data-micro text-purple-400 font-mono">ID: {sessionId.substring(0, 8)}...</span>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border ${
              connectionStatus === 'connected' 
                ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' 
                : connectionStatus === 'error' 
                ? 'bg-red-500/20 border-red-500/30 text-red-400' 
                : 'bg-amber-500/20 border-amber-500/30 text-amber-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-emerald-400 animate-quantum-pulse' :
                connectionStatus === 'error' ? 'bg-red-400' : 'bg-amber-400 animate-pulse'
              }`}></div>
              <span className="data-micro font-medium capitalize">{connectionStatus}</span>
            </div>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="neural-network-button p-2"
            >
              <Settings className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Settings Panel */}
      {showSettings && (
        <div className="relative z-10 glass-morph-premium rounded-xl p-6 mx-6 mb-4 border border-purple-500/30">
          <h4 className="intelligence-title text-purple-300 mb-4">AI Configuration</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <label className="precision-small text-gray-300 mb-2 block">Task Type</label>
              <select
                value={taskType}
                onChange={(e) => setTaskType(e.target.value)}
                className="w-full bg-gray-800/50 border border-purple-500/30 rounded-lg px-3 py-2 text-white focus-quantum"
              >
                {TASK_TYPE_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="precision-small text-gray-300 mb-2 block">AI Provider</label>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                className="w-full bg-gray-800/50 border border-purple-500/30 rounded-lg px-3 py-2 text-white focus-quantum"
              >
                {PROVIDER_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center justify-between">
              <label className="precision-small text-gray-300">Streaming Mode</label>
              <button
                onClick={() => setStreamingEnabled(!streamingEnabled)}
                className={`w-12 h-6 rounded-full transition-all duration-300 ${
                  streamingEnabled ? 'bg-purple-500' : 'bg-gray-600'
                }`}
              >
                <div className={`w-5 h-5 bg-white rounded-full transition-transform duration-300 ${
                  streamingEnabled ? 'transform translate-x-6' : 'transform translate-x-0.5'
                }`} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Revolutionary Messages Container */}
      <div className="flex-1 overflow-y-auto quantum-scroll relative z-10 px-6">
        <div className="max-w-4xl mx-auto space-y-6 py-4">
          {messages.map((message) => (
            <QuantumMessageBubble key={message.id} message={message} />
          ))}
          {(isLoading || isStreaming) && <QuantumLoadingIndicator />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Revolutionary Input Area */}
      <div className="relative z-10 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="glass-morph-premium rounded-xl p-4 border border-purple-500/30">
            <div className="flex space-x-4">
              <div className="flex space-x-2">
                <button className="neural-network-button p-2">
                  <Paperclip className="h-4 w-4" />
                </button>
                <button className="neural-network-button p-2">
                  <Camera className="h-4 w-4" />
                </button>
                <button className="neural-network-button p-2">
                  <Mic className="h-4 w-4" />
                </button>
              </div>

              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask me anything or explore quantum learning concepts..."
                className="flex-1 bg-transparent text-white placeholder-gray-400 border-none outline-none resize-none focus-quantum rounded-lg px-4 py-3"
                rows={3}
                disabled={isLoading || isStreaming}
              />

              <button
                onClick={handleSendMessage}
                disabled={isLoading || isStreaming || !inputMessage.trim()}
                className="quantum-button px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="flex items-center space-x-2">
                  <Send className="h-5 w-5" />
                  <span className="hidden md:inline">Send</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Revolutionary Message Bubble Component
function QuantumMessageBubble({ message }: { message: Message }) {
  return (
    <div className={cn(
      'flex space-x-4 animate-[fadeIn_0.5s_ease-in-out]',
      message.sender === 'user' ? 'justify-end' : 'justify-start'
    )}>
      {message.sender === 'ai' && (
        <div className="flex-shrink-0">
          <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-cyan-600 flex items-center justify-center animate-quantum-pulse">
            <Brain className="h-6 w-6 text-white" />
          </div>
        </div>
      )}
      
      <div className={cn(
        'max-w-3xl quantum-card relative',
        message.sender === 'user' 
          ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white' 
          : 'glass-morph-premium border-purple-500/20'
      )}>
        <div className="relative z-10">
          <p className="learning-body whitespace-pre-wrap leading-relaxed">{message.content}</p>
          
          {message.metadata && message.sender === 'ai' && (
            <div className="mt-4 pt-4 border-t border-purple-500/20">
              <div className="flex flex-wrap gap-2">
                {message.metadata.learningMode && (
                  <span className="data-micro px-3 py-1 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center space-x-1">
                    <Zap className="h-3 w-3" />
                    <span>{message.metadata.learningMode.replace('_', ' ')}</span>
                  </span>
                )}
                {message.metadata.intelligence_level && (
                  <span className="data-micro px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 flex items-center space-x-1">
                    <Brain className="h-3 w-3" />
                    <span>{message.metadata.intelligence_level}</span>
                  </span>
                )}
                {message.metadata.confidence && (
                  <span className="data-micro px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    {Math.round(message.metadata.confidence * 100)}% confidence
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
        
        <p className="data-micro text-gray-400 mt-3 opacity-70">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>

      {message.sender === 'user' && (
        <div className="flex-shrink-0">
          <div className="w-10 h-10 rounded-full bg-gradient-to-r from-gray-600 to-gray-700 flex items-center justify-center border border-purple-500/30">
            <User className="h-6 w-6 text-white" />
          </div>
        </div>
      )}
    </div>
  )
}

// Enhanced Loading Indicator
function QuantumLoadingIndicator() {
  return (
    <div className="flex space-x-4 justify-start">
      <div className="flex-shrink-0">
        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-cyan-600 flex items-center justify-center animate-quantum-pulse">
          <Brain className="h-6 w-6 text-white" />
        </div>
      </div>
      <div className="quantum-card max-w-xs">
        <div className="flex items-center space-x-3">
          <Loader2 className="h-5 w-5 animate-spin text-purple-400" />
          <span className="precision-small text-gray-300">AI is thinking...</span>
          <div className="flex space-x-1">
            {[...Array(3)].map((_, i) => (
              <div 
                key={i}
                className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" 
                style={{ animationDelay: `${i * 0.1}s` }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}