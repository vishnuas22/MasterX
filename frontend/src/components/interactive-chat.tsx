/**
 * Interactive Chat Component for MasterX Quantum Intelligence Platform
 * 
 * Full-featured chat interface with real-time messaging, multi-LLM support,
 * session management, and advanced AI capabilities.
 */

'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { 
  Send, 
  Mic, 
  Paperclip, 
  MoreVertical, 
  Bot, 
  User, 
  Zap, 
  Brain,
  Sparkles,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  Settings,
  Download,
  Copy,
  RefreshCw
} from 'lucide-react'
import { API, ChatMessage, ChatRequest, ChatResponse, ChatSession } from '@/lib/api-services'
import { QuantumDropdown, TASK_TYPE_OPTIONS, PROVIDER_OPTIONS } from '@/components/ui/quantum-dropdown'

interface InteractiveChatProps {
  sessionId?: string
  onSessionChange?: (session: ChatSession) => void
  className?: string
}

export function InteractiveChat({ sessionId, onSessionChange, className = '' }: InteractiveChatProps) {
  // State management
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedProvider, setSelectedProvider] = useState<'groq' | 'gemini' | 'auto'>('auto')
  const [taskType, setTaskType] = useState<'general' | 'reasoning' | 'coding' | 'creative'>('general')
  const [isListening, setIsListening] = useState(false)
  const [speechRecognition, setSpeechRecognition] = useState<any>(null)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Initialize session
  useEffect(() => {
    const initializeSession = async () => {
      try {
        // Create a simple session object for development
        const simpleSession = {
          session_id: sessionId || `session_${Date.now()}`,
          user_id: 'dev_user',
          created_at: new Date().toISOString(),
          last_activity: new Date().toISOString(),
          messages: [],
          context: {},
          learning_insights: {}
        }
        setCurrentSession(simpleSession)
        onSessionChange?.(simpleSession)
      } catch (error: any) {
        console.error('Failed to initialize session:', error)
        setError('Failed to initialize chat session')
      }
    }

    initializeSession()
  }, [sessionId, onSessionChange])

  // Send message function
  const sendMessage = async () => {
    console.log('🚀 SendMessage called:', { inputMessage, isLoading, currentSession })

    if (!inputMessage.trim() || isLoading) {
      console.log('❌ SendMessage early return:', { hasMessage: !!inputMessage.trim(), isLoading })
      return
    }

    const userMessage = inputMessage.trim()
    console.log('📤 Sending message:', userMessage)

    setInputMessage('')
    setIsLoading(true)
    setError(null)

    // Add user message to UI immediately
    const userMsg: ChatMessage = {
      message_id: `temp_${Date.now()}`,
      content: userMessage,
      message_type: 'user',
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const chatRequest: ChatRequest = {
        message: userMessage,
        session_id: currentSession?.session_id || `session_${Date.now()}`,
        task_type: taskType,
        provider: selectedProvider === 'auto' ? undefined : selectedProvider,
        stream: true
      }

      // Use regular API call instead of streaming for development
      setIsTyping(true)

      try {
        console.log('🌐 Making API call:', chatRequest)

        // Use the correct backend URL from environment
        const backendUrl = process.env.REACT_APP_BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001'
        const response = await fetch(`${backendUrl}/api/chat/send`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(chatRequest),
        })

        console.log('📡 API response status:', response.status)

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        console.log('✅ API response data:', data)

        // Add assistant response to messages
        const assistantMsg: ChatMessage = {
          message_id: data.message_id || `assistant_${Date.now()}`,
          content: data.response || 'I apologize, but I encountered an issue generating a response.',
          message_type: 'assistant',
          timestamp: new Date().toISOString()
        }

        console.log('💬 Adding assistant message:', assistantMsg)
        setMessages(prev => [...prev, assistantMsg])

      } catch (error) {
        console.error('Chat error:', error)
        setError('Failed to send message. Please try again.')

        // Add error message
        const errorMsg: ChatMessage = {
          message_id: `error_${Date.now()}`,
          content: 'Sorry, I encountered an error. Please try again.',
          message_type: 'assistant',
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev, errorMsg])
      }



    } catch (error: any) {
      console.error('Send message error:', error)
      setError('Failed to send message')

      // Remove the user message if sending failed
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setIsLoading(false)
      setIsTyping(false)
    }
  }

  // Handle input key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Voice chat functionality
  const toggleVoiceChat = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.')
      return
    }

    if (isListening) {
      // Stop listening
      if (speechRecognition) {
        speechRecognition.stop()
      }
      setIsListening(false)
    } else {
      // Start listening
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      const recognition = new SpeechRecognition()

      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onstart = () => {
        setIsListening(true)
        console.log('🎤 Voice recognition started')
      }

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript
        console.log('🗣️ Voice input:', transcript)
        setInputMessage(transcript)
        setIsListening(false)
      }

      recognition.onerror = (event: any) => {
        console.error('🚫 Voice recognition error:', event.error)
        setIsListening(false)
        alert(`Voice recognition error: ${event.error}`)
      }

      recognition.onend = () => {
        setIsListening(false)
        console.log('🎤 Voice recognition ended')
      }

      setSpeechRecognition(recognition)
      recognition.start()
    }
  }

  // File upload functionality
  const handleFileUpload = () => {
    // Create a file input element
    const fileInput = document.createElement('input')
    fileInput.type = 'file'
    fileInput.accept = 'image/*,text/*,.pdf,.doc,.docx'
    fileInput.multiple = false

    fileInput.onchange = (event: any) => {
      const file = event.target.files[0]
      if (file) {
        console.log('📎 File selected:', file.name, file.type, file.size)
        // For now, just show an alert - file upload can be implemented later
        alert(`File selected: ${file.name}\nType: ${file.type}\nSize: ${(file.size / 1024).toFixed(1)} KB\n\nFile upload functionality will be implemented in a future update.`)
      }
    }

    fileInput.click()
  }

  // Copy message content
  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  // Regenerate response
  const regenerateResponse = async (messageIndex: number) => {
    if (messageIndex < 1) return
    
    const userMessage = messages[messageIndex - 1]
    if (userMessage.message_type !== 'user') return

    // Remove the assistant message and regenerate
    setMessages(prev => prev.slice(0, messageIndex))
    setInputMessage(userMessage.content)
    await sendMessage()
  }

  return (
    <div className={`flex flex-col h-full bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 ${className}`}>
      {/* Chat Header */}
      <div className="glass-morph border-b border-purple-500/20 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Bot className="h-8 w-8 text-purple-400 animate-quantum-pulse" />
              <div className="absolute inset-0 h-8 w-8 border border-purple-400/30 rounded-full animate-spin" style={{ animationDuration: '8s' }} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
              <p className="text-sm text-purple-300">
                {isTyping ? 'Thinking...' : 'Ready to help'}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="w-48">
              <QuantumDropdown
                options={PROVIDER_OPTIONS}
                value={selectedProvider}
                onChange={(value) => setSelectedProvider(value as any)}
                placeholder="Select provider..."
                className="text-sm"
              />
            </div>

            <div className="w-48">
              <QuantumDropdown
                options={TASK_TYPE_OPTIONS}
                value={taskType}
                onChange={(value) => setTaskType(value as any)}
                placeholder="Select task type..."
                className="text-sm"
              />
            </div>

            <button className="p-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200">
              <Settings className="h-4 w-4 text-purple-300" />
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-3 p-3 bg-red-500/20 border border-red-500/30 rounded-lg flex items-center space-x-2">
            <AlertCircle className="h-4 w-4 text-red-400" />
            <span className="text-red-300 text-sm">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              ×
            </button>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="relative mb-6">
              <Brain className="h-16 w-16 text-purple-400 animate-quantum-pulse" />
              <Sparkles className="absolute -top-2 -right-2 h-6 w-6 text-cyan-400 animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Welcome to MasterX AI</h3>
            <p className="text-purple-300 max-w-md">
              Start a conversation with our quantum intelligence engine. Ask questions, get help with coding, 
              or explore creative ideas.
            </p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={message.message_id}
              className={`flex ${message.message_type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] ${message.message_type === 'user' ? 'order-2' : 'order-1'}`}>
                <div className={`flex items-start space-x-3 ${message.message_type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.message_type === 'user' 
                      ? 'bg-gradient-to-br from-purple-500 to-cyan-500' 
                      : 'bg-gradient-to-br from-slate-700 to-slate-600'
                  }`}>
                    {message.message_type === 'user' ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>

                  {/* Message Content */}
                  <div className={`glass-morph p-4 rounded-lg ${
                    message.message_type === 'user' 
                      ? 'bg-purple-500/20 border-purple-500/30' 
                      : 'bg-slate-800/30 border-slate-600/30'
                  }`}>
                    <div className="prose prose-invert max-w-none">
                      <p className="text-white whitespace-pre-wrap">{message.content}</p>
                    </div>

                    {/* Message Actions */}
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/10">
                      <span className="text-xs text-gray-400">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => copyMessage(message.content)}
                          className="p-1 rounded hover:bg-white/10 transition-colors"
                        >
                          <Copy className="h-3 w-3 text-gray-400" />
                        </button>
                        
                        {message.message_type === 'assistant' && (
                          <button
                            onClick={() => regenerateResponse(index)}
                            className="p-1 rounded hover:bg-white/10 transition-colors"
                          >
                            <RefreshCw className="h-3 w-3 text-gray-400" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-700 to-slate-600 flex items-center justify-center">
                <Bot className="h-4 w-4 text-white" />
              </div>
              <div className="glass-morph bg-slate-800/30 border-slate-600/30 p-4 rounded-lg">
                <div className="flex items-center space-x-2">
                  <Loader2 className="h-4 w-4 text-purple-400 animate-spin" />
                  <span className="text-purple-300">AI is thinking...</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="glass-morph border-t border-purple-500/20 p-4">
        <div className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="w-full bg-slate-800/50 border border-purple-500/30 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-400 focus:outline-none focus:border-purple-400 resize-none focus:text-white"
              rows={1}
              style={{
                minHeight: '44px',
                maxHeight: '120px',
                color: '#f1f5f9' // Ensure text is visible
              }}
              disabled={isLoading}
            />
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleFileUpload}
              className="p-3 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200"
              title="Upload file"
            >
              <Paperclip className="h-4 w-4 text-purple-300" />
            </button>

            <button
              onClick={toggleVoiceChat}
              className={`p-3 rounded-lg glass-morph transition-all duration-200 ${
                isListening
                  ? 'bg-red-500/20 hover:bg-red-500/30 border border-red-500/30'
                  : 'hover:bg-purple-500/20'
              }`}
              title={isListening ? "Stop voice input" : "Start voice input"}
            >
              <Mic className={`h-4 w-4 ${isListening ? 'text-red-400 animate-pulse' : 'text-purple-300'}`} />
            </button>

            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="p-3 rounded-lg bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 text-white animate-spin" />
              ) : (
                <Send className="h-4 w-4 text-white" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
