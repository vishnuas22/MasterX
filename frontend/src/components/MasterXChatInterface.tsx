'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, 
  Mic, 
  Paperclip, 
  Bot, 
  User, 
  Sparkles,
  Loader2,
  Brain,
  Zap,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Volume2,
  VolumeX,
  Settings,
  MoreHorizontal,
  ChevronRight,
  BarChart3
} from 'lucide-react'
import { API, ChatMessage, ChatRequest } from '@/lib/api-services'
import { cn } from '@/lib/utils'

// Speech Recognition types
declare global {
  interface Window {
    webkitSpeechRecognition: any
    SpeechRecognition: any
  }
}

interface MasterXChatInterfaceProps {
  className?: string
}

interface Message {
  id: string
  content: string
  type: 'user' | 'assistant'
  timestamp: string
  provider?: string
  model?: string
  isStreaming?: boolean
}

export function MasterXChatInterface({ className = '' }: MasterXChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState('')
  const [selectedProvider, setSelectedProvider] = useState<'groq' | 'gemini' | 'auto'>('auto')
  const [taskType, setTaskType] = useState<'general' | 'reasoning' | 'coding' | 'creative'>('general')
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingMessage])

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
      inputRef.current.style.height = `${inputRef.current.scrollHeight}px`
    }
  }, [inputMessage])

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage = inputMessage.trim()
    setInputMessage('')
    setIsLoading(true)

    // Add user message
    const userMsg: Message = {
      id: `user_${Date.now()}`,
      content: userMessage,
      type: 'user',
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMsg])

    try {
      // Simulate streaming response
      setStreamingMessage('')
      const response = await API.sendMessage({
        message: userMessage,
        session_id: `session_${Date.now()}`,
        provider: selectedProvider,
        task_type: taskType
      })

      // Add AI response
      const aiMsg: Message = {
        id: `ai_${Date.now()}`,
        content: response.response,
        type: 'assistant',
        timestamp: new Date().toISOString(),
        provider: response.provider,
        model: response.model
      }
      setMessages(prev => [...prev, aiMsg])

    } catch (error) {
      console.error('Failed to send message:', error)
      // Add error message
      const errorMsg: Message = {
        id: `error_${Date.now()}`,
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        type: 'assistant',
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
      setStreamingMessage('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleVoiceInput = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
      const recognition = new SpeechRecognition()
      
      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onstart = () => setIsListening(true)
      recognition.onend = () => setIsListening(false)
      
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setInputMessage(prev => prev + transcript)
      }

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
      }

      recognition.start()
    }
  }

  const handleFileUpload = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (files && files.length > 0) {
      // Handle file upload logic here
      console.log('Files selected:', files)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Chat Header */}
      <div className="h-16 glass-morph border-b border-purple-500/20 flex items-center justify-between px-quantum-6">
        <div className="flex items-center space-x-quantum-3">
          <motion.div
            className="relative"
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <Brain className="h-8 w-8 text-purple-400" />
            <div className="absolute inset-0 h-8 w-8 border border-cyan-400/30 rounded-full animate-ping" />
          </motion.div>
          <div>
            <h2 className="text-quantum-lg font-semibold text-plasma-white">Quantum Intelligence</h2>
            <p className="text-quantum-sm text-plasma-white/60">
              {isLoading ? 'Processing...' : 'Ready to assist'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-quantum-3">
          <div className="flex items-center space-x-quantum-2 text-quantum-sm text-plasma-white/60">
            <Zap className="h-4 w-4 text-cyan-400" />
            <span>{selectedProvider}</span>
          </div>
          <button className="p-quantum-2 rounded-lg glass-morph hover:bg-purple-500/20 transition-all duration-200">
            <Settings className="h-4 w-4 text-plasma-white/70" />
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-quantum-6">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="max-w-4xl mx-auto space-y-quantum-6">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onCopy={() => copyToClipboard(message.content)}
              />
            ))}

            {/* Streaming message */}
            {streamingMessage && (
              <MessageBubble
                message={{
                  id: 'streaming',
                  content: streamingMessage,
                  type: 'assistant',
                  timestamp: new Date().toISOString(),
                  isStreaming: true
                }}
                onCopy={() => copyToClipboard(streamingMessage)}
              />
            )}

            {/* Loading indicator */}
            {isLoading && !streamingMessage && (
              <div className="flex items-start space-x-quantum-4">
                <div className="w-quantum-10 h-quantum-10 rounded-full bg-gradient-to-r from-purple-600 to-cyan-600 flex items-center justify-center">
                  <Bot className="h-5 w-5 text-white" />
                </div>
                <div className="flex-1">
                  <div className="glass-morph rounded-2xl p-quantum-4 max-w-xs">
                    <div className="flex items-center space-x-quantum-3">
                      <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                      <span className="text-quantum-sm text-plasma-white/70">Quantum processing...</span>
                      <Sparkles className="h-4 w-4 text-cyan-400 animate-pulse" />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-purple-500/20 p-quantum-6">
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            <div className="flex items-end space-x-quantum-3">
              {/* File Upload */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleFileUpload}
                className="p-quantum-3 glass-morph rounded-xl hover:bg-purple-500/20 transition-all duration-200"
              >
                <Paperclip className="h-5 w-5 text-plasma-white/70" />
              </motion.button>

              {/* Text Input */}
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask MasterX anything..."
                  className="w-full resize-none glass-morph rounded-2xl px-quantum-6 py-quantum-4 pr-14 text-quantum-base text-plasma-white placeholder-plasma-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent transition-all max-h-32 min-h-[56px]"
                  rows={1}
                />

                {/* Send Button */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isLoading}
                  className={cn(
                    "absolute right-3 bottom-3 p-quantum-2 rounded-lg transition-all duration-200",
                    inputMessage.trim() && !isLoading
                      ? "bg-gradient-to-r from-purple-600 to-cyan-600 text-white hover:from-purple-700 hover:to-cyan-700"
                      : "glass-morph text-plasma-white/40 cursor-not-allowed"
                  )}
                >
                  <Send className="h-4 w-4" />
                </motion.button>
              </div>

              {/* Voice Input */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleVoiceInput}
                className={cn(
                  "p-quantum-3 rounded-xl transition-all duration-200",
                  isListening
                    ? "bg-red-500 text-white animate-pulse"
                    : "glass-morph hover:bg-purple-500/20 text-plasma-white/70"
                )}
              >
                <Mic className="h-5 w-5" />
              </motion.button>
            </div>
          </div>

          <p className="text-quantum-xs text-plasma-white/40 text-center mt-quantum-3">
            MasterX uses quantum intelligence to provide accurate responses. Always verify important information.
          </p>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,.pdf,.doc,.docx,.txt"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  )
}

// Modern Empty State Component
function EmptyState() {
  const suggestions = [
    {
      icon: Brain,
      title: "Explain quantum computing",
      description: "Learn about quantum mechanics and computing principles",
      category: "Science"
    },
    {
      icon: Zap,
      title: "Write Python code",
      description: "Generate and optimize code solutions",
      category: "Development"
    },
    {
      icon: BarChart3,
      title: "Analyze data patterns",
      description: "Extract insights from complex datasets",
      category: "Analytics"
    },
    {
      icon: Sparkles,
      title: "Creative brainstorming",
      description: "Generate innovative ideas and solutions",
      category: "Creative"
    }
  ]

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8 py-16">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="mb-16 max-w-3xl"
      >
        {/* Icon */}
        <motion.div
          className="w-16 h-16 mx-auto mb-8 rounded-2xl bg-gradient-primary flex items-center justify-center shadow-lg"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          whileHover={{ scale: 1.05 }}
        >
          <Brain className="h-8 w-8 text-white" />
        </motion.div>

        {/* Welcome Text */}
        <motion.h1
          className="text-display-xl mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          Welcome to{' '}
          <span className="bg-gradient-primary bg-clip-text text-transparent">
            MasterX
          </span>
        </motion.h1>

        <motion.p
          className="text-body-lg text-secondary max-w-2xl mx-auto leading-relaxed"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          Your AI assistant is ready to help. Choose a starting point below or ask me anything.
        </motion.p>
      </motion.div>

      {/* Suggestion Cards */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl w-full"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
      >
        {suggestions.map((suggestion, index) => (
          <motion.button
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.4,
              delay: 0.6 + index * 0.1
            }}
            whileHover={{
              scale: 1.02,
              y: -4,
              transition: { duration: 0.2 }
            }}
            whileTap={{ scale: 0.98 }}
            className="glass-modern rounded-xl p-6 text-left hover:glass-premium transition-all duration-300 group"
          >
            <div className="flex items-start space-x-4">
              {/* Icon */}
              <div className="w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
                <suggestion.icon className="h-5 w-5 text-white" />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-heading-sm text-primary group-hover:text-accent transition-colors">
                    {suggestion.title}
                  </h3>
                  <span className="text-caption text-quaternary bg-surface-tertiary px-2 py-1 rounded-md">
                    {suggestion.category}
                  </span>
                </div>
                <p className="text-body-sm text-secondary group-hover:text-primary transition-colors">
                  {suggestion.description}
                </p>
              </div>
            </div>
          </motion.button>
        ))}
      </motion.div>

      {/* Bottom CTA */}
      <motion.div
        className="mt-12 text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1 }}
      >
        <p className="text-body-sm text-tertiary">
          Start typing below to begin your conversation
        </p>
      </motion.div>
    </div>
  )
}

// Message Bubble Component
interface MessageBubbleProps {
  message: Message
  onCopy: () => void
}

function MessageBubble({ message, onCopy }: MessageBubbleProps) {
  const [showActions, setShowActions] = useState(false)
  const isUser = message.type === 'user'

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={cn(
        "flex items-start space-x-4",
        isUser ? "flex-row-reverse space-x-reverse" : ""
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Avatar */}
      <motion.div
        className={cn(
          "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0",
          isUser
            ? "bg-gradient-to-r from-cyan-500 to-purple-500"
            : "bg-gradient-to-r from-purple-600 to-cyan-600"
        )}
        whileHover={{ scale: 1.05 }}
      >
        {isUser ? (
          <User className="h-5 w-5 text-white" />
        ) : (
          <Bot className="h-5 w-5 text-white" />
        )}
      </motion.div>

      {/* Message Content */}
      <div className={cn(
        "flex-1 max-w-3xl",
        isUser ? "flex justify-end" : ""
      )}>
        <div className={cn(
          "rounded-2xl px-6 py-4 relative",
          isUser
            ? "bg-gradient-to-r from-cyan-600 to-purple-600 text-white ml-auto max-w-md"
            : "glass-morph text-plasma-white"
        )}>
          {/* Message Text */}
          <div className="prose prose-sm max-w-none">
            <p className="whitespace-pre-wrap break-words m-0 leading-relaxed">
              {message.content}
              {message.isStreaming && (
                <motion.span
                  className="inline-block w-2 h-5 bg-current ml-1"
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              )}
            </p>
          </div>

          {/* Timestamp and Provider */}
          <div className={cn(
            "text-xs mt-3 flex items-center justify-between",
            isUser ? "text-cyan-100" : "text-plasma-white/50"
          )}>
            <span>{formatTime(message.timestamp)}</span>
            {message.provider && (
              <div className="flex items-center space-x-1">
                <Zap className="h-3 w-3" />
                <span>{message.provider}</span>
              </div>
            )}
          </div>
        </div>

        {/* Message Actions */}
        {!isUser && (
          <AnimatePresence>
            {showActions && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="flex items-center space-x-2 mt-3"
              >
                <button
                  onClick={onCopy}
                  className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200"
                  title="Copy message"
                >
                  <Copy className="h-3 w-3 text-plasma-white/60" />
                </button>
                <button
                  className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200"
                  title="Good response"
                >
                  <ThumbsUp className="h-3 w-3 text-plasma-white/60" />
                </button>
                <button
                  className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200"
                  title="Poor response"
                >
                  <ThumbsDown className="h-3 w-3 text-plasma-white/60" />
                </button>
                <button
                  className="p-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all duration-200"
                  title="Regenerate"
                >
                  <RotateCcw className="h-3 w-3 text-plasma-white/60" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </div>
    </motion.div>
  )
}
