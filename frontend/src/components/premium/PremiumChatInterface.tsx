'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import {
  Send,
  Mic,
  MicOff,
  Paperclip,
  Settings,
  Volume2,
  VolumeX,
  Loader2,
  ArrowDown
} from 'lucide-react'
import EnhancedMessage from './EnhancedMessage'
import TypingIndicator from './TypingIndicator'
import ScrollToBottomButton from './ScrollToBottomButton'

interface MessageMetadata {
  model?: string
  provider?: string
  tokens_used?: number
  response_time_ms?: number
  confidence?: number
  learning_mode?: string
}

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  type?: 'text' | 'voice' | 'file'
  metadata?: MessageMetadata
  isStreaming?: boolean
}

export default function PremiumChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Welcome to MasterX Quantum Intelligence! I\'m your advanced AI assistant powered by quantum computing principles. How can I help you explore the frontiers of knowledge today?',
      sender: 'ai',
      timestamp: new Date(),
      metadata: {
        confidence: 0.98,
        response_time_ms: 100,
        model: 'gemini-2.0-flash-exp',
        provider: 'gemini',
        tokens_used: 45
      }
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const [unreadCount, setUnreadCount] = useState(0)
  const [isUserScrolling, setIsUserScrolling] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const lastScrollTop = useRef(0)

  // Intersection observer for scroll detection
  const { ref: bottomRef, inView: isBottomInView } = useInView({
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
  })

  // Smooth scroll to bottom
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
      })
    }
  }, [])

  // Handle scroll events
  const handleScroll = useCallback(() => {
    if (!messagesContainerRef.current) return

    const container = messagesContainerRef.current
    const scrollTop = container.scrollTop
    const scrollHeight = container.scrollHeight
    const clientHeight = container.clientHeight

    // Detect if user is scrolling up
    const isScrollingUp = scrollTop < lastScrollTop.current
    lastScrollTop.current = scrollTop

    // Show scroll button if not at bottom
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100
    setShowScrollButton(!isAtBottom && isScrollingUp)

    // Update user scrolling state
    setIsUserScrolling(!isAtBottom)

    // Reset unread count if at bottom
    if (isAtBottom) {
      setUnreadCount(0)
    }
  }, [])

  // Auto-scroll for new messages
  useEffect(() => {
    if (!isUserScrolling || isBottomInView) {
      scrollToBottom()
    } else {
      // Increment unread count for new AI messages
      const lastMessage = messages[messages.length - 1]
      if (lastMessage?.sender === 'ai' && !lastMessage.isStreaming) {
        setUnreadCount(prev => prev + 1)
      }
    }
  }, [messages, isUserScrolling, isBottomInView, scrollToBottom])

  // Scroll event listener
  useEffect(() => {
    const container = messagesContainerRef.current
    if (container) {
      container.addEventListener('scroll', handleScroll, { passive: true })
      return () => container.removeEventListener('scroll', handleScroll)
    }
  }, [handleScroll])

  // Enhanced message sending with streaming support
  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date()
    }

    const messageContent = inputValue
    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // Create streaming AI message
    const aiMessageId = Date.now().toString() + '_ai'
    const streamingMessage: Message = {
      id: aiMessageId,
      content: '',
      sender: 'ai',
      timestamp: new Date(),
      isStreaming: true
    }

    setMessages(prev => [...prev, streamingMessage])
    setStreamingMessageId(aiMessageId)

    try {
      const response = await fetch('http://localhost:8000/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageContent,
          session_id: 'quantum-session-1'
        }),
      })

      if (response.ok) {
        const data = await response.json()

        // Simulate streaming effect for better UX
        const fullContent = data.response
        const words = fullContent.split(' ')

        for (let i = 0; i < words.length; i++) {
          const partialContent = words.slice(0, i + 1).join(' ')

          setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId
              ? { ...msg, content: partialContent }
              : msg
          ))

          // Add delay for streaming effect
          await new Promise(resolve => setTimeout(resolve, 50))
        }

        // Final message with complete content and metadata
        setMessages(prev => prev.map(msg =>
          msg.id === aiMessageId
            ? {
                ...msg,
                content: fullContent,
                metadata: data.metadata,
                isStreaming: false
              }
            : msg
        ))
      } else {
        throw new Error('Failed to send message')
      }
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: Date.now().toString() + '_error',
        content: 'I apologize, but I\'m experiencing some technical difficulties. Please try again in a moment.',
        sender: 'ai',
        timestamp: new Date(),
        metadata: {
          confidence: 0.0,
          response_time_ms: 0,
          model: 'Error Handler',
          provider: 'system'
        }
      }
      setMessages(prev => [...prev.slice(0, -1), errorMessage])
    } finally {
      setIsLoading(false)
      setStreamingMessageId(null)
    }
  }

  // Message actions
  const handleRegenerate = useCallback((messageId: string) => {
    // Find the user message before this AI message
    const messageIndex = messages.findIndex(msg => msg.id === messageId)
    if (messageIndex > 0) {
      const userMessage = messages[messageIndex - 1]
      if (userMessage.sender === 'user') {
        // Remove the AI message and regenerate
        setMessages(prev => prev.filter(msg => msg.id !== messageId))
        setInputValue(userMessage.content)
        setTimeout(() => sendMessage(), 100)
      }
    }
  }, [messages])

  const handleEditMessage = useCallback((content: string) => {
    setInputValue(content)
  }, [])

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const toggleRecording = () => {
    setIsRecording(!isRecording)
  }

  const toggleSpeaking = () => {
    setIsSpeaking(!isSpeaking)
  }

  const handleFileUpload = () => {
    fileInputRef.current?.click()
  }

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      console.log('File selected:', file.name)
    }
  }

  const handleScrollToBottom = () => {
    setUnreadCount(0)
    setIsUserScrolling(false)
    scrollToBottom()
  }

  return (
    <div className="h-full flex flex-col relative overflow-hidden">
      {/* Chat Container */}
      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full p-6 min-h-0">
        {/* Messages Area */}
        <motion.div
          ref={messagesContainerRef}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="flex-1 overflow-y-auto mb-6 pr-2 scroll-smooth chat-container min-h-0"
          style={{ scrollBehavior: 'smooth' }}
        >
          <motion.div
            className="space-y-8 py-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <EnhancedMessage
                  key={message.id}
                  id={message.id}
                  content={message.content}
                  sender={message.sender}
                  timestamp={message.timestamp}
                  metadata={message.metadata}
                  isStreaming={message.isStreaming}
                  onRegenerate={() => handleRegenerate(message.id)}
                  onEdit={handleEditMessage}
                />
              ))}
            </AnimatePresence>
            {/* Typing Indicator */}
            {isLoading && (
              <TypingIndicator message="Quantum processing your request..." />
            )}
            <div ref={messagesEndRef} />
            <div ref={bottomRef} className="h-1" />
          </motion.div>
        </motion.div>

        {/* Input Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-gradient-to-r from-black/30 to-gray-900/30 backdrop-blur-2xl border border-white/25 rounded-3xl p-6 shadow-2xl relative z-10 flex-shrink-0 hover:border-white/30 transition-all duration-300"
        >
          <div className="flex items-end space-x-4">
            {/* Text Input */}
            <div className="flex-1">
              <motion.textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about quantum intelligence, technology, or any topic you're curious about..."
                whileFocus={{ scale: 1.01 }}
                transition={{ duration: 0.2 }}
                className="
                  w-full px-6 py-4 bg-gradient-to-br from-white/8 to-white/12 border border-white/25 rounded-2xl
                  text-white placeholder-gray-400 resize-none backdrop-blur-xl
                  focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50
                  transition-all duration-300 shadow-xl hover:shadow-2xl hover:border-white/30
                  hover:from-white/10 hover:to-white/14
                "
                rows={1}
                style={{ minHeight: '56px', maxHeight: '140px' }}
              />
            </div>

            {/* Action Buttons */}
            <div className="flex items-center space-x-3">
              {/* Voice Recording */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleRecording}
                className={`
                  p-4 rounded-2xl transition-all duration-300 shadow-xl backdrop-blur-xl
                  ${isRecording
                    ? 'bg-gradient-to-br from-red-500 to-red-600 text-white animate-pulse shadow-red-500/30'
                    : 'bg-gradient-to-br from-white/10 to-white/15 text-gray-400 hover:text-white hover:from-white/15 hover:to-white/20 border border-white/20'
                  }
                `}
              >
                {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </motion.button>

              {/* File Upload */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleFileUpload}
                className="p-4 bg-gradient-to-br from-white/10 to-white/15 text-gray-400 hover:text-white hover:from-white/15 hover:to-white/20 rounded-2xl transition-all duration-300 shadow-xl backdrop-blur-xl border border-white/20"
              >
                <Paperclip className="w-5 h-5" />
              </motion.button>

              {/* Send Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="
                  p-4 bg-gradient-to-r from-purple-500 to-cyan-500 text-white rounded-2xl
                  disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
                  transition-all duration-300 shadow-xl backdrop-blur-xl
                  hover:from-purple-600 hover:to-cyan-600 hover:shadow-2xl
                  shadow-purple-500/20
                "
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </motion.button>
            </div>
          </div>

          {/* Settings Panel */}
          <AnimatePresence>
            {showSettings && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 pt-4 border-t border-white/10"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Text-to-Speech</span>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={toggleSpeaking}
                    className={`
                      p-2 rounded-lg transition-colors
                      ${isSpeaking 
                        ? 'bg-purple-500 text-white' 
                        : 'bg-white/10 text-gray-400 hover:text-white'
                      }
                    `}
                  >
                    {isSpeaking ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                  </motion.button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Scroll to Bottom Button */}
      <ScrollToBottomButton
        visible={showScrollButton}
        onClick={handleScrollToBottom}
        unreadCount={unreadCount}
      />

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        onChange={onFileSelect}
        className="hidden"
        accept="*/*"
      />
    </div>
  )
}
