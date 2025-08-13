'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  Brain,
  Send,
  Paperclip,
  Mic,
  MicOff,
  Plus,
  Settings,
  History,
  ArrowLeft,
  BarChart3,
  Wand2,
  GitBranch,
  Glasses,
  Eye,
  Hand,
  Heart,
  Link as LinkIcon,
  Building,
  Code,
  Lightbulb,
  Loader2,
  User,
  Bot
} from 'lucide-react'
import { apiClient, formatTimestamp, generateSessionTitle } from '../../lib/api'
import MarkdownRenderer from '../../components/MarkdownRenderer'
import { motion, AnimatePresence } from 'framer-motion'
import { messageAnimations, messageListAnimation, typingIndicatorAnimation, buttonAnimations, inputAnimations, iconButtonAnimations } from '../../lib/animations'
import { LazyVirtualizedMessageList } from '../../components/LazyComponents'
import { VirtualizedMessageListRef } from '../../components/VirtualizedMessageList'
import { generateTestMessages, performanceMonitor, FPSMonitor, testScrollPerformance } from '../../lib/performanceUtils'
import { useChatSessions, useFlattenedMessages, useSendMessage, useBackgroundSync, useCacheWarming, useOfflineMessageQueue } from '../../hooks/useChatQueries'
import VoiceNavigation, { VoiceStatusIndicator } from '../../components/VoiceNavigation'
import { useChatState, useChatActions, useUIState, useUIActions, useUserState } from '../../store'
import FileUpload from '../../components/FileUpload'
import { CollaborationPanel, CollaborationStatusBar } from '../../components/Collaboration'
import { useTypingIndicator } from '../../hooks/useWebSocket'
import AnalyticsDashboard from '../../components/Analytics'
import QuantumIntelligence from '../../components/QuantumIntelligence'
import CreativeAI from '../../components/CreativeAI'
import DecisionSupport from '../../components/DecisionSupport'
import ARInterface from '../../components/ARInterface'
import AdvancedInput from '../../components/AdvancedInput'
import GestureRecognition from '../../components/GestureRecognition'
import EmotionRecognition from '../../components/EmotionRecognition'
import APIIntegration from '../../components/APIIntegration'
import EnterpriseCloud from '../../components/EnterpriseCloud'

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  isStreaming?: boolean
}

interface ChatSession {
  session_id: string
  title: string
  created_at: string
  message_count: number
}

export default function ChatPage() {
  // Zustand store state with safe defaults
  const chatState = useChatState()
  const chatActions = useChatActions()
  const uiState = useUIState()
  const uiActions = useUIActions()
  const userState = useUserState()

  // Verify all store hooks returned valid objects
  if (!chatState || !chatActions || !uiState || !uiActions || !userState) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Initializing MasterX...</p>
        </div>
      </div>
    )
  }

  // Safe destructuring with multiple fallback levels
  const safeDefaults = {
    currentSessionId: null,
    currentMessages: [],
    sessions: [],
    isLoading: false,
    inputMessage: '',
    sidebarOpen: false,
  }

  // Ensure chatState is an object before destructuring
  const safeChatState = chatState && typeof chatState === 'object' ? chatState : safeDefaults

  const {
    currentSessionId,
    currentMessages,
    sessions,
    isLoading,
    inputMessage,
    sidebarOpen,
  } = safeChatState

  // React Query hooks for data management (keeping for now, will migrate gradually)
  let sessionsData, sessionsLoading, messages, messagesLoading, fetchNextPage, hasNextPage, sendMessageMutation, warmCache, queueMessage, offlineQueue

  try {
    const sessionsResult = useChatSessions() || { data: null, isLoading: false }
    sessionsData = sessionsResult.data
    sessionsLoading = sessionsResult.isLoading

    const messagesResult = useFlattenedMessages(currentSessionId) || {
      messages: [],
      isLoading: false,
      fetchNextPage: () => {},
      hasNextPage: false
    }
    messages = messagesResult.messages
    messagesLoading = messagesResult.isLoading
    fetchNextPage = messagesResult.fetchNextPage
    hasNextPage = messagesResult.hasNextPage

    sendMessageMutation = useSendMessage(currentSessionId) || { mutate: () => {}, isLoading: false }
    const cacheResult = useCacheWarming() || { warmCache: () => {} }
    warmCache = cacheResult.warmCache
    const queueResult = useOfflineMessageQueue() || { queueMessage: () => {}, offlineQueue: [] }
    queueMessage = queueResult.queueMessage
    offlineQueue = queueResult.offlineQueue
  } catch (error) {
    console.error('ChatPage: React Query hook error:', error)
    // Provide fallbacks
    sessionsData = null
    sessionsLoading = false
    messages = []
    messagesLoading = false
    fetchNextPage = () => {}
    hasNextPage = false
    sendMessageMutation = { mutate: () => {}, isLoading: false }
    warmCache = () => {}
    queueMessage = () => {}
    offlineQueue = []
  }

  // Local UI state
  const [showFileUpload, setShowFileUpload] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [showQuantumIntelligence, setShowQuantumIntelligence] = useState(false)
  const [showCreativeAI, setShowCreativeAI] = useState(false)
  const [showDecisionSupport, setShowDecisionSupport] = useState(false)
  const [showARInterface, setShowARInterface] = useState(false)
  const [showAdvancedInput, setShowAdvancedInput] = useState(false)
  const [showGestureRecognition, setShowGestureRecognition] = useState(false)
  const [showEmotionRecognition, setShowEmotionRecognition] = useState(false)
  const [showAPIIntegration, setShowAPIIntegration] = useState(false)
  const [showEnterpriseCloud, setShowEnterpriseCloud] = useState(false)

  // Collaboration features - with safe error handling
  let startTyping, stopTyping, typingUsers
  try {
    const typingResult = useTypingIndicator(currentSessionId || undefined)
    if (typingResult && typeof typingResult === 'object') {
      startTyping = typingResult.startTyping || (() => {})
      stopTyping = typingResult.stopTyping || (() => {})
      typingUsers = typingResult.typingUsers || []
    } else {
      throw new Error('useTypingIndicator returned invalid result')
    }
  } catch (error) {
    console.warn('useTypingIndicator failed, using fallbacks:', error)
    startTyping = () => {}
    stopTyping = () => {}
    typingUsers = []
  }

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const virtualizedListRef = useRef<VirtualizedMessageListRef>(null)
  const fpsMonitorRef = useRef<FPSMonitor | null>(null)

  // Enable background sync for active session
  useBackgroundSync(currentSessionId, !uiState?.performanceMode) // Disable during performance testing

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    if (virtualizedListRef.current) {
      virtualizedListRef.current.scrollToBottom()
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [])

  useEffect(() => {
    // Small delay to ensure virtualized list is ready
    const timer = setTimeout(() => {
      scrollToBottom()
    }, 100)
    
    return () => clearTimeout(timer)
  }, [messages, scrollToBottom])

  // Warm cache on mount
  useEffect(() => {
    warmCache()
  }, [warmCache])

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const content = inputMessage.trim()

    // Use store action to send message
    try {
      await chatActions.sendMessage(content)
    } catch (error) {
      console.error('Error sending message:', error)
      uiActions.addNotification({
        type: 'error',
        title: 'Message Failed',
        message: 'Failed to send message. Please try again.',
        duration: 5000,
      })
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const startNewChat = () => {
    chatActions.setCurrentSession(null)
    chatActions.setSidebarOpen(false)
    uiActions.setPerformanceMode(false)
  }

  const loadSession = (session: ChatSession) => {
    chatActions.setCurrentSession(session.session_id)
    chatActions.setSidebarOpen(false)
    uiActions.setPerformanceMode(false)
  }

  // Performance testing functions
  const loadTestMessages = (count: number) => {
    console.log(`🧪 Loading ${count} test messages for performance testing...`)
    performanceMonitor.startMeasurement('loadMessages')

    const testMessages = generateTestMessages(count)

    // Create a temporary session for testing
    const testSessionId = `test-session-${Date.now()}`
    chatActions.setCurrentSession(testSessionId)

    // Manually set test messages in cache (bypassing API)
    const queryClient = (window as any).queryCache?.client
    if (queryClient) {
      queryClient.setQueryData(['chat', 'session', testSessionId, 'messages'], {
        pages: [{ messages: testMessages, hasMore: false }],
        pageParams: [0],
      })
    }

    const loadTime = performanceMonitor.endMeasurement('loadMessages')
    console.log(`✅ Loaded ${count} messages in ${loadTime.toFixed(2)}ms`)

    // Start FPS monitoring
    if (!fpsMonitorRef.current) {
      fpsMonitorRef.current = new FPSMonitor()
    }
    fpsMonitorRef.current.start()

    uiActions.setPerformanceMode(true)
  }

  const runScrollPerformanceTest = async () => {
    if (!virtualizedListRef.current) {
      console.warn('Virtual list not available for testing')
      return
    }

    console.log('🚀 Starting scroll performance test...')

    // Find the scrollable element
    const listElement = document.querySelector('.scrollbar-hide') as HTMLElement
    if (!listElement) {
      console.warn('Scrollable element not found')
      return
    }

    try {
      const results = await testScrollPerformance(listElement, 5000)
      console.log('📊 Scroll Performance Results:', results)

      // Show results in UI (you could create a modal for this)
      alert(`Scroll Performance Test Results:
Average FPS: ${results.averageFPS}
Memory Usage: ${results.memoryUsage.difference || 'N/A'}MB
Test Duration: ${results.duration}ms`)
    } catch (error) {
      console.error('Performance test failed:', error)
    }
  }

  return (
    <div className="h-screen bg-white flex overflow-hidden">
      {/* Voice Status Indicator */}
      <VoiceStatusIndicator />

      {/* Sidebar */}
      <nav
        id="navigation"
        role="navigation"
        aria-label="Chat navigation and settings"
        className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed left-0 top-0 h-full w-80 bg-slate-50 border-r border-slate-200 z-50 lg:relative lg:translate-x-0 lg:z-auto flex flex-col transition-all duration-300 shadow-lg`}
      >
        
        {/* Header */}
        <header className="flex items-center justify-between p-6 border-b border-slate-200">
          <Link href="/" className="flex items-center space-x-3" aria-label="MasterX home page">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl flex items-center justify-center shadow-lg" role="img" aria-label="MasterX logo">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-slate-900 text-lg">MasterX</h1>
              <p className="text-xs text-slate-500 font-medium">Quantum Intelligence</p>
            </div>
          </Link>
        </header>

        {/* New Chat Button */}
        <div className="p-6">
          <motion.button
            onClick={startNewChat}
            data-testid="new-chat-button"
            className="w-full enterprise-button enterprise-button-primary mb-4"
            variants={buttonAnimations}
            initial="idle"
            whileHover="hover"
            whileTap="tap"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Conversation
          </motion.button>

          {/* Voice Navigation */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Voice Navigation
            </h4>
            <VoiceNavigation showCommands={false} />
          </div>

          {/* Performance Testing Buttons (Development Only) */}
          {process.env.NODE_ENV === 'development' && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Performance Testing
              </h4>

              <motion.button
                onClick={() => loadTestMessages(100)}
                className="w-full text-xs px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
                variants={buttonAnimations}
                initial="idle"
                whileHover="hover"
                whileTap="tap"
              >
                Load 100 Messages
              </motion.button>

              <motion.button
                onClick={() => loadTestMessages(1000)}
                className="w-full text-xs px-3 py-2 bg-orange-100 hover:bg-orange-200 text-orange-700 rounded-lg transition-colors"
                variants={buttonAnimations}
                initial="idle"
                whileHover="hover"
                whileTap="tap"
              >
                Load 1,000 Messages
              </motion.button>

              <motion.button
                onClick={() => loadTestMessages(10000)}
                className="w-full text-xs px-3 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors"
                variants={buttonAnimations}
                initial="idle"
                whileHover="hover"
                whileTap="tap"
              >
                Load 10,000 Messages
              </motion.button>

              <motion.button
                onClick={runScrollPerformanceTest}
                className="w-full text-xs px-3 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors"
                variants={buttonAnimations}
                initial="idle"
                whileHover="hover"
                whileTap="tap"
                disabled={messages.length === 0}
              >
                Test Scroll Performance
              </motion.button>

              {uiState?.performanceMode && (
                <div className="text-xs text-slate-600 bg-slate-100 p-2 rounded">
                  Performance Mode Active
                  <br />
                  {messages.length} messages loaded
                  <br />
                  {offlineQueue.length > 0 && `${offlineQueue.length} queued offline`}
                  <br />
                  Check console for metrics
                  <br />
                  <button
                    onClick={() => (window as any).queryCache?.log()}
                    className="text-blue-600 underline"
                  >
                    Log Cache
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Chat History */}
        <div className="flex-1 px-6 pb-6 overflow-y-auto">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
            Recent Conversations
          </h3>
          <div className="space-y-2">
            {sessions.map((session) => (
              <button
                key={session.session_id}
                onClick={() => loadSession(session)}
                className="w-full text-left p-3 rounded-xl hover:bg-slate-100 transition-colors group"
              >
                <div className="text-sm font-medium text-slate-900 truncate mb-1">
                  {session.title || 'Untitled Conversation'}
                </div>
                <div className="text-xs text-slate-500">
                  {session.message_count} messages
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Collaboration Panel */}
        <div className="mt-auto p-4 border-t border-slate-200">
          <CollaborationPanel sessionId={currentSessionId || undefined} />
        </div>
      </nav>

      {/* Mobile Menu Button */}
      <motion.button
        onClick={() => chatActions.setSidebarOpen(!sidebarOpen)}
        className="fixed top-6 left-6 z-50 p-3 bg-white border border-slate-200 rounded-xl shadow-lg lg:hidden"
        variants={buttonAnimations}
        initial="idle"
        whileHover="hover"
        whileTap="tap"
      >
        <motion.div
          animate={{ rotate: sidebarOpen ? 0 : 180 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
        >
          {sidebarOpen ? <ArrowLeft className="h-5 w-5" /> : <History className="h-5 w-5" />}
        </motion.div>
      </motion.button>

      {/* Main Chat Area */}
      <main
        id="main-content"
        role="main"
        aria-label="Chat conversation"
        className="flex-1 flex flex-col"
      >
        {/* Chat Header */}
        <header className="border-b border-slate-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-slate-900">Quantum Intelligence Chat</h2>
              <p className="text-sm text-slate-500">Powered by advanced AI algorithms</p>
            </div>
            <button
              onClick={() => setShowQuantumIntelligence(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Quantum Intelligence"
            >
              <Brain className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowCreativeAI(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Creative AI Studio"
            >
              <Wand2 className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowDecisionSupport(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Decision Support System"
            >
              <GitBranch className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowARInterface(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Augmented Reality"
            >
              <Glasses className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowAdvancedInput(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Advanced Input Methods"
            >
              <Eye className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowGestureRecognition(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Gesture Recognition"
            >
              <Hand className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowEmotionRecognition(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Emotion Recognition"
            >
              <Heart className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowAPIIntegration(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="API Integration & Plugins"
            >
              <LinkIcon className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowEnterpriseCloud(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Enterprise & Cloud Services"
            >
              <Building className="h-5 w-5 text-slate-600" />
            </button>

            <button
              onClick={() => setShowAnalytics(true)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              title="Analytics Dashboard"
            >
              <BarChart3 className="h-5 w-5 text-slate-600" />
            </button>

            <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
              <Settings className="h-5 w-5 text-slate-600" />
            </button>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center p-6">
              <motion.div 
                className="text-center"
                variants={messageListAnimation}
                initial="hidden"
                animate="visible"
              >
                <div className="w-20 h-20 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-6">
                  <Brain className="h-10 w-10 text-indigo-600" />
                </div>
                <h3 className="text-2xl font-bold text-slate-900 mb-4">Welcome to MasterX</h3>
                <p className="text-lg text-slate-600 mb-8 max-w-md mx-auto">
                  Start a conversation with our quantum intelligence AI. Ask questions, explore ideas, or get help with complex problems.
                </p>
                
                <motion.div 
                  className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto"
                  variants={messageListAnimation}
                  initial="hidden"
                  animate="visible"
                >
                  <motion.button
                    onClick={() => chatActions.setInputMessage("Explain quantum computing in simple terms")}
                    className="p-6 bg-white border border-slate-200 rounded-2xl hover:border-indigo-300 hover:shadow-lg transition-all duration-200 text-left group"
                    variants={messageAnimations}
                    whileHover={{ scale: 1.02, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className="w-10 h-10 rounded-xl bg-indigo-100 flex items-center justify-center group-hover:bg-indigo-200 transition-colors">
                        <Lightbulb className="h-5 w-5 text-indigo-600" />
                      </div>
                      <span className="font-semibold text-slate-900">Explain Concepts</span>
                    </div>
                    <p className="text-sm text-slate-600">Get clear explanations of complex topics</p>
                  </motion.button>
                  
                  <motion.button
                    onClick={() => chatActions.setInputMessage("Write a Python function to analyze data")}
                    className="p-6 bg-white border border-slate-200 rounded-2xl hover:border-blue-300 hover:shadow-lg transition-all duration-200 text-left group"
                    variants={messageAnimations}
                    whileHover={{ scale: 1.02, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                        <Code className="h-5 w-5 text-blue-600" />
                      </div>
                      <span className="font-semibold text-slate-900">Code Generation</span>
                    </div>
                    <p className="text-sm text-slate-600">Generate and optimize code solutions</p>
                  </motion.button>
                </motion.div>
              </motion.div>
            </div>
          ) : (
            <LazyVirtualizedMessageList
              ref={virtualizedListRef}
              messages={messages}
              isLoading={isLoading}
              className="h-full"
            />
          )}
        </div>

        {/* Input Area */}
        <section
          id="chat-input"
          aria-label="Message input area"
          className="border-t border-slate-200 p-6"
        >
          <div className="max-w-4xl mx-auto">
            <motion.div 
              className="bg-white border border-slate-300 rounded-2xl shadow-lg p-4"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            >
              <div className="flex items-end space-x-4">
                <div className="flex space-x-2">
                  <motion.button 
                    className="p-3 hover:bg-slate-100 rounded-xl transition-colors"
                    variants={iconButtonAnimations}
                    initial="idle"
                    whileHover="hover"
                    whileTap="tap"
                  >
                    <Paperclip className="h-5 w-5 text-slate-600" />
                  </motion.button>
                  <motion.button 
                    className="p-3 hover:bg-slate-100 rounded-xl transition-colors"
                    variants={iconButtonAnimations}
                    initial="idle"
                    whileHover="hover"
                    whileTap="tap"
                  >
                    <Mic className="h-5 w-5 text-slate-600" />
                  </motion.button>
                </div>
                
                <motion.textarea
                  ref={inputRef}
                  data-testid="message-input"
                  value={inputMessage}
                  onChange={(e) => {
                    chatActions.setInputMessage(e.target.value)
                    startTyping()
                  }}
                  onBlur={stopTyping}
                  onKeyDown={handleKeyPress}
                  placeholder="Message MasterX..."
                  aria-label="Type your message to MasterX"
                  aria-describedby="input-help"
                  aria-required="false"
                  className="flex-1 bg-transparent text-slate-900 placeholder-slate-400 border-none outline-none resize-none rounded-lg px-4 py-3 text-base leading-relaxed focus:outline-none"
                  rows={1}
                  disabled={isLoading}
                  style={{ minHeight: '44px', maxHeight: '120px' }}
                  whileFocus={{ scale: 1.01 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />

                {/* File Upload Button */}
                <button
                  onClick={() => setShowFileUpload(!showFileUpload)}
                  className="p-3 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-xl transition-all duration-200"
                  title="Attach files"
                  aria-label="Attach files"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                </button>

                <motion.button
                  onClick={sendMessage}
                  data-testid="send-button"
                  disabled={!inputMessage.trim() || isLoading}
                  aria-label={isLoading ? 'Sending message...' : 'Send message'}
                  aria-describedby="send-help"
                  className={`p-3 rounded-xl transition-all duration-200 ${
                    inputMessage.trim() && !isLoading
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white shadow-lg'
                      : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  }`}
                  variants={buttonAnimations}
                  initial="idle"
                  whileHover={inputMessage.trim() && !isLoading ? "hover" : "idle"}
                  whileTap={inputMessage.trim() && !isLoading ? "tap" : "idle"}
                >
                  <Send className="h-5 w-5" />
                </motion.button>
              </div>
            </motion.div>

            {/* Input Help Text */}
            <div className="mt-2 text-center">
              <p id="input-help" className="text-xs text-slate-500">
                Press Enter to send, Shift+Enter for new line
              </p>
              <p id="send-help" className="text-xs text-slate-500">
                {isLoading ? 'Processing your message...' : 'Click to send your message to MasterX'}
              </p>
            </div>

            {/* File Upload Panel */}
            {showFileUpload && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 p-4 bg-slate-50 rounded-lg border border-slate-200"
              >
                <FileUpload
                  maxFiles={5}
                  maxFileSize={25}
                  onFileSelect={(files) => {
                    console.log('Files selected:', files)
                    uiActions.addNotification({
                      type: 'info',
                      title: 'Files Selected',
                      message: `${files.length} file(s) ready to upload`,
                      duration: 2000,
                    })
                  }}
                  onFileUpload={(files) => {
                    console.log('Files uploaded:', files)
                    uiActions.addNotification({
                      type: 'success',
                      title: 'Upload Complete',
                      message: 'Files are ready to be processed by AI',
                      duration: 3000,
                    })
                  }}
                  showPreview={true}
                />
              </motion.div>
            )}
          </div>
        </section>

        {/* Collaboration Status Bar */}
        <CollaborationStatusBar sessionId={currentSessionId || undefined} />
      </main>

      {/* Analytics Dashboard Modal */}
      <AnimatePresence>
        {showAnalytics && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowAnalytics(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <AnalyticsDashboard sessionId={currentSessionId || undefined} />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowAnalytics(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quantum Intelligence Modal */}
      <AnimatePresence>
        {showQuantumIntelligence && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowQuantumIntelligence(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-7xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <QuantumIntelligence
                query="How can we enhance the MasterX AI platform with breakthrough features?"
                onSolutionSelect={(solution) => {
                  console.log('Selected solution:', solution)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Solution Selected',
                    message: `Selected: ${solution.title}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowQuantumIntelligence(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Creative AI Modal */}
      <AnimatePresence>
        {showCreativeAI && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowCreativeAI(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <CreativeAI
                onIdeaGenerated={(idea) => {
                  console.log('Idea generated:', idea)
                  uiActions.addNotification({
                    type: 'success',
                    title: 'Creative Idea Generated',
                    message: `New idea: ${idea.title}`,
                    duration: 3000,
                  })
                }}
                onWorkflowCreated={(workflow) => {
                  console.log('Workflow created:', workflow)
                  uiActions.addNotification({
                    type: 'success',
                    title: 'Workflow Created',
                    message: `New workflow: ${workflow.name}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowCreativeAI(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Decision Support Modal */}
      <AnimatePresence>
        {showDecisionSupport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowDecisionSupport(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-7xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <DecisionSupport
                decision="Should we prioritize advanced AI features or focus on performance optimization?"
                onDecisionMade={(decision) => {
                  console.log('Decision made:', decision)
                  uiActions.addNotification({
                    type: 'success',
                    title: 'Decision Made',
                    message: `Selected: ${decision.selectedOption.title}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowDecisionSupport(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* AR Interface Modal */}
      <AnimatePresence>
        {showARInterface && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowARInterface(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <ARInterface
                onARStart={() => {
                  console.log('AR session started')
                  uiActions.addNotification({
                    type: 'success',
                    title: 'AR Session Started',
                    message: 'Augmented reality is now active',
                    duration: 3000,
                  })
                }}
                onAREnd={() => {
                  console.log('AR session ended')
                  uiActions.addNotification({
                    type: 'info',
                    title: 'AR Session Ended',
                    message: 'Augmented reality session has ended',
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowARInterface(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Advanced Input Modal */}
      <AnimatePresence>
        {showAdvancedInput && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowAdvancedInput(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <AdvancedInput
                onEyeTracking={(data) => {
                  console.log('Eye tracking data:', data)
                }}
                onGestureDetected={(gesture) => {
                  console.log('Gesture detected:', gesture)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Gesture Detected',
                    message: `${gesture.type} gesture detected`,
                    duration: 2000,
                  })
                }}
                onEmotionDetected={(emotion) => {
                  console.log('Emotion detected:', emotion)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Emotion Detected',
                    message: `${emotion.primary} emotion detected`,
                    duration: 2000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowAdvancedInput(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Gesture Recognition Modal */}
      <AnimatePresence>
        {showGestureRecognition && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowGestureRecognition(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <GestureRecognition
                onGestureCommand={(command) => {
                  console.log('Gesture command:', command)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Gesture Command',
                    message: `${command.action} gesture executed`,
                    duration: 2000,
                  })
                }}
                onCalibrationComplete={() => {
                  console.log('Gesture calibration complete')
                  uiActions.addNotification({
                    type: 'success',
                    title: 'Calibration Complete',
                    message: 'Gesture recognition has been calibrated',
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowGestureRecognition(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Emotion Recognition Modal */}
      <AnimatePresence>
        {showEmotionRecognition && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowEmotionRecognition(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <EmotionRecognition
                onEmotionChange={(emotion) => {
                  console.log('Emotion changed:', emotion)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Emotion Detected',
                    message: `${emotion.primary} emotion detected`,
                    duration: 2000,
                  })
                }}
                onMoodAnalysis={(analysis) => {
                  console.log('Mood analysis:', analysis)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Mood Analysis Updated',
                    message: `Dominant emotion: ${analysis.dominantEmotion}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowEmotionRecognition(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* API Integration Modal */}
      <AnimatePresence>
        {showAPIIntegration && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowAPIIntegration(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-7xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <APIIntegration
                onConnectionChange={(connection) => {
                  console.log('Connection changed:', connection)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Connection Updated',
                    message: `${connection.name} status: ${connection.status}`,
                    duration: 3000,
                  })
                }}
                onPluginActivated={(plugin) => {
                  console.log('Plugin activated:', plugin)
                  uiActions.addNotification({
                    type: 'success',
                    title: 'Plugin Activated',
                    message: `${plugin.name} is now active`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowAPIIntegration(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Enterprise Cloud Modal */}
      <AnimatePresence>
        {showEnterpriseCloud && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={() => setShowEnterpriseCloud(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-7xl max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <EnterpriseCloud
                onConnectorChange={(connector) => {
                  console.log('Connector changed:', connector)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Enterprise Connector',
                    message: `${connector.name} status: ${connector.status}`,
                    duration: 3000,
                  })
                }}
                onServiceChange={(service) => {
                  console.log('Service changed:', service)
                  uiActions.addNotification({
                    type: 'info',
                    title: 'Cloud Service',
                    message: `${service.name} status: ${service.status}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowEnterpriseCloud(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
