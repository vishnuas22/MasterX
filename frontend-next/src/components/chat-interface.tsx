'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Brain, Zap, Target } from 'lucide-react'
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
      // For now, generate a local session ID
      // Later we'll call: const response = await api.chat.createSession()
      const newSessionId = `session_${Date.now()}`
      setSessionId(newSessionId)
    } catch (error) {
      console.error('Failed to initialize session:', error)
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
      // For now, simulate AI response
      // Later we'll call: const response = await api.chat.send(inputMessage, sessionId)
      
      setTimeout(() => {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: generateMockAIResponse(inputMessage),
          sender: 'ai',
          timestamp: new Date(),
          metadata: {
            learningMode: 'socratic_discovery',
            concepts: extractConcepts(inputMessage),
            confidence: 0.88
          }
        }
        setMessages(prev => [...prev, aiMessage])
        setIsLoading(false)
      }, 1500)

    } catch (error) {
      console.error('Failed to send message:', error)
      setIsLoading(false)
    }
  }

  const generateMockAIResponse = (userInput: string): string => {
    const responses = [
      `That's a fascinating question about "${userInput}". Let me break this down using the Quantum Intelligence approach. First, let's explore the fundamental concepts...`,
      `I can see you're curious about this topic. Using our adaptive learning system, I'll guide you through this step by step. What specific aspect interests you most?`,
      `Excellent! This connects to several key learning principles. Let me engage our Socratic questioning mode to help you discover the answer yourself...`,
      `This is a perfect opportunity for our Debug Mastery mode. Let's identify any knowledge gaps and build a solid foundation...`
    ]
    return responses[Math.floor(Math.random() * responses.length)]
  }

  const extractConcepts = (input: string): string[] => {
    // Simple concept extraction (later replace with AI)
    const concepts = ['learning', 'quantum', 'AI', 'education']
    return concepts.filter(concept => 
      input.toLowerCase().includes(concept.toLowerCase())
    )
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="max-w-6xl mx-auto p-6 h-screen flex flex-col">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold quantum-text-glow mb-2">AI Learning Mentor</h1>
        <p className="text-gray-400">Powered by Quantum Intelligence Engine</p>
        {sessionId && (
          <p className="text-xs text-purple-400 mt-1">Session: {sessionId}</p>
        )}
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto bg-slate-800/30 rounded-xl p-4 mb-6 border border-purple-500/20">
        <div className="space-y-4">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isLoading && <LoadingIndicator />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-slate-800/50 rounded-xl p-4 border border-purple-500/20">
        <div className="flex space-x-3">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about learning, or let me know what you'd like to explore..."
            className="flex-1 bg-slate-700/50 text-white placeholder-gray-400 border border-slate-600 rounded-lg px-4 py-3 focus:border-purple-500 focus:outline-none resize-none"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-semibold quantum-glow transition-all flex items-center space-x-2"
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
      'flex space-x-3',
      message.sender === 'user' ? 'justify-end' : 'justify-start'
    )}>
      {message.sender === 'ai' && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
            <Brain className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
      
      <div className={cn(
        'max-w-3xl p-4 rounded-xl',
        message.sender === 'user' 
          ? 'bg-purple-600 text-white ml-auto' 
          : 'bg-slate-700/50 text-gray-100 border border-purple-500/20'
      )}>
        <p className="whitespace-pre-wrap">{message.content}</p>
        
        {message.metadata && message.sender === 'ai' && (
          <div className="mt-3 pt-3 border-t border-purple-500/20">
            <div className="flex flex-wrap gap-2 text-xs">
              {message.metadata.learningMode && (
                <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded-full flex items-center space-x-1">
                  <Zap className="h-3 w-3" />
                  <span>{message.metadata.learningMode.replace('_', ' ')}</span>
                </span>
              )}
              {message.metadata.concepts && message.metadata.concepts.length > 0 && (
                <span className="bg-blue-500/20 text-blue-300 px-2 py-1 rounded-full flex items-center space-x-1">
                  <Target className="h-3 w-3" />
                  <span>{message.metadata.concepts.join(', ')}</span>
                </span>
              )}
              {message.metadata.confidence && (
                <span className="bg-green-500/20 text-green-300 px-2 py-1 rounded-full">
                  {Math.round(message.metadata.confidence * 100)}% confidence
                </span>
              )}
            </div>
          </div>
        )}
        
        <p className="text-xs text-gray-400 mt-2">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>

      {message.sender === 'user' && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-slate-600 flex items-center justify-center">
            <User className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
    </div>
  )
}

function LoadingIndicator() {
  return (
    <div className="flex space-x-3 justify-start">
      <div className="flex-shrink-0">
        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
          <Brain className="h-5 w-5 text-white" />
        </div>
      </div>
      <div className="bg-slate-700/50 p-4 rounded-xl border border-purple-500/20">
        <div className="flex items-center space-x-2">
          <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
          <span className="text-gray-300">AI is thinking...</span>
        </div>
      </div>
    </div>
  )
}