'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { formatDistanceToNow } from 'date-fns'
import { 
  User, 
  Bot, 
  Copy, 
  RotateCcw, 
  Share2, 
  Edit3,
  Check,
  Sparkles,
  Zap,
  Clock,
  Hash
} from 'lucide-react'

interface MessageMetadata {
  model?: string
  provider?: string
  tokens_used?: number
  response_time_ms?: number
  confidence?: number
  learning_mode?: string
}

interface EnhancedMessageProps {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  metadata?: MessageMetadata
  isStreaming?: boolean
  onRegenerate?: () => void
  onEdit?: (content: string) => void
}

export default function EnhancedMessage({
  id,
  content,
  sender,
  timestamp,
  metadata,
  isStreaming = false,
  onRegenerate,
  onEdit
}: EnhancedMessageProps) {
  const [showActions, setShowActions] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showTimestamp, setShowTimestamp] = useState(false)
  const messageRef = useRef<HTMLDivElement>(null)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text:', err)
    }
  }

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'MasterX Chat Message',
          text: content,
        })
      } catch (err) {
        console.error('Failed to share:', err)
      }
    } else {
      handleCopy()
    }
  }

  const formatTimestamp = (date: Date) => {
    return formatDistanceToNow(date, { addSuffix: true })
  }

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'text-gray-400'
    if (confidence >= 0.9) return 'text-green-400'
    if (confidence >= 0.7) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <motion.div
      ref={messageRef}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{
        duration: 0.3,
        ease: "easeOut"
      }}
      className={`w-full mb-8 group`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className={`flex items-start gap-4 ${sender === 'user' ? 'flex-row-reverse' : ''}`}>
        {/* Avatar */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          className={`
            w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-xl relative
            ${sender === 'user'
              ? 'bg-gradient-to-br from-blue-500 to-purple-600'
              : 'bg-gradient-to-br from-purple-500 to-cyan-500'
            }
          `}
        >
          {sender === 'user' ? (
            <User className="w-6 h-6 text-white" />
          ) : (
            <Bot className="w-6 h-6 text-white" />
          )}

          {/* Status indicator for AI */}
          {sender === 'ai' && (
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full flex items-center justify-center">
              <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
            </div>
          )}
        </motion.div>

        {/* Message Content */}
        <div className={`flex-1 max-w-[75%] ${sender === 'user' ? 'text-right' : 'text-left'}`}>
          <motion.div
            whileHover={{
              scale: 1.002,
              transition: { duration: 0.15, ease: "easeOut" }
            }}
            className={`
              inline-block px-7 py-5 rounded-3xl backdrop-blur-2xl border shadow-2xl relative transition-all duration-300
              ${sender === 'user'
                ? 'bg-gradient-to-br from-blue-500/25 to-purple-500/25 border-blue-400/30 text-white shadow-blue-500/20'
                : 'bg-gradient-to-br from-black/50 to-gray-900/50 border-white/25 text-white shadow-black/30'
              }
            `}
          >
            {/* Message Content with Markdown */}
            <div className="prose prose-invert prose-sm max-w-none">
              {sender === 'ai' ? (
                <ReactMarkdown
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '')
                      const language = match ? match[1] : ''
                      
                      return !inline && language ? (
                        <div className="relative group">
                          <SyntaxHighlighter
                            style={oneDark}
                            language={language}
                            PreTag="div"
                            className="rounded-lg !bg-gray-900/50 !border border-white/10"
                            {...props}
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                          <button
                            onClick={() => navigator.clipboard.writeText(String(children))}
                            className="absolute top-2 right-2 p-2 bg-white/10 hover:bg-white/20 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <Copy className="w-4 h-4 text-white" />
                          </button>
                        </div>
                      ) : (
                        <code className="bg-gray-800/50 px-2 py-1 rounded text-purple-300" {...props}>
                          {children}
                        </code>
                      )
                    },
                    h1: ({ children }) => <h1 className="text-2xl font-bold text-white mb-4">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-xl font-semibold text-white mb-3">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-lg font-medium text-white mb-2">{children}</h3>,
                    p: ({ children }) => <p className="text-gray-100 leading-relaxed mb-3 last:mb-0">{children}</p>,
                    ul: ({ children }) => <ul className="list-disc list-inside text-gray-100 space-y-1 mb-3">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-inside text-gray-100 space-y-1 mb-3">{children}</ol>,
                    li: ({ children }) => <li className="text-gray-100">{children}</li>,
                    strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                    em: ({ children }) => <em className="italic text-purple-300">{children}</em>,
                    a: ({ href, children }) => (
                      <a href={href} className="text-cyan-400 hover:text-cyan-300 underline" target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-purple-500 pl-4 italic text-gray-300 my-3">
                        {children}
                      </blockquote>
                    )
                  }}
                >
                  {isStreaming ? content + '▊' : content}
                </ReactMarkdown>
              ) : (
                <p className="text-white leading-relaxed">{content}</p>
              )}
            </div>

            {/* Metadata for AI messages */}
            {metadata && sender === 'ai' && (
              <div className="mt-4 pt-3 border-t border-white/20 flex items-center justify-between text-xs">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <Sparkles className="w-3 h-3 text-purple-400" />
                    <span className="text-gray-300">{metadata.model || 'AI'}</span>
                  </div>
                  {metadata.confidence && (
                    <div className="flex items-center space-x-1">
                      <Zap className={`w-3 h-3 ${getConfidenceColor(metadata.confidence)}`} />
                      <span className={getConfidenceColor(metadata.confidence)}>
                        {(metadata.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
                <div className="flex items-center space-x-4">
                  {metadata.tokens_used && (
                    <div className="flex items-center space-x-1">
                      <Hash className="w-3 h-3 text-cyan-400" />
                      <span className="text-cyan-400">{metadata.tokens_used}</span>
                    </div>
                  )}
                  {metadata.response_time_ms && (
                    <div className="flex items-center space-x-1">
                      <Clock className="w-3 h-3 text-green-400" />
                      <span className="text-green-400">{metadata.response_time_ms}ms</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Timestamp */}
            <div 
              className="mt-2 text-xs text-gray-400 cursor-pointer"
              onMouseEnter={() => setShowTimestamp(true)}
              onMouseLeave={() => setShowTimestamp(false)}
            >
              {showTimestamp ? timestamp.toLocaleString() : formatTimestamp(timestamp)}
            </div>

          </motion.div>

          {/* Message Actions */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: showActions ? 1 : 0, scale: showActions ? 1 : 0.8 }}
            className={`mt-2 flex ${sender === 'user' ? 'justify-end' : 'justify-start'} space-x-2`}
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleCopy}
              className="p-1.5 bg-black/50 hover:bg-black/70 rounded-lg backdrop-blur-sm border border-white/10 transition-all shadow-lg"
              title="Copy message"
            >
              {copied ? (
                <Check className="w-3 h-3 text-green-400" />
              ) : (
                <Copy className="w-3 h-3 text-gray-300" />
              )}
            </motion.button>

            {sender === 'ai' && onRegenerate && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onRegenerate}
                className="p-1.5 bg-black/50 hover:bg-black/70 rounded-lg backdrop-blur-sm border border-white/10 transition-all shadow-lg"
                title="Regenerate response"
              >
                <RotateCcw className="w-3 h-3 text-gray-300" />
              </motion.button>
            )}

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleShare}
              className="p-1.5 bg-black/50 hover:bg-black/70 rounded-lg backdrop-blur-sm border border-white/10 transition-all shadow-lg"
              title="Share message"
            >
              <Share2 className="w-3 h-3 text-gray-300" />
            </motion.button>

            {onEdit && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => onEdit(content)}
                className="p-1.5 bg-black/50 hover:bg-black/70 rounded-lg backdrop-blur-sm border border-white/10 transition-all shadow-lg"
                title="Edit message"
              >
                <Edit3 className="w-3 h-3 text-gray-300" />
              </motion.button>
            )}
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
