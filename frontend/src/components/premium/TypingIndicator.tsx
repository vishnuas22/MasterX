'use client'

import { motion } from 'framer-motion'
import { Bot, Sparkles, Zap } from 'lucide-react'

interface TypingIndicatorProps {
  message?: string
}

export default function TypingIndicator({ message = "Quantum processing..." }: TypingIndicatorProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex justify-start"
    >
      <div className="flex items-start space-x-4">
        {/* AI Avatar */}
        <motion.div 
          animate={{ 
            scale: [1, 1.05, 1],
            rotate: [0, 1, -1, 0]
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
          className="w-12 h-12 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg relative"
        >
          <Bot className="w-6 h-6 text-white" />
          
          {/* Pulsing status indicator */}
          <motion.div 
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [1, 0.7, 1]
            }}
            transition={{ 
              duration: 1.5, 
              repeat: Infinity, 
              ease: "easeInOut" 
            }}
            className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full flex items-center justify-center"
          >
            <div className="w-2 h-2 bg-white rounded-full" />
          </motion.div>
        </motion.div>

        {/* Typing Bubble */}
        <motion.div 
          animate={{ 
            scale: [1, 1.02, 1]
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
          className="px-6 py-4 bg-black/30 border border-white/20 rounded-2xl backdrop-blur-sm shadow-lg max-w-xs"
        >
          <div className="flex items-center space-x-3">
            {/* Quantum-themed typing dots */}
            <div className="flex space-x-1">
              <motion.div
                animate={{ 
                  scale: [1, 1.3, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{ 
                  duration: 1.2, 
                  repeat: Infinity, 
                  delay: 0 
                }}
                className="w-2 h-2 bg-gradient-to-r from-purple-400 to-cyan-400 rounded-full"
              />
              <motion.div
                animate={{ 
                  scale: [1, 1.3, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{ 
                  duration: 1.2, 
                  repeat: Infinity, 
                  delay: 0.2 
                }}
                className="w-2 h-2 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full"
              />
              <motion.div
                animate={{ 
                  scale: [1, 1.3, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{ 
                  duration: 1.2, 
                  repeat: Infinity, 
                  delay: 0.4 
                }}
                className="w-2 h-2 bg-gradient-to-r from-purple-400 to-cyan-400 rounded-full"
              />
            </div>

            {/* Message text */}
            <span className="text-sm text-gray-300 font-medium">{message}</span>

            {/* Quantum effects */}
            <motion.div
              animate={{ 
                rotate: 360,
                scale: [1, 1.1, 1]
              }}
              transition={{ 
                rotate: { duration: 3, repeat: Infinity, ease: "linear" },
                scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
              }}
            >
              <Sparkles className="w-4 h-4 text-purple-400" />
            </motion.div>
          </div>

          {/* Processing indicator */}
          <div className="mt-2 flex items-center space-x-2">
            <motion.div
              animate={{ 
                width: ["0%", "100%", "0%"]
              }}
              transition={{ 
                duration: 2, 
                repeat: Infinity, 
                ease: "easeInOut" 
              }}
              className="h-0.5 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full"
            />
            <Zap className="w-3 h-3 text-cyan-400" />
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}
