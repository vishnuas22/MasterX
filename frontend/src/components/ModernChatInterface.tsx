'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { 
  Send,
  Paperclip,
  Mic,
  Brain,
  Code,
  BarChart3,
  Lightbulb,
  ArrowRight
} from 'lucide-react'

const suggestions = [
  {
    icon: Brain,
    title: "Explain quantum computing",
    description: "Learn about quantum mechanics and computing principles"
  },
  {
    icon: Code,
    title: "Write Python code",
    description: "Generate and optimize code solutions"
  },
  {
    icon: BarChart3,
    title: "Analyze data",
    description: "Extract insights from complex datasets"
  },
  {
    icon: Lightbulb,
    title: "Brainstorm ideas",
    description: "Generate innovative solutions"
  }
]

export default function ModernChatInterface() {
  return (
    <div className="flex flex-col h-full bg-white">
      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12 max-w-2xl"
        >
          <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center mx-auto mb-6">
            <Brain className="h-6 w-6 text-white" />
          </div>
          
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            How can I help you today?
          </h1>
          
          <p className="text-lg text-gray-600">
            I'm your AI assistant. Ask me anything or choose a suggestion below.
          </p>
        </motion.div>

        {/* Suggestions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl mb-12"
        >
          {suggestions.map((suggestion, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.3 + index * 0.1 }}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              className="p-4 border border-gray-200 rounded-xl text-left hover:border-gray-300 hover:shadow-sm transition-all duration-200 group"
            >
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center group-hover:bg-gray-200 transition-colors">
                  <suggestion.icon className="h-4 w-4 text-gray-600" />
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900 mb-1">
                    {suggestion.title}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {suggestion.description}
                  </p>
                </div>
                <ArrowRight className="h-4 w-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </motion.button>
          ))}
        </motion.div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            <div className="flex items-end space-x-3 bg-gray-50 rounded-2xl p-3">
              <button className="p-2 hover:bg-gray-200 rounded-lg transition-colors">
                <Paperclip className="h-4 w-4 text-gray-500" />
              </button>
              
              <div className="flex-1">
                <textarea
                  placeholder="Message MasterX..."
                  className="w-full bg-transparent resize-none border-0 outline-none text-gray-900 placeholder-gray-500 text-sm leading-6"
                  rows={1}
                  style={{ minHeight: '24px', maxHeight: '200px' }}
                />
              </div>
              
              <div className="flex items-center space-x-2">
                <button className="p-2 hover:bg-gray-200 rounded-lg transition-colors">
                  <Mic className="h-4 w-4 text-gray-500" />
                </button>
                <button className="p-2 bg-black hover:bg-gray-800 rounded-lg transition-colors">
                  <Send className="h-4 w-4 text-white" />
                </button>
              </div>
            </div>
          </div>
          
          <p className="text-xs text-gray-500 text-center mt-3">
            MasterX can make mistakes. Consider checking important information.
          </p>
        </div>
      </div>
    </div>
  )
}
