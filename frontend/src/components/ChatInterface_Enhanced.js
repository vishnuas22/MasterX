import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { GlassCard, GlassButton, GlassInput } from './GlassCard';
import { TypingIndicator, LoadingStates } from './LoadingSpinner';
import { 
  SendIcon, 
  AIBrainIcon, 
  UserIcon, 
  SparkleIcon, 
  BookIcon, 
  TargetIcon, 
  SettingsIcon, 
  TrophyIcon, 
  ZapIcon, 
  ArrowDownIcon,
  PulsingDot,
  CheckIcon,
  MicrophoneIcon
} from './PremiumIcons';
import { PremiumLearningModes, LearningModeIndicator } from './PremiumLearningModes';
import { ModelManagement } from './ModelManagement';
import { GamificationDashboard } from './GamificationDashboard';
import { AdvancedStreamingInterface } from './AdvancedStreamingInterface';
import ContextAwareChatInterface from './ContextAwareChatInterface';
import LiveLearningInterface from './LiveLearningInterface';
import { VoiceInterface } from './VoiceInterface';
import { GestureControl } from './GestureControl';
import { ARVRInterface } from './ARVRInterface';
import { ThemeProvider, AdaptiveThemePanel, useAdaptiveTheme } from './AdaptiveThemeSystem';
import { useApp } from '../context/AppContext';
import { cn } from '../utils/cn';

// Enhanced ChatMessage Component with Premium Animations
function EnhancedChatMessage({ message, isExpanded = true }) {
  const isUser = message.sender === 'user';
  const isTyping = message.sender === 'mentor' && !message.message;
  const isPremium = message.learning_mode && message.learning_mode !== 'adaptive';

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.98 }}
      transition={{ 
        duration: 0.4, 
        ease: [0.4, 0.0, 0.2, 1],
        delay: 0.1 
      }}
      className="mb-6 flex justify-start group"
    >
      <div className="flex items-start space-x-3 max-w-full w-full">
        {/* Enhanced Avatar with Premium Glow */}
        <motion.div 
          className="flex-shrink-0 mt-1"
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 400, damping: 20 }}
        >
          <div className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center relative",
            "transition-all duration-300 group-hover:shadow-lg",
            isUser 
              ? "bg-gradient-to-br from-ai-green-400 to-ai-blue-500 shadow-lg" 
              : isPremium 
                ? "glass-ai-secondary shadow-glow-purple group-hover:shadow-glow-purple-intense" 
                : "glass-ai-primary shadow-glow-blue group-hover:shadow-glow-blue-intense"
          )}>
            {/* Premium Animated Border */}
            {!isUser && (
              <div className="absolute -inset-0.5 rounded-full bg-gradient-to-r from-ai-blue-500 via-ai-purple-500 to-ai-green-500 opacity-0 group-hover:opacity-60 animate-pulse transition-opacity duration-300" />
            )}
            <div className="relative">
              {isUser ? (
                <span className="text-white text-sm font-bold">U</span>
              ) : (
                <AIBrainIcon 
                  size="sm" 
                  className={isPremium ? 'text-ai-purple-400' : 'text-ai-blue-400'}
                  animated
                />
              )}
            </div>
          </div>
        </motion.div>
        
        {/* Enhanced Message Content */}
        <div className="flex-1 min-w-0">
          {/* User Label */}
          <motion.div 
            className="mb-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <span className="text-sm font-medium text-text-primary">
              {isUser ? 'You' : 'MasterX'}
            </span>
            {isPremium && !isUser && (
              <motion.span 
                className="ml-2 text-xs text-ai-purple-400 flex items-center space-x-1"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3, type: "spring" }}
              >
                <SparkleIcon size="xs" />
                <span>Premium</span>
              </motion.span>
            )}
          </motion.div>
          
          {/* Premium Glass Message Container */}
          <motion.div
            className="relative group/message"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            {/* Premium Gradient Border */}
            <div className={cn(
              "absolute -inset-[1px] rounded-xl transition-opacity duration-300",
              "bg-gradient-to-r from-ai-blue-500/30 via-ai-purple-500/30 to-ai-green-500/30",
              "opacity-0 group-hover/message:opacity-100",
              !isUser && "animate-gradient-x"
            )} />
            
            <GlassCard 
              variant={
                isUser 
                  ? 'ai-primary'
                  : isPremium
                    ? 'ai-secondary'
                    : 'medium'
              }
              size="sm"
              className={cn(
                "relative glass-medium shadow-lg backdrop-blur-xl",
                "transition-all duration-300 group-hover/message:shadow-xl",
                isUser 
                  ? "bg-ai-blue-500/10 border border-ai-blue-500/20 group-hover/message:bg-ai-blue-500/15" 
                  : isPremium
                    ? "bg-ai-purple-500/5 border border-ai-purple-500/20 group-hover/message:bg-ai-purple-500/10"
                    : "bg-glass-light border border-border-subtle group-hover/message:bg-glass-medium"
              )}
              hover={false}
            >
              {/* Premium Mode Indicator */}
              {isPremium && !isUser && (
                <motion.div 
                  className="flex items-center space-x-2 mb-4 pb-3 border-b border-border-subtle"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <motion.div
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                  >
                    <SparkleIcon size="sm" className="text-ai-purple-400" />
                  </motion.div>
                  <span className="text-sm font-semibold text-ai-purple-300 capitalize">
                    {message.learning_mode} Mode
                  </span>
                  <div className="ml-auto">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.5, type: "spring" }}
                    >
                      <CheckIcon size="sm" className="text-ai-green-400" />
                    </motion.div>
                  </div>
                </motion.div>
              )}
              
              {/* Message Content with Premium Formatting */}
              {message.message ? (
                <div className="prose prose-lg prose-invert max-w-none">
                  <div className="premium-response text-text-primary leading-relaxed">
                    <ReactMarkdown
                      components={{
                        h1: ({children}) => <h1 className="text-xl font-bold text-gradient-primary mb-4 flex items-center space-x-2"><SparkleIcon size="sm" className="text-ai-blue-400" /><span>{children}</span></h1>,
                        h2: ({children}) => <h2 className="text-lg font-semibold text-ai-blue-300 mb-3 mt-6">{children}</h2>,
                        h3: ({children}) => <h3 className="text-base font-semibold text-ai-purple-300 mb-2 mt-4">{children}</h3>,
                        p: ({children}) => <p className="mb-4 text-text-secondary leading-relaxed">{children}</p>,
                        ul: ({children}) => <ul className="mb-4 space-y-2 ml-4">{children}</ul>,
                        ol: ({children}) => <ol className="mb-4 space-y-2 ml-4 list-decimal">{children}</ol>,
                        li: ({children}) => <li className="text-text-secondary flex items-start space-x-2"><span className="text-ai-blue-400 mt-1">•</span><span className="flex-1">{children}</span></li>,
                        strong: ({children}) => <strong className="font-semibold text-ai-blue-300">{children}</strong>,
                        em: ({children}) => <em className="italic text-ai-purple-300">{children}</em>,
                        code: ({children}) => <code className="px-2 py-1 bg-glass-thick rounded text-ai-green-300 text-sm font-mono">{children}</code>,
                        pre: ({children}) => <pre className="mb-4 p-4 bg-glass-thick rounded-lg overflow-x-auto border border-border-subtle">{children}</pre>,
                        blockquote: ({children}) => <blockquote className="border-l-4 border-ai-blue-500 pl-4 mb-4 italic text-text-tertiary bg-ai-blue-500/5 py-2 rounded-r-lg">{children}</blockquote>
                      }}
                    >
                      {message.message}
                    </ReactMarkdown>
                  </div>
                </div>
              ) : (
                <TypingIndicator size="sm" />
              )}
              
              {/* Message Suggestions */}
              {!isUser && message.suggestions && message.suggestions.length > 0 && (
                <div className="mt-6 pt-4 border-t border-border-subtle">
                  <p className="text-sm text-ai-blue-400 mb-3 flex items-center space-x-2 font-semibold">
                    <SparkleIcon size="sm" />
                    <span>Suggested Actions</span>
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {message.suggestions.slice(0, 3).map((suggestion, index) => (
                      <GlassButton
                        key={index}
                        size="sm"
                        variant="tertiary"
                        className="text-sm hover:variant-secondary"
                      >
                        {suggestion}
                      </GlassButton>
                    ))}
                  </div>
                </div>
              )}

              {/* Next Steps for Premium Responses */}
              {!isUser && message.next_steps && (
                <div className="mt-6 pt-4 border-t border-border-subtle">
                  <p className="text-sm text-ai-green-400 mb-3 flex items-center space-x-2 font-semibold">
                    <TargetIcon size="sm" />
                    <span>Next Steps</span>
                  </p>
                  <div className="bg-ai-green-500/5 border border-ai-green-500/20 rounded-lg p-3">
                    <p className="text-sm text-text-secondary leading-relaxed">
                      {message.next_steps}
                    </p>
                  </div>
                </div>
              )}
            </GlassCard>
          </motion.div>
          
          {/* Message Metadata */}
          <div className="flex items-center mt-2 text-xs text-text-quaternary space-x-2">
            <span>{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
            {isPremium && (
              <>
                <span>•</span>
                <div className="flex items-center space-x-1">
                  <SparkleIcon size="xs" className="text-ai-purple-400" />
                  <span className="text-ai-purple-400 font-medium">Premium Response</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export { EnhancedChatMessage };