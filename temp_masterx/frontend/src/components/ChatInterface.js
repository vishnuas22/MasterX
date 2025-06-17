import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles, BookOpen, Target, Settings, Brain } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { GlassCard, GlassButton, GlassInput } from './GlassCard';
import { TypingIndicator } from './LoadingSpinner';
import { PremiumLearningModes, LearningModeIndicator } from './PremiumLearningModes';
import { ModelManagement } from './ModelManagement';
import { useApp } from '../context/AppContext';

export function ChatInterface() {
  const { state, actions } = useApp();
  const [inputMessage, setInputMessage] = useState('');
  const [learningMode, setLearningMode] = useState('adaptive');
  const [showLearningModes, setShowLearningModes] = useState(false);
  const [showModelManagement, setShowModelManagement] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [state.messages, state.streamingMessage]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || !state.currentSession || state.isTyping) return;

    const message = inputMessage.trim();
    setInputMessage('');

    try {
      await actions.sendPremiumMessage(state.currentSession.id, message, {
        learning_mode: learningMode,
        user_preferences: {
          difficulty_preference: 'adaptive',
          interaction_style: 'collaborative'
        }
      });
    } catch (error) {
      console.error('Error sending premium message:', error);
      // Fallback to regular message
      try {
        await actions.sendMessage(state.currentSession.id, message);
      } catch (fallbackError) {
        console.error('Error sending fallback message:', fallbackError);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  if (!state.currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <GlassCard className="p-8 text-center max-w-md">
          <Bot className="h-16 w-16 text-blue-400 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-gray-100 mb-2">Welcome to MasterX</h3>
          <p className="text-gray-400 mb-6">
            Your AI-powered learning companion. Start a new session to begin your learning journey.
          </p>
          <GlassButton
            onClick={() => {
              // This would typically open a session creation modal
              console.log('Create new session');
            }}
            className="w-full"
          >
            <BookOpen className="h-4 w-4 mr-2" />
            Start Learning
          </GlassButton>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Chat Header */}
      <div className="border-b border-white/10 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Bot className="h-8 w-8 text-blue-400" />
              <div className="absolute -bottom-1 -right-1 h-3 w-3 bg-green-400 rounded-full border-2 border-gray-900"></div>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-100">MasterX AI Mentor</h2>
              <p className="text-sm text-gray-400">
                {state.currentSession.subject || 'General Learning'} • {state.currentSession.difficulty_level}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <LearningModeIndicator 
              currentMode={learningMode}
              onClick={() => setShowLearningModes(true)}
            />
            <GlassButton 
              size="sm" 
              variant="secondary"
              onClick={() => setShowModelManagement(true)}
            >
              <Brain className="h-4 w-4" />
            </GlassButton>
            <GlassButton size="sm" variant="secondary">
              <Target className="h-4 w-4" />
            </GlassButton>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {state.messages.map((message, index) => (
            <ChatMessage key={message.id || index} message={message} />
          ))}
          
          {/* Streaming message */}
          {state.isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex items-start space-x-3"
            >
              <div className="flex-shrink-0">
                <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-white" />
                </div>
              </div>
              <div className="flex-1">
                <GlassCard className="p-4">
                  {state.streamingMessage ? (
                    <div className="prose prose-invert max-w-none">
                      <ReactMarkdown>{state.streamingMessage}</ReactMarkdown>
                      <span className="inline-block w-2 h-5 bg-blue-400 animate-pulse ml-1"></span>
                    </div>
                  ) : (
                    <TypingIndicator />
                  )}
                </GlassCard>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-white/10 p-4">
        <form onSubmit={handleSendMessage} className="flex space-x-3">
          <div className="flex-1 relative">
            <GlassInput
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything..."
              className="w-full pr-12"
              disabled={state.isTyping}
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <Sparkles className="h-4 w-4 text-blue-400" />
            </div>
          </div>
          <GlassButton
            type="submit"
            disabled={!inputMessage.trim() || state.isTyping}
            className="px-4"
          >
            <Send className="h-4 w-4" />
          </GlassButton>
        </form>
        
        {/* Quick actions */}
        <div className="flex space-x-2 mt-3">
          <GlassButton
            size="sm"
            variant="secondary"
            onClick={() => setInputMessage("Can you create an exercise for me?")}
            disabled={state.isTyping}
          >
            Generate Exercise
          </GlassButton>
          <GlassButton
            size="sm"
            variant="secondary"
            onClick={() => setInputMessage("Explain this concept step by step")}
            disabled={state.isTyping}
          >
            Step-by-Step
          </GlassButton>
          <GlassButton
            size="sm"
            variant="secondary"
            onClick={() => setInputMessage("Give me a real-world example")}
            disabled={state.isTyping}
          >
            Real Example
          </GlassButton>
          <GlassButton
            size="sm"
            variant="secondary"
            onClick={() => {
              setLearningMode('challenge');
              setInputMessage("Give me a challenge problem");
            }}
            disabled={state.isTyping}
          >
            Challenge Me
          </GlassButton>
        </div>
      </div>

      {/* Premium Learning Modes Modal */}
      <PremiumLearningModes
        currentMode={learningMode}
        onModeChange={setLearningMode}
        isVisible={showLearningModes}
        onClose={() => setShowLearningModes(false)}
      />

      {/* Model Management Modal */}
      <ModelManagement
        isVisible={showModelManagement}
        onClose={() => setShowModelManagement(false)}
      />
    </div>
  );
}

function ChatMessage({ message }) {
  const isUser = message.sender === 'user';
  const isTyping = message.sender === 'mentor' && !message.message;
  const isPremium = message.learning_mode && message.learning_mode !== 'adaptive';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start space-x-3`}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
            isPremium 
              ? 'bg-gradient-to-r from-purple-500 to-pink-500' 
              : 'bg-gradient-to-r from-blue-500 to-purple-500'
          }`}>
            <Bot className="h-4 w-4 text-white" />
          </div>
        </div>
      )}
      
      <div className={`max-w-[80%] ${isUser ? 'order-first' : ''}`}>
        <GlassCard 
          className={`p-4 ${isUser 
            ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 border-blue-400/30' 
            : isPremium
              ? 'bg-gradient-to-r from-purple-500/10 to-pink-500/10 border-purple-400/20'
              : 'bg-white/5 border-white/10'
          }`}
        >
          {isPremium && (
            <div className="flex items-center space-x-2 mb-2 pb-2 border-b border-white/10">
              <Sparkles className="h-3 w-3 text-purple-400" />
              <span className="text-xs text-purple-300 capitalize">{message.learning_mode} Mode</span>
            </div>
          )}
          
          {message.message ? (
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown>{message.message}</ReactMarkdown>
            </div>
          ) : (
            <TypingIndicator />
          )}
          
          {/* Message suggestions */}
          {!isUser && message.suggestions && message.suggestions.length > 0 && (
            <div className="mt-3 pt-3 border-t border-white/10">
              <p className="text-xs text-gray-400 mb-2">Suggested actions:</p>
              <div className="flex flex-wrap gap-2">
                {message.suggestions.map((suggestion, index) => (
                  <GlassButton
                    key={index}
                    size="sm"
                    variant="secondary"
                    className="text-xs"
                  >
                    {suggestion}
                  </GlassButton>
                ))}
              </div>
            </div>
          )}

          {/* Next steps for premium responses */}
          {!isUser && message.next_steps && (
            <div className="mt-3 pt-3 border-t border-white/10">
              <p className="text-xs text-purple-400 mb-1 flex items-center space-x-1">
                <Target className="h-3 w-3" />
                <span>Next Steps:</span>
              </p>
              <p className="text-xs text-gray-300">{message.next_steps}</p>
            </div>
          )}
        </GlassCard>
        
        <div className={`flex items-center mt-2 text-xs text-gray-500 ${isUser ? 'justify-end' : 'justify-start'}`}>
          <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
          {isPremium && (
            <>
              <span className="mx-1">•</span>
              <span className="text-purple-400">Premium</span>
            </>
          )}
        </div>
      </div>
      
      {isUser && (
        <div className="flex-shrink-0">
          <div className="h-8 w-8 rounded-full bg-gradient-to-r from-gray-600 to-gray-700 flex items-center justify-center">
            <User className="h-4 w-4 text-white" />
          </div>
        </div>
      )}
    </motion.div>
  );
}
