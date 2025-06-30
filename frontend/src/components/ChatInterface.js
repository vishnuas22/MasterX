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

// Helper function to record learning analytics events
const recordLearningEvent = async (userId, sessionId, eventData) => {
  try {
    const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/analytics/learning-event`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        session_id: sessionId,
        concept_id: eventData.concept || 'general_learning',
        event_type: eventData.type || 'interaction',
        duration_seconds: eventData.duration || 0,
        performance_score: eventData.performance || 0.5,
        confidence_level: eventData.confidence || 0.5,
        context: eventData.context || {}
      })
    });
    
    if (!response.ok) {
      console.warn('Failed to record learning event');
    }
  } catch (error) {
    console.warn('Error recording learning event:', error);
  }
};

export function ChatInterface() {
  const { state, actions } = useApp();
  const { getThemeClasses } = useAdaptiveTheme();
  const [inputMessage, setInputMessage] = useState('');
  const [learningMode, setLearningMode] = useState('adaptive');
  const [showLearningModes, setShowLearningModes] = useState(false);
  const [showModelManagement, setShowModelManagement] = useState(false);
  const [showGamification, setShowGamification] = useState(false);
  const [showLiveLearning, setShowLiveLearning] = useState(false);
  const [showThemePanel, setShowThemePanel] = useState(false);
  const [useAdvancedStreaming, setUseAdvancedStreaming] = useState(false);
  const [useContextAwareness, setUseContextAwareness] = useState(true);
  const [currentView, setCurrentView] = useState('chat'); // 'chat', 'live-learning'
  const [isChatExpanded, setIsChatExpanded] = useState(false);
  const [isInputFocused, setIsInputFocused] = useState(false);
  
  // Simplified and robust scroll management
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);
  const [isUserNearBottom, setIsUserNearBottom] = useState(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const autoScrollTimeoutRef = useRef(null);
  const scrollCheckIntervalRef = useRef(null);
  const isAutoScrollingRef = useRef(false);

  // Helper function for quick start - defined early with useCallback to avoid hoisting issues
  const handleQuickStart = useCallback(async () => {
    if (!inputMessage.trim() || !state.user?.id) return;

    const message = inputMessage.trim();
    setInputMessage('');

    try {
      actions.setLoading(true);
      
      const sessionData = {
        user_id: state.user.id,
        subject: 'General Learning',
        difficulty_level: 'intermediate',
        learning_objectives: ['Interactive learning', 'Skill development']
      };
      
      await actions.createSession(sessionData);
      
      // Send the initial message
      setTimeout(async () => {
        if (state.currentSession?.id) {
          await actions.sendMessage(state.currentSession.id, message);
        }
      }, 100);
      
    } catch (error) {
      console.error('Error starting conversation:', error);
      actions.setError('Failed to start conversation. Please try again.');
    } finally {
      actions.setLoading(false);
    }
  }, [inputMessage, state.user, state.currentSession, actions]);

  // Helper function for subject-specific start
  const handleSubjectStart = useCallback(async (subject) => {
    if (!state.user?.id) return;

    try {
      actions.setLoading(true);
      
      const sessionData = {
        user_id: state.user.id,
        subject: subject,
        difficulty_level: 'intermediate',
        learning_objectives: [`Learn ${subject}`, 'Practice skills', 'Build understanding']
      };
      
      await actions.createSession(sessionData);
    } catch (error) {
      console.error('Error creating session:', error);
      actions.setError('Failed to create session. Please try again.');
    } finally {
      actions.setLoading(false);
    }
  }, [state.user, actions]);

  // Expand chat when messages exist or conversation starts
  useEffect(() => {
    if (state.messages.length > 0 || state.isTyping) {
      setIsChatExpanded(true);
    } else {
      setIsChatExpanded(false);
    }
  }, [state.messages.length, state.isTyping]);

  // Initialize context awareness component when session and user are available
  const contextAwareChat = useCallback(() => {
    if (!state.currentSession?.id || !state.user?.id) return null;
    
    return ContextAwareChatInterface({
      sessionId: state.currentSession.id,
      userId: state.user.id,
      onMessage: (response) => {
        // Handle context-aware message response
        actions.addMessage({
          id: Date.now().toString(),
          message: response.response,
          sender: 'mentor',
          timestamp: new Date().toISOString(),
          learning_mode: 'context_aware',
          suggestions: response.suggested_actions,
          next_steps: response.next_steps,
          metadata: response.metadata
        });
      }
    });
  }, [state.currentSession?.id, state.user?.id, actions]);

  // Improved scroll to bottom function
  const scrollToBottom = useCallback((force = false, smooth = true) => {
    if (!messagesEndRef.current || !messagesContainerRef.current) return;
    
    // Don't auto-scroll if user is reading above and not forced
    if (!force && !isUserNearBottom) return;
    
    isAutoScrollingRef.current = true;
    
    try {
      const container = messagesContainerRef.current;
      const targetScroll = container.scrollHeight - container.clientHeight;
      
      if (smooth) {
        container.scrollTo({
          top: targetScroll,
          behavior: 'smooth'
        });
      } else {
        container.scrollTop = targetScroll;
      }
      
      // Reset auto-scrolling flag after animation
      setTimeout(() => {
        isAutoScrollingRef.current = false;
      }, 200);
      
    } catch (error) {
      console.warn('Scroll error:', error);
      isAutoScrollingRef.current = false;
    }
  }, [isUserNearBottom]);

  // Simplified scroll detection
  const checkScrollPosition = useCallback(() => {
    if (!messagesContainerRef.current || isAutoScrollingRef.current) return;

    const container = messagesContainerRef.current;
    const { scrollTop, scrollHeight, clientHeight } = container;
    
    // Consider user "near bottom" if within 100px
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    const nearBottom = distanceFromBottom <= 100;
    
    setIsUserNearBottom(nearBottom);
    setShowScrollToBottom(!nearBottom);
  }, []);

  // Optimized scroll handler with debouncing
  const handleScroll = useCallback((e) => {
    // Clear previous timeout
    if (autoScrollTimeoutRef.current) {
      clearTimeout(autoScrollTimeoutRef.current);
    }
    
    // Debounce scroll position check
    autoScrollTimeoutRef.current = setTimeout(checkScrollPosition, 50);
  }, [checkScrollPosition]);

  // Auto-scroll for new messages
  useEffect(() => {
    if (state.messages.length > 0) {
      // Small delay to ensure DOM has updated
      setTimeout(() => {
        scrollToBottom(false, true);
      }, 50);
    }
  }, [state.messages.length, scrollToBottom]);

  // Auto-scroll for streaming messages  
  useEffect(() => {
    if (state.streamingMessage && isUserNearBottom) {
      scrollToBottom(false, false); // Use immediate scroll for streaming
    }
  }, [state.streamingMessage, isUserNearBottom, scrollToBottom]);

  // Periodic scroll position check (for edge cases)
  useEffect(() => {
    scrollCheckIntervalRef.current = setInterval(checkScrollPosition, 1000);
    return () => {
      if (scrollCheckIntervalRef.current) {
        clearInterval(scrollCheckIntervalRef.current);
      }
    };
  }, [checkScrollPosition]);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (autoScrollTimeoutRef.current) {
        clearTimeout(autoScrollTimeoutRef.current);
      }
      if (scrollCheckIntervalRef.current) {
        clearInterval(scrollCheckIntervalRef.current);
      }
    };
  }, []);

  // Force scroll to bottom
  const forceScrollToBottom = useCallback(() => {
    setIsUserNearBottom(true);
    scrollToBottom(true, true);
  }, [scrollToBottom]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || !state.currentSession || state.isTyping) return;

    const message = inputMessage.trim();
    const startTime = Date.now();
    setInputMessage('');

    // Force scroll to bottom when sending a message
    forceScrollToBottom();

    // Record learning event for user interaction
    if (state.user?.id && state.currentSession?.id) {
      await recordLearningEvent(state.user.id, state.currentSession.id, {
        type: 'question',
        concept: 'general_learning',
        duration: 0,
        performance: 0.5,
        confidence: 0.5,
        context: {
          message_length: message.length,
          learning_mode: learningMode,
          timestamp: startTime
        }
      });
    }

    try {
      if (useContextAwareness) {
        // Use context-aware premium chat
        const conversationContext = state.messages.map(msg => ({
          sender: msg.sender,
          message: msg.message,
          timestamp: msg.timestamp
        }));
        
        const contextChat = contextAwareChat();
        if (contextChat) {
          await contextChat.sendContextAwareMessage(message, conversationContext);
        } else {
          // Fallback to regular premium chat if context chat not available
          await actions.sendPremiumMessage(state.currentSession.id, message, {
            learning_mode: learningMode,
            user_preferences: {
              difficulty_preference: 'adaptive',
              interaction_style: 'collaborative'
            }
          });
        }
      } else if (useAdvancedStreaming) {
        // Use advanced streaming interface
        // The AdvancedStreamingInterface component will handle the streaming
        return;
      } else {
        // Use regular premium message
        await actions.sendPremiumMessage(state.currentSession.id, message, {
          learning_mode: learningMode,
          user_preferences: {
            difficulty_preference: 'adaptive',
            interaction_style: 'collaborative'
          }
        });
      }
      
      // Ensure scroll to bottom after message is sent
      setTimeout(() => {
        scrollToBottom(true, 'smooth');
      }, 150);
      
    } catch (error) {
      console.error('Error sending premium message:', error);
      // Fallback to regular message
      try {
        await actions.sendMessage(state.currentSession.id, message);
        setTimeout(() => {
          scrollToBottom(true, 'smooth');
        }, 150);
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

  // Add keyboard shortcut for scroll to bottom
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl/Cmd + End to scroll to bottom
      if ((e.ctrlKey || e.metaKey) && e.key === 'End') {
        e.preventDefault();
        forceScrollToBottom();
      }
      // End key to scroll to bottom when in message area
      if (e.key === 'End' && document.activeElement === messagesContainerRef.current) {
        e.preventDefault();
        forceScrollToBottom();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [forceScrollToBottom]);

  if (!state.currentSession) {
    return (
      <div className="h-full flex flex-col">
        {/* Minimal Header for No Session */}
        <div className="flex-shrink-0 glass-medium border-b border-border-subtle p-4">
          <div className="flex items-center justify-center">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 glass-ai-primary rounded-xl flex items-center justify-center">
                <AIBrainIcon size="lg" className="text-ai-blue-400" animated />
              </div>
              <h1 className="text-title font-bold text-gradient-primary">MasterX</h1>
            </div>
          </div>
        </div>

        {/* Centered Welcome Content */}
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="w-full max-w-2xl mx-auto">
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-center mb-8"
            >
              <div className="relative mx-auto mb-6 w-20 h-20">
                <div className="w-full h-full glass-ai-primary rounded-3xl flex items-center justify-center shadow-glow-blue">
                  <AIBrainIcon size="3xl" className="text-ai-blue-400" animated glow />
                </div>
                <PulsingDot size="sm" className="absolute -top-2 -right-2" />
              </div>
              <h1 className="text-4xl font-bold text-gradient-primary mb-4">
                Welcome to MasterX
              </h1>
              <p className="text-lg text-text-secondary mb-8 leading-relaxed max-w-lg mx-auto">
                Your premium AI-powered learning companion. Start a conversation to begin your personalized learning journey.
              </p>
            </motion.div>

            {/* Centered Input Box - Premium Dark Outline Style */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="relative mb-8"
            >
              <div className="relative group">
                {/* Premium Gradient Outline Border */}
                <div className="absolute -inset-[2px] rounded-2xl bg-gradient-to-r from-ai-blue-500 via-ai-purple-500 to-ai-green-500 opacity-60 group-hover:opacity-100 transition-opacity duration-300 animate-gradient-x"></div>
                
                {/* Dark Input Container */}
                <div className="relative bg-gray-900/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-700/50">
                  <div className="flex items-center space-x-4">
                    <div className="flex-1">
                      <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey && inputMessage.trim()) {
                            e.preventDefault();
                            handleQuickStart();
                          }
                        }}
                        onFocus={() => setIsInputFocused(true)}
                        onBlur={() => setIsInputFocused(false)}
                        placeholder="Ask me anything to start learning..."
                        className="w-full bg-transparent border-0 text-lg text-white placeholder-gray-400 focus:outline-none font-medium"
                        disabled={state.isLoading}
                      />
                    </div>
                    
                    {/* Voice Search Icon */}
                    <GlassButton
                      variant="secondary"
                      onClick={() => console.log('Voice search clicked')} // UI placeholder for now
                      className="px-3 py-2"
                      title="Voice search"
                    >
                      <MicrophoneIcon size="md" className="text-gray-400 hover:text-white" />
                    </GlassButton>
                    
                    <GlassButton
                      variant="gradient"
                      onClick={handleQuickStart}
                      disabled={!inputMessage.trim() || state.isLoading}
                      className="px-4 py-2"
                      loading={state.isLoading}
                    >
                      <SendIcon size="md" />
                    </GlassButton>
                  </div>
                  
                  {/* Premium Focus Effect */}
                  <div className={cn(
                    "absolute inset-0 rounded-2xl transition-opacity duration-300 pointer-events-none",
                    "bg-gradient-to-r from-ai-blue-500/5 via-ai-purple-500/5 to-ai-green-500/5",
                    isInputFocused ? "opacity-100" : "opacity-0"
                  )}></div>
                </div>
              </div>
            </motion.div>

            {/* Quick Start Suggestions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="grid grid-cols-2 gap-3 max-w-lg mx-auto"
            >
              {[
                { subject: 'Programming', icon: '💻', color: 'ai-blue-500' },
                { subject: 'Mathematics', icon: '🔢', color: 'ai-purple-500' },
                { subject: 'Science', icon: '🔬', color: 'ai-green-500' },
                { subject: 'Language', icon: '🗣️', color: 'ai-red-500' }
              ].map(({ subject, icon, color }) => (
                <GlassButton
                  key={subject}
                  variant="secondary"
                  onClick={() => handleSubjectStart(subject)}
                  disabled={state.isLoading}
                  className="flex flex-col items-center p-4 hover:glass-thick group"
                >
                  <span className="text-2xl mb-2 transform group-hover:scale-110 transition-transform">
                    {icon}
                  </span>
                  <span className="text-sm font-medium text-text-secondary group-hover:text-text-primary">
                    {subject}
                  </span>
                </GlassButton>
              ))}
            </motion.div>
          </div>
        </div>
      </div>
    );
  }





  return (
    <div className="h-full flex flex-col">
      {/* Minimal Centered Header */}
      <div className="flex-shrink-0 glass-medium border-b border-border-subtle p-4">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="w-10 h-10 glass-ai-primary rounded-xl flex items-center justify-center shadow-glow-blue">
                <AIBrainIcon size="lg" className="text-ai-blue-400" animated />
              </div>
              <div className="absolute -bottom-1 -right-1">
                <PulsingDot size="sm" color="ai-green-500" />
              </div>
            </div>
            <div>
              <h1 className="text-lg font-bold text-gradient-primary">
                MasterX
              </h1>
              <div className="flex items-center space-x-2 text-sm text-text-secondary">
                <span>{state.currentSession.subject || 'Learning'}</span>
                {useContextAwareness && (
                  <>
                    <span>•</span>
                    <span className="text-ai-purple-400 flex items-center space-x-1">
                      <SparkleIcon size="xs" />
                      <span>Smart</span>
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* AR/VR Controls */}
            <GlassButton
              size="sm"
              variant="secondary"
              onClick={() => console.log('AR/VR clicked')} // UI placeholder for now
              title="AR/VR Interface"
            >
              <div className="text-xs font-medium">AR/VR</div>
            </GlassButton>
            
            {/* Gestures Controls */}
            <GlassButton
              size="sm"
              variant="secondary"
              onClick={() => console.log('Gestures clicked')} // UI placeholder for now
              title="Gesture Controls"
            >
              <div className="text-xs font-medium">Gestures</div>
            </GlassButton>
            
            {/* Normal/Live Chat Toggle */}
            <GlassButton
              size="sm"
              variant={currentView === 'live-learning' ? 'gradient' : 'secondary'}
              onClick={() => setCurrentView(currentView === 'chat' ? 'live-learning' : 'chat')}
              title="Toggle Chat Mode"
            >
              <div className="text-xs font-medium">{currentView === 'live-learning' ? 'Live' : 'Chat'}</div>
            </GlassButton>
            
            <GlassButton
              size="sm"
              variant={useContextAwareness ? 'gradient' : 'secondary'}
              onClick={() => setUseContextAwareness(!useContextAwareness)}
              title="Toggle Context Awareness"
            >
              <AIBrainIcon size="sm" />
            </GlassButton>
            
            <GlassButton 
              size="sm" 
              variant="secondary"
              onClick={() => setShowThemePanel(true)}
              title="Settings"
            >
              <SettingsIcon size="sm" />
            </GlassButton>
            
            {/* User Profile Button - ChatGPT Style */}
            {state.user && (
              <GlassButton
                size="sm"
                variant="secondary"
                onClick={() => console.log('User profile clicked')} // UI placeholder for now
                title={`${state.user.name} Profile`}
                className="p-1"
              >
                <div className="w-8 h-8 bg-gradient-to-br from-ai-green-400 to-ai-blue-500 rounded-lg flex items-center justify-center shadow-lg">
                  <span className="text-white text-sm font-bold">
                    {state.user.name.charAt(0).toUpperCase()}
                  </span>
                </div>
              </GlassButton>
            )}
            
            <LearningModeIndicator 
              currentMode={learningMode}
              onClick={() => setShowLearningModes(true)}
            />
          </div>
        </div>
      </div>

      {/* Main Chat Area - Centered and Expandable */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Messages Container - ChatGPT Style */}
        <motion.div 
          className={cn(
            "flex-1 overflow-y-auto transition-all duration-500 ease-out",
            isChatExpanded ? "p-4" : "flex items-end pb-4"
          )}
          ref={messagesContainerRef}
          onScroll={handleScroll}
        >
          <div className={cn(
            "w-full mx-auto transition-all duration-500",
            isChatExpanded ? "max-w-4xl" : "max-w-2xl"
          )}>
            <AnimatePresence>
              {state.messages.map((message, index) => (
                <ChatMessage key={message.id || index} message={message} isExpanded={isChatExpanded} />
              ))}
              
              {/* Premium Streaming message */}
              {state.isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mb-6 flex justify-start"
                >
                  <div className="flex items-start space-x-3 max-w-full w-full">
                    <div className="flex-shrink-0 mt-1">
                      <div className="w-8 h-8 glass-ai-primary rounded-full flex items-center justify-center shadow-glow-blue">
                        <AIBrainIcon size="sm" className="text-ai-blue-400" animated />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0 max-w-4xl">
                      <div className="mb-2">
                        <span className="text-sm font-medium text-text-primary">MasterX</span>
                        <span className="ml-2 text-xs text-ai-blue-400 flex items-center space-x-1">
                          <SparkleIcon size="xs" />
                          <span>Streaming</span>
                        </span>
                      </div>
                      
                      <GlassCard 
                        variant="ai-primary" 
                        size="sm" 
                        className="glass-medium shadow-lg bg-glass-light border border-border-subtle"
                      >
                        {state.streamingMessage ? (
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
                                  code: ({children}) => <code className="px-2 py-1 bg-glass-thick rounded text-ai-green-300 text-sm font-mono">{children}</code>
                                }}
                              >
                                {state.streamingMessage}
                              </ReactMarkdown>
                              <motion.span 
                                className="inline-block w-0.5 h-4 bg-ai-blue-400 ml-1"
                                animate={{ opacity: [1, 0] }}
                                transition={{ duration: 0.8, repeat: Infinity }}
                              />
                            </div>
                          </div>
                        ) : (
                          <TypingIndicator size="sm" message="MasterX is thinking..." />
                        )}
                      </GlassCard>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        </motion.div>

        {/* Enhanced Scroll to Bottom Button */}
        <AnimatePresence>
          {showScrollToBottom && isChatExpanded && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={forceScrollToBottom}
              className="absolute bottom-24 right-8 w-12 h-12 glass-ai-primary rounded-full flex items-center justify-center hover:glass-thick transition-all duration-300 z-20 shadow-glow-blue"
            >
              <ArrowDownIcon size="sm" className="text-ai-blue-400" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Input Area - Premium Dark Outline Style */}
        <div className="flex-shrink-0 p-4">
          <div className={cn(
            "mx-auto transition-all duration-500",
            // Match the chat messages width with slightly wider margins
            isChatExpanded ? "max-w-3xl" : "max-w-xl"
          )}>
            <form onSubmit={handleSendMessage} className="relative">
              <div className="relative group">
                {/* Premium Gradient Outline Border */}
                <div className={cn(
                  "absolute -inset-[2px] rounded-2xl transition-all duration-300",
                  "bg-gradient-to-r from-ai-blue-500 via-ai-purple-500 to-ai-green-500",
                  isInputFocused ? "opacity-100 animate-gradient-x" : "opacity-60"
                )}></div>
                
                {/* Dark Input Container with Transparent Background */}
                <div className="relative bg-gray-900/80 backdrop-blur-xl rounded-2xl border border-gray-700/50">
                  <div className="flex items-center p-4 space-x-4">
                    <div className="flex-1 relative">
                      <GlassInput
                        ref={inputRef}
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        onFocus={() => setIsInputFocused(true)}
                        onBlur={() => setIsInputFocused(false)}
                        placeholder={useContextAwareness ? "Message MasterX..." : "Ask me anything..."}
                        className="w-full bg-transparent border-0 text-base text-white placeholder-gray-400 focus:outline-none resize-none min-h-[24px] max-h-32"
                        disabled={state.isTyping}
                        multiline
                      />
                      
                      {/* Context Indicator */}
                      {useContextAwareness && (
                        <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
                          <div className="flex items-center space-x-1 px-2 py-1 bg-ai-purple-500/20 rounded-lg border border-ai-purple-400/30">
                            <AIBrainIcon size="xs" className="text-ai-purple-400" />
                            <span className="text-xs text-ai-purple-300">Smart</span>
                          </div>
                        </div>
                      )}
                    </div>
                    
                    {/* Voice Search Icon */}
                    <GlassButton
                      variant="secondary"
                      onClick={() => console.log('Voice search clicked')} // UI placeholder for now
                      className="px-3 py-2 flex-shrink-0"
                      title="Voice search"
                    >
                      <MicrophoneIcon size="sm" className="text-gray-400 hover:text-white" />
                    </GlassButton>
                    
                    <GlassButton
                      type="submit"
                      variant="gradient"
                      disabled={!inputMessage.trim() || state.isTyping}
                      className="px-4 py-2 flex-shrink-0"
                      loading={state.isTyping}
                    >
                      <SendIcon size="sm" />
                    </GlassButton>
                  </div>
                  
                  {/* Premium Focus Effect */}
                  <div className={cn(
                    "absolute inset-0 rounded-2xl transition-opacity duration-300 pointer-events-none",
                    "bg-gradient-to-r from-ai-blue-500/5 via-ai-purple-500/5 to-ai-green-500/5",
                    isInputFocused ? "opacity-100" : "opacity-0"
                  )}></div>
                </div>
              </div>
              
              {/* Quick Actions - Only show when expanded */}
              <AnimatePresence>
                {isChatExpanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex flex-wrap gap-2 mt-3 justify-center"
                  >
                    <GlassButton
                      size="sm"
                      variant={useContextAwareness ? "gradient" : "secondary"}
                      onClick={() => setUseContextAwareness(!useContextAwareness)}
                    >
                      <AIBrainIcon size="xs" className="mr-1" />
                      Smart Mode
                    </GlassButton>
                    <GlassButton
                      size="sm"
                      variant="tertiary"
                      onClick={() => setInputMessage("Can you create an exercise for me?")}
                      disabled={state.isTyping}
                    >
                      Exercise
                    </GlassButton>
                    <GlassButton
                      size="sm"
                      variant="tertiary"
                      onClick={() => setInputMessage("Explain this step by step")}
                      disabled={state.isTyping}
                    >
                      Step-by-Step
                    </GlassButton>
                    <GlassButton
                      size="sm"
                      variant="tertiary"
                      onClick={() => setInputMessage("Give me a real-world example")}
                      disabled={state.isTyping}
                    >
                      Example
                    </GlassButton>
                  </motion.div>
                )}
              </AnimatePresence>
            </form>
          </div>
        </div>
      </div>

      {/* Modals - Keep existing functionality */}
      <GamificationDashboard
        isVisible={showGamification}
        onClose={() => setShowGamification(false)}
      />

      <PremiumLearningModes
        currentMode={learningMode}
        onModeChange={setLearningMode}
        isVisible={showLearningModes}
        onClose={() => setShowLearningModes(false)}
      />

      <ModelManagement
        isVisible={showModelManagement}
        onClose={() => setShowModelManagement(false)}
      />

      <AdaptiveThemePanel
        isOpen={showThemePanel}
        onClose={() => setShowThemePanel(false)}
      />
    </div>
  );
}

function ChatMessage({ message, isExpanded = true }) {
  const isUser = message.sender === 'user';
  const isTyping = message.sender === 'mentor' && !message.message;
  const isPremium = message.learning_mode && message.learning_mode !== 'adaptive';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="mb-6 flex justify-start"
    >
      <div className="flex items-start space-x-3 max-w-full w-full">
        {/* Avatar */}
        <div className="flex-shrink-0 mt-1">
          <div className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center",
            isUser 
              ? "bg-gradient-to-br from-ai-green-400 to-ai-blue-500 shadow-lg" 
              : isPremium 
                ? "glass-ai-secondary shadow-glow-purple" 
                : "glass-ai-primary shadow-glow-blue"
          )}>
            {isUser ? (
              <span className="text-white text-sm font-bold">
                {/* Get first letter of user name if available */}
                U
              </span>
            ) : (
              <AIBrainIcon 
                size="sm" 
                className={isPremium ? 'text-ai-purple-400' : 'text-ai-blue-400'}
                animated
              />
            )}
          </div>
        </div>
        
        {/* Message Content */}
        <div className="flex-1 min-w-0 max-w-4xl">
          {/* User Label */}
          <div className="mb-2">
            <span className="text-sm font-medium text-text-primary">
              {isUser ? 'You' : 'MasterX'}
            </span>
            {isPremium && !isUser && (
              <span className="ml-2 text-xs text-ai-purple-400 flex items-center space-x-1">
                <SparkleIcon size="xs" />
                <span>Premium</span>
              </span>
            )}
          </div>
          
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
              "glass-medium shadow-lg",
              isUser 
                ? "bg-ai-blue-500/10 border border-ai-blue-500/20" 
                : isPremium
                  ? "bg-ai-purple-500/5 border border-ai-purple-500/20"
                  : "bg-glass-light border border-border-subtle"
            )}
            hover={false}
          >
            {/* Premium Mode Indicator */}
            {isPremium && !isUser && (
              <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-border-subtle">
                <SparkleIcon size="sm" className="text-ai-purple-400" />
                <span className="text-sm font-semibold text-ai-purple-300 capitalize">
                  {message.learning_mode} Mode
                </span>
                <div className="ml-auto">
                  <CheckIcon size="sm" className="text-ai-green-400" />
                </div>
              </div>
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
