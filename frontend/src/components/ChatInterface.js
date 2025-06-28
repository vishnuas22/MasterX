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
  CheckIcon
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
import { cn } from '../utils/cn';

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
  
  // Simplified and robust scroll management
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);
  const [isUserNearBottom, setIsUserNearBottom] = useState(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const autoScrollTimeoutRef = useRef(null);
  const scrollCheckIntervalRef = useRef(null);
  const isAutoScrollingRef = useRef(false);

  // Initialize context awareness component
  const contextAwareChat = ContextAwareChatInterface({
    sessionId: state.currentSession?.id,
    userId: state.user?.id,
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
        
        await contextAwareChat.sendContextAwareMessage(message, conversationContext);
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
      <div className="flex-1 flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <GlassCard variant="ai-primary" size="lg" className="text-center max-w-lg">
            <div className="mb-8">
              <div className="relative mx-auto mb-6">
                <div className="w-20 h-20 glass-ai-primary rounded-3xl mx-auto flex items-center justify-center shadow-glow-blue">
                  <AIBrainIcon size="3xl" className="text-ai-blue-400" animated glow />
                </div>
                <PulsingDot size="sm" className="absolute -top-2 -right-2" />
              </div>
              <h1 className="text-headline-large font-bold text-gradient-primary mb-3">
                Welcome to MasterX
              </h1>
              <p className="text-body text-text-secondary mb-8 leading-relaxed">
                Your premium AI-powered learning companion. Start a new session to begin your personalized learning journey with advanced AI mentoring.
              </p>
            </div>
            
            {/* Enhanced Quick start options */}
            <div className="space-y-6">
              <GlassButton
                variant="gradient"
                size="lg"
                onClick={async () => {
                  try {
                    if (!state.user?.id) {
                      actions.setError('User not found. Please refresh and try again.');
                      return;
                    }

                    actions.setLoading(true);
                    
                    const sessionData = {
                      user_id: state.user.id,
                      subject: 'General Learning',
                      difficulty_level: 'intermediate',
                      learning_objectives: ['Explore new topics', 'Interactive learning', 'Skill development']
                    };
                    
                    await actions.createSession(sessionData);
                  } catch (error) {
                    console.error('Error creating session:', error);
                    actions.setError('Failed to create session. Please try again.');
                  } finally {
                    actions.setLoading(false);
                  }
                }}
                className="w-full"
                disabled={state.isLoading}
                loading={state.isLoading}
              >
                <BookIcon size="md" className="mr-3" />
                {state.isLoading ? 'Creating Session...' : 'Start Learning Journey'}
              </GlassButton>
              
              {/* Subject-specific quick start buttons */}
              <div>
                <h3 className="text-title font-semibold text-text-primary mb-4">
                  Or choose a subject:
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { subject: 'Programming', icon: '💻', color: 'ai-blue-500' },
                    { subject: 'Mathematics', icon: '🔢', color: 'ai-purple-500' },
                    { subject: 'Science', icon: '🔬', color: 'ai-green-500' },
                    { subject: 'Language', icon: '🗣️', color: 'ai-red-500' }
                  ].map(({ subject, icon, color }) => (
                    <GlassButton
                      key={subject}
                      variant="secondary"
                      onClick={async () => {
                        try {
                          if (!state.user?.id) {
                            actions.setError('User not found. Please refresh and try again.');
                            return;
                          }

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
                      }}
                      disabled={state.isLoading}
                      className="flex flex-col items-center p-4 hover:glass-thick"
                    >
                      <span className="text-2xl mb-2 transform group-hover:scale-110 transition-transform">
                        {icon}
                      </span>
                      <span className="text-caption font-medium text-text-secondary group-hover:text-text-primary">
                        {subject}
                      </span>
                    </GlassButton>
                  ))}
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Premium Chat Header */}
      <div className="flex-shrink-0 glass-medium border-b border-border-subtle p-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="w-12 h-12 glass-ai-primary rounded-2xl flex items-center justify-center shadow-glow-blue">
                <AIBrainIcon size="xl" className="text-ai-blue-400" animated />
              </div>
              <div className="absolute -bottom-1 -right-1">
                <PulsingDot size="sm" color="ai-green-500" />
              </div>
              {useContextAwareness && (
                <div className="absolute -top-1 -left-1 w-4 h-4 glass-ai-secondary rounded-full flex items-center justify-center" title="Context Awareness Active">
                  <AIBrainIcon size="xs" className="text-ai-purple-400" />
                </div>
              )}
            </div>
            <div>
              <h1 className="text-title-large font-bold text-gradient-primary">
                MasterX AI Mentor
              </h1>
              <div className="flex items-center space-x-2 text-body text-text-secondary">
                <span>{state.currentSession.subject || 'General Learning'}</span>
                <span>•</span>
                <span className="capitalize">{state.currentSession.difficulty_level}</span>
                {useContextAwareness && (
                  <>
                    <span>•</span>
                    <span className="text-ai-purple-400 flex items-center space-x-1">
                      <SparkleIcon size="xs" />
                      <span>Context Aware</span>
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* View Toggle */}
            <div className="flex glass-medium rounded-xl p-1">
              <GlassButton
                size="sm"
                variant={currentView === 'chat' ? 'primary' : 'tertiary'}
                onClick={() => setCurrentView('chat')}
                className="px-4 py-2"
              >
                Chat
              </GlassButton>
              <GlassButton
                size="sm"
                variant={currentView === 'live-learning' ? 'primary' : 'tertiary'}
                onClick={() => setCurrentView('live-learning')}
                className="px-4 py-2"
              >
                <ZapIcon size="sm" className="mr-2" />
                Live
              </GlassButton>
            </div>
            
            {/* Premium Action Buttons */}
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
              title="Adaptive Theme Settings"
            >
              <SettingsIcon size="sm" />
            </GlassButton>
            
            <LearningModeIndicator 
              currentMode={learningMode}
              onClick={() => setShowLearningModes(true)}
            />
            
            <GlassButton 
              size="sm" 
              variant="secondary"
              onClick={() => setShowGamification(true)}
              title="Gamification Dashboard"
            >
              <TrophyIcon size="sm" />
            </GlassButton>
            
            <GlassButton 
              size="sm" 
              variant="secondary"
              onClick={() => setShowModelManagement(true)}
              title="AI Model Management"
            >
              <SettingsIcon size="sm" />
            </GlassButton>
          </div>
        </div>
        
        {/* Context Awareness Panel */}
        {useContextAwareness && currentView === 'chat' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 pt-4 border-t border-border-subtle"
          >
            <contextAwareChat.ContextInsightsPanel />
          </motion.div>
        )}
      </div>

      {/* Main Content Area */}
      <AnimatePresence mode="wait">
        {currentView === 'chat' ? (
          <motion.div
            key="chat"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="flex-1 flex flex-col overflow-hidden"
          >
            {/* Premium Messages Area with enhanced scroll behavior */}
            <div 
              ref={messagesContainerRef}
              onScroll={handleScroll}
              className="flex-1 overflow-y-auto p-6 space-y-6"
              data-chat-container
            >
              {/* Enhanced Scroll Indicator */}
              <div className="absolute top-0 right-2 w-1 h-full glass-ultra-thin rounded-full overflow-hidden">
                <motion.div 
                  className="w-full bg-gradient-to-b from-ai-blue-400 to-ai-purple-500 rounded-full transition-all duration-300"
                  style={{
                    height: `${Math.min(100, Math.max(10, ((messagesContainerRef.current?.scrollTop || 0) / Math.max(1, (messagesContainerRef.current?.scrollHeight || 1) - (messagesContainerRef.current?.clientHeight || 1))) * 100))}%`,
                    transform: `translateY(${((messagesContainerRef.current?.scrollTop || 0) / Math.max(1, (messagesContainerRef.current?.scrollHeight || 1) - (messagesContainerRef.current?.clientHeight || 1))) * 100}%)`
                  }}
                />
              </div>
              
              <AnimatePresence>
                {state.messages.map((message, index) => (
                  <ChatMessage key={message.id || index} message={message} />
                ))}
                
                {/* Advanced Streaming Interface */}
                {useAdvancedStreaming && inputMessage && (
                  <AdvancedStreamingInterface
                    message={inputMessage}
                    sessionId={state.currentSession?.id}
                    onStreamComplete={(result) => {
                      setUseAdvancedStreaming(false);
                      setInputMessage('');
                    }}
                    streamingConfig={{
                      multiBranch: true,
                      factCheck: true,
                      interruptible: true
                    }}
                  />
                )}
                
                {/* Premium Streaming message */}
                {state.isTyping && !useAdvancedStreaming && (
                  <motion.div
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -20, scale: 0.95 }}
                    transition={{ duration: 0.3, ease: "easeOut" }}
                    className="flex items-start space-x-4"
                  >
                    <div className="flex-shrink-0">
                      <div className="w-10 h-10 glass-ai-primary rounded-2xl flex items-center justify-center shadow-glow-blue">
                        <AIBrainIcon size="lg" className="text-ai-blue-400" animated />
                      </div>
                    </div>
                    <div className="flex-1">
                      <GlassCard variant="ai-primary" size="md">
                        {state.streamingMessage ? (
                          <div className="prose-premium max-w-none">
                            <ReactMarkdown>{state.streamingMessage}</ReactMarkdown>
                            <motion.span 
                              className="inline-block w-0.5 h-5 bg-ai-blue-400 ml-1"
                              animate={{ opacity: [1, 0] }}
                              transition={{ duration: 0.8, repeat: Infinity }}
                            />
                          </div>
                        ) : (
                          <TypingIndicator size="md" message="AI is thinking..." />
                        )}
                      </GlassCard>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>

            {/* Enhanced Scroll to Bottom Button */}
            <AnimatePresence>
              {showScrollToBottom && (
                <motion.button
                  initial={{ opacity: 0, scale: 0.8, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.8, y: 20 }}
                  transition={{ type: "spring", damping: 25, stiffness: 300 }}
                  onClick={forceScrollToBottom}
                  className="absolute bottom-24 right-8 w-14 h-14 glass-ai-primary rounded-2xl flex items-center justify-center hover:glass-thick transition-all duration-300 z-20 shadow-glow-blue hover:scale-110 group"
                  title="Scroll to bottom (Ctrl+End)"
                  whileHover={{ scale: 1.1, rotate: 5 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <ArrowDownIcon size="lg" className="text-ai-blue-400 group-hover:text-ai-blue-300" />
                  {state.messages.length > 0 && (
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-ai-blue-500 rounded-full flex items-center justify-center shadow-lg">
                      <span className="text-xs text-white font-bold font-primary">!</span>
                    </div>
                  )}
                </motion.button>
              )}
            </AnimatePresence>

            {/* Premium Input Area */}
            <div className="flex-shrink-0 glass-medium border-t border-border-subtle p-6 shadow-xl">
              <form onSubmit={handleSendMessage} className="space-y-4">
                <div className="flex space-x-4">
                  <div className="flex-1 relative">
                    <GlassInput
                      ref={inputRef}
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={useContextAwareness ? "Ask me anything... (Context aware AI)" : "Ask me anything..."}
                      className="w-full text-body pr-20"
                      disabled={state.isTyping}
                    />
                    <div className="absolute right-4 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
                      {useContextAwareness && (
                        <div className="flex items-center space-x-1 px-2 py-1 glass-ai-secondary rounded-lg">
                          <AIBrainIcon size="xs" className="text-ai-purple-400" />
                          <span className="text-footnote text-ai-purple-300">Smart</span>
                        </div>
                      )}
                      <SparkleIcon size="sm" className="text-ai-blue-400" animated />
                    </div>
                  </div>
                  <GlassButton
                    type="submit"
                    variant="gradient"
                    disabled={!inputMessage.trim() || state.isTyping}
                    className="px-6"
                    loading={state.isTyping}
                  >
                    <SendIcon size="md" />
                  </GlassButton>
                </div>
                
                {/* Enhanced Quick Actions */}
                <div className="flex flex-wrap gap-2">
                  <GlassButton
                    size="sm"
                    variant={useContextAwareness ? "gradient" : "secondary"}
                    onClick={() => setUseContextAwareness(!useContextAwareness)}
                  >
                    <AIBrainIcon size="sm" className="mr-2" />
                    Context Aware
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant={useAdvancedStreaming ? "primary" : "secondary"}
                    onClick={() => setUseAdvancedStreaming(!useAdvancedStreaming)}
                  >
                    <SparkleIcon size="sm" className="mr-2" />
                    Advanced Stream
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant="tertiary"
                    onClick={() => setInputMessage("Can you create an exercise for me?")}
                    disabled={state.isTyping}
                  >
                    Generate Exercise
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant="tertiary"
                    onClick={() => setInputMessage("Explain this concept step by step")}
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
                    Real Example
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant="tertiary"
                    onClick={() => {
                      setLearningMode('challenge');
                      setInputMessage("Give me a challenge problem");
                    }}
                    disabled={state.isTyping}
                  >
                    <TargetIcon size="sm" className="mr-1" />
                    Challenge Me
                  </GlassButton>
                </div>
              </form>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="live-learning"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="flex-1"
          >
            <LiveLearningInterface
              userId={state.user?.id}
              onSessionUpdate={(session) => {
                console.log('Live session updated:', session);
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Advanced UI Features */}
      <VoiceInterface />
      <GestureControl />
      <ARVRInterface />

      {/* Modals */}
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

function ChatMessage({ message }) {
  const isUser = message.sender === 'user';
  const isTyping = message.sender === 'mentor' && !message.message;
  const isPremium = message.learning_mode && message.learning_mode !== 'adaptive';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={cn(
        'flex items-start space-x-4',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div className={cn(
            'w-10 h-10 rounded-2xl flex items-center justify-center shadow-lg',
            isPremium 
              ? 'glass-ai-secondary shadow-glow-purple' 
              : 'glass-ai-primary shadow-glow-blue'
          )}>
            <AIBrainIcon 
              size="lg" 
              className={isPremium ? 'text-ai-purple-400' : 'text-ai-blue-400'}
              animated
            />
          </div>
        </div>
      )}
      
      <div className={cn('max-w-[75%] min-w-0', isUser ? 'order-first' : '')}>
        <GlassCard 
          variant={
            isUser 
              ? 'ai-primary'
              : isPremium
                ? 'ai-secondary'
                : 'medium'
          }
          size="md"
          className={cn(
            'relative',
            isUser && 'ml-auto'
          )}
          hover={false}
        >
          {/* Premium Mode Indicator */}
          {isPremium && (
            <div className="flex items-center space-x-2 mb-3 pb-3 border-b border-border-subtle">
              <SparkleIcon size="sm" className="text-ai-purple-400" />
              <span className="text-caption font-semibold text-ai-purple-300 capitalize">
                {message.learning_mode} Mode
              </span>
              <div className="ml-auto">
                <CheckIcon size="sm" className="text-ai-green-400" />
              </div>
            </div>
          )}
          
          {/* Message Content */}
          {message.message ? (
            <div className="prose-premium max-w-none">
              <ReactMarkdown>{message.message}</ReactMarkdown>
            </div>
          ) : (
            <TypingIndicator />
          )}
          
          {/* Message Suggestions */}
          {!isUser && message.suggestions && message.suggestions.length > 0 && (
            <div className="mt-4 pt-4 border-t border-border-subtle">
              <p className="text-caption text-text-tertiary mb-3 flex items-center space-x-2">
                <SparkleIcon size="xs" />
                <span>Suggested actions:</span>
              </p>
              <div className="flex flex-wrap gap-2">
                {message.suggestions.map((suggestion, index) => (
                  <GlassButton
                    key={index}
                    size="sm"
                    variant="tertiary"
                    className="text-caption hover:variant-secondary"
                  >
                    {suggestion}
                  </GlassButton>
                ))}
              </div>
            </div>
          )}

          {/* Next Steps for Premium Responses */}
          {!isUser && message.next_steps && (
            <div className="mt-4 pt-4 border-t border-border-subtle">
              <p className="text-caption text-ai-blue-400 mb-2 flex items-center space-x-2 font-semibold">
                <TargetIcon size="sm" />
                <span>Next Steps:</span>
              </p>
              <p className="text-caption text-text-secondary leading-relaxed">
                {message.next_steps}
              </p>
            </div>
          )}
        </GlassCard>
        
        {/* Message Metadata */}
        <div className={cn(
          'flex items-center mt-2 text-footnote text-text-quaternary space-x-2',
          isUser ? 'justify-end' : 'justify-start'
        )}>
          <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
          {isPremium && (
            <>
              <span>•</span>
              <div className="flex items-center space-x-1">
                <SparkleIcon size="xs" className="text-ai-purple-400" />
                <span className="text-ai-purple-400 font-medium">Premium</span>
              </div>
            </>
          )}
          {message.metadata?.personalization_score && (
            <>
              <span>•</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-ai-green-500 rounded-full" />
                <span className="text-ai-green-400">
                  {Math.round(message.metadata.personalization_score * 100)}% personalized
                </span>
              </div>
            </>
          )}
        </div>
      </div>
      
      {isUser && (
        <div className="flex-shrink-0">
          <div className="w-10 h-10 glass-thick rounded-2xl flex items-center justify-center shadow-lg">
            <UserIcon size="lg" className="text-text-primary" />
          </div>
        </div>
      )}
    </motion.div>
  );
}
