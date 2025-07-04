import React, { useState, useEffect, lazy, Suspense } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { AppProvider, useApp } from "./context/AppContext";
import { ThemeProvider } from "./components/AdaptiveThemeSystem";
import { UserOnboarding } from "./components/UserOnboarding";
import { Sidebar } from "./components/Sidebar";
import { ChatInterface } from "./components/ChatInterface";
import { PageLoadingOverlay, LoadingStates } from "./components/LoadingSpinner";
import { GlassCard, GlassButton } from "./components/GlassCard";
import { AlertIcon, CheckIcon, MasterXIcon } from "./components/PremiumIcons";
import { api } from "./services/api";
import { cn } from "./utils/cn";

// ===============================
// 🎨 LAZY LOADED COMPONENTS
// ===============================

const LearningPsychologyDashboard = lazy(() => import("./components/LearningPsychologyDashboard"));
const MetacognitiveTraining = lazy(() => import("./components/MetacognitiveTraining"));
const MemoryPalaceBuilder = lazy(() => import("./components/MemoryPalaceBuilder"));
const PersonalizationDashboard = lazy(() => import("./components/PersonalizationDashboard"));
const AdvancedAnalyticsLearningDashboard = lazy(() => import("./components/AdvancedAnalyticsLearningDashboard"));

// ===============================
// 🎨 PREMIUM ERROR BOUNDARY
// ===============================

class PremiumErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-bg-primary via-bg-secondary to-bg-tertiary flex items-center justify-center p-6">
          <GlassCard size="lg" className="text-center max-w-md">
            <AlertIcon size="3xl" className="text-ai-red-500 mx-auto mb-6" />
            <h1 className="text-headline font-bold text-text-primary mb-4">
              Something went wrong
            </h1>
            <p className="text-body text-text-secondary mb-6">
              We encountered an unexpected error. Please refresh the page to continue.
            </p>
            <GlassButton 
              variant="primary"
              onClick={() => window.location.reload()}
              className="w-full"
            >
              Refresh Page
            </GlassButton>
          </GlassCard>
        </div>
      );
    }

    return this.props.children;
  }
}

// ===============================
// 🎨 MAIN APP CONTENT COMPONENT
// ===============================

function AppContent() {
  const { state, actions } = useApp();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [appReady, setAppReady] = useState(false);

  // Enhanced connection check with retry logic
  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 3;
    const retryDelay = 2000;

    const checkConnection = async () => {
      try {
        await api.healthCheck();
        setConnectionStatus('connected');
        setAppReady(true);
      } catch (error) {
        console.error('Backend connection failed:', error);
        
        if (retryCount < maxRetries) {
          retryCount++;
          setConnectionStatus('retrying');
          setTimeout(checkConnection, retryDelay);
        } else {
          setConnectionStatus('error');
        }
      }
    };

    checkConnection();
  }, []);

  // Show onboarding if no user
  if (!state.user && appReady) {
    console.log('📱 Showing onboarding - no user found, app ready');
    return <UserOnboarding />;
  }

  // Show main app if user exists
  if (state.user && appReady) {
    console.log('🎯 Showing main app - user found:', state.user.name || state.user.email);
    
    // Main application layout
    return (
      <div className="h-screen bg-gradient-to-br from-bg-primary via-bg-secondary to-bg-tertiary text-text-primary overflow-hidden">
        {/* Premium Background Effects */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Animated Gradient Mesh */}
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-ai-blue-500/10 to-transparent rounded-full blur-3xl animate-float" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-ai-purple-500/10 to-transparent rounded-full blur-3xl animate-float" style={{ animationDelay: '1s' }} />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-ai-green-500/5 to-transparent rounded-full blur-3xl animate-pulse-soft" />
          
          {/* Premium Grid Pattern */}
          <div 
            className="absolute inset-0 opacity-[0.02]"
            style={{
              backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0)`,
              backgroundSize: '24px 24px'
            }}
          />
        </div>

        {/* Main Layout Container */}
        <div className="relative z-10 flex h-full">
          {/* Enhanced Sidebar */}
          <AnimatePresence>
            <motion.div
              initial={{ x: -300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -300, opacity: 0 }}
              transition={{ 
                type: "spring", 
                damping: 25, 
                stiffness: 200,
                staggerChildren: 0.1 
              }}
            >
              <Sidebar 
                isCollapsed={sidebarCollapsed} 
                onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} 
              />
            </motion.div>
          </AnimatePresence>

          {/* Main Content Area */}
          <motion.main
            className="flex-1 flex flex-col overflow-hidden relative min-h-0"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.6, ease: "easeOut" }}
          >
            <Suspense 
              fallback={
                <div className="flex-1 flex items-center justify-center">
                  <LoadingStates state="loading" size="xl" />
                </div>
              }
            >
              <AnimatePresence mode="wait">
                {renderActiveView(state.activeView, state.user)}
              </AnimatePresence>
            </Suspense>
          </motion.main>
        </div>

        {/* Premium Error Display */}
        <AnimatePresence>
          {state.error && (
            <motion.div
              initial={{ opacity: 0, y: 100, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 100, scale: 0.9 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="fixed bottom-6 right-6 z-50 max-w-sm"
            >
              <GlassCard variant="ai-secondary" className="border-ai-red-500/30">
                <div className="flex items-start space-x-3">
                  <AlertIcon size="lg" className="text-ai-red-400 flex-shrink-0 mt-1" />
                  <div className="flex-1 min-w-0">
                    <h4 className="text-body font-semibold text-ai-red-300 mb-1">
                      Error
                    </h4>
                    <p className="text-caption text-text-secondary break-words">
                      {state.error}
                    </p>
                  </div>
                  <button
                    onClick={() => actions.setError(null)}
                    className="text-ai-red-300 hover:text-ai-red-100 transition-colors p-1 rounded-lg hover:bg-ai-red-500/10"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  // Show connection error with premium styling
  if (connectionStatus === 'error') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-bg-primary via-bg-secondary to-bg-tertiary flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <GlassCard size="lg" className="max-w-md">
            <div className="mb-6">
              <div className="w-16 h-16 mx-auto mb-4 text-ai-red-500">
                <AlertIcon size="3xl" />
              </div>
              <h2 className="text-headline font-bold text-text-primary mb-2">
                Connection Error
              </h2>
              <p className="text-body text-text-secondary mb-6">
                Unable to connect to MasterX AI Mentor System. Please check your internet connection and try again.
              </p>
            </div>
            
            <div className="space-y-3">
              <GlassButton
                variant="primary"
                onClick={() => window.location.reload()}
                className="w-full"
              >
                Retry Connection
              </GlassButton>
              <GlassButton
                variant="secondary"
                onClick={() => setConnectionStatus('checking')}
                className="w-full"
              >
                Check Again
              </GlassButton>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    );
  }

  // Show loading while checking connection or app not ready
  if (connectionStatus === 'checking' || connectionStatus === 'retrying' || !appReady) {
    return (
      <PageLoadingOverlay
        isVisible={true}
        message={connectionStatus === 'retrying' ? 'Retrying connection...' : 'Connecting to MasterX...'}
      />
    );
  }


}

// ===============================
// 🎨 VIEW RENDERING FUNCTION
// ===============================

function renderActiveView(activeView, user) {
  const viewTransition = {
    initial: { opacity: 0, x: 20, scale: 0.98 },
    animate: { opacity: 1, x: 0, scale: 1 },
    exit: { opacity: 0, x: -20, scale: 0.98 },
    transition: { duration: 0.4, ease: "easeInOut" }
  };

  const comingSoonViews = [
    'elaborative-questions',
    'transfer-learning', 
    'achievements',
    'settings'
  ];

  if (comingSoonViews.includes(activeView)) {
    return (
      <motion.div
        key={activeView}
        {...viewTransition}
        className="flex-1 flex items-center justify-center p-6"
      >
        <GlassCard size="lg" className="text-center max-w-md">
          <div className="mb-6">
            <MasterXIcon size="3xl" className="text-ai-blue-500 mx-auto mb-4" animated />
          </div>
          <h2 className="text-headline font-bold text-text-primary mb-3">
            {getViewTitle(activeView)}
          </h2>
          <p className="text-body text-text-secondary mb-6">
            {getViewDescription(activeView)}
          </p>
          <div className="flex items-center justify-center space-x-2 text-ai-blue-400">
            <div className="w-2 h-2 bg-ai-blue-400 rounded-full animate-pulse" />
            <span className="text-caption font-medium">Coming Soon</span>
          </div>
        </GlassCard>
      </motion.div>
    );
  }

  switch (activeView) {
    case 'chat':
      return (
        <motion.div key="chat" {...viewTransition} className="flex-1 flex flex-col min-h-0">
          <ChatInterface />
        </motion.div>
      );
    case 'personalization':
      return (
        <motion.div key="personalization" {...viewTransition} className="flex-1">
          <PersonalizationDashboard />
        </motion.div>
      );
    case 'learning-psychology':
      return (
        <motion.div key="learning-psychology" {...viewTransition} className="flex-1">
          <LearningPsychologyDashboard />
        </motion.div>
      );
    case 'metacognitive-training':
      return (
        <motion.div key="metacognitive-training" {...viewTransition} className="flex-1">
          <MetacognitiveTraining />
        </motion.div>
      );
    case 'memory-palace':
      return (
        <motion.div key="memory-palace" {...viewTransition} className="flex-1">
          <MemoryPalaceBuilder />
        </motion.div>
      );
    case 'analytics':
      return (
        <motion.div key="analytics" {...viewTransition} className="flex-1">
          <AdvancedAnalyticsLearningDashboard userId={user?.id} />
        </motion.div>
      );
    default:
      return (
        <motion.div key="default" {...viewTransition} className="flex-1 flex flex-col min-h-0">
          <ChatInterface />
        </motion.div>
      );
  }
}

// ===============================
// 🎨 UTILITY FUNCTIONS
// ===============================

function getViewTitle(viewId) {
  const titles = {
    'elaborative-questions': 'Elaborative Questions',
    'transfer-learning': 'Transfer Learning',
    'analytics': 'Learning Analytics',
    'achievements': 'Your Achievements',
    'settings': 'Settings & Preferences'
  };
  return titles[viewId] || 'Feature';
}

function getViewDescription(viewId) {
  const descriptions = {
    'elaborative-questions': 'Advanced questioning techniques to deepen your understanding and critical thinking skills.',
    'transfer-learning': 'Apply knowledge across different domains and contexts for enhanced learning outcomes.',
    'analytics': 'Comprehensive insights into your learning patterns, progress, and optimization recommendations.',
    'achievements': 'Track your learning milestones, badges, and celebrate your educational journey.',
    'settings': 'Customize your learning experience with personalized preferences and configurations.'
  };
  return descriptions[viewId] || 'This feature is under development and will be available soon.';
}

// ===============================
// 🎨 ROOT APP COMPONENT
// ===============================

function App() {
  return (
    <PremiumErrorBoundary>
      <ThemeProvider>
        <AppProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/*" element={<AppContent />} />
            </Routes>
          </BrowserRouter>
        </AppProvider>
      </ThemeProvider>
    </PremiumErrorBoundary>
  );
}

export default App;