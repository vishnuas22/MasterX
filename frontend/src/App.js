import React, { useState, useEffect, lazy, Suspense } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { AppProvider, useApp } from "./context/AppContext";
import { ThemeProvider } from "./components/AdaptiveThemeSystem";
import { UserOnboarding } from "./components/UserOnboarding";
import { Sidebar } from "./components/Sidebar";
import { ChatInterface } from "./components/ChatInterface";
import { LoadingSpinner } from "./components/LoadingSpinner";
import ConnectionStatus from "./components/ConnectionStatus";
import { api } from "./services/api";

// Lazy load heavy components for better performance
const LearningPsychologyDashboard = lazy(() => import("./components/LearningPsychologyDashboard"));
const MetacognitiveTraining = lazy(() => import("./components/MetacognitiveTraining"));
const MemoryPalaceBuilder = lazy(() => import("./components/MemoryPalaceBuilder"));
const PersonalizationDashboard = lazy(() => import("./components/PersonalizationDashboard"));

// Main App Content Component
function AppContent() {
  const { state, actions } = useApp();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('checking');

  // Check backend connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await api.healthCheck();
        setConnectionStatus('connected');
      } catch (error) {
        console.error('Backend connection failed:', error);
        setConnectionStatus('error');
      }
    };

    checkConnection();
  }, []);

  // Show onboarding if no user
  if (!state.user) {
    return <UserOnboarding />;
  }

  // Show connection error
  if (connectionStatus === 'error') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center">
        <div className="text-center text-white">
          <div className="text-red-400 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold mb-2">Connection Error</h2>
          <p className="text-gray-400 mb-4">Unable to connect to MasterX AI Mentor System</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  // Show loading while checking connection
  if (connectionStatus === 'checking') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center">
        <LoadingSpinner size="xl" message="Connecting to MasterX..." />
      </div>
    );
  }

  // Main application layout
  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl animate-pulse"></div>
      </div>

      {/* Main Layout */}
      <div className="relative z-10 flex h-full">
        <AnimatePresence>
          <motion.div
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
          >
            <Sidebar 
              isCollapsed={sidebarCollapsed} 
              onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} 
            />
          </motion.div>
        </AnimatePresence>

        <motion.main
          className="flex-1 flex flex-col overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {/* Render different components based on active view */}
          {state.activeView === 'chat' && <ChatInterface />}
          {state.activeView === 'personalization' && (
            <Suspense fallback={<LoadingSpinner />}>
              <PersonalizationDashboard />
            </Suspense>
          )}
          {state.activeView === 'learning-psychology' && (
            <Suspense fallback={<LoadingSpinner />}>
              <LearningPsychologyDashboard />
            </Suspense>
          )}
          {state.activeView === 'metacognitive-training' && (
            <Suspense fallback={<LoadingSpinner />}>
              <MetacognitiveTraining />
            </Suspense>
          )}
          {state.activeView === 'memory-palace' && (
            <Suspense fallback={<LoadingSpinner />}>
              <MemoryPalaceBuilder />
            </Suspense>
          )}
          {state.activeView === 'elaborative-questions' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-4">Elaborative Questions</h2>
                <p className="text-gray-400">Coming soon! Deep questioning skills development.</p>
              </div>
            </div>
          )}
          {state.activeView === 'transfer-learning' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-4">Transfer Learning</h2>
                <p className="text-gray-400">Coming soon! Knowledge application across domains.</p>
              </div>
            </div>
          )}
          {state.activeView === 'analytics' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-4">Analytics Dashboard</h2>
                <p className="text-gray-400">Coming soon! Comprehensive learning analytics.</p>
              </div>
            </div>
          )}
          {state.activeView === 'achievements' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-4">Achievements</h2>
                <p className="text-gray-400">Coming soon! Your learning milestones and badges.</p>
              </div>
            </div>
          )}
          {state.activeView === 'settings' && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-4">Settings</h2>
                <p className="text-gray-400">Coming soon! Customize your learning experience.</p>
              </div>
            </div>
          )}
        </motion.main>
      </div>

      {/* Error Display */}
      <AnimatePresence>
        {state.error && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-4 right-4 z-50"
          >
            <div className="bg-red-500/20 border border-red-400/50 backdrop-blur-xl rounded-xl p-4 max-w-sm">
              <div className="flex items-start space-x-3">
                <div className="text-red-400 text-xl">⚠️</div>
                <div className="flex-1">
                  <h4 className="text-red-300 font-medium">Error</h4>
                  <p className="text-red-200 text-sm mt-1">{state.error}</p>
                </div>
                <button
                  onClick={() => actions.setError(null)}
                  className="text-red-300 hover:text-red-100 transition-colors"
                >
                  ✕
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Connection Status Debug Component */}
      <ConnectionStatus />
    </div>
  );
}

// Root App Component
function App() {
  return (
    <ThemeProvider>
      <AppProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/*" element={<AppContent />} />
          </Routes>
        </BrowserRouter>
      </AppProvider>
    </ThemeProvider>
  );
}

export default App;
