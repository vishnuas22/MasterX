import React, { useState, useEffect } from 'react';
import './App.css';
import { QuantumIntelligenceDashboard } from './components/QuantumIntelligenceDashboard';
import { EmotionDetectionInterface } from './components/EmotionDetectionInterface';
import { AdaptiveLearningInterface } from './components/AdaptiveLearningInterface';
import { QuantumChatInterface } from './components/QuantumChatInterface';
import { PerformanceMonitoringDashboard } from './components/PerformanceMonitoringDashboard';
import { QuantumIntelligenceProvider } from './providers/QuantumIntelligenceProvider';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
console.log('🚀 MasterX Frontend V6.0 - Backend URL:', BACKEND_URL);

function App() {
  const [activeView, setActiveView] = useState('quantum-chat');
  const [systemStatus, setSystemStatus] = useState('initializing');

  useEffect(() => {
    // Initialize system health check
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/health`);
      const health = await response.json();
      
      if (health.status === 'healthy') {
        setSystemStatus('operational');
        console.log('✅ MasterX Quantum Intelligence System: OPERATIONAL');
      } else {
        setSystemStatus('degraded');
        console.warn('⚠️ MasterX System Health Issues Detected');
      }
    } catch (error) {
      console.error('❌ System Health Check Failed:', error);
      setSystemStatus('error');
    }
  };

  const renderActiveView = () => {
    switch (activeView) {
      case 'quantum-dashboard':
        return <QuantumIntelligenceDashboard />;
      case 'emotion-detection':
        return <EmotionDetectionInterface />;
      case 'adaptive-learning':
        return <AdaptiveLearningInterface />;
      case 'quantum-chat':
        return <QuantumChatInterface />;
      case 'performance-monitoring':
        return <PerformanceMonitoringDashboard />;
      default:
        return <QuantumChatInterface />;
    }
  };

  return (
    <QuantumIntelligenceProvider>
      <div className="masterx-app">
        {/* Header */}
        <header className="masterx-header">
          <div className="header-content">
            <div className="logo-section">
              <h1 className="masterx-logo">
                🧠 MasterX
                <span className="version-badge">V6.0</span>
              </h1>
              <p className="tagline">Quantum Intelligence Learning Platform</p>
            </div>
            
            <div className="system-status">
              <div className={`status-indicator status-${systemStatus}`}>
                <div className="status-dot"></div>
                <span>{systemStatus.toUpperCase()}</span>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="main-navigation">
            <button 
              className={`nav-btn ${activeView === 'quantum-chat' ? 'active' : ''}`}
              onClick={() => setActiveView('quantum-chat')}
              data-testid="nav-quantum-chat"
            >
              🚀 Quantum Chat
            </button>
            <button 
              className={`nav-btn ${activeView === 'quantum-dashboard' ? 'active' : ''}`}
              onClick={() => setActiveView('quantum-dashboard')}
              data-testid="nav-quantum-dashboard"
            >
              🧠 Quantum Dashboard
            </button>
            <button 
              className={`nav-btn ${activeView === 'emotion-detection' ? 'active' : ''}`}
              onClick={() => setActiveView('emotion-detection')}
              data-testid="nav-emotion-detection"
            >
              💭 Emotion Detection
            </button>
            <button 
              className={`nav-btn ${activeView === 'adaptive-learning' ? 'active' : ''}`}
              onClick={() => setActiveView('adaptive-learning')}
              data-testid="nav-adaptive-learning"
            >
              🎯 Adaptive Learning
            </button>
            <button 
              className={`nav-btn ${activeView === 'performance-monitoring' ? 'active' : ''}`}
              onClick={() => setActiveView('performance-monitoring')}
              data-testid="nav-performance-monitoring"
            >
              📊 Performance
            </button>
          </nav>
        </header>

        {/* Main Content */}
        <main className="main-content">
          <div className="view-container">
            {renderActiveView()}
          </div>
        </main>

        {/* Footer */}
        <footer className="masterx-footer">
          <p>
            🚀 MasterX V6.0 - Revolutionary AI Learning Platform | 
            Quantum Intelligence Engine: {systemStatus} | 
            Backend: {BACKEND_URL ? 'Connected' : 'Disconnected'}
          </p>
        </footer>
      </div>
    </QuantumIntelligenceProvider>
  );
}

export default App;