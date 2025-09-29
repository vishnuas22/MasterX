import React, { createContext, useContext, useState, useEffect } from 'react';

const QuantumIntelligenceContext = createContext();

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export const QuantumIntelligenceProvider = ({ children }) => {
  const [systemHealth, setSystemHealth] = useState({
    status: 'initializing',
    quantum_intelligence: { available: false },
    performance: { health_score: 0 }
  });
  const [emotionalState, setEmotionalState] = useState({
    primary_emotion: 'neutral',
    confidence: 0,
    learning_readiness: 'unknown'
  });
  const [adaptiveMetrics, setAdaptiveMetrics] = useState({
    comprehension_level: 'unknown',
    difficulty_score: 0.5,
    quantum_coherence: 0
  });
  const [aiProviders, setAiProviders] = useState({
    groq: { status: 'unknown' },
    emergent: { status: 'unknown' },
    gemini: { status: 'unknown' }
  });
  const [isLoading, setIsLoading] = useState(false);

  // Initialize system monitoring
  useEffect(() => {
    const interval = setInterval(updateSystemHealth, 10000); // Update every 10 seconds
    updateSystemHealth(); // Initial call
    return () => clearInterval(interval);
  }, []);

  const updateSystemHealth = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/health`);
      const health = await response.json();
      setSystemHealth(health);
      
      // Update AI provider status from health data
      if (health.quantum_intelligence?.available) {
        setAiProviders({
          groq: { status: 'operational' },
          emergent: { status: 'operational' },
          gemini: { status: 'operational' }
        });
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setSystemHealth(prev => ({ ...prev, status: 'error' }));
    }
  };

  const sendQuantumMessage = async (message, userId = 'user_001', taskType = 'GENERAL') => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/quantum/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          message: message,
          task_type: taskType,
          session_id: `session_${Date.now()}`,
          force_real_ai: true,
          enable_caching: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Update emotional state
      if (result.analytics?.authentic_emotion_result) {
        setEmotionalState({
          primary_emotion: result.analytics.authentic_emotion_result.primary_emotion || 'neutral',
          confidence: result.analytics.authentic_emotion_result.emotion_confidence || 0,
          learning_readiness: result.analytics.authentic_emotion_result.learning_readiness || 'unknown'
        });
      }

      // Update adaptive metrics
      if (result.analytics?.adaptation_analysis?.analytics) {
        const analytics = result.analytics.adaptation_analysis.analytics;
        setAdaptiveMetrics({
          comprehension_level: analytics.comprehension_level || 'unknown',
          difficulty_score: analytics.difficulty_score || 0.5,
          quantum_coherence: result.quantum_metrics?.quantum_coherence || 0
        });
      }

      return result;
    } catch (error) {
      console.error('Quantum message failed:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    systemHealth,
    emotionalState,
    adaptiveMetrics,
    aiProviders,
    isLoading,
    sendQuantumMessage,
    updateSystemHealth
  };

  return (
    <QuantumIntelligenceContext.Provider value={value}>
      {children}
    </QuantumIntelligenceContext.Provider>
  );
};

export const useQuantumIntelligence = () => {
  const context = useContext(QuantumIntelligenceContext);
  if (!context) {
    throw new Error('useQuantumIntelligence must be used within a QuantumIntelligenceProvider');
  }
  return context;
};

export default QuantumIntelligenceProvider;