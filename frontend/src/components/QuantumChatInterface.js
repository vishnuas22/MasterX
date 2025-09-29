import React, { useState, useRef, useEffect } from 'react';
import { useQuantumIntelligence } from '../providers/QuantumIntelligenceProvider';

export const QuantumChatInterface = () => {
  const { sendQuantumMessage, isLoading, emotionalState, adaptiveMetrics } = useQuantumIntelligence();
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'system',
      content: '🧠 Welcome to MasterX V6.0! I\'m your AI learning companion powered by quantum intelligence. How can I help you learn today?',
      timestamp: Date.now()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedTaskType, setSelectedTaskType] = useState('GENERAL');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const taskTypes = [
    { value: 'GENERAL', label: '🎯 General Learning', description: 'General questions and learning' },
    { value: 'BEGINNER_CONCEPTS', label: '🌱 Beginner Concepts', description: 'Simple explanations for beginners' },
    { value: 'ADVANCED_CONCEPTS', label: '🚀 Advanced Concepts', description: 'Complex and advanced topics' },
    { value: 'EMOTIONAL_SUPPORT', label: '💭 Emotional Support', description: 'Need encouragement or help with frustration' },
    { value: 'COMPLEX_EXPLANATION', label: '🧠 Complex Explanation', description: 'Detailed analysis and explanations' },
    { value: 'QUICK_RESPONSE', label: '⚡ Quick Response', description: 'Fast and concise answers' },
    { value: 'PERSONALIZED_LEARNING', label: '🎯 Personalized Learning', description: 'Tailored to your learning style' },
    { value: 'QUANTUM_LEARNING', label: '🌌 Quantum Learning', description: 'Advanced quantum-enhanced learning' }
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: Date.now(),
      taskType: selectedTaskType
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');

    try {
      const result = await sendQuantumMessage(inputMessage.trim(), 'user_001', selectedTaskType);
      
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: result.response.content,
        timestamp: Date.now(),
        provider: result.response.provider,
        model: result.response.model,
        confidence: result.response.confidence,
        empathy_score: result.response.empathy_score,
        task_completion_score: result.response.task_completion_score,
        performance: result.performance,
        quantum_metrics: result.quantum_metrics,
        emotion_detected: result.analytics?.authentic_emotion_result?.primary_emotion,
        adaptations: result.analytics?.adaptations || []
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `❌ Sorry, I encountered an error: ${error.message}. Please try again.`,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      'joy': 'text-yellow-400',
      'confused': 'text-orange-400',
      'frustrated': 'text-red-400',
      'curious': 'text-blue-400',
      'confident': 'text-green-400',
      'neutral': 'text-gray-400'
    };
    return colors[emotion] || 'text-gray-400';
  };

  return (
    <div className="quantum-chat-interface">
      <div className="chat-header">
        <div className="header-info">
          <h2 className="text-2xl font-bold text-quantum-primary mb-2">
            🚀 Quantum Chat Interface
          </h2>
          <div className="status-indicators">
            <div className="status-item">
              <span className="label">Emotion:</span>
              <span className={`value ${getEmotionColor(emotionalState.primary_emotion)}`}>
                {emotionalState.primary_emotion.toUpperCase()}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Learning Readiness:</span>
              <span className="value text-quantum-accent">
                {emotionalState.learning_readiness?.replace('_', ' ').toUpperCase()}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Quantum Coherence:</span>
              <span className="value text-quantum-success">
                {Math.round(adaptiveMetrics.quantum_coherence * 100)}%
              </span>
            </div>
          </div>
        </div>
        
        <div className="task-type-selector">
          <label htmlFor="taskType" className="block text-sm font-medium mb-2">
            Learning Mode:
          </label>
          <select
            id="taskType"
            value={selectedTaskType}
            onChange={(e) => setSelectedTaskType(e.target.value)}
            className="task-select"
            data-testid="task-type-selector"
          >
            {taskTypes.map(type => (
              <option key={type.value} value={type.value} title={type.description}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="messages-container" data-testid="messages-container">
        {messages.map(message => (
          <div key={message.id} className={`message message-${message.type}`}>
            <div className="message-header">
              <span className="message-sender">
                {message.type === 'user' ? '👤 You' : 
                 message.type === 'ai' ? `🧠 MasterX ${message.provider ? `(${message.provider})` : ''}` :
                 message.type === 'system' ? '🎯 System' : '❌ Error'}
              </span>
              <span className="message-time">{formatTime(message.timestamp)}</span>
              {message.taskType && (
                <span className="task-badge">
                  {taskTypes.find(t => t.value === message.taskType)?.label}
                </span>
              )}
            </div>
            
            <div className="message-content">
              {message.content}
            </div>

            {message.type === 'ai' && message.provider && (
              <div className="message-metadata">
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span>Confidence:</span>
                    <span>{Math.round(message.confidence * 100)}%</span>
                  </div>
                  <div className="metadata-item">
                    <span>Empathy:</span>
                    <span>{Math.round(message.empathy_score * 100)}%</span>
                  </div>
                  <div className="metadata-item">
                    <span>Task Completion:</span>
                    <span>{Math.round(message.task_completion_score * 100)}%</span>
                  </div>
                  <div className="metadata-item">
                    <span>Response Time:</span>
                    <span>{Math.round(message.performance?.total_processing_time_ms)}ms</span>
                  </div>
                </div>
                
                {message.emotion_detected && (
                  <div className="emotion-detected">
                    <span>🎭 Emotion Detected: </span>
                    <span className={getEmotionColor(message.emotion_detected)}>
                      {message.emotion_detected.toUpperCase()}
                    </span>
                  </div>
                )}

                {message.adaptations && message.adaptations.length > 0 && (
                  <div className="adaptations">
                    <span>🎯 Adaptations Applied:</span>
                    <ul>
                      {message.adaptations.map((adaptation, index) => (
                        <li key={index}>
                          {adaptation.strategy.replace('_', ' ')} 
                          (Confidence: {Math.round(adaptation.confidence * 100)}%)
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="message message-ai loading">
            <div className="message-header">
              <span className="message-sender">🧠 MasterX</span>
              <span className="message-time">Processing...</span>
            </div>
            <div className="message-content">
              <div className="thinking-animation">
                <div className="thinking-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span>Quantum Intelligence processing your request...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="message-input-form">
        <div className="input-container">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me anything about learning, quantum mechanics, programming, or just chat..."
            className="message-input"
            disabled={isLoading}
            data-testid="message-input"
          />
          <button
            type="submit"
            disabled={isLoading || !inputMessage.trim()}
            className="send-button"
            data-testid="send-button"
          >
            {isLoading ? '⏳' : '🚀'}
          </button>
        </div>
        <div className="input-hint">
          Press Enter to send • Current mode: {taskTypes.find(t => t.value === selectedTaskType)?.label}
        </div>
      </form>

      <style jsx>{`
        .quantum-chat-interface {
          display: flex;
          flex-direction: column;
          height: calc(100vh - 200px);
          background: rgba(13, 17, 23, 0.6);
          border-radius: var(--quantum-border-radius);
          border: 1px solid var(--quantum-border);
          overflow: hidden;
        }

        .chat-header {
          background: rgba(33, 38, 45, 0.8);
          padding: 1.5rem;
          border-bottom: 1px solid var(--quantum-border);
        }

        .header-info h2 {
          margin-bottom: 1rem;
        }

        .status-indicators {
          display: flex;
          gap: 2rem;
          margin-bottom: 1rem;
        }

        .status-item {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .status-item .label {
          font-size: 0.8rem;
          color: var(--quantum-accent);
          font-weight: 500;
        }

        .status-item .value {
          font-weight: 600;
          font-size: 0.9rem;
        }

        .task-type-selector {
          max-width: 300px;
        }

        .task-select {
          width: 100%;
          padding: 0.75rem;
          background: rgba(13, 17, 23, 0.8);
          border: 1px solid var(--quantum-border);
          border-radius: var(--quantum-border-radius);
          color: var(--quantum-light);
          font-size: 0.9rem;
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 1rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .message {
          max-width: 85%;
          background: rgba(33, 38, 45, 0.6);
          border-radius: var(--quantum-border-radius);
          padding: 1rem;
          border: 1px solid var(--quantum-border);
        }

        .message-user {
          align-self: flex-end;
          background: rgba(0, 245, 255, 0.1);
          border-color: var(--quantum-primary);
        }

        .message-ai {
          align-self: flex-start;
          background: rgba(199, 125, 255, 0.1);
          border-color: var(--quantum-accent);
        }

        .message-system {
          align-self: center;
          background: rgba(0, 255, 136, 0.1);
          border-color: var(--quantum-success);
          text-align: center;
        }

        .message-error {
          align-self: flex-start;
          background: rgba(255, 0, 110, 0.1);
          border-color: var(--quantum-danger);
        }

        .message-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
          font-size: 0.8rem;
        }

        .message-sender {
          font-weight: 600;
          color: var(--quantum-primary);
        }

        .message-time {
          color: var(--quantum-accent);
        }

        .task-badge {
          background: var(--quantum-accent);
          color: var(--quantum-dark);
          padding: 0.2rem 0.5rem;
          border-radius: 10px;
          font-size: 0.7rem;
        }

        .message-content {
          line-height: 1.6;
          white-space: pre-wrap;
        }

        .message-metadata {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid var(--quantum-border);
          font-size: 0.8rem;
        }

        .metadata-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 0.5rem;
          margin-bottom: 0.5rem;
        }

        .metadata-item {
          display: flex;
          justify-content: space-between;
        }

        .emotion-detected,
        .adaptations {
          margin-top: 0.5rem;
          color: var(--quantum-accent);
        }

        .adaptations ul {
          margin: 0.25rem 0 0 1rem;
          list-style-type: disc;
        }

        .thinking-animation {
          display: flex;
          align-items: center;
          gap: 1rem;
          color: var(--quantum-accent);
        }

        .thinking-dots {
          display: flex;
          gap: 0.25rem;
        }

        .thinking-dots span {
          width: 6px;
          height: 6px;
          background: var(--quantum-primary);
          border-radius: 50%;
          animation: bounce 1.4s infinite ease-in-out both;
        }

        .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
          } 40% {
            transform: scale(1);
          }
        }

        .message-input-form {
          background: rgba(33, 38, 45, 0.8);
          padding: 1.5rem;
          border-top: 1px solid var(--quantum-border);
        }

        .input-container {
          display: flex;
          gap: 1rem;
          margin-bottom: 0.5rem;
        }

        .message-input {
          flex: 1;
          padding: 1rem;
          background: rgba(13, 17, 23, 0.8);
          border: 1px solid var(--quantum-border);
          border-radius: var(--quantum-border-radius);
          color: var(--quantum-light);
          font-size: 1rem;
          transition: border-color 0.3s ease;
        }

        .message-input:focus {
          outline: none;
          border-color: var(--quantum-primary);
          box-shadow: 0 0 0 2px rgba(0, 245, 255, 0.1);
        }

        .send-button {
          padding: 1rem 1.5rem;
          background: var(--quantum-gradient-primary);
          border: none;
          border-radius: var(--quantum-border-radius);
          color: var(--quantum-dark);
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          font-size: 1.2rem;
        }

        .send-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: var(--quantum-shadow);
        }

        .send-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .input-hint {
          font-size: 0.8rem;
          color: var(--quantum-accent);
          text-align: center;
        }

        @media (max-width: 768px) {
          .status-indicators {
            flex-direction: column;
            gap: 1rem;
          }

          .message {
            max-width: 95%;
          }

          .metadata-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};