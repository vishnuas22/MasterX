import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Play, CheckCircle, ArrowRight, Lightbulb, Target, BarChart3 } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { LoadingSpinner } from './LoadingSpinner';
import { api } from '../services/api';
import { useApp } from '../context/AppContext';

const MetacognitiveTraining = () => {
  const { state } = useApp();
  const [currentSession, setCurrentSession] = useState(null);
  const [sessionStep, setSessionStep] = useState(0);
  const [userResponse, setUserResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState('self_questioning');
  const [topic, setTopic] = useState('');
  const [level, setLevel] = useState('intermediate');

  const strategies = [
    {
      id: 'self_questioning',
      name: 'Self-Questioning',
      description: 'Learn to ask yourself the right questions while studying',
      icon: Brain,
      color: 'from-purple-500 to-blue-600'
    },
    {
      id: 'goal_setting',
      name: 'Goal Setting',
      description: 'Set and achieve specific, measurable learning objectives',
      icon: Target,
      color: 'from-green-500 to-teal-600'
    },
    {
      id: 'progress_monitoring',
      name: 'Progress Monitoring',
      description: 'Track and evaluate your learning effectiveness',
      icon: BarChart3,
      color: 'from-orange-500 to-red-600'
    },
    {
      id: 'reflection',
      name: 'Reflection',
      description: 'Develop deeper insights through structured reflection',
      icon: Lightbulb,
      color: 'from-cyan-500 to-blue-600'
    }
  ];

  const startSession = async () => {
    if (!topic.trim()) {
      alert('Please enter a topic to study');
      return;
    }

    setLoading(true);
    try {
      const sessionData = {
        strategy: selectedStrategy,
        topic: topic,
        level: level
      };

      const response = await api.startMetacognitiveSession(state.user.id, sessionData);
      setCurrentSession(response);
      setSessionStep(1);
      setSessionHistory([]);
    } catch (error) {
      console.error('Error starting session:', error);
      alert('Failed to start metacognitive session: ' + (error.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const submitResponse = async () => {
    if (!userResponse.trim()) return;

    setLoading(true);
    try {
      const responseData = {
        user_response: userResponse
      };

      const response = await api.respondToMetacognitiveSession(currentSession.session_id, responseData);

      // Add to history
      setSessionHistory(prev => [...prev, {
        type: 'user',
        content: userResponse,
        timestamp: new Date()
      }, {
        type: 'ai',
        content: response.feedback || response.next_prompt || 'Session completed!',
        analysis: response.analysis,
        timestamp: new Date()
      }]);

      setUserResponse('');
      setSessionStep(prev => prev + 1);
    } catch (error) {
      console.error('Error submitting response:', error);
      alert('Failed to submit response: ' + (error.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const resetSession = () => {
    setCurrentSession(null);
    setSessionStep(0);
    setSessionHistory([]);
    setUserResponse('');
  };

  if (!currentSession) {
    return (
      <div className="p-6">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-blue-600 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Metacognitive Training</h1>
              <p className="text-gray-400">Develop advanced self-awareness in your learning process</p>
            </div>
          </div>
        </motion.div>

        {/* Strategy Selection */}
        <GlassCard className="p-6 mb-6">
          <h3 className="text-xl font-bold text-white mb-4">Choose Your Learning Strategy</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {strategies.map((strategy) => (
              <motion.div
                key={strategy.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedStrategy(strategy.id)}
                className={`p-4 rounded-lg cursor-pointer transition-all duration-300 border-2 ${
                  selectedStrategy === strategy.id 
                    ? 'border-white/30 bg-white/10' 
                    : 'border-transparent bg-white/5 hover:bg-white/10'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${strategy.color} flex items-center justify-center`}>
                    <strategy.icon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-white">{strategy.name}</h4>
                    <p className="text-sm text-gray-400 mt-1">{strategy.description}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </GlassCard>

        {/* Session Setup */}
        <GlassCard className="p-6">
          <h3 className="text-xl font-bold text-white mb-4">Session Setup</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                What topic would you like to study?
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Machine Learning, History of Rome, Calculus..."
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-gray-600 text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Learning Level
              </label>
              <select
                value={level}
                onChange={(e) => setLevel(e.target.value)}
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-gray-600 text-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              >
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={startSession}
              disabled={loading}
              className="w-full px-6 py-3 rounded-lg bg-gradient-to-r from-purple-500 to-blue-600 text-white font-medium flex items-center justify-center space-x-2 hover:shadow-lg transition-all duration-300 disabled:opacity-50"
            >
              {loading ? <LoadingSpinner size="sm" /> : <Play className="w-5 h-5" />}
              <span>{loading ? 'Starting Session...' : 'Start Metacognitive Session'}</span>
            </motion.button>
          </div>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="p-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-blue-600 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Metacognitive Session</h1>
              <p className="text-gray-400">{currentSession.topic} • {currentSession.strategy.replace('_', ' ')}</p>
            </div>
          </div>
          <button
            onClick={resetSession}
            className="px-4 py-2 rounded-lg border border-gray-600 text-gray-300 hover:bg-white/5 transition-colors"
          >
            New Session
          </button>
        </div>
      </motion.div>

      {/* Initial Prompt */}
      {sessionStep === 1 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <GlassCard className="p-6">
            <div className="flex items-start space-x-3 mb-4">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-blue-600 flex items-center justify-center flex-shrink-0">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <div>
                <h4 className="font-semibold text-blue-400 mb-2">AI Mentor</h4>
                <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {currentSession.initial_prompt}
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Session History */}
      <div className="space-y-4 mb-6">
        {sessionHistory.map((entry, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <GlassCard className="p-4">
              <div className="flex items-start space-x-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  entry.type === 'user' 
                    ? 'bg-gradient-to-r from-green-500 to-teal-600' 
                    : 'bg-gradient-to-r from-purple-500 to-blue-600'
                }`}>
                  {entry.type === 'user' ? (
                    <span className="text-sm font-bold text-white">U</span>
                  ) : (
                    <Brain className="w-4 h-4 text-white" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <h4 className={`font-semibold ${
                      entry.type === 'user' ? 'text-green-400' : 'text-blue-400'
                    }`}>
                      {entry.type === 'user' ? 'You' : 'AI Mentor'}
                    </h4>
                    <span className="text-xs text-gray-500">
                      {entry.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                    {entry.content}
                  </div>
                  
                  {/* Analysis Display */}
                  {entry.analysis && (
                    <div className="mt-4 p-3 rounded-lg bg-white/5 border border-gray-600">
                      <h5 className="text-sm font-semibold text-purple-400 mb-2">Learning Analysis</h5>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-400">Self-Awareness: </span>
                          <span className="text-white">{(entry.analysis.self_awareness_score * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Reflection Quality: </span>
                          <span className="text-white">{entry.analysis.reflection_quality}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      {/* Response Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <GlassCard className="p-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Your Response
              </label>
              <textarea
                value={userResponse}
                onChange={(e) => setUserResponse(e.target.value)}
                placeholder="Share your thoughts, reflections, or answers here..."
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-gray-600 text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 resize-none"
                rows={4}
              />
            </div>

            <div className="flex justify-end space-x-3">
              <button
                onClick={resetSession}
                className="px-6 py-2 rounded-lg border border-gray-600 text-gray-300 hover:bg-white/5 transition-colors"
              >
                End Session
              </button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={submitResponse}
                disabled={loading || !userResponse.trim()}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-blue-600 text-white font-medium flex items-center space-x-2 hover:shadow-lg transition-all duration-300 disabled:opacity-50"
              >
                {loading ? <LoadingSpinner size="sm" /> : <ArrowRight className="w-4 h-4" />}
                <span>{loading ? 'Processing...' : 'Submit Response'}</span>
              </motion.button>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
};

export default MetacognitiveTraining;