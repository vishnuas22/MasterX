import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, BookOpen, Target, Shuffle, Trophy, Play, Settings, HelpCircle } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { LoadingSpinner } from './LoadingSpinner';
import { api } from '../services/api';
import { useApp } from '../context/AppContext';

const LearningPsychologyDashboard = () => {
  const { state, actions } = useApp();
  const [activeFeature, setActiveFeature] = useState(null);
  const [features, setFeatures] = useState(null);
  const [loading, setLoading] = useState(true);
  const [userProgress, setUserProgress] = useState(null);

  useEffect(() => {
    loadFeatures();
    loadUserProgress();
  }, []);

  const loadFeatures = async () => {
    try {
      const response = await api.getLearningPsychologyFeatures();
      setFeatures(response);
    } catch (error) {
      console.error('Error loading features:', error);
      // Set fallback features if API fails
      setFeatures({
        available: true,
        message: "Learning psychology features are available",
        features: [
          "Metacognitive Training",
          "Memory Palace Builder", 
          "Elaborative Interrogation",
          "Transfer Learning"
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const loadUserProgress = async () => {
    if (!state.user) return;
    
    try {
      const response = await api.getUserLearningProgress(state.user.id);
      setUserProgress(response);
    } catch (error) {
      console.error('Error loading progress:', error);
      // Set fallback progress data
      setUserProgress({
        metacognitive_sessions: 0,
        memory_palaces: 0,
        elaborative_questions_answered: 0,
        transfer_scenarios_completed: 0
      });
    }
  };

  const handleStartFeature = (featureId) => {
    switch (featureId) {
      case 'metacognitive':
        actions.setActiveView('metacognitive-training');
        break;
      case 'memory_palace':
        actions.setActiveView('memory-palace');
        break;
      case 'elaborative':
        actions.setActiveView('elaborative-questions');
        break;
      case 'transfer':
        actions.setActiveView('transfer-learning');
        break;
      default:
        console.log('Unknown feature:', featureId);
    }
  };

  const featureCards = [
    {
      id: 'metacognitive',
      title: 'Metacognitive Training',
      description: 'Master how you learn with advanced self-awareness techniques',
      icon: Brain,
      gradient: 'from-purple-500 to-blue-600',
      features: ['Self-Questioning', 'Goal Setting', 'Progress Monitoring', 'Strategy Selection']
    },
    {
      id: 'memory_palace',
      title: 'Memory Palace Builder',
      description: 'Create AI-powered spatial memory systems for enhanced retention',
      icon: BookOpen,
      gradient: 'from-green-500 to-teal-600',
      features: ['AI-Generated Palaces', 'Visual Associations', 'Practice Sessions', 'Effectiveness Tracking']
    },
    {
      id: 'elaborative',
      title: 'Elaborative Interrogation',
      description: 'Develop deep questioning skills for critical thinking',
      icon: HelpCircle,
      gradient: 'from-orange-500 to-red-600',
      features: ['Deep Questions', 'Multi-Level Difficulty', 'Subject Adaptation', 'Critical Analysis']
    },
    {
      id: 'transfer',
      title: 'Transfer Learning',
      description: 'Apply knowledge across domains with AI-guided analogies',
      icon: Shuffle,
      gradient: 'from-cyan-500 to-blue-600',
      features: ['Cross-Domain Mapping', 'Analogy Generation', 'Application Scenarios', 'Pattern Recognition']
    }
  ];

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <LoadingSpinner size="xl" message="Loading Advanced Learning Psychology..." />
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 overflow-auto">
      {/* Header */}
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
            <h1 className="text-3xl font-bold text-white">Advanced Learning Psychology</h1>
            <p className="text-gray-400">Master your learning process with cutting-edge cognitive science</p>
          </div>
        </div>

        {/* Progress Overview */}
        {userProgress && (
          <GlassCard className="p-4 mb-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">{userProgress.metacognitive_sessions || 0}</div>
                <div className="text-sm text-gray-400">Metacognitive Sessions</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">{userProgress.memory_palaces || 0}</div>
                <div className="text-sm text-gray-400">Memory Palaces</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400">{userProgress.elaborative_questions_answered || 0}</div>
                <div className="text-sm text-gray-400">Questions Answered</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-cyan-400">{userProgress.transfer_scenarios_completed || 0}</div>
                <div className="text-sm text-gray-400">Transfer Scenarios</div>
              </div>
            </div>
          </GlassCard>
        )}
      </motion.div>

      {/* Feature Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {featureCards.map((feature, index) => (
          <motion.div
            key={feature.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <GlassCard 
              className={`p-6 cursor-pointer transition-all duration-300 hover:scale-105 border-2 border-transparent hover:border-white/20 ${
                activeFeature === feature.id ? 'border-white/30 ring-2 ring-white/20' : ''
              }`}
              onClick={() => setActiveFeature(feature.id)}
            >
              <div className="flex items-start space-x-4">
                <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${feature.gradient} flex items-center justify-center flex-shrink-0`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                  <p className="text-gray-400 mb-4">{feature.description}</p>
                  <div className="space-y-1">
                    {feature.features.map((item, idx) => (
                      <div key={idx} className="flex items-center space-x-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-gray-400"></div>
                        <span className="text-sm text-gray-300">{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="mt-6 flex justify-end">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleStartFeature(feature.id)}
                  className={`px-6 py-2 rounded-lg bg-gradient-to-r ${feature.gradient} text-white font-medium flex items-center space-x-2 hover:shadow-lg transition-all duration-300`}
                >
                  <Play className="w-4 h-4" />
                  <span>Start Session</span>
                </motion.button>
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      {/* Quick Access Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <GlassCard className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white">Quick Access</h3>
            <Settings className="w-5 h-5 text-gray-400" />
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-center group">
              <Target className="w-6 h-6 text-blue-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-sm text-white font-medium">Set Learning Goals</div>
            </button>
            
            <button className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-center group">
              <BookOpen className="w-6 h-6 text-green-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-sm text-white font-medium">Browse Palaces</div>
            </button>
            
            <button className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-center group">
              <HelpCircle className="w-6 h-6 text-orange-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-sm text-white font-medium">Practice Questions</div>
            </button>
            
            <button className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-center group">
              <Trophy className="w-6 h-6 text-purple-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-sm text-white font-medium">View Progress</div>
            </button>
          </div>
        </GlassCard>
      </motion.div>

      {/* Feature Details Panel */}
      <AnimatePresence>
        {activeFeature && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-6"
          >
            <GlassCard className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">
                  {featureCards.find(f => f.id === activeFeature)?.title} - Advanced Features
                </h3>
                <button
                  onClick={() => setActiveFeature(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ✕
                </button>
              </div>
              
              <div className="text-gray-300">
                {activeFeature === 'metacognitive' && (
                  <div>
                    <p className="mb-4">Develop advanced metacognitive awareness through guided self-reflection and strategic learning approaches.</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-purple-400 mb-2">Self-Questioning Techniques</h4>
                        <p className="text-sm">Learn to ask the right questions that enhance understanding and retention.</p>
                      </div>
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-blue-400 mb-2">Strategic Planning</h4>
                        <p className="text-sm">Develop personalized learning strategies based on your cognitive strengths.</p>
                      </div>
                    </div>
                  </div>
                )}
                
                {activeFeature === 'memory_palace' && (
                  <div>
                    <p className="mb-4">Build powerful spatial memory systems using AI-generated environments tailored to your learning content.</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-green-400 mb-2">AI Palace Generation</h4>
                        <p className="text-sm">Automatically create memorable spaces optimized for your specific information.</p>
                      </div>
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-teal-400 mb-2">Practice & Refinement</h4>
                        <p className="text-sm">Regular practice sessions to strengthen memory associations and recall speed.</p>
                      </div>
                    </div>
                  </div>
                )}
                
                {activeFeature === 'elaborative' && (
                  <div>
                    <p className="mb-4">Master the art of deep questioning to enhance critical thinking and comprehensive understanding.</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-orange-400 mb-2">Progressive Difficulty</h4>
                        <p className="text-sm">Questions that adapt to your level and gradually increase in complexity.</p>
                      </div>
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-red-400 mb-2">Critical Analysis</h4>
                        <p className="text-sm">Develop skills to analyze, synthesize, and evaluate information effectively.</p>
                      </div>
                    </div>
                  </div>
                )}
                
                {activeFeature === 'transfer' && (
                  <div>
                    <p className="mb-4">Learn to apply knowledge across different domains using AI-powered analogy generation and pattern recognition.</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-cyan-400 mb-2">Cross-Domain Mapping</h4>
                        <p className="text-sm">Identify connections between seemingly unrelated fields of knowledge.</p>
                      </div>
                      <div className="p-4 rounded-lg bg-white/5">
                        <h4 className="font-semibold text-blue-400 mb-2">Practical Applications</h4>
                        <p className="text-sm">Real-world scenarios that demonstrate knowledge transfer principles.</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="mt-6 flex justify-end space-x-4">
                <button className="px-6 py-2 rounded-lg border border-gray-600 text-gray-300 hover:bg-white/5 transition-colors">
                  Learn More
                </button>
                <button className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-blue-600 text-white font-medium hover:shadow-lg transition-all duration-300">
                  Start Now
                </button>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LearningPsychologyDashboard;