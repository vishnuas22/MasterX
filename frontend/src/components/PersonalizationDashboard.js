import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, Brain, Target, TrendingUp, Calendar, 
  BookOpen, Heart, Zap, Star, Settings,
  BarChart3, PieChart, Activity, Clock,
  Award, Lightbulb, ArrowRight, Plus
} from 'lucide-react';
import { api } from '../services/api';
import { useApp } from '../context/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

const PersonalizationDashboard = () => {
  const { state } = useApp();
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [learningDNA, setLearningDNA] = useState(null);
  const [goals, setGoals] = useState([]);
  const [recommendations, setRecommendations] = useState({});
  const [insights, setInsights] = useState([]);
  const [memories, setMemories] = useState([]);
  const [showCreateGoal, setShowCreateGoal] = useState(false);

  useEffect(() => {
    loadPersonalizationData();
  }, [state.user]);

  const loadPersonalizationData = async () => {
    if (!state.user) return;
    
    setLoading(true);
    try {
      // Load all personalization data
      const [dnaRes, goalsRes, recommendationsRes, insightsRes, memoriesRes] = await Promise.all([
        api.get(`/users/${state.user.id}/learning-dna`),
        api.get(`/users/${state.user.id}/goals`),
        api.get(`/users/${state.user.id}/recommendations`),
        api.get(`/users/${state.user.id}/insights`),
        api.get(`/users/${state.user.id}/memories?limit=20`)
      ]);

      setLearningDNA(dnaRes.data.learning_dna);
      setGoals(goalsRes.data.goals);
      setRecommendations(recommendationsRes.data.recommendations);
      setInsights(insightsRes.data.insights);
      setMemories(memoriesRes.data.memories);
    } catch (error) {
      console.error('Error loading personalization data:', error);
    } finally {
      setLoading(false);
    }
  };

  const createGoal = async (goalData) => {
    try {
      await api.post(`/users/${state.user.id}/goals`, goalData);
      await loadPersonalizationData(); // Refresh data
      setShowCreateGoal(false);
    } catch (error) {
      console.error('Error creating goal:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <LoadingSpinner size="xl" message="Loading your personalization profile..." />
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: User },
    { id: 'learning-dna', label: 'Learning DNA', icon: Brain },
    { id: 'goals', label: 'Goals & Progress', icon: Target },
    { id: 'insights', label: 'Insights', icon: Lightbulb },
    { id: 'memories', label: 'Learning Memories', icon: BookOpen }
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              Personalization Hub
            </h1>
            <p className="text-gray-400">
              Your AI-powered learning profile and adaptive insights
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-blue-500/20 border border-blue-400/30 rounded-lg px-4 py-2">
              <Star className="w-5 h-5 text-blue-400" />
              <span className="text-blue-300 font-medium">
                Confidence: {learningDNA ? (learningDNA.confidence_score * 100).toFixed(0) : 0}%
              </span>
            </div>
            <button className="p-2 bg-gray-800/50 border border-gray-700/50 rounded-lg hover:bg-gray-700/50 transition-colors">
              <Settings className="w-5 h-5 text-gray-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center space-x-1 p-6 border-b border-gray-800/30">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-blue-500/20 border border-blue-400/30 text-blue-300'
                  : 'hover:bg-gray-800/30 text-gray-400 hover:text-gray-300'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="font-medium">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <OverviewTab 
              key="overview"
              learningDNA={learningDNA}
              goals={goals}
              recommendations={recommendations}
              insights={insights}
            />
          )}
          {activeTab === 'learning-dna' && (
            <LearningDNATab key="learning-dna" learningDNA={learningDNA} />
          )}
          {activeTab === 'goals' && (
            <GoalsTab 
              key="goals" 
              goals={goals} 
              onCreateGoal={() => setShowCreateGoal(true)}
              onRefresh={loadPersonalizationData}
            />
          )}
          {activeTab === 'insights' && (
            <InsightsTab key="insights" insights={insights} />
          )}
          {activeTab === 'memories' && (
            <MemoriesTab key="memories" memories={memories} />
          )}
        </AnimatePresence>
      </div>

      {/* Create Goal Modal */}
      <AnimatePresence>
        {showCreateGoal && (
          <CreateGoalModal
            onClose={() => setShowCreateGoal(false)}
            onCreate={createGoal}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// Overview Tab Component
const OverviewTab = ({ learningDNA, goals, recommendations, insights }) => {
  const activeGoals = goals.filter(g => g.status === 'active');
  const recentInsights = insights.slice(0, 3);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard
          icon={Brain}
          title="Learning Style"
          value={learningDNA?.learning_style || 'Analyzing...'}
          color="blue"
        />
        <StatsCard
          icon={Target}
          title="Active Goals"
          value={activeGoals.length}
          color="green"
        />
        <StatsCard
          icon={TrendingUp}
          title="Learning Velocity"
          value={learningDNA ? `${(learningDNA.learning_velocity * 100).toFixed(0)}%` : 'Calculating...'}
          color="purple"
        />
        <StatsCard
          icon={Heart}
          title="Energy Level"
          value={learningDNA ? `${(learningDNA.current_energy_level * 100).toFixed(0)}%` : 'Monitoring...'}
          color="red"
        />
      </div>

      {/* Current Learning Profile */}
      {learningDNA && (
        <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Brain className="w-6 h-6 text-blue-400 mr-2" />
            Current Learning Profile
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <ProfileMetric
              label="Preferred Pace"
              value={learningDNA.preferred_pace}
              icon={Clock}
            />
            <ProfileMetric
              label="Motivation Style"
              value={learningDNA.motivation_style}
              icon={Zap}
            />
            <ProfileMetric
              label="Attention Span"
              value={`${learningDNA.attention_span_minutes} min`}
              icon={Activity}
            />
          </div>
        </div>
      )}

      {/* Recent Insights */}
      {recentInsights.length > 0 && (
        <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Lightbulb className="w-6 h-6 text-yellow-400 mr-2" />
            Recent Insights
          </h3>
          <div className="space-y-3">
            {recentInsights.map((insight, index) => (
              <div key={index} className="p-3 bg-gray-700/30 rounded-lg">
                <h4 className="font-medium text-white">{insight.title}</h4>
                <p className="text-gray-400 text-sm mt-1">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next Actions */}
      {recommendations.next_actions && (
        <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <ArrowRight className="w-6 h-6 text-green-400 mr-2" />
            Recommended Next Actions
          </h3>
          <div className="space-y-2">
            {recommendations.next_actions.map((action, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                <span className="text-gray-300">{action}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
};

// Learning DNA Tab Component
const LearningDNATab = ({ learningDNA }) => {
  if (!learningDNA) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center justify-center h-64"
      >
        <LoadingSpinner message="Analyzing your learning DNA..." />
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      {/* DNA Overview */}
      <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-400/30 rounded-xl p-6">
        <h3 className="text-2xl font-bold text-white mb-4">Your Learning DNA</h3>
        <p className="text-blue-100 mb-4">
          Based on {learningDNA.confidence_score > 0.7 ? 'extensive' : 'initial'} analysis of your learning patterns
        </p>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Star className="w-5 h-5 text-yellow-400" />
            <span className="text-white">Confidence: {(learningDNA.confidence_score * 100).toFixed(0)}%</span>
          </div>
          <div className="flex items-center space-x-2">
            <Calendar className="w-5 h-5 text-blue-400" />
            <span className="text-white">Last Updated: {new Date(learningDNA.last_updated).toLocaleDateString()}</span>
          </div>
        </div>
      </div>

      {/* Core Learning Profile */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DNAMetricCard
          title="Learning Style"
          value={learningDNA.learning_style}
          description="Your preferred way of processing information"
          color="blue"
          icon={Brain}
        />
        <DNAMetricCard
          title="Cognitive Patterns"
          value={learningDNA.cognitive_patterns.join(', ')}
          description="How you organize and process thoughts"
          color="purple"
          icon={PieChart}
        />
        <DNAMetricCard
          title="Preferred Pace"
          value={learningDNA.preferred_pace}
          description="Your optimal learning speed"
          color="green"
          icon={Clock}
        />
        <DNAMetricCard
          title="Motivation Style"
          value={learningDNA.motivation_style}
          description="What drives your learning"
          color="orange"
          icon={Zap}
        />
      </div>

      {/* Advanced Metrics */}
      <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6">
        <h4 className="text-xl font-bold text-white mb-4">Advanced Learning Metrics</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <AdvancedMetric
            label="Learning Velocity"
            value={learningDNA.learning_velocity}
            unit="concepts/hour"
            color="blue"
          />
          <AdvancedMetric
            label="Curiosity Index"
            value={learningDNA.curiosity_index}
            unit="/1.0"
            color="purple"
          />
          <AdvancedMetric
            label="Perseverance Score"
            value={learningDNA.perseverance_score}
            unit="/1.0"
            color="green"
          />
          <AdvancedMetric
            label="Retention Rate"
            value={learningDNA.concept_retention_rate}
            unit="/1.0"
            color="yellow"
          />
          <AdvancedMetric
            label="Transfer Ability"
            value={learningDNA.knowledge_transfer_ability}
            unit="/1.0"
            color="red"
          />
          <AdvancedMetric
            label="Metacognitive Awareness"
            value={learningDNA.metacognitive_awareness}
            unit="/1.0"
            color="indigo"
          />
        </div>
      </div>
    </motion.div>
  );
};

// Goals Tab Component
const GoalsTab = ({ goals, onCreateGoal, onRefresh }) => {
  const activeGoals = goals.filter(g => g.status === 'active');
  const completedGoals = goals.filter(g => g.status === 'completed');

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-white">Learning Goals</h3>
        <button
          onClick={onCreateGoal}
          className="flex items-center space-x-2 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>Create Goal</span>
        </button>
      </div>

      {/* Active Goals */}
      {activeGoals.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Active Goals</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {activeGoals.map(goal => (
              <GoalCard key={goal.goal_id} goal={goal} onRefresh={onRefresh} />
            ))}
          </div>
        </div>
      )}

      {/* Completed Goals */}
      {completedGoals.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Completed Goals</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {completedGoals.map(goal => (
              <GoalCard key={goal.goal_id} goal={goal} completed onRefresh={onRefresh} />
            ))}
          </div>
        </div>
      )}

      {/* No Goals State */}
      {goals.length === 0 && (
        <div className="text-center py-12">
          <Target className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <h4 className="text-xl font-semibold text-white mb-2">No Learning Goals Yet</h4>
          <p className="text-gray-400 mb-6">Create your first goal to start tracking your learning journey</p>
          <button
            onClick={onCreateGoal}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg transition-colors"
          >
            Create Your First Goal
          </button>
        </div>
      )}
    </motion.div>
  );
};

// Insights Tab Component
const InsightsTab = ({ insights }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    className="space-y-4"
  >
    <h3 className="text-2xl font-bold text-white mb-6">Learning Insights</h3>
    {insights.length > 0 ? (
      insights.map((insight, index) => (
        <InsightCard key={index} insight={insight} />
      ))
    ) : (
      <div className="text-center py-12">
        <Lightbulb className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <h4 className="text-xl font-semibold text-white mb-2">No Insights Yet</h4>
        <p className="text-gray-400">Continue learning to generate personalized insights</p>
      </div>
    )}
  </motion.div>
);

// Memories Tab Component
const MemoriesTab = ({ memories }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    className="space-y-4"
  >
    <h3 className="text-2xl font-bold text-white mb-6">Learning Memories</h3>
    {memories.length > 0 ? (
      memories.map((memory, index) => (
        <MemoryCard key={index} memory={memory} />
      ))
    ) : (
      <div className="text-center py-12">
        <BookOpen className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <h4 className="text-xl font-semibold text-white mb-2">No Memories Yet</h4>
        <p className="text-gray-400">Your learning memories will appear here as you progress</p>
      </div>
    )}
  </motion.div>
);

// Helper Components
const StatsCard = ({ icon: Icon, title, value, color }) => {
  const colorClasses = {
    blue: 'border-blue-400/30 bg-blue-500/20 text-blue-300',
    green: 'border-green-400/30 bg-green-500/20 text-green-300',
    purple: 'border-purple-400/30 bg-purple-500/20 text-purple-300',
    red: 'border-red-400/30 bg-red-500/20 text-red-300'
  };

  return (
    <div className={`border rounded-xl p-4 ${colorClasses[color]}`}>
      <div className="flex items-center space-x-3">
        <Icon className="w-8 h-8" />
        <div>
          <div className="text-2xl font-bold">{value}</div>
          <div className="text-sm opacity-80">{title}</div>
        </div>
      </div>
    </div>
  );
};

const ProfileMetric = ({ label, value, icon: Icon }) => (
  <div className="flex items-center space-x-3">
    <Icon className="w-5 h-5 text-gray-400" />
    <div>
      <div className="text-white font-medium">{value}</div>
      <div className="text-gray-400 text-sm">{label}</div>
    </div>
  </div>
);

const DNAMetricCard = ({ title, value, description, color, icon: Icon }) => {
  const colorClasses = {
    blue: 'border-blue-400/30 bg-blue-500/10',
    purple: 'border-purple-400/30 bg-purple-500/10',
    green: 'border-green-400/30 bg-green-500/10',
    orange: 'border-orange-400/30 bg-orange-500/10'
  };

  return (
    <div className={`border rounded-xl p-4 ${colorClasses[color]}`}>
      <div className="flex items-start space-x-3">
        <Icon className="w-6 h-6 text-white mt-1" />
        <div className="flex-1">
          <h4 className="font-semibold text-white">{title}</h4>
          <div className="text-lg font-bold text-white mt-1">{value}</div>
          <p className="text-gray-400 text-sm mt-2">{description}</p>
        </div>
      </div>
    </div>
  );
};

const AdvancedMetric = ({ label, value, unit, color }) => {
  const percentage = Math.round(value * 100);
  const colorClasses = {
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    indigo: 'bg-indigo-500'
  };

  return (
    <div className="bg-gray-700/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-300 text-sm">{label}</span>
        <span className="text-white font-medium">{value.toFixed(2)}{unit}</span>
      </div>
      <div className="w-full bg-gray-600 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${colorClasses[color]}`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
};

const GoalCard = ({ goal, completed = false, onRefresh }) => {
  const progressColor = completed ? 'bg-green-500' : 'bg-blue-500';
  
  return (
    <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-4">
      <div className="flex items-start justify-between mb-3">
        <h4 className="font-semibold text-white">{goal.title}</h4>
        {completed && <Award className="w-5 h-5 text-yellow-400" />}
      </div>
      <p className="text-gray-400 text-sm mb-4">{goal.description}</p>
      
      <div className="mb-3">
        <div className="flex items-center justify-between text-sm mb-1">
          <span className="text-gray-400">Progress</span>
          <span className="text-white">{goal.progress_percentage.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-600 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${progressColor}`}
            style={{ width: `${goal.progress_percentage}%` }}
          ></div>
        </div>
      </div>
      
      <div className="flex items-center space-x-4 text-sm text-gray-400">
        <span>Type: {goal.goal_type}</span>
        <span>•</span>
        <span>{goal.time_invested_hours.toFixed(1)}h invested</span>
      </div>
    </div>
  );
};

const InsightCard = ({ insight }) => (
  <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-4">
    <div className="flex items-start space-x-3">
      <Lightbulb className="w-6 h-6 text-yellow-400 mt-1 flex-shrink-0" />
      <div className="flex-1">
        <h4 className="font-semibold text-white mb-2">{insight.title}</h4>
        <p className="text-gray-300 mb-3">{insight.description}</p>
        <div className="flex items-center space-x-4 text-sm text-gray-400">
          <span>Confidence: {(insight.confidence * 100).toFixed(0)}%</span>
          <span>•</span>
          <span>{new Date(insight.created_at).toLocaleDateString()}</span>
        </div>
      </div>
    </div>
  </div>
);

const MemoryCard = ({ memory }) => (
  <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-4">
    <div className="flex items-start space-x-3">
      <BookOpen className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" />
      <div className="flex-1">
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-xs bg-blue-500/20 text-blue-300 px-2 py-1 rounded">
            {memory.memory_type}
          </span>
          <span className="text-xs text-gray-500">
            {new Date(memory.created_at).toLocaleDateString()}
          </span>
        </div>
        <p className="text-gray-300">{memory.content}</p>
        {memory.importance > 0.7 && (
          <div className="mt-2">
            <span className="text-xs bg-yellow-500/20 text-yellow-300 px-2 py-1 rounded">
              High Importance
            </span>
          </div>
        )}
      </div>
    </div>
  </div>
);

// Create Goal Modal Component
const CreateGoalModal = ({ onClose, onCreate }) => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    goal_type: 'skill_mastery',
    target_date: '',
    skills_required: [],
    success_criteria: []
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onCreate(formData);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-gray-800 border border-gray-700 rounded-xl p-6 w-full max-w-md"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-xl font-bold text-white mb-4">Create Learning Goal</h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Goal Title
            </label>
            <input
              type="text"
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
              placeholder="e.g., Learn React Development"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Description
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white h-20"
              placeholder="Describe what you want to achieve..."
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Goal Type
            </label>
            <select
              value={formData.goal_type}
              onChange={(e) => setFormData({ ...formData, goal_type: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
            >
              <option value="skill_mastery">Skill Mastery</option>
              <option value="knowledge_area">Knowledge Area</option>
              <option value="certification">Certification</option>
              <option value="project_completion">Project Completion</option>
              <option value="habit_formation">Habit Formation</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Target Date (Optional)
            </label>
            <input
              type="date"
              value={formData.target_date}
              onChange={(e) => setFormData({ ...formData, target_date: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
            />
          </div>

          <div className="flex space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-2 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 bg-blue-500 hover:bg-blue-600 text-white py-2 rounded-lg transition-colors"
            >
              Create Goal
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
};

export default PersonalizationDashboard;