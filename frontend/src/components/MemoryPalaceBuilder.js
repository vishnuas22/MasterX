import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BookOpen, Home, School, TreePine, Castle, Building2, Sparkles, Eye, Navigation, Target, Plus, Trash2 } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { LoadingSpinner } from './LoadingSpinner';
import { api } from '../services/api';
import { useApp } from '../context/AppContext';

const MemoryPalaceBuilder = () => {
  const { state } = useApp();
  const [currentPalace, setCurrentPalace] = useState(null);
  const [userPalaces, setUserPalaces] = useState([]);
  const [showBuilder, setShowBuilder] = useState(false);
  const [loading, setLoading] = useState(false);
  const [practiceMode, setPracticeMode] = useState(null);
  
  // Builder state
  const [palaceName, setPalaceName] = useState('');
  const [palaceType, setPalaceType] = useState('home');
  const [topic, setTopic] = useState('');
  const [informationItems, setInformationItems] = useState(['']);

  const palaceTypes = [
    { id: 'home', name: 'Home', icon: Home, description: 'Familiar home environment', color: 'from-blue-500 to-cyan-600' },
    { id: 'library', name: 'Library', icon: BookOpen, description: 'Grand library with sections', color: 'from-green-500 to-teal-600' },
    { id: 'school', name: 'School', icon: School, description: 'Educational institution layout', color: 'from-purple-500 to-blue-600' },
    { id: 'nature', name: 'Nature', icon: TreePine, description: 'Natural environment', color: 'from-emerald-500 to-green-600' },
    { id: 'castle', name: 'Castle', icon: Castle, description: 'Medieval castle structure', color: 'from-orange-500 to-red-600' },
    { id: 'laboratory', name: 'Laboratory', icon: Building2, description: 'Scientific laboratory', color: 'from-cyan-500 to-blue-600' }
  ];

  useEffect(() => {
    loadUserPalaces();
  }, []);

  const loadUserPalaces = async () => {
    if (!state.user) return;
    
    setLoading(true);
    try {
      const response = await api.getUserMemoryPalaces(state.user.id);
      setUserPalaces(response.palaces || []);
    } catch (error) {
      console.error('Error loading palaces:', error);
    } finally {
      setLoading(false);
    }
  };

  const addInformationItem = () => {
    setInformationItems([...informationItems, '']);
  };

  const updateInformationItem = (index, value) => {
    const newItems = [...informationItems];
    newItems[index] = value;
    setInformationItems(newItems);
  };

  const removeInformationItem = (index) => {
    if (informationItems.length > 1) {
      setInformationItems(informationItems.filter((_, i) => i !== index));
    }
  };

  const createPalace = async () => {
    if (!palaceName.trim() || !topic.trim()) {
      alert('Please fill in palace name and topic');
      return;
    }

    const validItems = informationItems.filter(item => item.trim());
    if (validItems.length === 0) {
      alert('Please add at least one information item');
      return;
    }

    setLoading(true);
    try {
      const palaceData = {
        name: palaceName,
        palace_type: palaceType,
        topic: topic,
        information_items: validItems
      };

      const response = await api.createMemoryPalace(state.user.id, palaceData);
      setCurrentPalace(response);
      setShowBuilder(false);
      loadUserPalaces();
      
      // Reset form
      setPalaceName('');
      setTopic('');
      setInformationItems(['']);
    } catch (error) {
      console.error('Error creating palace:', error);
      alert('Failed to create memory palace: ' + (error.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const startPractice = async (palaceId, mode) => {
    setLoading(true);
    try {
      const response = await api.post(`/learning-psychology/memory-palace/${palaceId}/practice`, null, {
        params: { practice_type: mode }
      });
      
      setPracticeMode({ mode, data: response.data });
    } catch (error) {
      console.error('Error starting practice:', error);
      alert('Failed to start practice session');
    } finally {
      setLoading(false);
    }
  };

  if (showBuilder) {
    return (
      <div className="p-6">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-green-500 to-teal-600 flex items-center justify-center">
                <BookOpen className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Memory Palace Builder</h1>
                <p className="text-gray-400">Create your AI-powered spatial memory system</p>
              </div>
            </div>
            <button
              onClick={() => setShowBuilder(false)}
              className="px-4 py-2 rounded-lg border border-gray-600 text-gray-300 hover:bg-white/5 transition-colors"
            >
              Back to Palaces
            </button>
          </div>
        </motion.div>

        {/* Palace Type Selection */}
        <GlassCard className="p-6 mb-6">
          <h3 className="text-xl font-bold text-white mb-4">Choose Palace Type</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {palaceTypes.map((type) => (
              <motion.div
                key={type.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setPalaceType(type.id)}
                className={`p-4 rounded-lg cursor-pointer transition-all duration-300 border-2 ${
                  palaceType === type.id 
                    ? 'border-white/30 bg-white/10' 
                    : 'border-transparent bg-white/5 hover:bg-white/10'
                }`}
              >
                <div className="text-center">
                  <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${type.color} flex items-center justify-center mx-auto mb-3`}>
                    <type.icon className="w-8 h-8 text-white" />
                  </div>
                  <h4 className="font-semibold text-white mb-1">{type.name}</h4>
                  <p className="text-sm text-gray-400">{type.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </GlassCard>

        {/* Palace Configuration */}
        <GlassCard className="p-6">
          <h3 className="text-xl font-bold text-white mb-6">Palace Configuration</h3>
          
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Palace Name
              </label>
              <input
                type="text"
                value={palaceName}
                onChange={(e) => setPalaceName(e.target.value)}
                placeholder="e.g., My Python Concepts Palace"
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-gray-600 text-white placeholder-gray-400 focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Learning Topic
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Python Programming Fundamentals"
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-gray-600 text-white placeholder-gray-400 focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="block text-sm font-medium text-gray-300">
                  Information Items to Remember
                </label>
                <button
                  onClick={addInformationItem}
                  className="flex items-center space-x-1 text-green-400 hover:text-green-300 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span className="text-sm">Add Item</span>
                </button>
              </div>
              
              <div className="space-y-3">
                {informationItems.map((item, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <input
                      type="text"
                      value={item}
                      onChange={(e) => updateInformationItem(index, e.target.value)}
                      placeholder={`Information item ${index + 1}`}
                      className="flex-1 px-4 py-2 rounded-lg bg-white/10 border border-gray-600 text-white placeholder-gray-400 focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
                    />
                    {informationItems.length > 1 && (
                      <button
                        onClick={() => removeInformationItem(index)}
                        className="text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={createPalace}
              disabled={loading}
              className="w-full px-6 py-3 rounded-lg bg-gradient-to-r from-green-500 to-teal-600 text-white font-medium flex items-center justify-center space-x-2 hover:shadow-lg transition-all duration-300 disabled:opacity-50"
            >
              {loading ? <LoadingSpinner size="sm" /> : <Sparkles className="w-5 h-5" />}
              <span>{loading ? 'Creating Palace...' : 'Create Memory Palace'}</span>
            </motion.button>
          </div>
        </GlassCard>
      </div>
    );
  }

  if (practiceMode) {
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
                <Target className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Practice Session</h1>
                <p className="text-gray-400">{practiceMode.mode.replace('_', ' ')} mode</p>
              </div>
            </div>
            <button
              onClick={() => setPracticeMode(null)}
              className="px-4 py-2 rounded-lg border border-gray-600 text-gray-300 hover:bg-white/5 transition-colors"
            >
              End Practice
            </button>
          </div>
        </motion.div>

        <GlassCard className="p-6">
          <div className="text-white">
            <h3 className="text-xl font-bold mb-4">Practice Instructions</h3>
            <p className="text-gray-300 mb-6">{practiceMode.data.instructions}</p>
            
            {practiceMode.data.questions && (
              <div className="space-y-4">
                <h4 className="text-lg font-semibold">Practice Questions:</h4>
                {practiceMode.data.questions.map((question, index) => (
                  <div key={index} className="p-4 rounded-lg bg-white/5 border border-gray-600">
                    <p className="text-gray-300">{question.question}</p>
                  </div>
                ))}
              </div>
            )}
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
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-green-500 to-teal-600 flex items-center justify-center">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Memory Palace Builder</h1>
              <p className="text-gray-400">Create and manage your spatial memory systems</p>
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setShowBuilder(true)}
            className="px-6 py-2 rounded-lg bg-gradient-to-r from-green-500 to-teal-600 text-white font-medium flex items-center space-x-2 hover:shadow-lg transition-all duration-300"
          >
            <Plus className="w-4 h-4" />
            <span>Create New Palace</span>
          </motion.button>
        </div>
      </motion.div>

      {/* Current Palace Display */}
      {currentPalace && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <GlassCard className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">{currentPalace.name}</h3>
                <p className="text-gray-400 mb-4">{currentPalace.description}</p>
              </div>
              <div className="text-right">
                <div className="text-sm text-gray-400">Palace Type</div>
                <div className="text-lg font-semibold text-green-400 capitalize">{currentPalace.palace_type}</div>
              </div>
            </div>

            {/* Palace Overview */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center p-3 rounded-lg bg-white/5">
                <div className="text-xl font-bold text-blue-400">{currentPalace.room_count || currentPalace.rooms?.length || 0}</div>
                <div className="text-sm text-gray-400">Rooms</div>
              </div>
              <div className="text-center p-3 rounded-lg bg-white/5">
                <div className="text-xl font-bold text-green-400">{currentPalace.information_count || currentPalace.information_nodes?.length || 0}</div>
                <div className="text-sm text-gray-400">Information Items</div>
              </div>
              <div className="text-center p-3 rounded-lg bg-white/5">
                <div className="text-xl font-bold text-purple-400">{(currentPalace.effectiveness_score * 100).toFixed(0)}%</div>
                <div className="text-sm text-gray-400">Effectiveness</div>
              </div>
            </div>

            {/* Practice Modes */}
            <div className="flex space-x-3">
              <button
                onClick={() => startPractice(currentPalace.palace_id, 'recall')}
                className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium flex items-center justify-center space-x-2 hover:shadow-lg transition-all duration-300"
              >
                <Eye className="w-4 h-4" />
                <span>Recall Practice</span>
              </button>
              <button
                onClick={() => startPractice(currentPalace.palace_id, 'navigation')}
                className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-green-500 to-teal-600 text-white font-medium flex items-center justify-center space-x-2 hover:shadow-lg transition-all duration-300"
              >
                <Navigation className="w-4 h-4" />
                <span>Navigation</span>
              </button>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* User Palaces Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {loading ? (
          <div className="col-span-full flex justify-center py-12">
            <LoadingSpinner size="lg" message="Loading your memory palaces..." />
          </div>
        ) : userPalaces.length === 0 ? (
          <div className="col-span-full text-center py-12">
            <BookOpen className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-400 mb-2">No Memory Palaces Yet</h3>
            <p className="text-gray-500 mb-6">Create your first AI-powered memory palace to get started</p>
            <button
              onClick={() => setShowBuilder(true)}
              className="px-6 py-2 rounded-lg bg-gradient-to-r from-green-500 to-teal-600 text-white font-medium hover:shadow-lg transition-all duration-300"
            >
              Create Your First Palace
            </button>
          </div>
        ) : (
          userPalaces.map((palace, index) => (
            <motion.div
              key={palace.palace_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setCurrentPalace(palace)}
              className="cursor-pointer group"
            >
              <GlassCard className="p-6 h-full hover:border-white/20 transition-all duration-300 group-hover:scale-105">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${
                      palaceTypes.find(t => t.id === palace.palace_type)?.color || 'from-gray-500 to-gray-600'
                    } flex items-center justify-center`}>
                      {React.createElement(
                        palaceTypes.find(t => t.id === palace.palace_type)?.icon || BookOpen,
                        { className: "w-6 h-6 text-white" }
                      )}
                    </div>
                    <div>
                      <h4 className="font-bold text-white group-hover:text-green-400 transition-colors">{palace.name}</h4>
                      <p className="text-sm text-gray-400 capitalize">{palace.palace_type.replace('_', ' ')}</p>
                    </div>
                  </div>
                </div>

                <p className="text-gray-300 text-sm mb-4 line-clamp-2">{palace.description}</p>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="text-center p-2 rounded-lg bg-white/5">
                    <div className="text-lg font-bold text-blue-400">{palace.room_count}</div>
                    <div className="text-xs text-gray-400">Rooms</div>
                  </div>
                  <div className="text-center p-2 rounded-lg bg-white/5">
                    <div className="text-lg font-bold text-green-400">{palace.information_count}</div>
                    <div className="text-xs text-gray-400">Items</div>
                  </div>
                </div>

                <div className="text-xs text-gray-500">
                  Created {new Date(palace.created_at).toLocaleDateString()}
                </div>
              </GlassCard>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
};

export default MemoryPalaceBuilder;