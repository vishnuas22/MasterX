import React from 'react';
import { motion } from 'framer-motion';
import { 
  MessageCircle, 
  Brain, 
  Users, 
  TrendingUp, 
  Settings, 
  ChevronLeft,
  ChevronRight,
  User,
  BookOpen,
  Target,
  Lightbulb,
  Activity,
  Award,
  BarChart3,
  Zap
} from 'lucide-react';
import { useApp } from '../context/AppContext';

export function Sidebar({ isCollapsed, onToggle }) {
  const { state, actions } = useApp();

  const menuItems = [
    {
      id: 'chat',
      label: 'AI Mentor Chat',
      icon: MessageCircle,
      description: 'Interactive learning conversations'
    },
    {
      id: 'personalization',
      label: 'Personalization Hub',
      icon: User,
      description: 'Your learning DNA & adaptive insights'
    },
    {
      id: 'learning-psychology',
      label: 'Learning Psychology',
      icon: Brain,
      description: 'Advanced cognitive techniques'
    },
    {
      id: 'metacognitive-training',
      label: 'Metacognitive Training',
      icon: Lightbulb,
      description: 'Think about thinking'
    },
    {
      id: 'memory-palace',
      label: 'Memory Palace',
      icon: BookOpen,
      description: 'Spatial memory techniques'
    },
    {
      id: 'elaborative-questions',
      label: 'Elaborative Questions',
      icon: Target,
      description: 'Deep questioning skills'
    },
    {
      id: 'transfer-learning',
      label: 'Transfer Learning',
      icon: TrendingUp,
      description: 'Knowledge application across domains'
    }
  ];

  const bottomItems = [
    {
      id: 'analytics',
      label: 'Analytics',
      icon: BarChart3,
      description: 'Learning progress insights'
    },
    {
      id: 'achievements',
      label: 'Achievements',
      icon: Award,
      description: 'Your learning milestones'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Settings,
      description: 'Customize your experience'
    }
  ];

  const handleItemClick = (itemId) => {
    actions.setActiveView(itemId);
  };

  return (
    <motion.div
      initial={{ width: isCollapsed ? 80 : 280 }}
      animate={{ width: isCollapsed ? 80 : 280 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="bg-gray-900/80 backdrop-blur-xl border-r border-gray-800/50 flex flex-col relative"
    >
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="absolute -right-3 top-6 w-6 h-6 bg-gray-800 border border-gray-700 rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-gray-700 transition-colors z-10"
      >
        {isCollapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
      </button>

      {/* Header */}
      <div className="p-6 border-b border-gray-800/50">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          {!isCollapsed && (
            <div>
              <h1 className="text-lg font-bold text-white">MasterX</h1>
              <p className="text-xs text-gray-400">AI Learning Mentor</p>
            </div>
          )}
        </div>
      </div>

      {/* User Info */}
      {state.user && (
        <div className="p-4 border-b border-gray-800/30">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm font-bold">
                {state.user.name.charAt(0).toUpperCase()}
              </span>
            </div>
            {!isCollapsed && (
              <div className="min-w-0 flex-1">
                <p className="text-white text-sm font-medium truncate">{state.user.name}</p>
                <p className="text-gray-400 text-xs truncate">{state.user.email}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Navigation */}
      <nav className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = state.activeView === item.id;
            
            return (
              <motion.button
                key={item.id}
                onClick={() => handleItemClick(item.id)}
                className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 group ${
                  isActive
                    ? 'bg-blue-500/20 border border-blue-400/30 text-blue-300'
                    : 'hover:bg-gray-800/50 text-gray-400 hover:text-gray-300'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-blue-400' : ''}`} />
                {!isCollapsed && (
                  <div className="text-left min-w-0 flex-1">
                    <div className={`font-medium text-sm ${isActive ? 'text-blue-300' : ''}`}>
                      {item.label}
                    </div>
                    <div className="text-xs text-gray-500 truncate group-hover:text-gray-400">
                      {item.description}
                    </div>
                  </div>
                )}
                
                {/* Active indicator */}
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="w-1 h-6 bg-blue-400 rounded-full"
                    initial={false}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
              </motion.button>
            );
          })}
        </div>
      </nav>

      {/* Bottom Navigation */}
      <div className="p-4 border-t border-gray-800/30">
        <div className="space-y-2">
          {bottomItems.map((item) => {
            const Icon = item.icon;
            const isActive = state.activeView === item.id;
            
            return (
              <motion.button
                key={item.id}
                onClick={() => handleItemClick(item.id)}
                className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 ${
                  isActive
                    ? 'bg-blue-500/20 border border-blue-400/30 text-blue-300'
                    : 'hover:bg-gray-800/50 text-gray-400 hover:text-gray-300'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-blue-400' : ''}`} />
                {!isCollapsed && (
                  <div className="text-left min-w-0 flex-1">
                    <div className={`font-medium text-sm ${isActive ? 'text-blue-300' : ''}`}>
                      {item.label}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      {item.description}
                    </div>
                  </div>
                )}
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Premium Badge */}
      {!isCollapsed && (
        <div className="p-4">
          <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-400/30 rounded-xl p-3">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="w-4 h-4 text-purple-400" />
              <span className="text-purple-300 text-sm font-medium">Premium Features</span>
            </div>
            <p className="text-xs text-gray-400">
              Advanced personalization and AI models active
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
}
