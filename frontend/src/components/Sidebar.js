import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../context/AppContext';
import { GlassCard, GlassBadge } from './GlassCard';
import { 
  MessageIcon, 
  AIBrainIcon, 
  UserIcon,
  BookIcon,
  TargetIcon,
  LightbulbIcon,
  TrendingUpIcon,
  BarChartIcon,
  TrophyIcon,
  SettingsIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ZapIcon,
  SparkleIcon,
  PulsingDot 
} from './PremiumIcons';
import { cn } from '../utils/cn';

// ===============================
// 🎨 PREMIUM SIDEBAR COMPONENT
// ===============================

export function Sidebar({ isCollapsed, onToggle }) {
  const { state, actions } = useApp();

  const menuItems = [
    {
      id: 'chat',
      label: 'AI Mentor Chat',
      icon: MessageIcon,
      description: 'Interactive learning conversations',
      badge: null,
      gradient: true
    },
    {
      id: 'personalization',
      label: 'Personalization Hub',
      icon: UserIcon,
      description: 'Your learning DNA & adaptive insights',
      badge: 'NEW',
      premium: true
    },
    {
      id: 'learning-psychology',
      label: 'Learning Psychology',
      icon: AIBrainIcon,
      description: 'Advanced cognitive techniques',
      badge: 'BETA',
      premium: true
    },
    {
      id: 'metacognitive-training',
      label: 'Metacognitive Training',
      icon: LightbulbIcon,
      description: 'Think about thinking',
      comingSoon: false
    },
    {
      id: 'memory-palace',
      label: 'Memory Palace',
      icon: BookIcon,
      description: 'Spatial memory techniques',
      comingSoon: false
    },
    {
      id: 'elaborative-questions',
      label: 'Elaborative Questions',
      icon: TargetIcon,
      description: 'Deep questioning skills',
      comingSoon: true
    },
    {
      id: 'transfer-learning',
      label: 'Transfer Learning',
      icon: TrendingUpIcon,
      description: 'Knowledge application across domains',
      comingSoon: true
    }
  ];

  const bottomItems = [
    {
      id: 'analytics',
      label: 'Advanced Analytics',
      icon: BarChartIcon,
      description: 'AI-powered learning insights',
      comingSoon: false
    },
    {
      id: 'achievements',
      label: 'Achievements',
      icon: TrophyIcon,
      description: 'Your learning milestones',
      comingSoon: true
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: SettingsIcon,
      description: 'Customize your experience',
      comingSoon: true
    }
  ];

  const handleItemClick = (itemId) => {
    actions.setActiveView(itemId);
  };

  return (
    <motion.div
      initial={{ width: isCollapsed ? 80 : 320 }}
      animate={{ width: isCollapsed ? 80 : 320 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      className="relative flex flex-col glass-ultra-thick border-r border-border-medium shadow-xl"
    >
      {/* Toggle Button */}
      <motion.button
        onClick={onToggle}
        className="absolute -right-4 top-8 w-8 h-8 glass-medium rounded-full flex items-center justify-center text-text-tertiary hover:text-text-primary hover:glass-thick transition-all duration-200 z-20 shadow-lg"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        {isCollapsed ? (
          <ChevronRightIcon size="sm" />
        ) : (
          <ChevronLeftIcon size="sm" />
        )}
      </motion.button>

      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <motion.div 
          className="flex items-center space-x-3"
          layout
        >
          <div className="relative">
            <div className="w-10 h-10 glass-ai-primary rounded-xl flex items-center justify-center shadow-glow-blue">
              <ZapIcon size="lg" className="text-ai-blue-400" glow />
            </div>
            <PulsingDot size="xs" className="absolute -top-1 -right-1" />
          </div>
          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3, delay: 0.1 }}
              >
                <h1 className="text-title-large font-bold text-gradient-primary">
                  MasterX
                </h1>
                <p className="text-caption text-text-tertiary font-medium">
                  AI Learning Mentor
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* User Info */}
      {state.user && (
        <div className="p-4 border-b border-border-subtle/50">
          <motion.div 
            className="flex items-center space-x-3"
            layout
          >
            <div className="relative">
              <div className="w-9 h-9 bg-gradient-to-br from-ai-green-400 to-ai-blue-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm font-bold font-primary">
                  {state.user.name.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="absolute -bottom-0.5 -right-0.5">
                <PulsingDot size="xs" color="ai-green-500" />
              </div>
            </div>
            <AnimatePresence>
              {!isCollapsed && (
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="min-w-0 flex-1"
                >
                  <p className="text-body font-semibold text-text-primary truncate">
                    {state.user.name}
                  </p>
                  <p className="text-caption text-text-tertiary truncate">
                    {state.user.email}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      )}

      {/* Main Navigation */}
      <nav className="flex-1 overflow-y-auto p-4 space-y-2">
        <motion.div layout className="space-y-1">
          {menuItems.map((item, index) => (
            <SidebarItem
              key={item.id}
              item={item}
              isActive={state.activeView === item.id}
              isCollapsed={isCollapsed}
              onClick={() => handleItemClick(item.id)}
              index={index}
            />
          ))}
        </motion.div>
      </nav>

      {/* Bottom Navigation */}
      <div className="p-4 border-t border-border-subtle space-y-1">
        {bottomItems.map((item, index) => (
          <SidebarItem
            key={item.id}
            item={item}
            isActive={state.activeView === item.id}
            isCollapsed={isCollapsed}
            onClick={() => handleItemClick(item.id)}
            index={index}
            isBottom
          />
        ))}
      </div>

      {/* Premium Badge */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ delay: 0.3 }}
            className="p-4"
          >
            <GlassCard variant="ai-primary" size="sm" className="text-center">
              <div className="flex items-center space-x-2 mb-2">
                <SparkleIcon size="sm" className="text-ai-blue-400" />
                <span className="text-caption font-semibold text-ai-blue-300">
                  Premium Features
                </span>
              </div>
              <p className="text-footnote text-text-tertiary">
                Advanced AI models and personalization active
              </p>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ===============================
// 🎨 SIDEBAR ITEM COMPONENT
// ===============================

function SidebarItem({ 
  item, 
  isActive, 
  isCollapsed, 
  onClick, 
  index, 
  isBottom = false 
}) {
  const Icon = item.icon;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ 
        delay: index * 0.05,
        duration: 0.3,
        ease: "easeOut"
      }}
    >
      <motion.button
        onClick={onClick}
        disabled={item.comingSoon}
        className={cn(
          'w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-200 group relative',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-ai-blue-500/50',
          isActive
            ? 'glass-thick border border-ai-blue-500/30 text-ai-blue-300 shadow-glow-blue'
            : item.comingSoon
              ? 'opacity-60 cursor-not-allowed hover:opacity-70'
              : 'hover:glass-medium text-text-tertiary hover:text-text-primary',
          item.premium && !isActive && 'hover:glass-ai-primary hover:border-ai-blue-500/20'
        )}
        whileHover={!item.comingSoon ? { scale: 1.02, x: 2 } : undefined}
        whileTap={!item.comingSoon ? { scale: 0.98 } : undefined}
      >
        <div className="flex-shrink-0 relative">
          <Icon 
            size="lg" 
            className={cn(
              'transition-colors duration-200',
              isActive 
                ? 'text-ai-blue-400' 
                : item.premium 
                  ? 'text-ai-purple-400'
                  : ''
            )}
            animated={!item.comingSoon}
            gradient={item.gradient && isActive}
          />
          {item.premium && !isCollapsed && (
            <div className="absolute -top-1 -right-1">
              <SparkleIcon size="xs" className="text-ai-purple-400" />
            </div>
          )}
        </div>

        <AnimatePresence>
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
              className="text-left min-w-0 flex-1"
            >
              <div className="flex items-center justify-between mb-1">
                <div className={cn(
                  'font-semibold text-body truncate',
                  isActive ? 'text-ai-blue-300' : ''
                )}>
                  {item.label}
                </div>
                {item.badge && (
                  <GlassBadge 
                    variant={item.badge === 'NEW' ? 'success' : 'primary'}
                    className="ml-2 flex-shrink-0"
                  >
                    {item.badge}
                  </GlassBadge>
                )}
                {item.comingSoon && (
                  <GlassBadge variant="secondary" className="ml-2 flex-shrink-0">
                    Soon
                  </GlassBadge>
                )}
              </div>
              <div className="text-caption text-text-quaternary truncate group-hover:text-text-tertiary transition-colors">
                {item.description}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Active indicator */}
        {isActive && (
          <motion.div
            layoutId="sidebarActiveIndicator"
            className="absolute right-2 w-1 h-8 bg-ai-blue-400 rounded-full shadow-glow-blue"
            initial={false}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />
        )}
      </motion.button>
    </motion.div>
  );
}

export default Sidebar;