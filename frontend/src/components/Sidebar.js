import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../context/AppContext';
import { GlassCard, GlassBadge, GlassButton } from './GlassCard';
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
  PulsingDot,
  SearchIcon,
  PlusIcon,
  MoreHorizontalIcon,
  ShareIcon,
  EditIcon,
  TrashIcon,
  CrownIcon,
  StarIcon
} from './PremiumIcons';
import { cn } from '../utils/cn';

// ===============================
// 🎨 CHATGPT-STYLE SIDEBAR COMPONENT
// ===============================

export function Sidebar({ isCollapsed, onToggle }) {
  const { state, actions } = useApp();
  const [searchQuery, setSearchQuery] = useState('');
  const [showChatOptions, setShowChatOptions] = useState(null);
  const searchRef = useRef(null);

  // Premium Features - Alphabetical order as requested
  const premiumFeatures = [
    {
      id: 'analytics',
      label: 'Analytics',
      icon: BarChartIcon,
      description: 'Advanced learning insights',
      implemented: true
    },
    {
      id: 'achievements',
      label: 'Achievements',
      icon: TrophyIcon,
      description: 'Learning milestones',
      implemented: false
    },
    {
      id: 'memory-palace',
      label: 'Memory Palace',
      icon: BookIcon,
      description: 'Spatial memory techniques',
      implemented: true
    },
    {
      id: 'personalization',
      label: 'Personalization Hub',
      icon: UserIcon,
      description: 'Learning DNA & insights',
      implemented: true
    }
  ];

  // Get actual chats from app state - no mock data
  const userChats = state.userChats || []; // Use actual chat data from state

  const handleNewChat = () => {
    // Will implement backend functionality later
    console.log('New chat clicked');
    // For now, just navigate to chat
    actions.setActiveView('chat');
  };

  const handleSearchChats = (query) => {
    setSearchQuery(query);
    // Will implement search functionality later
    console.log('Searching chats:', query);
  };

  const handleChatAction = (chatId, action) => {
    console.log(`Chat ${chatId} - ${action}`);
    setShowChatOptions(null);
    // Will implement backend functionality later
  };

  const handleItemClick = (itemId) => {
    actions.setActiveView(itemId);
  };

  return (
    <motion.div
      initial={{ width: isCollapsed ? 64 : 280 }}
      animate={{ width: isCollapsed ? 64 : 280 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
      className="relative h-full flex flex-col glass-ultra-thick border-r border-border-medium shadow-xl bg-gray-900/95"
    >
      {/* Header Section */}
      <div className="flex-shrink-0 p-3 border-b border-border-subtle/30">
        <div className="flex items-center justify-between">
          {/* MasterX Icon - Top Left */}
          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex items-center space-x-3"
              >
                <div className="relative">
                  <div className="w-8 h-8 glass-ai-primary rounded-lg flex items-center justify-center">
                    <ZapIcon size="md" className="text-ai-blue-400" />
                  </div>
                  <PulsingDot size="xs" className="absolute -top-1 -right-1" />
                </div>
                <h1 className="text-lg font-bold text-white">MasterX</h1>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Sidebar Toggle - Top Right */}
          <motion.button
            onClick={onToggle}
            className="w-8 h-8 glass-medium rounded-lg flex items-center justify-center text-gray-400 hover:text-white hover:glass-thick transition-all duration-200"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isCollapsed ? (
              <ChevronRightIcon size="sm" />
            ) : (
              <ChevronLeftIcon size="sm" />
            )}
          </motion.button>
        </div>
      </div>

      {/* New Chat Button */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex-shrink-0 p-3"
          >
            <GlassButton
              onClick={handleNewChat}
              className="w-full justify-start space-x-3 p-3 glass-medium hover:glass-thick border border-gray-700/50 hover:border-gray-600/50"
              variant="secondary"
            >
              <PlusIcon size="sm" className="text-gray-400" />
              <span className="text-sm font-medium text-white">New chat</span>
            </GlassButton>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Search Chats */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex-shrink-0 px-3 pb-4"
          >
            <div className="relative">
              <SearchIcon size="sm" className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                ref={searchRef}
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearchChats(e.target.value)}
                placeholder="Search chats..."
                className="w-full pl-10 pr-4 py-2 text-sm bg-gray-800/50 border border-gray-700/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-ai-blue-500/50 focus:bg-gray-800/70 transition-all"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Premium Features Section */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex-shrink-0 px-3 pb-4"
          >
            <div className="flex items-center space-x-2 mb-3">
              <CrownIcon size="sm" className="text-ai-purple-400" />
              <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Premium Features</h3>
            </div>
            <div className="space-y-1">
              {premiumFeatures.map((feature) => (
                <SidebarItem
                  key={feature.id}
                  item={feature}
                  isActive={state.activeView === feature.id}
                  onClick={() => handleItemClick(feature.id)}
                  isPremium
                  isCompact
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chats Section - Flexible Height */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex-1 flex flex-col min-h-0 px-3"
          >
            <div className="flex items-center space-x-2 mb-3">
              <MessageIcon size="sm" className="text-gray-400" />
              <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Chats</h3>
              {userChats.length > 0 && (
                <span className="text-xs text-gray-500">({userChats.length})</span>
              )}
            </div>
            
            {/* Scrollable Chat List */}
            <div className="flex-1 overflow-y-auto space-y-1 pb-4">
              {userChats.length > 0 ? (
                userChats
                  .filter(chat => chat.title?.toLowerCase().includes(searchQuery.toLowerCase()))
                  .map((chat) => (
                    <ChatItem
                      key={chat.id}
                      chat={chat}
                      isActive={state.currentSession?.id === chat.id}
                      onAction={handleChatAction}
                      showOptions={showChatOptions === chat.id}
                      onToggleOptions={() => setShowChatOptions(showChatOptions === chat.id ? null : chat.id)}
                    />
                  ))
              ) : (
                <div className="text-center py-8">
                  <MessageIcon size="lg" className="text-gray-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">No chats yet</p>
                  <p className="text-xs text-gray-600">Start a conversation to see your chats here</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upgrade Plan - Fixed Bottom */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="flex-shrink-0 p-3 border-t border-border-subtle/30 bg-gray-900/98"
          >
            <GlassCard variant="ai-primary" size="sm" className="text-center bg-gradient-to-r from-ai-purple-500/10 to-ai-blue-500/10 border border-ai-purple-500/30">
              <div className="flex items-center space-x-2 mb-2">
                <StarIcon size="sm" className="text-ai-purple-400" />
                <span className="text-xs font-semibold text-ai-purple-300">Upgrade Plan</span>
              </div>
              <p className="text-xs text-gray-400 mb-3">Ultra Premium Models</p>
              <GlassButton
                size="sm"
                variant="gradient"
                className="w-full text-xs"
              >
                Upgrade Now
              </GlassButton>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ===============================
// 🎨 COMPACT SIDEBAR ITEM COMPONENT
// ===============================

function SidebarItem({ 
  item, 
  isActive, 
  onClick, 
  isPremium = false,
  isCompact = false
}) {
  const Icon = item.icon;

  return (
    <motion.button
      onClick={onClick}
      disabled={!item.implemented}
      className={cn(
        'w-full flex items-center space-x-3 p-2 rounded-lg transition-all duration-200 group relative text-left',
        'focus:outline-none focus-visible:ring-1 focus-visible:ring-ai-blue-500/50',
        isActive
          ? 'bg-ai-blue-500/20 text-ai-blue-300 border border-ai-blue-500/30'
          : item.implemented
            ? 'text-gray-300 hover:bg-gray-800/50 hover:text-white'
            : 'opacity-40 cursor-not-allowed text-gray-500',
        isCompact && 'py-2'
      )}
      whileHover={item.implemented ? { x: 2 } : undefined}
      whileTap={item.implemented ? { scale: 0.98 } : undefined}
    >
      <div className="flex-shrink-0 relative">
        <Icon 
          size={isCompact ? "sm" : "md"} 
          className={cn(
            'transition-colors duration-200',
            isActive 
              ? 'text-ai-blue-400' 
              : isPremium 
                ? 'text-ai-purple-400'
                : 'text-gray-400 group-hover:text-white'
          )}
        />
        {isPremium && (
          <div className="absolute -top-1 -right-1">
            <SparkleIcon size="xs" className="text-ai-purple-400" />
          </div>
        )}
      </div>

      <div className="text-left min-w-0 flex-1">
        <div className={cn(
          'font-medium truncate',
          isCompact ? 'text-sm' : 'text-sm',
          isActive ? 'text-ai-blue-300' : 'text-inherit'
        )}>
          {item.label}
        </div>
        {!item.implemented && (
          <div className="text-xs text-gray-500">Coming Soon</div>
        )}
      </div>
    </motion.button>
  );
}

// ===============================
// 🎨 CHAT ITEM COMPONENT
// ===============================

function ChatItem({ 
  chat, 
  isActive, 
  onAction, 
  showOptions, 
  onToggleOptions 
}) {
  return (
    <div className="relative group">
      <motion.button
        className={cn(
          'w-full flex items-center justify-between p-2 rounded-lg transition-all duration-200 text-left',
          'focus:outline-none focus-visible:ring-1 focus-visible:ring-ai-blue-500/50',
          isActive
            ? 'bg-ai-blue-500/20 text-ai-blue-300 border border-ai-blue-500/30'
            : 'text-gray-300 hover:bg-gray-800/50 hover:text-white'
        )}
        whileHover={{ x: 2 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="flex items-center space-x-3 min-w-0 flex-1">
          <MessageIcon size="sm" className="flex-shrink-0 text-gray-400" />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium truncate">{chat.title}</div>
            <div className="text-xs text-gray-500 truncate">{chat.timestamp}</div>
          </div>
        </div>
        
        <motion.button
          onClick={(e) => {
            e.stopPropagation();
            onToggleOptions();
          }}
          className="opacity-0 group-hover:opacity-100 p-1 rounded-md hover:bg-gray-700/50 transition-all duration-200"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <MoreHorizontalIcon size="sm" className="text-gray-400" />
        </motion.button>
      </motion.button>

      {/* Chat Options Menu */}
      <AnimatePresence>
        {showOptions && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            className="absolute right-0 top-full mt-1 w-36 bg-gray-800/95 backdrop-blur-xl border border-gray-700/50 rounded-lg shadow-xl z-50"
          >
            <div className="py-1">
              <ChatOptionButton
                onClick={() => onAction(chat.id, 'share')}
                icon={ShareIcon}
                label="Share"
              />
              <ChatOptionButton
                onClick={() => onAction(chat.id, 'rename')}
                icon={EditIcon}
                label="Rename"
              />
              <ChatOptionButton
                onClick={() => onAction(chat.id, 'delete')}
                icon={TrashIcon}
                label="Delete"
                danger
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ===============================
// 🎨 CHAT OPTION BUTTON
// ===============================

function ChatOptionButton({ onClick, icon: Icon, label, danger = false }) {
  return (
    <motion.button
      onClick={onClick}
      className={cn(
        'w-full flex items-center space-x-3 px-3 py-2 text-sm transition-colors',
        danger 
          ? 'text-red-400 hover:bg-red-500/10 hover:text-red-300'
          : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'
      )}
      whileHover={{ x: 2 }}
    >
      <Icon size="sm" />
      <span>{label}</span>
    </motion.button>
  );
}

export default Sidebar;