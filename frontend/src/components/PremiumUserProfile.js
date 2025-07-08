import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GlassCard, GlassButton } from './GlassCard';
import { 
  UserIcon, 
  SettingsIcon, 
  TrophyIcon, 
  BarChartIcon,
  ZapIcon,
  ChevronRightIcon,
  SparkleIcon,
  CrownIcon
} from './PremiumIcons';
import { cn } from '../utils/cn';

// Additional icons for user profile
export function PremiumUserProfile({ user, onAction, className }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleAction = (action) => {
    setIsOpen(false);
    onAction(action);
  };

  const menuItems = [
    {
      icon: UserIcon,
      label: 'Profile Settings',
      action: 'profile',
      description: 'Manage your account'
    },
    {
      icon: BarChartIcon,
      label: 'Learning Progress',
      action: 'progress',
      description: 'View analytics'
    },
    {
      icon: TrophyIcon,
      label: 'Achievements',
      action: 'achievements',
      description: 'Your milestones'
    },
    {
      icon: CrownIcon,
      label: 'Upgrade Plan',
      action: 'upgrade',
      description: 'Go premium'
    },
    {
      icon: SettingsIcon,
      label: 'Settings',
      action: 'settings',
      description: 'App preferences'
    }
  ];

  return (
    <div className={cn("relative", className)} ref={dropdownRef}>
      {/* User Avatar Button */}
      <GlassButton
        size="sm"
        variant="secondary"
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 hover:glass-thick"
      >
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-ai-blue-400 to-ai-purple-500 flex items-center justify-center">
            <span className="text-xs font-bold text-white">
              {user.name ? user.name.charAt(0).toUpperCase() : 'U'}
            </span>
          </div>
          {isOpen ? (
            <motion.div
              animate={{ rotate: 180 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronRightIcon size="xs" className="text-text-tertiary" />
            </motion.div>
          ) : (
            <ChevronRightIcon size="xs" className="text-text-tertiary" />
          )}
        </div>
      </GlassButton>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="absolute top-full right-0 mt-2 w-64 z-50"
          >
            <GlassCard variant="dark" className="glass-thick border border-border-subtle shadow-2xl">
              {/* User Info Header */}
              <div className="p-4 border-b border-border-subtle">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-ai-blue-400 to-ai-purple-500 flex items-center justify-center shadow-glow-blue">
                    <span className="text-sm font-bold text-white">
                      {user.name ? user.name.charAt(0).toUpperCase() : 'U'}
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-text-primary truncate">
                      {user.name || 'User'}
                    </p>
                    <p className="text-xs text-text-tertiary truncate">
                      {user.email}
                    </p>
                    <div className="flex items-center space-x-1 mt-1">
                      <SparkleIcon size="xs" className="text-ai-purple-400" />
                      <span className="text-xs text-ai-purple-400 font-medium">Premium Member</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Menu Items */}
              <div className="p-2">
                {menuItems.map((item, index) => {
                  const IconComponent = item.icon;
                  return (
                    <motion.button
                      key={item.action}
                      onClick={() => handleAction(item.action)}
                      className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-glass-light transition-all duration-200 group"
                      whileHover={{ x: 2 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="w-8 h-8 flex items-center justify-center rounded-lg glass-medium group-hover:glass-thick transition-all duration-200">
                        <IconComponent size="sm" className="text-text-secondary group-hover:text-ai-blue-400" />
                      </div>
                      <div className="flex-1 text-left">
                        <p className="text-sm font-medium text-text-primary group-hover:text-ai-blue-300">
                          {item.label}
                        </p>
                        <p className="text-xs text-text-tertiary">
                          {item.description}
                        </p>
                      </div>
                      <ChevronRightIcon 
                        size="xs" 
                        className="text-text-quaternary group-hover:text-ai-blue-400 opacity-0 group-hover:opacity-100 transition-all duration-200" 
                      />
                    </motion.button>
                  );
                })}
              </div>

              {/* Logout Section */}
              <div className="p-2 border-t border-border-subtle">
                <motion.button
                  onClick={() => handleAction('logout')}
                  className="w-full flex items-center space-x-3 p-3 rounded-lg hover:bg-red-500/10 transition-all duration-200 group"
                  whileHover={{ x: 2 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="w-8 h-8 flex items-center justify-center rounded-lg glass-medium group-hover:bg-red-500/20 transition-all duration-200">
                    <ZapIcon size="sm" className="text-text-secondary group-hover:text-red-400" />
                  </div>
                  <div className="flex-1 text-left">
                    <p className="text-sm font-medium text-text-primary group-hover:text-red-300">
                      Sign Out
                    </p>
                    <p className="text-xs text-text-tertiary">
                      End your session
                    </p>
                  </div>
                </motion.button>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
