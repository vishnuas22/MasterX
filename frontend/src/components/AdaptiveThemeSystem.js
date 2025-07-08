import React, { useState, useEffect, useContext, createContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sun, Moon, Eye, Brain, Palette, Monitor, Sunset, Sunrise } from 'lucide-react';
import { GlassCard, GlassButton } from './GlassCard';

// Theme Context
const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Advanced Theme Types
const THEME_TYPES = {
  DARK_FOCUS: 'dark_focus',
  DARK_IMMERSIVE: 'dark_immersive',
  DARK_MINIMAL: 'dark_minimal',
  LIGHT_CLEAN: 'light_clean',
  LIGHT_WARM: 'light_warm',
  AUTO_ADAPTIVE: 'auto_adaptive',
  CIRCADIAN: 'circadian',
  FOCUS_MODE: 'focus_mode'
};

const THEME_CONFIGS = {
  [THEME_TYPES.DARK_FOCUS]: {
    name: 'Dark Focus',
    icon: Brain,
    description: 'Optimized for deep learning sessions',
    cssVars: {
      '--bg-primary': 'rgb(8, 8, 12)',
      '--bg-secondary': 'rgb(15, 15, 20)',
      '--bg-tertiary': 'rgb(20, 20, 28)',
      '--text-primary': 'rgb(248, 250, 252)',
      '--text-secondary': 'rgb(203, 213, 225)',
      '--accent-primary': 'rgb(59, 130, 246)',
      '--accent-secondary': 'rgb(147, 51, 234)',
      '--glass-bg': 'rgba(255, 255, 255, 0.05)',
      '--glass-border': 'rgba(255, 255, 255, 0.1)',
      '--shadow-color': 'rgba(0, 0, 0, 0.6)'
    }
  },
  [THEME_TYPES.DARK_IMMERSIVE]: {
    name: 'Dark Immersive',
    icon: Eye,
    description: 'Cinematic experience for long sessions',
    cssVars: {
      '--bg-primary': 'rgb(0, 0, 0)',
      '--bg-secondary': 'rgb(10, 10, 10)',
      '--bg-tertiary': 'rgb(20, 20, 20)',
      '--text-primary': 'rgb(255, 255, 255)',
      '--text-secondary': 'rgb(220, 220, 220)',
      '--accent-primary': 'rgb(34, 197, 94)',
      '--accent-secondary': 'rgb(168, 85, 247)',
      '--glass-bg': 'rgba(255, 255, 255, 0.03)',
      '--glass-border': 'rgba(255, 255, 255, 0.08)',
      '--shadow-color': 'rgba(0, 0, 0, 0.8)'
    }
  },
  [THEME_TYPES.LIGHT_CLEAN]: {
    name: 'Light Clean',
    icon: Sun,
    description: 'Bright and clean for daytime learning',
    cssVars: {
      '--bg-primary': 'rgb(255, 255, 255)',
      '--bg-secondary': 'rgb(248, 250, 252)',
      '--bg-tertiary': 'rgb(241, 245, 249)',
      '--text-primary': 'rgb(15, 23, 42)',
      '--text-secondary': 'rgb(51, 65, 85)',
      '--accent-primary': 'rgb(59, 130, 246)',
      '--accent-secondary': 'rgb(147, 51, 234)',
      '--glass-bg': 'rgba(255, 255, 255, 0.7)',
      '--glass-border': 'rgba(0, 0, 0, 0.1)',
      '--shadow-color': 'rgba(0, 0, 0, 0.1)'
    }
  },
  [THEME_TYPES.CIRCADIAN]: {
    name: 'Circadian Rhythm',
    icon: Sunset,
    description: 'Adapts to natural light cycles',
    dynamic: true
  },
  [THEME_TYPES.FOCUS_MODE]: {
    name: 'Ultra Focus',
    icon: Monitor,
    description: 'Minimal distractions for maximum concentration',
    cssVars: {
      '--bg-primary': 'rgb(5, 5, 10)',
      '--bg-secondary': 'rgb(10, 10, 15)',
      '--bg-tertiary': 'rgb(15, 15, 20)',
      '--text-primary': 'rgb(240, 240, 240)',
      '--text-secondary': 'rgb(180, 180, 180)',
      '--accent-primary': 'rgb(99, 102, 241)',
      '--accent-secondary': 'rgb(139, 92, 246)',
      '--glass-bg': 'rgba(255, 255, 255, 0.02)',
      '--glass-border': 'rgba(255, 255, 255, 0.05)',
      '--shadow-color': 'rgba(0, 0, 0, 0.9)'
    }
  }
};

export function ThemeProvider({ children }) {
  const [currentTheme, setCurrentTheme] = useState(THEME_TYPES.DARK_FOCUS);
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [focusIntensity, setFocusIntensity] = useState(0.5);
  const [adaptiveSettings, setAdaptiveSettings] = useState({
    enableCircadian: false,
    enableFocusDetection: true,
    enableEyeStrainReduction: true,
    enableMotionReduction: false
  });

  // Apply theme to CSS variables
  const applyTheme = (theme) => {
    const config = THEME_CONFIGS[theme];
    if (!config || config.dynamic) return;

    const root = document.documentElement;
    Object.entries(config.cssVars).forEach(([property, value]) => {
      root.style.setProperty(property, value);
    });
  };

  // Circadian rhythm calculation
  const getCircadianTheme = () => {
    const hour = new Date().getHours();
    
    if (hour >= 6 && hour < 10) {
      return generateDynamicTheme('sunrise');
    } else if (hour >= 10 && hour < 16) {
      return generateDynamicTheme('day');
    } else if (hour >= 16 && hour < 20) {
      return generateDynamicTheme('sunset');
    } else {
      return generateDynamicTheme('night');
    }
  };

  // Generate dynamic theme based on time/focus
  const generateDynamicTheme = (timeOfDay) => {
    const themes = {
      sunrise: {
        '--bg-primary': 'rgb(15, 12, 20)',
        '--accent-primary': 'rgb(251, 146, 60)',
        '--accent-secondary': 'rgb(244, 114, 182)'
      },
      day: {
        '--bg-primary': 'rgb(248, 250, 252)',
        '--accent-primary': 'rgb(59, 130, 246)',
        '--accent-secondary': 'rgb(16, 185, 129)'
      },
      sunset: {
        '--bg-primary': 'rgb(20, 15, 12)',
        '--accent-primary': 'rgb(251, 113, 133)',
        '--accent-secondary': 'rgb(168, 85, 247)'
      },
      night: {
        '--bg-primary': 'rgb(8, 8, 12)',
        '--accent-primary': 'rgb(99, 102, 241)',
        '--accent-secondary': 'rgb(147, 51, 234)'
      }
    };

    const root = document.documentElement;
    Object.entries(themes[timeOfDay]).forEach(([property, value]) => {
      root.style.setProperty(property, value);
    });
  };

  // Auto theme detection based on system preference
  useEffect(() => {
    if (isAutoMode) {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleChange = () => {
        setCurrentTheme(mediaQuery.matches ? THEME_TYPES.DARK_FOCUS : THEME_TYPES.LIGHT_CLEAN);
      };
      
      handleChange();
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [isAutoMode]);

  // Circadian rhythm updates
  useEffect(() => {
    if (adaptiveSettings.enableCircadian) {
      const interval = setInterval(() => {
        getCircadianTheme();
      }, 60000); // Update every minute
      
      getCircadianTheme(); // Initial application
      return () => clearInterval(interval);
    }
  }, [adaptiveSettings.enableCircadian]);

  // Apply theme changes
  useEffect(() => {
    if (!adaptiveSettings.enableCircadian) {
      applyTheme(currentTheme);
    }
  }, [currentTheme, adaptiveSettings.enableCircadian]);

  // Focus detection (simulated - would integrate with eye tracking in real implementation)
  useEffect(() => {
    if (adaptiveSettings.enableFocusDetection) {
      const handleFocusChange = () => {
        if (document.hasFocus()) {
          setFocusIntensity(prev => Math.min(1, prev + 0.1));
        } else {
          setFocusIntensity(prev => Math.max(0, prev - 0.2));
        }
      };

      window.addEventListener('focus', handleFocusChange);
      window.addEventListener('blur', handleFocusChange);
      
      return () => {
        window.removeEventListener('focus', handleFocusChange);
        window.removeEventListener('blur', handleFocusChange);
      };
    }
  }, [adaptiveSettings.enableFocusDetection]);

  const value = {
    currentTheme,
    setCurrentTheme,
    isAutoMode,
    setIsAutoMode,
    focusIntensity,
    adaptiveSettings,
    setAdaptiveSettings,
    THEME_TYPES,
    THEME_CONFIGS,
    applyTheme,
    getCircadianTheme
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function AdaptiveThemePanel({ isOpen, onClose }) {
  const { 
    currentTheme, 
    setCurrentTheme, 
    isAutoMode, 
    setIsAutoMode,
    adaptiveSettings,
    setAdaptiveSettings,
    THEME_TYPES,
    THEME_CONFIGS,
    focusIntensity
  } = useTheme();

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="w-full max-w-4xl max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <GlassCard className="p-6" variant="premium">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-gradient-to-r from-purple-500 to-blue-500">
                  <Palette className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Adaptive Theme System</h2>
                  <p className="text-gray-400">Optimize your learning environment</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Current Status */}
            <div className="mb-6 p-4 rounded-xl bg-white/5 border border-white/10">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-1">Current Theme</h3>
                  <p className="text-gray-300">{THEME_CONFIGS[currentTheme]?.name}</p>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-400">Focus Intensity</div>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-300"
                        style={{ width: `${focusIntensity * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-white">{Math.round(focusIntensity * 100)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Theme Selection */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Available Themes</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(THEME_CONFIGS).map(([key, config]) => {
                  const IconComponent = config.icon;
                  const isSelected = currentTheme === key;
                  
                  return (
                    <motion.button
                      key={key}
                      onClick={() => setCurrentTheme(key)}
                      className={`p-4 rounded-xl text-left transition-all duration-300 ${
                        isSelected
                          ? 'bg-blue-500/20 border border-blue-400/50'
                          : 'bg-white/5 border border-white/10 hover:bg-white/10'
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center space-x-3 mb-3">
                        <div className="p-2 rounded-lg bg-blue-500/20">
                          <IconComponent className="h-5 w-5 text-blue-400" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-white">{config.name}</h4>
                          {isSelected && (
                            <div className="flex items-center space-x-1 text-xs text-green-400">
                              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                              <span>Active</span>
                            </div>
                          )}
                        </div>
                      </div>
                      <p className="text-sm text-gray-300">{config.description}</p>
                    </motion.button>
                  );
                })}
              </div>
            </div>

            {/* Adaptive Settings */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Adaptive Features</h3>
              <div className="space-y-4">
                {[
                  {
                    key: 'enableCircadian',
                    label: 'Circadian Rhythm',
                    description: 'Automatically adjust theme based on time of day',
                    icon: Sunrise
                  },
                  {
                    key: 'enableFocusDetection',
                    label: 'Focus Detection',
                    description: 'Adapt interface based on attention level',
                    icon: Brain
                  },
                  {
                    key: 'enableEyeStrainReduction',
                    label: 'Eye Strain Reduction',
                    description: 'Optimize colors and contrast for eye comfort',
                    icon: Eye
                  },
                  {
                    key: 'enableMotionReduction',
                    label: 'Reduce Motion',
                    description: 'Minimize animations for sensitive users',
                    icon: Monitor
                  }
                ].map(({ key, label, description, icon: Icon }) => (
                  <div key={key} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                    <div className="flex items-center space-x-3">
                      <Icon className="h-5 w-5 text-blue-400" />
                      <div>
                        <div className="font-medium text-white">{label}</div>
                        <div className="text-sm text-gray-400">{description}</div>
                      </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={adaptiveSettings[key]}
                        onChange={(e) => setAdaptiveSettings(prev => ({
                          ...prev,
                          [key]: e.target.checked
                        }))}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* Auto Mode Toggle */}
            <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-400/20">
              <div className="flex items-center space-x-3">
                <Monitor className="h-6 w-6 text-purple-400" />
                <div>
                  <div className="font-medium text-white">System Auto Mode</div>
                  <div className="text-sm text-gray-400">Follow system dark/light preference</div>
                </div>
              </div>
              <GlassButton
                variant={isAutoMode ? "primary" : "secondary"}
                onClick={() => setIsAutoMode(!isAutoMode)}
              >
                {isAutoMode ? "Enabled" : "Disabled"}
              </GlassButton>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3 mt-6">
              <GlassButton variant="secondary" onClick={onClose}>
                Close
              </GlassButton>
              <GlassButton onClick={onClose}>
                <Palette className="h-4 w-4 mr-2" />
                Apply Settings
              </GlassButton>
            </div>
          </GlassCard>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

// Hook for theme-aware components
export function useAdaptiveTheme() {
  const { currentTheme, focusIntensity, adaptiveSettings } = useTheme();
  
  const getThemeClasses = (baseClasses = '') => {
    const intensityClass = focusIntensity > 0.7 ? 'high-focus' : focusIntensity > 0.3 ? 'medium-focus' : 'low-focus';
    const motionClass = adaptiveSettings.enableMotionReduction ? 'reduced-motion' : '';
    
    return `${baseClasses} theme-${currentTheme} ${intensityClass} ${motionClass}`.trim();
  };

  return {
    currentTheme,
    focusIntensity,
    adaptiveSettings,
    getThemeClasses
  };
}