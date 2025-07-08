import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '../utils/cn';

// ===============================
// 🎨 PREMIUM LOADING SYSTEM
// ===============================

const sizes = {
  xs: 'w-3 h-3',
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
  xl: 'w-12 h-12',
  '2xl': 'w-16 h-16',
};

// Premium Circular Spinner
export const LoadingSpinner = ({ 
  size = 'md', 
  color = 'ai-blue-500',
  className = '',
  ...props 
}) => (
  <div className={cn('relative', sizes[size], className)} {...props}>
    <svg 
      className="animate-spin text-current" 
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="2"
        className="opacity-20"
      />
      <path
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        className={`text-${color}`}
      />
    </svg>
  </div>
);

// AI Typing Indicator
export const TypingIndicator = ({ 
  size = 'sm',
  message = 'AI is thinking...',
  showMessage = true,
  className = '' 
}) => (
  <div className={cn('flex items-center space-x-3', className)}>
    <div className="flex space-x-1">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className={cn(
            'bg-ai-blue-400 rounded-full',
            size === 'xs' && 'w-1 h-1',
            size === 'sm' && 'w-1.5 h-1.5',
            size === 'md' && 'w-2 h-2',
            size === 'lg' && 'w-2.5 h-2.5'
          )}
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.6, 1, 0.6],
          }}
          transition={{
            duration: 1.4,
            repeat: Infinity,
            delay: i * 0.2,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
    {showMessage && (
      <span className="text-text-tertiary text-sm font-primary">
        {message}
      </span>
    )}
  </div>
);

// Pulsing Dot Indicator
export const PulsingDot = ({ 
  size = 'sm', 
  color = 'ai-blue-500',
  intensity = 'medium',
  className = '' 
}) => {
  const intensityClasses = {
    low: 'animate-pulse',
    medium: 'animate-pulse-soft',
    high: 'animate-ping',
  };

  return (
    <div className={cn('relative', sizes[size], className)}>
      <div className={cn(
        'absolute inset-0 rounded-full opacity-75',
        `bg-${color}`,
        intensityClasses[intensity]
      )} />
      <div className={cn(
        'relative rounded-full',
        sizes[size],
        `bg-${color}`
      )} />
    </div>
  );
};

// Skeleton Loader
export const SkeletonLoader = ({ 
  className = '',
  lines = 3,
  avatar = false,
  button = false 
}) => (
  <div className={cn('animate-shimmer', className)}>
    <div className="flex items-start space-x-4">
      {avatar && (
        <div className="w-10 h-10 rounded-full glass-thin" />
      )}
      <div className="flex-1 space-y-3">
        {Array.from({ length: lines }).map((_, i) => (
          <div 
            key={i}
            className={cn(
              'h-4 glass-thin rounded-lg',
              i === lines - 1 ? 'w-3/4' : 'w-full'
            )}
          />
        ))}
        {button && (
          <div className="w-24 h-8 glass-thin rounded-lg mt-4" />
        )}
      </div>
    </div>
  </div>
);

// Progress Bar
export const ProgressBar = ({ 
  progress = 0, 
  size = 'md',
  color = 'ai-blue-500',
  showPercentage = false,
  animated = true,
  className = '' 
}) => {
  const heightClasses = {
    xs: 'h-1',
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4',
  };

  return (
    <div className={cn('w-full', className)}>
      <div className={cn(
        'w-full glass-thin rounded-full overflow-hidden',
        heightClasses[size]
      )}>
        <motion.div
          className={cn(
            'h-full rounded-full',
            `bg-${color}`,
            animated && 'transition-all duration-500 ease-out'
          )}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      {showPercentage && (
        <div className="flex justify-between mt-1 text-xs text-text-tertiary">
          <span>Progress</span>
          <span>{Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
};

// Circular Progress
export const CircularProgress = ({ 
  progress = 0,
  size = 'md',
  color = 'ai-blue-500',
  strokeWidth = 4,
  showPercentage = true,
  className = '' 
}) => {
  const sizeMap = {
    sm: 40,
    md: 60,
    lg: 80,
    xl: 100,
  };

  const radius = (sizeMap[size] - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div className={cn('relative', className)}>
      <svg
        width={sizeMap[size]}
        height={sizeMap[size]}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={sizeMap[size] / 2}
          cy={sizeMap[size] / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-border-subtle"
        />
        {/* Progress circle */}
        <motion.circle
          cx={sizeMap[size] / 2}
          cy={sizeMap[size] / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          className={`text-${color}`}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: 'easeInOut' }}
          style={{
            strokeDasharray,
          }}
        />
      </svg>
      {showPercentage && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-semibold text-text-primary">
            {Math.round(progress)}%
          </span>
        </div>
      )}
    </div>
  );
};

// Loading States Component
export const LoadingStates = ({ 
  state = 'loading',
  message,
  size = 'md',
  className = '' 
}) => {
  const stateConfig = {
    loading: {
      component: LoadingSpinner,
      defaultMessage: 'Loading...',
      color: 'ai-blue-500',
    },
    thinking: {
      component: TypingIndicator,
      defaultMessage: 'AI is thinking...',
      color: 'ai-purple-500',
    },
    processing: {
      component: PulsingDot,
      defaultMessage: 'Processing...',
      color: 'ai-green-500',
    },
    error: {
      component: ({ size }) => (
        <div className={cn('text-ai-red-500', sizes[size])}>
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM13 17h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
        </div>
      ),
      defaultMessage: 'Something went wrong',
      color: 'ai-red-500',
    },
    success: {
      component: ({ size }) => (
        <div className={cn('text-ai-green-500', sizes[size])}>
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        </div>
      ),
      defaultMessage: 'Success!',
      color: 'ai-green-500',
    },
  };

  const config = stateConfig[state];
  const Component = config.component;

  return (
    <div className={cn('flex items-center space-x-3', className)}>
      <Component size={size} color={config.color} />
      {message !== false && (
        <span className="text-text-secondary text-sm font-primary">
          {message || config.defaultMessage}
        </span>
      )}
    </div>
  );
};

// Floating Action Loading
export const FloatingActionLoading = ({ 
  isVisible = true,
  className = '' 
}) => {
  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      className={cn(
        'fixed bottom-6 right-6 z-50',
        'glass-thick rounded-2xl p-4',
        'shadow-xl',
        className
      )}
    >
      <LoadingStates state="thinking" size="lg" />
    </motion.div>
  );
};

// Page Loading Overlay
export const PageLoadingOverlay = ({ 
  isVisible = true,
  message = 'Loading MasterX...',
  className = '' 
}) => {
  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={cn(
        'fixed inset-0 z-max',
        'bg-bg-primary/80 backdrop-blur-lg',
        'flex items-center justify-center',
        className
      )}
    >
      <div className="text-center">
        <div className="mb-6">
          <LoadingSpinner size="2xl" />
        </div>
        <h2 className="text-title font-semibold text-text-primary mb-2">
          {message}
        </h2>
        <p className="text-body text-text-tertiary">
          Preparing your AI learning experience...
        </p>
      </div>
    </motion.div>
  );
};

export default LoadingSpinner;