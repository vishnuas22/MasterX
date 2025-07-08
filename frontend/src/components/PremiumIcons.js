import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '../utils/cn';

// ===============================
// 🎨 PREMIUM ICON SYSTEM
// ===============================

const iconSizes = {
  xs: 'w-3 h-3',
  sm: 'w-4 h-4', 
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
  xl: 'w-8 h-8',
  '2xl': 'w-10 h-10',
  '3xl': 'w-12 h-12',
};

const IconWrapper = ({ 
  children, 
  size = 'md', 
  className = '', 
  animated = false,
  gradient = false,
  glow = false,
  ...props 
}) => {
  const baseClasses = cn(
    iconSizes[size],
    gradient && 'text-gradient-primary',
    glow && 'filter drop-shadow-glow-blue',
    className
  );

  if (animated) {
    return (
      <motion.div
        className={baseClasses}
        whileHover={{ scale: 1.1, rotate: 5 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <div className={baseClasses} {...props}>
      {children}
    </div>
  );
};

// AI & Learning Icons
export const AIBrainIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3C7.03 3 3 7.03 3 12s4.03 9 9 9 9-4.03 9-9-4.03-9-9-9z"/>
      <path d="M8 12h.01"/>
      <path d="M16 12h.01"/>
      <path d="M12 16h.01"/>
      <path d="M9.5 9.5h.01"/>
      <path d="M14.5 9.5h.01"/>
      <circle cx="12" cy="12" r="2"/>
    </svg>
  </IconWrapper>
);

export const SparkleIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0l2.4 7.2L22 9.6l-7.6 2.4L12 24l-2.4-7.2L2 14.4l7.6-2.4L12 0z"/>
      <path d="M18 2l1.2 3.6L23 7.2l-3.8 1.2L18 12l-1.2-3.6L13 7.2l3.8-1.2L18 2z"/>
      <path d="M6 10l.6 1.8L9 12.6l-1.9.6L6 16l-.6-1.8L3 13.6l1.9-.6L6 10z"/>
    </svg>
  </IconWrapper>
);

export const MindIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5A2.5 2.5 0 0 1 14.5 2a2.5 2.5 0 0 1 2.5 2.5c0 .51-.15 1-.4 1.4l1.9 1.9c.25-.4.4-.89.4-1.4A2.5 2.5 0 0 1 21.5 8.5a2.5 2.5 0 0 1-2.5 2.5c-.51 0-1-.15-1.4-.4L16 12.2c.25.4.4.89.4 1.4a2.5 2.5 0 0 1-2.5 2.5c-.51 0-1-.15-1.4-.4l-1.9 1.9c.25.4.4.89.4 1.4A2.5 2.5 0 0 1 8.5 21.5a2.5 2.5 0 0 1-2.5-2.5c0-.51.15-1 .4-1.4L4.5 16a2.5 2.5 0 0 1-1.4.4A2.5 2.5 0 0 1 .5 13.9a2.5 2.5 0 0 1 2.5-2.5c.51 0 1 .15 1.4.4l1.9-1.9a2.5 2.5 0 0 1-.4-1.4A2.5 2.5 0 0 1 8 6a2.5 2.5 0 0 1 1.5 2.5z"/>
    </svg>
  </IconWrapper>
);

export const LightbulbIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21h6"/>
      <path d="M12 3C8.5 3 6 6 6 9.5c0 1 .2 2 .5 2.8L8 15h8l1.5-2.7c.3-.8.5-1.8.5-2.8C18 6 15.5 3 12 3z"/>
      <path d="M9 18h6"/>
    </svg>
  </IconWrapper>
);

// Chat & Communication Icons
export const MessageIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
    </svg>
  </IconWrapper>
);

export const SendIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 2 11 13"/>
      <path d="M22 2 15 22 11 13 2 9z"/>
    </svg>
  </IconWrapper>
);

// Learning & Education Icons
export const BookIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
    </svg>
  </IconWrapper>
);

export const TargetIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <circle cx="12" cy="12" r="6"/>
      <circle cx="12" cy="12" r="2"/>
    </svg>
  </IconWrapper>
);

export const TrophyIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/>
      <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>
      <path d="M4 22h16"/>
      <path d="M10 14.66V17c0 .55.45 1 1 1h2c.55 0 1-.45 1-1v-2.34"/>
      <path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/>
    </svg>
  </IconWrapper>
);

// Progress & Analytics Icons
export const TrendingUpIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22,7 13.5,15.5 8.5,10.5 2,17"/>
      <polyline points="16,7 22,7 22,13"/>
    </svg>
  </IconWrapper>
);

export const BarChartIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="20" x2="12" y2="10"/>
      <line x1="18" y1="20" x2="18" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="16"/>
    </svg>
  </IconWrapper>
);

// Interface Icons
export const SettingsIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </svg>
  </IconWrapper>
);

export const UserIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>
  </IconWrapper>
);

// Action Icons
export const PlayIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z"/>
    </svg>
  </IconWrapper>
);

export const PauseIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="4" width="4" height="16"/>
      <rect x="14" y="4" width="4" height="16"/>
    </svg>
  </IconWrapper>
);

export const ArrowDownIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19"/>
      <polyline points="19,12 12,19 5,12"/>
    </svg>
  </IconWrapper>
);

export const ChevronLeftIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15,18 9,12 15,6"/>
    </svg>
  </IconWrapper>
);

export const ChevronRightIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9,18 15,12 9,6"/>
    </svg>
  </IconWrapper>
);

export const CloseIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  </IconWrapper>
);

// Status Icons
export const CheckIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20,6 9,17 4,12"/>
    </svg>
  </IconWrapper>
);

export const AlertIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  </IconWrapper>
);

// Advanced UI Icons
export const MagicWandIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 4V2m0 14v-2"/>
      <path d="M8 9h2m8 0h2"/>
      <path d="M15 10L7.5 2.5"/>
      <path d="M9.5 12.5L15 7"/>
      <path d="M9 19l6-6"/>
      <path d="M21 15l-6 6"/>
      <circle cx="16" cy="8" r="2"/>
    </svg>
  </IconWrapper>
);

export const EyeIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  </IconWrapper>
);

export const HeartIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
    </svg>
  </IconWrapper>
);

export const ZapIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
  </IconWrapper>
);

// Brand Icon
export const MasterXIcon = ({ size, animated, gradient, glow, className, ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      <circle cx="12" cy="12" r="2" fill="currentColor" opacity="0.6"/>
      <path d="M8 8l8 8M16 8l-8 8" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.4"/>
    </svg>
  </IconWrapper>
);

// Animated Icon Compositions
export const PulsingDot = ({ size = 'sm', color = 'ai-blue-500' }) => (
  <div className={cn('relative', iconSizes[size])}>
    <div className={cn('absolute inset-0 rounded-full animate-ping opacity-75', `bg-${color}`)} />
    <div className={cn('relative rounded-full', iconSizes[size], `bg-${color}`)} />
  </div>
);

export const LoadingSpinner = ({ size = 'md', color = 'ai-blue-500' }) => (
  <div className={cn(iconSizes[size], 'animate-spin')}>
    <svg viewBox="0 0 24 24">
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
        className="opacity-25"
        fill="none"
      />
      <path
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        className={`text-${color}`}
      />
    </svg>
  </div>
);

export const TypingIndicator = () => (
  <div className="flex items-center space-x-1">
    <div className="flex space-x-1">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-2 h-2 bg-ai-blue-400 rounded-full"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.7, 1, 0.7],
          }}
          transition={{
            duration: 1.4,
            repeat: Infinity,
            delay: i * 0.2,
          }}
        />
      ))}
    </div>
    <span className="text-text-tertiary text-sm ml-2">AI is thinking...</span>
  </div>
);

export const SearchIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8"/>
      <path d="M21 21l-4.35-4.35"/>
    </svg>
  </IconWrapper>
);

export const PlusIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 5v14M5 12h14"/>
    </svg>
  </IconWrapper>
);

export const MoreHorizontalIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="1"/>
      <circle cx="19" cy="12" r="1"/>
      <circle cx="5" cy="12" r="1"/>
    </svg>
  </IconWrapper>
);

export const ShareIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>
      <polyline points="16,6 12,2 8,6"/>
      <line x1="12" y1="2" x2="12" y2="15"/>
    </svg>
  </IconWrapper>
);

export const EditIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>
    </svg>
  </IconWrapper>
);

export const TrashIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="3,6 5,6 21,6"/>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>
  </IconWrapper>
);

export const CrownIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 6L9 9l3-8 3 8-3-3zM19 15l-7-7-7 7h14zM21 16H3l-1 4h20l-1-4z"/>
    </svg>
  </IconWrapper>
);

export const StarIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
  </IconWrapper>
);

export const MicrophoneIcon = ({ size = 'md', animated = false, gradient = false, glow = false, className = '', ...props }) => (
  <IconWrapper size={size} animated={animated} gradient={gradient} glow={glow} className={className} {...props}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  </IconWrapper>
);

export default {
  AIBrainIcon,
  SparkleIcon,
  MindIcon,
  LightbulbIcon,
  MessageIcon,
  SendIcon,
  BookIcon,
  TargetIcon,
  TrophyIcon,
  TrendingUpIcon,
  BarChartIcon,
  SettingsIcon,
  UserIcon,
  PlayIcon,
  PauseIcon,
  ArrowDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  CheckIcon,
  AlertIcon,
  MagicWandIcon,
  EyeIcon,
  HeartIcon,
  ZapIcon,
  MasterXIcon,
  PulsingDot,
  LoadingSpinner,
  TypingIndicator,
  SearchIcon,
  PlusIcon,
  MoreHorizontalIcon,
  ShareIcon,
  EditIcon,
  TrashIcon,
  CrownIcon,
  StarIcon,
  MicrophoneIcon,
};