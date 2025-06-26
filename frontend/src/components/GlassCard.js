import React, { forwardRef } from 'react';
import { motion } from 'framer-motion';
import { cn } from '../utils/cn';

// ===============================
// 🎨 PREMIUM GLASS CARD SYSTEM
// ===============================

const glassVariants = {
  'ultra-thin': 'glass-ultra-thin',
  'thin': 'glass-thin',
  'medium': 'glass-medium',
  'thick': 'glass-thick',
  'ultra-thick': 'glass-ultra-thick',
  'premium': 'glass-medium hover-glow',
  'ai-primary': 'bg-gradient-to-br from-ai-blue-500/8 to-ai-purple-500/5 border border-ai-blue-500/20 backdrop-blur-xl shadow-glow-blue',
  'ai-secondary': 'bg-gradient-to-br from-ai-purple-500/8 to-ai-red-500/5 border border-ai-purple-500/20 backdrop-blur-xl shadow-glow-purple',
};

const sizeVariants = {
  'sm': 'p-3 rounded-lg',
  'md': 'p-4 rounded-xl',
  'lg': 'p-6 rounded-2xl',
  'xl': 'p-8 rounded-3xl',
};

export const GlassCard = forwardRef(({ 
  children, 
  className = '', 
  variant = 'medium',
  size = 'md',
  hover = true,
  animated = true,
  ...props 
}, ref) => {
  const baseClasses = cn(
    'relative',
    'backdrop-blur-premium',
    'transition-all duration-300 ease-apple',
    glassVariants[variant],
    sizeVariants[size],
    hover && 'hover-lift',
    className
  );

  if (animated) {
    return (
      <motion.div
        ref={ref}
        className={baseClasses}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ 
          duration: 0.4, 
          ease: [0.175, 0.885, 0.32, 1.275] 
        }}
        whileHover={hover ? { 
          y: -2, 
          scale: 1.01,
          transition: { duration: 0.2 }
        } : undefined}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <div ref={ref} className={baseClasses} {...props}>
      {children}
    </div>
  );
});

GlassCard.displayName = 'GlassCard';

// ===============================
// 🎨 PREMIUM GLASS BUTTON SYSTEM
// ===============================

const buttonVariants = {
  primary: cn(
    'bg-ai-blue-500/20 hover:bg-ai-blue-500/30',
    'border border-ai-blue-500/30 hover:border-ai-blue-500/50',
    'text-ai-blue-100 hover:text-white',
    'shadow-glow-blue hover:shadow-xl'
  ),
  secondary: cn(
    'glass-thin hover:glass-medium',
    'border-border-medium hover:border-border-strong',
    'text-text-secondary hover:text-text-primary'
  ),
  tertiary: cn(
    'bg-transparent hover:glass-thin',
    'border border-transparent hover:border-border-subtle',
    'text-text-tertiary hover:text-text-secondary'
  ),
  danger: cn(
    'bg-ai-red-500/20 hover:bg-ai-red-500/30',
    'border border-ai-red-500/30 hover:border-ai-red-500/50',
    'text-ai-red-100 hover:text-white',
    'shadow-glow-red'
  ),
  success: cn(
    'bg-ai-green-500/20 hover:bg-ai-green-500/30',
    'border border-ai-green-500/30 hover:border-ai-green-500/50',
    'text-ai-green-100 hover:text-white',
    'shadow-glow-green'
  ),
  gradient: cn(
    'bg-gradient-to-r from-ai-blue-500/20 to-ai-purple-500/20',
    'hover:from-ai-blue-500/30 hover:to-ai-purple-500/30',
    'border border-ai-blue-500/20 hover:border-ai-purple-500/30',
    'text-gradient-primary hover:text-white'
  ),
};

const buttonSizes = {
  xs: 'px-2 py-1 text-xs rounded-md',
  sm: 'px-3 py-1.5 text-sm rounded-lg',
  md: 'px-4 py-2 text-base rounded-xl',
  lg: 'px-6 py-3 text-lg rounded-2xl',
  xl: 'px-8 py-4 text-xl rounded-3xl',
};

export const GlassButton = forwardRef(({ 
  children, 
  className = '', 
  variant = 'secondary',
  size = 'md',
  disabled = false,
  loading = false,
  animated = true,
  ...props 
}, ref) => {
  const baseClasses = cn(
    'relative inline-flex items-center justify-center',
    'font-medium font-primary',
    'backdrop-blur-premium',
    'transition-all duration-200 ease-apple',
    'focus:outline-none focus-visible:ring-2 focus-visible:ring-ai-blue-500/50 focus-visible:ring-offset-2 focus-visible:ring-offset-transparent',
    'active:scale-95',
    disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
    buttonVariants[variant],
    buttonSizes[size],
    className
  );

  const content = (
    <>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      <span className={cn('flex items-center', loading && 'opacity-0')}>
        {children}
      </span>
    </>
  );

  if (animated && !disabled) {
    return (
      <motion.button
        ref={ref}
        className={baseClasses}
        whileHover={{ scale: 1.02, y: -1 }}
        whileTap={{ scale: 0.98 }}
        transition={{ duration: 0.1 }}
        disabled={disabled || loading}
        {...props}
      >
        {content}
      </motion.button>
    );
  }

  return (
    <button 
      ref={ref} 
      className={baseClasses} 
      disabled={disabled || loading}
      {...props}
    >
      {content}
    </button>
  );
});

GlassButton.displayName = 'GlassButton';

// ===============================
// 🎨 PREMIUM GLASS INPUT SYSTEM
// ===============================

export const GlassInput = forwardRef(({ 
  className = '', 
  error = false,
  icon,
  rightIcon,
  ...props 
}, ref) => {
  const inputClasses = cn(
    'w-full',
    'glass-medium',
    'rounded-xl',
    'px-4 py-3',
    'text-body text-text-primary placeholder:text-text-quaternary',
    'font-primary font-normal',
    'backdrop-blur-premium',
    'border transition-all duration-200 ease-apple',
    error 
      ? 'border-ai-red-500/50 focus:border-ai-red-500 focus:shadow-glow-red' 
      : 'border-border-medium focus:border-ai-blue-500/50 focus:shadow-glow-blue',
    'focus:outline-none focus:ring-0',
    'focus:scale-[1.01]',
    icon && 'pl-12',
    rightIcon && 'pr-12',
    className
  );

  return (
    <div className="relative">
      {icon && (
        <div className="absolute left-4 top-1/2 transform -translate-y-1/2 text-text-quaternary">
          {icon}
        </div>
      )}
      <input ref={ref} className={inputClasses} {...props} />
      {rightIcon && (
        <div className="absolute right-4 top-1/2 transform -translate-y-1/2 text-text-quaternary">
          {rightIcon}
        </div>
      )}
    </div>
  );
});

GlassInput.displayName = 'GlassInput';

// ===============================
// 🎨 PREMIUM GLASS TEXTAREA
// ===============================

export const GlassTextarea = forwardRef(({ 
  className = '', 
  error = false,
  ...props 
}, ref) => {
  const textareaClasses = cn(
    'w-full',
    'glass-medium',
    'rounded-xl',
    'px-4 py-3',
    'text-body text-text-primary placeholder:text-text-quaternary',
    'font-primary font-normal',
    'backdrop-blur-premium',
    'border transition-all duration-200 ease-apple',
    'resize-none',
    error 
      ? 'border-ai-red-500/50 focus:border-ai-red-500 focus:shadow-glow-red' 
      : 'border-border-medium focus:border-ai-blue-500/50 focus:shadow-glow-blue',
    'focus:outline-none focus:ring-0',
    'focus:scale-[1.01]',
    className
  );

  return <textarea ref={ref} className={textareaClasses} {...props} />;
});

GlassTextarea.displayName = 'GlassTextarea';

// ===============================
// 🎨 PREMIUM GLASS MODAL
// ===============================

export const GlassModal = ({ 
  isOpen, 
  onClose, 
  children, 
  className = '',
  size = 'md',
  showCloseButton = true 
}) => {
  const sizeClasses = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-7xl',
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-modal flex items-center justify-center p-4"
    >
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className={cn(
          'relative w-full',
          'glass-ultra-thick',
          'rounded-3xl',
          'shadow-2xl',
          sizeClasses[size],
          className
        )}
      >
        {showCloseButton && (
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center rounded-full glass-thin hover:glass-medium transition-all duration-200"
          >
            <svg className="w-4 h-4 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
        {children}
      </motion.div>
    </motion.div>
  );
};

// ===============================
// 🎨 PREMIUM GLASS BADGE
// ===============================

const badgeVariants = {
  primary: 'bg-ai-blue-500/20 text-ai-blue-100 border-ai-blue-500/30',
  secondary: 'glass-thin text-text-secondary border-border-medium',
  success: 'bg-ai-green-500/20 text-ai-green-100 border-ai-green-500/30',
  warning: 'bg-yellow-500/20 text-yellow-100 border-yellow-500/30',
  danger: 'bg-ai-red-500/20 text-ai-red-100 border-ai-red-500/30',
  gradient: 'bg-gradient-to-r from-ai-blue-500/20 to-ai-purple-500/20 text-gradient-primary border-ai-blue-500/20',
};

export const GlassBadge = ({ 
  children, 
  variant = 'secondary', 
  className = '',
  ...props 
}) => {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5',
        'text-xs font-medium font-primary',
        'rounded-lg border',
        'backdrop-blur-premium',
        badgeVariants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};

// ===============================
// 🎨 PREMIUM GLASS TOOLTIP
// ===============================

export const GlassTooltip = ({ 
  children, 
  content, 
  placement = 'top',
  className = '' 
}) => {
  const [isVisible, setIsVisible] = React.useState(false);

  const placementClasses = {
    top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 transform -translate-y-1/2 ml-2',
  };

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className={cn(
            'absolute z-tooltip',
            'glass-thick',
            'rounded-lg px-3 py-1.5',
            'text-sm text-text-primary font-primary',
            'whitespace-nowrap',
            'pointer-events-none',
            placementClasses[placement],
            className
          )}
        >
          {content}
        </motion.div>
      )}
    </div>
  );
};

// ===============================
// 🎨 UTILITY FUNCTIONS
// ===============================



export default GlassCard;