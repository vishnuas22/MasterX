import React, { forwardRef, useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '../utils/cn';

// ===============================
// 🍎 AUTHENTIC APPLE DESIGN SYSTEM
// ===============================

// Apple Materials System (iOS/macOS/visionOS)
const appleMaterials = {
  'ultraThin': {
    light: 'bg-white/50 backdrop-blur-[2px] border-black/[0.04]',
    dark: 'bg-white/[0.08] backdrop-blur-[2px] border-white/[0.08]'
  },
  'thin': {
    light: 'bg-white/70 backdrop-blur-[4px] border-black/[0.06]',
    dark: 'bg-white/[0.12] backdrop-blur-[4px] border-white/[0.12]'
  },
  'regular': {
    light: 'bg-white/80 backdrop-blur-[8px] border-black/[0.08]',
    dark: 'bg-white/[0.18] backdrop-blur-[8px] border-white/[0.18]'
  },
  'thick': {
    light: 'bg-white/90 backdrop-blur-[12px] border-black/[0.1]',
    dark: 'bg-white/[0.24] backdrop-blur-[12px] border-white/[0.24]'
  },
  'ultraThick': {
    light: 'bg-white/95 backdrop-blur-[20px] border-black/[0.12]',
    dark: 'bg-white/[0.32] backdrop-blur-[20px] border-white/[0.32]'
  },
  // macOS Tahoe (v26) Beta Materials
  'tahoeCard': {
    light: 'bg-gradient-to-br from-white/85 via-white/80 to-white/75 backdrop-blur-[16px] border-black/[0.08]',
    dark: 'bg-gradient-to-br from-white/[0.22] via-white/[0.18] to-white/[0.15] backdrop-blur-[16px] border-white/[0.15]'
  },
  'tahoeSheet': {
    light: 'bg-white/92 backdrop-blur-[24px] border-black/[0.06]',
    dark: 'bg-white/[0.28] backdrop-blur-[24px] border-white/[0.12]'
  },
  // System Colors
  'systemBlue': {
    light: 'bg-blue-500/10 backdrop-blur-[8px] border-blue-500/20',
    dark: 'bg-blue-400/15 backdrop-blur-[8px] border-blue-400/25'
  },
  'systemGreen': {
    light: 'bg-green-500/10 backdrop-blur-[8px] border-green-500/20',
    dark: 'bg-green-400/15 backdrop-blur-[8px] border-green-400/25'
  },
  'systemOrange': {
    light: 'bg-orange-500/10 backdrop-blur-[8px] border-orange-500/20',
    dark: 'bg-orange-400/15 backdrop-blur-[8px] border-orange-400/25'
  },
  'systemRed': {
    light: 'bg-red-500/10 backdrop-blur-[8px] border-red-500/20',
    dark: 'bg-red-400/15 backdrop-blur-[8px] border-red-400/25'
  },
  'systemPurple': {
    light: 'bg-purple-500/10 backdrop-blur-[8px] border-purple-500/20',
    dark: 'bg-purple-400/15 backdrop-blur-[8px] border-purple-400/25'
  }
};

// Apple Corner Radius System
const appleCornerRadius = {
  'xs': 'rounded-[4px]',    // iOS small elements
  'sm': 'rounded-[8px]',    // iOS buttons
  'md': 'rounded-[12px]',   // iOS cards
  'lg': 'rounded-[16px]',   // iOS large cards
  'xl': 'rounded-[20px]',   // macOS panels
  'xxl': 'rounded-[24px]',  // macOS sheets
  'continuous': 'rounded-[28px]' // Apple's continuous corner radius
};

// Apple Spacing System (8pt grid)
const appleSpacing = {
  'xs': 'p-2',    // 8pt
  'sm': 'p-3',    // 12pt  
  'md': 'p-4',    // 16pt
  'lg': 'p-6',    // 24pt
  'xl': 'p-8',    // 32pt
  'xxl': 'p-12',  // 48pt
  'xxxl': 'p-16'  // 64pt
};

// Apple Shadow System
const appleShadows = {
  'level1': 'shadow-[0_1px_3px_rgba(0,0,0,0.1)]',
  'level2': 'shadow-[0_2px_8px_rgba(0,0,0,0.15)]',
  'level3': 'shadow-[0_4px_16px_rgba(0,0,0,0.2)]',
  'level4': 'shadow-[0_8px_32px_rgba(0,0,0,0.25)]',
  'level5': 'shadow-[0_16px_64px_rgba(0,0,0,0.3)]',
  // macOS Tahoe Enhanced Shadows
  'tahoe': 'shadow-[0_8px_32px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.08)]',
  'tahoeFloating': 'shadow-[0_16px_64px_rgba(0,0,0,0.15),0_4px_16px_rgba(0,0,0,0.1)]'
};

// Apple Spring Physics (iOS/macOS native feel)
const appleSpring = {
  type: "spring",
  damping: 25,
  stiffness: 300,
  mass: 0.8
};

const appleSpringBouncy = {
  type: "spring", 
  damping: 18,
  stiffness: 200,
  mass: 0.6
};

// Apple Easing Curves
const appleEasing = {
  standard: [0.25, 0.1, 0.25, 1],
  decelerate: [0.0, 0.0, 0.2, 1],
  accelerate: [0.4, 0.0, 1, 1],
  sharp: [0.4, 0.0, 0.6, 1]
};

// Dark Mode Detection Hook
const useDarkMode = () => {
  const [isDark, setIsDark] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  React.useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e) => setIsDark(e.matches);
    
    mediaQuery.addListener(handler);
    return () => mediaQuery.removeListener(handler);
  }, []);

  return isDark;
};

// Main Apple Card Component
export const GlassCard = forwardRef(({ 
  children, 
  className = '', 
  material = 'regular',
  cornerRadius = 'md',
  spacing = 'md',
  shadow = 'level2',
  hover = true,
  animated = true,
  haptic = false,
  elevated = false,
  ...props 
}, ref) => {
  const isDark = useDarkMode();
  const [isPressed, setIsPressed] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const materialClasses = appleMaterials[material] 
    ? appleMaterials[material][isDark ? 'dark' : 'light']
    : appleMaterials.regular[isDark ? 'dark' : 'light'];

  const baseClasses = cn(
    'relative',
    'border',
    'transition-all duration-300 ease-out',
    'will-change-transform',
    // Apple Materials
    materialClasses,
    // Apple Corner Radius
    appleCornerRadius[cornerRadius],
    // Apple Spacing
    appleSpacing[spacing],
    // Apple Shadows
    appleShadows[shadow],
    // Elevated state (floating)
    elevated && appleShadows.tahoeFloating,
    // Hover effects
    hover && 'cursor-pointer',
    // Pressed state
    isPressed && 'scale-[0.98] brightness-95',
    className
  );

  const handleMouseEnter = () => {
    setIsHovered(true);
    if (haptic) {
      // Simulate haptic feedback with subtle visual cue
      document.body.style.setProperty('--haptic-strength', '1');
    }
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setIsPressed(false);
    if (haptic) {
      document.body.style.setProperty('--haptic-strength', '0');
    }
  };

  const handleMouseDown = () => {
    setIsPressed(true);
  };

  const handleMouseUp = () => {
    setIsPressed(false);
  };

  if (animated) {
    return (
      <motion.div
        ref={ref}
        className={baseClasses}
        initial={{ opacity: 0, scale: 0.96, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={appleSpring}
        whileHover={hover ? { 
          y: -2, 
          scale: 1.005,
          transition: { duration: 0.15, ease: appleEasing.decelerate }
        } : undefined}
        whileTap={hover ? {
          scale: 0.98,
          transition: { duration: 0.1, ease: appleEasing.sharp }
        } : undefined}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        style={{
          // Apple's signature subtle gradient overlay
          background: isHovered ? 
            `linear-gradient(135deg, rgba(255,255,255,${isDark ? '0.1' : '0.3'}) 0%, rgba(255,255,255,${isDark ? '0.05' : '0.1'}) 100%)` : 
            undefined
        }}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <div 
      ref={ref} 
      className={baseClasses} 
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      {...props}
    >
      {children}
    </div>
  );
});

GlassCard.displayName = 'AppleCard';

// ===============================
// 🍎 APPLE RADIO BUTTON SYSTEM (iOS Style)
// ===============================

export const AppleRadio = ({
  checked = false,
  onChange,
  disabled = false,
  variant = 'default',
  size = 'md',
  label,
  description,
  value,
  name,
  animated = true,
  className = '',
  ...props
}) => {
  const isDark = useDarkMode();
  const [isPressed, setIsPressed] = useState(false);

  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  const variants = {
    default: isDark ? 'border-gray-600' : 'border-gray-300',
    blue: isDark ? 'border-blue-400' : 'border-blue-500',
    green: isDark ? 'border-green-400' : 'border-green-500',
    orange: isDark ? 'border-orange-400' : 'border-orange-500',
    red: isDark ? 'border-red-400' : 'border-red-500'
  };

  const sizeClass = sizes[size];
  const variantClass = variants[variant] || variants.default;

  const handleChange = () => {
    if (disabled) return;
    onChange?.(value);
  };

  const radioElement = (
    <motion.div
      className={cn(
        'relative inline-flex items-center justify-center',
        'border-2 rounded-full transition-all duration-200',
        'focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2',
        sizeClass,
        checked 
          ? (isDark ? 'bg-blue-600 border-blue-600' : 'bg-blue-500 border-blue-500')
          : (isDark ? 'bg-transparent border-gray-600' : 'bg-white border-gray-300'),
        disabled && 'opacity-50 cursor-not-allowed',
        className
      )}
      whileHover={!disabled ? { scale: 1.05 } : {}}
      whileTap={!disabled ? { scale: 0.95 } : {}}
      animate={{ scale: isPressed ? 0.9 : 1 }}
    >
      <input
        type="radio"
        checked={checked}
        onChange={handleChange}
        disabled={disabled}
        value={value}
        name={name}
        className="sr-only"
        onMouseDown={() => setIsPressed(true)}
        onMouseUp={() => setIsPressed(false)}
        onMouseLeave={() => setIsPressed(false)}
        {...props}
      />
      
      {checked && (
        <motion.div
          initial={animated ? { scale: 0, opacity: 0 } : {}}
          animate={{ scale: 1, opacity: 1 }}
          exit={animated ? { scale: 0, opacity: 0 } : {}}
          transition={appleSpring}
          className={cn(
            'w-2 h-2 rounded-full bg-white',
            size === 'sm' && 'w-1.5 h-1.5',
            size === 'lg' && 'w-2.5 h-2.5'
          )}
        />
      )}
    </motion.div>
  );

  if (label || description) {
    return (
      <label className="flex items-start space-x-3 cursor-pointer">
        <div className="flex-shrink-0 pt-0.5">
          {radioElement}
        </div>
        <div className="flex-1 min-w-0">
          {label && (
            <p className={cn(
              'text-sm font-medium',
              isDark ? 'text-gray-100' : 'text-gray-900'
            )}>
              {label}
            </p>
          )}
          {description && (
            <p className={cn(
              'text-sm',
              isDark ? 'text-gray-400' : 'text-gray-500'
            )}>
              {description}
            </p>
          )}
        </div>
      </label>
    );
  }

  return radioElement;
};

// ===============================
// 🍎 APPLE SELECT/DROPDOWN SYSTEM (iOS/macOS Style)
// ===============================

export const AppleSelect = ({
  options = [],
  value,
  onChange,
  placeholder = 'Select an option',
  disabled = false,
  variant = 'default',
  size = 'md',
  searchable = false,
  multiSelect = false,
  animated = true,
  className = '',
  ...props
}) => {
  const isDark = useDarkMode();
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const selectRef = React.useRef(null);

  const sizes = {
    sm: 'px-3 py-1.5 text-sm min-h-[32px]',
    md: 'px-4 py-2 text-base min-h-[44px]',
    lg: 'px-6 py-3 text-lg min-h-[52px]'
  };

  const sizeClass = sizes[size];

  const inputStyleClasses = appleInputStyles[variant] 
    ? appleInputStyles[variant][isDark ? 'dark' : 'light']
    : appleInputStyles.default[isDark ? 'dark' : 'light'];

  const filteredOptions = searchable && searchTerm
    ? options.filter(option => 
        option.label.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : options;

  const selectedOption = options.find(option => option.value === value);
  const selectedOptions = multiSelect 
    ? options.filter(option => value?.includes(option.value))
    : [];

  const handleSelect = (option) => {
    if (multiSelect) {
      const currentValues = value || [];
      const newValues = currentValues.includes(option.value)
        ? currentValues.filter(v => v !== option.value)
        : [...currentValues, option.value];
      onChange?.(newValues);
    } else {
      onChange?.(option.value);
      setIsOpen(false);
    }
  };

  const handleClickOutside = (event) => {
    if (selectRef.current && !selectRef.current.contains(event.target)) {
      setIsOpen(false);
    }
  };

  React.useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const getDisplayValue = () => {
    if (multiSelect && selectedOptions.length > 0) {
      return selectedOptions.length === 1 
        ? selectedOptions[0].label
        : `${selectedOptions.length} selected`;
    }
    return selectedOption?.label || placeholder;
  };

  return (
    <div ref={selectRef} className={cn('relative w-full', className)} {...props}>
      {/* Select Button */}
      <motion.button
        type="button"
        className={cn(
          'relative w-full text-left border cursor-pointer',
          'transition-all duration-200 ease-out',
          'focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:ring-offset-0',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          // Apple Input Styles
          inputStyleClasses,
          // Apple Corner Radius
          appleCornerRadius.md,
          // Size
          sizeClass
        )}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        whileHover={!disabled ? { scale: 1.005 } : {}}
        whileTap={!disabled ? { scale: 0.995 } : {}}
      >
        <span className={cn(
          'block truncate',
          !selectedOption && !selectedOptions.length && (isDark ? 'text-gray-400' : 'text-gray-500')
        )}>
          {getDisplayValue()}
        </span>
        <span className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
          <motion.svg
            className={cn('w-5 h-5', isDark ? 'text-gray-400' : 'text-gray-500')}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </motion.svg>
        </span>
      </motion.button>

      {/* Dropdown Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={appleSpring}
            className={cn(
              'absolute z-50 mt-2 w-full',
              'border rounded-xl shadow-2xl',
              'max-h-60 overflow-auto',
              // Apple Material
              isDark 
                ? 'bg-gray-800/95 border-gray-600 backdrop-blur-xl' 
                : 'bg-white/95 border-gray-200 backdrop-blur-xl'
            )}
          >
            {/* Search Input */}
            {searchable && (
              <div className="p-2 border-b border-gray-200 dark:border-gray-600">
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={cn(
                    'w-full px-3 py-2 text-sm border rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500/20',
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-400' 
                      : 'bg-gray-50 border-gray-200 text-gray-900 placeholder-gray-500'
                  )}
                />
              </div>
            )}

            {/* Options */}
            <div className="py-1">
              {filteredOptions.length === 0 ? (
                <div className={cn(
                  'px-4 py-3 text-sm',
                  isDark ? 'text-gray-400' : 'text-gray-500'
                )}>
                  No options found
                </div>
              ) : (
                filteredOptions.map((option, index) => {
                  const isSelected = multiSelect 
                    ? value?.includes(option.value)
                    : option.value === value;

                  return (
                    <motion.button
                      key={option.value}
                      type="button"
                      className={cn(
                        'relative w-full px-4 py-3 text-left text-sm',
                        'transition-colors duration-150',
                        'focus:outline-none focus:bg-blue-500/10',
                        'flex items-center justify-between',
                        isSelected 
                          ? (isDark ? 'bg-blue-600/20 text-blue-300' : 'bg-blue-50 text-blue-700')
                          : (isDark ? 'text-gray-100 hover:bg-gray-700/50' : 'text-gray-900 hover:bg-gray-50')
                      )}
                      onClick={() => handleSelect(option)}
                      whileHover={{ backgroundColor: isDark ? 'rgba(55, 65, 81, 0.5)' : 'rgba(249, 250, 251, 1)' }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center">
                        {option.icon && <span className="mr-3">{option.icon}</span>}
                        <span className="block truncate">{option.label}</span>
                      </div>
                      {isSelected && (
                        <motion.svg
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="w-5 h-5 text-blue-500"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </motion.svg>
                      )}
                    </motion.button>
                  );
                })
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GlassCard;