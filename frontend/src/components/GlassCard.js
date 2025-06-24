import React from 'react';
import { motion } from 'framer-motion';

export function GlassCard({ 
  children, 
  className = '', 
  animate = true, 
  hover = true,
  variant = 'default',
  ...props 
}) {
  const variants = {
    default: `
      backdrop-blur-xl bg-white/5 
      border border-white/10 
      rounded-2xl shadow-2xl 
    `,
    premium: `
      backdrop-blur-2xl bg-gradient-to-br from-white/10 to-white/5
      border border-white/20 
      rounded-2xl shadow-2xl
      before:absolute before:inset-0 before:rounded-2xl 
      before:bg-gradient-to-br before:from-white/10 before:to-transparent 
      before:opacity-50 before:pointer-events-none
      relative overflow-hidden
      after:absolute after:inset-[1px] after:rounded-2xl
      after:bg-gradient-to-br after:from-white/5 after:to-transparent
      after:opacity-60 after:pointer-events-none
    `,
    solid: `
      backdrop-blur-2xl bg-white/10 
      border border-white/20 
      rounded-2xl shadow-2xl 
    `,
    minimal: `
      backdrop-blur-lg bg-white/[0.02] 
      border border-white/5 
      rounded-xl shadow-lg 
    `,
    glow: `
      backdrop-blur-2xl bg-gradient-to-br from-white/15 to-white/5
      border border-white/30 
      rounded-2xl shadow-2xl
      shadow-[0_0_50px_rgba(255,255,255,0.1)]
      before:absolute before:inset-0 before:rounded-2xl 
      before:bg-gradient-to-br before:from-white/20 before:to-transparent 
      before:opacity-40 before:pointer-events-none
      relative overflow-hidden
    `
  };

  const baseClasses = variants[variant] || variants.default;

  const hoverClasses = hover ? `
    hover:bg-white/15 hover:border-white/40 
    hover:shadow-[0_25px_50px_rgba(0,0,0,0.4)] 
    hover:scale-[1.02] 
    hover:shadow-[0_0_60px_rgba(255,255,255,0.15)]
    transition-all duration-500 ease-out
    hover:before:opacity-70
  ` : '';

  const Card = animate ? motion.div : 'div';

  const animationProps = animate ? {
    initial: { opacity: 0, y: 20, scale: 0.95 },
    animate: { opacity: 1, y: 0, scale: 1 },
    transition: { 
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1] // Custom easing for premium feel
    }
  } : {};

  return (
    <Card
      className={`${baseClasses} ${hoverClasses} ${className}`}
      {...animationProps}
      {...props}
    >
      {children}
    </Card>
  );
}

export function GlassButton({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  disabled = false,
  onClick,
  ...props 
}) {
  const baseClasses = `
    backdrop-blur-xl border rounded-xl font-medium
    transition-all duration-300 transform
    focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-transparent
    disabled:opacity-50 disabled:cursor-not-allowed
    active:scale-95 relative overflow-hidden
    before:absolute before:inset-0 before:bg-gradient-to-r 
    before:from-white/10 before:to-transparent before:opacity-0
    hover:before:opacity-100 before:transition-opacity before:duration-300
  `;

  const variants = {
    primary: `
      bg-gradient-to-r from-blue-500/20 to-purple-500/20 
      border-blue-400/30 text-blue-100
      hover:from-blue-500/30 hover:to-purple-500/30 
      hover:border-blue-400/50 hover:shadow-lg hover:shadow-blue-500/25
      focus:ring-blue-400/50
      hover:shadow-[0_0_20px_rgba(59,130,246,0.4)]
    `,
    secondary: `
      bg-white/5 border-white/20 text-gray-100
      hover:bg-white/10 hover:border-white/30
      focus:ring-white/50
      hover:shadow-[0_8px_32px_rgba(255,255,255,0.1)]
    `,
    success: `
      bg-gradient-to-r from-green-500/20 to-emerald-500/20 
      border-green-400/30 text-green-100
      hover:from-green-500/30 hover:to-emerald-500/30 
      hover:border-green-400/50
      focus:ring-green-400/50
      hover:shadow-[0_0_20px_rgba(16,185,129,0.4)]
    `,
    danger: `
      bg-gradient-to-r from-red-500/20 to-pink-500/20 
      border-red-400/30 text-red-100
      hover:from-red-500/30 hover:to-pink-500/30 
      hover:border-red-400/50
      focus:ring-red-400/50
      hover:shadow-[0_0_20px_rgba(239,68,68,0.4)]
    `,
    premium: `
      bg-gradient-to-r from-purple-500/20 via-blue-500/20 to-cyan-500/20 
      border-purple-400/30 text-white
      hover:from-purple-500/30 hover:via-blue-500/30 hover:to-cyan-500/30 
      hover:border-purple-400/50
      focus:ring-purple-400/50
      hover:shadow-[0_0_30px_rgba(147,51,234,0.5)]
    `
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
    xl: 'px-8 py-4 text-xl'
  };

  return (
    <motion.button
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`}
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      transition={{ duration: 0.2 }}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      <span className="relative z-10">{children}</span>
    </motion.button>
  );
}

export function GlassInput({ 
  className = '', 
  error = false,
  variant = 'default',
  ...props 
}) {
  const baseClasses = `
    backdrop-blur-xl border rounded-xl
    px-4 py-3 text-gray-100 placeholder-gray-400
    transition-all duration-300 relative
    focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-transparent
  `;

  const variants = {
    default: `
      bg-white/5 border-white/20 
      hover:border-white/30 hover:bg-white/10
      focus:border-blue-400/50 focus:bg-white/10 focus:ring-blue-400/50
      focus:shadow-[0_0_20px_rgba(59,130,246,0.3)]
    `,
    premium: `
      bg-gradient-to-r from-white/10 to-white/5 border-white/30 
      hover:border-white/40 hover:from-white/15 hover:to-white/10
      focus:border-purple-400/50 focus:from-white/15 focus:to-white/10 focus:ring-purple-400/50
      focus:shadow-[0_0_20px_rgba(147,51,234,0.3)]
    `
  };

  const normalClasses = variants[variant] || variants.default;

  const errorClasses = `
    border-red-400/50 bg-red-500/10
    focus:border-red-400 focus:ring-red-400/50
    focus:shadow-[0_0_20px_rgba(239,68,68,0.3)]
  `;

  return (
    <input
      className={`${baseClasses} ${error ? errorClasses : normalClasses} ${className}`}
      {...props}
    />
  );
}

// New premium components
export function PremiumCard({ children, className = '', ...props }) {
  return (
    <motion.div
      className={`
        relative backdrop-blur-2xl bg-gradient-to-br from-white/10 to-white/5
        border border-white/20 rounded-2xl shadow-2xl overflow-hidden
        before:absolute before:inset-0 before:bg-gradient-to-br 
        before:from-white/10 before:via-transparent before:to-white/5
        before:opacity-50 before:pointer-events-none
        after:absolute after:inset-0 after:bg-gradient-to-t
        after:from-black/10 after:to-transparent after:pointer-events-none
        hover:shadow-[0_20px_60px_rgba(0,0,0,0.4)]
        hover:border-white/30 hover:scale-[1.02]
        transition-all duration-500 ease-out
        ${className}
      `}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      {...props}
    >
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}

export function NeonButton({ children, color = 'blue', className = '', ...props }) {
  const colors = {
    blue: 'from-blue-500 to-cyan-500 shadow-blue-500/50',
    purple: 'from-purple-500 to-pink-500 shadow-purple-500/50',
    green: 'from-green-500 to-emerald-500 shadow-green-500/50',
    orange: 'from-orange-500 to-red-500 shadow-orange-500/50'
  };

  return (
    <motion.button
      className={`
        relative px-6 py-3 rounded-xl font-medium text-white
        bg-gradient-to-r ${colors[color]} border border-white/20
        backdrop-blur-xl overflow-hidden
        before:absolute before:inset-0 before:bg-gradient-to-r 
        before:${colors[color]} before:opacity-0 before:blur-xl
        hover:before:opacity-100 hover:shadow-lg hover:${colors[color]}
        focus:outline-none focus:ring-2 focus:ring-white/50
        transition-all duration-300
        ${className}
      `}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      {...props}
    >
      <span className="relative z-10">{children}</span>
    </motion.button>
  );
}
