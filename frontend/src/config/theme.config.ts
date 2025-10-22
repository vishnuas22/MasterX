/**
 * Theme Configuration
 * 
 * Design tokens matching Tailwind config and Apple HIG 2025
 * All values match tailwind.config.js exactly
 */

// ============================================================================
// COLOR PALETTE
// ============================================================================

export const colors = {
  // Background colors (Dark theme)
  bgPrimary: '#0A0A0A',
  bgSecondary: '#1C1C1E',
  bgTertiary: '#2C2C2E',
  
  // Text colors
  textPrimary: '#FFFFFF',
  textSecondary: '#E5E5E7',
  textTertiary: '#8E8E93',
  
  // Accent colors
  accent: {
    primary: '#0A84FF',
    success: '#30D158',
    warning: '#FF9F0A',
    error: '#FF453A',
    purple: '#BF5AF2',
  },
  
  // Emotion colors
  emotion: {
    joy: '#FFD60A',
    calm: '#64D2FF',
    focus: '#BF5AF2',
    frustration: '#FF453A',
    curiosity: '#30D158',
    confusion: '#FF9F0A',
    excitement: '#FFD60A',
    neutral: '#8E8E93',
  },
} as const;

// ============================================================================
// TYPOGRAPHY
// ============================================================================

export const typography = {
  fontFamily: {
    sans: '-apple-system, BlinkMacSystemFont, SF Pro Display, Inter, sans-serif',
    mono: 'SF Mono, Monaco, Consolas, monospace',
  },
  
  fontSize: {
    xs: 'clamp(0.75rem, 0.7rem + 0.2vw, 0.875rem)',
    sm: 'clamp(0.875rem, 0.8rem + 0.3vw, 1rem)',
    base: 'clamp(1rem, 0.9rem + 0.4vw, 1.125rem)',
    lg: 'clamp(1.125rem, 1rem + 0.5vw, 1.25rem)',
    xl: 'clamp(1.25rem, 1.1rem + 0.6vw, 1.5rem)',
    '2xl': 'clamp(1.5rem, 1.3rem + 0.8vw, 2rem)',
    '3xl': 'clamp(2rem, 1.7rem + 1.2vw, 3rem)',
  },
  
  fontWeight: {
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  
  lineHeight: {
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.75,
  },
} as const;

// ============================================================================
// SPACING SCALE (8-point grid)
// ============================================================================

export const spacing = {
  0: '0',
  1: '0.25rem',  // 4px
  2: '0.5rem',   // 8px
  3: '0.75rem',  // 12px
  4: '1rem',     // 16px
  5: '1.5rem',   // 24px
  6: '2rem',     // 32px
  8: '3rem',     // 48px
  10: '4rem',    // 64px
  12: '6rem',    // 96px
} as const;

// ============================================================================
// BORDER RADIUS
// ============================================================================

export const borderRadius = {
  sm: '8px',
  md: '12px',
  lg: '16px',
  xl: '24px',
  full: '9999px',
} as const;

// ============================================================================
// SHADOWS
// ============================================================================

export const shadows = {
  sm: '0 1px 2px rgba(0, 0, 0, 0.2)',
  md: '0 4px 8px rgba(0, 0, 0, 0.3)',
  lg: '0 8px 16px rgba(0, 0, 0, 0.4)',
  xl: '0 16px 32px rgba(0, 0, 0, 0.5)',
  glass: '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
} as const;

// ============================================================================
// TRANSITIONS
// ============================================================================

export const transitions = {
  duration: {
    fast: '150ms',
    normal: '250ms',
    slow: '400ms',
  },
  
  timing: {
    linear: 'linear',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
} as const;

// ============================================================================
// BREAKPOINTS
// ============================================================================

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const;

// ============================================================================
// Z-INDEX SCALE
// ============================================================================

export const zIndex = {
  dropdown: 1000,
  sticky: 1020,
  fixed: 1030,
  modalBackdrop: 1040,
  modal: 1050,
  popover: 1060,
  tooltip: 1070,
} as const;

// ============================================================================
// ANIMATION PRESETS
// ============================================================================

export const animations = {
  fadeIn: 'fadeIn 250ms ease-out',
  slideUp: 'slideUp 400ms cubic-bezier(0.4, 0, 0.2, 1)',
  pulseSubtle: 'pulseSubtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
  spin: 'spin 1s linear infinite',
} as const;

// ============================================================================
// GLASS MORPHISM
// ============================================================================

export const glassMorphism = {
  background: 'rgba(28, 28, 30, 0.7)',
  backdropFilter: 'blur(40px)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type ColorKey = keyof typeof colors;
export type SpacingKey = keyof typeof spacing;
export type RadiusKey = keyof typeof borderRadius;
export type ShadowKey = keyof typeof shadows;
export type BreakpointKey = keyof typeof breakpoints;
export type ZIndexKey = keyof typeof zIndex;

// ============================================================================
// THEME OBJECT (Combines all tokens)
// ============================================================================

export const theme = {
  colors,
  typography,
  spacing,
  borderRadius,
  shadows,
  transitions,
  breakpoints,
  zIndex,
  animations,
  glassMorphism,
} as const;

export default theme;
