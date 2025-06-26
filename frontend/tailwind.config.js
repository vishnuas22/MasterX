/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      // Premium Font System
      fontFamily: {
        'primary': ['SF Pro Display', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'system-ui', 'sans-serif'],
        'mono': ['SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', 'monospace'],
        'inter': ['Inter', 'sans-serif'],
        'sf-pro': ['SF Pro Display', 'sans-serif'],
      },
      
      // Enhanced Color System
      colors: {
        // Core grays with better names
        'glass': {
          50: 'rgba(255, 255, 255, 0.03)',
          100: 'rgba(255, 255, 255, 0.05)',
          200: 'rgba(255, 255, 255, 0.08)',
          300: 'rgba(255, 255, 255, 0.12)',
          400: 'rgba(255, 255, 255, 0.15)',
          500: 'rgba(255, 255, 255, 0.18)',
        },
        
        // AI Brand Colors
        'ai': {
          'blue': {
            50: 'rgb(240, 249, 255)',
            100: 'rgb(224, 242, 254)',
            200: 'rgb(186, 230, 253)',
            300: 'rgb(125, 211, 252)',
            400: 'rgb(56, 189, 248)',
            500: 'rgb(0, 122, 255)',  // Primary iOS Blue
            600: 'rgb(2, 132, 199)',
            700: 'rgb(3, 105, 161)',
            800: 'rgb(7, 89, 133)',
            900: 'rgb(12, 74, 110)',
          },
          'purple': {
            50: 'rgb(250, 245, 255)',
            100: 'rgb(243, 232, 255)',
            200: 'rgb(233, 213, 255)',
            300: 'rgb(216, 180, 254)',
            400: 'rgb(196, 144, 253)',
            500: 'rgb(175, 82, 222)',   // AI Purple
            600: 'rgb(147, 51, 234)',
            700: 'rgb(126, 34, 206)',
            800: 'rgb(107, 33, 168)',
            900: 'rgb(88, 28, 135)',
          },
          'green': {
            50: 'rgb(240, 253, 244)',
            100: 'rgb(220, 252, 231)',
            200: 'rgb(187, 247, 208)',
            300: 'rgb(134, 239, 172)',
            400: 'rgb(74, 222, 128)',
            500: 'rgb(50, 215, 75)',    // Success Green
            600: 'rgb(34, 197, 94)',
            700: 'rgb(21, 128, 61)',
            800: 'rgb(22, 101, 52)',
            900: 'rgb(20, 83, 45)',
          },
          'red': {
            50: 'rgb(254, 242, 242)',
            100: 'rgb(254, 226, 226)',
            200: 'rgb(254, 202, 202)',
            300: 'rgb(252, 165, 165)',
            400: 'rgb(248, 113, 113)',
            500: 'rgb(255, 69, 58)',     // Error Red
            600: 'rgb(220, 38, 38)',
            700: 'rgb(185, 28, 28)',
            800: 'rgb(153, 27, 27)',
            900: 'rgb(127, 29, 29)',
          },
        },
        
        // Background System
        'bg': {
          'primary': 'rgb(8, 8, 12)',
          'secondary': 'rgb(15, 15, 20)',
          'tertiary': 'rgb(20, 20, 28)',
          'elevated': 'rgb(25, 25, 35)',
        },
        
        // Text Hierarchy
        'text': {
          'primary': 'rgb(255, 255, 255)',
          'secondary': 'rgb(235, 235, 245)',
          'tertiary': 'rgb(174, 174, 178)',
          'quaternary': 'rgb(134, 134, 138)',
        },
        
        // Border System
        'border': {
          'subtle': 'rgba(255, 255, 255, 0.08)',
          'medium': 'rgba(255, 255, 255, 0.12)',
          'strong': 'rgba(255, 255, 255, 0.18)',
        },
      },
      
      // Enhanced Spacing (8pt grid)
      spacing: {
        '1': '0.25rem',    // 4px
        '2': '0.5rem',     // 8px
        '3': '0.75rem',    // 12px
        '4': '1rem',       // 16px
        '5': '1.25rem',    // 20px
        '6': '1.5rem',     // 24px
        '7': '1.75rem',    // 28px
        '8': '2rem',       // 32px
        '9': '2.25rem',    // 36px
        '10': '2.5rem',    // 40px
        '11': '2.75rem',   // 44px
        '12': '3rem',      // 48px
        '14': '3.5rem',    // 56px
        '16': '4rem',      // 64px
        '18': '4.5rem',    // 72px
        '20': '5rem',      // 80px
        '24': '6rem',      // 96px
        '28': '7rem',      // 112px
        '32': '8rem',      // 128px
        '36': '9rem',      // 144px
        '40': '10rem',     // 160px
        '44': '11rem',     // 176px
        '48': '12rem',     // 192px
        '52': '13rem',     // 208px
        '56': '14rem',     // 224px
        '60': '15rem',     // 240px
        '64': '16rem',     // 256px
        '72': '18rem',     // 288px
        '80': '20rem',     // 320px
        '96': '24rem',     // 384px
      },
      
      // Enhanced Border Radius
      borderRadius: {
        'none': '0',
        'xs': '0.25rem',   // 4px
        'sm': '0.375rem',  // 6px
        'DEFAULT': '0.5rem', // 8px
        'md': '0.5rem',    // 8px
        'lg': '0.75rem',   // 12px
        'xl': '1rem',      // 16px
        '2xl': '1.5rem',   // 24px
        '3xl': '2rem',     // 32px
        '4xl': '2.5rem',   // 40px
        'full': '9999px',
      },
      
      // Enhanced Typography Scale
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],      // 12px
        'sm': ['0.8125rem', { lineHeight: '1.25rem' }], // 13px
        'base': ['0.9375rem', { lineHeight: '1.5rem' }], // 15px
        'lg': ['1.0625rem', { lineHeight: '1.75rem' }],  // 17px (Apple's preferred)
        'xl': ['1.125rem', { lineHeight: '1.75rem' }],   // 18px
        '2xl': ['1.375rem', { lineHeight: '2rem' }],     // 22px
        '3xl': ['1.75rem', { lineHeight: '2.25rem' }],   // 28px
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],    // 36px
        '5xl': ['3rem', { lineHeight: '3.5rem' }],       // 48px
        '6xl': ['4rem', { lineHeight: '4.5rem' }],       // 64px
        
        // Apple-inspired naming
        'caption': ['0.75rem', { lineHeight: '1rem' }],     // 12px
        'footnote': ['0.8125rem', { lineHeight: '1.25rem' }], // 13px
        'body': ['0.9375rem', { lineHeight: '1.5rem' }],     // 15px
        'body-large': ['1.0625rem', { lineHeight: '1.75rem' }], // 17px
        'title': ['1.125rem', { lineHeight: '1.75rem' }],    // 18px
        'title-large': ['1.375rem', { lineHeight: '2rem' }], // 22px
        'headline': ['1.75rem', { lineHeight: '2.25rem' }],  // 28px
        'headline-large': ['2.25rem', { lineHeight: '2.5rem' }], // 36px
        'display': ['3rem', { lineHeight: '3.5rem' }],       // 48px
        'display-large': ['4rem', { lineHeight: '4.5rem' }], // 64px
      },
      
      // Enhanced Font Weights
      fontWeight: {
        'thin': '100',
        'light': '300',
        'normal': '400',
        'medium': '500',
        'semibold': '600',
        'bold': '700',
        'heavy': '800',
        'black': '900',
      },
      
      // Letter Spacing
      letterSpacing: {
        'tighter': '-0.05em',
        'tight': '-0.025em',
        'normal': '0',
        'wide': '0.025em',
        'wider': '0.05em',
        'widest': '0.1em',
      },
      
      // Enhanced Box Shadow System
      boxShadow: {
        'xs': '0 1px 2px rgba(0, 0, 0, 0.12)',
        'sm': '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
        'DEFAULT': '0 4px 6px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
        'md': '0 4px 6px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
        'lg': '0 10px 15px rgba(0, 0, 0, 0.12), 0 4px 6px rgba(0, 0, 0, 0.08)',
        'xl': '0 20px 25px rgba(0, 0, 0, 0.15), 0 10px 10px rgba(0, 0, 0, 0.04)',
        '2xl': '0 25px 50px rgba(0, 0, 0, 0.25)',
        'inner': 'inset 0 2px 4px rgba(0, 0, 0, 0.06)',
        'none': 'none',
        
        // AI-specific glows
        'glow-blue': '0 0 20px rgba(0, 122, 255, 0.3)',
        'glow-purple': '0 0 20px rgba(175, 82, 222, 0.3)',
        'glow-green': '0 0 20px rgba(50, 215, 75, 0.3)',
        'glow-red': '0 0 20px rgba(255, 69, 58, 0.3)',
        
        // Glass shadows
        'glass-sm': '0 4px 6px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        'glass-md': '0 10px 15px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        'glass-lg': '0 20px 25px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
      },
      
      // Enhanced Animation System
      animation: {
        'fade-in-up': 'fadeInUp 0.6s cubic-bezier(0, 0, 0.2, 1)',
        'fade-in-down': 'fadeInDown 0.6s cubic-bezier(0, 0, 0.2, 1)',
        'slide-in-left': 'slideInLeft 0.6s cubic-bezier(0, 0, 0.2, 1)',
        'slide-in-right': 'slideInRight 0.6s cubic-bezier(0, 0, 0.2, 1)',
        'scale-in': 'scaleIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
        'float': 'float 3s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s infinite',
        'gradient-shift': 'gradientShift 8s ease infinite',
        'pulse-soft': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      
      // Enhanced Timing Functions
      transitionTimingFunction: {
        'apple': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'spring': 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
        'bounce': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      },
      
      // Backdrop Blur
      backdropBlur: {
        'xs': '2px',
        'sm': '4px',
        'DEFAULT': '8px',
        'md': '12px',
        'lg': '16px',
        'xl': '24px',
        '2xl': '40px',
        '3xl': '64px',
        'premium': '20px',
      },
      
      // Backdrop Saturate
      backdropSaturate: {
        '150': '1.5',
        '160': '1.6',
        '170': '1.7',
        '180': '1.8',
        '200': '2',
      },
      
      // Z-Index Scale
      zIndex: {
        'behind': '-1',
        'auto': 'auto',
        '0': '0',
        '10': '10',
        '20': '20',
        '30': '30',
        '40': '40',
        '50': '50',
        'modal': '100',
        'popover': '200',
        'tooltip': '300',
        'toast': '400',
        'max': '9999',
      },
    },
  },
  plugins: [
    // Custom utilities
    function({ addUtilities, theme }) {
      const newUtilities = {
        // Glass effect utilities
        '.glass-ultra-thin': {
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(10px) saturate(150%)',
        },
        '.glass-thin': {
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(15px) saturate(160%)',
        },
        '.glass-medium': {
          background: 'rgba(255, 255, 255, 0.08)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          backdropFilter: 'blur(20px) saturate(170%)',
          boxShadow: '0 10px 15px rgba(0, 0, 0, 0.12), 0 4px 6px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        },
        '.glass-thick': {
          background: 'rgba(255, 255, 255, 0.12)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          backdropFilter: 'blur(25px) saturate(180%)',
          boxShadow: '0 20px 25px rgba(0, 0, 0, 0.15), 0 10px 10px rgba(0, 0, 0, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
        },
        '.glass-ultra-thick': {
          background: 'rgba(255, 255, 255, 0.15)',
          border: '1px solid rgba(255, 255, 255, 0.18)',
          backdropFilter: 'blur(30px) saturate(200%)',
          boxShadow: '0 25px 50px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
        },
        
        // Text gradient utilities
        '.text-gradient-primary': {
          background: 'linear-gradient(135deg, rgb(0, 122, 255), rgb(175, 82, 222))',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
        '.text-gradient-secondary': {
          background: 'linear-gradient(135deg, rgb(175, 82, 222), rgb(50, 215, 75))',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
        
        // Hover effects
        '.hover-lift': {
          transition: 'all 0.4s cubic-bezier(0, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-4px) scale(1.01)',
            boxShadow: '0 25px 50px rgba(0, 0, 0, 0.25)',
          },
        },
        '.hover-glow': {
          transition: 'all 0.3s cubic-bezier(0, 0, 0.2, 1)',
          '&:hover': {
            boxShadow: '0 0 20px rgba(0, 122, 255, 0.3), 0 20px 25px rgba(0, 0, 0, 0.15)',
          },
        },
        '.hover-scale': {
          transition: 'transform 0.2s cubic-bezier(0, 0, 0.2, 1)',
          '&:hover': {
            transform: 'scale(1.05)',
          },
        },
      }
      
      addUtilities(newUtilities)
    }
  ],
}