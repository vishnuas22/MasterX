/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0A0A0A',
        'bg-secondary': '#1C1C1E',
        'bg-tertiary': '#2C2C2E',
        'text-primary': '#FFFFFF',
        'text-secondary': '#E5E5E7',
        'text-tertiary': '#8E8E93',
        'accent': {
          primary: '#0A84FF',
          success: '#30D158',
          warning: '#FF9F0A',
          error: '#FF453A',
          purple: '#BF5AF2',
        },
        'emotion': {
          joy: '#FFD60A',
          calm: '#64D2FF',
          focus: '#BF5AF2',
          frustration: '#FF453A',
          curiosity: '#30D158',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'Inter', 'sans-serif'],
      },
      borderRadius: {
        'sm': '8px',
        'md': '12px',
        'lg': '16px',
        'xl': '24px',
      },
      boxShadow: {
        'sm': '0 1px 2px rgba(0, 0, 0, 0.2)',
        'md': '0 4px 8px rgba(0, 0, 0, 0.3)',
        'lg': '0 8px 16px rgba(0, 0, 0, 0.4)',
        'xl': '0 16px 32px rgba(0, 0, 0, 0.5)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
      },
      transitionTimingFunction: {
        'apple': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'spring': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
      },
      animation: {
        'fade-in': 'fadeIn 250ms ease-out',
        'slide-up': 'slideUp 400ms cubic-bezier(0.4, 0, 0.2, 1)',
        'pulse-subtle': 'pulseSubtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        slideUp: {
          from: { transform: 'translateY(10px)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}