/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-900': '#111827',
        'dark-800': '#16181c',
        'dark-700': '#23272f',
        'neon-100': '#e0e7ff',
        'neon-200': '#a5b4fc',
        'neon-300': '#818cf8',
        'neon-400': '#6366f1',
        'neon-500': '#7f1dff',
        'neon-700': '#312e81',
        'neon-800': '#1e1b4b',
      },
      boxShadow: {
        'neon': '0 0 16px 2px #6366f1, 0 0 2px 0 #7f1dff',
      },
      backdropBlur: {
        glass: '18px',
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}