import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@store': path.resolve(__dirname, './src/store'),
      '@services': path.resolve(__dirname, './src/services'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@types': path.resolve(__dirname, './src/types'),
      '@config': path.resolve(__dirname, './src/config'),
    },
  },
  build: {
    target: 'es2022',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui-vendor': ['framer-motion', 'lucide-react'],
          'data-vendor': ['axios', '@tanstack/react-query', 'zustand'],
          'chart-vendor': ['recharts'],
        },
      },
    },
    chunkSizeWarningLimit: 500,
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      '4cc3583d-34e4-45c4-8925-cb018c0d93f1.preview.emergentagent.com',
      'localhost',
      '.preview.emergentagent.com'
    ],
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'zustand', 'axios'],
  },
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },
});