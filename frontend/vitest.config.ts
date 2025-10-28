/**
 * vitest.config.ts - Vitest Testing Configuration
 * 
 * Purpose: Configure unit and integration testing with Vitest
 * 
 * Features:
 * - JSDOM environment for React component testing
 * - Path aliases matching vite.config.ts
 * - Coverage reporting with V8
 * - Global test utilities
 * - Mock browser APIs
 * 
 * Following AGENTS_FRONTEND.md:
 * - Test coverage > 80%
 * - Fast test execution
 * - Isolated test environment
 */

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    // Environment setup
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    
    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockData',
        'dist/',
        '.eslintrc.cjs',
        '**/*.test.{ts,tsx}',
        '**/*.spec.{ts,tsx}',
      ],
      // Target: >80% coverage per AGENTS_FRONTEND.md
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
    
    // Test execution
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', 'dist', '.idea', '.git', '.cache'],
    
    // Performance
    threads: true,
    isolate: true,
    
    // Timeouts
    testTimeout: 10000,
    hookTimeout: 10000,
    
    // Watch mode
    watchExclude: ['**/node_modules/**', '**/dist/**'],
    
    // Reporters
    reporters: ['verbose'],
    
    // Mock configuration
    mockReset: true,
    restoreMocks: true,
    clearMocks: true,
  },
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
});
