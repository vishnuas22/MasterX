/**
 * setup.ts - Test Environment Setup
 * 
 * Purpose: Configure global test utilities and mocks
 * 
 * Features:
 * - DOM matchers from @testing-library/jest-dom
 * - Mock browser APIs (IntersectionObserver, ResizeObserver, matchMedia)
 * - Global test utilities
 * - Environment cleanup
 * 
 * Following AGENTS_FRONTEND.md:
 * - Comprehensive test environment
 * - Isolated test execution
 * - Browser API mocking
 */

import '@testing-library/jest-dom';
import { expect, afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// ============================================================================
// CLEANUP
// ============================================================================

/**
 * Cleanup after each test
 * Prevents memory leaks and test pollution
 */
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});

// ============================================================================
// BROWSER API MOCKS
// ============================================================================

/**
 * Mock IntersectionObserver
 * Used by: useIntersection hook, lazy loading
 */
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn(() => []),
}));

/**
 * Mock ResizeObserver
 * Used by: responsive components, charts
 */
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

/**
 * Mock matchMedia
 * Used by: responsive design, theme detection
 */
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

/**
 * Mock window.scrollTo
 * Used by: scroll behavior, navigation
 */
Object.defineProperty(window, 'scrollTo', {
  writable: true,
  value: vi.fn(),
});

/**
 * Mock HTMLMediaElement methods
 * Used by: voice interaction, audio playback
 */
Object.defineProperty(HTMLMediaElement.prototype, 'play', {
  writable: true,
  value: vi.fn().mockResolvedValue(undefined),
});

Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
  writable: true,
  value: vi.fn(),
});

/**
 * Mock navigator.mediaDevices
 * Used by: voice interaction, microphone access
 */
Object.defineProperty(navigator, 'mediaDevices', {
  writable: true,
  value: {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [
        {
          stop: vi.fn(),
          getSettings: () => ({ deviceId: 'test-device' }),
        },
      ],
    }),
    enumerateDevices: vi.fn().mockResolvedValue([]),
  },
});

/**
 * Mock Web Speech API
 * Used by: voice interaction
 */
Object.defineProperty(window, 'SpeechRecognition', {
  writable: true,
  value: vi.fn().mockImplementation(() => ({
    start: vi.fn(),
    stop: vi.fn(),
    abort: vi.fn(),
    onresult: null,
    onerror: null,
    onend: null,
  })),
});

/**
 * Mock clipboard API
 * Used by: copy functionality
 */
Object.defineProperty(navigator, 'clipboard', {
  writable: true,
  value: {
    writeText: vi.fn().mockResolvedValue(undefined),
    readText: vi.fn().mockResolvedValue(''),
  },
});

// ============================================================================
// ENVIRONMENT VARIABLES
// ============================================================================

/**
 * Mock environment variables for testing
 */
process.env.VITE_BACKEND_URL = 'http://localhost:8001/api';
process.env.VITE_APP_NAME = 'MasterX';
process.env.VITE_APP_VERSION = '1.0.0';
process.env.VITE_ENABLE_VOICE = 'true';
process.env.VITE_ENABLE_ANALYTICS = 'true';
process.env.VITE_ENABLE_GAMIFICATION = 'true';

// ============================================================================
// CUSTOM MATCHERS
// ============================================================================

/**
 * Extend Vitest's expect with custom matchers
 * Can add domain-specific assertions here
 */
expect.extend({
  // Example custom matcher
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () =>
          `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// ============================================================================
// CONSOLE SUPPRESSION (Optional)
// ============================================================================

/**
 * Suppress console errors/warnings in tests
 * Comment out during debugging
 */
// const originalError = console.error;
// const originalWarn = console.warn;

// beforeAll(() => {
//   console.error = (...args: any[]) => {
//     if (
//       typeof args[0] === 'string' &&
//       (args[0].includes('Warning: ReactDOM.render') ||
//        args[0].includes('Not implemented: HTMLFormElement.prototype.submit'))
//     ) {
//       return;
//     }
//     originalError.call(console, ...args);
//   };
//   console.warn = (...args: any[]) => {
//     if (typeof args[0] === 'string' && args[0].includes('deprecated')) {
//       return;
//     }
//     originalWarn.call(console, ...args);
//   };
// });

// afterAll(() => {
//   console.error = originalError;
//   console.warn = originalWarn;
// });
