/**
 * testUtils.tsx - Testing Utilities & Helpers
 * 
 * Purpose: Provide reusable test utilities for component testing
 * 
 * Features:
 * - Custom render with providers (Router, Query, Store)
 * - Mock data generators
 * - Test helpers for common scenarios
 * - Type-safe testing utilities
 * 
 * Following AGENTS_FRONTEND.md:
 * - DRY principle (Don't Repeat Yourself)
 * - Type safety
 * - Comprehensive provider setup
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import userEvent from '@testing-library/user-event';

// ============================================================================
// CUSTOM RENDER
// ============================================================================

/**
 * Custom render function with all providers
 * 
 * Wraps component with:
 * - React Router (for navigation)
 * - React Query (for server state)
 * - Any other global providers
 * 
 * Usage:
 * ```typescript
 * const { getByText } = renderWithProviders(<MyComponent />);
 * ```
 */
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialRoute?: string;
  queryClient?: QueryClient;
}

export function renderWithProviders(
  ui: ReactElement,
  {
    initialRoute = '/',
    queryClient = createTestQueryClient(),
    ...renderOptions
  }: CustomRenderOptions = {}
) {
  // Set initial route
  window.history.pushState({}, 'Test page', initialRoute);

  const Wrapper = ({ children }: { children: React.ReactNode }) => {
    return (
      <BrowserRouter>
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      </BrowserRouter>
    );
  };

  return {
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
    user: userEvent.setup(),
  };
}

/**
 * Create a test QueryClient with sensible defaults
 * Disables retries and logging for tests
 */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity,
      },
      mutations: {
        retry: false,
      },
    },
    logger: {
      log: () => {},
      warn: () => {},
      error: () => {},
    },
  });
}

// ============================================================================
// MOCK DATA GENERATORS
// ============================================================================

/**
 * Generate mock user data
 */
export const createMockUser = (overrides = {}) => ({
  id: 'test-user-1',
  email: 'test@example.com',
  name: 'Test User',
  avatar: null,
  learningGoals: ['Math', 'Science'],
  preferences: {
    theme: 'dark',
    difficulty: 'intermediate',
    voiceEnabled: false,
  },
  ...overrides,
});

/**
 * Generate mock message data
 */
export const createMockMessage = (overrides = {}) => ({
  id: 'msg-1',
  content: 'Test message',
  role: 'user' as const,
  timestamp: new Date().toISOString(),
  emotion: null,
  ...overrides,
});

/**
 * Generate mock chat session
 */
export const createMockChatSession = (overrides = {}) => ({
  sessionId: 'session-1',
  userId: 'test-user-1',
  messages: [
    createMockMessage({ id: 'msg-1', role: 'user', content: 'Hello' }),
    createMockMessage({ id: 'msg-2', role: 'assistant', content: 'Hi there!' }),
  ],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  ...overrides,
});

/**
 * Generate mock emotion data
 */
export const createMockEmotion = (overrides = {}) => ({
  primary: 'joy',
  confidence: 0.85,
  pad: {
    pleasure: 0.7,
    arousal: 0.5,
    dominance: 0.6,
  },
  categories: {
    joy: 0.85,
    excitement: 0.6,
    neutral: 0.2,
  },
  learningReadiness: 0.8,
  cognitiveLoad: 0.4,
  flowState: 0.7,
  timestamp: new Date().toISOString(),
  ...overrides,
});

/**
 * Generate mock analytics data
 */
export const createMockAnalytics = (overrides = {}) => ({
  totalSessions: 10,
  totalMessages: 150,
  averageSessionLength: 25,
  topicMastery: {
    math: 0.75,
    science: 0.6,
    history: 0.8,
  },
  emotionTrends: {
    joy: 0.4,
    excitement: 0.3,
    neutral: 0.2,
    confusion: 0.1,
  },
  learningVelocity: 0.7,
  ...overrides,
});

/**
 * Generate mock gamification data
 */
export const createMockGamification = (overrides = {}) => ({
  points: 1500,
  level: 5,
  streak: 7,
  achievements: [
    {
      id: 'first-lesson',
      name: 'First Steps',
      description: 'Complete your first lesson',
      icon: '🎓',
      unlockedAt: new Date().toISOString(),
    },
  ],
  badges: [],
  ...overrides,
});

// ============================================================================
// TEST HELPERS
// ============================================================================

/**
 * Wait for loading states to complete
 */
export const waitForLoadingToFinish = () => {
  return new Promise((resolve) => setTimeout(resolve, 0));
};

/**
 * Simulate typing in an input field
 */
export const typeIntoInput = async (
  user: ReturnType<typeof userEvent.setup>,
  input: HTMLElement,
  text: string
) => {
  await user.clear(input);
  await user.type(input, text);
};

/**
 * Mock localStorage
 */
export const mockLocalStorage = () => {
  const store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      Object.keys(store).forEach((key) => delete store[key]);
    },
  };
};

/**
 * Mock sessionStorage
 */
export const mockSessionStorage = () => mockLocalStorage();

/**
 * Create mock fetch response
 */
export const createMockResponse = <T,>(data: T, ok = true, status = 200) => {
  return {
    ok,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
    headers: new Headers(),
    redirected: false,
    statusText: ok ? 'OK' : 'Error',
    type: 'basic' as ResponseType,
    url: '',
    clone: () => createMockResponse(data, ok, status),
    body: null,
    bodyUsed: false,
    arrayBuffer: async () => new ArrayBuffer(0),
    blob: async () => new Blob(),
    formData: async () => new FormData(),
  };
};

/**
 * Wait for specific condition to be true
 */
export const waitForCondition = async (
  condition: () => boolean,
  timeout = 1000,
  interval = 50
): Promise<void> => {
  const startTime = Date.now();

  while (!condition()) {
    if (Date.now() - startTime > timeout) {
      throw new Error('Timeout waiting for condition');
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
};

/**
 * Suppress console errors for specific test
 */
export const suppressConsoleErrors = () => {
  const originalError = console.error;
  beforeEach(() => {
    console.error = vi.fn();
  });
  afterEach(() => {
    console.error = originalError;
  });
};

// Re-export everything from @testing-library/react
export * from '@testing-library/react';
export { userEvent };
