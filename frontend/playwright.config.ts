/**
 * playwright.config.ts - Playwright E2E Testing Configuration
 * 
 * Purpose: Configure end-to-end testing with Playwright
 * 
 * Features:
 * - Multi-browser testing (Chromium, Firefox, WebKit)
 * - Mobile viewport testing
 * - Screenshot and video on failure
 * - Parallel test execution
 * - CI/CD integration ready
 * 
 * Following AGENTS_FRONTEND.md:
 * - E2E tests for critical user journeys
 * - Cross-browser compatibility
 * - Responsive testing
 */

import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_TEST_BASE_URL || 'http://localhost:3000';

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.e2e\.ts/,
  
  // Timeout configuration
  timeout: 30 * 1000, // 30 seconds per test
  expect: {
    timeout: 5000, // 5 seconds for assertions
  },
  
  // Test execution
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: [
    ['html', { open: 'never' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['list'],
  ],
  
  // Global test setup
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    
    // Navigation
    actionTimeout: 10000,
    navigationTimeout: 30000,
    
    // Viewport
    viewport: { width: 1280, height: 720 },
    
    // Locale
    locale: 'en-US',
    timezoneId: 'America/New_York',
    
    // Color scheme
    colorScheme: 'dark',
  },
  
  // Projects for different browsers and devices
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    
    // Mobile testing
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
    
    // Tablet testing
    {
      name: 'iPad',
      use: { ...devices['iPad Pro'] },
    },
  ],
  
  // Web server configuration
  webServer: {
    command: 'yarn preview',
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
  
  // Output directories
  outputDir: 'test-results',
});
