/**
 * auth.e2e.ts - E2E Tests for Authentication Flow
 * 
 * Purpose: Test complete authentication user journey
 * 
 * Coverage:
 * - Landing page
 * - Login flow
 * - Signup flow
 * - Protected routes
 * - Logout flow
 * 
 * Following AGENTS_FRONTEND.md:
 * - E2E tests for critical user journeys
 * - Real browser testing
 */

import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Start from the landing page
    await page.goto('/');
  });

  // ============================================================================
  // LANDING PAGE
  // ============================================================================

  test('should display landing page correctly', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/MasterX/);

    // Check main heading
    await expect(page.locator('h1')).toContainText(/MasterX/i);

    // Check CTA buttons
    await expect(page.getByRole('link', { name: /get started/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /sign in/i })).toBeVisible();
  });

  // ============================================================================
  // NAVIGATION
  // ============================================================================

  test('should navigate to login page', async ({ page }) => {
    // Click on sign in button
    await page.getByRole('link', { name: /sign in/i }).click();

    // Wait for navigation
    await page.waitForURL('/login');

    // Check we're on login page
    await expect(page.locator('h2')).toContainText(/sign in/i);
    await expect(page.getByLabel(/email/i)).toBeVisible();
    await expect(page.getByLabel(/password/i)).toBeVisible();
  });

  test('should navigate to signup page', async ({ page }) => {
    // Click on get started button
    await page.getByRole('link', { name: /get started/i }).click();

    // Wait for navigation
    await page.waitForURL('/signup');

    // Check we're on signup page
    await expect(page.locator('h2')).toContainText(/sign up/i);
    await expect(page.getByLabel(/name/i)).toBeVisible();
    await expect(page.getByLabel(/email/i)).toBeVisible();
    await expect(page.getByLabel(/password/i)).toBeVisible();
  });

  // ============================================================================
  // LOGIN FLOW
  // ============================================================================

  test('should successfully login with valid credentials', async ({ page }) => {
    // Navigate to login
    await page.goto('/login');

    // Fill in form
    await page.getByLabel(/email/i).fill('test@example.com');
    await page.getByLabel(/password/i).fill('password123');

    // Submit form
    await page.getByRole('button', { name: /sign in/i }).click();

    // Wait for redirect to app
    await page.waitForURL('/app');

    // Check we're authenticated
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should show error with invalid credentials', async ({ page }) => {
    // Navigate to login
    await page.goto('/login');

    // Fill in form with invalid credentials
    await page.getByLabel(/email/i).fill('invalid@example.com');
    await page.getByLabel(/password/i).fill('wrongpassword');

    // Submit form
    await page.getByRole('button', { name: /sign in/i }).click();

    // Check for error message
    await expect(page.getByText(/invalid credentials/i)).toBeVisible();
  });

  test('should validate email format', async ({ page }) => {
    // Navigate to login
    await page.goto('/login');

    // Fill in invalid email
    await page.getByLabel(/email/i).fill('not-an-email');
    await page.getByLabel(/password/i).fill('password123');

    // Try to submit
    await page.getByRole('button', { name: /sign in/i }).click();

    // Check for validation error
    await expect(page.getByText(/valid email/i)).toBeVisible();
  });

  test('should validate required fields', async ({ page }) => {
    // Navigate to login
    await page.goto('/login');

    // Try to submit without filling fields
    await page.getByRole('button', { name: /sign in/i }).click();

    // Check for validation errors
    await expect(page.getByText(/required/i).first()).toBeVisible();
  });

  // ============================================================================
  // SIGNUP FLOW
  // ============================================================================

  test('should successfully signup with valid data', async ({ page }) => {
    // Navigate to signup
    await page.goto('/signup');

    // Fill in form
    await page.getByLabel(/name/i).fill('Test User');
    await page.getByLabel(/email/i).fill(`test${Date.now()}@example.com`); // Unique email
    await page.getByLabel(/password/i).fill('password123');

    // Submit form
    await page.getByRole('button', { name: /sign up/i }).click();

    // Wait for redirect to onboarding or app
    await page.waitForURL(/\/(onboarding|app)/);

    // Check we're authenticated
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should show error for existing email', async ({ page }) => {
    // Navigate to signup
    await page.goto('/signup');

    // Fill in form with existing email
    await page.getByLabel(/name/i).fill('Test User');
    await page.getByLabel(/email/i).fill('existing@example.com');
    await page.getByLabel(/password/i).fill('password123');

    // Submit form
    await page.getByRole('button', { name: /sign up/i }).click();

    // Check for error message
    await expect(page.getByText(/already exists/i)).toBeVisible();
  });

  // ============================================================================
  // PROTECTED ROUTES
  // ============================================================================

  test('should redirect to login when accessing protected route', async ({ page }) => {
    // Try to access protected route directly
    await page.goto('/app');

    // Should be redirected to login
    await page.waitForURL('/login');
    await expect(page.locator('h2')).toContainText(/sign in/i);
  });

  // ============================================================================
  // LOGOUT FLOW
  // ============================================================================

  test('should successfully logout', async ({ page, context }) => {
    // First login
    await page.goto('/login');
    await page.getByLabel(/email/i).fill('test@example.com');
    await page.getByLabel(/password/i).fill('password123');
    await page.getByRole('button', { name: /sign in/i }).click();
    await page.waitForURL('/app');

    // Open user menu
    await page.locator('[data-testid="user-menu-button"]').click();

    // Click logout
    await page.getByRole('button', { name: /logout/i }).click();

    // Should be redirected to landing
    await page.waitForURL('/');

    // Check we're logged out
    await expect(page.getByRole('link', { name: /sign in/i })).toBeVisible();

    // Check token is cleared
    const cookies = await context.cookies();
    const hasAuthToken = cookies.some(cookie => cookie.name === 'accessToken');
    expect(hasAuthToken).toBe(false);
  });

  // ============================================================================
  // REMEMBER ME
  // ============================================================================

  test('should persist session when remember me is checked', async ({ page, context }) => {
    // Navigate to login
    await page.goto('/login');

    // Fill in form and check remember me
    await page.getByLabel(/email/i).fill('test@example.com');
    await page.getByLabel(/password/i).fill('password123');
    await page.getByLabel(/remember me/i).check();

    // Submit form
    await page.getByRole('button', { name: /sign in/i }).click();
    await page.waitForURL('/app');

    // Close and reopen browser (simulate new session)
    const cookies = await context.cookies();
    await context.clearCookies();
    await context.addCookies(cookies);

    // Reload page
    await page.reload();

    // Should still be authenticated
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  // ============================================================================
  // MOBILE RESPONSIVENESS
  // ============================================================================

  test('should be responsive on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    // Navigate to login
    await page.goto('/login');

    // Check form is visible and usable
    await expect(page.getByLabel(/email/i)).toBeVisible();
    await expect(page.getByLabel(/password/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();

    // Fill and submit should work
    await page.getByLabel(/email/i).fill('test@example.com');
    await page.getByLabel(/password/i).fill('password123');
    await page.getByRole('button', { name: /sign in/i }).click();
  });

  // ============================================================================
  // ACCESSIBILITY
  // ============================================================================

  test('should be keyboard accessible', async ({ page }) => {
    // Navigate to login
    await page.goto('/login');

    // Tab through form
    await page.keyboard.press('Tab'); // Email field
    await page.keyboard.type('test@example.com');
    
    await page.keyboard.press('Tab'); // Password field
    await page.keyboard.type('password123');
    
    await page.keyboard.press('Tab'); // Remember me checkbox
    await page.keyboard.press('Space'); // Check it
    
    await page.keyboard.press('Tab'); // Submit button
    await page.keyboard.press('Enter'); // Submit

    // Should navigate to app
    await page.waitForURL('/app');
  });

  test('should have proper focus indicators', async ({ page }) => {
    // Navigate to login
    await page.goto('/login');

    // Focus email field
    await page.getByLabel(/email/i).focus();

    // Check focus ring is visible
    const emailField = page.getByLabel(/email/i);
    await expect(emailField).toBeFocused();
    
    // Verify focus styling (check for focus ring classes)
    const className = await emailField.getAttribute('class');
    expect(className).toContain('focus:ring');
  });
});
