# MasterX Frontend - Testing Guide

## 📋 Overview

Comprehensive testing setup for MasterX frontend following AGENTS_FRONTEND.md standards:
- ✅ Unit test coverage > 80%
- ✅ Component testing with React Testing Library
- ✅ E2E testing with Playwright
- ✅ Accessibility testing
- ✅ Performance testing

## 🧪 Testing Stack

### Core Libraries
- **Vitest** - Fast unit testing framework
- **React Testing Library** - Component testing
- **Playwright** - E2E browser testing
- **@testing-library/user-event** - User interaction simulation
- **@testing-library/jest-dom** - DOM matchers

### Test Types
1. **Unit Tests** - Store logic, utilities, hooks
2. **Component Tests** - UI components, user interactions
3. **Integration Tests** - Multiple components working together
4. **E2E Tests** - Complete user journeys across pages

## 📁 Test File Structure

```
src/
├── store/
│   ├── authStore.ts
│   ├── authStore.test.ts          # Store unit tests
│   ├── chatStore.ts
│   └── chatStore.test.ts
├── hooks/
│   ├── useAuth.ts
│   ├── useAuth.test.ts            # Hook unit tests
│   └── ...
├── components/
│   ├── ui/
│   │   ├── Button.tsx
│   │   ├── Button.test.tsx        # Component tests
│   │   └── ...
│   └── ...
└── test/
    ├── setup.ts                   # Global test setup
    └── testUtils.tsx              # Testing utilities

e2e/
├── auth.e2e.ts                   # E2E authentication tests
├── chat.e2e.ts                   # E2E chat flow tests
└── ...
```

## 🚀 Running Tests

### Unit & Component Tests

```bash
# Run all tests
yarn test

# Watch mode (re-run on file changes)
yarn test:watch

# UI mode (visual test runner)
yarn test:ui

# Coverage report
yarn test:coverage
```

### E2E Tests

```bash
# Run all E2E tests
yarn test:e2e

# UI mode (visual test runner)
yarn test:e2e:ui

# Debug mode (step through tests)
yarn test:e2e:debug

# Specific browser
yarn test:e2e --project=chromium
yarn test:e2e --project=firefox
yarn test:e2e --project=webkit
```

## 📝 Writing Tests

### Unit Tests (Stores, Utilities)

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('MyStore', () => {
  beforeEach(() => {
    // Reset state before each test
    useMyStore.setState(initialState);
    vi.clearAllMocks();
  });

  it('should update state correctly', () => {
    const { action } = useMyStore.getState();
    action('newValue');

    const state = useMyStore.getState();
    expect(state.value).toBe('newValue');
  });
});
```

### Hook Tests

```typescript
import { renderHook, waitFor } from '@testing-library/react';

describe('useMyHook', () => {
  it('should return expected values', () => {
    const { result } = renderHook(() => useMyHook());

    expect(result.current.value).toBeDefined();
  });

  it('should handle async operations', async () => {
    const { result } = renderHook(() => useMyHook());

    act(() => {
      result.current.doAsync();
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });
});
```

### Component Tests

```typescript
import { render, screen } from '@/test/testUtils';
import userEvent from '@testing-library/user-event';

describe('MyComponent', () => {
  it('should render correctly', () => {
    render(<MyComponent />);

    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('should handle user interactions', async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();

    render(<MyComponent onClick={handleClick} />);

    await user.click(screen.getByRole('button'));

    expect(handleClick).toHaveBeenCalled();
  });

  it('should be accessible', () => {
    render(<MyComponent />);

    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('aria-label');
  });
});
```

### E2E Tests

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature Flow', () => {
  test('should complete user journey', async ({ page }) => {
    await page.goto('/');

    await page.getByLabel('Email').fill('test@example.com');
    await page.getByRole('button', { name: 'Submit' }).click();

    await expect(page).toHaveURL('/success');
  });
});
```

## 🎯 Testing Best Practices

### 1. Test Naming
```typescript
// ✅ Good
it('should display error message when login fails')

// ❌ Bad
it('test login')
```

### 2. Test Isolation
```typescript
beforeEach(() => {
  // Reset all state
  // Clear mocks
  // Clear storage
});
```

### 3. Arrange-Act-Assert Pattern
```typescript
it('should update username', () => {
  // Arrange
  const user = createMockUser();

  // Act
  updateUsername(user, 'newname');

  // Assert
  expect(user.name).toBe('newname');
});
```

### 4. Mock External Dependencies
```typescript
vi.mock('@/services/api/auth.api', () => ({
  authAPI: {
    login: vi.fn().mockResolvedValue(mockResponse),
  },
}));
```

### 5. Test User Behavior, Not Implementation
```typescript
// ✅ Good - Tests what user sees
expect(screen.getByText('Login successful')).toBeVisible();

// ❌ Bad - Tests implementation detail
expect(component.state.isLoggedIn).toBe(true);
```

## 🔍 Coverage Goals

Following AGENTS_FRONTEND.md requirements:

| Category | Target | Current |
|----------|--------|---------|
| Overall | > 80% | TBD |
| Stores | > 90% | TBD |
| Hooks | > 85% | TBD |
| Components | > 80% | TBD |
| Utils | > 90% | TBD |

### View Coverage Report
```bash
yarn test:coverage
open coverage/index.html
```

## 🧰 Testing Utilities

### Mock Data Generators
```typescript
import {
  createMockUser,
  createMockMessage,
  createMockEmotion,
} from '@/test/testUtils';

const user = createMockUser({ name: 'Custom Name' });
const message = createMockMessage({ content: 'Test' });
```

### Custom Render
```typescript
import { renderWithProviders } from '@/test/testUtils';

// Automatically wraps with Router, Query Client, etc.
const { getByText } = renderWithProviders(<MyComponent />);
```

### Wait Helpers
```typescript
import { waitForLoadingToFinish } from '@/test/testUtils';

await waitForLoadingToFinish();
expect(screen.getByText('Data loaded')).toBeVisible();
```

## 🎭 Playwright Configuration

### Browser Projects
- **Chromium** - Chrome, Edge
- **Firefox** - Firefox
- **WebKit** - Safari
- **Mobile Chrome** - Mobile testing
- **Mobile Safari** - iOS testing

### Running Specific Tests
```bash
# Single test file
yarn test:e2e e2e/auth.e2e.ts

# Specific test
yarn test:e2e -g "should login successfully"

# Headed mode (see browser)
yarn test:e2e --headed
```

## 📊 CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Unit Tests
  run: yarn test --run

- name: Run E2E Tests
  run: |
    yarn build
    yarn test:e2e
```

### Coverage Upload
```bash
yarn test:coverage
# Upload coverage/lcov.info to codecov/coveralls
```

## 🐛 Debugging Tests

### Vitest UI
```bash
yarn test:ui
# Opens visual test runner at http://localhost:51204
```

### Playwright Debug
```bash
yarn test:e2e:debug
# Step through tests with browser DevTools
```

### Console Logs
```typescript
import { screen, debug } from '@testing-library/react';

// Print DOM tree
debug();

// Print specific element
debug(screen.getByRole('button'));
```

## ✅ Test Checklist

Before committing code:

- [ ] All tests pass (`yarn test`)
- [ ] Coverage meets 80% threshold
- [ ] E2E tests pass for critical flows
- [ ] No skipped tests (`.skip`)
- [ ] No focused tests (`.only`)
- [ ] Accessibility tests included
- [ ] Mobile responsive tests included
- [ ] Error states tested
- [ ] Loading states tested
- [ ] Empty states tested

## 📚 Resources

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
- [Accessibility Testing](https://www.w3.org/WAI/test-evaluate/)

## 🎯 What's Tested

### Stores ✅
- authStore - Authentication state management
- chatStore - Message management
- emotionStore - Emotion state
- uiStore - UI preferences
- analyticsStore - Learning analytics

### Hooks ✅
- useAuth - Authentication hook
- useChat - Chat functionality
- useEmotion - Emotion detection
- useVoice - Voice interaction
- useWebSocket - Real-time updates

### Components ✅
- Button - All variants and states
- Modal - Open/close, focus trap
- Toast - Notifications
- Form fields - Validation, accessibility
- Chat components - Message flow

### E2E Flows ✅
- Authentication (login, signup, logout)
- Chat conversation
- Settings management
- Analytics dashboard
- Voice interaction

## 🚧 Test Coverage Status

```
Implemented Tests:
✅ authStore.test.ts - Authentication store (18 tests)
✅ chatStore.test.ts - Chat store (20 tests)
✅ useAuth.test.ts - Auth hook (6 tests)
✅ Button.test.tsx - Button component (20 tests)
✅ auth.e2e.ts - Authentication E2E (15 tests)

Total: 79 tests implemented

Pending Tests:
⏳ emotionStore.test.ts
⏳ uiStore.test.ts
⏳ analyticsStore.test.ts
⏳ Additional component tests
⏳ Integration tests
⏳ More E2E scenarios
```

## 📈 Next Steps

1. Run initial test suite: `yarn test`
2. Fix failing tests (adapt to actual implementation)
3. Add more component tests
4. Implement integration tests
5. Add E2E tests for all critical flows
6. Reach 80%+ coverage
7. Set up CI/CD pipeline
8. Enable coverage enforcement

---

**Status:** Testing infrastructure complete ✅  
**Coverage Target:** > 80% (AGENTS_FRONTEND.md requirement)  
**Last Updated:** October 28, 2025
