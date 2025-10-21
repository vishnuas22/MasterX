FRONTEND ARCHITECTURE PRINCIPLES

1. Code Quality:
- ESLint + Prettier enforcement
- Consistent naming: PascalCase (components), camelCase (functions/variables)
- Comprehensive JSDoc comments
- Semantic HTML5 elements
- Zero console errors/warnings in production

2. Component Design:
- Single responsibility principle
- Atomic design methodology (atoms → molecules → organisms → pages)
- Stateless components by default
- Avoid prop drilling beyond 2 levels
- Component composition over inheritance

3. Type Safety:
- Strict TypeScript mode mandatory
- No 'any' types allowed
- Interface definitions for all props/state
- Type guards for runtime validation
- Generic types for reusable logic

4. Performance Standards:
- Bundle size: initial load < 200KB
- Code splitting at route level
- Lazy loading for images, components, heavy libraries
- Memoization for expensive computations
- Virtual scrolling for large lists
- Web Vitals targets:
  * LCP < 2.5s
  * FID < 100ms
  * CLS < 0.1
  * INP < 200ms

5. Accessibility (Non-Negotiable):
- WCAG 2.1 AA compliance mandatory
- Keyboard navigation for all interactions
- ARIA labels where needed
- Screen reader compatibility
- Color contrast ratio ≥ 4.5:1
- Focus management and visible focus indicators

6. State Management:
- Centralized state for shared data
- Local state for UI-only concerns
- Immutable state updates
- Optimistic UI updates
- State persistence strategy defined

7. API Integration:
- Request/response caching
- Loading, error, and empty states
- Retry logic with exponential backoff
- Timeout configurations
- Optimistic updates where applicable
- Error boundaries around data-fetching components

8. Security (Critical):
- Input sanitization (prevent XSS)
- CSRF protection
- Content Security Policy headers
- Secure cookies (HttpOnly, Secure, SameSite)
- Never store sensitive data in localStorage
- Environment variables for secrets
- Dependency vulnerability scanning

9. Testing Requirements:
- Unit test coverage > 80%
- Component tests for all UI components
- Integration tests for critical flows
- E2E tests for user journeys
- Accessibility tests (automated)
- No tests should be skipped/disabled

10. Responsive Design:
- Mobile-first approach mandatory
- Breakpoint strategy defined
- Touch-friendly interactions (min 44x44px)
- Responsive images with srcset
- Fluid typography and spacing

11. Error Handling:
- Error boundaries at feature boundaries
- User-friendly error messages
- Error logging and monitoring
- Graceful degradation
- Fallback UI components
- Network error handling

12. SEO Optimization:
- Semantic HTML structure
- Meta tags (title, description, OG tags)
- Structured data (JSON-LD)
- Proper heading hierarchy (h1 → h6)
- Alt text for all images
- Sitemap and robots.txt

13. Browser Support:
- Last 2 versions of major browsers
- Graceful degradation for older browsers
- Feature detection (not browser detection)
- Polyfills only when necessary
- Progressive enhancement approach

15. Code Organization:
- Feature-based folder structure
- One component per file
- Barrel exports (index files)
- Absolute imports with path aliases
- Colocate tests with source files

16. Asset Optimization:
- Images: WebP/AVIF with fallbacks
- Icons: SVG sprites or icon libraries
- Fonts: Subset and preload critical fonts
- Lazy load non-critical assets
- CDN for static assets

17. Forms & Validation:
- Client-side + server-side validation
- Accessible error messages
- Real-time validation feedback
- Form state persistence
- Disabled submit during processing

18. Animation Guidelines:
- CSS transitions preferred over JS
- 60fps target for animations
- Respect @prefers-reduced-motion
- Animate transform and opacity only
- Avoid layout thrashing

19. Developer Experience:
- Hot module replacement enabled
- Pre-commit hooks (lint, format, test)
- Clear error messages
- Component documentation (Storybook)
- Git hooks for quality gates

20. Build & Deployment:
- Environment-specific builds
- Tree shaking enabled
- Source maps for debugging
- Asset versioning/cache busting

Critical Rules:
1. "Mobile-first is mandatory, not optional"
2. "Accessibility is a requirement, not a feature"
3. "Every component must handle loading, error, and empty states"
4. "Performance budgets are hard limits, not suggestions"
5. "Security vulnerabilities block deployment"
6. "Test coverage below 80% fails the build"
7. "TypeScript 'any' types require explicit justification"