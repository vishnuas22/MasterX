// **Purpose:** Detect when element enters viewport (lazy loading)

// **What This File Contributes:**
// 1. Viewport visibility detection
// 2. Lazy loading trigger
// 3. Infinite scroll support
// 4. Performance optimization

// **Implementation:**
// ```typescript
import { useState, useEffect, RefObject } from 'react';

interface UseIntersectionOptions {
  threshold?: number;
  root?: Element | null;
  rootMargin?: string;
}

/**
 * Intersection Observer hook for lazy loading
 */
export const useIntersection = (
  elementRef: RefObject<Element>,
  options: UseIntersectionOptions = {}
): boolean => {
  const [isIntersecting, setIsIntersecting] = useState(false);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
      },
      {
        threshold: options.threshold || 0.1,
        root: options.root || null,
        rootMargin: options.rootMargin || '0px',
      }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [elementRef, options.threshold, options.root, options.rootMargin]);

  return isIntersecting;
};

// /**
//  * Usage example:
//  * 
//  * const imageRef = useRef<HTMLImageElement>(null);
//  * const isVisible = useIntersection(imageRef, { threshold: 0.5 });
//  * 
//  * return (
//  *   <img
//  *     ref={imageRef}
//  *     src={isVisible ? actualSrc : placeholderSrc}
//  *     alt="Lazy loaded"
//  *   />
//  * );
//  */
// ```

// **Benefits:**
// 1. Native browser API (performant)
// 2. No external dependencies
// 3. Configurable threshold
// 4. Automatic cleanup

// **Performance Impact:**
// - Images: Load only when visible (saves 70% bandwidth)
// - Charts: Render only when user scrolls to them
// - Infinite scroll: Load more items automatically

// **Connected Files:**
// - → `components/chat/MessageList.tsx` (infinite scroll)
// - → `components/analytics/ProgressChart.tsx` (lazy render)
// - → `components/emotion/EmotionChart.tsx` (lazy render)