import { useState, useEffect, RefObject } from 'react';

interface UseIntersectionOptions {
  /** Threshold (0-1) at which to trigger intersection */
  threshold?: number;
  /** Root element for intersection (default: viewport) */
  root?: Element | null;
  /** Margin around root element */
  rootMargin?: string;
}

/**
 * Intersection Observer hook for lazy loading
 * 
 * Use cases:
 * - Lazy load images when they enter viewport
 * - Infinite scroll (load more when reaching bottom)
 * - Analytics (track which sections user views)
 * - Animations (trigger when element is visible)
 * 
 * Performance benefits:
 * - Reduces initial page load time
 * - Saves bandwidth (only loads visible content)
 * - Improves Core Web Vitals (LCP, CLS)
 * 
 * @param elementRef - Ref to element to observe
 * @param options - Intersection observer options
 * @returns Boolean indicating if element is in viewport
 * 
 * @example
 * const imageRef = useRef<HTMLImageElement>(null);
 * const isVisible = useIntersection(imageRef, { threshold: 0.5 });
 * 
 * return (
 *   <img
 *     ref={imageRef}
 *     src={isVisible ? actualImage : placeholder}
 *     alt="Lazy loaded image"
 *   />
 * );
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
        threshold: options.threshold || 0.1, // Trigger at 10% visibility
        root: options.root || null, // Use viewport as root
        rootMargin: options.rootMargin || '0px',
      }
    );

    observer.observe(element);

    // Cleanup: disconnect observer when component unmounts
    return () => {
      observer.disconnect();
    };
  }, [elementRef, options.threshold, options.root, options.rootMargin]);

  return isIntersecting;
};

/**
 * Advanced usage example: Infinite scroll
 * 
 * const lastItemRef = useRef<HTMLDivElement>(null);
 * const isLastItemVisible = useIntersection(lastItemRef, {
 *   threshold: 1.0, // Fully visible
 *   rootMargin: '100px' // Trigger 100px before reaching bottom
 * });
 * 
 * useEffect(() => {
 *   if (isLastItemVisible && !isLoading) {
 *     loadMoreItems();
 *   }
 * }, [isLastItemVisible]);
 */
