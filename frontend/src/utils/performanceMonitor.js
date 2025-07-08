// Performance monitoring utility for MasterX
class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.isEnabled = process.env.NODE_ENV === 'development' || window.location.hostname.includes('localhost');
  }

  // Start timing an operation
  startTiming(operation) {
    if (!this.isEnabled) return;
    this.metrics.set(operation, performance.now());
  }

  // End timing and log result
  endTiming(operation) {
    if (!this.isEnabled) return;
    const startTime = this.metrics.get(operation);
    if (startTime) {
      const duration = performance.now() - startTime;
      if (duration > 100) { // Only log slow operations
        console.log(`⏱️ ${operation}: ${duration.toFixed(2)}ms`);
      }
      this.metrics.delete(operation);
    }
  }

  // Log large console outputs (like the 1600+ messages issue)
  throttleConsole() {
    if (!this.isEnabled) return;
    
    const originalLog = console.log;
    let logCount = 0;
    const maxLogs = 50; // Limit logs in production

    console.log = function(...args) {
      if (logCount < maxLogs) {
        originalLog.apply(console, args);
        logCount++;
      } else if (logCount === maxLogs) {
        originalLog('🚫 Console logging limit reached. Suppressing further logs.');
        logCount++;
      }
    };
  }

  // Monitor bundle size impact
  logBundleMetrics() {
    if (!this.isEnabled || !window.performance) return;
    
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0];
      if (navigation) {
        console.log(`📦 Bundle Performance:`);
        console.log(`   Load: ${navigation.loadEventEnd - navigation.loadEventStart}ms`);
        console.log(`   DOM Ready: ${navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart}ms`);
        console.log(`   Total: ${navigation.loadEventEnd - navigation.fetchStart}ms`);
      }
    });
  }

  // Debounce utility for reducing excessive API calls
  debounce(func, delay) {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(null, args), delay);
    };
  }

  // Cache frequently accessed data
  createCache(maxSize = 100) {
    const cache = new Map();
    return {
      get: (key) => cache.get(key),
      set: (key, value) => {
        if (cache.size >= maxSize) {
          const firstKey = cache.keys().next().value;
          cache.delete(firstKey);
        }
        cache.set(key, value);
      },
      has: (key) => cache.has(key),
      clear: () => cache.clear()
    };
  }
}

export const performanceMonitor = new PerformanceMonitor();

// Auto-initialize monitoring
if (typeof window !== 'undefined') {
  performanceMonitor.throttleConsole();
  performanceMonitor.logBundleMetrics();
}