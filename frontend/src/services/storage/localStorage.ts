/**
 * LocalStorage Service - Type-safe wrapper with error handling
 * 
 * Features:
 * - Type-safe get/set operations
 * - Automatic JSON serialization/deserialization
 * - Error handling (quota exceeded, parse errors)
 * - Size tracking
 * - Key enumeration
 * 
 * Storage Limits:
 * - Browser limit: ~5-10MB
 * - Automatically handles quota exceeded errors
 * 
 * Use Cases:
 * - User preferences (theme, language)
 * - Auth token persistence
 * - UI state (sidebar collapsed, etc.)
 * - Small cached data
 * 
 * For larger data (>1MB), use IndexedDB instead
 */
class LocalStorageService {
  /**
   * Get item from localStorage
   * @param key - Storage key
   * @returns Parsed value or null if not found
   */
  get<T>(key: string): T | null {
    try {
      const item = localStorage.getItem(key);
      if (!item) return null;
      
      return JSON.parse(item) as T;
    } catch (error) {
      console.error(`Error reading ${key} from localStorage:`, error);
      return null;
    }
  }

  /**
   * Set item in localStorage
   * @param key - Storage key
   * @param value - Value to store (will be JSON serialized)
   * @returns True if successful, false otherwise
   */
  set<T>(key: string, value: T): boolean {
    try {
      const serialized = JSON.stringify(value);
      localStorage.setItem(key, serialized);
      return true;
    } catch (error) {
      console.error(`Error writing ${key} to localStorage:`, error);
      
      // Check if quota exceeded
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        console.warn('LocalStorage quota exceeded');
        // Could clear old items here if needed
      }
      
      return false;
    }
  }

  /**
   * Remove item from localStorage
   * @param key - Storage key to remove
   */
  remove(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing ${key} from localStorage:`, error);
    }
  }

  /**
   * Clear all items from localStorage
   * Use with caution - clears everything!
   */
  clear(): void {
    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  }

  /**
   * Check if key exists
   * @param key - Storage key to check
   * @returns True if key exists
   */
  has(key: string): boolean {
    return localStorage.getItem(key) !== null;
  }

  /**
   * Get all keys in localStorage
   * @returns Array of all storage keys
   */
  keys(): string[] {
    return Object.keys(localStorage);
  }

  /**
   * Get storage size (approximate)
   * @returns Size in bytes
   */
  getSize(): number {
    let size = 0;
    for (const key in localStorage) {
      if (localStorage.hasOwnProperty(key)) {
        size += localStorage[key].length + key.length;
      }
    }
    return size; // bytes
  }

  /**
   * Get storage size in human-readable format
   * @returns Size string (e.g., "2.3 MB")
   */
  getSizeFormatted(): string {
    const bytes = this.getSize();
    const kb = bytes / 1024;
    const mb = kb / 1024;
    
    if (mb >= 1) {
      return `${mb.toFixed(2)} MB`;
    } else if (kb >= 1) {
      return `${kb.toFixed(2)} KB`;
    } else {
      return `${bytes} bytes`;
    }
  }
}

export const localStorageService = new LocalStorageService();
export default localStorageService;
