/**
 * Storage Services Barrel Exports
 * 
 * Client-side storage solutions:
 * - LocalStorage: Small data, user preferences
 * - IndexedDB: Large data, offline functionality
 */

export { localStorageService, default as defaultLocalStorage } from './localStorage';
export { indexedDBService, default as defaultIndexedDB } from './indexedDB';
