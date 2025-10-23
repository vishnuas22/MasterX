import { openDB, DBSchema, IDBPDatabase } from 'idb';

/**
 * IndexedDB Schema Definition
 * Defines structure of client-side database
 */
interface MasterXDB extends DBSchema {
  messages: {
    key: string;
    value: {
      id: string;
      session_id: string;
      content: string;
      role: 'user' | 'assistant';
      timestamp: string;
      emotion: any;
      synced: boolean;
    };
    indexes: { 'by-session': string };
  };
  sessions: {
    key: string;
    value: {
      id: string;
      user_id: string;
      created_at: string;
      message_count: number;
    };
  };
  audio_cache: {
    key: string;
    value: {
      id: string;
      audio_blob: Blob;
      text: string;
      created_at: string;
    };
  };
}

/**
 * IndexedDB Service - Client-side database for offline functionality
 * 
 * Features:
 * - Store chat history offline
 * - Cache API responses
 * - Store large files (audio blobs)
 * - Sync when connection restored
 * 
 * Benefits:
 * - No size limits (can store GBs)
 * - Indexed queries (fast lookups)
 * - Async, non-blocking operations
 * - Transactional (ACID guarantees)
 * 
 * Performance:
 * - Indexed queries: O(log n)
 * - Supports hundreds of thousands of records
 * 
 * Use Cases:
 * - Offline chat messages
 * - Audio TTS response caching
 * - Large data sets (conversation history)
 * - Progressive Web App (PWA) support
 */
class IndexedDBService {
  private db: IDBPDatabase<MasterXDB> | null = null;
  private readonly DB_NAME = 'masterx-db';
  private readonly DB_VERSION = 1;

  /**
   * Initialize database
   * Creates object stores and indexes on first run
   */
  async init(): Promise<void> {
    this.db = await openDB<MasterXDB>(this.DB_NAME, this.DB_VERSION, {
      upgrade(db) {
        // Messages store
        if (!db.objectStoreNames.contains('messages')) {
          const messageStore = db.createObjectStore('messages', { keyPath: 'id' });
          messageStore.createIndex('by-session', 'session_id');
        }

        // Sessions store
        if (!db.objectStoreNames.contains('sessions')) {
          db.createObjectStore('sessions', { keyPath: 'id' });
        }

        // Audio cache store
        if (!db.objectStoreNames.contains('audio_cache')) {
          db.createObjectStore('audio_cache', { keyPath: 'id' });
        }
      },
    });
  }

  /**
   * Save message offline
   * @param message - Message to save
   */
  async saveMessage(message: MasterXDB['messages']['value']): Promise<void> {
    if (!this.db) await this.init();
    await this.db!.put('messages', message);
  }

  /**
   * Get messages for session
   * @param sessionId - Session ID
   * @returns Array of messages
   */
  async getSessionMessages(sessionId: string): Promise<MasterXDB['messages']['value'][]> {
    if (!this.db) await this.init();
    return await this.db!.getAllFromIndex('messages', 'by-session', sessionId);
  }

  /**
   * Get unsynced messages
   * Used for syncing offline messages when connection restored
   * @returns Array of unsynced messages
   */
  async getUnsyncedMessages(): Promise<MasterXDB['messages']['value'][]> {
    if (!this.db) await this.init();
    const allMessages = await this.db!.getAll('messages');
    return allMessages.filter((msg) => !msg.synced);
  }

  /**
   * Mark message as synced
   * @param messageId - Message ID to mark as synced
   */
  async markMessageSynced(messageId: string): Promise<void> {
    if (!this.db) await this.init();
    const message = await this.db!.get('messages', messageId);
    if (message) {
      message.synced = true;
      await this.db!.put('messages', message);
    }
  }

  /**
   * Cache audio blob (TTS responses)
   * @param id - Audio ID (hash of text)
   * @param audioBlob - Audio data
   * @param text - Original text
   */
  async cacheAudio(id: string, audioBlob: Blob, text: string): Promise<void> {
    if (!this.db) await this.init();
    await this.db!.put('audio_cache', {
      id,
      audio_blob: audioBlob,
      text,
      created_at: new Date().toISOString(),
    });
  }

  /**
   * Get cached audio
   * @param id - Audio ID
   * @returns Audio blob or null if not found
   */
  async getCachedAudio(id: string): Promise<Blob | null> {
    if (!this.db) await this.init();
    const cached = await this.db!.get('audio_cache', id);
    return cached?.audio_blob || null;
  }

  /**
   * Clear old cache (keep last 7 days)
   * Call periodically to prevent database bloat
   */
  async clearOldCache(): Promise<void> {
    if (!this.db) await this.init();
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    
    const allAudio = await this.db!.getAll('audio_cache');
    for (const audio of allAudio) {
      if (new Date(audio.created_at) < sevenDaysAgo) {
        await this.db!.delete('audio_cache', audio.id);
      }
    }
  }

  /**
   * Get database size estimate
   * Note: Not all browsers support this
   * @returns Size in bytes (approximate)
   */
  async getSize(): Promise<number> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      return estimate.usage || 0;
    }
    return 0;
  }

  /**
   * Close database connection
   * Call when app is closing or logging out
   */
  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

export const indexedDBService = new IndexedDBService();
export default indexedDBService;
