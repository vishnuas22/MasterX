/**
 * WebSocket Services Barrel Exports
 * 
 * Real-time bidirectional communication with backend
 * - Auto-reconnection
 * - JWT authentication
 * - Type-safe event handling
 */

export { socketClient, default as defaultSocketClient } from './socket.client';
export {
  initializeSocketHandlers,
  cleanupSocketHandlers,
  emitTypingIndicator,
  joinSession,
  leaveSession,
} from './socket.handlers';
