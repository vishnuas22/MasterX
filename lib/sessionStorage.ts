// Utility functions for loading/saving sessions to localStorage

import { SessionsData, Session, ChatMessage } from "../types/session";
import { v4 as uuidv4 } from "uuid";

const STORAGE_KEY = "alpha_sessions";

// Patch all messages to ensure they have a unique id (for backward compatibility)
function patchMessages(sessionsData: SessionsData): SessionsData {
  if (!sessionsData || !Array.isArray(sessionsData.sessions)) return { sessions: [], activeSessionId: null };
  // Map of seen IDs to ensure uniqueness
  const seen = new Set<string>();
  return {
    ...sessionsData,
    sessions: sessionsData.sessions.map(session => ({
      ...session,
      messages: (session.messages || []).map((msg: any) => {
        let id = msg.id;
        if (!id || seen.has(id)) {
          id = uuidv4();
        }
        seen.add(id);
        return { ...msg, id };
      }),
    })),
  };
}

export function loadSessions(): SessionsData {
  if (typeof window === "undefined") return { sessions: [], activeSessionId: null };
  const data = localStorage.getItem(STORAGE_KEY);
  if (data) return patchMessages(JSON.parse(data));
  return { sessions: [], activeSessionId: null };
}

export function saveSessions(sessionsData: SessionsData) {
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessionsData));
  }
}

// Add, update, delete sessions
export function addSession(newSession: Session, sessionsData: SessionsData): SessionsData {
  return {
    ...sessionsData,
    sessions: [...sessionsData.sessions, newSession],
    activeSessionId: newSession.id,
  };
}

export function updateSession(updatedSession: Session, sessionsData: SessionsData): SessionsData {
  return {
    ...sessionsData,
    sessions: sessionsData.sessions.map((s) => (s.id === updatedSession.id ? updatedSession : s)),
  };
}

export function deleteSession(sessionId: string, sessionsData: SessionsData): SessionsData {
  const filtered = sessionsData.sessions.filter((s) => s.id !== sessionId);
  return {
    ...sessionsData,
    sessions: filtered,
    activeSessionId: filtered.length ? filtered[0].id : null,
  };
}