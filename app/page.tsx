"use client";

import React, { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import { SessionSidebar } from "../components/SessionSidebar";
import TopBar from "../components/TopBar";
import ChatPanel from "../components/ChatPanel";
import InputBar from "../components/InputBar";
import AchievementBanner from "../components/AchievementBanner";
import OnboardingModal from "../components/OnboardingModal";
import {
  loadSessions,
  saveSessions,
  addSession,
  updateSession,
  deleteSession,
} from "../lib/sessionStorage";
import { SessionsData, Session, ChatMessage } from "../types/session";

const STORAGE_KEY_NAME = "alpha_user_name";

const getInitialMessages = (name: string): ChatMessage[] => [
  {
    id: uuidv4(),
    sender: "ai",
    content: `Welcome${name ? `, ${name}` : ""}! I’m your AI mentor—ready to help you unlock your full potential. What’s your learning goal today?`,
    timestamp: new Date().toISOString(),
  },
];

export default function Home() {
  // All hooks MUST be at the top level
  const [sessionsData, setSessionsData] = useState<SessionsData | null>(null);
  const [userName, setUserName] = useState<string | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [showAchievement, setShowAchievement] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  // Load sessions from localStorage client-side only
  useEffect(() => {
    setSessionsData(loadSessions());
  }, []);

  // On mount: Load username
  useEffect(() => {
    if (typeof window === "undefined") return;
    const savedName = localStorage.getItem(STORAGE_KEY_NAME);
    if (savedName) {
      setUserName(savedName);
      setShowOnboarding(false);
    }
  }, []);

  // Persist username
  useEffect(() => {
    if (userName) {
      localStorage.setItem(STORAGE_KEY_NAME, userName);
    }
  }, [userName]);

  // Persist sessions on change
  useEffect(() => {
    if (sessionsData) saveSessions(sessionsData);
  }, [sessionsData]);

  // Show loading while sessions load (avoids hydration errors)
  if (!sessionsData) {
    return <div>Loading...</div>;
  }

  // Onboarding modal complete
  const handleOnboarding = (name: string) => {
    setUserName(name);
    setShowOnboarding(false);

    // Personalize all sessions with initial message if not present
    if (!sessionsData.sessions.length) {
      const firstSession: Session = {
        id: uuidv4(),
        title: `Session 1`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        messages: getInitialMessages(name),
      };
      setSessionsData({
        sessions: [firstSession],
        activeSessionId: firstSession.id,
      });
    } else {
      // For existing sessions, update first message if it's generic
      const updated = sessionsData.sessions.map((s) => ({
        ...s,
        messages:
          s.messages.length === 1 && s.messages[0].sender === "ai"
            ? getInitialMessages(name)
            : s.messages,
      }));
      setSessionsData({ ...sessionsData, sessions: updated });
    }
  };

  // Session management
  const handleAddSession = () => {
    const newSession: Session = {
      id: uuidv4(),
      title: `Session ${sessionsData.sessions.length + 1}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: getInitialMessages(userName || ""),
    };
    setSessionsData(addSession(newSession, sessionsData));
  };

  const handleSelectSession = (sessionId: string) => {
    setSessionsData({ ...sessionsData, activeSessionId: sessionId });
  };

  const handleDeleteSession = (sessionId: string) => {
    setSessionsData(deleteSession(sessionId, sessionsData));
  };

  // Reset all (username, sessions)
  const handleReset = () => {
    localStorage.removeItem(STORAGE_KEY_NAME);
    setUserName(null);
    setShowOnboarding(true);
    setSessionsData({ sessions: [], activeSessionId: null });
  };

  // Get active session
  const activeSession =
    sessionsData.sessions.find((s) => s.id === sessionsData.activeSessionId) || null;

  // Handle chat input send for active session
  const handleSend = async (text: string) => {
    if (!sessionsData.activeSessionId || !activeSession) return;
    setIsTyping(true);

    // Add user message
    const userMsg: ChatMessage = {
      id: uuidv4(),
      sender: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };
    const updatedSessions = sessionsData.sessions.map((session) => {
      if (session.id === sessionsData.activeSessionId) {
        return {
          ...session,
          messages: [...session.messages, userMsg],
          updatedAt: new Date().toISOString(),
        };
      }
      return session;
    });
    setSessionsData({ ...sessionsData, sessions: updatedSessions });

    // Prepare messages for API (OpenAI-style)
    const apiMessages = [
      ...(activeSession.messages || []).map(({ sender, content }) => ({
        role: sender === "ai" ? "assistant" : "user",
        content: content,
      })),
      { role: "user", content: text },
    ];

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: apiMessages }),
      });

      const data = await res.json();
      const aiReply: ChatMessage = {
        id: uuidv4(),
        sender: "ai",
        content: data.choices?.[0]?.message?.content ?? "Sorry, I had trouble responding.",
        timestamp: new Date().toISOString(),
      };

      // Add AI reply to session
      const newSessions = sessionsData.sessions.map((session) => {
        if (session.id === sessionsData.activeSessionId) {
          return {
            ...session,
            messages: [
              ...(session.messages || []),
              aiReply,
            ],
            updatedAt: new Date().toISOString(),
          };
        }
        return session;
      });
      setSessionsData({ ...sessionsData, sessions: newSessions });
    } catch (err) {
      // Add error message to chat
      const aiError: ChatMessage = {
        id: uuidv4(),
        sender: "ai",
        content: "Sorry, there was an error connecting to the AI.",
        timestamp: new Date().toISOString(),
      };
      const newSessions = sessionsData.sessions.map((session) => {
        if (session.id === sessionsData.activeSessionId) {
          return {
            ...session,
            messages: [
              ...(session.messages || []),
              aiError,
            ],
            updatedAt: new Date().toISOString(),
          };
        }
        return session;
      });
      setSessionsData({ ...sessionsData, sessions: newSessions });
    }
    setIsTyping(false);
    setTimeout(() => setShowAchievement(true), 1200);
  };

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw", overflow: "hidden" }}>
      {/* Session Sidebar */}
      <SessionSidebar
        sessionsData={sessionsData}
        onSelect={handleSelectSession}
        onAdd={handleAddSession}
        onDelete={handleDeleteSession}
      />
      <main style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative" }}>
        {/* Onboarding */}
        <OnboardingModal show={showOnboarding} onComplete={handleOnboarding} />

        {/* Top Bar with Reset */}
        <TopBar onReset={handleReset} />

        {/* Session Title */}
        <h1 style={{margin: "16px 0 0 0", textAlign: "left", fontSize: "1.8em", fontWeight: 700}}>
          {activeSession?.title || "No Session Selected"}
        </h1>

        {/* Chat */}
        <div style={{flex: 1, display: "flex", flexDirection: "column", minHeight: 0}}>
          <ChatPanel messages={activeSession?.messages || []} isTyping={isTyping} />
        </div>

        {/* Input */}
        <InputBar onSend={handleSend} disabled={!activeSession || showOnboarding} />

        {/* Achievement banner */}
        <AchievementBanner
          message="Milestone Unlocked!"
          show={showAchievement}
          onClose={() => setShowAchievement(false)}
        />
      </main>
    </div>
  );
}