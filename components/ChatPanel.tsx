"use client";

import { useRef, useEffect } from "react";
import TypingIndicator from "./TypingIndicator";
import { ChatMessage } from "../types/session";

function Avatar({ sender }: { sender: "ai" | "user" }) {
  return sender === "ai" ? (
    <div className="mr-3 flex items-center justify-center w-10 h-10 bg-neon-800 border-2 border-neon-400 rounded-full shadow-neon">
      <span className="text-neon-200 text-xl font-bold">A</span>
    </div>
  ) : (
    <div className="ml-3 flex items-center justify-center w-10 h-10 bg-dark-800 border-2 border-neon-300 rounded-full">
      <span className="text-neon-100 text-xl">🧑‍💻</span>
    </div>
  );
}

export default function ChatPanel({
  messages,
  isTyping,
}: {
  messages: ChatMessage[];
  isTyping: boolean;
}) {
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // Debug: log message IDs
  console.log("ChatPanel messages", messages.map(m => m.id));

  return (
    <section ref={chatRef}>
      <div className="flex flex-col gap-6">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-end transition-all duration-300 ease-in-out ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {msg.sender === "ai" && <Avatar sender="ai" />}
            <div
              className={`rounded-2xl px-6 py-4 shadow-lg max-w-[75%] break-words text-base sm:text-lg transition-all duration-300 ${
                msg.sender === "ai"
                  ? "bg-glass border border-neon-700 text-neon-100 neon-glow"
                  : "bg-neon-800 text-white border border-neon-300"
              }`}
            >
              {msg.content}
            </div>
            {msg.sender === "user" && <Avatar sender="user" />}
          </div>
        ))}
        {isTyping && <TypingIndicator />}
      </div>
    </section>
  );
}