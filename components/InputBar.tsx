"use client";

import { useState, useRef } from "react";

export default function InputBar({
  onSend,
  disabled = false,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
}) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || disabled) return;
    onSend(input.trim());
    setInput("");
    inputRef.current?.focus();
  };

  return (
    <footer className="w-full px-2 sm:px-8 py-5 absolute bottom-0 left-0 glassy-input z-20 flex items-center justify-center">
      <form
        className="w-full max-w-2xl flex gap-3"
        onSubmit={handleSend}
        autoComplete="off"
      >
        <input
          ref={inputRef}
          className="flex-1 px-6 py-3 rounded-full glassy-input border-none outline-none text-lg bg-dark-800 text-white shadow-inner focus:ring-2 focus:ring-neon-500 transition"
          type="text"
          placeholder="Type your message here…"
          aria-label="Message input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={disabled}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) handleSend(e as any);
          }}
        />
        <button
          type="submit"
          className="px-6 py-3 rounded-full bg-gradient-to-r from-neon-500 to-neon-300 text-white font-bold shadow-md neon-glow hover:scale-105 transition-all active:scale-95"
          disabled={disabled}
        >
          Send ▶️
        </button>
      </form>
    </footer>
  );
}