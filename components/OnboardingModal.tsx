"use client";

import { useState, useEffect, useRef } from "react";

export default function OnboardingModal({
  show,
  onComplete,
}: {
  show: boolean;
  onComplete: (name: string) => void;
}) {
  const [name, setName] = useState("");
  const [touched, setTouched] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (show) {
      inputRef.current?.focus();
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [show]);

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-dark-900/80 backdrop-blur-sm">
      <form
        className="glassy-topbar rounded-3xl shadow-neon px-12 py-10 flex flex-col items-center relative max-w-md mx-auto animate-pop-in"
        onSubmit={e => {
          e.preventDefault();
          setTouched(true);
          if (name.trim().length === 0) return;
          onComplete(name.trim());
        }}
      >
        <span className="text-4xl mb-4 neon-text">👋</span>
        <h2 className="text-2xl font-bold neon-text mb-2">Welcome to Alpha Mentor</h2>
        <p className="text-neon-200 text-lg mb-4 text-center">
          Your AI-powered learning partner for mastering any skill.<br />
          <span className="text-neon-400 font-semibold">What should Alpha call you?</span>
        </p>
        <input
          ref={inputRef}
          type="text"
          className="mb-3 px-6 py-3 rounded-full glassy-input text-lg text-neon-100 bg-dark-800 border-none outline-none focus:ring-2 focus:ring-neon-500 transition w-full"
          placeholder="Enter your name…"
          value={name}
          onChange={e => setName(e.target.value)}
          onFocus={() => setTouched(true)}
          maxLength={32}
          autoComplete="off"
        />
        {touched && name.trim().length === 0 && (
          <span className="text-neon-500 text-sm mb-2">Please enter your name.</span>
        )}
        <button
          type="submit"
          className="px-8 py-3 rounded-full bg-gradient-to-r from-neon-500 to-neon-300 text-white text-lg font-bold neon-glow mt-2 hover:scale-105 transition-all"
        >
          Get Started
        </button>
        <span
          className="absolute top-3 right-5 text-neon-300 cursor-pointer text-xl"
          title="Close"
          onClick={() => onComplete(name.trim() || "Learner")}
        >
          ×
        </span>
      </form>
    </div>
  );
}