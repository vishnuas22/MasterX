"use client";

import { useState, useEffect } from "react";

// Simulate dynamic progress for demo
const useDemoProgress = () => {
  const [progress, setProgress] = useState(35);
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((p) => (p >= 100 ? 35 : p + 1));
    }, 90);
    return () => clearInterval(interval);
  }, []);
  return progress;
};

export default function TopBar({
  onReset,
}: {
  onReset?: () => void;
}) {
  const progress = useDemoProgress();

  return (
    <header className="w-full px-6 py-4 flex items-center justify-between glassy-topbar z-20">
      <div className="flex items-center gap-4">
        <span className="text-2xl font-bold tracking-wide neon-text">Alpha</span>
        <span className="ml-6 text-md text-muted-300 tracking-widest uppercase">
          AI Mentor
        </span>
      </div>
      <div className="flex items-center gap-6 min-w-[340px]">
        <div className="flex flex-col w-48">
          {/* Animated Neon Progress Bar */}
          <div className="relative h-3 rounded-full bg-dark-800 overflow-hidden border border-neon-700 shadow">
            <div
              className="absolute top-0 left-0 h-full rounded-full animate-glow"
              style={{
                width: `${progress}%`,
                background:
                  "linear-gradient(90deg, #7f1dff 0%, #6366f1 100%)",
                boxShadow:
                  "0 0 16px 2px #7f1dff, 0 0 2px 0 #6366f1, 0 0 36px 8px #818cf8",
                transition: "width 0.4s cubic-bezier(.4,2,.3,1)",
              }}
            />
          </div>
          <span className="text-xs text-neon-200 mt-1 ml-1 select-none">
            Progress: {progress}%
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-neon-200 text-sm font-semibold">
            Current Topic: <span className="text-neon-100">Basics</span>
          </span>
          <div className="w-10 h-10 rounded-full bg-glass border border-neon-300 shadow-neon flex items-center justify-center">
            {/* User Profile Icon */}
            <span className="text-xl font-bold text-neon-400">A</span>
          </div>
          {onReset && (
            <button
              onClick={onReset}
              className="ml-4 px-4 py-2 rounded-full bg-dark-800 border border-neon-400 text-neon-200 text-sm font-semibold shadow hover:bg-neon-900 hover:text-neon-100 transition"
            >
              Reset Chat & Name
            </button>
          )}
        </div>
      </div>
    </header>
  );
}