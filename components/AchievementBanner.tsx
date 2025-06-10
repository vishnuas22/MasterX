"use client";

import { useEffect, useState } from "react";
import Confetti from "react-confetti";

export default function AchievementBanner({
  message,
  show,
  onClose,
}: {
  message: string;
  show: boolean;
  onClose?: () => void;
}) {
  const [visible, setVisible] = useState(show);

  useEffect(() => {
    if (show) {
      setVisible(true);
      const timer = setTimeout(() => {
        setVisible(false);
        if (onClose) onClose();
      }, 3200); // Show for 3.2 seconds
      return () => clearTimeout(timer);
    }
  }, [show, onClose]);

  if (!visible) return null;

  return (
    <>
      <Confetti
        width={typeof window !== "undefined" ? window.innerWidth : 1200}
        height={typeof window !== "undefined" ? window.innerHeight : 900}
        numberOfPieces={200}
        recycle={false}
        gravity={0.18}
        initialVelocityY={8}
      />
      <div className="fixed top-10 left-1/2 z-50 -translate-x-1/2 neon-glow">
        <div className="glassy-topbar px-8 py-5 rounded-2xl flex items-center gap-4 shadow-neon border-2 border-neon-400 animate-pop-in">
          <span className="text-3xl">🏆</span>
          <span className="text-xl font-bold text-neon-200 tracking-wide">
            {message}
          </span>
        </div>
      </div>
    </>
  );
}