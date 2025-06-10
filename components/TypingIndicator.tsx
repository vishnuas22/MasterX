export default function TypingIndicator() {
    return (
      <div className="flex items-end animate-fade-in">
        <div className="mr-3 flex items-center justify-center w-10 h-10 bg-neon-800 border-2 border-neon-400 rounded-full shadow-neon">
          <span className="text-neon-200 text-xl font-bold">A</span>
        </div>
        <div className="rounded-2xl px-6 py-4 shadow-lg max-w-[75%] bg-glass border border-neon-700 text-neon-100 flex items-center gap-3">
          <span>Alpha is typing</span>
          <span className="flex items-center h-6">
            <span className="dot dot1"></span>
            <span className="dot dot2"></span>
            <span className="dot dot3"></span>
          </span>
        </div>
      </div>
    );
  }