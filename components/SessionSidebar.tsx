import React, { useState } from "react";
import { SessionsData } from "../types/session";

interface SessionSidebarProps {
  sessionsData: SessionsData;
  onSelect: (sessionId: string) => void;
  onAdd: () => void;
  onDelete: (sessionId: string) => void;
}

const curriculumSteps = [
  "Intro & Goal Setup",
  "Foundations",
  "Practice",
  "Milestones",
  "Review",
];

export const SessionSidebar: React.FC<SessionSidebarProps> = ({
  sessionsData,
  onSelect,
  onAdd,
  onDelete,
}) => {
  const [open, setOpen] = useState(true);
  const [activeStep, setActiveStep] = useState(1);

  return (
    <aside
      className={`h-full glassy-sidebar transition-all duration-300 z-30 flex flex-col items-center py-6 ${
        open ? "w-72" : "w-16"
      }`}
      // Remove the white/gray background, let CSS class handle it
      style={{
        minHeight: "100vh",
        boxSizing: "border-box",
      }}
    >
      {/* Collapse/expand button */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="mb-6 p-2 rounded-full bg-dark-800 hover:bg-neon-800 transition"
        aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
      >
        <span className="text-neon-300">{open ? "⏴" : "⏵"}</span>
      </button>
      {open && (
        <div className="w-full px-4 flex flex-col gap-6">
          {/* Curriculum Section */}
          <div>
            <h2 className="text-md font-bold text-neon-200 mb-2">Curriculum</h2>
            <ul className="space-y-2 text-muted-200 text-sm">
              {curriculumSteps.map((step, idx) => (
                <li
                  key={step}
                  className={`flex items-center gap-2 cursor-pointer group transition relative ${
                    idx === activeStep
                      ? "text-neon-100 font-semibold"
                      : "hover:text-neon-400"
                  }`}
                  onClick={() => setActiveStep(idx)}
                >
                  <span
                    className={`w-2 h-2 rounded-full mr-1 transition-all
                      ${
                        idx === activeStep
                          ? "bg-gradient-to-r from-neon-400 to-neon-500 shadow-neon scale-125"
                          : "bg-dark-700 group-hover:bg-neon-400"
                      }
                    `}
                  />
                  <span
                    className={`transition-all ${
                      idx === activeStep ? "neon-text" : ""
                    }`}
                  >
                    {step}
                  </span>
                  {idx === activeStep && (
                    <span className="absolute -right-3 w-2 h-8 rounded bg-gradient-to-b from-neon-500 to-neon-400 opacity-70 animate-glow" />
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Achievements Section */}
          <div>
            <h2 className="text-md font-bold text-neon-200 mb-2">Achievements</h2>
            <ul className="space-y-1 text-muted-300 text-xs">
              <li>🎉 Milestone 1 unlocked</li>
              <li>🔒 Next: Consistency</li>
            </ul>
          </div>

          {/* Resources Section */}
          <div>
            <h2 className="text-md font-bold text-neon-200 mb-2">Resources</h2>
            <ul className="space-y-1 text-muted-300 text-xs">
              <li>
                <a href="#" className="hover:text-neon-400 underline">
                  Free Tutorials
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-neon-400 underline">
                  Tools & Guides
                </a>
              </li>
            </ul>
          </div>

          {/* Sessions List */}
          <div>
            <h2 className="text-md font-bold text-neon-200 mb-2">Sessions</h2>
            <button
              onClick={onAdd}
              className="w-full mb-2 p-2 bg-neon-200 text-dark-900 rounded hover:bg-neon-300 font-bold"
            >
              + New Session
            </button>
            <ul>
              {sessionsData.sessions.map((session) => (
                <li
                  key={session.id}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer ${
                    session.id === sessionsData.activeSessionId
                      ? "bg-neon-100 font-bold"
                      : "hover:bg-neon-50"
                  }`}
                  onClick={() => onSelect(session.id)}
                >
                  <span className="truncate">{session.title}</span>
                  <button
                    className="text-red-400 ml-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(session.id);
                    }}
                    aria-label="Delete session"
                  >
                    🗑️
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </aside>
  );
};