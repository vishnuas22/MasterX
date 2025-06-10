// Data types for session management

export interface ChatMessage {
    id: string;
    sender: "user" | "ai";
    content: string;
    timestamp: string;
  }
  
  export interface RoadmapStep {
    id: string;
    title: string;
    description?: string;
    completed: boolean;
  }
  
  export interface Session {
    id: string;
    title: string;
    createdAt: string;
    updatedAt: string;
    roadmap?: RoadmapStep[];
    currentStep?: number;
    messages: ChatMessage[];
    archived?: boolean;
    // You can add more metadata as needed (goal, timeline, etc.)
  }
  
  export interface SessionsData {
    sessions: Session[];
    activeSessionId: string | null;
  }
  
  export interface ChatMessage {
    id: string;
    sender: "user" | "ai";
    content: string;
    timestamp: string;
  }