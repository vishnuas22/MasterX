import { create } from 'zustand';
import type { EmotionMetrics, LearningReadiness, CognitiveLoadLevel } from '@/types/emotion.types';

interface EmotionHistory {
  timestamp: string;
  emotion: string;
  intensity: number;
  valence: number;
  arousal: number;
  learningReadiness: LearningReadiness | null;
  cognitiveLoad: CognitiveLoadLevel | null;
}

interface EmotionStoreState {
  // State
  currentEmotion: EmotionMetrics | null;
  emotionHistory: EmotionHistory[];
  dominantEmotion: string | null;
  learningReadiness: LearningReadiness | null;
  cognitiveLoad: CognitiveLoadLevel | null;
  
  // Actions
  addEmotionData: (emotion: EmotionMetrics) => void;
  getEmotionTrend: (minutes: number) => EmotionHistory[];
  clearHistory: () => void;
  calculateDominantEmotion: () => void;
}

export const useEmotionStore = create<EmotionStoreState>((set, get) => ({
  // Initial state
  currentEmotion: null,
  emotionHistory: [],
  dominantEmotion: null,
  learningReadiness: null,
  cognitiveLoad: null,
  
  // Add new emotion data
  addEmotionData: (emotion) => {
    const historyEntry: EmotionHistory = {
      timestamp: new Date().toISOString(),
      emotion: emotion.primary_emotion,
      intensity: emotion.emotion_scores[emotion.primary_emotion] || 0,
      valence: emotion.pad_dimensions?.pleasure || 0,
      arousal: emotion.pad_dimensions?.arousal || 0,
      learningReadiness: emotion.learning_readiness,
      cognitiveLoad: emotion.cognitive_load,
    };
    
    set((state) => ({
      currentEmotion: emotion,
      emotionHistory: [...state.emotionHistory, historyEntry].slice(-100), // Keep last 100
      learningReadiness: emotion.learning_readiness,
      cognitiveLoad: emotion.cognitive_load,
    }));
    
    // Calculate dominant emotion
    get().calculateDominantEmotion();
  },
  
  // Get emotion trend for last N minutes
  getEmotionTrend: (minutes) => {
    const cutoffTime = new Date(Date.now() - minutes * 60 * 1000);
    return get().emotionHistory.filter(
      (entry) => new Date(entry.timestamp) > cutoffTime
    );
  },
  
  // Clear emotion history
  clearHistory: () => {
    set({
      emotionHistory: [],
      dominantEmotion: null,
      currentEmotion: null,
    });
  },
  
  // Calculate dominant emotion (most frequent in last 10 entries)
  calculateDominantEmotion: () => {
    const { emotionHistory } = get();
    const recent = emotionHistory.slice(-10);
    
    if (recent.length === 0) return;
    
    // Count emotion frequencies
    const counts: Record<string, number> = {};
    recent.forEach((entry) => {
      counts[entry.emotion] = (counts[entry.emotion] || 0) + 1;
    });
    
    // Find most frequent
    const dominant = Object.entries(counts).reduce((a, b) =>
      b[1] > a[1] ? b : a
    )[0];
    
    set({ dominantEmotion: dominant });
  },
}));
