// **Purpose:** Track emotion history, patterns, analytics

// **What This File Contributes:**
// 1. Emotion timeline (visualize changes over time)
// 2. Dominant emotion detection
// 3. Emotion pattern analysis
// 4. Learning readiness tracking

// **Implementation:**
// ```typescript
import { create } from 'zustand';
import type { EmotionMetrics, EmotionHistory } from '@types/emotion.types';

interface EmotionState {
  // State
  currentEmotion: EmotionMetrics | null;
  emotionHistory: EmotionHistory[];
  dominantEmotion: string | null;
  learningReadiness: string | null;
  cognitiveLoad: number | null;
  
  // Actions
  addEmotionData: (emotion: EmotionMetrics) => void;
  getEmotionTrend: (minutes: number) => EmotionHistory[];
  clearHistory: () => void;
  calculateDominantEmotion: () => void;
}

export const useEmotionStore = create<EmotionState>((set, get) => ({
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
      valence: emotion.pad_dimensions?.valence || 0,
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


// **Key Features:**
// 1. **History tracking:** Last 100 emotion data points
// 2. **Trend analysis:** Get emotions for time range
// 3. **Dominant emotion:** Most frequent emotion
// 4. **Real-time updates:** Updates as AI detects emotions

// **Performance:**
// - Array operations: O(n) where n max 100
// - Memory: ~5KB for 100 entries
// - No backend calls (derived from chatStore)

// **Connected Files:**
// - ← `store/chatStore.ts` (receives emotion data)
// - → `components/emotion/EmotionTimeline.tsx` (visualizes history)
// - → `components/emotion/MoodTracker.tsx` (displays dominant emotion)
// - → `components/analytics/EmotionChart.tsx` (charts trends)
