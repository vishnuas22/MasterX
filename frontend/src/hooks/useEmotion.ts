// **Purpose:** Access and analyze emotion data

// **What This File Contributes:**
// 1. Current emotion access
// 2. Emotion history
// 3. Trend analysis
// 4. Dominant emotion calculation

// **Implementation:**
// ```typescript
import { useEmotionStore } from '@store/emotionStore';
import { useMemo } from 'react';
import type { EmotionMetrics } from '@types/emotion.types';

export const useEmotion = () => {
  const {
    currentEmotion,
    emotionHistory,
    dominantEmotion,
    learningReadiness,
    cognitiveLoad,
    getEmotionTrend,
  } = useEmotionStore();

  /**
   * Get emotion color for visualization
   */
  const getEmotionColor = (emotion: string): string => {
    const colorMap: Record<string, string> = {
      joy: '#FFD60A',
      calm: '#64D2FF',
      focus: '#BF5AF2',
      frustration: '#FF453A',
      curiosity: '#30D158',
      confusion: '#FF9F0A',
      excitement: '#FFD60A',
      neutral: '#8E8E93',
    };
    return colorMap[emotion] || colorMap.neutral;
  };

  /**
   * Get learning readiness indicator
   */
  const getReadinessIndicator = (): {
    label: string;
    color: string;
    icon: string;
  } => {
    if (!learningReadiness) {
      return { label: 'Unknown', color: '#8E8E93', icon: '?' };
    }

    const indicators: Record<string, any> = {
      OPTIMAL: { label: 'Optimal', color: '#30D158', icon: '🎯' },
      READY: { label: 'Ready', color: '#64D2FF', icon: '✓' },
      STRUGGLING: { label: 'Struggling', color: '#FF9F0A', icon: '⚠️' },
      BLOCKED: { label: 'Blocked', color: '#FF453A', icon: '🚫' },
    };

    return indicators[learningReadiness] || indicators.READY;
  };

  /**
   * Get emotion trend for last N minutes
   */
  const getRecentTrend = (minutes: number = 30) => {
    return getEmotionTrend(minutes);
  };

  /**
   * Calculate emotion diversity (entropy)
   */
  const emotionDiversity = useMemo(() => {
    if (emotionHistory.length < 5) return 0;

    const recent = emotionHistory.slice(-20);
    const emotionCounts: Record<string, number> = {};
    
    recent.forEach((entry) => {
      emotionCounts[entry.emotion] = (emotionCounts[entry.emotion] || 0) + 1;
    });

    const total = recent.length;
    let entropy = 0;

    Object.values(emotionCounts).forEach((count) => {
      const p = count / total;
      entropy -= p * Math.log2(p);
    });

    return entropy;
  }, [emotionHistory]);

  return {
    // State
    currentEmotion,
    emotionHistory,
    dominantEmotion,
    learningReadiness,
    cognitiveLoad,
    emotionDiversity,
    
    // Helpers
    getEmotionColor,
    getReadinessIndicator,
    getRecentTrend,
  };
};


// **Benefits:**
// 1. Color mapping for UI consistency
// 2. Readiness indicators for visual feedback
// 3. Trend analysis helpers
// 4. Emotion diversity metric

// **Performance:**
// - useMemo for expensive calculations
// - Only recalculates when emotion history changes

// **Connected Files:**
// - ← `store/emotionStore.ts`
// - → `components/emotion/EmotionWidget.tsx`
// - → `components/emotion/EmotionChart.tsx`