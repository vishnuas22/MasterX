// **Purpose:** Complete emotion detection type system matching backend

// **What This File Contributes:**
// 1. 27 emotion categories (GoEmotions)
// 2. PAD dimensions (Pleasure-Arousal-Dominance)
// 3. Learning readiness states
// 4. Cognitive load levels
// 5. Flow state indicators

// **Implementation:**

// /**
//  * Emotion Detection Types
//  * 
//  * Matches backend emotion_core.py exactly:
//  * - EmotionCategory (lines 29-78)
//  * - LearningReadiness (lines 84-101)
//  * - CognitiveLoadLevel (lines 104-121)
//  * - FlowStateIndicator (lines 124-145)
//  * - EmotionMetrics (lines 283-507)
//  */

// ============================================================================
// EMOTION CATEGORIES (27 from GoEmotions Dataset)
// ============================================================================

export enum EmotionCategory {
  // Positive emotions (12)
  ADMIRATION = 'admiration',
  AMUSEMENT = 'amusement',
  APPROVAL = 'approval',
  CARING = 'caring',
  DESIRE = 'desire',
  EXCITEMENT = 'excitement',
  GRATITUDE = 'gratitude',
  JOY = 'joy',
  LOVE = 'love',
  OPTIMISM = 'optimism',
  PRIDE = 'pride',
  RELIEF = 'relief',
  
  // Negative emotions (11)
  ANGER = 'anger',
  ANNOYANCE = 'annoyance',
  DISAPPOINTMENT = 'disappointment',
  DISAPPROVAL = 'disapproval',
  DISGUST = 'disgust',
  EMBARRASSMENT = 'embarrassment',
  FEAR = 'fear',
  GRIEF = 'grief',
  NERVOUSNESS = 'nervousness',
  REMORSE = 'remorse',
  SADNESS = 'sadness',
  
  // Ambiguous emotions (4)
  CONFUSION = 'confusion',
  CURIOSITY = 'curiosity',
  REALIZATION = 'realization',
  SURPRISE = 'surprise',
  
  // Neutral state (1)
  NEUTRAL = 'neutral',
}

// ============================================================================
// LEARNING-SPECIFIC STATES
// ============================================================================

export enum LearningReadiness {
  OPTIMAL = 'optimal',
  GOOD = 'good',
  MODERATE = 'moderate',
  LOW = 'low',
  BLOCKED = 'blocked',
}

export enum CognitiveLoadLevel {
  UNDER_STIMULATED = 'under_stimulated',
  OPTIMAL = 'optimal',
  MODERATE = 'moderate',
  HIGH = 'high',
  OVERLOADED = 'overloaded',
}

export enum FlowStateIndicator {
  DEEP_FLOW = 'deep_flow',
  FLOW = 'flow',
  NEAR_FLOW = 'near_flow',
  NOT_IN_FLOW = 'not_in_flow',
  ANXIETY = 'anxiety',
  BOREDOM = 'boredom',
}

export enum InterventionLevel {
  NONE = 'none',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

export interface EmotionScore {
  emotion: EmotionCategory;
  confidence: number; // 0.0 - 1.0
}

export interface PADDimensions {
  pleasure: number; // -1.0 to 1.0 (negative to positive)
  arousal: number; // 0.0 to 1.0 (calm to excited)
  dominance: number; // 0.0 to 1.0 (submissive to in control)
}

export interface EmotionMetrics {
  // Raw ML predictions
  primary_emotion: EmotionCategory;
  primary_confidence: number; // 0.0 - 1.0
  secondary_emotions: EmotionScore[];
  emotion_scores: Record<string, number>;
  
  // PAD psychological dimensions
  pad_dimensions: PADDimensions;
  
  // Learning-specific assessments (ML-derived)
  learning_readiness: LearningReadiness;
  cognitive_load: CognitiveLoadLevel;
  flow_state: FlowStateIndicator;
  
  // Intervention recommendations (ML-driven)
  needs_intervention: boolean;
  intervention_level: InterventionLevel;
  suggested_actions: string[];
  
  // Metadata
  text_analyzed: string;
  processing_time_ms: number;
  model_version: string;
  timestamp: string; // ISO 8601
}

export interface EmotionState {
  primary_emotion: string;
  arousal: number; // 0.0 - 1.0
  valence: number; // 0.0 - 1.0
  learning_readiness: LearningReadiness;
}

export interface EmotionHistory {
  timestamp: string; // ISO 8601
  emotion: string;
  intensity: number;
  valence: number;
  arousal: number;
  learningReadiness?: LearningReadiness;
  cognitiveLoad?: CognitiveLoadLevel;
}

// ============================================================================
// EMOTION UI HELPERS
// ============================================================================

export interface EmotionColorMap {
  [key: string]: string; // CSS color
}

export const EMOTION_COLORS: EmotionColorMap = {
  joy: '#FFD60A',
  calm: '#64D2FF',
  focus: '#BF5AF2',
  frustration: '#FF453A',
  curiosity: '#30D158',
  confusion: '#FF9F0A',
  excitement: '#FFD60A',
  neutral: '#8E8E93',
  // Add all 27 emotions...
};

export interface EmotionVisualization {
  emotion: EmotionCategory;
  color: string;
  intensity: number;
  icon?: string; // emoji or icon name
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

export const isEmotionMetrics = (obj: unknown): obj is EmotionMetrics => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'primary_emotion' in obj &&
    'primary_confidence' in obj &&
    'pad_dimensions' in obj
  );
};

export const isPADDimensions = (obj: unknown): obj is PADDimensions => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'pleasure' in obj &&
    'arousal' in obj &&
    'dominance' in obj
  );
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export const getEmotionCategory = (emotion: string): EmotionCategory => {
  return (EmotionCategory[emotion.toUpperCase() as keyof typeof EmotionCategory] || EmotionCategory.NEUTRAL);
};

export const isPositiveEmotion = (emotion: EmotionCategory): boolean => {
  const positiveEmotions = [
    EmotionCategory.JOY,
    EmotionCategory.LOVE,
    EmotionCategory.EXCITEMENT,
    EmotionCategory.GRATITUDE,
    EmotionCategory.OPTIMISM,
    EmotionCategory.AMUSEMENT,
    EmotionCategory.ADMIRATION,
    EmotionCategory.APPROVAL,
    EmotionCategory.CARING,
    EmotionCategory.DESIRE,
    EmotionCategory.PRIDE,
    EmotionCategory.RELIEF,
  ];
  return positiveEmotions.includes(emotion);
};

export const isNegativeEmotion = (emotion: EmotionCategory): boolean => {
  const negativeEmotions = [
    EmotionCategory.ANGER,
    EmotionCategory.DISGUST,
    EmotionCategory.FEAR,
    EmotionCategory.GRIEF,
    EmotionCategory.SADNESS,
    EmotionCategory.DISAPPOINTMENT,
    EmotionCategory.EMBARRASSMENT,
    EmotionCategory.NERVOUSNESS,
    EmotionCategory.REMORSE,
    EmotionCategory.ANNOYANCE,
    EmotionCategory.DISAPPROVAL,
  ];
  return negativeEmotions.includes(emotion);
};


// **Key Features:**
// 1. **Complete emotion taxonomy:** All 27 GoEmotions categories
// 2. **Learning psychology:** Readiness, flow, cognitive load
// 3. **Type safety:** Enums for all categorical data
// 4. **Helper functions:** Common emotion operations
// 5. **Visualization support:** Color mapping for UI

// **Connected Files:**
// - ← Backend: `services/emotion/emotion_core.py`
// - → `store/emotionStore.ts` (emotion state management)
// - → `components/emotion/*` (emotion visualization)
// - → `hooks/useEmotion.ts` (emotion operations)