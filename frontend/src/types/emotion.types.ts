/**
 * Emotion Detection Types
 * 
 * Matches backend emotion_core.py exactly:
 * - EmotionCategory (lines 29-78)
 * - LearningReadiness (lines 84-101)
 * - CognitiveLoadLevel (lines 104-121)
 * - FlowStateIndicator (lines 124-145)
 * - EmotionMetrics (lines 283-507)
 * 
 * @module types/emotion
 */

// ============================================================================
// EMOTION CATEGORIES (27 from GoEmotions Dataset)
// ============================================================================

/**
 * 27 emotion categories from Google's GoEmotions dataset
 * Used by RoBERTa/ModernBERT models in backend emotion detection
 */
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

/**
 * Learning readiness levels (ML-derived via Logistic Regression)
 * Indicates student's current capacity to learn effectively
 */
export enum LearningReadiness {
  OPTIMAL = 'optimal',
  GOOD = 'good',
  MODERATE = 'moderate',
  LOW = 'low',
  BLOCKED = 'blocked',
}

/**
 * Cognitive load levels (ML-derived via MLP Neural Network)
 * Measures mental effort required by working memory
 */
export enum CognitiveLoadLevel {
  UNDER_STIMULATED = 'under_stimulated',
  OPTIMAL = 'optimal',
  MODERATE = 'moderate',
  HIGH = 'high',
  OVERLOADED = 'overloaded',
}

/**
 * Flow state indicators (ML-derived via Random Forest)
 * Based on Csikszentmihalyi's flow theory
 */
export enum FlowStateIndicator {
  DEEP_FLOW = 'deep_flow',
  FLOW = 'flow',
  NEAR_FLOW = 'near_flow',
  NOT_IN_FLOW = 'not_in_flow',
  ANXIETY = 'anxiety',
  BOREDOM = 'boredom',
}

/**
 * Intervention level recommendations (ML-driven)
 * Indicates urgency of pedagogical intervention needed
 */
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

/**
 * Individual emotion score with confidence
 */
export interface EmotionScore {
  emotion: EmotionCategory;
  confidence: number; // 0.0 - 1.0
}

/**
 * PAD (Pleasure-Arousal-Dominance) psychological dimensions
 * Three-dimensional emotion representation from psychology
 */
export interface PADDimensions {
  pleasure: number; // -1.0 to 1.0 (negative to positive)
  arousal: number; // 0.0 to 1.0 (calm to excited)
  dominance: number; // 0.0 to 1.0 (submissive to in control)
}

/**
 * Complete emotion analysis result from backend
 * Contains all ML predictions and derived metrics
 */
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

/**
 * Simplified emotion state for UI display
 * Used in chat messages and real-time indicators
 */
export interface EmotionState {
  primary_emotion: string;
  arousal: number; // 0.0 - 1.0
  valence: number; // 0.0 - 1.0
  learning_readiness: LearningReadiness;
}

/**
 * Historical emotion data point for timeline/charts
 */
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

/**
 * Color mapping for emotion visualization
 * Uses Apple HIG color palette for consistency
 */
export interface EmotionColorMap {
  [key: string]: string; // CSS color (hex)
}

/**
 * Complete color palette for all 27 emotions
 * Colors chosen for accessibility (WCAG 2.1 AA compliant)
 */
export const EMOTION_COLORS: EmotionColorMap = {
  // Positive emotions - warm, vibrant colors
  joy: '#FFD60A', // Yellow
  love: '#FF375F', // Pink
  excitement: '#FFD60A', // Yellow
  gratitude: '#30D158', // Green
  optimism: '#64D2FF', // Light Blue
  amusement: '#FFD60A', // Yellow
  admiration: '#BF5AF2', // Purple
  approval: '#30D158', // Green
  caring: '#FF9F0A', // Orange
  desire: '#FF375F', // Pink
  pride: '#BF5AF2', // Purple
  relief: '#64D2FF', // Light Blue
  
  // Negative emotions - cool, muted colors
  anger: '#FF453A', // Red
  disgust: '#AC8E68', // Brown
  fear: '#8E8E93', // Gray
  grief: '#636366', // Dark Gray
  sadness: '#5E5CE6', // Indigo
  disappointment: '#FF9F0A', // Orange
  embarrassment: '#FF375F', // Pink
  nervousness: '#8E8E93', // Gray
  remorse: '#636366', // Dark Gray
  annoyance: '#FF9F0A', // Orange
  disapproval: '#FF453A', // Red
  
  // Ambiguous emotions - mixed colors
  confusion: '#FF9F0A', // Orange
  curiosity: '#30D158', // Green
  realization: '#BF5AF2', // Purple
  surprise: '#FFD60A', // Yellow
  
  // Neutral
  neutral: '#8E8E93', // Gray
};

/**
 * Emotion visualization data for UI components
 */
export interface EmotionVisualization {
  emotion: EmotionCategory;
  color: string;
  intensity: number; // 0.0 - 1.0
  icon?: string; // emoji or icon name
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

/**
 * Type guard to check if object is EmotionMetrics
 * @param obj - Object to check
 * @returns True if object matches EmotionMetrics interface
 */
export const isEmotionMetrics = (obj: unknown): obj is EmotionMetrics => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'primary_emotion' in obj &&
    'primary_confidence' in obj &&
    'pad_dimensions' in obj
  );
};

/**
 * Type guard to check if object is PADDimensions
 * @param obj - Object to check
 * @returns True if object matches PADDimensions interface
 */
export const isPADDimensions = (obj: unknown): obj is PADDimensions => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'pleasure' in obj &&
    'arousal' in obj &&
    'dominance' in obj
  );
};

/**
 * Type guard to check if object is EmotionState
 * @param obj - Object to check
 * @returns True if object matches EmotionState interface
 */
export const isEmotionState = (obj: unknown): obj is EmotionState => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'primary_emotion' in obj &&
    'arousal' in obj &&
    'valence' in obj &&
    'learning_readiness' in obj
  );
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Convert string to EmotionCategory enum
 * @param emotion - Emotion string (case-insensitive)
 * @returns Corresponding EmotionCategory or NEUTRAL if not found
 */
export const getEmotionCategory = (emotion: string): EmotionCategory => {
  const normalized = emotion.toUpperCase() as keyof typeof EmotionCategory;
  return EmotionCategory[normalized] || EmotionCategory.NEUTRAL;
};

/**
 * Check if emotion is positive
 * @param emotion - EmotionCategory to check
 * @returns True if emotion is positive
 */
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

/**
 * Check if emotion is negative
 * @param emotion - EmotionCategory to check
 * @returns True if emotion is negative
 */
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

/**
 * Check if emotion is ambiguous (neither clearly positive nor negative)
 * @param emotion - EmotionCategory to check
 * @returns True if emotion is ambiguous
 */
export const isAmbiguousEmotion = (emotion: EmotionCategory): boolean => {
  const ambiguousEmotions = [
    EmotionCategory.CONFUSION,
    EmotionCategory.CURIOSITY,
    EmotionCategory.REALIZATION,
    EmotionCategory.SURPRISE,
  ];
  return ambiguousEmotions.includes(emotion);
};

/**
 * Get color for emotion (with fallback to neutral)
 * @param emotion - Emotion string or EmotionCategory
 * @returns CSS color string
 */
export const getEmotionColor = (emotion: string | EmotionCategory): string => {
  const emotionKey = typeof emotion === 'string' ? emotion.toLowerCase() : emotion;
  return EMOTION_COLORS[emotionKey] || EMOTION_COLORS.neutral;
};

/**
 * Calculate valence from PAD dimensions
 * @param pad - PAD dimensions
 * @returns Valence score (0.0 to 1.0, where 0 is negative, 1 is positive)
 */
export const calculateValence = (pad: PADDimensions): number => {
  // Convert pleasure from [-1, 1] to [0, 1]
  return (pad.pleasure + 1) / 2;
};

/**
 * Get emoji representation of emotion
 * @param emotion - EmotionCategory
 * @returns Emoji string
 */
export const getEmotionEmoji = (emotion: EmotionCategory): string => {
  const emojiMap: Record<EmotionCategory, string> = {
    [EmotionCategory.JOY]: '😊',
    [EmotionCategory.LOVE]: '❤️',
    [EmotionCategory.EXCITEMENT]: '🤩',
    [EmotionCategory.GRATITUDE]: '🙏',
    [EmotionCategory.OPTIMISM]: '🌟',
    [EmotionCategory.AMUSEMENT]: '😄',
    [EmotionCategory.ADMIRATION]: '😍',
    [EmotionCategory.APPROVAL]: '👍',
    [EmotionCategory.CARING]: '🤗',
    [EmotionCategory.DESIRE]: '😍',
    [EmotionCategory.PRIDE]: '😎',
    [EmotionCategory.RELIEF]: '😌',
    [EmotionCategory.ANGER]: '😠',
    [EmotionCategory.DISGUST]: '🤢',
    [EmotionCategory.FEAR]: '😨',
    [EmotionCategory.GRIEF]: '😢',
    [EmotionCategory.SADNESS]: '😞',
    [EmotionCategory.DISAPPOINTMENT]: '😔',
    [EmotionCategory.EMBARRASSMENT]: '😳',
    [EmotionCategory.NERVOUSNESS]: '😰',
    [EmotionCategory.REMORSE]: '😔',
    [EmotionCategory.ANNOYANCE]: '😒',
    [EmotionCategory.DISAPPROVAL]: '👎',
    [EmotionCategory.CONFUSION]: '😕',
    [EmotionCategory.CURIOSITY]: '🤔',
    [EmotionCategory.REALIZATION]: '💡',
    [EmotionCategory.SURPRISE]: '😲',
    [EmotionCategory.NEUTRAL]: '😐',
  };
  
  return emojiMap[emotion] || '😐';
};

/**
 * Format learning readiness for display
 * @param readiness - LearningReadiness enum value
 * @returns Human-readable string
 */
export const formatLearningReadiness = (readiness: LearningReadiness): string => {
  const formatMap: Record<LearningReadiness, string> = {
    [LearningReadiness.OPTIMAL]: 'Optimal',
    [LearningReadiness.GOOD]: 'Good',
    [LearningReadiness.MODERATE]: 'Moderate',
    [LearningReadiness.LOW]: 'Low',
    [LearningReadiness.BLOCKED]: 'Blocked',
  };
  return formatMap[readiness];
};

/**
 * Format cognitive load for display
 * @param load - CognitiveLoadLevel enum value
 * @returns Human-readable string
 */
export const formatCognitiveLoad = (load: CognitiveLoadLevel): string => {
  const formatMap: Record<CognitiveLoadLevel, string> = {
    [CognitiveLoadLevel.UNDER_STIMULATED]: 'Under-stimulated',
    [CognitiveLoadLevel.OPTIMAL]: 'Optimal',
    [CognitiveLoadLevel.MODERATE]: 'Moderate',
    [CognitiveLoadLevel.HIGH]: 'High',
    [CognitiveLoadLevel.OVERLOADED]: 'Overloaded',
  };
  return formatMap[load];
};

/**
 * Format flow state for display
 * @param flowState - FlowStateIndicator enum value
 * @returns Human-readable string
 */
export const formatFlowState = (flowState: FlowStateIndicator): string => {
  const formatMap: Record<FlowStateIndicator, string> = {
    [FlowStateIndicator.DEEP_FLOW]: 'Deep Flow',
    [FlowStateIndicator.FLOW]: 'Flow',
    [FlowStateIndicator.NEAR_FLOW]: 'Near Flow',
    [FlowStateIndicator.NOT_IN_FLOW]: 'Not in Flow',
    [FlowStateIndicator.ANXIETY]: 'Anxious',
    [FlowStateIndicator.BOREDOM]: 'Bored',
  };
  return formatMap[flowState];
};
