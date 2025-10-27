/**
 * EmotionIndicator Component - Real-time Emotion Display
 * 
 * WCAG 2.1 AA Compliant:
 * - Color is not the only visual means (icons + text)
 * - Sufficient contrast ratios (>4.5:1)
 * - Accessible tooltips with explanations
 * - Screen reader announcements for emotion changes
 * 
 * Performance:
 * - Smooth animations (60fps)
 * - Memoized calculations
 * - Lightweight (<3KB)
 * 
 * Backend Integration:
 * - Emotion data from ChatResponse.emotion_state
 * - Real-time updates via WebSocket
 * - Historical data from messages collection
 */

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Smile, Frown, Meh, AlertCircle, Sparkles,
  TrendingUp, TrendingDown, Minus, Zap,
  Brain, Target, Activity
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { Tooltip } from '@/components/ui/Tooltip';
import { Badge } from '@/components/ui/Badge';
import type { EmotionState } from '@/types/emotion.types';

// ============================================================================
// TYPES
// ============================================================================

export interface EmotionIndicatorProps {
  /**
   * Current emotion state from backend
   */
  emotion: EmotionState;
  
  /**
   * Is emotion being analyzed
   */
  isAnalyzing?: boolean;
  
  /**
   * Compact mode (icon only)
   */
  compact?: boolean;
  
  /**
   * Show detailed breakdown
   */
  showDetails?: boolean;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

// ============================================================================
// EMOTION CONFIGURATION
// ============================================================================

/**
 * Emotion metadata (icons, colors, descriptions)
 */
const emotionConfig: Record<string, {
  icon: React.FC<any>;
  color: string;
  bgColor: string;
  label: string;
  description: string;
}> = {
  joy: {
    icon: Smile,
    color: 'text-emotion-joy',
    bgColor: 'bg-emotion-joy/20',
    label: 'Joyful',
    description: 'Experiencing happiness and positive engagement'
  },
  curiosity: {
    icon: Sparkles,
    color: 'text-emotion-curiosity',
    bgColor: 'bg-emotion-curiosity/20',
    label: 'Curious',
    description: 'Engaged and interested in learning more'
  },
  frustration: {
    icon: Frown,
    color: 'text-emotion-frustration',
    bgColor: 'bg-emotion-frustration/20',
    label: 'Frustrated',
    description: 'Experiencing difficulty, may need support'
  },
  confusion: {
    icon: AlertCircle,
    color: 'text-accent-warning',
    bgColor: 'bg-accent-warning/20',
    label: 'Confused',
    description: 'Uncertain or unclear about the concept'
  },
  calm: {
    icon: Meh,
    color: 'text-emotion-calm',
    bgColor: 'bg-emotion-calm/20',
    label: 'Calm',
    description: 'Relaxed and focused learning state'
  },
  focus: {
    icon: Target,
    color: 'text-emotion-focus',
    bgColor: 'bg-emotion-focus/20',
    label: 'Focused',
    description: 'Deep concentration on the task'
  }
};

/**
 * Learning readiness levels
 */
const readinessConfig: Record<string, {
  label: string;
  color: string;
  icon: React.FC<any>;
  description: string;
}> = {
  optimal_readiness: {
    label: 'Optimal',
    color: 'text-accent-success',
    icon: TrendingUp,
    description: 'Perfect state for learning - energized and positive'
  },
  high_readiness: {
    label: 'High',
    color: 'text-accent-success',
    icon: TrendingUp,
    description: 'Great state for learning - engaged and ready'
  },
  moderate_readiness: {
    label: 'Moderate',
    color: 'text-accent-warning',
    icon: Minus,
    description: 'Can learn but may need some support'
  },
  low_readiness: {
    label: 'Low',
    color: 'text-accent-warning',
    icon: TrendingDown,
    description: 'Challenging state - may need encouragement'
  },
  not_ready: {
    label: 'Not Ready',
    color: 'text-accent-error',
    icon: TrendingDown,
    description: 'Difficult learning conditions - needs intervention'
  }
};

// ============================================================================
// PAD METER COMPONENT
// ============================================================================

const PADMeter = React.memo<{
  label: string;
  value: number;
  icon: React.FC<any>;
  description: string;
}>(({ label, value, icon: Icon, description }) => {
  // Normalize value (-1 to 1) to percentage (0 to 100)
  const percentage = ((value + 1) / 2) * 100;
  
  return (
    <Tooltip content={description}>
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-1 text-text-secondary">
            <Icon className="w-3 h-3" />
            <span>{label}</span>
          </div>
          <span className="text-text-primary font-semibold">
            {value.toFixed(2)}
          </span>
        </div>
        
        {/* Progress bar */}
        <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className={cn(
              'h-full rounded-full',
              value > 0.5
                ? 'bg-accent-success'
                : value > 0
                ? 'bg-accent-warning'
                : 'bg-accent-error'
            )}
          />
        </div>
      </div>
    </Tooltip>
  );
});

PADMeter.displayName = 'PADMeter';

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const EmotionIndicator: React.FC<EmotionIndicatorProps> = ({
  emotion,
  isAnalyzing = false,
  compact = false,
  showDetails = false,
  className
}) => {
  // ============================================================================
  // MEMOIZED VALUES
  // ============================================================================
  
  const emotionData = useMemo(
    () => emotionConfig[emotion.primary_emotion] || emotionConfig.calm,
    [emotion.primary_emotion]
  );
  
  const readinessData = useMemo(
    () => readinessConfig[emotion.learning_readiness] || readinessConfig.moderate_readiness,
    [emotion.learning_readiness]
  );
  
  const EmotionIcon = emotionData.icon;
  const ReadinessIcon = readinessData.icon;
  
  // ============================================================================
  // COMPACT MODE
  // ============================================================================
  
  if (compact) {
    return (
      <Tooltip content={`${emotionData.label} • ${readinessData.label} readiness`}>
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className={cn(
            'relative p-2 rounded-lg',
            emotionData.bgColor,
            className
          )}
        >
          <EmotionIcon className={cn('w-5 h-5', emotionData.color)} />
          
          {isAnalyzing && (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              className="absolute inset-0 border-2 border-accent-primary/30 border-t-accent-primary rounded-lg"
            />
          )}
        </motion.div>
      </Tooltip>
    );
  }
  
  // ============================================================================
  // DETAILED MODE
  // ============================================================================
  
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-bg-secondary rounded-xl border border-white/10 overflow-hidden',
        className
      )}
      role="status"
      aria-live="polite"
      aria-label={`Current emotion: ${emotionData.label}`}
    >
      {/* Header */}
      <div className={cn('p-4 border-b border-white/10', emotionData.bgColor)}>
        <div className="flex items-center gap-3">
          <div className={cn(
            'p-3 rounded-lg bg-white/10',
            isAnalyzing && 'animate-pulse'
          )}>
            <EmotionIcon className={cn('w-6 h-6', emotionData.color)} />
          </div>
          
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-text-primary">
              {emotionData.label}
            </h3>
            <p className="text-xs text-text-secondary">
              {emotionData.description}
            </p>
          </div>
          
          {isAnalyzing && (
            <Badge variant="primary" size="sm">
              Analyzing...
            </Badge>
          )}
        </div>
      </div>
      
      {/* Learning Readiness */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-text-secondary">
            Learning Readiness
          </span>
          <div className={cn('flex items-center gap-1', readinessData.color)}>
            <ReadinessIcon className="w-4 h-4" />
            <span className="text-sm font-semibold">
              {readinessData.label}
            </span>
          </div>
        </div>
        <p className="text-xs text-text-tertiary">
          {readinessData.description}
        </p>
      </div>
      
      {/* PAD Dimensions (if showDetails) */}
      {showDetails && (
        <div className="p-4 space-y-3">
          <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
            Emotional Dimensions
          </h4>
          
          <PADMeter
            label="Pleasure"
            value={emotion.valence}
            icon={Smile}
            description="How positive or negative the emotion is"
          />
          
          <PADMeter
            label="Arousal"
            value={emotion.arousal}
            icon={Activity}
            description="Energy level - calm to excited"
          />
          
          <PADMeter
            label="Dominance"
            value={emotion.dominance}
            icon={Brain}
            description="Sense of control over the situation"
          />
        </div>
      )}
      
      {/* Cognitive Load (if available) */}
      {emotion.cognitive_load !== undefined && showDetails && (
        <div className="p-4 border-t border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-text-secondary">
              Cognitive Load
            </span>
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3 text-accent-warning" />
              <span className="text-sm font-semibold text-text-primary">
                {Math.round(emotion.cognitive_load * 100)}%
              </span>
            </div>
          </div>
          
          {/* Load bar */}
          <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${emotion.cognitive_load * 100}%` }}
              transition={{ duration: 0.5 }}
              className={cn(
                'h-full',
                emotion.cognitive_load > 0.7
                  ? 'bg-accent-error'
                  : emotion.cognitive_load > 0.4
                  ? 'bg-accent-warning'
                  : 'bg-accent-success'
              )}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
};

EmotionIndicator.displayName = 'EmotionIndicator';

export default EmotionIndicator;
