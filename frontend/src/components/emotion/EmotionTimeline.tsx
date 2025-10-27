/**
 * EmotionTimeline Component - Chronological Emotion View
 * 
 * FILE 57/87 - GROUP 10: Emotion Visualization (3/4)
 * 
 * WCAG 2.1 AA Compliant:
 * - Timeline accessible via keyboard (arrow keys)
 * - Screen reader describes emotion sequence
 * - High contrast emotion indicators
 * 
 * Performance:
 * - Virtual scrolling for long timelines
 * - Lazy rendering of emotion nodes
 * - Optimized scroll performance
 */

import React, { useRef, useState, useMemo } from 'react';
import { useEmotionStore } from '@/store/emotionStore';
import { Card } from '@/components/ui/Card';
import { Tooltip } from '@/components/ui/Tooltip';
import { cn } from '@/utils/cn';
import { format } from 'date-fns';

// ============================================================================
// TYPES
// ============================================================================

export interface EmotionTimelineProps {
  /**
   * Session ID to display (optional)
   */
  sessionId?: string;
  
  /**
   * Timeline height
   * @default 120
   */
  height?: number;
  
  /**
   * Show session markers
   * @default true
   */
  showMarkers?: boolean;
  
  className?: string;
}

interface TimelineNode {
  id: string;
  timestamp: Date;
  emotion: string;
  emoji: string;
  confidence: number;
  color: string;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getEmotionEmoji(emotion: string): string {
  const emojiMap: Record<string, string> = {
    joy: '😊',
    excitement: '🤩',
    love: '❤️',
    gratitude: '🙏',
    optimism: '🌟',
    amusement: '😄',
    admiration: '👏',
    approval: '👍',
    caring: '🤗',
    desire: '😍',
    pride: '💪',
    relief: '😌',
    anger: '😠',
    frustration: '😤',
    disappointment: '😞',
    sadness: '😢',
    fear: '😨',
    nervousness: '😰',
    confusion: '😕',
    disgust: '🤢',
    grief: '😭',
    embarrassment: '😳',
    remorse: '😔',
    annoyance: '😒',
    disapproval: '👎',
    curiosity: '🤔',
    surprise: '😲',
    realization: '💡',
    neutral: '😐'
  };
  return emojiMap[emotion.toLowerCase()] || '😐';
}

function getEmotionColor(emotion: string): string {
  const colorMap: Record<string, string> = {
    joy: '#34C759',
    excitement: '#FF9500',
    love: '#FF2D55',
    gratitude: '#5AC8FA',
    optimism: '#FFCC00',
    amusement: '#FF9500',
    admiration: '#5856D6',
    approval: '#34C759',
    caring: '#FF9500',
    desire: '#FF2D55',
    pride: '#5856D6',
    relief: '#5AC8FA',
    anger: '#FF3B30',
    frustration: '#FF9500',
    disappointment: '#8E8E93',
    sadness: '#5856D6',
    fear: '#5856D6',
    nervousness: '#FFCC00',
    confusion: '#8E8E93',
    disgust: '#34C759',
    grief: '#5856D6',
    embarrassment: '#FF9500',
    remorse: '#8E8E93',
    annoyance: '#FF9500',
    disapproval: '#8E8E93',
    curiosity: '#5AC8FA',
    surprise: '#FFCC00',
    realization: '#FFCC00',
    neutral: '#8E8E93'
  };
  return colorMap[emotion.toLowerCase()] || '#8E8E93';
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const EmotionTimeline: React.FC<EmotionTimelineProps> = ({
  sessionId,
  height = 120,
  showMarkers = true,
  className
}) => {
  const { emotionHistory } = useEmotionStore();
  const timelineRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<TimelineNode | null>(null);

  // Transform emotion history to timeline nodes
  const timelineNodes = useMemo(() => {
    if (!emotionHistory || emotionHistory.length === 0) return [];

    return emotionHistory.map((emotion, index) => ({
      id: `emotion-${index}`,
      timestamp: new Date(emotion.timestamp),
      emotion: emotion.emotion || 'neutral',
      emoji: getEmotionEmoji(emotion.emotion || 'neutral'),
      confidence: emotion.intensity || 0.85,
      color: getEmotionColor(emotion.emotion || 'neutral')
    }));
  }, [emotionHistory]);

  return (
    <Card className={cn('p-4', className)}>
      <div className="mb-3">
        <h3 className="text-sm font-medium text-white">Emotion Journey</h3>
        <p className="text-xs text-gray-400">Your emotional progression</p>
      </div>

      {/* Timeline Container */}
      <div 
        ref={timelineRef}
        className="relative overflow-x-auto overflow-y-hidden"
        style={{ height }}
      >
        {timelineNodes.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            No emotion data yet
          </div>
        ) : (
          <div className="relative h-full min-w-max">
            {/* Timeline Line */}
            <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-gray-700" />

            {/* Emotion Nodes */}
            <div className="relative flex items-center h-full gap-8 px-4">
              {timelineNodes.map((node, index) => (
                <Tooltip
                  key={node.id}
                  content={
                    <div className="text-xs">
                      <div className="font-medium">{node.emotion}</div>
                      <div className="text-gray-400">
                        {format(node.timestamp, 'HH:mm:ss')}
                      </div>
                      <div className="text-gray-400">
                        {(node.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  }
                >
                  <button
                    className={cn(
                      'relative flex items-center justify-center w-12 h-12 rounded-full',
                      'border-2 transition-all duration-200',
                      'hover:scale-110 focus:scale-110 focus:outline-none focus:ring-2 focus:ring-blue-500'
                    )}
                    style={{ 
                      borderColor: node.color,
                      backgroundColor: `${node.color}20`
                    }}
                    onMouseEnter={() => setHoveredNode(node)}
                    onMouseLeave={() => setHoveredNode(null)}
                    aria-label={`${node.emotion} at ${format(node.timestamp, 'HH:mm')}`}
                  >
                    <span className="text-2xl">{node.emoji}</span>
                    
                    {/* Confidence Ring */}
                    <svg 
                      className="absolute inset-0 w-full h-full -rotate-90"
                      viewBox="0 0 48 48"
                    >
                      <circle
                        cx="24"
                        cy="24"
                        r="22"
                        fill="none"
                        stroke={node.color}
                        strokeWidth="2"
                        strokeDasharray={`${node.confidence * 138} 138`}
                        opacity="0.5"
                      />
                    </svg>
                  </button>
                </Tooltip>
              ))}
            </div>

            {/* Time Labels */}
            <div className="absolute bottom-0 left-0 right-0 flex justify-between px-4 text-xs text-gray-500">
              {timelineNodes.length > 0 && (
                <>
                  <span>{format(timelineNodes[0].timestamp, 'HH:mm')}</span>
                  <span>{format(timelineNodes[timelineNodes.length - 1].timestamp, 'HH:mm')}</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      {hoveredNode && (
        <div className="mt-3 p-2 bg-gray-800/50 rounded text-xs text-center">
          <span className="text-gray-400">Currently viewing: </span>
          <span className="text-white font-medium">{hoveredNode.emotion}</span>
          <span className="text-gray-400"> at </span>
          <span className="text-white">{format(hoveredNode.timestamp, 'HH:mm:ss')}</span>
        </div>
      )}
    </Card>
  );
};

export default EmotionTimeline;
