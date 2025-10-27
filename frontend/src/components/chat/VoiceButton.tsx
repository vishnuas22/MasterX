/**
 * VoiceButton Component - Speech-to-Text Input
 * 
 * WCAG 2.1 AA Compliant:
 * - Keyboard accessible (Space to toggle)
 * - Clear visual feedback (recording state)
 * - Error announcements to screen readers
 * - Alternative text input always available
 * 
 * Performance:
 * - Efficient audio processing
 * - Chunked uploads for large files
 * - Background transcription
 * 
 * Backend Integration:
 * - POST /api/v1/voice/transcribe
 * - Groq Whisper model (fast, accurate)
 * - Supports 50+ languages
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Mic, MicOff, Loader2, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/utils/cn';
import { Tooltip } from '@/components/ui/Tooltip';
import { useVoice } from '@/hooks/useVoice';
import { toast } from '@/components/ui/Toast';

// ============================================================================
// TYPES
// ============================================================================

export interface VoiceButtonProps {
  /**
   * Callback when transcription is complete
   */
  onTranscription: (text: string) => void;
  
  /**
   * Is recording disabled
   */
  disabled?: boolean;
  
  /**
   * Language code
   * @default "en"
   */
  language?: string;
  
  /**
   * Maximum recording duration (seconds)
   * @default 60
   */
  maxDuration?: number;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

// ============================================================================
// AUDIO VISUALIZER COMPONENT
// ============================================================================

const AudioVisualizer = React.memo<{ isRecording: boolean; audioLevel: number }>(
  ({ isRecording, audioLevel }) => {
    if (!isRecording) return null;
    
    return (
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0 }}
        className="absolute inset-0 flex items-center justify-center"
      >
        <motion.div
          animate={{
            scale: [1, 1 + audioLevel * 0.5, 1],
            opacity: [0.3, 0.6, 0.3]
          }}
          transition={{
            duration: 0.5,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
          className="absolute w-20 h-20 bg-accent-error rounded-full"
        />
      </motion.div>
    );
  }
);

AudioVisualizer.displayName = 'AudioVisualizer';

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const VoiceButton: React.FC<VoiceButtonProps> = ({
  onTranscription,
  disabled = false,
  language = 'en',
  maxDuration = 60,
  className
}) => {
  // ============================================================================
  // STATE
  // ============================================================================
  
  const [recordingDuration, setRecordingDuration] = useState(0);
  const timerRef = useRef<NodeJS.Timeout>();
  
  // ============================================================================
  // VOICE HOOK
  // ============================================================================
  
  const {
    isRecording,
    isTranscribing,
    audioLevel,
    error,
    startRecording,
    stopRecording,
    cancelRecording
  } = useVoice({
    language,
    onTranscription: useCallback((text: string) => {
      if (text.trim()) {
        onTranscription(text);
        toast({
          title: 'Transcription Complete',
          description: text.slice(0, 50) + (text.length > 50 ? '...' : ''),
          variant: 'success'
        });
      }
    }, [onTranscription]),
    onError: useCallback((err: Error) => {
      toast({
        title: 'Voice Input Error',
        description: err.message,
        variant: 'error'
      });
    }, [])
  });
  
  // ============================================================================
  // RECORDING TIMER
  // ============================================================================
  
  useEffect(() => {
    if (isRecording) {
      setRecordingDuration(0);
      timerRef.current = setInterval(() => {
        setRecordingDuration(prev => {
          const next = prev + 1;
          
          // Auto-stop at max duration
          if (next >= maxDuration) {
            stopRecording();
            return maxDuration;
          }
          
          return next;
        });
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      setRecordingDuration(0);
    }
    
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isRecording, maxDuration, stopRecording]);
  
  // ============================================================================
  // HANDLERS
  // ============================================================================
  
  const handleToggle = useCallback(() => {
    if (disabled) return;
    
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [disabled, isRecording, startRecording, stopRecording]);
  
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.code === 'Space' && !disabled) {
      e.preventDefault();
      handleToggle();
    }
  }, [disabled, handleToggle]);
  
  // ============================================================================
  // RENDER
  // ============================================================================
  
  return (
    <div className={cn('relative', className)}>
      {/* Main Button */}
      <Tooltip content={
        disabled
          ? 'Voice input disabled'
          : isRecording
          ? 'Stop recording (Space)'
          : isTranscribing
          ? 'Transcribing audio...'
          : 'Start voice input (Space)'
      }>
        <motion.button
          onClick={handleToggle}
          onKeyDown={handleKeyDown}
          disabled={disabled || isTranscribing}
          whileTap={{ scale: 0.95 }}
          className={cn(
            'relative p-3 rounded-full transition-all',
            'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2',
            isRecording
              ? 'bg-accent-error text-white animate-pulse'
              : isTranscribing
              ? 'bg-accent-primary text-white'
              : 'bg-bg-tertiary text-text-secondary hover:bg-bg-tertiary/80',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
          aria-label={
            isRecording
              ? `Recording... ${recordingDuration} seconds`
              : 'Start voice input'
          }
          aria-pressed={isRecording}
        >
          {/* Audio Visualizer */}
          <AnimatePresence>
            {isRecording && (
              <AudioVisualizer
                isRecording={isRecording}
                audioLevel={audioLevel}
              />
            )}
          </AnimatePresence>
          
          {/* Icon */}
          {isTranscribing ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : isRecording ? (
            <MicOff className="w-5 h-5" />
          ) : error ? (
            <AlertCircle className="w-5 h-5 text-accent-error" />
          ) : (
            <Mic className="w-5 h-5" />
          )}
        </motion.button>
      </Tooltip>
      
      {/* Recording Timer */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap"
          >
            <div className="px-2 py-1 bg-accent-error text-white text-xs font-semibold rounded-full">
              {Math.floor(recordingDuration / 60)}:{(recordingDuration % 60).toString().padStart(2, '0')}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

VoiceButton.displayName = 'VoiceButton';

export default VoiceButton;
