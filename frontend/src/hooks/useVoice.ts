/**
 * useVoice Hook - Voice input/output operations
 * 
 * Features:
 * - Start/stop audio recording
 * - Transcription (Speech-to-Text)
 * - Text-to-speech synthesis
 * - Audio playback with state management
 * - Recording state tracking
 * - Error handling with user feedback
 * 
 * Audio Flow:
 * 1. User clicks microphone → startRecording()
 * 2. User speaks
 * 3. User clicks stop → stopRecording() → auto-transcribes
 * 4. Returns transcribed text
 * 
 * TTS Flow:
 * 1. Call speak(text, emotion)
 * 2. Backend generates audio
 * 3. Auto-plays audio
 * 4. Returns when complete
 * 
 * Performance:
 * - Microphone access: ~100ms
 * - Transcription: 1-3 seconds
 * - TTS: 0.5-2 seconds
 * - Audio playback: Real-time
 * 
 * Usage:
 * ```tsx
 * const { isRecording, startRecording, stopRecording, speak } = useVoice();
 * 
 * // Voice input
 * const handleVoiceInput = async () => {
 *   await startRecording();
 *   // ... user speaks ...
 *   const text = await stopRecording();
 *   console.log('Transcribed:', text);
 * };
 * 
 * // Voice output
 * await speak('Hello! How can I help you today?', 'friendly');
 * ```
 */

import { useState, useRef } from 'react';
import { voiceAPI } from '@/services/api/voice.api';
import { useUIStore } from '@/store/uiStore';

export interface UseVoiceReturn {
  /**
   * Recording state
   */
  isRecording: boolean;
  
  /**
   * Audio playback state
   */
  isPlaying: boolean;
  
  /**
   * Transcription processing state
   */
  isTranscribing: boolean;
  
  /**
   * Start recording audio
   */
  startRecording: () => Promise<void>;
  
  /**
   * Stop recording and transcribe
   * @returns Transcribed text or null if failed
   */
  stopRecording: () => Promise<string | null>;
  
  /**
   * Speak text using TTS
   * @param text - Text to speak
   * @param emotion - Optional emotion for voice selection
   */
  speak: (text: string, emotion?: string) => Promise<void>;
  
  /**
   * Stop current audio playback
   */
  stopSpeaking: () => void;
}

export const useVoice = (): UseVoiceReturn => {
  const { showToast } = useUIStore();
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  /**
   * Start recording audio
   */
  const startRecording = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error: any) {
      console.error('Microphone access error:', error);
      
      if (error.name === 'NotAllowedError') {
        showToast({
          type: 'error',
          message: 'Microphone access denied. Please allow microphone access in your browser settings.',
        });
      } else if (error.name === 'NotFoundError') {
        showToast({
          type: 'error',
          message: 'No microphone found. Please connect a microphone and try again.',
        });
      } else {
        showToast({
          type: 'error',
          message: 'Failed to access microphone.',
        });
      }
    }
  };

  /**
   * Stop recording and transcribe
   */
  const stopRecording = async (): Promise<string | null> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current) {
        resolve(null);
        return;
      }

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setIsRecording(false);
        
        // Stop all tracks
        mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());

        // Transcribe
        setIsTranscribing(true);
        try {
          const result = await voiceAPI.transcribe(audioBlob);
          setIsTranscribing(false);
          resolve(result.text);
        } catch (error: any) {
          console.error('Transcription error:', error);
          showToast({
            type: 'error',
            message: error.response?.data?.detail || 'Transcription failed. Please try again.',
          });
          setIsTranscribing(false);
          resolve(null);
        }
      };

      mediaRecorderRef.current.stop();
    });
  };

  /**
   * Speak text using TTS
   */
  const speak = async (text: string, emotion?: string): Promise<void> => {
    try {
      setIsPlaying(true);
      
      // Get audio blob from backend
      const audioBlob = await voiceAPI.synthesize(text, emotion);
      
      // Create audio URL
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Create and play audio
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl); // Clean up
      };
      
      audio.onerror = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl); // Clean up
        showToast({
          type: 'error',
          message: 'Audio playback failed',
        });
      };
      
      await audio.play();
    } catch (error: any) {
      console.error('TTS error:', error);
      showToast({
        type: 'error',
        message: error.response?.data?.detail || 'Text-to-speech failed. Please try again.',
      });
      setIsPlaying(false);
    }
  };

  /**
   * Stop audio playback
   */
  const stopSpeaking = (): void => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  return {
    // State
    isRecording,
    isPlaying,
    isTranscribing,
    
    // Actions
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  };
};
