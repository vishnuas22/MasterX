/**
 * Voice API Service
 * 
 * Handles voice input (STT) and output (TTS):
 * - Speech-to-text transcription (Groq Whisper)
 * - Text-to-speech synthesis (ElevenLabs)
 * - Pronunciation assessment with feedback
 * - Voice-based learning interactions
 * 
 * Backend Integration:
 * POST /api/v1/voice/transcribe            - Audio to text (Whisper)
 * POST /api/v1/voice/synthesize            - Text to speech (ElevenLabs)
 * POST /api/v1/voice/assess-pronunciation  - Pronunciation feedback
 * POST /api/v1/voice/chat                  - Complete voice interaction
 * 
 * Audio Format Requirements:
 * - Input: webm, mp3, wav, m4a (browser-dependent)
 * - Output: audio/mpeg (mp3)
 * - Recommended: webm (best compression)
 * 
 * @module services/api/voice.api
 */

import apiClient from './client';

/**
 * Transcription Response
 */
export interface TranscriptionResponse {
  text: string;
  language: string;
  duration: number;
  confidence: number;
}

/**
 * Pronunciation Assessment Response
 */
export interface PronunciationAssessment {
  overall_score: number;
  accuracy_score: number;
  fluency_score: number;
  completeness_score: number;
  pronunciation_score: number;
  transcribed_text: string;
  expected_text: string;
  word_level_feedback: Array<{
    word: string;
    accuracy: number;
    error_type?: string;
  }>;
  phoneme_analysis: Array<{
    phoneme: string;
    accuracy: number;
  }>;
  suggestions: string[];
}

/**
 * Voice Chat Response
 */
export interface VoiceChatResponse {
  transcription: TranscriptionResponse;
  chat_response: {
    message: string;
    emotion_state: any;
    session_id: string;
  };
  audio_response: {
    audio_url: string;
    voice_id: string;
    duration: number;
  };
}

/**
 * Voice API endpoints
 */
export const voiceAPI = {
  /**
   * Transcribe audio to text
   * 
   * Uses Groq Whisper (whisper-large-v3-turbo) for fast, accurate transcription.
   * Supports 99+ languages with automatic detection.
   * 
   * Performance: 200-1250x faster than real-time
   * 
   * @param audioBlob - Audio data (webm, mp3, wav, m4a)
   * @param language - Optional: language code (default: 'en')
   * @returns Transcription with confidence score
   * @throws 400 - No audio file provided
   * @throws 503 - Voice service unavailable
   * @throws 500 - Transcription failed
   * 
   * @example
   * ```typescript
   * const audioBlob = await mediaRecorder.requestData();
   * const result = await voiceAPI.transcribe(audioBlob);
   * console.log(result.text); // "I need help with calculus"
   * console.log(result.confidence); // 0.95
   * ```
   */
  transcribe: async (
    audioBlob: Blob,
    language: string = 'en'
  ): Promise<TranscriptionResponse> => {
    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'audio.webm');

    const { data } = await apiClient.post<TranscriptionResponse>(
      `/api/v1/voice/transcribe?language=${language}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds for large audio files
      }
    );
    return data;
  },

  /**
   * Synthesize text to speech
   * 
   * Uses ElevenLabs TTS with emotion-aware voice selection.
   * Returns high-quality audio with natural intonation.
   * 
   * Voices are selected based on emotion:
   * - encouraging: Warm, supportive female voice
   * - calm: Deep, reassuring male voice
   * - excited: Energetic, enthusiastic voice
   * - professional: Clear, authoritative voice
   * - friendly: Casual, approachable voice
   * 
   * @param text - Text to synthesize (max ~5000 chars)
   * @param emotion - Optional: emotion for voice selection
   * @param voiceStyle - Optional: specific voice style
   * @returns Audio blob (mp3 format)
   * @throws 503 - Voice service unavailable
   * @throws 500 - Synthesis failed
   * 
   * @example
   * ```typescript
   * const audio = await voiceAPI.synthesize(
   *   "Great job! Let's try another problem.",
   *   "encouraging"
   * );
   * 
   * // Play audio
   * const audioUrl = URL.createObjectURL(audio);
   * const audioElement = new Audio(audioUrl);
   * audioElement.play();
   * ```
   */
  synthesize: async (
    text: string,
    emotion?: string,
    voiceStyle?: string
  ): Promise<Blob> => {
    const response = await apiClient.post(
      '/api/v1/voice/synthesize',
      {
        text,
        emotion: emotion || 'neutral',
        voice_style: voiceStyle,
      },
      {
        responseType: 'blob', // Important: get binary data
        timeout: 30000,
      }
    );
    return response.data;
  },

  /**
   * Assess pronunciation quality
   * 
   * Compares user's pronunciation with expected text.
   * Provides detailed feedback on accuracy, fluency, and completeness.
   * 
   * Uses ML-based phoneme analysis for word-level feedback.
   * 
   * @param audioBlob - User's audio pronunciation
   * @param expectedText - The text they should be pronouncing
   * @param language - Optional: language code (default: 'en')
   * @returns Detailed pronunciation assessment
   * @throws 400 - Missing audio or expected text
   * @throws 503 - Voice service unavailable
   * @throws 500 - Assessment failed
   * 
   * @example
   * ```typescript
   * const assessment = await voiceAPI.assessPronunciation(
   *   audioBlob,
   *   "The derivative of x squared is 2x"
   * );
   * 
   * console.log(assessment.overall_score); // 0.85
   * console.log(assessment.word_level_feedback);
   * // [{ word: "derivative", accuracy: 0.92 }, ...]
   * 
   * console.log(assessment.suggestions);
   * // ["Pay attention to the 'th' sound in 'the'"]
   * ```
   */
  assessPronunciation: async (
    audioBlob: Blob,
    expectedText: string,
    language: string = 'en'
  ): Promise<PronunciationAssessment> => {
    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'audio.webm');

    const { data } = await apiClient.post<PronunciationAssessment>(
      `/api/v1/voice/assess-pronunciation?expected_text=${encodeURIComponent(expectedText)}&language=${language}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      }
    );
    return data;
  },

  /**
   * Complete voice-based learning interaction
   * 
   * Full pipeline:
   * 1. Transcribe user's spoken question
   * 2. Process through MasterX engine (emotion detection, AI response)
   * 3. Synthesize AI response to voice
   * 4. Return transcription, text response, and audio
   * 
   * This is the main endpoint for voice-only learning experiences.
   * 
   * @param audioBlob - User's spoken question
   * @param userId - User identifier
   * @param sessionId - Session identifier
   * @param language - Optional: language code (default: 'en')
   * @returns Complete interaction with text and audio
   * @throws 400 - Missing required parameters
   * @throws 503 - Voice service unavailable
   * @throws 500 - Voice interaction failed
   * 
   * @example
   * ```typescript
   * const result = await voiceAPI.voiceChat(
   *   audioBlob,
   *   'user-123',
   *   'session-abc'
   * );
   * 
   * console.log(result.transcription.text);
   * // "I'm confused about derivatives"
   * 
   * console.log(result.chat_response.message);
   * // "Let me help you understand derivatives..."
   * 
   * // Play AI's voice response
   * const audio = new Audio(result.audio_response.audio_url);
   * audio.play();
   * ```
   */
  voiceChat: async (
    audioBlob: Blob,
    userId: string,
    sessionId: string,
    language: string = 'en'
  ): Promise<VoiceChatResponse> => {
    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'audio.webm');

    const { data } = await apiClient.post<VoiceChatResponse>(
      `/api/v1/voice/chat?user_id=${userId}&session_id=${sessionId}&language=${language}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // Voice processing can take time
      }
    );
    return data;
  },
};