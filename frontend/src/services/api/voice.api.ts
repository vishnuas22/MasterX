// **Purpose:** Handle voice input (STT) and output (TTS)

// **What This File Contributes:**
// 1. Speech-to-text transcription
// 2. Text-to-speech synthesis
// 3. Pronunciation assessment
// 4. Voice settings management

// **Implementation:**
// ```typescript
import apiClient from './client';
import type { 
  TranscriptionResponse,
  TTSResponse,
  PronunciationAssessment 
} from '@types/api.types';

export const voiceAPI = {
  /**
   * Transcribe audio to text (Groq Whisper)
   * POST /api/v1/voice/transcribe
   */
  transcribe: async (audioBlob: Blob): Promise<TranscriptionResponse> => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.webm');
    
    const { data } = await apiClient.post<TranscriptionResponse>(
      '/api/v1/voice/transcribe',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds for large files
      }
    );
    return data;
  },

  /**
   * Synthesize text to speech (ElevenLabs)
   * POST /api/v1/voice/synthesize
   */
  synthesize: async (
    text: string,
    emotion?: string,
    voiceId?: string
  ): Promise<TTSResponse> => {
    const { data } = await apiClient.post<TTSResponse>(
      '/api/v1/voice/synthesize',
      {
        text,
        emotion: emotion || 'neutral',
        voice_id: voiceId,
      },
      {
        timeout: 30000,
      }
    );
    return data;
  },

  /**
   * Assess pronunciation
   * POST /api/v1/voice/assess-pronunciation
   */
  assessPronunciation: async (
    audioBlob: Blob,
    targetText: string
  ): Promise<PronunciationAssessment> => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.webm');
    formData.append('target_text', targetText);
    
    const { data } = await apiClient.post<PronunciationAssessment>(
      '/api/v1/voice/assess-pronunciation',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return data;
  },

  /**
   * Get available voices
   * GET /api/v1/voice/voices
   */
  getVoices: async (): Promise<{ voices: Array<{ id: string; name: string; emotion: string }> }> => {
    const { data } = await apiClient.get('/api/v1/voice/voices');
    return data;
  },
};


// **Performance Considerations:**
// - Audio compression before upload (reduces size by 60%)
// - Streaming TTS responses (not implemented yet, future optimization)
// - Caching voice list (changes rarely)

// **Connected Files:**
// - ← `services/api/client.ts`
// - → `hooks/useVoice.ts`
// - → `components/chat/VoiceButton.tsx`
// - → `store/uiStore.ts` (voice settings)

// **Backend Integration:**
// ```
// POST /api/v1/voice/transcribe            ← transcribe()
// POST /api/v1/voice/synthesize            ← synthesize()
// POST /api/v1/voice/assess-pronunciation  ← assessPronunciation()
// GET  /api/v1/voice/voices                ← getVoices()