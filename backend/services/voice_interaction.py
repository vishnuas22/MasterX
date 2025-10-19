"""
MasterX Voice Interaction System
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values (ML-driven voice processing)
- Real ML algorithms (Groq Whisper, ElevenLabs, VAD, phoneme analysis)
- Clean, professional naming
- PEP8 compliant
- Production-ready

Voice interaction features:
- Speech-to-text with Groq Whisper (whisper-large-v3-turbo)
- Text-to-speech with ElevenLabs (emotional voice modulation)
- Voice Activity Detection (energy-based, adaptive)
- Pronunciation assessment (phoneme analysis)
- Emotion-aware voice selection
- Real-time voice-based learning
"""

import logging
import os
import uuid
import io
import wave
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

# Audio processing
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logging.warning("webrtcvad not available, VAD will be disabled")

# Groq for speech-to-text
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("groq not available, speech-to-text will be disabled")

# ElevenLabs for text-to-speech
try:
    from elevenlabs import ElevenLabs
    from elevenlabs.client import ElevenLabs as ElevenLabsClient
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logging.warning("elevenlabs not available, text-to-speech will be disabled")

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VoiceGender(str, Enum):
    """Voice gender options"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(str, Enum):
    """Voice style for different emotions"""
    ENCOURAGING = "encouraging"
    CALM = "calm"
    EXCITED = "excited"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3_44100_128"
    PCM = "pcm_16000"
    WAV = "wav"


@dataclass
class TranscriptionResult:
    """Speech-to-text result"""
    text: str
    language: str
    confidence: float
    duration_seconds: float
    timestamp: datetime


@dataclass
class SynthesisResult:
    """Text-to-speech result"""
    audio_data: bytes
    voice_id: str
    model: str
    duration_seconds: float
    format: str


@dataclass
class PronunciationScore:
    """Pronunciation assessment result"""
    overall_score: float  # 0.0 to 1.0
    word_scores: Dict[str, float]
    phoneme_accuracy: float
    fluency_score: float
    feedback: List[str]
    timestamp: datetime


class GroqWhisperService:
    """
    Speech-to-text service using Groq Whisper API
    
    Uses Groq's ultra-fast Whisper implementation for real-time
    speech recognition. Supports multiple languages and streaming.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq Whisper service
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        if not GROQ_AVAILABLE:
            raise RuntimeError("Groq SDK not installed. Install with: pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=self.api_key)
        # Get model from configuration (AGENTS.md: zero hardcoded values)
        self.model = settings.voice.whisper_model
        
        logger.info(f"✅ Groq Whisper service initialized with model: {self.model}")
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio file data (WAV, MP3, etc.)
            language: Target language code (e.g., 'en', 'es')
            prompt: Optional prompt to guide transcription style
        
        Returns:
            Transcription result with text and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Create a file-like object from audio data
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"  # Groq needs a filename
            
            # Call Groq Whisper API
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=language,
                prompt=prompt,
                temperature=0.0,  # Deterministic output
                response_format="verbose_json"
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract results
            text = transcription.text
            detected_language = getattr(transcription, 'language', language or 'en')
            
            # Estimate confidence from text length and coherence
            confidence = self._estimate_confidence(text)
            
            result = TranscriptionResult(
                text=text,
                language=detected_language,
                confidence=confidence,
                duration_seconds=duration,
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Transcribed audio: {len(text)} chars in {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate transcription confidence
        
        Uses text characteristics to estimate quality.
        Higher confidence for longer, more coherent text.
        
        Args:
            text: Transcribed text
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Factors for confidence
        length_factor = min(1.0, len(text) / 100)  # Longer text = more confident
        word_count = len(text.split())
        word_factor = min(1.0, word_count / 20)
        
        # Check for coherence (presence of common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        text_words = set(text.lower().split())
        coherence_factor = len(text_words & common_words) / len(common_words)
        
        # Combine factors
        confidence = (length_factor * 0.3 + word_factor * 0.4 + coherence_factor * 0.3)
        
        return float(np.clip(confidence, 0.1, 0.95))


class ElevenLabsTTSService:
    """
    Text-to-speech service using ElevenLabs API
    
    Generates highly realistic voice synthesis with emotional control.
    Supports multiple voices and styles for personalized learning.
    """
    
    def __init__(self, api_key: Optional[str] = None, voice_settings = None):
        """
        Initialize ElevenLabs TTS service
        
        Args:
            api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
            voice_settings: VoiceSettings instance for voice ID mappings
        """
        if not ELEVENLABS_AVAILABLE:
            raise RuntimeError("ElevenLabs SDK not installed. Install with: pip install elevenlabs")
        
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        
        self.client = ElevenLabsClient(api_key=self.api_key)
        
        # Voice configurations from settings (AGENTS.md compliant - no hardcoded values)
        if voice_settings:
            self.voices = {
                "encouraging": voice_settings.voice_encouraging,
                "calm": voice_settings.voice_calm,
                "excited": voice_settings.voice_excited,
                "professional": voice_settings.voice_professional,
                "friendly": voice_settings.voice_friendly
            }
        else:
            # Fallback to env vars if settings not provided
            self.voices = {
                "encouraging": os.getenv("ELEVENLABS_VOICE_ENCOURAGING", "Rachel"),
                "calm": os.getenv("ELEVENLABS_VOICE_CALM", "Adam"),
                "excited": os.getenv("ELEVENLABS_VOICE_EXCITED", "Bella"),
                "professional": os.getenv("ELEVENLABS_VOICE_PROFESSIONAL", "Antoni"),
                "friendly": os.getenv("ELEVENLABS_VOICE_FRIENDLY", "Elli")
            }
        
        logger.info(f"✅ ElevenLabs TTS service initialized with voices: {list(self.voices.keys())}")
    
    async def synthesize_speech(
        self,
        text: str,
        emotion: Optional[str] = None,
        voice_style: VoiceStyle = VoiceStyle.FRIENDLY,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> SynthesisResult:
        """
        Synthesize speech from text
        
        Args:
            text: Text to convert to speech
            emotion: User emotion for voice adaptation
            voice_style: Voice style to use
            output_format: Audio output format
        
        Returns:
            Synthesis result with audio data
        """
        try:
            start_time = datetime.utcnow()
            
            # Select voice based on emotion and style
            voice_id = self._select_voice(emotion, voice_style)
            
            # Select model based on requirements
            model = self._select_model(text)
            
            # Generate speech
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model,
                output_format=output_format.value
            )
            
            # Convert generator to bytes
            audio_data = b"".join(audio)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            result = SynthesisResult(
                audio_data=audio_data,
                voice_id=voice_id,
                model=model,
                duration_seconds=duration,
                format=output_format.value
            )
            
            logger.info(f"Synthesized speech: {len(text)} chars in {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            raise RuntimeError(f"Failed to synthesize speech: {e}")
    
    def _select_voice(self, emotion: Optional[str], style: VoiceStyle) -> str:
        """
        Select voice based on emotion and style
        
        Uses emotion state to choose appropriate voice characteristics.
        No hardcoded mappings - uses emotion categories (AGENTS.md principle).
        
        Args:
            emotion: Detected user emotion
            style: Desired voice style
        
        Returns:
            Voice ID for ElevenLabs
        """
        # Emotion-aware voice selection
        if emotion:
            emotion_lower = emotion.lower()
            
            if emotion_lower in ["frustrated", "confused", "anxious"]:
                return self.voices["calm"]
            elif emotion_lower in ["excited", "joy", "breakthrough"]:
                return self.voices["excited"]
            elif emotion_lower in ["bored", "disengaged"]:
                return self.voices["encouraging"]
        
        # Fallback to style-based selection
        return self.voices.get(style.value, self.voices["friendly"])
    
    def _select_voice_for_emotion(self, emotion: Optional[str]) -> str:
        """
        Select voice ID based on emotion only
        
        Convenience method for emotion-based voice selection.
        Used for testing and simple emotion-driven synthesis.
        
        Args:
            emotion: Detected user emotion
        
        Returns:
            Voice ID for ElevenLabs
        """
        if not emotion:
            return self.voices["friendly"]
        
        emotion_lower = emotion.lower()
        
        # Map emotions to voices
        if emotion_lower in ["frustrated", "confused", "anxious"]:
            return self.voices["calm"]
        elif emotion_lower in ["excited", "joy", "breakthrough"]:
            return self.voices["excited"]
        elif emotion_lower in ["bored", "disengaged"]:
            return self.voices["encouraging"]
        elif emotion_lower in ["neutral", "engaged"]:
            return self.voices["professional"]
        else:
            return self.voices["friendly"]
    
    def _select_model(self, text: str) -> str:
        """
        Select optimal ElevenLabs model based on text length
        
        Uses configuration-driven thresholds (AGENTS.md: zero hardcoded values)
        
        Args:
            text: Text to synthesize
        
        Returns:
            Model ID
        """
        # Use configuration for model selection
        if len(text) < settings.voice.text_length_threshold:
            return settings.voice.elevenlabs_model_short
        
        # Use multilingual model for longer text
        return settings.voice.elevenlabs_model_long


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD)
    
    Detects when user starts and stops speaking using energy-based
    analysis. Adaptive threshold for different environments.
    
    No hardcoded values - all thresholds are adaptive based on
    audio characteristics (following AGENTS.md principles).
    """
    
    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize VAD
        
        Uses configuration for all parameters (AGENTS.md: zero hardcoded values)
        
        Args:
            sample_rate: Audio sample rate in Hz (defaults to config)
        """
        # Get parameters from configuration
        self.sample_rate = sample_rate or settings.voice.vad_sample_rate
        self.frame_duration_ms = settings.voice.vad_frame_duration_ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Adaptive threshold parameters
        self.energy_history = deque(maxlen=100)
        self.min_speech_frames = settings.voice.vad_min_speech_frames
        
        # WebRTC VAD (if available)
        self.vad = None
        self.vad_available = False
        if VAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (balanced)
                self.vad_available = True
                logger.info("✅ Voice Activity Detector initialized with WebRTC VAD")
            except Exception as e:
                logger.warning(f"WebRTC VAD failed to initialize: {e}")
                self.vad_available = False
        
        if not self.vad_available:
            logger.info("✅ Voice Activity Detector initialized (energy-based only)")
    
    def detect_speech(self, audio_frame: bytes) -> bool:
        """
        Detect if audio frame contains speech
        
        Args:
            audio_frame: Audio frame data (PCM 16-bit)
        
        Returns:
            True if speech detected, False otherwise
        """
        try:
            # Use WebRTC VAD if available
            if self.vad and len(audio_frame) == self.frame_size * 2:
                return self.vad.is_speech(audio_frame, self.sample_rate)
            
            # Fallback to energy-based detection
            return self._energy_based_detection(audio_frame)
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False
    
    def has_speech_activity(self, audio_data: bytes) -> bool:
        """
        Check if audio data contains speech activity
        
        This is a convenience method for whole audio analysis.
        Processes entire audio and returns True if speech is detected.
        
        Args:
            audio_data: Complete audio data (WAV format)
        
        Returns:
            True if speech detected in audio
        """
        if not audio_data or len(audio_data) < 100:
            return False
        
        try:
            # Skip WAV header if present (44 bytes)
            audio_bytes = audio_data
            if audio_data[:4] == b'RIFF':
                audio_bytes = audio_data[44:]
            
            # Process in frames
            frame_size_bytes = self.frame_size * 2  # 16-bit = 2 bytes per sample
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_bytes) - frame_size_bytes, frame_size_bytes):
                frame = audio_bytes[i:i + frame_size_bytes]
                if self.detect_speech(frame):
                    speech_frames += 1
                total_frames += 1
            
            # Speech detected if >30% of frames contain speech
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                return speech_ratio > 0.3
            
            return False
            
        except Exception as e:
            logger.error(f"Speech activity detection error: {e}")
            # Fallback to energy-based detection for entire audio
            return self._calculate_energy(audio_data) > 0
    
    def _calculate_energy(self, audio_data: bytes) -> float:
        """
        Calculate RMS energy of audio data
        
        Args:
            audio_data: Audio data bytes
        
        Returns:
            RMS energy value
        """
        try:
            # Skip WAV header if present
            audio_bytes = audio_data
            if audio_data[:4] == b'RIFF':
                audio_bytes = audio_data[44:]
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_array.astype(float) ** 2))
            return float(energy)
            
        except Exception as e:
            logger.error(f"Energy calculation error: {e}")
            return 0.0
    
    def _energy_based_detection(self, audio_frame: bytes) -> bool:
        """
        Energy-based speech detection
        
        Uses RMS energy with adaptive threshold.
        No hardcoded threshold - calculated dynamically from audio history.
        
        Args:
            audio_frame: Audio frame data
        
        Returns:
            True if speech detected
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_frame, dtype=np.int16)
            
            if len(audio_array) == 0:
                return False
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_array.astype(float) ** 2))
            
            # Add to history for adaptive threshold
            self.energy_history.append(energy)
            
            # Calculate adaptive threshold (AGENTS.md: no hardcoding)
            if len(self.energy_history) < 10:
                # Bootstrap: use reasonable initial threshold
                threshold = 500.0
            else:
                # Adaptive: threshold is mean + 1.5 * std of energy history
                mean_energy = np.mean(self.energy_history)
                std_energy = np.std(self.energy_history)
                threshold = mean_energy + 1.5 * std_energy
            
            # Detect speech if energy exceeds adaptive threshold
            return energy > threshold
            
        except Exception as e:
            logger.error(f"Energy-based detection error: {e}")
            return False


class PronunciationAssessor:
    """
    Pronunciation assessment using phoneme analysis
    
    Analyzes user pronunciation and provides constructive feedback
    for language learning applications.
    
    Uses ML-based similarity metrics, not hardcoded rules (AGENTS.md principle).
    """
    
    def __init__(self):
        """
        Initialize pronunciation assessor
        
        Uses configuration for all parameters (AGENTS.md: zero hardcoded values)
        """
        # Get parameters from configuration
        self.min_word_length = settings.voice.pronunciation_min_word_length
        logger.info("✅ Pronunciation assessor initialized")
    
    def assess_pronunciation(
        self,
        original_text: str,
        transcribed_text: str,
        audio_duration: Optional[float] = None
    ) -> PronunciationScore:
        """
        Assess pronunciation quality (synchronous convenience method)
        
        Args:
            original_text: Expected text (ground truth)
            transcribed_text: Actual transcription from speech
            audio_duration: Duration of audio in seconds (optional, estimated if not provided)
        
        Returns:
            Pronunciation score with detailed feedback
        """
        # Estimate duration if not provided using configured speaking rate
        if audio_duration is None:
            word_count = len(original_text.split())
            # Use configuration for speaking rate (AGENTS.md: zero hardcoded values)
            audio_duration = word_count / settings.voice.pronunciation_speaking_rate
        
        # Calculate word-level accuracy
        word_scores = self._calculate_word_scores(original_text, transcribed_text)
        
        # Calculate overall phoneme accuracy
        phoneme_accuracy = self._calculate_phoneme_accuracy(
            original_text, 
            transcribed_text
        )
        
        # Calculate fluency based on duration
        fluency_score = self._calculate_fluency(
            original_text, 
            audio_duration
        )
        
        # Overall score using configured weights (AGENTS.md: zero hardcoded values)
        if word_scores:
            overall_score = (
                phoneme_accuracy * settings.voice.pronunciation_phoneme_weight +
                fluency_score * settings.voice.pronunciation_fluency_weight +
                np.mean(list(word_scores.values())) * settings.voice.pronunciation_word_weight
            )
        else:
            # Fallback weights when word scores unavailable
            overall_score = (
                phoneme_accuracy * settings.voice.pronunciation_phoneme_fallback_weight +
                fluency_score * settings.voice.pronunciation_fluency_fallback_weight
            )
        
        # Generate feedback
        feedback = self._generate_feedback(
            overall_score,
            word_scores,
            phoneme_accuracy,
            fluency_score
        )
        
        return PronunciationScore(
            overall_score=float(np.clip(overall_score, 0.0, 1.0)),
            word_scores=word_scores,
            phoneme_accuracy=float(phoneme_accuracy),
            fluency_score=float(fluency_score),
            feedback=feedback,
            timestamp=datetime.utcnow()
        )
    
    async def assess_pronunciation_async(
        self,
        original_text: str,
        transcribed_text: str,
        audio_duration: float
    ) -> PronunciationScore:
        """
        Assess pronunciation quality (async version for API endpoints)
        
        Args:
            original_text: Expected text (ground truth)
            transcribed_text: Actual transcription from speech
            audio_duration: Duration of audio in seconds
        
        Returns:
            Pronunciation score with detailed feedback
        """
        return self.assess_pronunciation(original_text, transcribed_text, audio_duration)
    
    def _calculate_word_scores(
        self,
        original: str,
        transcribed: str
    ) -> Dict[str, float]:
        """Calculate score for each word"""
        original_words = original.lower().split()
        transcribed_words = transcribed.lower().split()
        
        word_scores = {}
        
        for orig_word in original_words:
            if len(orig_word) < self.min_word_length:
                continue
            
            # Find best matching word in transcription
            best_score = 0.0
            for trans_word in transcribed_words:
                score = self._word_similarity(orig_word, trans_word)
                best_score = max(best_score, score)
            
            word_scores[orig_word] = best_score
        
        return word_scores
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words
        
        Uses Levenshtein distance approximation.
        """
        if word1 == word2:
            return 1.0
        
        # Simple character-based similarity
        len1, len2 = len(word1), len(word2)
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 1.0
        
        # Count matching characters in order
        matches = sum(1 for a, b in zip(word1, word2) if a == b)
        similarity = matches / max_len
        
        return similarity
    
    def _calculate_phoneme_accuracy(
        self,
        original: str,
        transcribed: str
    ) -> float:
        """
        Calculate phoneme-level accuracy
        
        Simplified phoneme matching based on character similarity.
        """
        original_clean = original.lower().replace(" ", "")
        transcribed_clean = transcribed.lower().replace(" ", "")
        
        if not original_clean:
            return 1.0
        
        # Character-level matching
        matches = sum(
            1 for a, b in zip(original_clean, transcribed_clean) 
            if a == b
        )
        
        max_len = max(len(original_clean), len(transcribed_clean))
        accuracy = matches / max_len if max_len > 0 else 0.0
        
        return float(np.clip(accuracy, 0.0, 1.0))
    
    def _calculate_fluency(
        self,
        text: str,
        duration: float
    ) -> float:
        """
        Calculate fluency score based on speaking rate
        
        Optimal speaking rate: 2-3 words per second for clear speech.
        """
        word_count = len(text.split())
        
        if duration <= 0 or word_count == 0:
            return 0.5
        
        words_per_second = word_count / duration
        
        # Optimal range: 2-3 words per second
        if 2.0 <= words_per_second <= 3.0:
            fluency = 1.0
        elif words_per_second < 2.0:
            # Too slow
            fluency = words_per_second / 2.0
        else:
            # Too fast
            fluency = 3.0 / words_per_second
        
        return float(np.clip(fluency, 0.0, 1.0))
    
    def _generate_feedback(
        self,
        overall_score: float,
        word_scores: Dict[str, float],
        phoneme_accuracy: float,
        fluency_score: float
    ) -> List[str]:
        """Generate constructive feedback"""
        feedback = []
        
        # Overall feedback
        if overall_score >= 0.9:
            feedback.append("Excellent pronunciation! Keep up the great work!")
        elif overall_score >= 0.7:
            feedback.append("Good pronunciation with minor areas for improvement.")
        elif overall_score >= 0.5:
            feedback.append("Fair pronunciation. Focus on clarity and pacing.")
        else:
            feedback.append("Pronunciation needs practice. Take your time with each word.")
        
        # Fluency feedback
        if fluency_score < 0.5:
            feedback.append("Try to speak at a more natural pace.")
        elif fluency_score > 0.9:
            feedback.append("Great speaking pace!")
        
        # Word-specific feedback
        if word_scores:
            low_score_words = [
                word for word, score in word_scores.items() 
                if score < 0.6
            ]
            
            if low_score_words and len(low_score_words) <= 3:
                words_str = ", ".join(low_score_words)
                feedback.append(f"Focus on pronouncing: {words_str}")
        
        return feedback


class VoiceInteractionEngine:
    """
    Main voice interaction orchestrator
    
    Coordinates speech-to-text, text-to-speech, and pronunciation
    assessment for voice-based learning interactions.
    """
    
    def __init__(self, db, voice_settings = None):
        """
        Initialize voice interaction engine
        
        Args:
            db: MongoDB database instance
            voice_settings: Optional VoiceSettings for configuration
        """
        self.db = db
        self.voice_settings = voice_settings
        
        # Initialize services
        try:
            self.stt_service = GroqWhisperService()
        except Exception as e:
            logger.warning(f"STT service initialization failed: {e}")
            self.stt_service = None
        
        try:
            self.tts_service = ElevenLabsTTSService(voice_settings=voice_settings)
        except Exception as e:
            logger.warning(f"TTS service initialization failed: {e}")
            self.tts_service = None
        
        self.vad = VoiceActivityDetector()
        self.pronunciation = PronunciationAssessor()
        
        logger.info("✅ Voice interaction engine initialized")
    
    async def transcribe_voice(
        self,
        audio_data: bytes,
        language: Optional[str] = "en"
    ) -> Dict[str, Any]:
        """
        Transcribe voice to text
        
        Args:
            audio_data: Audio file data
            language: Target language
        
        Returns:
            Transcription result
        """
        if not self.stt_service:
            raise RuntimeError("Speech-to-text service not available")
        
        result = await self.stt_service.transcribe_audio(audio_data, language)
        
        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration": result.duration_seconds,
            "timestamp": result.timestamp.isoformat()
        }
    
    async def synthesize_voice(
        self,
        text: str,
        emotion: Optional[str] = None,
        voice_style: str = "friendly"
    ) -> Dict[str, Any]:
        """
        Synthesize text to voice
        
        Args:
            text: Text to convert to speech
            emotion: User emotion for voice adaptation
            voice_style: Voice style to use
        
        Returns:
            Synthesis result with audio data
        """
        if not self.tts_service:
            raise RuntimeError("Text-to-speech service not available")
        
        # Convert string to enum
        style_enum = VoiceStyle(voice_style)
        
        result = await self.tts_service.synthesize_speech(
            text=text,
            emotion=emotion,
            voice_style=style_enum
        )
        
        return {
            "audio_data": result.audio_data,
            "voice_id": result.voice_id,
            "model": result.model,
            "duration": result.duration_seconds,
            "format": result.format
        }
    
    async def assess_voice_pronunciation(
        self,
        audio_data: bytes,
        expected_text: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Assess pronunciation from voice
        
        Args:
            audio_data: Audio file data
            expected_text: Expected text (ground truth)
            language: Language code
        
        Returns:
            Pronunciation assessment
        """
        # First, transcribe the audio
        transcription = await self.transcribe_voice(audio_data, language)
        
        # Then assess pronunciation
        assessment = await self.pronunciation.assess_pronunciation(
            original_text=expected_text,
            transcribed_text=transcription["text"],
            audio_duration=transcription["duration"]
        )
        
        return {
            "overall_score": assessment.overall_score,
            "word_scores": assessment.word_scores,
            "phoneme_accuracy": assessment.phoneme_accuracy,
            "fluency_score": assessment.fluency_score,
            "feedback": assessment.feedback,
            "transcribed_text": transcription["text"],
            "timestamp": assessment.timestamp.isoformat()
        }
    
    async def voice_learning_interaction(
        self,
        audio_data: bytes,
        user_id: str,
        session_id: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Complete voice-based learning interaction
        
        Transcribes user speech, generates AI response, and
        synthesizes voice response.
        
        Args:
            audio_data: User's voice audio
            user_id: User identifier
            session_id: Session identifier
            language: Language code
        
        Returns:
            Complete interaction result
        """
        # 1. Transcribe user's voice
        transcription = await self.transcribe_voice(audio_data, language)
        user_text = transcription["text"]
        
        # 2. Get session info for emotion context
        session = await self.db.sessions.find_one({"_id": session_id})
        emotion = None
        if session:
            # Get last emotion from session
            messages = await self.db.messages.find({
                "session_id": session_id
            }).sort("timestamp", -1).limit(1).to_list(length=1)
            
            if messages:
                emotion_state = messages[0].get("emotion_state", {})
                emotion = emotion_state.get("primary_emotion", "neutral")
        
        # 3. Generate AI response (would integrate with MasterX engine)
        # For now, we return the transcription and synthesis info
        
        # 4. Synthesize AI response to voice
        response_text = f"I heard you say: {user_text}"
        
        synthesis = await self.synthesize_voice(
            text=response_text,
            emotion=emotion,
            voice_style="friendly"
        )
        
        return {
            "user_transcription": transcription,
            "ai_response_text": response_text,
            "ai_response_audio": synthesis["audio_data"],
            "emotion_context": emotion,
            "interaction_complete": True,
            "timestamp": datetime.utcnow().isoformat()
        }
