import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Settings, 
  Waveform,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Headphones,
  Radio
} from 'lucide-react';
import { GlassCard, GlassButton } from './GlassCard';
import { useApp } from '../context/AppContext';

// Voice Interface Hook
export function useVoiceInterface() {
  const { state, actions } = useApp();
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState({
    enabled: false,
    autoSpeak: true,
    voice: null,
    rate: 1.0,
    pitch: 1.0,
    volume: 0.8,
    language: 'en-US',
    wakeWord: 'hey master',
    voiceCommands: true,
    speechToText: true,
    textToSpeech: true
  });
  
  const recognition = useRef(null);
  const synthesis = useRef(null);
  const audioContext = useRef(null);
  const analyser = useRef(null);
  const dataArray = useRef(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isWaitingForWakeWord, setIsWaitingForWakeWord] = useState(false);

  // Initialize Speech APIs
  useEffect(() => {
    // Initialize Speech Recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition.current = new SpeechRecognition();
      recognition.current.continuous = true;
      recognition.current.interimResults = true;
      recognition.current.lang = voiceSettings.language;
    }

    // Initialize Speech Synthesis
    if ('speechSynthesis' in window) {
      synthesis.current = window.speechSynthesis;
    }

    // Initialize Audio Context for visualization
    if ('AudioContext' in window || 'webkitAudioContext' in window) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContext.current = new AudioContext();
    }

    return () => {
      if (recognition.current) {
        recognition.current.stop();
      }
      if (synthesis.current) {
        synthesis.current.cancel();
      }
      if (audioContext.current) {
        audioContext.current.close();
      }
    };
  }, []);

  // Setup audio visualization
  const setupAudioVisualization = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioContext.current.createMediaStreamSource(stream);
      analyser.current = audioContext.current.createAnalyser();
      analyser.current.fftSize = 256;
      
      const bufferLength = analyser.current.frequencyBinCount;
      dataArray.current = new Uint8Array(bufferLength);
      
      source.connect(analyser.current);
      
      const updateAudioLevel = () => {
        if (analyser.current && dataArray.current) {
          analyser.current.getByteFrequencyData(dataArray.current);
          const average = dataArray.current.reduce((a, b) => a + b) / dataArray.current.length;
          setAudioLevel(average / 255);
        }
        if (isListening) {
          requestAnimationFrame(updateAudioLevel);
        }
      };
      
      updateAudioLevel();
    } catch (error) {
      console.error('Error setting up audio visualization:', error);
    }
  }, [isListening]);

  // Start listening
  const startListening = useCallback(async () => {
    if (!recognition.current || !voiceSettings.enabled) return;

    try {
      setIsListening(true);
      setupAudioVisualization();
      
      recognition.current.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        if (finalTranscript) {
          handleVoiceCommand(finalTranscript.toLowerCase().trim());
        }
      };

      recognition.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognition.current.onend = () => {
        if (voiceSettings.enabled && isWaitingForWakeWord) {
          // Restart recognition for wake word detection
          setTimeout(() => {
            recognition.current.start();
          }, 100);
        } else {
          setIsListening(false);
        }
      };

      recognition.current.start();
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      setIsListening(false);
    }
  }, [voiceSettings.enabled, isWaitingForWakeWord, setupAudioVisualization]);

  // Stop listening
  const stopListening = useCallback(() => {
    if (recognition.current) {
      recognition.current.stop();
    }
    setIsListening(false);
    setIsWaitingForWakeWord(false);
  }, []);

  // Handle voice commands
  const handleVoiceCommand = useCallback(async (command) => {
    console.log('Voice command received:', command);

    // Wake word detection
    if (isWaitingForWakeWord) {
      if (command.includes(voiceSettings.wakeWord)) {
        setIsWaitingForWakeWord(false);
        speak("Yes, I'm listening. How can I help you?");
        return;
      } else {
        return; // Ignore other speech when waiting for wake word
      }
    }

    // Process commands
    if (command.includes('hey master') || command.includes('ok master')) {
      speak("Yes, I'm here. What would you like to learn?");
      return;
    }

    if (command.includes('stop speaking') || command.includes('stop talking')) {
      stopSpeaking();
      return;
    }

    if (command.includes('speak slower')) {
      setVoiceSettings(prev => ({ ...prev, rate: Math.max(0.5, prev.rate - 0.2) }));
      speak("I'll speak slower now.");
      return;
    }

    if (command.includes('speak faster')) {
      setVoiceSettings(prev => ({ ...prev, rate: Math.min(2.0, prev.rate + 0.2) }));
      speak("I'll speak faster now.");
      return;
    }

    if (command.includes('new session') || command.includes('start new session')) {
      speak("Starting a new learning session.");
      // Trigger new session creation
      return;
    }

    if (command.includes('what did you just say') || command.includes('repeat that')) {
      // Repeat last AI response
      const lastMessage = state.messages.filter(m => m.sender === 'mentor').pop();
      if (lastMessage) {
        speak(lastMessage.message);
      }
      return;
    }

    // Default: send as chat message
    if (state.currentSession) {
      try {
        await actions.sendMessage(state.currentSession.id, command);
      } catch (error) {
        speak("Sorry, I couldn't process that message. Please try again.");
      }
    } else {
      speak("Please start a learning session first.");
    }
  }, [state.currentSession, state.messages, actions, voiceSettings.wakeWord, isWaitingForWakeWord]);

  // Text to speech
  const speak = useCallback((text, options = {}) => {
    if (!synthesis.current || !voiceSettings.textToSpeech) return;

    // Cancel any ongoing speech
    synthesis.current.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = options.rate || voiceSettings.rate;
    utterance.pitch = options.pitch || voiceSettings.pitch;
    utterance.volume = options.volume || voiceSettings.volume;
    utterance.lang = options.language || voiceSettings.language;

    if (voiceSettings.voice) {
      utterance.voice = voiceSettings.voice;
    }

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    synthesis.current.speak(utterance);
  }, [voiceSettings]);

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    if (synthesis.current) {
      synthesis.current.cancel();
    }
    setIsSpeaking(false);
  }, []);

  // Auto-speak new AI messages
  useEffect(() => {
    if (voiceSettings.autoSpeak && voiceSettings.enabled) {
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && lastMessage.sender === 'mentor' && lastMessage.message) {
        // Clean the message for better speech
        const cleanText = lastMessage.message
          .replace(/#{1,6}\s*/g, '') // Remove markdown headers
          .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold formatting
          .replace(/\*(.*?)\*/g, '$1') // Remove italic formatting
          .replace(/`(.*?)`/g, '$1') // Remove code formatting
          .replace(/\[(.*?)\]\(.*?\)/g, '$1') // Remove links, keep text
          .replace(/\n+/g, '. '); // Replace newlines with periods
        
        speak(cleanText);
      }
    }
  }, [state.messages, voiceSettings.autoSpeak, voiceSettings.enabled, speak]);

  // Wake word monitoring
  const enableWakeWordMode = useCallback(() => {
    setIsWaitingForWakeWord(true);
    startListening();
  }, [startListening]);

  return {
    isListening,
    isSpeaking,
    audioLevel,
    voiceSettings,
    setVoiceSettings,
    isWaitingForWakeWord,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    enableWakeWordMode,
    handleVoiceCommand
  };
}

// Voice Control Component
export function VoiceInterface() {
  const {
    isListening,
    isSpeaking,
    audioLevel,
    voiceSettings,
    setVoiceSettings,
    isWaitingForWakeWord,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    enableWakeWordMode
  } = useVoiceInterface();

  const [showSettings, setShowSettings] = useState(false);
  const [availableVoices, setAvailableVoices] = useState([]);

  // Load available voices
  useEffect(() => {
    const loadVoices = () => {
      const voices = speechSynthesis.getVoices();
      setAvailableVoices(voices.filter(voice => voice.lang.startsWith('en')));
    };

    loadVoices();
    speechSynthesis.onvoiceschanged = loadVoices;
  }, []);

  return (
    <div className="fixed bottom-4 left-4 z-40">
      {/* Main Voice Button */}
      <motion.div
        className="relative"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        <GlassButton
          size="lg"
          variant={voiceSettings.enabled ? (isListening ? "success" : "primary") : "secondary"}
          onClick={() => {
            if (!voiceSettings.enabled) {
              setVoiceSettings(prev => ({ ...prev, enabled: true }));
              enableWakeWordMode();
            } else if (isListening) {
              stopListening();
            } else {
              startListening();
            }
          }}
          className="w-16 h-16 rounded-full flex items-center justify-center"
        >
          {isListening ? (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ repeat: Infinity, duration: 1 }}
            >
              <Mic className="h-6 w-6" />
            </motion.div>
          ) : (
            <MicOff className="h-6 w-6" />
          )}
        </GlassButton>

        {/* Audio Visualization */}
        {isListening && (
          <div className="absolute -top-2 -right-2 w-6 h-6">
            <div className="relative w-full h-full">
              {[...Array(3)].map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute inset-0 rounded-full border-2 border-green-400"
                  animate={{
                    scale: [1, 1.5, 1],
                    opacity: [0.8, 0, 0.8]
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 1.5,
                    delay: i * 0.2
                  }}
                  style={{
                    scale: 1 + audioLevel * 0.5
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Speaking Indicator */}
        {isSpeaking && (
          <div className="absolute -bottom-2 -right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
            >
              <Volume2 className="h-3 w-3 text-white" />
            </motion.div>
          </div>
        )}

        {/* Wake Word Mode Indicator */}
        {isWaitingForWakeWord && (
          <div className="absolute -top-8 left-1/2 transform -translate-x-1/2">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-purple-500/90 text-white text-xs px-2 py-1 rounded-full whitespace-nowrap"
            >
              Say "{voiceSettings.wakeWord}"
            </motion.div>
          </div>
        )}
      </motion.div>

      {/* Quick Controls */}
      <AnimatePresence>
        {voiceSettings.enabled && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="absolute bottom-0 left-20 flex space-x-2"
          >
            {/* Stop Speaking */}
            {isSpeaking && (
              <GlassButton
                size="sm"
                variant="danger"
                onClick={stopSpeaking}
                className="rounded-full"
              >
                <VolumeX className="h-4 w-4" />
              </GlassButton>
            )}

            {/* Settings */}
            <GlassButton
              size="sm"
              variant="secondary"
              onClick={() => setShowSettings(true)}
              className="rounded-full"
            >
              <Settings className="h-4 w-4" />
            </GlassButton>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Settings Modal */}
      <VoiceSettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        voiceSettings={voiceSettings}
        setVoiceSettings={setVoiceSettings}
        availableVoices={availableVoices}
        speak={speak}
      />
    </div>
  );
}

// Voice Settings Modal
function VoiceSettingsModal({ 
  isOpen, 
  onClose, 
  voiceSettings, 
  setVoiceSettings, 
  availableVoices,
  speak 
}) {
  if (!isOpen) return null;

  const testVoice = () => {
    speak("This is how I sound with the current settings.", voiceSettings);
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="w-full max-w-2xl max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <GlassCard className="p-6" variant="premium">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-gradient-to-r from-green-500 to-blue-500">
                  <Headphones className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Voice Settings</h2>
                  <p className="text-gray-400">Customize your voice experience</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Voice Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Voice
              </label>
              <select
                value={voiceSettings.voice?.name || ''}
                onChange={(e) => {
                  const selectedVoice = availableVoices.find(v => v.name === e.target.value);
                  setVoiceSettings(prev => ({ ...prev, voice: selectedVoice }));
                }}
                className="w-full p-3 rounded-lg bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Default Voice</option>
                {availableVoices.map(voice => (
                  <option key={voice.name} value={voice.name} className="bg-gray-800">
                    {voice.name} ({voice.lang})
                  </option>
                ))}
              </select>
            </div>

            {/* Voice Controls */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Rate */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Speaking Rate: {voiceSettings.rate.toFixed(1)}x
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={voiceSettings.rate}
                  onChange={(e) => setVoiceSettings(prev => ({ 
                    ...prev, 
                    rate: parseFloat(e.target.value) 
                  }))}
                  className="w-full"
                />
              </div>

              {/* Pitch */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Pitch: {voiceSettings.pitch.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={voiceSettings.pitch}
                  onChange={(e) => setVoiceSettings(prev => ({ 
                    ...prev, 
                    pitch: parseFloat(e.target.value) 
                  }))}
                  className="w-full"
                />
              </div>

              {/* Volume */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Volume: {Math.round(voiceSettings.volume * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={voiceSettings.volume}
                  onChange={(e) => setVoiceSettings(prev => ({ 
                    ...prev, 
                    volume: parseFloat(e.target.value) 
                  }))}
                  className="w-full"
                />
              </div>

              {/* Wake Word */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Wake Word
                </label>
                <input
                  type="text"
                  value={voiceSettings.wakeWord}
                  onChange={(e) => setVoiceSettings(prev => ({ 
                    ...prev, 
                    wakeWord: e.target.value 
                  }))}
                  className="w-full p-3 rounded-lg bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="hey master"
                />
              </div>
            </div>

            {/* Feature Toggles */}
            <div className="space-y-4 mb-6">
              {[
                { key: 'autoSpeak', label: 'Auto-speak AI responses' },
                { key: 'voiceCommands', label: 'Voice commands' },
                { key: 'speechToText', label: 'Speech to text' },
                { key: 'textToSpeech', label: 'Text to speech' }
              ].map(({ key, label }) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-gray-300">{label}</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={voiceSettings[key]}
                      onChange={(e) => setVoiceSettings(prev => ({
                        ...prev,
                        [key]: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              ))}
            </div>

            {/* Test Voice */}
            <div className="mb-6">
              <GlassButton onClick={testVoice} className="w-full">
                <Play className="h-4 w-4 mr-2" />
                Test Voice Settings
              </GlassButton>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3">
              <GlassButton variant="secondary" onClick={onClose}>
                Cancel
              </GlassButton>
              <GlassButton onClick={onClose}>
                <Radio className="h-4 w-4 mr-2" />
                Save Settings
              </GlassButton>
            </div>
          </GlassCard>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}