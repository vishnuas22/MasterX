import { useState, useRef, useCallback, useEffect } from 'react'

interface TTSOptions {
  voice?: SpeechSynthesisVoice | null
  rate?: number
  pitch?: number
  volume?: number
  onStart?: () => void
  onEnd?: () => void
  onError?: (error: string) => void
}

interface TTSState {
  isSupported: boolean
  isSpeaking: boolean
  isPaused: boolean
  voices: SpeechSynthesisVoice[]
  currentVoice: SpeechSynthesisVoice | null
  rate: number
  pitch: number
  volume: number
}

export const useTextToSpeech = (options: TTSOptions = {}) => {
  const {
    voice = null,
    rate = 1,
    pitch = 1,
    volume = 1,
    onStart,
    onEnd,
    onError
  } = options

  const [state, setState] = useState<TTSState>({
    isSupported: typeof window !== 'undefined' && 'speechSynthesis' in window,
    isSpeaking: false,
    isPaused: false,
    voices: [],
    currentVoice: null,
    rate,
    pitch,
    volume
  })

  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
  const textQueueRef = useRef<string[]>([])

  // Load available voices
  useEffect(() => {
    if (!state.isSupported) return

    const loadVoices = () => {
      const voices = speechSynthesis.getVoices()
      setState(prev => ({
        ...prev,
        voices,
        currentVoice: voice || voices.find(v => v.default) || voices[0] || null
      }))
    }

    loadVoices()
    speechSynthesis.addEventListener('voiceschanged', loadVoices)

    return () => {
      speechSynthesis.removeEventListener('voiceschanged', loadVoices)
    }
  }, [state.isSupported, voice])

  const speak = useCallback((text: string, interrupt: boolean = false) => {
    if (!state.isSupported) {
      onError?.('Text-to-speech not supported in this browser')
      return
    }

    if (!text.trim()) return

    // Stop current speech if interrupting
    if (interrupt && state.isSpeaking) {
      speechSynthesis.cancel()
      textQueueRef.current = []
    }

    // Add to queue if currently speaking and not interrupting
    if (state.isSpeaking && !interrupt) {
      textQueueRef.current.push(text)
      return
    }

    const utterance = new SpeechSynthesisUtterance(text)
    
    // Configure utterance
    utterance.voice = state.currentVoice
    utterance.rate = state.rate
    utterance.pitch = state.pitch
    utterance.volume = state.volume

    utterance.onstart = () => {
      setState(prev => ({ ...prev, isSpeaking: true, isPaused: false }))
      onStart?.()
    }

    utterance.onend = () => {
      setState(prev => ({ ...prev, isSpeaking: false, isPaused: false }))
      onEnd?.()
      
      // Process queue
      if (textQueueRef.current.length > 0) {
        const nextText = textQueueRef.current.shift()
        if (nextText) {
          setTimeout(() => speak(nextText), 100) // Small delay between utterances
        }
      }
    }

    utterance.onerror = (event) => {
      setState(prev => ({ ...prev, isSpeaking: false, isPaused: false }))
      onError?.(`Speech synthesis error: ${event.error}`)
    }

    utterance.onpause = () => {
      setState(prev => ({ ...prev, isPaused: true }))
    }

    utterance.onresume = () => {
      setState(prev => ({ ...prev, isPaused: false }))
    }

    utteranceRef.current = utterance
    speechSynthesis.speak(utterance)
  }, [state.isSupported, state.currentVoice, state.rate, state.pitch, state.volume, state.isSpeaking, onStart, onEnd, onError])

  const pause = useCallback(() => {
    if (state.isSupported && state.isSpeaking && !state.isPaused) {
      speechSynthesis.pause()
    }
  }, [state.isSupported, state.isSpeaking, state.isPaused])

  const resume = useCallback(() => {
    if (state.isSupported && state.isSpeaking && state.isPaused) {
      speechSynthesis.resume()
    }
  }, [state.isSupported, state.isSpeaking, state.isPaused])

  const stop = useCallback(() => {
    if (state.isSupported) {
      speechSynthesis.cancel()
      textQueueRef.current = []
      setState(prev => ({ ...prev, isSpeaking: false, isPaused: false }))
    }
  }, [state.isSupported])

  const setVoice = useCallback((voice: SpeechSynthesisVoice) => {
    setState(prev => ({ ...prev, currentVoice: voice }))
  }, [])

  const setRate = useCallback((rate: number) => {
    setState(prev => ({ ...prev, rate: Math.max(0.1, Math.min(10, rate)) }))
  }, [])

  const setPitch = useCallback((pitch: number) => {
    setState(prev => ({ ...prev, pitch: Math.max(0, Math.min(2, pitch)) }))
  }, [])

  const setVolume = useCallback((volume: number) => {
    setState(prev => ({ ...prev, volume: Math.max(0, Math.min(1, volume)) }))
  }, [])

  // Get preferred voices (English, natural sounding)
  const getPreferredVoices = useCallback(() => {
    return state.voices.filter(voice => 
      voice.lang.startsWith('en') && 
      (voice.name.includes('Natural') || 
       voice.name.includes('Premium') || 
       voice.name.includes('Enhanced') ||
       voice.localService)
    ).sort((a, b) => {
      // Prefer local voices
      if (a.localService && !b.localService) return -1
      if (!a.localService && b.localService) return 1
      return 0
    })
  }, [state.voices])

  // Clean text for better TTS
  const cleanTextForSpeech = useCallback((text: string) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown
      .replace(/\*(.*?)\*/g, '$1') // Remove italic markdown
      .replace(/`(.*?)`/g, '$1') // Remove inline code
      .replace(/```[\s\S]*?```/g, '[code block]') // Replace code blocks
      .replace(/#{1,6}\s/g, '') // Remove markdown headers
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Replace links with text
      .replace(/\n{2,}/g, '. ') // Replace multiple newlines with periods
      .replace(/\n/g, ' ') // Replace single newlines with spaces
      .replace(/\s{2,}/g, ' ') // Replace multiple spaces with single space
      .trim()
  }, [])

  const speakClean = useCallback((text: string, interrupt: boolean = false) => {
    const cleanedText = cleanTextForSpeech(text)
    speak(cleanedText, interrupt)
  }, [speak, cleanTextForSpeech])

  return {
    ...state,
    speak,
    speakClean,
    pause,
    resume,
    stop,
    setVoice,
    setRate,
    setPitch,
    setVolume,
    getPreferredVoices,
    cleanTextForSpeech
  }
}
