import { useState, useRef, useCallback, useEffect } from 'react'

interface VoiceInputOptions {
  continuous?: boolean
  interimResults?: boolean
  language?: string
  onTranscript?: (transcript: string, isFinal: boolean) => void
  onError?: (error: string) => void
  onStart?: () => void
  onEnd?: () => void
}

interface VoiceInputState {
  isListening: boolean
  isSupported: boolean
  transcript: string
  interimTranscript: string
  error: string | null
}

export const useVoiceInput = (options: VoiceInputOptions = {}) => {
  const {
    continuous = true,
    interimResults = true,
    language = 'en-US',
    onTranscript,
    onError,
    onStart,
    onEnd
  } = options

  const [state, setState] = useState<VoiceInputState>({
    isListening: false,
    isSupported: typeof window !== 'undefined' && 'webkitSpeechRecognition' in window,
    transcript: '',
    interimTranscript: '',
    error: null
  })

  const recognitionRef = useRef<any>(null)
  const timeoutRef = useRef<NodeJS.Timeout>()

  // Initialize speech recognition
  useEffect(() => {
    if (!state.isSupported) return

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    if (!SpeechRecognition) return

    const recognition = new SpeechRecognition()
    recognition.continuous = continuous
    recognition.interimResults = interimResults
    recognition.lang = language
    recognition.maxAlternatives = 1

    recognition.onstart = () => {
      setState(prev => ({ ...prev, isListening: true, error: null }))
      onStart?.()
    }

    recognition.onresult = (event: any) => {
      let finalTranscript = ''
      let interimTranscript = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscript += transcript
        } else {
          interimTranscript += transcript
        }
      }

      setState(prev => ({
        ...prev,
        transcript: prev.transcript + finalTranscript,
        interimTranscript
      }))

      if (finalTranscript) {
        onTranscript?.(finalTranscript, true)
      } else if (interimTranscript) {
        onTranscript?.(interimTranscript, false)
      }

      // Auto-stop after silence
      if (finalTranscript && !continuous) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = setTimeout(() => {
          stop()
        }, 2000)
      }
    }

    recognition.onerror = (event: any) => {
      const errorMessage = `Speech recognition error: ${event.error}`
      setState(prev => ({ ...prev, error: errorMessage, isListening: false }))
      onError?.(errorMessage)
    }

    recognition.onend = () => {
      setState(prev => ({ ...prev, isListening: false }))
      onEnd?.()
    }

    recognitionRef.current = recognition

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
      clearTimeout(timeoutRef.current)
    }
  }, [continuous, interimResults, language, onTranscript, onError, onStart, onEnd, state.isSupported])

  const start = useCallback(() => {
    if (!state.isSupported || !recognitionRef.current) {
      const error = 'Speech recognition not supported in this browser'
      setState(prev => ({ ...prev, error }))
      onError?.(error)
      return
    }

    try {
      setState(prev => ({ ...prev, transcript: '', interimTranscript: '', error: null }))
      recognitionRef.current.start()
    } catch (error) {
      const errorMessage = 'Failed to start speech recognition'
      setState(prev => ({ ...prev, error: errorMessage }))
      onError?.(errorMessage)
    }
  }, [state.isSupported, onError])

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    clearTimeout(timeoutRef.current)
  }, [])

  const toggle = useCallback(() => {
    if (state.isListening) {
      stop()
    } else {
      start()
    }
  }, [state.isListening, start, stop])

  const reset = useCallback(() => {
    setState(prev => ({ ...prev, transcript: '', interimTranscript: '', error: null }))
  }, [])

  return {
    ...state,
    start,
    stop,
    toggle,
    reset
  }
}
