import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Hand, 
  Eye, 
  Zap, 
  Settings, 
  Play, 
  Pause, 
  Volume2, 
  SkipForward,
  SkipBack,
  MousePointer,
  Camera,
  WifiOff
} from 'lucide-react';
import { GlassCard, GlassButton } from './GlassCard';
import { useApp } from '../context/AppContext';

// Gesture Recognition Hook
export function useGestureControl() {
  const { state, actions } = useApp();
  const [isEnabled, setIsEnabled] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [cameraPermission, setCameraPermission] = useState('prompt');
  const [detectedGestures, setDetectedGestures] = useState([]);
  const [gestureSettings, setGestureSettings] = useState({
    sensitivity: 0.7,
    gestureTimeout: 2000,
    enabledGestures: {
      scroll: true,
      navigate: true,
      voice: true,
      speed: true,
      volume: true
    },
    customGestures: []
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const lastGestureRef = useRef(null);
  const gestureBufferRef = useRef([]);

  // Gesture definitions
  const GESTURES = {
    PALM_OPEN: {
      name: 'Palm Open',
      action: 'pause_resume',
      description: 'Pause/Resume speaking',
      landmarks: 'open_palm'
    },
    FIST: {
      name: 'Fist',
      action: 'stop',
      description: 'Stop current action',
      landmarks: 'closed_fist'
    },
    POINT_UP: {
      name: 'Point Up',
      action: 'scroll_up',
      description: 'Scroll up',
      landmarks: 'index_up'
    },
    POINT_DOWN: {
      name: 'Point Down',
      action: 'scroll_down',
      description: 'Scroll down',
      landmarks: 'index_down'
    },
    THUMB_UP: {
      name: 'Thumbs Up',
      action: 'like_continue',
      description: 'Like/Continue',
      landmarks: 'thumb_up'
    },
    THUMB_DOWN: {
      name: 'Thumbs Down',
      action: 'dislike_stop',
      description: 'Dislike/Stop',
      landmarks: 'thumb_down'
    },
    PEACE: {
      name: 'Peace Sign',
      action: 'new_session',
      description: 'Start new session',
      landmarks: 'peace_sign'
    },
    OK: {
      name: 'OK Sign',
      action: 'confirm',
      description: 'Confirm action',
      landmarks: 'ok_sign'
    },
    SWIPE_LEFT: {
      name: 'Swipe Left',
      action: 'previous',
      description: 'Previous message',
      landmarks: 'swipe_left'
    },
    SWIPE_RIGHT: {
      name: 'Swipe Right',
      action: 'next',
      description: 'Next message',
      landmarks: 'swipe_right'
    }
  };

  // Initialize camera and gesture detection
  const initializeGestureDetection = useCallback(async () => {
    try {
      // Request camera permission
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      setCameraPermission('granted');
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // Load MediaPipe Hands model (would need to be imported)
      // This is a simplified version - in reality, you'd use MediaPipe or TensorFlow.js
      // await loadHandPoseModel();
      
      return true;
    } catch (error) {
      console.error('Failed to initialize gesture detection:', error);
      setCameraPermission('denied');
      return false;
    }
  }, []);

  // Simplified gesture detection (would use actual ML model in production)
  const detectGestures = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // In a real implementation, this would use MediaPipe Hands or similar
    // For demo purposes, we'll simulate gesture detection
    const simulatedGesture = Math.random();
    
    if (simulatedGesture > 0.95) {
      const gestures = Object.keys(GESTURES);
      const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
      handleGestureDetected(randomGesture);
    }
  }, []);

  // Handle detected gesture
  const handleGestureDetected = useCallback((gestureType) => {
    const now = Date.now();
    const gesture = GESTURES[gestureType];
    
    if (!gesture || !gestureSettings.enabledGestures[gesture.action]) return;

    // Prevent duplicate gestures within timeout period
    if (lastGestureRef.current && 
        now - lastGestureRef.current.timestamp < gestureSettings.gestureTimeout &&
        lastGestureRef.current.type === gestureType) {
      return;
    }

    lastGestureRef.current = { type: gestureType, timestamp: now };
    
    // Add to gesture buffer for smoothing
    gestureBufferRef.current.push({ type: gestureType, timestamp: now, confidence: 0.8 });
    if (gestureBufferRef.current.length > 5) {
      gestureBufferRef.current.shift();
    }

    // Update detected gestures state
    setDetectedGestures(prev => [
      { type: gestureType, timestamp: now, confidence: 0.8 },
      ...prev.slice(0, 4)
    ]);

    // Execute gesture action
    executeGestureAction(gesture.action);
  }, [gestureSettings.gestureTimeout, gestureSettings.enabledGestures]);

  // Execute gesture action
  const executeGestureAction = useCallback(async (action) => {
    switch (action) {
      case 'pause_resume':
        // Toggle speech or pause current action
        if (window.speechSynthesis.speaking) {
          window.speechSynthesis.pause();
        } else {
          window.speechSynthesis.resume();
        }
        break;

      case 'stop':
        // Stop current speech or action
        window.speechSynthesis.cancel();
        break;

      case 'scroll_up':
        // Scroll chat up
        const chatContainer = document.querySelector('[data-chat-container]');
        if (chatContainer) {
          chatContainer.scrollBy({ top: -200, behavior: 'smooth' });
        }
        break;

      case 'scroll_down':
        // Scroll chat down
        const chatContainerDown = document.querySelector('[data-chat-container]');
        if (chatContainerDown) {
          chatContainerDown.scrollBy({ top: 200, behavior: 'smooth' });
        }
        break;

      case 'like_continue':
        // Send positive feedback or continue
        if (state.currentSession) {
          await actions.sendMessage(state.currentSession.id, "👍 Please continue");
        }
        break;

      case 'dislike_stop':
        // Send negative feedback or request to stop
        if (state.currentSession) {
          await actions.sendMessage(state.currentSession.id, "Please stop or try a different approach");
        }
        break;

      case 'new_session':
        // Start new learning session
        // This would trigger session creation
        break;

      case 'confirm':
        // Confirm current action
        // Could be used to confirm prompts or actions
        break;

      case 'previous':
        // Navigate to previous message or content
        break;

      case 'next':
        // Navigate to next message or content
        break;

      default:
        console.log(`Unknown gesture action: ${action}`);
    }
  }, [state.currentSession, actions]);

  // Start gesture detection
  const startGestureDetection = useCallback(async () => {
    const initialized = await initializeGestureDetection();
    if (!initialized) return false;

    setIsEnabled(true);
    
    // Start detection loop
    detectionIntervalRef.current = setInterval(detectGestures, 100); // 10 FPS
    
    return true;
  }, [initializeGestureDetection, detectGestures]);

  // Stop gesture detection
  const stopGestureDetection = useCallback(() => {
    setIsEnabled(false);
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Calibration process
  const startCalibration = useCallback(async () => {
    setIsCalibrating(true);
    
    // Simulate calibration process
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    setIsCalibrating(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopGestureDetection();
    };
  }, [stopGestureDetection]);

  return {
    isEnabled,
    isCalibrating,
    cameraPermission,
    detectedGestures,
    gestureSettings,
    setGestureSettings,
    GESTURES,
    videoRef,
    canvasRef,
    startGestureDetection,
    stopGestureDetection,
    startCalibration,
    handleGestureDetected
  };
}

// Gesture Control Component
export function GestureControl() {
  const {
    isEnabled,
    isCalibrating,
    cameraPermission,
    detectedGestures,
    gestureSettings,
    setGestureSettings,
    GESTURES,
    videoRef,
    canvasRef,
    startGestureDetection,
    stopGestureDetection,
    startCalibration
  } = useGestureControl();

  const [showSettings, setShowSettings] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  return (
    <div className="fixed top-4 left-4 z-40">
      {/* Main Gesture Button */}
      <motion.div
        className="relative"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        <GlassButton
          size="lg"
          variant={isEnabled ? "success" : cameraPermission === 'denied' ? "danger" : "secondary"}
          onClick={async () => {
            if (!isEnabled) {
              const success = await startGestureDetection();
              if (success) {
                setShowPreview(true);
              }
            } else {
              stopGestureDetection();
              setShowPreview(false);
            }
          }}
          className="w-14 h-14 rounded-full flex items-center justify-center"
        >
          {cameraPermission === 'denied' ? (
            <WifiOff className="h-6 w-6" />
          ) : isEnabled ? (
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              <Hand className="h-6 w-6" />
            </motion.div>
          ) : (
            <Hand className="h-6 w-6" />
          )}
        </GlassButton>

        {/* Gesture Indicators */}
        <AnimatePresence>
          {detectedGestures.length > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="absolute -top-12 left-1/2 transform -translate-x-1/2"
            >
              <div className="bg-green-500/90 text-white text-xs px-2 py-1 rounded-full">
                {GESTURES[detectedGestures[0].type]?.name}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Calibration Indicator */}
        {isCalibrating && (
          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
              className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full"
            />
          </div>
        )}
      </motion.div>

      {/* Quick Controls */}
      <AnimatePresence>
        {isEnabled && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-16 left-0 flex flex-col space-y-2"
          >
            {/* Settings */}
            <GlassButton
              size="sm"
              variant="secondary"
              onClick={() => setShowSettings(true)}
              className="rounded-full w-10 h-10"
            >
              <Settings className="h-4 w-4" />
            </GlassButton>

            {/* Camera Preview Toggle */}
            <GlassButton
              size="sm"
              variant={showPreview ? "primary" : "secondary"}
              onClick={() => setShowPreview(!showPreview)}
              className="rounded-full w-10 h-10"
            >
              <Camera className="h-4 w-4" />
            </GlassButton>

            {/* Calibrate */}
            <GlassButton
              size="sm"
              variant="secondary"
              onClick={startCalibration}
              className="rounded-full w-10 h-10"
              disabled={isCalibrating}
            >
              <Zap className="h-4 w-4" />
            </GlassButton>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Camera Preview */}
      <AnimatePresence>
        {showPreview && isEnabled && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute top-0 left-20"
          >
            <GlassCard className="p-2" variant="minimal">
              <div className="relative">
                <video
                  ref={videoRef}
                  className="w-48 h-36 rounded-lg object-cover"
                  muted
                  playsInline
                />
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={480}
                  className="hidden"
                />
                
                {/* Gesture Overlay */}
                <div className="absolute inset-0 pointer-events-none">
                  {detectedGestures.slice(0, 1).map((gesture, index) => (
                    <motion.div
                      key={gesture.timestamp}
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.5 }}
                      className="absolute top-2 left-2 bg-green-500/80 text-white text-xs px-2 py-1 rounded"
                    >
                      {GESTURES[gesture.type]?.name}
                    </motion.div>
                  ))}
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <GestureSettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        gestureSettings={gestureSettings}
        setGestureSettings={setGestureSettings}
        GESTURES={GESTURES}
        startCalibration={startCalibration}
        isCalibrating={isCalibrating}
      />
    </div>
  );
}

// Gesture Settings Modal
function GestureSettingsModal({ 
  isOpen, 
  onClose, 
  gestureSettings, 
  setGestureSettings, 
  GESTURES,
  startCalibration,
  isCalibrating
}) {
  if (!isOpen) return null;

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
          className="w-full max-w-3xl max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <GlassCard className="p-6" variant="premium">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500">
                  <Hand className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Gesture Settings</h2>
                  <p className="text-gray-400">Configure gesture recognition</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Sensitivity Setting */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Sensitivity: {(gestureSettings.sensitivity * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={gestureSettings.sensitivity}
                onChange={(e) => setGestureSettings(prev => ({ 
                  ...prev, 
                  sensitivity: parseFloat(e.target.value) 
                }))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Less sensitive</span>
                <span>More sensitive</span>
              </div>
            </div>

            {/* Gesture Timeout */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Gesture Timeout: {gestureSettings.gestureTimeout}ms
              </label>
              <input
                type="range"
                min="500"
                max="5000"
                step="100"
                value={gestureSettings.gestureTimeout}
                onChange={(e) => setGestureSettings(prev => ({ 
                  ...prev, 
                  gestureTimeout: parseInt(e.target.value) 
                }))}
                className="w-full"
              />
            </div>

            {/* Available Gestures */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Available Gestures</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(GESTURES).map(([key, gesture]) => (
                  <div key={key} className="p-3 rounded-lg bg-white/5 border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-medium text-white">{gesture.name}</div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={gestureSettings.enabledGestures[gesture.action] || false}
                          onChange={(e) => setGestureSettings(prev => ({
                            ...prev,
                            enabledGestures: {
                              ...prev.enabledGestures,
                              [gesture.action]: e.target.checked
                            }
                          }))}
                          className="sr-only peer"
                        />
                        <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                      </label>
                    </div>
                    <div className="text-sm text-gray-400">{gesture.description}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Calibration */}
            <div className="mb-6 p-4 rounded-xl bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-400/20">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-white">Camera Calibration</div>
                  <div className="text-sm text-gray-400">Improve gesture recognition accuracy</div>
                </div>
                <GlassButton
                  onClick={startCalibration}
                  disabled={isCalibrating}
                  variant="primary"
                >
                  {isCalibrating ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                    >
                      <Zap className="h-4 w-4" />
                    </motion.div>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      Calibrate
                    </>
                  )}
                </GlassButton>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3">
              <GlassButton variant="secondary" onClick={onClose}>
                Cancel
              </GlassButton>
              <GlassButton onClick={onClose}>
                <Hand className="h-4 w-4 mr-2" />
                Save Settings
              </GlassButton>
            </div>
          </GlassCard>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}