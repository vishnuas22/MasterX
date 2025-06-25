import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cube, 
  VrHeadset, 
  Eye, 
  RotateCcw, 
  ZoomIn, 
  ZoomOut, 
  Move3D,
  Layers,
  Settings,
  Play,
  Pause,
  Volume2,
  Maximize2,
  Minimize2
} from 'lucide-react';
import { GlassCard, GlassButton } from './GlassCard';
import { useApp } from '../context/AppContext';

// AR/VR Visualization Hook
export function useARVRVisualization() {
  const { state } = useApp();
  const [isVRMode, setIsVRMode] = useState(false);
  const [isARMode, setIsARMode] = useState(false);
  const [is3DMode, setIs3DMode] = useState(false);
  const [vrSupported, setVRSupported] = useState(false);
  const [arSupported, setARSupported] = useState(false);
  const [currentVisualization, setCurrentVisualization] = useState(null);
  const [visualizationSettings, setVisualizationSettings] = useState({
    renderQuality: 'high',
    enablePhysics: true,
    enableShadows: true,
    enableLighting: true,
    fov: 75,
    nearPlane: 0.1,
    farPlane: 1000,
    autoRotate: false,
    rotationSpeed: 1,
    zoomLevel: 1,
    cameraPosition: { x: 0, y: 0, z: 5 },
    backgroundColor: '#000011'
  });

  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const vrSessionRef = useRef(null);
  const arSessionRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Check VR/AR support
  useEffect(() => {
    const checkSupport = async () => {
      // Check WebXR support
      if ('xr' in navigator) {
        try {
          const vrSupport = await navigator.xr.isSessionSupported('immersive-vr');
          const arSupport = await navigator.xr.isSessionSupported('immersive-ar');
          setVRSupported(vrSupport);
          setARSupported(arSupport);
        } catch (error) {
          console.log('WebXR not supported:', error);
        }
      }
    };

    checkSupport();
  }, []);

  // Initialize 3D scene (would use Three.js in real implementation)
  const initialize3DScene = useCallback((container) => {
    if (!container) return;

    // In a real implementation, this would use Three.js
    // const scene = new THREE.Scene();
    // const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    // const renderer = new THREE.WebGLRenderer({ antialias: true });
    
    // For demo purposes, we'll create a simulated 3D environment
    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.background = 'linear-gradient(45deg, #001122, #002244)';
    
    container.appendChild(canvas);
    
    // Simulate 3D rendering
    const ctx = canvas.getContext('2d');
    const render3DScene = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw a rotating cube simulation
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const time = Date.now() * 0.001;
      
      // Draw cube wireframe
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const size = 100 * visualizationSettings.zoomLevel;
      const rotation = visualizationSettings.autoRotate ? time * visualizationSettings.rotationSpeed : 0;
      
      // Simulate 3D cube projection
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2 + rotation;
        const x = centerX + Math.cos(angle) * size;
        const y = centerY + Math.sin(angle) * size * 0.6;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();
      ctx.stroke();
      
      // Add some floating particles
      for (let i = 0; i < 20; i++) {
        const particleTime = time + i;
        const x = centerX + Math.cos(particleTime) * (150 + i * 10);
        const y = centerY + Math.sin(particleTime * 0.7) * (100 + i * 5);
        
        ctx.fillStyle = `rgba(0, 255, 136, ${0.3 + Math.sin(particleTime) * 0.2})`;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      
      if (is3DMode) {
        animationFrameRef.current = requestAnimationFrame(render3DScene);
      }
    };
    
    render3DScene();
    return canvas;
  }, [visualizationSettings, is3DMode]);

  // Start VR session
  const startVRSession = useCallback(async () => {
    if (!vrSupported) return false;

    try {
      const session = await navigator.xr.requestSession('immersive-vr');
      vrSessionRef.current = session;
      setIsVRMode(true);
      
      session.addEventListener('end', () => {
        setIsVRMode(false);
        vrSessionRef.current = null;
      });
      
      return true;
    } catch (error) {
      console.error('Failed to start VR session:', error);
      return false;
    }
  }, [vrSupported]);

  // Start AR session
  const startARSession = useCallback(async () => {
    if (!arSupported) return false;

    try {
      const session = await navigator.xr.requestSession('immersive-ar');
      arSessionRef.current = session;
      setIsARMode(true);
      
      session.addEventListener('end', () => {
        setIsARMode(false);
        arSessionRef.current = null;
      });
      
      return true;
    } catch (error) {
      console.error('Failed to start AR session:', error);
      return false;
    }
  }, [arSupported]);

  // Stop VR/AR session
  const stopXRSession = useCallback(() => {
    if (vrSessionRef.current) {
      vrSessionRef.current.end();
    }
    if (arSessionRef.current) {
      arSessionRef.current.end();
    }
  }, []);

  // Generate 3D visualization for concept
  const generateVisualization = useCallback((concept, type = 'molecule') => {
    const visualizations = {
      molecule: {
        type: '3d_molecule',
        objects: [
          { type: 'sphere', position: [0, 0, 0], color: '#ff4444', scale: 1.2 },
          { type: 'sphere', position: [2, 0, 0], color: '#4444ff', scale: 1.0 },
          { type: 'cylinder', position: [1, 0, 0], rotation: [0, 0, 90], color: '#888888' }
        ],
        animation: 'rotate',
        labels: ['Atom A', 'Atom B', 'Bond']
      },
      network: {
        type: '3d_network',
        objects: [
          { type: 'sphere', position: [0, 0, 0], color: '#44ff44', scale: 0.8 },
          { type: 'sphere', position: [3, 2, 1], color: '#ff44ff', scale: 0.8 },
          { type: 'sphere', position: [-2, 3, -1], color: '#ffff44', scale: 0.8 },
          { type: 'line', from: [0, 0, 0], to: [3, 2, 1], color: '#ffffff' },
          { type: 'line', from: [0, 0, 0], to: [-2, 3, -1], color: '#ffffff' }
        ],
        animation: 'pulse',
        labels: ['Node 1', 'Node 2', 'Node 3']
      },
      graph: {
        type: '3d_graph',
        objects: [
          { type: 'plane', position: [0, 0, 0], color: '#333333', scale: 10 },
          { type: 'curve', points: [[0, 0, 0], [1, 2, 0], [2, 1, 0], [3, 3, 0]], color: '#00ff00' }
        ],
        animation: 'draw',
        labels: ['X-axis', 'Y-axis', 'Function']
      }
    };

    return visualizations[type] || visualizations.molecule;
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      stopXRSession();
    };
  }, [stopXRSession]);

  return {
    isVRMode,
    isARMode,
    is3DMode,
    setIs3DMode,
    vrSupported,
    arSupported,
    currentVisualization,
    setCurrentVisualization,
    visualizationSettings,
    setVisualizationSettings,
    initialize3DScene,
    startVRSession,
    startARSession,
    stopXRSession,
    generateVisualization
  };
}

// AR/VR Interface Component
export function ARVRInterface() {
  const {
    isVRMode,
    isARMode,
    is3DMode,
    setIs3DMode,
    vrSupported,
    arSupported,
    currentVisualization,
    setCurrentVisualization,
    visualizationSettings,
    setVisualizationSettings,
    initialize3DScene,
    startVRSession,
    startARSession,
    stopXRSession,
    generateVisualization
  } = useARVRVisualization();

  const [showSettings, setShowSettings] = useState(false);
  const [showVisualizationPanel, setShowVisualizationPanel] = useState(false);
  const containerRef = useRef(null);

  // Initialize 3D scene when container is ready
  useEffect(() => {
    if (is3DMode && containerRef.current) {
      initialize3DScene(containerRef.current);
    }
  }, [is3DMode, initialize3DScene]);

  return (
    <>
      {/* Main AR/VR Controls */}
      <div className="fixed top-4 right-4 z-40">
        <motion.div
          className="flex flex-col space-y-2"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          {/* 3D Mode Toggle */}
          <GlassButton
            size="lg"
            variant={is3DMode ? "success" : "secondary"}
            onClick={() => {
              setIs3DMode(!is3DMode);
              if (!is3DMode) {
                setShowVisualizationPanel(true);
              }
            }}
            className="w-14 h-14 rounded-full flex items-center justify-center"
          >
            <motion.div
              animate={is3DMode ? { rotateY: 360 } : {}}
              transition={{ duration: 2, repeat: is3DMode ? Infinity : 0, ease: "linear" }}
            >
              <Cube className="h-6 w-6" />
            </motion.div>
          </GlassButton>

          {/* VR Mode */}
          {vrSupported && (
            <GlassButton
              size="md"
              variant={isVRMode ? "primary" : "secondary"}
              onClick={isVRMode ? stopXRSession : startVRSession}
              className="w-12 h-12 rounded-full flex items-center justify-center"
            >
              <VrHeadset className="h-5 w-5" />
            </GlassButton>
          )}

          {/* AR Mode */}
          {arSupported && (
            <GlassButton
              size="md"
              variant={isARMode ? "primary" : "secondary"}
              onClick={isARMode ? stopXRSession : startARSession}
              className="w-12 h-12 rounded-full flex items-center justify-center"
            >
              <Eye className="h-5 w-5" />
            </GlassButton>
          )}

          {/* Settings */}
          <GlassButton
            size="md"
            variant="secondary"
            onClick={() => setShowSettings(true)}
            className="w-12 h-12 rounded-full flex items-center justify-center"
          >
            <Settings className="h-5 w-5" />
          </GlassButton>
        </motion.div>
      </div>

      {/* 3D Visualization Container */}
      <AnimatePresence>
        {is3DMode && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="fixed inset-4 z-30 pointer-events-none"
          >
            <GlassCard className="h-full p-4" variant="minimal">
              <div className="h-full relative pointer-events-auto">
                <div
                  ref={containerRef}
                  className="w-full h-full rounded-lg overflow-hidden"
                />
                
                {/* 3D Controls Overlay */}
                <div className="absolute bottom-4 left-4 flex space-x-2">
                  <GlassButton
                    size="sm"
                    variant="secondary"
                    onClick={() => setVisualizationSettings(prev => ({
                      ...prev,
                      autoRotate: !prev.autoRotate
                    }))}
                  >
                    <RotateCcw className="h-4 w-4" />
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant="secondary"
                    onClick={() => setVisualizationSettings(prev => ({
                      ...prev,
                      zoomLevel: Math.min(3, prev.zoomLevel + 0.2)
                    }))}
                  >
                    <ZoomIn className="h-4 w-4" />
                  </GlassButton>
                  <GlassButton
                    size="sm"
                    variant="secondary"
                    onClick={() => setVisualizationSettings(prev => ({
                      ...prev,
                      zoomLevel: Math.max(0.5, prev.zoomLevel - 0.2)
                    }))}
                  >
                    <ZoomOut className="h-4 w-4" />
                  </GlassButton>
                </div>

                {/* Close Button */}
                <div className="absolute top-4 right-4">
                  <GlassButton
                    size="sm"
                    variant="secondary"
                    onClick={() => setIs3DMode(false)}
                  >
                    <Minimize2 className="h-4 w-4" />
                  </GlassButton>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Visualization Panel */}
      <VisualizationPanel
        isOpen={showVisualizationPanel}
        onClose={() => setShowVisualizationPanel(false)}
        generateVisualization={generateVisualization}
        setCurrentVisualization={setCurrentVisualization}
      />

      {/* Settings Modal */}
      <ARVRSettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        visualizationSettings={visualizationSettings}
        setVisualizationSettings={setVisualizationSettings}
        vrSupported={vrSupported}
        arSupported={arSupported}
      />
    </>
  );
}

// Visualization Panel Component
function VisualizationPanel({ isOpen, onClose, generateVisualization, setCurrentVisualization }) {
  if (!isOpen) return null;

  const visualizationTypes = [
    {
      id: 'molecule',
      name: 'Molecular Structure',
      description: 'Visualize chemical compounds and molecular bonds',
      icon: '🧪',
      examples: ['Water (H2O)', 'Carbon Dioxide (CO2)', 'Methane (CH4)']
    },
    {
      id: 'network',
      name: 'Network Graph',
      description: 'Display relationships and connections',
      icon: '🕸️',
      examples: ['Neural Network', 'Social Network', 'Computer Network']
    },
    {
      id: 'graph',
      name: 'Mathematical Graph',
      description: 'Plot mathematical functions and data',
      icon: '📊',
      examples: ['Linear Function', 'Quadratic Function', 'Sine Wave']
    },
    {
      id: 'anatomy',
      name: 'Anatomical Model',
      description: 'Explore biological structures',
      icon: '🫀',
      examples: ['Heart', 'Brain', 'DNA Helix']
    },
    {
      id: 'physics',
      name: 'Physics Simulation',
      description: 'Simulate physical phenomena',
      icon: '⚡',
      examples: ['Gravity', 'Electromagnetic Field', 'Wave Motion']
    },
    {
      id: 'architecture',
      name: 'Architectural Model',
      description: 'Visualize structures and buildings',
      icon: '🏗️',
      examples: ['Building Design', 'Bridge Structure', 'City Layout']
    }
  ];

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
          className="w-full max-w-4xl max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <GlassCard className="p-6" variant="premium">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500">
                  <Layers className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">3D Visualizations</h2>
                  <p className="text-gray-400">Choose a visualization type</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Visualization Types */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {visualizationTypes.map((type) => (
                <motion.button
                  key={type.id}
                  onClick={() => {
                    const visualization = generateVisualization('demo', type.id);
                    setCurrentVisualization(visualization);
                    onClose();
                  }}
                  className="p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300 text-left"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="text-2xl mb-2">{type.icon}</div>
                  <h3 className="font-semibold text-white mb-2">{type.name}</h3>
                  <p className="text-sm text-gray-400 mb-3">{type.description}</p>
                  
                  <div className="space-y-1">
                    <div className="text-xs font-medium text-gray-300">Examples:</div>
                    {type.examples.map((example, index) => (
                      <div key={index} className="text-xs text-gray-500">
                        • {example}
                      </div>
                    ))}
                  </div>
                </motion.button>
              ))}
            </div>

            {/* Quick Actions */}
            <div className="flex justify-end space-x-3 mt-6">
              <GlassButton variant="secondary" onClick={onClose}>
                Cancel
              </GlassButton>
              <GlassButton onClick={onClose}>
                <Cube className="h-4 w-4 mr-2" />
                Start Visualizing
              </GlassButton>
            </div>
          </GlassCard>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

// AR/VR Settings Modal
function ARVRSettingsModal({ 
  isOpen, 
  onClose, 
  visualizationSettings, 
  setVisualizationSettings, 
  vrSupported, 
  arSupported 
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
          className="w-full max-w-2xl max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <GlassCard className="p-6" variant="premium">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500">
                  <VrHeadset className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">AR/VR Settings</h2>
                  <p className="text-gray-400">Configure immersive experience</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Support Status */}
            <div className="mb-6 p-4 rounded-xl bg-white/5 border border-white/10">
              <h3 className="font-semibold text-white mb-3">Device Support</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Virtual Reality (VR)</span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    vrSupported ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                  }`}>
                    {vrSupported ? 'Supported' : 'Not Supported'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Augmented Reality (AR)</span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    arSupported ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                  }`}>
                    {arSupported ? 'Supported' : 'Not Supported'}
                  </span>
                </div>
              </div>
            </div>

            {/* Visualization Settings */}
            <div className="space-y-6">
              {/* Render Quality */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Render Quality
                </label>
                <select
                  value={visualizationSettings.renderQuality}
                  onChange={(e) => setVisualizationSettings(prev => ({ 
                    ...prev, 
                    renderQuality: e.target.value 
                  }))}
                  className="w-full p-3 rounded-lg bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="low" className="bg-gray-800">Low (Better Performance)</option>
                  <option value="medium" className="bg-gray-800">Medium (Balanced)</option>
                  <option value="high" className="bg-gray-800">High (Best Quality)</option>
                  <option value="ultra" className="bg-gray-800">Ultra (Experimental)</option>
                </select>
              </div>

              {/* Field of View */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Field of View: {visualizationSettings.fov}°
                </label>
                <input
                  type="range"
                  min="30"
                  max="120"
                  value={visualizationSettings.fov}
                  onChange={(e) => setVisualizationSettings(prev => ({ 
                    ...prev, 
                    fov: parseInt(e.target.value) 
                  }))}
                  className="w-full"
                />
              </div>

              {/* Auto Rotation Speed */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Auto Rotation Speed: {visualizationSettings.rotationSpeed}x
                </label>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.1"
                  value={visualizationSettings.rotationSpeed}
                  onChange={(e) => setVisualizationSettings(prev => ({ 
                    ...prev, 
                    rotationSpeed: parseFloat(e.target.value) 
                  }))}
                  className="w-full"
                />
              </div>

              {/* Feature Toggles */}
              <div className="space-y-4">
                <h3 className="font-semibold text-white">Visual Features</h3>
                {[
                  { key: 'enablePhysics', label: 'Physics Simulation' },
                  { key: 'enableShadows', label: 'Dynamic Shadows' },
                  { key: 'enableLighting', label: 'Realistic Lighting' },
                  { key: 'autoRotate', label: 'Auto Rotation' }
                ].map(({ key, label }) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-gray-300">{label}</span>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={visualizationSettings[key]}
                        onChange={(e) => setVisualizationSettings(prev => ({
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
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3 mt-6">
              <GlassButton variant="secondary" onClick={onClose}>
                Cancel
              </GlassButton>
              <GlassButton onClick={onClose}>
                <Cube className="h-4 w-4 mr-2" />
                Apply Settings
              </GlassButton>
            </div>
          </GlassCard>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}