/**
 * Quantum Background Component for MasterX Quantum Intelligence Platform
 * 
 * Sophisticated background animations that are hydration-safe and performant.
 * Uses CSS-only animations to prevent server/client mismatch issues.
 */

'use client'

export function QuantumBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {/* Primary Cosmic Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-cosmic-drift" />
      <div 
        className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-cosmic-drift" 
        style={{ animationDelay: '2s' }} 
      />
      <div 
        className="absolute top-3/4 left-3/4 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl animate-cosmic-drift" 
        style={{ animationDelay: '4s' }} 
      />

      {/* Neural Network Grid */}
      <div className="absolute inset-0 opacity-5">
        <div className="grid grid-cols-12 grid-rows-8 h-full w-full">
          {Array.from({ length: 96 }).map((_, i) => (
            <div
              key={i}
              className="border border-purple-500/20 animate-pulse"
              style={{ animationDelay: `${(i * 0.1) % 3}s` }}
            />
          ))}
        </div>
      </div>

      {/* Floating Data Streams */}
      <div className="absolute inset-0">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="absolute h-px bg-gradient-to-r from-transparent via-purple-400/30 to-transparent animate-data-stream"
            style={{
              top: `${10 + i * 12}%`,
              width: '200px',
              animationDelay: `${i * 0.8}s`,
              animationDuration: `${3 + (i % 2)}s`
            }}
          />
        ))}
      </div>

      {/* Quantum Particles */}
      <div className="absolute inset-0">
        {Array.from({ length: 15 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-purple-400/40 rounded-full animate-quantum-float"
            style={{
              left: `${5 + (i * 7) % 90}%`,
              top: `${10 + (i * 11) % 80}%`,
              animationDelay: `${(i * 0.4) % 4}s`,
              animationDuration: `${4 + (i % 3)}s`
            }}
          />
        ))}
      </div>

      {/* Neural Sparks */}
      <div className="absolute inset-0">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-cyan-400/60 rounded-full animate-neural-spark"
            style={{
              left: `${20 + (i * 15) % 60}%`,
              top: `${15 + (i * 20) % 70}%`,
              animationDelay: `${(i * 1.2) % 6}s`
            }}
          />
        ))}
      </div>

      {/* Gradient Overlays */}
      <div className="absolute inset-0 bg-gradient-to-br from-purple-900/5 via-transparent to-cyan-900/5" />
      <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-purple-500/5 to-transparent" />
    </div>
  )
}

export default QuantumBackground
