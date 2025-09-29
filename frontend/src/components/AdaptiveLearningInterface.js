import React, { useState, useEffect } from 'react';
import { useQuantumIntelligence } from '../providers/QuantumIntelligenceProvider';

export const AdaptiveLearningInterface = () => {
  const { adaptiveMetrics } = useQuantumIntelligence();
  const [learningDNA, setLearningDNA] = useState({
    learning_style: 'visual',
    pace_preference: 'moderate',
    difficulty_tolerance: 0.7,
    engagement_patterns: ['interactive', 'gamified'],
    preferred_modalities: ['visual', 'auditory'],
    quantum_coherence: 0
  });
  const [adaptationHistory, setAdaptationHistory] = useState([]);
  const [personalizedContent, setPersonalizedContent] = useState(null);

  // Mock learning DNA analysis
  useEffect(() => {
    setLearningDNA(prev => ({
      ...prev,
      quantum_coherence: adaptiveMetrics.quantum_coherence || 0
    }));
  }, [adaptiveMetrics]);

  // Mock adaptation recommendations
  const generateAdaptations = () => {
    const adaptations = [
      {
        id: Date.now(),
        type: 'difficulty_adjustment',
        strategy: 'Reduce complexity by 20%',
        confidence: 0.92,
        reasoning: 'User showing signs of cognitive overload',
        expected_improvement: 0.75,
        timestamp: Date.now()
      },
      {
        id: Date.now() + 1,
        type: 'learning_modality',
        strategy: 'Add visual diagrams and interactive elements',
        confidence: 0.88,
        reasoning: 'User responds well to visual learning',
        expected_improvement: 0.68,
        timestamp: Date.now()
      },
      {
        id: Date.now() + 2,
        type: 'pacing_adjustment',
        strategy: 'Increase practice time by 30%',
        confidence: 0.85,
        reasoning: 'User needs more time to master concepts',
        expected_improvement: 0.72,
        timestamp: Date.now()
      }
    ];

    setAdaptationHistory(prev => [...adaptations, ...prev.slice(0, 7)]);
  };

  const getDifficultyColor = (score) => {
    if (score < 0.3) return 'text-quantum-success';
    if (score < 0.7) return 'text-quantum-warning';
    return 'text-quantum-danger';
  };

  const getCoherenceColor = (coherence) => {
    if (coherence > 0.8) return 'text-quantum-success';
    if (coherence > 0.5) return 'text-quantum-warning';
    return 'text-quantum-danger';
  };

  return (
    <div className="adaptive-learning-interface">
      <div className="interface-header">
        <h2 className="text-3xl font-bold text-quantum-accent mb-2">
          🎯 Adaptive Learning Engine
        </h2>
        <p className="text-quantum-accent mb-6">
          Revolutionary real-time personalization with Quantum Difficulty Scaling
        </p>
      </div>

      <div className="interface-grid">
        {/* Learning DNA Profile */}
        <div className="learning-card learning-dna">
          <div className="card-header">
            <h3>🧬 Learning DNA Profile</h3>
          </div>
          <div className="dna-visualization">
            <div className="dna-strand">
              <div className="dna-helix">
                <div className="helix-segment" data-gene="style">
                  <span className="gene-label">Learning Style</span>
                  <span className="gene-value">{learningDNA.learning_style.toUpperCase()}</span>
                </div>
                <div className="helix-segment" data-gene="pace">
                  <span className="gene-label">Pace Preference</span>
                  <span className="gene-value">{learningDNA.pace_preference.toUpperCase()}</span>
                </div>
                <div className="helix-segment" data-gene="tolerance">
                  <span className="gene-label">Difficulty Tolerance</span>
                  <span className="gene-value">{Math.round(learningDNA.difficulty_tolerance * 100)}%</span>
                </div>
                <div className="helix-segment" data-gene="coherence">
                  <span className="gene-label">Quantum Coherence</span>
                  <span className={`gene-value ${getCoherenceColor(learningDNA.quantum_coherence)}`}>
                    {Math.round(learningDNA.quantum_coherence * 100)}%
                  </span>
                </div>
              </div>
            </div>
            <div className="dna-patterns">
              <div className="pattern-item">
                <span className="pattern-label">Engagement Patterns:</span>
                <div className="pattern-tags">
                  {learningDNA.engagement_patterns.map(pattern => (
                    <span key={pattern} className="pattern-tag">{pattern}</span>
                  ))}
                </div>
              </div>
              <div className="pattern-item">
                <span className="pattern-label">Preferred Modalities:</span>
                <div className="pattern-tags">
                  {learningDNA.preferred_modalities.map(modality => (
                    <span key={modality} className="modality-tag">{modality}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Real-time Adaptation Metrics */}
        <div className="learning-card adaptation-metrics">
          <div className="card-header">
            <h3>📊 Real-time Adaptation Metrics</h3>
          </div>
          <div className="metrics-display">
            <div className="metric-item">
              <div className="metric-icon">🧠</div>
              <div className="metric-info">
                <div className="metric-label">Comprehension Level</div>
                <div className="metric-value text-quantum-primary">
                  {adaptiveMetrics.comprehension_level?.toUpperCase() || 'UNKNOWN'}
                </div>
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-icon">🎯</div>
              <div className="metric-info">
                <div className="metric-label">Difficulty Score</div>
                <div className={`metric-value ${getDifficultyColor(adaptiveMetrics.difficulty_score)}`}>
                  {Math.round(adaptiveMetrics.difficulty_score * 100)}%
                </div>
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-icon">🌌</div>
              <div className="metric-info">
                <div className="metric-label">Quantum Coherence</div>
                <div className={`metric-value ${getCoherenceColor(adaptiveMetrics.quantum_coherence)}`}>
                  {Math.round(adaptiveMetrics.quantum_coherence * 100)}%
                </div>
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-icon">⚡</div>
              <div className="metric-info">
                <div className="metric-label">Learning Velocity</div>
                <div className="metric-value text-quantum-success">OPTIMAL</div>
              </div>
            </div>
          </div>
        </div>

        {/* Quantum Difficulty Scaling */}
        <div className="learning-card difficulty-scaling">
          <div className="card-header">
            <h3>🎚️ Quantum Difficulty Scaling (QDS)</h3>
            <button
              onClick={generateAdaptations}
              className="generate-button"
              data-testid="generate-adaptations"
            >
              🚀 Generate Adaptations
            </button>
          </div>
          <div className="scaling-controls">
            <div className="control-item">
              <label>Base Difficulty</label>
              <div className="difficulty-slider">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={adaptiveMetrics.difficulty_score}
                  readOnly
                  className="slider"
                />
                <span className="slider-value">
                  {Math.round(adaptiveMetrics.difficulty_score * 100)}%
                </span>
              </div>
            </div>
            <div className="control-item">
              <label>Adaptive Range</label>
              <div className="range-indicator">
                <div className="range-bar">
                  <div 
                    className="range-fill"
                    style={{ width: '70%' }}
                  />
                </div>
                <span>±35% adjustment range</span>
              </div>
            </div>
            <div className="control-item">
              <label>Scaling Sensitivity</label>
              <div className="sensitivity-gauge">
                <div className="gauge-indicator moderate">MODERATE</div>
              </div>
            </div>
          </div>
        </div>

        {/* Adaptation History */}
        <div className="learning-card adaptation-history">
          <div className="card-header">
            <h3>📈 Adaptation History</h3>
          </div>
          <div className="adaptations-timeline">
            {adaptationHistory.length > 0 ? (
              adaptationHistory.map(adaptation => (
                <div key={adaptation.id} className="adaptation-item">
                  <div className="adaptation-header">
                    <span className="adaptation-type">
                      {adaptation.type.replace('_', ' ').toUpperCase()}
                    </span>
                    <span className="adaptation-confidence">
                      {Math.round(adaptation.confidence * 100)}% confident
                    </span>
                  </div>
                  <div className="adaptation-strategy">
                    {adaptation.strategy}
                  </div>
                  <div className="adaptation-details">
                    <div className="detail-item">
                      <span>Reasoning:</span>
                      <span>{adaptation.reasoning}</span>
                    </div>
                    <div className="detail-item">
                      <span>Expected Improvement:</span>
                      <span className="text-quantum-success">
                        +{Math.round(adaptation.expected_improvement * 100)}%
                      </span>
                    </div>
                    <div className="detail-item">
                      <span>Applied:</span>
                      <span>{new Date(adaptation.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="no-adaptations">
                <span>🔄</span>
                <p>No adaptations yet. Click "Generate Adaptations" to see personalized learning recommendations.</p>
              </div>
            )}
          </div>
        </div>

        {/* Personalization Engine */}
        <div className="learning-card personalization-engine">
          <div className="card-header">
            <h3>🎨 Personalization Engine</h3>
          </div>
          <div className="engine-features">
            <div className="feature-group">
              <h4>🧠 Cognitive Adaptation</h4>
              <div className="feature-list">
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Dynamic difficulty adjustment</span>
                </div>
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Cognitive load optimization</span>
                </div>
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Attention span management</span>
                </div>
              </div>
            </div>
            <div className="feature-group">
              <h4>🎭 Emotional Intelligence</h4>
              <div className="feature-list">
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Stress level monitoring</span>
                </div>
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Motivation enhancement</span>
                </div>
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Confidence building</span>
                </div>
              </div>
            </div>
            <div className="feature-group">
              <h4>⚡ Performance Optimization</h4>
              <div className="feature-list">
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Learning velocity tracking</span>
                </div>
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Retention rate optimization</span>
                </div>
                <div className="feature-item active">
                  <span className="feature-dot"></span>
                  <span>Success prediction modeling</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .adaptive-learning-interface {
          padding: 1rem;
        }

        .interface-header {
          text-align: center;
          margin-bottom: 2rem;
        }

        .interface-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 1.5rem;
        }

        .learning-card {
          background: rgba(33, 38, 45, 0.6);
          border: 1px solid var(--quantum-border);
          border-radius: var(--quantum-border-radius);
          padding: 1.5rem;
          backdrop-filter: blur(10px);
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .card-header h3 {
          color: var(--quantum-accent);
          font-size: 1.2rem;
          font-weight: 700;
        }

        .generate-button {
          background: var(--quantum-gradient-primary);
          color: var(--quantum-dark);
          border: none;
          padding: 0.5rem 1rem;
          border-radius: var(--quantum-border-radius);
          font-weight: 600;
          cursor: pointer;
          font-size: 0.8rem;
          transition: all 0.3s ease;
        }

        .generate-button:hover {
          transform: translateY(-2px);
          box-shadow: var(--quantum-shadow);
        }

        .dna-visualization {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .dna-strand {
          background: linear-gradient(45deg, rgba(0, 245, 255, 0.1), rgba(199, 125, 255, 0.1));
          border-radius: var(--quantum-border-radius);
          padding: 1rem;
        }

        .dna-helix {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }

        .helix-segment {
          background: rgba(13, 17, 23, 0.6);
          border-radius: 8px;
          padding: 1rem;
          text-align: center;
          border: 1px solid var(--quantum-border);
        }

        .gene-label {
          display: block;
          font-size: 0.8rem;
          color: var(--quantum-accent);
          margin-bottom: 0.5rem;
        }

        .gene-value {
          display: block;
          font-weight: 600;
          font-size: 1rem;
          color: var(--quantum-primary);
        }

        .dna-patterns {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .pattern-item {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .pattern-label {
          font-weight: 600;
          color: var(--quantum-accent);
        }

        .pattern-tags {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
        }

        .pattern-tag,
        .modality-tag {
          background: var(--quantum-accent);
          color: var(--quantum-dark);
          padding: 0.25rem 0.75rem;
          border-radius: 20px;
          font-size: 0.8rem;
          font-weight: 500;
        }

        .modality-tag {
          background: var(--quantum-secondary);
        }

        .metrics-display {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }

        .metric-item {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: rgba(13, 17, 23, 0.4);
          border-radius: var(--quantum-border-radius);
        }

        .metric-icon {
          font-size: 1.5rem;
        }

        .metric-info {
          flex: 1;
        }

        .metric-label {
          font-size: 0.8rem;
          color: var(--quantum-accent);
          margin-bottom: 0.25rem;
        }

        .metric-value {
          font-weight: 600;
          font-size: 1rem;
        }

        .scaling-controls {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .control-item {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .control-item label {
          font-weight: 600;
          color: var(--quantum-accent);
        }

        .difficulty-slider {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .slider {
          flex: 1;
          height: 8px;
          border-radius: 4px;
          background: var(--quantum-gray);
          outline: none;
          cursor: pointer;
        }

        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: var(--quantum-primary);
          cursor: pointer;
        }

        .slider-value {
          font-weight: 600;
          color: var(--quantum-primary);
        }

        .range-indicator {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .range-bar {
          flex: 1;
          height: 8px;
          background: var(--quantum-gray);
          border-radius: 4px;
          overflow: hidden;
        }

        .range-fill {
          height: 100%;
          background: var(--quantum-gradient-primary);
          border-radius: 4px;
        }

        .sensitivity-gauge {
          display: flex;
          justify-content: center;
        }

        .gauge-indicator {
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-weight: 600;
          text-align: center;
        }

        .gauge-indicator.moderate {
          background: rgba(255, 190, 11, 0.2);
          color: var(--quantum-warning);
          border: 1px solid var(--quantum-warning);
        }

        .adaptations-timeline {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          max-height: 400px;
          overflow-y: auto;
        }

        .adaptation-item {
          background: rgba(13, 17, 23, 0.4);
          border-radius: var(--quantum-border-radius);
          padding: 1rem;
          border-left: 3px solid var(--quantum-accent);
        }

        .adaptation-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .adaptation-type {
          font-weight: 600;
          color: var(--quantum-primary);
          font-size: 0.8rem;
        }

        .adaptation-confidence {
          color: var(--quantum-success);
          font-size: 0.8rem;
        }

        .adaptation-strategy {
          font-weight: 500;
          margin-bottom: 0.75rem;
          color: var(--quantum-light);
        }

        .adaptation-details {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .detail-item {
          display: flex;
          justify-content: space-between;
          font-size: 0.8rem;
        }

        .detail-item span:first-child {
          color: var(--quantum-accent);
        }

        .no-adaptations {
          text-align: center;
          padding: 2rem;
          color: var(--quantum-accent);
        }

        .no-adaptations span {
          font-size: 2rem;
          display: block;
          margin-bottom: 1rem;
        }

        .engine-features {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .feature-group h4 {
          color: var(--quantum-primary);
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
        }

        .feature-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .feature-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.5rem;
          border-radius: 6px;
          background: rgba(13, 17, 23, 0.3);
        }

        .feature-item.active {
          background: rgba(0, 245, 255, 0.05);
          border: 1px solid rgba(0, 245, 255, 0.2);
        }

        .feature-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--quantum-success);
          flex-shrink: 0;
        }

        @media (max-width: 768px) {
          .interface-grid {
            grid-template-columns: 1fr;
          }

          .metrics-display,
          .dna-helix {
            grid-template-columns: 1fr;
          }

          .pattern-tags {
            justify-content: center;
          }
        }
      `}</style>
    </div>
  );
};