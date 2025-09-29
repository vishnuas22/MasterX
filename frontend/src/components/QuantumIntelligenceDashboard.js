import React, { useState, useEffect } from 'react';
import { useQuantumIntelligence } from '../providers/QuantumIntelligenceProvider';

export const QuantumIntelligenceDashboard = () => {
  const { systemHealth, updateSystemHealth } = useQuantumIntelligence();
  const [performanceMetrics, setPerformanceMetrics] = useState({
    response_times: { avg_ms: 0, p95_ms: 0, p99_ms: 0 },
    real_ai_metrics: { avg_response_time_ms: 0, success_rate: 0 },
    system: { cpu_percent: 0, memory_percent: 0 },
    cache: { hit_rate: 0, efficiency: 'low' },
    connections: { status: 'unknown', pool_utilization_pct: 0 }
  });

  useEffect(() => {
    if (systemHealth.performance) {
      setPerformanceMetrics(systemHealth.performance);
    }
  }, [systemHealth]);

  const getHealthColor = (score) => {
    if (score >= 0.8) return 'text-quantum-success';
    if (score >= 0.6) return 'text-quantum-warning';
    return 'text-quantum-danger';
  };

  const getStatusColor = (status) => {
    const colors = {
      'operational': 'text-quantum-success',
      'healthy': 'text-quantum-success',
      'degraded': 'text-quantum-warning',
      'error': 'text-quantum-danger',
      'unknown': 'text-quantum-accent'
    };
    return colors[status] || 'text-quantum-accent';
  };

  const formatMs = (ms) => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="quantum-intelligence-dashboard">
      <div className="dashboard-header">
        <h2 className="text-3xl font-bold text-quantum-primary mb-2">
          🧠 Quantum Intelligence Dashboard V6.0
        </h2>
        <p className="text-quantum-accent mb-6">
          Real-time monitoring of the world's most advanced AI learning system
        </p>
        
        <button
          onClick={updateSystemHealth}
          className="refresh-button"
          data-testid="refresh-dashboard"
        >
          🔄 Refresh System Status
        </button>
      </div>

      <div className="dashboard-grid">
        {/* System Health Overview */}
        <div className="dashboard-card system-health">
          <div className="card-header">
            <h3>🎯 System Health</h3>
            <div className={`health-score ${getHealthColor(systemHealth.performance?.health_score || 0)}`}>
              {Math.round((systemHealth.performance?.health_score || 0) * 100)}%
            </div>
          </div>
          <div className="health-indicators">
            <div className="health-item">
              <span>Status:</span>
              <span className={getStatusColor(systemHealth.status)}>
                {systemHealth.status?.toUpperCase()}
              </span>
            </div>
            <div className="health-item">
              <span>Quantum Intelligence:</span>
              <span className={systemHealth.quantum_intelligence?.available ? 'text-quantum-success' : 'text-quantum-danger'}>
                {systemHealth.quantum_intelligence?.available ? 'OPERATIONAL' : 'UNAVAILABLE'}
              </span>
            </div>
            <div className="health-item">
              <span>Engine Status:</span>
              <span className="text-quantum-success">
                {systemHealth.quantum_intelligence?.engine_status?.toUpperCase() || 'UNKNOWN'}
              </span>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="dashboard-card performance-metrics">
          <div className="card-header">
            <h3>⚡ Performance Metrics</h3>
          </div>
          <div className="metrics-grid">
            <div className="metric-item">
              <div className="metric-label">Average Response Time</div>
              <div className="metric-value text-quantum-primary">
                {formatMs(performanceMetrics.response_times?.avg_ms || 0)}
              </div>
              <div className="metric-target">
                Target: {formatMs(performanceMetrics.response_times?.target_ms || 15000)}
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">95th Percentile</div>
              <div className="metric-value text-quantum-accent">
                {formatMs(performanceMetrics.response_times?.p95_ms || 0)}
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">99th Percentile</div>
              <div className="metric-value text-quantum-secondary">
                {formatMs(performanceMetrics.response_times?.p99_ms || 0)}
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Real AI Success Rate</div>
              <div className="metric-value text-quantum-success">
                {Math.round((performanceMetrics.real_ai_metrics?.success_rate || 0) * 100)}%
              </div>
            </div>
          </div>
        </div>

        {/* AI Provider Status */}
        <div className="dashboard-card ai-providers">
          <div className="card-header">
            <h3>🤖 AI Provider Status</h3>
          </div>
          <div className="providers-list">
            <div className="provider-item">
              <div className="provider-info">
                <span className="provider-name">🚀 Groq</span>
                <span className="provider-model">llama-3.3-70b-versatile</span>
              </div>
              <div className="provider-status operational">OPERATIONAL</div>
            </div>
            <div className="provider-item">
              <div className="provider-info">
                <span className="provider-name">🧠 Emergent LLM</span>
                <span className="provider-model">openai/gpt-4o</span>
              </div>
              <div className="provider-status operational">OPERATIONAL</div>
            </div>
            <div className="provider-item">
              <div className="provider-info">
                <span className="provider-name">💎 Gemini</span>
                <span className="provider-model">gemini-2.5-flash</span>
              </div>
              <div className="provider-status operational">OPERATIONAL</div>
            </div>
          </div>
        </div>

        {/* System Resources */}
        <div className="dashboard-card system-resources">
          <div className="card-header">
            <h3>💻 System Resources</h3>
          </div>
          <div className="resource-metrics">
            <div className="resource-item">
              <div className="resource-header">
                <span>CPU Usage</span>
                <span>{Math.round(performanceMetrics.system?.cpu_percent || 0)}%</span>
              </div>
              <div className="resource-bar">
                <div 
                  className="resource-fill cpu"
                  style={{ width: `${performanceMetrics.system?.cpu_percent || 0}%` }}
                />
              </div>
            </div>
            <div className="resource-item">
              <div className="resource-header">
                <span>Memory Usage</span>
                <span>{Math.round(performanceMetrics.system?.memory_percent || 0)}%</span>
              </div>
              <div className="resource-bar">
                <div 
                  className="resource-fill memory"
                  style={{ width: `${performanceMetrics.system?.memory_percent || 0}%` }}
                />
              </div>
            </div>
            <div className="resource-item">
              <div className="resource-header">
                <span>Available Memory</span>
                <span>{Math.round(performanceMetrics.system?.memory_available_gb || 0)} GB</span>
              </div>
            </div>
          </div>
        </div>

        {/* Cache Performance */}
        <div className="dashboard-card cache-performance">
          <div className="card-header">
            <h3>🧮 Cache Performance</h3>
          </div>
          <div className="cache-metrics">
            <div className="cache-item">
              <span>Hit Rate:</span>
              <span className="text-quantum-primary">{performanceMetrics.cache?.hit_rate || 0}%</span>
            </div>
            <div className="cache-item">
              <span>Efficiency:</span>
              <span className={`cache-efficiency-${performanceMetrics.cache?.efficiency || 'low'}`}>
                {(performanceMetrics.cache?.efficiency || 'low').toUpperCase()}
              </span>
            </div>
            <div className="cache-item">
              <span>Real AI Cached:</span>
              <span className="text-quantum-accent">{performanceMetrics.cache?.real_ai_responses_cached || 0}</span>
            </div>
            <div className="cache-item">
              <span>Total Hits:</span>
              <span className="text-quantum-success">{performanceMetrics.cache?.hits || 0}</span>
            </div>
          </div>
        </div>

        {/* Connection Pool Status */}
        <div className="dashboard-card connection-pool">
          <div className="card-header">
            <h3>🔗 Connection Pool</h3>
          </div>
          <div className="connection-metrics">
            <div className="connection-item">
              <span>Status:</span>
              <span className={getStatusColor(performanceMetrics.connections?.status)}>
                {performanceMetrics.connections?.status?.toUpperCase() || 'UNKNOWN'}
              </span>
            </div>
            <div className="connection-item">
              <span>Pool Utilization:</span>
              <span className="text-quantum-primary">
                {Math.round(performanceMetrics.connections?.pool_utilization_pct || 0)}%
              </span>
            </div>
            <div className="connection-item">
              <span>Active Connections:</span>
              <span className="text-quantum-accent">
                {performanceMetrics.connections?.metrics?.active_connections || 0}
              </span>
            </div>
            <div className="connection-item">
              <span>Circuit Breaker:</span>
              <span className="text-quantum-success">
                {performanceMetrics.connections?.circuit_breaker_state || 'UNKNOWN'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .quantum-intelligence-dashboard {
          padding: 1rem;
        }

        .dashboard-header {
          margin-bottom: 2rem;
          text-align: center;
        }

        .refresh-button {
          background: var(--quantum-gradient-primary);
          color: var(--quantum-dark);
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: var(--quantum-border-radius);
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .refresh-button:hover {
          transform: translateY(-2px);
          box-shadow: var(--quantum-shadow);
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 1.5rem;
        }

        .dashboard-card {
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
          color: var(--quantum-primary);
          font-size: 1.2rem;
          font-weight: 700;
        }

        .health-score {
          font-size: 1.5rem;
          font-weight: 800;
        }

        .health-indicators,
        .cache-metrics,
        .connection-metrics {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .health-item,
        .cache-item,
        .connection-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 0;
          border-bottom: 1px solid rgba(48, 54, 61, 0.5);
        }

        .health-item:last-child,
        .cache-item:last-child,
        .connection-item:last-child {
          border-bottom: none;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }

        .metric-item {
          text-align: center;
          padding: 1rem;
          background: rgba(13, 17, 23, 0.4);
          border-radius: var(--quantum-border-radius);
        }

        .metric-label {
          font-size: 0.8rem;
          color: var(--quantum-accent);
          margin-bottom: 0.5rem;
        }

        .metric-value {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 0.25rem;
        }

        .metric-target {
          font-size: 0.7rem;
          color: var(--quantum-accent);
        }

        .providers-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .provider-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          background: rgba(13, 17, 23, 0.4);
          border-radius: var(--quantum-border-radius);
        }

        .provider-info {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .provider-name {
          font-weight: 600;
          color: var(--quantum-primary);
        }

        .provider-model {
          font-size: 0.8rem;
          color: var(--quantum-accent);
        }

        .provider-status {
          padding: 0.25rem 0.75rem;
          border-radius: 20px;
          font-size: 0.8rem;
          font-weight: 600;
        }

        .provider-status.operational {
          background: rgba(0, 255, 136, 0.2);
          color: var(--quantum-success);
          border: 1px solid var(--quantum-success);
        }

        .resource-metrics {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .resource-item {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .resource-header {
          display: flex;
          justify-content: space-between;
          font-weight: 500;
        }

        .resource-bar {
          height: 8px;
          background: rgba(48, 54, 61, 0.8);
          border-radius: 4px;
          overflow: hidden;
        }

        .resource-fill {
          height: 100%;
          border-radius: 4px;
          transition: width 0.3s ease;
        }

        .resource-fill.cpu {
          background: var(--quantum-primary);
        }

        .resource-fill.memory {
          background: var(--quantum-accent);
        }

        .cache-efficiency-high {
          color: var(--quantum-success);
        }

        .cache-efficiency-medium {
          color: var(--quantum-warning);
        }

        .cache-efficiency-low {
          color: var(--quantum-danger);
        }

        @media (max-width: 768px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }

          .metrics-grid {
            grid-template-columns: 1fr;
          }

          .dashboard-card {
            padding: 1rem;
          }
        }
      `}</style>
    </div>
  );
};