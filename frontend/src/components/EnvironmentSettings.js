import React, { useState, useEffect } from 'react';
import { getEnvironmentConfig } from '../config/environment';
import { testBackendHealth } from '../utils/connectionManager';

const EnvironmentSettings = ({ onClose }) => {
  const [envConfig, setEnvConfig] = useState(null);
  const [testResults, setTestResults] = useState({});
  const [isTesting, setIsTesting] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    const config = getEnvironmentConfig();
    setEnvConfig(config);
  }, []);

  const testURL = async (url) => {
    setIsTesting(true);
    try {
      const response = await fetch(`${url}/api/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        const data = await response.json();
        return { success: true, status: data.status, latency: '< 1s' };
      }
      return { success: false, error: `HTTP ${response.status}` };
    } catch (error) {
      return { success: false, error: error.message };
    } finally {
      setIsTesting(false);
    }
  };

  const testAllURLs = async () => {
    if (!envConfig) return;
    
    setIsTesting(true);
    const results = {};
    
    // Test primary URL
    results[envConfig.backendURL] = await testURL(envConfig.backendURL);
    
    // Test fallbacks
    if (envConfig.fallbacks) {
      for (const fallback of envConfig.fallbacks) {
        results[fallback] = await testURL(fallback);
      }
    }
    
    setTestResults(results);
    setIsTesting(false);
  };

  const getStatusIcon = (result) => {
    if (!result) return '⏳';
    return result.success ? '✅' : '❌';
  };

  const getEnvironmentInfo = () => {
    return {
      local: {
        name: 'Local Development',
        description: 'Running on localhost for development',
        icon: '🏠',
        color: 'text-blue-400'
      },
      preview: {
        name: 'Emergent Preview',
        description: 'Running on emergent.sh preview environment',
        icon: '🌐',
        color: 'text-purple-400'
      },
      production: {
        name: 'Production',
        description: 'Production deployment',
        icon: '🚀',
        color: 'text-green-400'
      },
      unknown: {
        name: 'Unknown Environment',
        description: 'Environment could not be determined',
        icon: '❓',
        color: 'text-gray-400'
      }
    };
  };

  if (!envConfig) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-white">Loading environment configuration...</div>
      </div>
    );
  }

  const environmentInfo = getEnvironmentInfo()[envConfig.environment] || getEnvironmentInfo().unknown;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-gray-900/95 backdrop-blur-xl border border-gray-700 rounded-2xl p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">MasterX Connection Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Current Environment */}
        <div className="bg-gray-800/50 rounded-xl p-4 mb-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-2xl">{environmentInfo.icon}</span>
            <div>
              <h3 className={`text-lg font-semibold ${environmentInfo.color}`}>
                {environmentInfo.name}
              </h3>
              <p className="text-gray-400 text-sm">{environmentInfo.description}</p>
            </div>
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Primary Backend URL:</span>
              <span className="text-white font-mono">{envConfig.backendURL}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">API Endpoint:</span>
              <span className="text-white font-mono">{envConfig.apiURL}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Environment Variable:</span>
              <span className="text-white font-mono">
                {process.env.REACT_APP_BACKEND_URL || 'Not Set'}
              </span>
            </div>
          </div>
        </div>

        {/* Connection Test */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Connection Test</h3>
            <button
              onClick={testAllURLs}
              disabled={isTesting}
              className={`px-4 py-2 rounded-lg transition-colors ${
                isTesting 
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isTesting ? 'Testing...' : 'Test All URLs'}
            </button>
          </div>

          <div className="space-y-2">
            {/* Primary URL */}
            <div className="bg-gray-800/30 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span>{getStatusIcon(testResults[envConfig.backendURL])}</span>
                  <span className="text-white font-mono text-sm">{envConfig.backendURL}</span>
                  <span className="px-2 py-1 bg-blue-600 text-white text-xs rounded">PRIMARY</span>
                </div>
                {testResults[envConfig.backendURL]?.error && (
                  <span className="text-red-400 text-xs">
                    {testResults[envConfig.backendURL].error}
                  </span>
                )}
              </div>
            </div>

            {/* Fallback URLs */}
            {envConfig.fallbacks && envConfig.fallbacks.map((fallback, index) => (
              <div key={fallback} className="bg-gray-800/30 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span>{getStatusIcon(testResults[fallback])}</span>
                    <span className="text-white font-mono text-sm">{fallback}</span>
                    <span className="px-2 py-1 bg-gray-600 text-white text-xs rounded">
                      FALLBACK {index + 1}
                    </span>
                  </div>
                  {testResults[fallback]?.error && (
                    <span className="text-red-400 text-xs">
                      {testResults[fallback].error}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="mb-6">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-white hover:text-blue-400 transition-colors"
          >
            <span className={`transform transition-transform ${showAdvanced ? 'rotate-90' : ''}`}>
              ▶
            </span>
            Advanced Configuration
          </button>

          {showAdvanced && (
            <div className="mt-4 bg-gray-800/30 rounded-lg p-4">
              <div className="text-sm text-gray-300 space-y-2">
                <div><strong>Browser Location:</strong> {window.location.href}</div>
                <div><strong>Hostname:</strong> {window.location.hostname}</div>
                <div><strong>Protocol:</strong> {window.location.protocol}</div>
                <div><strong>Port:</strong> {window.location.port || 'default'}</div>
                <div><strong>User Agent:</strong> {navigator.userAgent}</div>
              </div>
            </div>
          )}
        </div>

        {/* Recommendations */}
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <h4 className="text-blue-400 font-semibold mb-2">💡 Connection Tips</h4>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• <strong>Local Development:</strong> Make sure your backend is running on localhost:8001</li>
            <li>• <strong>Preview Environment:</strong> The system automatically detects emergent.sh URLs</li>
            <li>• <strong>Network Issues:</strong> Check your firewall and internet connection</li>
            <li>• <strong>CORS Errors:</strong> The backend should have proper CORS configuration</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EnvironmentSettings;