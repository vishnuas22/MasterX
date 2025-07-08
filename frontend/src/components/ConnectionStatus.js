import React, { useState, useEffect } from 'react';
import { getConnectionHealth, refreshConnection } from '../services/api';
import { testBackendHealth } from '../utils/connectionManager';

const ConnectionStatus = ({ showDetails = false }) => {
  const [status, setStatus] = useState({
    connected: false,
    url: null,
    environment: 'unknown',
    loading: true,
    error: null
  });

  const [showDebug, setShowDebug] = useState(showDetails);

  const checkConnection = async () => {
    setStatus(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      // Test backend health
      const healthResult = await testBackendHealth();
      
      if (healthResult.success) {
        // Get detailed connection status
        const connectionStatus = getConnectionHealth();
        
        setStatus({
          connected: true,
          url: healthResult.url,
          environment: healthResult.environment,
          loading: false,
          error: null,
          details: connectionStatus
        });
      } else {
        setStatus({
          connected: false,
          url: null,
          environment: 'unknown',
          loading: false,
          error: healthResult.error
        });
      }
    } catch (error) {
      setStatus({
        connected: false,
        url: null,
        environment: 'unknown',
        loading: false,
        error: error.message
      });
    }
  };

  const handleRefreshConnection = async () => {
    try {
      await refreshConnection();
      await checkConnection();
    } catch (error) {
      console.error('Failed to refresh connection:', error);
    }
  };

  useEffect(() => {
    checkConnection();
  }, []);

  const getStatusColor = () => {
    if (status.loading) return 'text-yellow-500';
    return status.connected ? 'text-green-500' : 'text-red-500';
  };

  const getStatusIcon = () => {
    if (status.loading) return '🔄';
    return status.connected ? '✅' : '❌';
  };

  const getEnvironmentBadge = () => {
    const colors = {
      local: 'bg-blue-500',
      preview: 'bg-purple-500',
      production: 'bg-green-500',
      unknown: 'bg-gray-500'
    };
    
    return (
      <span className={`px-2 py-1 text-xs rounded-full text-white ${colors[status.environment] || colors.unknown}`}>
        {status.environment.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 text-white text-sm max-w-sm">
        {/* Main Status */}
        <div className="flex items-center gap-2 mb-2">
          <span className={getStatusColor()}>{getStatusIcon()}</span>
          <span className="font-medium">
            {status.loading ? 'Checking...' : status.connected ? 'Connected' : 'Disconnected'}
          </span>
          {getEnvironmentBadge()}
        </div>

        {/* URL Display */}
        {status.url && (
          <div className="text-xs text-gray-300 mb-2">
            {status.url}
          </div>
        )}

        {/* Error Display */}
        {status.error && (
          <div className="text-xs text-red-400 mb-2">
            Error: {status.error}
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-2">
          <button
            onClick={checkConnection}
            className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 rounded transition-colors"
          >
            Test
          </button>
          <button
            onClick={handleRefreshConnection}
            className="px-2 py-1 text-xs bg-green-600 hover:bg-green-700 rounded transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={() => setShowDebug(!showDebug)}
            className="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-700 rounded transition-colors"
          >
            {showDebug ? 'Hide' : 'Debug'}
          </button>
        </div>

        {/* Debug Information */}
        {showDebug && status.details && (
          <div className="mt-3 pt-3 border-t border-gray-700">
            <div className="text-xs space-y-1">
              <div><strong>Config URL:</strong> {status.details.configuredURL}</div>
              <div><strong>Working URL:</strong> {status.details.workingURL || 'None'}</div>
              <div><strong>Connection Tested:</strong> {status.details.connectionTested ? 'Yes' : 'No'}</div>
              <div><strong>Retries:</strong> {status.details.apiRetries}/{status.details.maxRetries}</div>
              <div><strong>Test in Progress:</strong> {status.details.testInProgress ? 'Yes' : 'No'}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConnectionStatus;