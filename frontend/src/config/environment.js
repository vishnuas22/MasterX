// Environment configuration for MasterX AI Mentor System
// This file ensures the app works in any environment - TRULY PORTABLE!

// Helper function to detect if we're in preview environment
const isPreviewEnvironment = () => {
  const hostname = window.location.hostname;
  return hostname.includes('emergentagent.com') || 
         process.env.REACT_APP_BACKEND_URL?.includes('emergentagent.com') ||
         process.env.REACT_APP_BACKEND_URL?.includes('preview');
};

// Helper function to detect if we're in local development
const isLocalEnvironment = () => {
  const hostname = window.location.hostname;
  return hostname === 'localhost' || 
         hostname === '127.0.0.1' || 
         hostname.startsWith('192.168.') || 
         hostname.startsWith('10.') || 
         hostname.endsWith('.local');
};

export const getEnvironmentConfig = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  const port = window.location.port;
  
  console.log(`🌍 Detecting environment for hostname: ${hostname}, protocol: ${protocol}, port: ${port}`);
  console.log(`🔧 REACT_APP_BACKEND_URL: ${process.env.REACT_APP_BACKEND_URL}`);
  
  // Configuration for different environments with improved detection
  let config;
  
  if (isLocalEnvironment()) {
    // Local development environment
    config = {
      environment: 'local',
      backendURL: 'http://localhost:8001',
      apiURL: 'http://localhost:8001/api'
    };
  } else if (isPreviewEnvironment()) {
    // Preview environment
    const backendURL = process.env.REACT_APP_BACKEND_URL || `${protocol}//${hostname}`;
    config = {
      environment: 'preview',
      backendURL: backendURL,
      apiURL: `${backendURL}/api`
    };
  } else {
    // Production or other environments
    const backendURL = process.env.REACT_APP_BACKEND_URL || `${protocol}//${hostname}:8001`;
    config = {
      environment: 'production',
      backendURL: backendURL,
      apiURL: `${backendURL}/api`
    };
  }
  
  console.log(`🌍 Environment detected: ${config.environment}`);
  console.log(`🔗 Backend URL: ${config.backendURL}`);
  console.log(`🚀 API URL: ${config.apiURL}`);
  
  return config;
};

export default getEnvironmentConfig;