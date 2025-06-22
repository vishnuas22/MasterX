// Environment configuration for MasterX AI Mentor System
// This file ensures the app works in any environment

export const getEnvironmentConfig = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  const port = window.location.port;
  
  // Configuration for different environments
  const configs = {
    // Preview environment
    preview: {
      condition: () => hostname.includes('emergentagent.com'),
      backendURL: `${protocol}//${hostname}`
    },
    
    // Local development
    local: {
      condition: () => hostname === 'localhost' || hostname === '127.0.0.1',
      backendURL: 'http://localhost:8001'
    },
    
    // Production or other environments
    production: {
      condition: () => true, // Default fallback
      backendURL: process.env.REACT_APP_BACKEND_URL || `${protocol}//${hostname}:8001`
    }
  };
  
  // Find matching configuration
  for (const [envName, config] of Object.entries(configs)) {
    if (config.condition()) {
      console.log(`🌍 Environment detected: ${envName}`);
      console.log(`🔗 Backend URL: ${config.backendURL}`);
      return {
        environment: envName,
        backendURL: config.backendURL,
        apiURL: `${config.backendURL}/api`
      };
    }
  }
  
  // This should never happen, but just in case
  return configs.production;
};

export default getEnvironmentConfig;