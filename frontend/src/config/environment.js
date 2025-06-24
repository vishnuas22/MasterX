// MasterX AI Mentor System - Practical Environment Configuration
// Hybrid approach: Supports both hardcoded URLs and auto-detection for maximum reliability

// Environment detection functions
const isLocalDevelopment = () => {
  const hostname = window.location.hostname;
  return hostname === 'localhost' || 
         hostname === '127.0.0.1' || 
         hostname.startsWith('192.168.') || 
         hostname.startsWith('10.') || 
         hostname.endsWith('.local');
};

const isEmergentPreview = () => {
  const hostname = window.location.hostname;
  return hostname.includes('emergentagent.com') || 
         hostname.includes('preview.emergentagent.com');
};

const hasConfiguredBackendURL = () => {
  return process.env.REACT_APP_BACKEND_URL && 
         process.env.REACT_APP_BACKEND_URL.trim() !== '' &&
         process.env.REACT_APP_BACKEND_URL !== 'undefined';
};

export const getEnvironmentConfig = () => {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  const port = window.location.port;
  
  console.log(`🌍 MasterX Environment Detection`);
  console.log(`📍 Hostname: ${hostname}, Protocol: ${protocol}, Port: ${port}`);
  console.log(`⚙️ Configured Backend URL: ${process.env.REACT_APP_BACKEND_URL || 'NOT SET'}`);
  
  let config;
  
  // PRIORITY 1: Local Development (Always prioritize localhost for development)
  if (isLocalDevelopment()) {
    config = {
      environment: 'local',
      backendURL: 'http://localhost:8001',
      apiURL: 'http://localhost:8001/api',
      fallbacks: [
        'http://127.0.0.1:8001',
        'http://localhost:3001', // Alternative port
      ]
    };
    console.log(`🏠 Local Development Mode - Using localhost:8001`);
    
  // PRIORITY 2: Configured Backend URL (Use hardcoded URLs when available)
  } else if (hasConfiguredBackendURL()) {
    const backendURL = process.env.REACT_APP_BACKEND_URL;
    config = {
      environment: isEmergentPreview() ? 'preview' : 'production',
      backendURL: backendURL,
      apiURL: `${backendURL}/api`,
      fallbacks: [
        `${protocol}//${hostname}`, // Current host as fallback
        `${protocol}//${hostname}:8001` // With port fallback
      ]
    };
    console.log(`🌐 Using Configured Backend URL: ${backendURL}`);
    
  // PRIORITY 3: Auto-detection for Emergent Preview
  } else if (isEmergentPreview()) {
    const backendURL = `${protocol}//${hostname}`;
    config = {
      environment: 'preview',
      backendURL: backendURL,
      apiURL: `${backendURL}/api`,
      fallbacks: [
        `${protocol}//${hostname}:8001`,
        `${protocol}//${hostname}:3001`
      ]
    };
    console.log(`🌐 Emergent Preview Auto-Detection: ${backendURL}`);
    
  // PRIORITY 4: Generic Production/Unknown Environment
  } else {
    const backendURL = `${protocol}//${hostname}:8001`;
    config = {
      environment: 'production',
      backendURL: backendURL,
      apiURL: `${backendURL}/api`,
      fallbacks: [
        `${protocol}//${hostname}`,
        `${protocol}//${hostname}:3001`,
        'http://localhost:8001' // Last resort
      ]
    };
    console.log(`🚀 Production Mode: ${backendURL}`);
  }
  
  console.log(`✅ Environment: ${config.environment.toUpperCase()}`);
  console.log(`🔗 Backend URL: ${config.backendURL}`);
  console.log(`🚀 API URL: ${config.apiURL}`);
  console.log(`🔄 Fallbacks: ${config.fallbacks.join(', ')}`);
  
  return config;
};

export default getEnvironmentConfig;