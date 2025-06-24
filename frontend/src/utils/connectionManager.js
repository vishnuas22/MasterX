// MasterX AI Mentor System - Practical Connection Manager
// Simplified and reliable connection testing with fallback support

import { getEnvironmentConfig } from '../config/environment';

export class ConnectionManager {
  constructor() {
    this.envConfig = getEnvironmentConfig();
    this.workingURL = null;
    this.testInProgress = false;
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1 second
  }

  // Get URLs to test in priority order
  getURLsToTest() {
    const urls = [this.envConfig.backendURL];
    
    // Add fallbacks if they exist and are different from primary
    if (this.envConfig.fallbacks) {
      this.envConfig.fallbacks.forEach(fallback => {
        if (fallback !== this.envConfig.backendURL && !urls.includes(fallback)) {
          urls.push(fallback);
        }
      });
    }
    
    console.log(`🔍 Connection test order:`, urls);
    return urls;
  }

  async testConnection(baseURL, timeout = 5000) {
    try {
      console.log(`🧪 Testing connection to: ${baseURL}`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      
      const response = await fetch(`${baseURL}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache'
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        const isHealthy = data.status === 'healthy';
        console.log(`${isHealthy ? '✅' : '❌'} ${baseURL}: ${data.status}`);
        return isHealthy;
      } else {
        console.log(`❌ ${baseURL}: HTTP ${response.status}`);
        return false;
      }
    } catch (error) {
      const errorType = error.name === 'AbortError' ? 'Timeout' : error.message;
      console.log(`❌ ${baseURL}: ${errorType}`);
      return false;
    }
  }

  async findWorkingURL(forceRetest = false) {
    // Return cached result if available and not forcing retest
    if (this.workingURL && !forceRetest && !this.testInProgress) {
      console.log(`🔄 Using cached connection: ${this.workingURL}`);
      return this.workingURL;
    }

    // If test is already in progress, wait for it
    if (this.testInProgress) {
      console.log('⏳ Connection test in progress, waiting...');
      while (this.testInProgress) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return this.workingURL;
    }

    this.testInProgress = true;
    console.log('🔍 Starting connection discovery...');
    
    try {
      const urlsToTest = this.getURLsToTest();
      
      // Test URLs in priority order
      for (const url of urlsToTest) {
        console.log(`Testing: ${url}`);
        const isWorking = await this.testConnection(url);
        
        if (isWorking) {
          this.workingURL = url;
          console.log(`✅ Found working backend: ${url}`);
          this.testInProgress = false;
          return url;
        }
        
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      // If no URL works, throw error
      this.testInProgress = false;
      const error = `Unable to connect to MasterX backend. Tested: ${urlsToTest.join(', ')}`;
      console.error(`❌ ${error}`);
      throw new Error(error);
      
    } catch (error) {
      this.testInProgress = false;
      throw error;
    }
  }

  async getAPIURL(forceRetest = false) {
    const baseURL = await this.findWorkingURL(forceRetest);
    return `${baseURL}/api`;
  }

  // Quick test of current working URL
  async testCurrentConnection() {
    if (!this.workingURL) {
      return false;
    }
    return await this.testConnection(this.workingURL, 3000);
  }

  // Reset and force new connection test
  resetConnection() {
    console.log('🔄 Resetting connection manager...');
    this.workingURL = null;
    this.testInProgress = false;
    // Update environment config in case it changed
    this.envConfig = getEnvironmentConfig();
  }

  // Get current connection status
  getConnectionStatus() {
    return {
      workingURL: this.workingURL,
      environment: this.envConfig.environment,
      configuredURL: this.envConfig.backendURL,
      hasConnection: !!this.workingURL,
      testInProgress: this.testInProgress
    };
  }
}

// Create singleton instance
export const connectionManager = new ConnectionManager();

// Export a simple function for easy health checks
export const testBackendHealth = async () => {
  try {
    const workingURL = await connectionManager.findWorkingURL();
    return { success: true, url: workingURL, environment: connectionManager.envConfig.environment };
  } catch (error) {
    return { success: false, error: error.message };
  }
};

export default connectionManager;