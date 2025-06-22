// Connection test utility for MasterX AI Mentor System
// Ensures robust connection handling across all environments

export class ConnectionManager {
  constructor() {
    this.possibleURLs = this.generatePossibleURLs();
    this.workingURL = null;
    this.testInProgress = false;
  }

  generatePossibleURLs() {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    
    const urls = [];
    
    // Priority order for URL testing
    
    // 1. Environment variable (highest priority)
    if (process.env.REACT_APP_BACKEND_URL) {
      urls.push(process.env.REACT_APP_BACKEND_URL);
    }
    
    // 2. Same origin (for preview environments)
    if (hostname.includes('emergentagent.com')) {
      urls.push(`${protocol}//${hostname}`);
    }
    
    // 3. Local development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      urls.push('http://localhost:8001');
      urls.push('http://127.0.0.1:8001');
    }
    
    // 4. Common port variations
    urls.push(`${protocol}//${hostname}:8001`);
    urls.push(`${protocol}//${hostname}:3001`);
    urls.push(`${protocol}//${hostname}:5000`);
    
    // Remove duplicates
    return [...new Set(urls)];
  }

  async testConnection(baseURL) {
    try {
      // Create an AbortController with a timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(`${baseURL}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal
      });
      
      // Clear the timeout
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        return data.status === 'healthy';
      }
      return false;
    } catch (error) {
      console.log(`Connection test failed for ${baseURL}:`, error.message);
      return false;
    }
  }

  async findWorkingURL() {
    if (this.workingURL && !this.testInProgress) {
      return this.workingURL;
    }

    if (this.testInProgress) {
      // Wait for ongoing test
      while (this.testInProgress) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return this.workingURL;
    }

    this.testInProgress = true;
    console.log('🔍 Testing connections to find working backend...');
    
    for (const url of this.possibleURLs) {
      console.log(`Testing: ${url}`);
      const isWorking = await this.testConnection(url);
      
      if (isWorking) {
        this.workingURL = url;
        console.log(`✅ Found working backend: ${url}`);
        this.testInProgress = false;
        return url;
      }
    }
    
    this.testInProgress = false;
    console.error('❌ No working backend URL found');
    throw new Error('Unable to connect to MasterX AI Mentor System. Please check if the backend is running.');
  }

  async getAPIURL() {
    const baseURL = await this.findWorkingURL();
    return `${baseURL}/api`;
  }

  // Method to force re-test connections (useful for error recovery)
  resetConnection() {
    this.workingURL = null;
    this.testInProgress = false;
  }
}

// Create singleton instance
export const connectionManager = new ConnectionManager();

export default connectionManager;