// Connection test utility for MasterX AI Mentor System
// Ensures robust connection handling across all environments - TRULY PORTABLE!

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
    
    console.log(`🔍 Generating possible URLs for hostname: ${hostname}`);
    
    // Priority order for URL testing - LOCAL DEVELOPMENT FIRST, then PREVIEW!
    
    // 1. Local development URLs (HIGHEST PRIORITY for localhost)
    if (hostname === 'localhost' || hostname === '127.0.0.1' || 
        hostname.startsWith('192.168.') || hostname.startsWith('10.') || 
        hostname.endsWith('.local')) {
      urls.push('http://localhost:8001');
      urls.push('http://127.0.0.1:8001');
      console.log('🏠 Local development detected - prioritizing localhost:8001');
    } else {
      // 2. Preview environment URLs (HIGHEST PRIORITY for non-localhost)
      if (process.env.REACT_APP_BACKEND_URL) {
        urls.push(process.env.REACT_APP_BACKEND_URL);
        console.log(`🌐 Preview URL from env: ${process.env.REACT_APP_BACKEND_URL}`);
      }
      
      // 3. Same origin (for preview environments)
      if (hostname.includes('emergentagent.com')) {
        urls.push(`${protocol}//${hostname}`);
        console.log('🌐 Preview environment detected');
      }
      
      // 4. Common port variations (lower priority)
      urls.push(`${protocol}//${hostname}:8001`);
      urls.push(`${protocol}//${hostname}:3001`);
      urls.push(`${protocol}//${hostname}:5000`);
    }
    
    // Remove duplicates and log final list
    const uniqueUrls = [...new Set(urls)];
    console.log('🔗 Generated URLs to test:', uniqueUrls);
    return uniqueUrls;
  }

  async testConnection(baseURL) {
    try {
      console.log(`🧪 Testing connection to: ${baseURL}`);
      
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
        const isHealthy = data.status === 'healthy';
        console.log(`${isHealthy ? '✅' : '❌'} Health check for ${baseURL}: ${data.status}`);
        return isHealthy;
      }
      console.log(`❌ Health check failed for ${baseURL}: ${response.status}`);
      return false;
    } catch (error) {
      console.log(`❌ Connection test failed for ${baseURL}:`, error.message);
      return false;
    }
  }

  async findWorkingURL() {
    if (this.workingURL && !this.testInProgress) {
      console.log(`🔄 Using cached working URL: ${this.workingURL}`);
      return this.workingURL;
    }

    if (this.testInProgress) {
      // Wait for ongoing test
      console.log('⏳ Waiting for ongoing connection test...');
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
    console.error('❌ No working backend URL found from:', this.possibleURLs);
    throw new Error('Unable to connect to MasterX AI Mentor System. Please check your internet connection.');
  }

  async getAPIURL() {
    const baseURL = await this.findWorkingURL();
    return `${baseURL}/api`;
  }

  // Method to force re-test connections (useful for error recovery)
  resetConnection() {
    console.log('🔄 Resetting connection manager...');
    this.workingURL = null;
    this.testInProgress = false;
    // Regenerate URLs to pick up any environment changes
    this.possibleURLs = this.generatePossibleURLs();
  }
}

// Create singleton instance
export const connectionManager = new ConnectionManager();

export default connectionManager;