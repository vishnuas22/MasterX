import axios from 'axios';
import { getEnvironmentConfig } from '../config/environment';
import { connectionManager } from '../utils/connectionManager';

// Get initial environment configuration
const envConfig = getEnvironmentConfig();
let BACKEND_URL = envConfig.backendURL;
let API = envConfig.apiURL;

// Only log essential information
console.log('🚀 MasterX API Service Initialized');
console.log(`🌍 Environment: ${envConfig.environment.toUpperCase()}`);
console.log(`🔗 Primary Backend URL: ${BACKEND_URL}`);
console.log(`🚀 API Endpoint: ${API}`);

// Configure axios defaults with optimized settings
axios.defaults.timeout = 15000; // Reduced from 30 seconds

class ApiService {
  constructor() {
    this.isConnectionTested = false;
    this.connectionRetries = 0;
    this.maxRetries = 2; // Reduced from 3
    this.axiosInstance = this.createAxiosInstance();
    this.debugMode = false; // Set to true for debugging
  }

  createAxiosInstance() {
    const instance = axios.create({
      baseURL: API,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor - Test connection and update URLs if needed
    instance.interceptors.request.use(
      async (config) => {
        // Test connection on first request or after errors
        if (!this.isConnectionTested) {
          try {
            if (this.debugMode) console.log('🔍 Testing backend connection...');
            const workingURL = await connectionManager.findWorkingURL();
            
            // Update URLs if different from initial configuration
            if (workingURL !== BACKEND_URL) {
              BACKEND_URL = workingURL;
              API = `${workingURL}/api`;
              config.baseURL = API;
              if (this.debugMode) console.log(`🔄 Updated API endpoint to: ${API}`);
            }
            
            this.isConnectionTested = true;
            this.connectionRetries = 0;
            console.log('✅ Backend connection verified');
            
          } catch (error) {
            console.error('❌ Backend connection failed:', error.message);
            // Continue with original URL as fallback
          }
        }
        
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor - Handle errors and connection recovery
    instance.interceptors.response.use(
      (response) => {
        // Reset connection retries on successful response
        this.connectionRetries = 0;
        return response;
      },
      async (error) => {
        if (this.debugMode) console.error('❌ API Request Failed:', error.response?.data || error.message);
        
        // Handle connection errors with retry logic
        if (this.shouldRetryConnection(error)) {
          this.connectionRetries++;
          
          if (this.connectionRetries <= this.maxRetries) {
            if (this.debugMode) console.log(`🔄 Connection error detected (retry ${this.connectionRetries}/${this.maxRetries})`);
            
            // Reset connection and try to find working URL
            connectionManager.resetConnection();
            this.isConnectionTested = false;
            
            // Reduced wait time before retrying
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Retry the original request if it's a simple GET
            if (error.config && error.config.method?.toLowerCase() === 'get') {
              if (this.debugMode) console.log('🔄 Retrying request...');
              return this.axiosInstance.request(error.config);
            }
          } else {
            console.error('❌ Max connection retries exceeded');
          }
        }
        
        // Transform error messages for better UX
        throw this.transformError(error);
      }
    );

    return instance;
  }

  shouldRetryConnection(error) {
    return (
      error.code === 'ECONNABORTED' ||
      error.code === 'NETWORK_ERROR' ||
      error.code === 'ERR_NETWORK' ||
      !error.response ||
      (error.response && error.response.status >= 500)
    );
  }

  transformError(error) {
    if (error.response?.status === 404) {
      return new Error('Resource not found');
    } else if (error.response?.status === 500) {
      return new Error('Server error. Please try again later.');
    } else if (error.code === 'ECONNABORTED') {
      return new Error('Request timeout. Please check your connection.');
    } else if (error.code === 'NETWORK_ERROR' || error.code === 'ERR_NETWORK' || !error.response) {
      return new Error('Unable to connect to MasterX AI Mentor System. Please check your internet connection.');
    }
    
    return new Error(error.response?.data?.detail || error.message || 'An error occurred');
  }

  // Connection utility methods
  async testConnection() {
    try {
      const response = await this.axiosInstance.get('/health');
      return response.data;
    } catch (error) {
      throw new Error('Unable to connect to MasterX AI Mentor System');
    }
  }

  async forceConnectionRefresh() {
    console.log('🔄 Forcing connection refresh...');
    connectionManager.resetConnection();
    this.isConnectionTested = false;
    this.connectionRetries = 0;
    return this.testConnection();
  }

  getConnectionStatus() {
    return {
      ...connectionManager.getConnectionStatus(),
      apiRetries: this.connectionRetries,
      maxRetries: this.maxRetries,
      connectionTested: this.isConnectionTested
    };
  }

  // Health check
  async healthCheck() {
    const response = await this.axiosInstance.get('/health');
    return response.data;
  }

  // User endpoints
  async createUser(userData) {
    const response = await this.axiosInstance.post('/users', userData);
    return response.data;
  }

  async getUser(userId) {
    const response = await this.axiosInstance.get(`/users/${userId}`);
    return response.data;
  }

  async getUserByEmail(email) {
    const response = await this.axiosInstance.get(`/users/email/${email}`);
    return response.data;
  }

  // Session endpoints
  async createSession(sessionData) {
    const response = await this.axiosInstance.post('/sessions', sessionData);
    return response.data;
  }

  async getSession(sessionId) {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}`);
    return response.data;
  }

  async getUserSessions(userId, activeOnly = true) {
    const response = await this.axiosInstance.get(`/users/${userId}/sessions`, {
      params: { active_only: activeOnly }
    });
    return response.data;
  }

  async endSession(sessionId) {
    const response = await this.axiosInstance.put(`/sessions/${sessionId}/end`);
    return response.data;
  }

  // Chat endpoints
  async sendChatMessage(requestData) {
    const response = await this.axiosInstance.post('/chat', requestData);
    return response.data;
  }

  async getSessionMessages(sessionId, limit = 50, offset = 0) {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/messages`, {
      params: { limit, offset }
    });
    return response.data;
  }

  // Chat Management endpoints
  async renameSession(sessionId, newTitle) {
    const response = await this.axiosInstance.put(`/sessions/${sessionId}/rename`, {
      title: newTitle
    });
    return response.data;
  }

  async shareSession(sessionId) {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/share`, {});
    return response.data;
  }

  async deleteSession(sessionId) {
    const response = await this.axiosInstance.delete(`/sessions/${sessionId}`);
    return response.data;
  }

  async searchSessionMessages(sessionId, query, limit = 20) {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/search`, {
      params: { query, limit }
    });
    return response.data;
  }

  async searchUserSessions(userId, query, limit = 50) {
    const response = await this.axiosInstance.get(`/users/${userId}/sessions/search`, {
      params: { query, limit }
    });
    return response.data;
  }

  // Exercise endpoints
  async generateExercise(topic, difficulty = 'medium', exerciseType = 'multiple_choice') {
    const response = await this.axiosInstance.post('/exercises/generate', null, {
      params: { topic, difficulty, exercise_type: exerciseType }
    });
    return response.data;
  }

  async analyzeExerciseResponse(question, userAnswer, correctAnswer = null) {
    const response = await this.axiosInstance.post('/exercises/analyze', null, {
      params: { question, user_answer: userAnswer, correct_answer: correctAnswer }
    });
    return response.data;
  }

  // Learning path endpoints
  async generateLearningPath(subject, userLevel = 'beginner', goals = []) {
    const response = await this.axiosInstance.post('/learning-paths/generate', null, {
      params: { subject, user_level: userLevel, goals: goals.join(',') }
    });
    return response.data;
  }

  async getUserProgress(userId, subject = null) {
    const params = subject ? { subject } : {};
    const response = await this.axiosInstance.get(`/users/${userId}/progress`, { params });
    return response.data;
  }

  // ================================
  // PREMIUM CONTEXT AWARENESS ENDPOINTS
  // ================================

  async analyzeUserContext(userId, sessionId, message, conversationContext) {
    const response = await this.axiosInstance.post('/context/analyze', {
      user_id: userId,
      session_id: sessionId,
      message: message,
      conversation_context: conversationContext
    });
    return response.data;
  }

  async getUserMemoryInsights(userId) {
    const response = await this.axiosInstance.get(`/context/${userId}/memory`);
    return response.data;
  }

  async sendContextAwareMessage(requestData) {
    const response = await this.axiosInstance.post('/chat/premium-context', requestData);
    return response.data;
  }

  // ================================
  // LIVE LEARNING SESSIONS ENDPOINTS
  // ================================

  async createLiveSession(userId, sessionType, title, durationMinutes = 60, features = {}) {
    const response = await this.axiosInstance.post('/live-sessions/create', {
      user_id: userId,
      session_type: sessionType,
      title: title,
      duration_minutes: durationMinutes,
      features: features
    });
    return response.data;
  }

  async handleVoiceInteraction(sessionId, userId, audioData) {
    const response = await this.axiosInstance.post(`/live-sessions/${sessionId}/voice`, {
      user_id: userId,
      audio_data: audioData
    });
    return response.data;
  }

  async handleScreenSharing(sessionId, userId, screenData) {
    const response = await this.axiosInstance.post(`/live-sessions/${sessionId}/screen-share`, {
      user_id: userId,
      screen_data: screenData
    });
    return response.data;
  }

  async handleLiveCoding(sessionId, userId, codeUpdate) {
    const response = await this.axiosInstance.post(`/live-sessions/${sessionId}/code`, {
      user_id: userId,
      code_update: codeUpdate
    });
    return response.data;
  }

  async handleInteractiveWhiteboard(sessionId, userId, whiteboardUpdate) {
    const response = await this.axiosInstance.post(`/live-sessions/${sessionId}/whiteboard`, {
      user_id: userId,
      whiteboard_update: whiteboardUpdate
    });
    return response.data;
  }

  async getLiveSessionStatus(sessionId) {
    const response = await this.axiosInstance.get(`/live-sessions/${sessionId}/status`);
    return response.data;
  }

  async endLiveSession(sessionId) {
    const response = await this.axiosInstance.post(`/live-sessions/${sessionId}/end`);
    return response.data;
  }

  // Learning Psychology Features
  async getLearningPsychologyFeatures() {
    const response = await this.axiosInstance.get('/learning-psychology/features');
    return response.data;
  }

  async getUserLearningProgress(userId) {
    const response = await this.axiosInstance.get(`/learning-psychology/progress/${userId}`);
    return response.data;
  }

  // Metacognitive Training
  async startMetacognitiveSession(userId, sessionData) {
    const response = await this.axiosInstance.post(`/learning-psychology/metacognitive/start?user_id=${userId}`, sessionData);
    return response.data;
  }

  async respondToMetacognitiveSession(sessionId, responseData) {
    const response = await this.axiosInstance.post(`/learning-psychology/metacognitive/${sessionId}/respond`, responseData);
    return response.data;
  }

  // Memory Palace
  async createMemoryPalace(userId, palaceData) {
    const response = await this.axiosInstance.post(`/learning-psychology/memory-palace/create?user_id=${userId}`, palaceData);
    return response.data;
  }

  async practiceMemoryPalace(palaceId, practiceType = 'recall') {
    const response = await this.axiosInstance.post(`/learning-psychology/memory-palace/${palaceId}/practice?practice_type=${practiceType}`);
    return response.data;
  }

  async getUserMemoryPalaces(userId) {
    const response = await this.axiosInstance.get(`/learning-psychology/memory-palace/user/${userId}`);
    return response.data;
  }

  // Elaborative Questions
  async generateElaborativeQuestions(questionData) {
    const response = await this.axiosInstance.post('/learning-psychology/elaborative-questions/generate', questionData);
    return response.data;
  }

  async evaluateElaborativeResponse(questionId, responseData) {
    const response = await this.axiosInstance.post(`/learning-psychology/elaborative-questions/${questionId}/evaluate`, responseData);
    return response.data;
  }

  // Transfer Learning
  async createTransferScenario(scenarioData) {
    const response = await this.axiosInstance.post('/learning-psychology/transfer-learning/create-scenario', scenarioData);
    return response.data;
  }

  // ===============================
  // 🎨 AR/VR AND GESTURE CONTROL API METHODS
  // ===============================

  // AR/VR Settings
  async getUserARVRSettings(userId) {
    const response = await this.axiosInstance.get(`/users/${userId}/arvr-settings`);
    return response.data;
  }

  async updateUserARVRSettings(userId, settings) {
    const response = await this.axiosInstance.post(`/users/${userId}/arvr-settings`, { settings });
    return response.data;
  }

  // Gesture Control Settings
  async getUserGestureSettings(userId) {
    const response = await this.axiosInstance.get(`/users/${userId}/gesture-settings`);
    return response.data;
  }

  async updateUserGestureSettings(userId, settings) {
    const response = await this.axiosInstance.post(`/users/${userId}/gesture-settings`, { settings });
    return response.data;
  }

  // Session AR/VR State
  async updateSessionARVRState(sessionId, state) {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/arvr-state`, { state });
    return response.data;
  }

  // User Profile and Learning DNA
  async getUserLearningDNA(userId) {
    const response = await this.axiosInstance.get(`/users/${userId}/learning-dna`);
    return response.data;
  }

  async updateUserLearningMode(userId, preferredMode, preferences = {}) {
    const response = await this.axiosInstance.post(`/users/${userId}/learning-mode`, { 
      preferred_mode: preferredMode, 
      preferences 
    });
    return response.data;
  }
}

// Create and export API instance
export const api = new ApiService();

// Export additional utilities
export const getConnectionHealth = () => api.getConnectionStatus();
export const refreshConnection = () => api.forceConnectionRefresh();

// Also export the axios instance for custom requests
export const axiosInstance = axios.create({
  baseURL: API,
  headers: {
    'Content-Type': 'application/json',
  },
});
