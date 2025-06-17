import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Configure axios defaults
axios.defaults.timeout = 30000; // 30 seconds

class ApiService {
  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for logging
    this.axiosInstance.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Add response interceptor for error handling
    this.axiosInstance.interceptors.response.use(
      (response) => {
        return response;
      },
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        if (error.response?.status === 404) {
          throw new Error('Resource not found');
        } else if (error.response?.status === 500) {
          throw new Error('Server error. Please try again later.');
        } else if (error.code === 'ECONNABORTED') {
          throw new Error('Request timeout. Please check your connection.');
        }
        throw new Error(error.response?.data?.detail || error.message || 'An error occurred');
      }
    );
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

  // Utility methods
  async testConnection() {
    try {
      const response = await this.axiosInstance.get('/');
      return response.data;
    } catch (error) {
      throw new Error('Unable to connect to MasterX AI Mentor System');
    }
  }
}

// Create and export API instance
export const api = new ApiService();

// Also export the axios instance for custom requests
export const axiosInstance = axios.create({
  baseURL: API,
  headers: {
    'Content-Type': 'application/json',
  },
});
