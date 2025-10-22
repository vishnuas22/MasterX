import axios, { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/store/authStore';
import { useUIStore } from '@/store/uiStore';

// Create axios instance
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor (add JWT token)
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Get token from auth store
    const token = useAuthStore.getState().token;
    
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Log request in dev
    if (import.meta.env.DEV) {
      console.log(`→ ${config.method?.toUpperCase()} ${config.url}`);
    }
    
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// Response interceptor (handle errors)
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log response in dev
    if (import.meta.env.DEV) {
      console.log(`← ${response.status} ${response.config.url}`);
    }
    
    return response;
  },
  async (error: AxiosError) => {
    const { response, config } = error;
    
    // Log error
    if (import.meta.env.DEV) {
      console.error(`✗ ${response?.status} ${config?.url}`, error);
    }
    
    // Handle specific error codes
    if (response) {
      switch (response.status) {
        case 401:
          // Unauthorized - token expired
          useAuthStore.getState().logout();
          useUIStore.getState().showToast({
            type: 'error',
            message: 'Session expired. Please log in again.',
          });
          break;
          
        case 429:
          // Rate limited
          useUIStore.getState().showToast({
            type: 'warning',
            message: 'Too many requests. Please slow down.',
          });
          break;
          
        case 500:
        case 502:
        case 503:
          // Server errors
          useUIStore.getState().showToast({
            type: 'error',
            message: 'Server error. Please try again later.',
          });
          break;
      }
    } else if (error.code === 'ECONNABORTED') {
      // Timeout
      useUIStore.getState().showToast({
        type: 'error',
        message: 'Request timeout. Check your connection.',
      });
    } else if (!navigator.onLine) {
      // Offline
      useUIStore.getState().showToast({
        type: 'error',
        message: 'No internet connection.',
      });
    }
    
    return Promise.reject(error);
  }
);

// Retry logic for failed requests
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;
    
    // Retry only on network errors or 5xx
    if (
      !config ||
      !config.retry ||
      config.__retryCount >= (config.retry || 0)
    ) {
      return Promise.reject(error);
    }
    
    config.__retryCount = config.__retryCount || 0;
    config.__retryCount += 1;
    
    // Exponential backoff
    const delay = Math.pow(2, config.__retryCount) * 1000;
    await new Promise((resolve) => setTimeout(resolve, delay));
    
    return apiClient(config);
  }
);

export default apiClient;
