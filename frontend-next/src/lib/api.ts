import axios from 'axios'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL
const API_BASE = `${BACKEND_URL}/api`

export const apiService = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
apiService.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiService.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

export const api = {
  // Basic endpoints
  get: (url: string) => apiService.get(url),
  post: (url: string, data?: any) => apiService.post(url, data),
  put: (url: string, data?: any) => apiService.put(url, data),
  delete: (url: string) => apiService.delete(url),

  // Health check
  healthCheck: () => apiService.get('/'),

  // Chat endpoints (to be implemented)
  chat: {
    send: (message: string, sessionId?: string) => 
      apiService.post('/chat/send', { message, session_id: sessionId }),
    createSession: () => 
      apiService.post('/chat/session'),
    getHistory: (sessionId: string) => 
      apiService.get(`/chat/history/${sessionId}`),
  },

  // User endpoints (to be implemented)
  user: {
    create: (userData: any) => 
      apiService.post('/users', userData),
    get: (userId: string) => 
      apiService.get(`/users/${userId}`),
  },
}