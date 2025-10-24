// **Purpose:** Reusable API types and patterns

// **What This File Contributes:**
// 1. Generic API response wrappers
// 2. Error types
// 3. Pagination types
// 4. Loading states
// 5. API status codes

// **Implementation:**

// /**
//  * Generic API Types
//  * 
//  * Reusable types for all API interactions
//  * Following REST best practices
//  */

// ============================================================================
// API RESPONSE WRAPPERS
// ============================================================================

export interface APIResponse<T> {
  data: T;
  status: number;
  message?: string;
  timestamp: string;
}

export interface APIError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  status: number;
  timestamp: string;
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

export interface APIValidationError extends APIError {
  validation_errors: ValidationError[];
}

// ============================================================================
// PAGINATION TYPES
// ============================================================================

export interface PaginationParams {
  page: number;
  limit: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    current_page: number;
    total_pages: number;
    total_items: number;
    items_per_page: number;
    has_next: boolean;
    has_previous: boolean;
  };
}

// ============================================================================
// LOADING STATES
// ============================================================================

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  timestamp?: string;
}

export interface QueryState<T> extends AsyncState<T> {
  isFetching: boolean;
  isStale: boolean;
}

// ============================================================================
// HTTP STATUS CODES
// ============================================================================

export enum HTTPStatus {
  OK = 200,
  CREATED = 201,
  NO_CONTENT = 204,
  BAD_REQUEST = 400,
  UNAUTHORIZED = 401,
  FORBIDDEN = 403,
  NOT_FOUND = 404,
  CONFLICT = 409,
  UNPROCESSABLE_ENTITY = 422,
  TOO_MANY_REQUESTS = 429,
  INTERNAL_SERVER_ERROR = 500,
  SERVICE_UNAVAILABLE = 503,
}

// ============================================================================
// API CONFIGURATION
// ============================================================================

export interface APIConfig {
  baseURL: string;
  timeout: number;
  headers: Record<string, string>;
  retries: number;
  retryDelay: number;
}

export interface RequestConfig {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  url: string;
  data?: unknown;
  params?: Record<string, unknown>;
  headers?: Record<string, string>;
  timeout?: number;
  retry?: boolean;
}

// ============================================================================
// TYPE GUARDS
// ============================================================================

export const isAPIError = (obj: unknown): obj is APIError => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'error' in obj &&
    'status' in obj
  );
};

export const isAPIValidationError = (obj: unknown): obj is APIValidationError => {
  return (
    isAPIError(obj) &&
    'validation_errors' in obj &&
    Array.isArray((obj as APIValidationError).validation_errors)
  );
};

// ============================================================================
// HELPER TYPES
// ============================================================================

export type APIPromise<T> = Promise<APIResponse<T>>;
export type QueryKey = readonly unknown[];


// **Key Features:**
// 1. **Generic wrappers:** Reusable response types
// 2. **Error handling:** Comprehensive error types
// 3. **Pagination:** Standard pagination pattern
// 4. **Loading states:** Async operation tracking
// 5. **Type safety:** Type guards for runtime checks

// **Connected Files:**
// - → All `*.api.ts` files (use these types)
// - → React Query hooks (QueryState)
// - → Error boundaries (APIError)