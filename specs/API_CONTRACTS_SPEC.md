# API Contracts Specification

## Purpose
Define all API endpoints, request/response formats, and interface contracts for MasterX platform with emphasis on emotion-aware learning and real-time personalization.

## Core Learning API

### 1.1 POST /api/quantum/chat - Emotion-Aware AI Tutoring
**Purpose**: Primary endpoint for interactive AI tutoring with emotional intelligence and personalization

**Request Format**:
```typescript
interface ChatRequest {
  user_id: string;
  message: string;
  session_id?: string; // optional, creates new session if not provided
  context?: {
    subject?: string; // 'mathematics', 'science', 'programming', etc.
    topic?: string; // specific topic within subject
    difficulty_level?: number; // 0.0-1.0, auto-detected if not provided
    learning_objective?: string; // what user wants to achieve
    previous_messages?: Array<{
      role: 'user' | 'assistant';
      content: string;
      timestamp: string;
      emotional_state?: EmotionalState;
    }>;
  };
  preferences?: {
    explanation_style?: 'concise' | 'detailed' | 'step-by-step';
    response_length?: 'short' | 'medium' | 'long';
    include_examples?: boolean;
    emotional_support_level?: 'minimal' | 'standard' | 'high';
  };
}
```

**Response Format**:
```typescript
interface ChatResponse {
  response: {
    text: string; // main AI response
    explanation_components?: {
      concept_explanation?: string;
      step_by_step_solution?: string[];
      examples?: string[];
      visual_aids_description?: string;
    };
  };
  
  emotional_analysis: {
    detected_emotion: EmotionalState;
    confidence_score: number; // 0.0-1.0
    emotional_intensity: number; // 0.0-1.0
    adaptation_applied: string[]; // list of adaptations made
    emotional_trend: 'improving' | 'stable' | 'declining';
  };
  
  learning_insights: {
    difficulty_assessment: number; // 0.0-1.0 how difficult this was for user
    knowledge_gaps: string[]; // identified areas needing work
    mastery_indicators: string[]; // concepts user has mastered
    recommended_next_steps: string[];
    estimated_understanding_level: number; // 0.0-1.0
  };
  
  personalization_data: {
    learning_style_detected?: 'visual' | 'auditory' | 'kinesthetic' | 'reading_writing';
    preferred_complexity: number; // user's optimal complexity level
    engagement_indicators: string[]; // signs of user engagement
    adaptation_effectiveness: number; // how well personalization worked
  };
  
  metadata: {
    session_id: string;
    provider_used: 'groq' | 'gemini' | 'emergent';
    response_time_ms: number;
    tokens_used: number;
    cost_usd: number;
    cache_hit: boolean;
    quality_score: number; // 0.0-1.0 internal quality assessment
  };
}
```

### 1.2 POST /api/quantum/assess - Adaptive Assessment
**Purpose**: Dynamic assessment that adapts difficulty based on performance and emotional state

**Request Format**:
```typescript
interface AssessmentRequest {
  user_id: string;
  assessment_type: 'diagnostic' | 'practice' | 'quiz' | 'exam_prep';
  subject: string;
  topics?: string[]; // specific topics to assess
  parameters: {
    target_duration_minutes?: number; // preferred assessment length
    difficulty_preference?: 'adaptive' | 'fixed'; // adaptive adjusts based on performance
    question_count?: number; // fixed number or 'adaptive'
    feedback_timing?: 'immediate' | 'end_of_section' | 'end_of_assessment';
  };
  context?: {
    upcoming_exam?: string; // 'SAT', 'AP_Calculus', etc.
    learning_goals?: string[];
    time_constraints?: number; // minutes available
  };
}
```

**Response Format**:
```typescript
interface AssessmentResponse {
  assessment_id: string;
  questions: Array<{
    question_id: string;
    question_text: string;
    question_type: 'multiple_choice' | 'short_answer' | 'essay' | 'problem_solving';
    options?: string[]; // for multiple choice
    difficulty_level: number; // 0.0-1.0
    estimated_time_minutes: number;
    learning_objective: string;
    hints_available: boolean;
  }>;
  
  adaptive_settings: {
    starting_difficulty: number;
    adjustment_algorithm: 'item_response_theory' | 'performance_based';
    emotional_adaptation_enabled: boolean;
    mastery_threshold: number; // when to consider topic mastered
  };
  
  metadata: {
    total_estimated_duration: number;
    learning_objectives: string[];
    prerequisite_topics: string[];
    difficulty_range: [number, number]; // min and max difficulty levels
  };
}
```

### 1.3 POST /api/quantum/feedback - Learning Feedback Processing
**Purpose**: Process user feedback on AI responses and learning effectiveness

**Request Format**:
```typescript
interface FeedbackRequest {
  user_id: string;
  session_id: string;
  feedback_type: 'response_quality' | 'emotional_adaptation' | 'difficulty_level' | 'explanation_style';
  rating: number; // 1-5 scale
  specific_feedback?: {
    what_worked_well?: string[];
    what_needs_improvement?: string[];
    suggested_alternatives?: string[];
    emotional_impact?: 'positive' | 'neutral' | 'negative';
  };
  context: {
    message_id?: string; // specific response being rated
    topic?: string;
    difficulty_level?: number;
    emotional_state_during_interaction?: EmotionalState;
  };
}
```

## User Management API

### 2.1 POST /api/user/register - User Registration
**Request Format**:
```typescript
interface UserRegistrationRequest {
  email: string;
  password: string;
  profile: {
    name: string;
    age?: number;
    education_level: 'middle_school' | 'high_school' | 'undergraduate' | 'graduate' | 'professional';
    learning_goals: string[];
    subjects_of_interest: string[];
    preferred_study_schedule?: {
      days_per_week: number;
      hours_per_day: number;
      preferred_time_of_day: 'morning' | 'afternoon' | 'evening' | 'late_night';
    };
  };
  preferences: {
    notification_settings: {
      email_notifications: boolean;
      study_reminders: boolean;
      progress_updates: boolean;
    };
    privacy_settings: {
      anonymous_analytics: boolean;
      emotional_data_usage: boolean;
      learning_data_sharing: boolean;
    };
  };
}
```

### 2.2 GET /api/user/profile - User Profile Retrieval
**Response Format**:
```typescript
interface UserProfileResponse {
  user_id: string;
  basic_info: {
    name: string;
    email: string;
    age?: number;
    education_level: string;
    registration_date: string;
    last_active: string;
  };
  
  learning_dna: {
    learning_style: {
      primary: string;
      confidence: number;
      last_assessed: string;
    };
    cognitive_profile: {
      processing_speed: string;
      working_memory: string;
      attention_span_minutes: number;
      preferred_complexity: number;
    };
    motivation_profile: {
      goal_orientation: string;
      achievement_sensitivity: number;
      feedback_preference: string;
    };
    emotional_learning_patterns: {
      optimal_emotional_states: EmotionalState[];
      stress_triggers: string[];
      confidence_builders: string[];
    };
  };
  
  progress_summary: {
    subjects_studied: Array<{
      subject: string;
      mastery_level: number; // 0.0-1.0
      time_spent_hours: number;
      last_studied: string;
    }>;
    overall_progress: {
      total_study_time_hours: number;
      concepts_mastered: number;
      current_streak_days: number;
      achievements_unlocked: number;
    };
  };
  
  subscription: {
    plan: 'free' | 'premium' | 'family';
    status: 'active' | 'inactive' | 'cancelled';
    expires_at?: string;
    usage_stats: {
      queries_this_month: number;
      query_limit: number;
      features_accessed: string[];
    };
  };
}
```

### 2.3 PUT /api/user/preferences - Update User Preferences
**Request Format**:
```typescript
interface UserPreferencesUpdate {
  learning_preferences?: {
    explanation_style: 'concise' | 'detailed' | 'comprehensive';
    difficulty_preference: 'easy' | 'moderate' | 'challenging' | 'adaptive';
    feedback_frequency: 'high' | 'moderate' | 'minimal';
    emotional_support_level: 'minimal' | 'standard' | 'high';
  };
  
  study_schedule?: {
    preferred_session_length_minutes: number;
    break_frequency_minutes: number;
    daily_goal_minutes: number;
    reminder_schedule: string[]; // ISO time strings
  };
  
  content_preferences?: {
    include_visual_aids: boolean;
    include_real_world_examples: boolean;
    include_practice_problems: boolean;
    gamification_level: 'none' | 'minimal' | 'moderate' | 'high';
  };
}
```

## Analytics API

### 3.1 GET /api/analytics/dashboard - Learning Analytics Dashboard
**Response Format**:
```typescript
interface AnalyticsDashboardResponse {
  overview: {
    total_study_time_hours: number;
    concepts_mastered_count: number;
    current_streak_days: number;
    next_milestone: {
      description: string;
      progress_percentage: number;
      estimated_completion_date: string;
    };
  };
  
  progress_tracking: {
    weekly_study_time: number[]; // last 7 days
    subject_progress: Array<{
      subject: string;
      mastery_percentage: number;
      time_invested_hours: number;
      recent_performance_trend: 'improving' | 'stable' | 'declining';
    }>;
    difficulty_progression: Array<{
      date: string;
      average_difficulty: number;
      success_rate: number;
    }>;
  };
  
  emotional_insights: {
    emotional_health_score: number; // 0.0-1.0, overall emotional wellness
    dominant_emotions: Array<{
      emotion: EmotionalState;
      percentage: number;
      trend: 'increasing' | 'stable' | 'decreasing';
    }>;
    stress_pattern_analysis: {
      high_stress_periods: string[]; // time patterns when stress is high
      stress_triggers: string[];
      effective_stress_reduction_strategies: string[];
    };
    optimal_learning_conditions: {
      best_emotional_states_for_learning: EmotionalState[];
      recommended_study_times: string[];
      ideal_session_duration_minutes: number;
    };
  };
  
  personalized_recommendations: {
    next_topics_to_study: Array<{
      topic: string;
      subject: string;
      priority: 'high' | 'medium' | 'low';
      estimated_time_to_master_hours: number;
      prerequisite_completion: number; // 0.0-1.0
    }>;
    
    skill_gap_analysis: Array<{
      skill_area: string;
      current_level: number; // 0.0-1.0
      target_level: number;
      recommended_actions: string[];
    }>;
    
    study_optimization_tips: Array<{
      category: 'schedule' | 'technique' | 'emotional' | 'content';
      recommendation: string;
      expected_impact: 'high' | 'medium' | 'low';
    }>;
  };
}
```

### 3.2 GET /api/analytics/performance - Performance Analytics
**Query Parameters**:
- `time_period`: 'week' | 'month' | 'quarter' | 'year'
- `subject`: optional subject filter
- `metric_type`: 'accuracy' | 'speed' | 'engagement' | 'emotional'

**Response Format**:
```typescript
interface PerformanceAnalyticsResponse {
  time_period: string;
  metrics: {
    accuracy_metrics: {
      overall_accuracy_percentage: number;
      accuracy_by_difficulty: Array<{
        difficulty_level: number;
        accuracy_percentage: number;
        sample_size: number;
      }>;
      improvement_rate: number; // percentage improvement over time period
    };
    
    speed_metrics: {
      average_response_time_seconds: number;
      response_time_trend: 'improving' | 'stable' | 'declining';
      optimal_response_time_range: [number, number];
    };
    
    engagement_metrics: {
      average_session_duration_minutes: number;
      questions_per_session: number;
      follow_up_question_rate: number; // indicates curiosity and engagement
      session_completion_rate: number;
    };
    
    emotional_metrics: {
      positive_emotion_percentage: number;
      stress_level_average: number; // 0.0-1.0
      emotional_resilience_score: number; // recovery from negative emotions
      intervention_success_rate: number; // effectiveness of emotional interventions
    };
  };
  
  comparative_analysis: {
    compared_to_previous_period: {
      accuracy_change: number;
      engagement_change: number;
      emotional_wellness_change: number;
    };
    peer_comparison?: {
      percentile_rank: number; // user's rank among peers (if opted in)
      areas_of_strength: string[];
      areas_for_improvement: string[];
    };
  };
}
```

## Health and System API

### 4.1 GET /api/health - System Health Check
**Response Format**:
```typescript
interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  
  components: {
    api_server: {
      status: 'up' | 'down';
      response_time_ms: number;
    };
    database: {
      status: 'connected' | 'disconnected';
      connection_pool_utilization: number;
      query_performance_ms: number;
    };
    ai_providers: {
      groq: {
        status: 'available' | 'unavailable' | 'degraded';
        average_response_time_ms: number;
        success_rate_percentage: number;
      };
      gemini: {
        status: 'available' | 'unavailable' | 'degraded';
        average_response_time_ms: number;
        success_rate_percentage: number;
      };
      emergent: {
        status: 'available' | 'unavailable' | 'degraded';
        average_response_time_ms: number;
        success_rate_percentage: number;
      };
    };
    cache_system: {
      status: 'operational' | 'degraded' | 'failed';
      hit_rate_percentage: number;
      memory_utilization_percentage: number;
    };
  };
  
  performance_metrics: {
    requests_per_second: number;
    average_response_time_ms: number;
    error_rate_percentage: number;
    concurrent_users: number;
  };
}
```

## Error Handling Standards

### 5.1 Standard Error Response Format
```typescript
interface ErrorResponse {
  error: {
    code: string; // machine-readable error code
    message: string; // human-readable error message
    details?: any; // additional error context
    timestamp: string;
    request_id: string; // for tracking and debugging
    suggestion?: string; // helpful suggestion for user
  };
  
  // AI-specific error handling
  fallback_available?: boolean; // whether fallback response is available
  retry_recommended?: boolean;
  retry_after_seconds?: number;
  
  // User experience enhancements
  user_friendly_message: string; // non-technical explanation
  support_contact?: string; // how to get help
  documentation_link?: string; // relevant help documentation
}
```

### 5.2 Common Error Codes
```typescript
enum APIErrorCodes {
  // Authentication & Authorization
  'INVALID_CREDENTIALS' = 'INVALID_CREDENTIALS',
  'TOKEN_EXPIRED' = 'TOKEN_EXPIRED',
  'INSUFFICIENT_PERMISSIONS' = 'INSUFFICIENT_PERMISSIONS',
  
  // AI Provider Errors
  'AI_PROVIDER_TIMEOUT' = 'AI_PROVIDER_TIMEOUT',
  'AI_PROVIDER_UNAVAILABLE' = 'AI_PROVIDER_UNAVAILABLE',
  'AI_RESPONSE_QUALITY_LOW' = 'AI_RESPONSE_QUALITY_LOW',
  'AI_CONTENT_FILTERED' = 'AI_CONTENT_FILTERED',
  
  // Personalization Errors
  'INSUFFICIENT_USER_DATA' = 'INSUFFICIENT_USER_DATA',
  'EMOTION_DETECTION_FAILED' = 'EMOTION_DETECTION_FAILED',
  'PERSONALIZATION_ENGINE_ERROR' = 'PERSONALIZATION_ENGINE_ERROR',
  
  // System Errors
  'RATE_LIMIT_EXCEEDED' = 'RATE_LIMIT_EXCEEDED',
  'DATABASE_UNAVAILABLE' = 'DATABASE_UNAVAILABLE',
  'CACHE_MISS_CRITICAL' = 'CACHE_MISS_CRITICAL',
  'SYSTEM_OVERLOADED' = 'SYSTEM_OVERLOADED',
  
  // User Input Errors
  'INVALID_REQUEST_FORMAT' = 'INVALID_REQUEST_FORMAT',
  'MISSING_REQUIRED_FIELD' = 'MISSING_REQUIRED_FIELD',
  'INVALID_PARAMETER_VALUE' = 'INVALID_PARAMETER_VALUE',
  'CONTENT_TOO_LONG' = 'CONTENT_TOO_LONG'
}
```

## API Versioning and Compatibility

### 6.1 Versioning Strategy
- **Current Version**: v1
- **URL Format**: `/api/v1/endpoint`
- **Backward Compatibility**: Maintain v1 for minimum 1 year after v2 release
- **Breaking Changes**: Only in major version updates
- **Deprecation Notice**: 6 months advance notice for deprecated endpoints

### 6.2 Rate Limiting
```typescript
interface RateLimitingPolicy {
  free_tier: {
    requests_per_minute: 30;
    requests_per_hour: 500;
    requests_per_day: 2000;
    ai_queries_per_day: 50;
  };
  
  premium_tier: {
    requests_per_minute: 200;
    requests_per_hour: 5000;
    requests_per_day: 50000;
    ai_queries_per_day: 1000;
  };
  
  rate_limit_headers: [
    'X-RateLimit-Limit',
    'X-RateLimit-Remaining',
    'X-RateLimit-Reset',
    'X-RateLimit-Retry-After'
  ];
}
```

## Success Metrics for API Design

### 6.3 API Performance Targets
- **Response Time**: 95% of requests under target times (defined per endpoint)
- **Availability**: 99.9% uptime for all endpoints
- **Error Rate**: <1% for 4xx errors, <0.1% for 5xx errors
- **Rate Limit Compliance**: <5% of users hitting rate limits
- **API Documentation**: >4.5/5 developer satisfaction rating