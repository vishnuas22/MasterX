# Database Schema Specification

## Purpose
Define comprehensive database schema, indexing strategy, and query patterns for MasterX platform with focus on performance, scalability, and educational data integrity.

## Database Technology Stack
- **Primary Database**: MongoDB (document-based for flexible educational content)
- **Caching Layer**: Redis (in-memory caching for performance)
- **Search Engine**: MongoDB Atlas Search (full-text search for educational content)
- **Analytics Database**: MongoDB with aggregation pipelines (real-time analytics)

## Core Collections

### 1.1 users Collection
```typescript
interface UserDocument {
  _id: ObjectId;
  user_id: string; // UUID for external references
  email: string;
  password_hash: string;
  email_verified: boolean;
  created_at: Date;
  updated_at: Date;
  last_login: Date;
  
  profile: {
    name: string;
    age?: number;
    education_level: 'middle_school' | 'high_school' | 'undergraduate' | 'graduate' | 'professional';
    timezone: string; // for scheduling and analytics
    location?: {
      country: string;
      region?: string;
    };
    learning_goals: string[];
    subjects_of_interest: string[];
  };
  
  subscription: {
    plan: 'free' | 'premium' | 'family';
    status: 'active' | 'inactive' | 'cancelled' | 'trial';
    subscribed_at?: Date;
    expires_at?: Date;
    payment_method_id?: string;
    billing_cycle: 'monthly' | 'yearly';
    usage_tracking: {
      ai_queries_used: number;
      ai_queries_limit: number;
      reset_date: Date;
    };
  };
  
  preferences: {
    notification_settings: {
      email_notifications: boolean;
      study_reminders: boolean;
      progress_updates: boolean;
      achievement_notifications: boolean;
    };
    privacy_settings: {
      analytics_participation: boolean;
      emotional_data_usage: boolean;
      learning_data_sharing: boolean;
      public_profile: boolean;
    };
    learning_preferences: {
      explanation_style: 'concise' | 'detailed' | 'comprehensive';
      difficulty_preference: 'easy' | 'moderate' | 'challenging' | 'adaptive';
      feedback_frequency: 'high' | 'moderate' | 'minimal';
      emotional_support_level: 'minimal' | 'standard' | 'high';
    };
  };
  
  // Account status and security
  account_status: 'active' | 'suspended' | 'deleted';
  security: {
    failed_login_attempts: number;
    last_password_change: Date;
    two_factor_enabled: boolean;
  };
}
```

### 1.2 learning_dna Collection
```typescript
interface LearningDNADocument {
  _id: ObjectId;
  user_id: string; // references users.user_id
  created_at: Date;
  updated_at: Date;
  version: number; // for tracking DNA evolution
  
  cognitive_profile: {
    learning_style: {
      primary: 'visual' | 'auditory' | 'kinesthetic' | 'reading_writing';
      secondary?: 'visual' | 'auditory' | 'kinesthetic' | 'reading_writing';
      confidence_score: number; // 0.0-1.0
      assessment_method: 'survey' | 'behavioral_analysis' | 'hybrid';
      last_assessed: Date;
    };
    
    processing_characteristics: {
      processing_speed: 'slow' | 'average' | 'fast';
      working_memory_capacity: 'low' | 'average' | 'high';
      attention_span_minutes: number;
      information_processing_preference: 'sequential' | 'global' | 'mixed';
      multitasking_ability: 'low' | 'average' | 'high';
    };
    
    complexity_preferences: {
      preferred_difficulty_level: number; // 0.0-1.0
      challenge_tolerance: 'low' | 'moderate' | 'high';
      explanation_depth_preference: 'concise' | 'detailed' | 'comprehensive';
      example_density_preference: 'minimal' | 'moderate' | 'extensive';
      abstract_vs_concrete: number; // -1.0 (concrete) to 1.0 (abstract)
    };
  };
  
  motivation_profile: {
    goal_orientation: 'mastery' | 'performance' | 'social' | 'mixed';
    intrinsic_motivation_level: number; // 0.0-1.0
    extrinsic_motivation_responsiveness: number; // 0.0-1.0
    feedback_preference: 'immediate' | 'periodic' | 'milestone_based';
    achievement_sensitivity: number; // response to gamification 0.0-1.0
    autonomy_preference: 'guided' | 'semi_independent' | 'self_directed';
  };
  
  emotional_learning_patterns: {
    optimal_emotional_states: EmotionalState[];
    problematic_emotional_states: EmotionalState[];
    stress_triggers: string[];
    confidence_builders: string[];
    engagement_factors: string[];
    emotional_recovery_strategies: string[];
    emotional_resilience_score: number; // 0.0-1.0
  };
  
  // Evolution tracking
  assessment_history: Array<{
    date: Date;
    assessment_type: 'initial' | 'behavioral_update' | 'user_feedback' | 'performance_based';
    changes_made: string[];
    confidence_delta: number;
  }>;
  
  // Performance correlation data
  learning_effectiveness: {
    subjects_performance: Record<string, {
      mastery_rate: number;
      engagement_level: number;
      optimal_difficulty: number;
    }>;
    overall_learning_velocity: number; // concepts mastered per hour
    personalization_effectiveness: number; // 0.0-1.0
  };
}
```

### 1.3 learning_sessions Collection
```typescript
interface LearningSessionDocument {
  _id: ObjectId;
  session_id: string; // UUID for external references
  user_id: string; // references users.user_id
  started_at: Date;
  ended_at?: Date;
  duration_minutes?: number;
  
  session_context: {
    session_type: 'study' | 'practice' | 'assessment' | 'revision' | 'exploration';
    subject: string;
    topics: string[];
    learning_objectives: string[];
    initial_difficulty_level: number; // 0.0-1.0
    device_type: 'mobile' | 'tablet' | 'desktop';
    platform: 'web' | 'ios' | 'android';
  };
  
  interactions: Array<{
    interaction_id: string;
    timestamp: Date;
    interaction_type: 'question' | 'response' | 'feedback' | 'assessment';
    user_input?: string;
    ai_response?: {
      text: string;
      provider_used: 'groq' | 'gemini' | 'emergent';
      response_time_ms: number;
      tokens_used: number;
      cost_usd: number;
    };
    emotional_context?: {
      detected_emotion: EmotionalState;
      confidence: number;
      emotional_intensity: number;
      adaptation_applied: string[];
    };
    personalization_applied?: {
      difficulty_adjustment: number;
      explanation_style: string;
      content_modifications: string[];
    };
  }>;
  
  learning_progress: {
    concepts_introduced: string[];
    concepts_practiced: string[];
    concepts_mastered: string[];
    knowledge_gaps_identified: string[];
    mastery_demonstrations: Array<{
      concept: string;
      evidence: string;
      confidence_level: number;
    }>;
  };
  
  emotional_journey: {
    initial_emotional_state?: EmotionalState;
    emotional_states_timeline: Array<{
      timestamp: Date;
      emotion: EmotionalState;
      intensity: number;
      trigger?: string;
    }>;
    interventions_applied: Array<{
      timestamp: Date;
      intervention_type: string;
      reason: string;
      effectiveness_rating?: number; // user feedback
    }>;
    final_emotional_state?: EmotionalState;
    overall_emotional_health_score: number; // 0.0-1.0
  };
  
  performance_metrics: {
    questions_attempted: number;
    questions_correct: number;
    accuracy_percentage: number;
    average_response_time_seconds: number;
    difficulty_progression: Array<{
      timestamp: Date;
      difficulty_level: number;
      success_rate: number;
    }>;
    engagement_indicators: {
      follow_up_questions: number;
      self_directed_exploration: boolean;
      session_completion_rate: number;
    };
  };
  
  ai_usage_analytics: {
    total_ai_queries: number;
    providers_used: Record<string, number>; // provider -> query count
    total_tokens_consumed: number;
    total_cost_usd: number;
    average_response_time_ms: number;
    cache_hit_rate: number;
    quality_scores: number[]; // quality rating for each response
  };
  
  // Session outcome and effectiveness
  session_outcome: {
    learning_objectives_met: string[];
    areas_needing_more_work: string[];
    user_satisfaction_rating?: number; // 1-5 if provided
    perceived_difficulty_rating?: number; // user's perception
    emotional_satisfaction_rating?: number; // how user felt about emotional support
    next_session_recommendations: string[];
  };
}
```

### 1.4 emotional_analytics Collection
```typescript
interface EmotionalAnalyticsDocument {
  _id: ObjectId;
  user_id: string;
  analysis_date: Date; // daily aggregation
  data_source: 'daily_aggregation' | 'weekly_summary' | 'monthly_report';
  
  emotional_summary: {
    dominant_emotion: EmotionalState;
    emotion_distribution: Record<EmotionalState, {
      percentage: number;
      total_time_minutes: number;
      session_count: number;
    }>;
    emotional_stability_score: number; // 0.0-1.0, less volatility = higher score
    overall_emotional_wellness: number; // 0.0-1.0
  };
  
  learning_correlation_analysis: {
    most_productive_emotions: Array<{
      emotion: EmotionalState;
      learning_effectiveness_score: number;
      concepts_mastered_rate: number;
    }>;
    
    problematic_emotional_patterns: Array<{
      emotion: EmotionalState;
      learning_hindrance_score: number;
      common_triggers: string[];
      recovery_time_minutes: number;
    }>;
    
    optimal_learning_conditions: {
      best_emotional_state_for_new_concepts: EmotionalState;
      best_emotional_state_for_practice: EmotionalState;
      best_emotional_state_for_assessment: EmotionalState;
    };
  };
  
  intervention_effectiveness: {
    interventions_applied: Array<{
      intervention_type: string;
      application_count: number;
      success_rate: number; // 0.0-1.0
      average_effectiveness_rating: number;
    }>;
    
    emotional_recovery_patterns: {
      average_recovery_time_minutes: Record<EmotionalState, number>;
      most_effective_recovery_strategies: Array<{
        strategy: string;
        success_rate: number;
        applicable_emotions: EmotionalState[];
      }>;
    };
  };
  
  trends_and_patterns: {
    emotional_trends: Array<{
      emotion: EmotionalState;
      trend_direction: 'increasing' | 'decreasing' | 'stable';
      change_percentage: number;
      time_period: string;
    }>;
    
    temporal_patterns: {
      best_learning_times: string[]; // time of day when emotions are optimal
      challenging_periods: string[]; // times when negative emotions are common
      emotional_biorhythm_insights: string[];
    };
  };
}
```

### 1.5 content_mastery Collection
```typescript
interface ContentMasteryDocument {
  _id: ObjectId;
  user_id: string;
  subject: string;
  topic: string;
  concept: string;
  
  mastery_tracking: {
    current_mastery_level: number; // 0.0-1.0
    mastery_confidence: number; // 0.0-1.0 confidence in the assessment
    first_exposure_date: Date;
    last_practiced_date: Date;
    mastery_achieved_date?: Date; // when reached >0.8 mastery
  };
  
  learning_history: Array<{
    session_id: string;
    date: Date;
    activity_type: 'introduction' | 'practice' | 'assessment' | 'revision';
    performance_score: number; // 0.0-1.0
    difficulty_level: number;
    emotional_state: EmotionalState;
    time_spent_minutes: number;
  }>;
  
  spaced_repetition_data: {
    repetition_interval_days: number;
    next_review_date: Date;
    review_history: Array<{
      date: Date;
      performance_score: number;
      interval_adjustment: number;
    }>;
    forgetting_curve_parameters: {
      initial_strength: number;
      decay_rate: number;
      stability: number;
    };
  };
  
  prerequisite_relationships: {
    prerequisite_concepts: Array<{
      concept: string;
      importance_weight: number; // 0.0-1.0
      mastery_required: number; // minimum mastery level needed
    }>;
    dependent_concepts: Array<{
      concept: string;
      dependency_strength: number; // how much this concept helps with dependent
    }>;
  };
  
  personalization_data: {
    optimal_difficulty_level: number; // personalized difficulty for this concept
    effective_learning_strategies: string[];
    emotional_associations: Record<EmotionalState, number>; // performance by emotion
    learning_style_effectiveness: Record<string, number>; // effectiveness by learning style
  };
}
```

## Indexing Strategy

### 2.1 Performance-Critical Indexes
```typescript
interface DatabaseIndexes {
  users: [
    { email: 1 }, // unique index for login
    { user_id: 1 }, // unique index for external references
    { "subscription.status": 1, "subscription.expires_at": 1 }, // subscription management
    { created_at: -1 }, // recent users analytics
    { last_login: -1 } // user activity tracking
  ];
  
  learning_dna: [
    { user_id: 1 }, // unique index - one DNA per user
    { updated_at: -1 }, // recent updates for analytics
    { "cognitive_profile.learning_style.primary": 1 }, // learning style analytics
    { "learning_effectiveness.overall_learning_velocity": -1 } // performance analytics
  ];
  
  learning_sessions: [
    { user_id: 1, started_at: -1 }, // user session history (most common query)
    { session_id: 1 }, // unique index for session lookups
    { "session_context.subject": 1, started_at: -1 }, // subject-based analytics
    { started_at: -1 }, // chronological session analysis
    { "ai_usage_analytics.total_cost_usd": -1 }, // cost tracking and optimization
    { 
      user_id: 1, 
      "session_context.subject": 1, 
      started_at: -1 
    } // compound index for user-subject queries
  ];
  
  emotional_analytics: [
    { user_id: 1, analysis_date: -1 }, // user emotional trends
    { analysis_date: -1 }, // platform-wide emotional health analytics
    { "emotional_summary.dominant_emotion": 1 }, // emotion-based insights
    { "emotional_summary.overall_emotional_wellness": -1 } // wellness ranking
  ];
  
  content_mastery: [
    { user_id: 1, subject: 1 }, // user progress by subject
    { user_id: 1, "mastery_tracking.current_mastery_level": -1 }, // user strengths
    { subject: 1, topic: 1, "mastery_tracking.current_mastery_level": -1 }, // content difficulty analysis
    { "spaced_repetition_data.next_review_date": 1 }, // upcoming reviews query
    { 
      user_id: 1, 
      "spaced_repetition_data.next_review_date": 1 
    } // user-specific review schedule
  ];
}
```

### 2.2 Text Search Indexes
```typescript
interface TextSearchIndexes {
  // Full-text search for educational content
  learning_sessions: {
    "interactions.user_input": "text",
    "interactions.ai_response.text": "text",
    "session_context.topics": "text"
  };
  
  // Search user profiles for admin and analytics
  users: {
    "profile.name": "text",
    email: "text",
    "profile.learning_goals": "text"
  };
}
```

## Query Performance Targets

### 3.1 Critical Query Performance SLAs
```typescript
interface QueryPerformanceSLAs {
  user_authentication: {
    target_time_ms: 50;
    query: "find user by email for login";
    expected_frequency: "high";
  };
  
  user_profile_lookup: {
    target_time_ms: 100;
    query: "get complete user profile and learning DNA";
    expected_frequency: "very high";
  };
  
  recent_session_history: {
    target_time_ms: 200;
    query: "get user's last 30 days of learning sessions";
    expected_frequency: "high";
  };
  
  emotional_analytics_dashboard: {
    target_time_ms: 500;
    query: "aggregate emotional analytics for dashboard";
    expected_frequency: "medium";
  };
  
  content_mastery_lookup: {
    target_time_ms: 150;
    query: "get user's mastery levels for personalization";
    expected_frequency: "very high";
  };
  
  spaced_repetition_due: {
    target_time_ms: 100;
    query: "find concepts due for review today";
    expected_frequency: "daily";
  };
}
```

## Data Archival and Retention

### 4.1 Data Lifecycle Management
```typescript
interface DataRetentionPolicy {
  user_accounts: {
    active_data: "retain indefinitely while account active";
    deleted_accounts: "hard delete after 30 days";
    inactive_accounts: "archive after 2 years, delete after 5 years";
  };
  
  learning_sessions: {
    detailed_sessions: "retain 2 years for personalization";
    aggregated_data: "retain indefinitely for research";
    ai_interaction_logs: "retain 90 days for debugging";
  };
  
  emotional_analytics: {
    raw_emotional_data: "aggregate daily, delete raw data after 90 days";
    aggregated_emotional_data: "retain 5 years for longitudinal studies";
    sensitive_emotional_patterns: "user-controlled retention settings";
  };
  
  content_mastery: {
    mastery_data: "retain indefinitely for spaced repetition";
    learning_history: "retain 3 years for pattern analysis";
    prerequisite_relationships: "retain indefinitely as educational metadata";
  };
}
```

### 4.2 Archival Strategy
```typescript
interface ArchivalStrategy {
  hot_storage: {
    timeframe: "last 90 days";
    performance_target: "sub-100ms queries";
    storage_type: "primary MongoDB cluster";
  };
  
  warm_storage: {
    timeframe: "90 days to 2 years";
    performance_target: "sub-500ms queries";
    storage_type: "MongoDB with slower disks";
  };
  
  cold_storage: {
    timeframe: "2+ years";
    performance_target: "research queries only";
    storage_type: "compressed archives, S3 Glacier";
  };
}
```

## Privacy and Security

### 5.1 Data Encryption and Security
```typescript
interface DataSecurityMeasures {
  encryption_at_rest: {
    sensitive_fields: [
      "email", "password_hash", 
      "emotional_analytics.detailed_patterns",
      "learning_sessions.interactions.user_input"
    ];
    encryption_method: "AES-256-GCM";
    key_management: "AWS KMS with key rotation";
  };
  
  encryption_in_transit: {
    all_connections: "TLS 1.3 minimum";
    internal_communication: "mTLS for microservices";
    client_connections: "HTTPS only with HSTS";
  };
  
  access_controls: {
    role_based_access: "strict RBAC for admin functions";
    user_data_isolation: "users can only access their own data";
    audit_logging: "all data access logged with retention";
  };
  
  privacy_compliance: {
    gdpr_compliance: "user data export and deletion capabilities";
    coppa_compliance: "enhanced protections for users under 13";
    data_minimization: "collect only necessary data for functionality";
    user_consent: "granular consent for analytics and emotional data";
  };
}
```

## Backup and Disaster Recovery

### 6.1 Backup Strategy
```typescript
interface BackupStrategy {
  continuous_replication: {
    replica_set_members: 3; // minimum for high availability
    geographic_distribution: "primary + 2 replicas in different regions";
    sync_lag_target: "<1 second";
  };
  
  snapshot_backups: {
    frequency: "every 6 hours";
    retention: "30 daily, 12 monthly, 5 yearly";
    storage_location: "encrypted S3 with cross-region replication";
    restoration_testing: "monthly automated restoration tests";
  };
  
  point_in_time_recovery: {
    oplog_retention: "48 hours";
    recovery_granularity: "to the second";
    recovery_time_objective: "15 minutes";
    recovery_point_objective: "1 minute data loss maximum";
  };
}
```

## Analytics and Aggregation Pipelines

### 7.1 Real-time Analytics Queries
```typescript
interface AnalyticsQueries {
  user_progress_dashboard: {
    aggregation_pipeline: [
      { $match: { user_id: "{{user_id}}" } },
      { $lookup: { from: "content_mastery", localField: "user_id", foreignField: "user_id" } },
      { $group: { _id: "$subject", mastery_avg: { $avg: "$mastery_tracking.current_mastery_level" } } }
    ];
    cache_duration: "15 minutes";
    performance_target: "<200ms";
  };
  
  emotional_wellness_trends: {
    aggregation_pipeline: [
      { $match: { user_id: "{{user_id}}", analysis_date: { $gte: "{{start_date}}" } } },
      { $project: { date: "$analysis_date", wellness: "$emotional_summary.overall_emotional_wellness" } },
      { $sort: { date: 1 } }
    ];
    cache_duration: "1 hour";
    performance_target: "<300ms";
  };
  
  platform_usage_analytics: {
    aggregation_pipeline: [
      { $match: { started_at: { $gte: "{{start_date}}" } } },
      { $group: { 
          _id: { $dateToString: { format: "%Y-%m-%d", date: "$started_at" } },
          active_users: { $addToSet: "$user_id" },
          total_sessions: { $sum: 1 },
          avg_session_duration: { $avg: "$duration_minutes" }
        } }
    ];
    cache_duration: "6 hours";
    performance_target: "<1000ms";
  };
}
```

## Success Metrics

### 8.1 Database Performance KPIs
- **Query Response Time**: 95% of critical queries under SLA targets
- **Database Availability**: 99.99% uptime with automatic failover
- **Data Consistency**: Zero data loss during normal operations
- **Index Efficiency**: >90% of queries using appropriate indexes
- **Storage Growth**: Predictable growth patterns with 6-month capacity planning
- **Backup Recovery**: 100% success rate for restoration testing