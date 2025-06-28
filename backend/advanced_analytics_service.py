"""
Advanced Learning Analytics Service
Provides Knowledge Graph Mapping, Competency Heat Maps, Learning Velocity Tracking,
Retention Curves, and Learning Path Optimization using AI
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph"""
    id: str
    name: str
    description: str
    difficulty_level: float  # 0.0 to 1.0
    category: str
    prerequisites: List[str]
    related_concepts: List[str]
    mastery_threshold: float = 0.8
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class LearningEvent:
    """Represents a learning interaction event"""
    id: str
    user_id: str
    concept_id: str
    event_type: str  # 'question', 'explanation', 'practice', 'assessment'
    timestamp: datetime
    duration_seconds: int
    performance_score: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    session_id: str
    context: Dict[str, Any]

@dataclass
class CompetencyLevel:
    """Represents user competency in a specific concept"""
    concept_id: str
    current_level: float  # 0.0 to 1.0
    confidence: float
    last_updated: datetime
    learning_velocity: float  # rate of improvement
    retention_score: float
    practice_count: int
    mastered: bool = False

class AdvancedAnalyticsService:
    """Advanced Learning Analytics Service with AI-powered insights"""
    
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.user_competencies = defaultdict(dict)  # user_id -> concept_id -> CompetencyLevel
        self.learning_events = defaultdict(list)    # user_id -> [LearningEvent]
        self.concept_library = {}                   # concept_id -> ConceptNode
        self.learning_paths = defaultdict(list)     # user_id -> [concept_ids in order]
        
        # Initialize default concepts
        self._initialize_default_concepts()
        
    def _initialize_default_concepts(self):
        """Initialize the knowledge graph with common learning concepts"""
        default_concepts = [
            ConceptNode(
                id="basic_math",
                name="Basic Mathematics",
                description="Fundamental arithmetic operations",
                difficulty_level=0.2,
                category="mathematics",
                prerequisites=[],
                related_concepts=["algebra", "geometry"]
            ),
            ConceptNode(
                id="algebra",
                name="Algebra",
                description="Mathematical expressions and equations",
                difficulty_level=0.4,
                category="mathematics",
                prerequisites=["basic_math"],
                related_concepts=["calculus", "geometry"]
            ),
            ConceptNode(
                id="calculus",
                name="Calculus",
                description="Mathematical analysis of change",
                difficulty_level=0.8,
                category="mathematics",
                prerequisites=["algebra"],
                related_concepts=["differential_equations"]
            ),
            ConceptNode(
                id="programming_basics",
                name="Programming Fundamentals",
                description="Basic programming concepts",
                difficulty_level=0.3,
                category="computer_science",
                prerequisites=[],
                related_concepts=["data_structures", "algorithms"]
            ),
            ConceptNode(
                id="data_structures",
                name="Data Structures",
                description="Ways to organize and store data",
                difficulty_level=0.6,
                category="computer_science",
                prerequisites=["programming_basics"],
                related_concepts=["algorithms", "databases"]
            ),
            ConceptNode(
                id="algorithms",
                name="Algorithms",
                description="Problem-solving procedures",
                difficulty_level=0.7,
                category="computer_science",
                prerequisites=["programming_basics", "data_structures"],
                related_concepts=["machine_learning"]
            )
        ]
        
        for concept in default_concepts:
            self.add_concept(concept)
    
    def add_concept(self, concept: ConceptNode):
        """Add a concept to the knowledge graph"""
        self.concept_library[concept.id] = concept
        self.knowledge_graph.add_node(concept.id, **concept.__dict__)
        
        # Add prerequisite edges
        for prereq in concept.prerequisites:
            if prereq in self.concept_library:
                self.knowledge_graph.add_edge(prereq, concept.id, relationship="prerequisite")
        
        # Add related concept edges
        for related in concept.related_concepts:
            if related in self.concept_library:
                self.knowledge_graph.add_edge(concept.id, related, relationship="related")
                self.knowledge_graph.add_edge(related, concept.id, relationship="related")

    async def record_learning_event(self, event: LearningEvent):
        """Record a learning interaction event"""
        self.learning_events[event.user_id].append(event)
        
        # Update competency level
        await self._update_competency(event)
        
        # No need to optimize learning path on every event
        # We'll optimize on demand when the endpoint is called

    async def _update_competency(self, event: LearningEvent):
        """Update user competency based on learning event"""
        user_id = event.user_id
        concept_id = event.concept_id
        
        if concept_id not in self.user_competencies[user_id]:
            self.user_competencies[user_id][concept_id] = CompetencyLevel(
                concept_id=concept_id,
                current_level=0.0,
                confidence=0.0,
                last_updated=datetime.utcnow(),
                learning_velocity=0.0,
                retention_score=1.0,
                practice_count=0
            )
        
        competency = self.user_competencies[user_id][concept_id]
        
        # Calculate time-based decay for retention
        time_diff = (event.timestamp - competency.last_updated).total_seconds() / 3600  # hours
        retention_decay = max(0.8, 1.0 - (time_diff * 0.01))  # Slight decay over time
        
        # Update competency using weighted average
        old_level = competency.current_level
        weight = min(1.0, 0.1 + (event.performance_score * 0.3))
        
        competency.current_level = (
            old_level * (1 - weight) + 
            event.performance_score * weight
        ) * retention_decay
        
        # Update confidence based on consistency
        competency.confidence = min(1.0, competency.confidence * 0.9 + 
                                  abs(event.performance_score - event.confidence_level) * 0.1)
        
        # Calculate learning velocity (improvement rate)
        if competency.practice_count > 0:
            improvement = competency.current_level - old_level
            competency.learning_velocity = improvement / (time_diff + 0.1)
        
        competency.retention_score = retention_decay
        competency.practice_count += 1
        competency.last_updated = event.timestamp
        
        # Check mastery
        concept = self.concept_library.get(concept_id)
        if concept and competency.current_level >= concept.mastery_threshold:
            competency.mastered = True

    async def generate_knowledge_graph_mapping(self, user_id: str) -> Dict[str, Any]:
        """Generate personalized knowledge graph mapping for user"""
        user_competencies = self.user_competencies.get(user_id, {})
        
        # Create personalized graph data
        nodes = []
        edges = []
        
        for concept_id, concept in self.concept_library.items():
            competency = user_competencies.get(concept_id)
            
            node_data = {
                "id": concept_id,
                "name": concept.name,
                "description": concept.description,
                "category": concept.category,
                "difficulty": concept.difficulty_level,
                "mastery_level": competency.current_level if competency else 0.0,
                "confidence": competency.confidence if competency else 0.0,
                "mastered": competency.mastered if competency else False,
                "size": 20 + (competency.current_level * 30 if competency else 10),
                "color": self._get_competency_color(competency.current_level if competency else 0.0)
            }
            nodes.append(node_data)
        
        # Add edges with relationship data
        for edge in self.knowledge_graph.edges(data=True):
            source, target, data = edge
            edge_data = {
                "source": source,
                "target": target,
                "relationship": data.get("relationship", "related"),
                "strength": self._calculate_edge_strength(user_id, source, target)
            }
            edges.append(edge_data)
        
        # Calculate learning recommendations
        recommendations = await self._generate_learning_recommendations(user_id)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "recommendations": recommendations,
            "user_progress": {
                "total_concepts": len(self.concept_library),
                "mastered_concepts": sum(1 for c in user_competencies.values() if c.mastered),
                "average_competency": np.mean([c.current_level for c in user_competencies.values()]) if user_competencies else 0.0,
                "learning_velocity": np.mean([c.learning_velocity for c in user_competencies.values()]) if user_competencies else 0.0
            }
        }

    def _get_competency_color(self, level: float) -> str:
        """Get color based on competency level"""
        if level >= 0.8:
            return "#4ade80"  # Green - Mastered
        elif level >= 0.6:
            return "#fbbf24"  # Yellow - Proficient
        elif level >= 0.3:
            return "#f97316"  # Orange - Learning
        else:
            return "#ef4444"  # Red - Beginner

    def _calculate_edge_strength(self, user_id: str, source: str, target: str) -> float:
        """Calculate the strength of connection between concepts for user"""
        source_comp = self.user_competencies.get(user_id, {}).get(source)
        target_comp = self.user_competencies.get(user_id, {}).get(target)
        
        if not source_comp or not target_comp:
            return 0.5
        
        # Stronger connection if both concepts are well understood
        return (source_comp.current_level + target_comp.current_level) / 2

    async def generate_competency_heat_map(self, user_id: str, time_period: int = 30) -> Dict[str, Any]:
        """Generate competency heat map for user over time period (days)"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_period)
        
        # Get user events in time period
        user_events = self.learning_events.get(user_id, [])
        period_events = [e for e in user_events if start_date <= e.timestamp <= end_date]
        
        # Group events by day and concept
        daily_competency = defaultdict(lambda: defaultdict(list))
        
        for event in period_events:
            day_key = event.timestamp.strftime("%Y-%m-%d")
            daily_competency[day_key][event.concept_id].append(event.performance_score)
        
        # Calculate daily averages
        heat_map_data = []
        concepts = list(self.concept_library.keys())
        
        for i in range(time_period):
            date = start_date + timedelta(days=i)
            day_key = date.strftime("%Y-%m-%d")
            
            day_data = {
                "date": day_key,
                "day_of_week": date.strftime("%A"),
                "competencies": {}
            }
            
            for concept_id in concepts:
                if concept_id in daily_competency[day_key]:
                    scores = daily_competency[day_key][concept_id]
                    avg_score = np.mean(scores)
                    day_data["competencies"][concept_id] = {
                        "score": avg_score,
                        "activity_count": len(scores),
                        "improvement": self._calculate_daily_improvement(user_id, concept_id, date)
                    }
                else:
                    day_data["competencies"][concept_id] = {
                        "score": 0.0,
                        "activity_count": 0,
                        "improvement": 0.0
                    }
            
            heat_map_data.append(day_data)
        
        return {
            "heat_map_data": heat_map_data,
            "concepts": [{"id": c.id, "name": c.name, "category": c.category} 
                        for c in self.concept_library.values()],
            "summary": {
                "most_active_day": self._find_most_active_day(daily_competency),
                "strongest_concepts": self._find_strongest_concepts(user_id),
                "improvement_trends": self._calculate_improvement_trends(user_id, time_period)
            }
        }

    def _calculate_daily_improvement(self, user_id: str, concept_id: str, date: datetime) -> float:
        """Calculate improvement for a concept on a specific day"""
        user_events = self.learning_events.get(user_id, [])
        day_start = date.replace(hour=0, minute=0, second=0)
        day_end = day_start + timedelta(days=1)
        
        day_events = [e for e in user_events 
                     if e.concept_id == concept_id and day_start <= e.timestamp < day_end]
        
        if len(day_events) < 2:
            return 0.0
        
        day_events.sort(key=lambda x: x.timestamp)
        return day_events[-1].performance_score - day_events[0].performance_score

    async def track_learning_velocity(self, user_id: str, window_days: int = 7) -> Dict[str, Any]:
        """Track learning velocity over a rolling window"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=window_days * 4)  # Look back further for trend
        
        user_events = self.learning_events.get(user_id, [])
        period_events = [e for e in user_events if start_date <= e.timestamp <= end_date]
        
        # Calculate velocity for each concept over rolling windows
        velocity_data = {}
        
        for concept_id in self.concept_library.keys():
            concept_events = [e for e in period_events if e.concept_id == concept_id]
            if len(concept_events) < 2:
                continue
            
            concept_events.sort(key=lambda x: x.timestamp)
            
            # Calculate velocity over different time windows
            velocities = []
            for i in range(len(concept_events) - 1):
                current = concept_events[i]
                next_event = concept_events[i + 1]
                
                time_diff = (next_event.timestamp - current.timestamp).total_seconds() / 3600
                if time_diff > 0:
                    score_diff = next_event.performance_score - current.performance_score
                    velocity = score_diff / time_diff
                    velocities.append({
                        "timestamp": next_event.timestamp,
                        "velocity": velocity,
                        "score": next_event.performance_score
                    })
            
            if velocities:
                velocity_data[concept_id] = {
                    "concept_name": self.concept_library[concept_id].name,
                    "velocities": velocities,
                    "average_velocity": np.mean([v["velocity"] for v in velocities]),
                    "current_velocity": velocities[-1]["velocity"] if velocities else 0.0,
                    "velocity_trend": self._calculate_velocity_trend(velocities)
                }
        
        return {
            "velocity_data": velocity_data,
            "overall_velocity": np.mean([v["average_velocity"] for v in velocity_data.values()]) 
                              if velocity_data else 0.0,
            "accelerating_concepts": [k for k, v in velocity_data.items() 
                                    if v["velocity_trend"] > 0.1],
            "stalling_concepts": [k for k, v in velocity_data.items() 
                                if v["velocity_trend"] < -0.1]
        }

    def _calculate_velocity_trend(self, velocities: List[Dict]) -> float:
        """Calculate if velocity is increasing or decreasing"""
        if len(velocities) < 3:
            return 0.0
        
        recent = velocities[-3:]
        return (recent[-1]["velocity"] - recent[0]["velocity"]) / 3

    async def generate_retention_curves(self, user_id: str) -> Dict[str, Any]:
        """Generate retention curves showing how well knowledge is retained"""
        user_competencies = self.user_competencies.get(user_id, {})
        user_events = self.learning_events.get(user_id, [])
        
        retention_curves = {}
        
        for concept_id, competency in user_competencies.items():
            concept_events = [e for e in user_events if e.concept_id == concept_id]
            if len(concept_events) < 2:
                continue
            
            concept_events.sort(key=lambda x: x.timestamp)
            
            # Calculate retention over time
            retention_points = []
            peak_performance = 0.0
            
            for i, event in enumerate(concept_events):
                # Update peak performance
                peak_performance = max(peak_performance, event.performance_score)
                
                # Calculate time since last practice
                if i > 0:
                    time_gap = (event.timestamp - concept_events[i-1].timestamp).total_seconds() / 3600
                    
                    # Calculate retention based on forgetting curve
                    retention_factor = self._calculate_retention_factor(time_gap)
                    retention_score = event.performance_score / max(0.1, peak_performance)
                    
                    retention_points.append({
                        "timestamp": event.timestamp,
                        "time_gap_hours": time_gap,
                        "retention_score": retention_score * retention_factor,
                        "performance_score": event.performance_score,
                        "peak_performance": peak_performance
                    })
            
            if retention_points:
                retention_curves[concept_id] = {
                    "concept_name": self.concept_library[concept_id].name,
                    "retention_points": retention_points,
                    "average_retention": np.mean([p["retention_score"] for p in retention_points]),
                    "retention_half_life": self._calculate_retention_half_life(retention_points),
                    "forgetting_curve_fit": self._fit_forgetting_curve(retention_points)
                }
        
        return {
            "retention_curves": retention_curves,
            "overall_retention": np.mean([c["average_retention"] for c in retention_curves.values()]) 
                               if retention_curves else 0.0,
            "strongest_retention": max(retention_curves.items(), 
                                     key=lambda x: x[1]["average_retention"]) if retention_curves else None,
            "weakest_retention": min(retention_curves.items(), 
                                   key=lambda x: x[1]["average_retention"]) if retention_curves else None
        }

    def _calculate_retention_factor(self, hours_gap: float) -> float:
        """Calculate retention factor based on time gap (forgetting curve)"""
        # Simple exponential decay model
        return np.exp(-hours_gap / 24)  # Half-life of 24 hours

    def _calculate_retention_half_life(self, retention_points: List[Dict]) -> float:
        """Calculate the half-life of retention for this concept"""
        if len(retention_points) < 3:
            return 24.0  # Default 24 hours
        
        # Find where retention drops to 50% of peak
        for point in retention_points:
            if point["retention_score"] <= 0.5:
                return point["time_gap_hours"]
        
        return 48.0  # Default if retention doesn't drop significantly

    def _fit_forgetting_curve(self, retention_points: List[Dict]) -> Dict[str, float]:
        """Fit an exponential forgetting curve to the data"""
        if len(retention_points) < 3:
            return {"decay_rate": -0.1, "initial_strength": 1.0, "r_squared": 0.0}
        
        # Simple exponential fit
        x = np.array([p["time_gap_hours"] for p in retention_points])
        y = np.array([p["retention_score"] for p in retention_points])
        
        # Fit y = a * exp(b * x)
        try:
            log_y = np.log(np.maximum(y, 0.01))  # Avoid log(0)
            coeffs = np.polyfit(x, log_y, 1)
            
            return {
                "decay_rate": coeffs[0],
                "initial_strength": np.exp(coeffs[1]),
                "r_squared": np.corrcoef(x, log_y)[0, 1]**2
            }
        except:
            return {"decay_rate": -0.1, "initial_strength": 1.0, "r_squared": 0.0}

    async def optimize_learning_path(self, user_id: str) -> Dict[str, Any]:
        """Generate AI-optimized learning path for user"""
        user_competencies = self.user_competencies.get(user_id, {})
        
        # Get unmastered concepts
        unmastered = []
        for concept_id, concept in self.concept_library.items():
            competency = user_competencies.get(concept_id)
            if not competency or not competency.mastered:
                unmastered.append(concept_id)
        
        # Calculate readiness score for each unmastered concept
        readiness_scores = {}
        for concept_id in unmastered:
            readiness_scores[concept_id] = await self._calculate_readiness_score(user_id, concept_id)
        
        # Sort by readiness score
        optimal_order = sorted(unmastered, key=lambda x: readiness_scores[x], reverse=True)
        
        # Group into learning phases
        learning_phases = self._group_into_phases(optimal_order, readiness_scores)
        
        # Generate detailed path with recommendations
        detailed_path = []
        for i, concept_id in enumerate(optimal_order[:10]):  # Top 10 recommendations
            concept = self.concept_library[concept_id]
            competency = user_competencies.get(concept_id)
            
            path_item = {
                "position": i + 1,
                "concept_id": concept_id,
                "concept_name": concept.name,
                "description": concept.description,
                "category": concept.category,
                "difficulty": concept.difficulty_level,
                "readiness_score": readiness_scores[concept_id],
                "current_competency": competency.current_level if competency else 0.0,
                "estimated_time_hours": self._estimate_learning_time(concept, competency),
                "prerequisites_met": self._check_prerequisites_met(user_id, concept_id),
                "learning_strategy": self._recommend_learning_strategy(concept, competency)
            }
            detailed_path.append(path_item)
        
        return {
            "optimal_path": detailed_path,
            "learning_phases": learning_phases,
            "total_estimated_hours": sum(item["estimated_time_hours"] for item in detailed_path),
            "priority_concepts": optimal_order[:3],
            "path_confidence": self._calculate_path_confidence(user_id, optimal_order),
            "adaptive_recommendations": await self._generate_adaptive_recommendations(user_id)
        }

    async def _calculate_readiness_score(self, user_id: str, concept_id: str) -> float:
        """Calculate how ready a user is to learn a specific concept"""
        concept = self.concept_library[concept_id]
        user_competencies = self.user_competencies.get(user_id, {})
        
        # Check prerequisite mastery
        prereq_score = 1.0
        if concept.prerequisites:
            prereq_levels = []
            for prereq in concept.prerequisites:
                competency = user_competencies.get(prereq)
                if competency:
                    prereq_levels.append(competency.current_level)
                else:
                    prereq_levels.append(0.0)
            prereq_score = np.mean(prereq_levels)
        
        # Current competency level (inverted - lower competency = higher priority)
        current_competency = user_competencies.get(concept_id)
        competency_gap = 1.0 - (current_competency.current_level if current_competency else 0.0)
        
        # Learning velocity potential (based on related concepts)
        velocity_potential = self._calculate_velocity_potential(user_id, concept_id)
        
        # Interest/engagement factor (based on recent activity in category)
        engagement_factor = self._calculate_engagement_factor(user_id, concept.category)
        
        # Combine factors
        readiness_score = (
            prereq_score * 0.4 +           # Can they learn it?
            competency_gap * 0.3 +         # Do they need it?
            velocity_potential * 0.2 +     # Will they learn it quickly?
            engagement_factor * 0.1        # Are they interested?
        )
        
        return readiness_score

    def _calculate_velocity_potential(self, user_id: str, concept_id: str) -> float:
        """Calculate potential learning velocity for a concept"""
        concept = self.concept_library[concept_id]
        user_competencies = self.user_competencies.get(user_id, {})
        
        # Look at velocity in related concepts
        related_velocities = []
        for related_id in concept.related_concepts:
            competency = user_competencies.get(related_id)
            if competency:
                related_velocities.append(competency.learning_velocity)
        
        if related_velocities:
            return np.mean(related_velocities)
        
        # Default based on concept difficulty (easier concepts = higher potential)
        return 1.0 - concept.difficulty_level

    def _calculate_engagement_factor(self, user_id: str, category: str) -> float:
        """Calculate user engagement with a specific category"""
        user_events = self.learning_events.get(user_id, [])
        recent_events = [e for e in user_events 
                        if (datetime.utcnow() - e.timestamp).days <= 7]
        
        category_events = [e for e in recent_events 
                          if self.concept_library.get(e.concept_id, {}).category == category]
        
        if not recent_events:
            return 0.5
        
        return len(category_events) / len(recent_events)

    def _group_into_phases(self, optimal_order: List[str], readiness_scores: Dict[str, float]) -> List[Dict]:
        """Group concepts into learning phases"""
        phases = []
        current_phase = []
        phase_threshold = 0.1
        
        last_score = 1.0
        for concept_id in optimal_order:
            score = readiness_scores[concept_id]
            
            if last_score - score > phase_threshold and current_phase:
                phases.append({
                    "phase_number": len(phases) + 1,
                    "concepts": current_phase.copy(),
                    "average_readiness": np.mean([readiness_scores[c] for c in current_phase]),
                    "estimated_duration_days": len(current_phase) * 3
                })
                current_phase = []
            
            current_phase.append(concept_id)
            last_score = score
        
        if current_phase:
            phases.append({
                "phase_number": len(phases) + 1,
                "concepts": current_phase,
                "average_readiness": np.mean([readiness_scores[c] for c in current_phase]),
                "estimated_duration_days": len(current_phase) * 3
            })
        
        return phases

    def _estimate_learning_time(self, concept: ConceptNode, competency: Optional[CompetencyLevel]) -> float:
        """Estimate time needed to master a concept"""
        base_time = concept.difficulty_level * 10  # Base hours based on difficulty
        
        if competency:
            # Reduce time based on current competency
            remaining_competency = max(0.1, concept.mastery_threshold - competency.current_level)
            time_modifier = remaining_competency / concept.mastery_threshold
            
            # Adjust based on learning velocity
            if competency.learning_velocity > 0:
                velocity_modifier = 1.0 / (1.0 + competency.learning_velocity)
            else:
                velocity_modifier = 1.2
            
            return base_time * time_modifier * velocity_modifier
        
        return base_time

    def _check_prerequisites_met(self, user_id: str, concept_id: str) -> bool:
        """Check if all prerequisites for a concept are met"""
        concept = self.concept_library[concept_id]
        user_competencies = self.user_competencies.get(user_id, {})
        
        for prereq in concept.prerequisites:
            competency = user_competencies.get(prereq)
            if not competency or competency.current_level < 0.7:
                return False
        
        return True

    def _recommend_learning_strategy(self, concept: ConceptNode, competency: Optional[CompetencyLevel]) -> str:
        """Recommend optimal learning strategy for a concept"""
        if not competency or competency.current_level < 0.3:
            return "foundational_learning"
        elif competency.current_level < 0.6:
            return "structured_practice"
        elif competency.current_level < 0.8:
            return "applied_practice"
        else:
            return "mastery_reinforcement"

    def _calculate_path_confidence(self, user_id: str, optimal_order: List[str]) -> float:
        """Calculate confidence in the recommended learning path"""
        user_competencies = self.user_competencies.get(user_id, {})
        
        if len(user_competencies) < 3:
            return 0.5  # Low confidence with little data
        
        # Calculate based on consistency of learning velocity
        velocities = [c.learning_velocity for c in user_competencies.values()]
        velocity_std = np.std(velocities) if velocities else 1.0
        
        # Lower standard deviation = more predictable learning = higher confidence
        confidence = max(0.3, 1.0 - velocity_std)
        
        return confidence

    async def _generate_adaptive_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate adaptive learning recommendations"""
        user_competencies = self.user_competencies.get(user_id, {})
        user_events = self.learning_events.get(user_id, [])
        
        recommendations = []
        
        # Time-based recommendations
        if user_events:
            last_event = max(user_events, key=lambda x: x.timestamp)
            hours_since = (datetime.utcnow() - last_event.timestamp).total_seconds() / 3600
            
            if hours_since > 24:
                recommendations.append({
                    "type": "review_reminder",
                    "message": "It's been a while! Consider reviewing your recent concepts to maintain retention.",
                    "action": "review_recent",
                    "priority": "medium"
                })
        
        # Performance-based recommendations
        struggling_concepts = [
            (concept_id, comp) for concept_id, comp in user_competencies.items()
            if comp.current_level < 0.5 and comp.practice_count > 3
        ]
        
        if struggling_concepts:
            concept_id, comp = min(struggling_concepts, key=lambda x: x[1].current_level)
            recommendations.append({
                "type": "struggling_concept",
                "message": f"Consider taking a break from {self.concept_library[concept_id].name} and reviewing prerequisites.",
                "action": "review_prerequisites",
                "concept_id": concept_id,
                "priority": "high"
            })
        
        return recommendations

    async def _generate_learning_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate learning recommendations for knowledge graph"""
        user_competencies = self.user_competencies.get(user_id, {})
        
        recommendations = []
        
        # Find next concepts to learn
        for concept_id, concept in self.concept_library.items():
            competency = user_competencies.get(concept_id)
            
            if not competency or competency.current_level < 0.8:
                prereq_ready = self._check_prerequisites_met(user_id, concept_id)
                
                if prereq_ready:
                    recommendations.append({
                        "concept_id": concept_id,
                        "concept_name": concept.name,
                        "type": "next_concept",
                        "reason": "Prerequisites met, ready to learn",
                        "priority": 1.0 - concept.difficulty_level
                    })
                elif competency and competency.current_level > 0.3:
                    recommendations.append({
                        "concept_id": concept_id,
                        "concept_name": concept.name,
                        "type": "review_prerequisites",
                        "reason": "Need to strengthen prerequisites first",
                        "priority": 0.5
                    })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations

    def _find_most_active_day(self, daily_competency: Dict) -> str:
        """Find the most active learning day"""
        activity_counts = {}
        for day, concepts in daily_competency.items():
            total_activity = sum(len(scores) for scores in concepts.values())
            activity_counts[day] = total_activity
        
        if activity_counts:
            return max(activity_counts.items(), key=lambda x: x[1])[0]
        return "No activity recorded"

    def _find_strongest_concepts(self, user_id: str) -> List[Dict[str, Any]]:
        """Find user's strongest concepts"""
        user_competencies = self.user_competencies.get(user_id, {})
        
        strongest = sorted(
            user_competencies.items(),
            key=lambda x: x[1].current_level,
            reverse=True
        )[:3]
        
        return [
            {
                "concept_id": concept_id,
                "concept_name": self.concept_library[concept_id].name,
                "level": comp.current_level
            }
            for concept_id, comp in strongest
        ]

    def _calculate_improvement_trends(self, user_id: str, days: int) -> Dict[str, float]:
        """Calculate improvement trends over time period"""
        user_events = self.learning_events.get(user_id, [])
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        period_events = [e for e in user_events if start_date <= e.timestamp <= end_date]
        
        trends = {}
        for concept_id in self.concept_library.keys():
            concept_events = [e for e in period_events if e.concept_id == concept_id]
            if len(concept_events) >= 2:
                concept_events.sort(key=lambda x: x.timestamp)
                start_score = concept_events[0].performance_score
                end_score = concept_events[-1].performance_score
                trends[concept_id] = end_score - start_score
        
        return trends

# Global instance
advanced_analytics_service = AdvancedAnalyticsService()