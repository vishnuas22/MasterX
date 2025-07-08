"""
MasterX Advanced Knowledge Graph & Contextual Intelligence Engine
=================================================================

This module provides sophisticated knowledge graph construction, semantic understanding,
and contextual intelligence for adaptive learning optimization.

Features:
- Dynamic knowledge graph construction and maintenance
- Semantic relationship discovery and mapping
- Contextual understanding and adaptation
- Concept mastery tracking and prediction
- Learning path optimization using graph algorithms
- Cross-domain knowledge transfer detection
- Prerequisite and dependency analysis
- Intelligent content sequencing

Author: MasterX AI Team
Version: 2.0 (Premium Algorithm Suite)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import asyncio
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import uuid
import math
import pickle
import os
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE ENUMS AND DATA STRUCTURES
# ============================================================================

class ConceptType(Enum):
    """Types of learning concepts"""
    FUNDAMENTAL = "fundamental"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    SPECIALIZED = "specialized"
    INTERDISCIPLINARY = "interdisciplinary"
    SKILL = "skill"
    THEORY = "theory"
    APPLICATION = "application"
    TOOL = "tool"
    METHODOLOGY = "methodology"

class RelationshipType(Enum):
    """Types of relationships between concepts"""
    PREREQUISITE = "prerequisite"
    BUILDS_ON = "builds_on"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    APPLIES_TO = "applies_to"
    EXEMPLIFIES = "exemplifies"
    CONTRASTS_WITH = "contrasts_with"
    PART_OF = "part_of"
    LEADS_TO = "leads_to"
    REQUIRES_UNDERSTANDING = "requires_understanding"

class MasteryLevel(Enum):
    """Levels of concept mastery"""
    UNKNOWN = 0
    AWARENESS = 1
    BASIC_UNDERSTANDING = 2
    FUNCTIONAL_KNOWLEDGE = 3
    PROFICIENT = 4
    ADVANCED = 5
    EXPERT = 6

class ContextType(Enum):
    """Types of learning contexts"""
    ACADEMIC = "academic"
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    RESEARCH = "research"
    CREATIVE = "creative"
    COLLABORATIVE = "collaborative"
    PRACTICAL = "practical"
    THEORETICAL = "theoretical"

@dataclass
class Concept:
    """Represents a learning concept in the knowledge graph"""
    concept_id: str
    name: str
    description: str
    concept_type: ConceptType
    domain: str
    subdomain: str
    difficulty_level: float
    complexity_score: float
    importance_score: float
    prerequisites: List[str]
    learning_objectives: List[str]
    keywords: List[str]
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConceptRelationship:
    """Represents a relationship between concepts"""
    relationship_id: str
    source_concept_id: str
    target_concept_id: str
    relationship_type: RelationshipType
    strength: float
    confidence: float
    bidirectional: bool
    context: List[ContextType]
    evidence: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserConceptMastery:
    """Tracks user's mastery of a concept"""
    user_id: str
    concept_id: str
    mastery_level: MasteryLevel
    confidence_score: float
    last_practiced: datetime
    practice_count: int
    time_to_mastery: Optional[timedelta]
    learning_path: List[str]
    difficulty_encountered: float
    help_needed: bool
    review_frequency: float
    transfer_applications: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class LearningPath:
    """Represents an optimized learning path"""
    path_id: str
    user_id: str
    goal_concepts: List[str]
    path_concepts: List[str]
    estimated_duration: timedelta
    difficulty_progression: List[float]
    personalization_factors: Dict[str, Any]
    success_probability: float
    alternative_paths: List[str]
    checkpoints: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ContextualState:
    """Represents current contextual state for learning"""
    user_id: str
    session_id: str
    current_context: ContextType
    domain_focus: str
    learning_objectives: List[str]
    available_time: timedelta
    difficulty_preference: float
    collaboration_available: bool
    tools_available: List[str]
    prior_knowledge: Dict[str, float]
    emotional_state: str
    motivation_level: float
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# NEURAL NETWORKS FOR KNOWLEDGE GRAPH PROCESSING
# ============================================================================

class ConceptEmbeddingNetwork(nn.Module):
    """
    Neural network for learning concept embeddings
    """
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 512, hidden_dims: List[int] = [1024, 512, 256]):
        super(ConceptEmbeddingNetwork, self).__init__()
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Concept description encoder
        self.description_encoder = nn.LSTM(embedding_dim, hidden_dims[0] // 2, num_layers=2, 
                                         batch_first=True, bidirectional=True)
        
        # Feature processing layers
        layers = []
        input_dim = hidden_dims[0] + 50  # LSTM output + additional features
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.feature_network = nn.Sequential(*layers)
        
        # Output embedding
        self.output_embedding = nn.Linear(input_dim, 256)
        
    def forward(self, description_tokens: torch.Tensor, additional_features: torch.Tensor):
        # Embed description tokens
        embedded_desc = self.word_embedding(description_tokens)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.description_encoder(embedded_desc)
        
        # Use final hidden state
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concatenate bidirectional
        
        # Combine with additional features
        combined_features = torch.cat([final_hidden, additional_features], dim=1)
        
        # Process through feature network
        processed_features = self.feature_network(combined_features)
        
        # Generate final embedding
        concept_embedding = self.output_embedding(processed_features)
        
        return F.normalize(concept_embedding, p=2, dim=1)  # L2 normalize

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for learning on knowledge graphs
    """
    def __init__(self, node_features: int = 256, hidden_dims: List[int] = [512, 256, 128], output_dim: int = 64):
        super(GraphNeuralNetwork, self).__init__()
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        input_dim = node_features
        for hidden_dim in hidden_dims:
            self.conv_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Attention mechanism for important nodes
        self.attention = GATConv(input_dim, output_dim, heads=4, concat=False)
        
        # Output layers
        self.classifier = nn.Linear(output_dim, 1)  # For mastery prediction
        self.difficulty_predictor = nn.Linear(output_dim, 1)
        self.prerequisite_predictor = nn.Linear(output_dim, 1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None):
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        
        # Apply attention
        x = self.attention(x, edge_index)
        
        # Generate predictions
        mastery_pred = torch.sigmoid(self.classifier(x))
        difficulty_pred = torch.sigmoid(self.difficulty_predictor(x)) * 6.0  # Scale to 1-6
        prerequisite_pred = torch.sigmoid(self.prerequisite_predictor(x))
        
        return {
            'node_embeddings': x,
            'mastery_prediction': mastery_pred,
            'difficulty_prediction': difficulty_pred,
            'prerequisite_prediction': prerequisite_pred
        }

class LearningPathOptimizer(nn.Module):
    """
    Neural network for optimizing learning paths
    """
    def __init__(self, concept_embedding_dim: int = 256, user_embedding_dim: int = 128, 
                 path_length: int = 20):
        super(LearningPathOptimizer, self).__init__()
        
        self.path_length = path_length
        
        # Encoder for current state
        self.state_encoder = nn.Sequential(
            nn.Linear(concept_embedding_dim + user_embedding_dim + 50, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # LSTM for path generation
        self.path_generator = nn.LSTM(256 + concept_embedding_dim, 512, num_layers=2, 
                                    batch_first=True, dropout=0.3)
        
        # Attention mechanism for concept selection
        self.concept_attention = nn.MultiheadAttention(512, num_heads=8, dropout=0.1)
        
        # Concept selector
        self.concept_selector = nn.Linear(512, 1)
        
        # Path quality estimator
        self.quality_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, current_state: torch.Tensor, user_embedding: torch.Tensor, 
                concept_embeddings: torch.Tensor, concept_mask: torch.Tensor = None):
        batch_size = current_state.size(0)
        num_concepts = concept_embeddings.size(1)
        
        # Encode current state
        state_features = self.state_encoder(current_state)
        
        # Initialize path generation
        paths = []
        hidden_state = None
        current_input = state_features.unsqueeze(1)
        
        for step in range(self.path_length):
            # Generate next step features
            lstm_out, hidden_state = self.path_generator(current_input, hidden_state)
            
            # Apply attention to select best concept
            query = lstm_out.transpose(0, 1)  # (1, batch, features)
            key = value = concept_embeddings.transpose(0, 1)  # (num_concepts, batch, features)
            
            attended_concepts, attention_weights = self.concept_attention(query, key, value)
            attended_concepts = attended_concepts.transpose(0, 1)  # (batch, 1, features)
            
            # Select concept
            concept_scores = self.concept_selector(attended_concepts).squeeze(-1)  # (batch, 1)
            
            if concept_mask is not None:
                concept_scores = concept_scores.masked_fill(~concept_mask, float('-inf'))
            
            selected_concept_idx = torch.argmax(concept_scores, dim=1)
            selected_concept = concept_embeddings[torch.arange(batch_size), selected_concept_idx]
            
            paths.append(selected_concept_idx)
            
            # Update input for next step
            current_input = torch.cat([state_features, selected_concept], dim=1).unsqueeze(1)
        
        # Estimate path quality
        final_hidden = hidden_state[0][-1]  # Last layer, last timestep
        path_quality = torch.sigmoid(self.quality_estimator(final_hidden))
        
        path_tensor = torch.stack(paths, dim=1)  # (batch, path_length)
        
        return {
            'path': path_tensor,
            'quality_score': path_quality,
            'attention_weights': attention_weights
        }

# ============================================================================
# ADVANCED KNOWLEDGE GRAPH ENGINE
# ============================================================================

class AdvancedKnowledgeGraphEngine:
    """
    Advanced knowledge graph engine for semantic understanding and learning optimization
    """
    def __init__(self):
        # Core graph structure
        self.graph = nx.MultiDiGraph()
        self.concepts = {}
        self.relationships = {}
        self.user_mastery = defaultdict(dict)
        
        # Neural networks
        self.concept_embedding_network = ConceptEmbeddingNetwork()
        self.graph_neural_network = GraphNeuralNetwork()
        self.path_optimizer = LearningPathOptimizer()
        
        # Semantic processing
        self.text_processor = SemanticTextProcessor()
        self.concept_extractor = ConceptExtractor()
        self.relationship_detector = RelationshipDetector()
        
        # Caching and optimization
        self.embedding_cache = {}
        self.path_cache = {}
        self.similarity_cache = {}
        
        # Analytics and metrics
        self.graph_metrics = GraphMetrics()
        self.learning_analytics = LearningAnalytics()
        
        logger.info("AdvancedKnowledgeGraphEngine initialized")
    
    async def add_concept(self, concept: Concept) -> bool:
        """Add a new concept to the knowledge graph"""
        try:
            # Store concept
            self.concepts[concept.concept_id] = concept
            
            # Add to graph
            self.graph.add_node(
                concept.concept_id,
                name=concept.name,
                concept_type=concept.concept_type.value,
                domain=concept.domain,
                difficulty=concept.difficulty_level,
                importance=concept.importance_score,
                **concept.metadata
            )
            
            # Generate embedding
            concept.embedding = await self._generate_concept_embedding(concept)
            
            # Update relationships
            await self._update_concept_relationships(concept)
            
            logger.info(f"Added concept: {concept.name} ({concept.concept_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding concept {concept.concept_id}: {str(e)}")
            return False
    
    async def add_relationship(self, relationship: ConceptRelationship) -> bool:
        """Add a relationship between concepts"""
        try:
            # Validate concepts exist
            if (relationship.source_concept_id not in self.concepts or 
                relationship.target_concept_id not in self.concepts):
                logger.error(f"Invalid concept IDs in relationship {relationship.relationship_id}")
                return False
            
            # Store relationship
            self.relationships[relationship.relationship_id] = relationship
            
            # Add to graph
            self.graph.add_edge(
                relationship.source_concept_id,
                relationship.target_concept_id,
                relationship_id=relationship.relationship_id,
                type=relationship.relationship_type.value,
                strength=relationship.strength,
                confidence=relationship.confidence,
                bidirectional=relationship.bidirectional
            )
            
            # Add reverse edge if bidirectional
            if relationship.bidirectional:
                self.graph.add_edge(
                    relationship.target_concept_id,
                    relationship.source_concept_id,
                    relationship_id=relationship.relationship_id + "_reverse",
                    type=relationship.relationship_type.value,
                    strength=relationship.strength,
                    confidence=relationship.confidence,
                    bidirectional=True
                )
            
            logger.info(f"Added relationship: {relationship.relationship_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship {relationship.relationship_id}: {str(e)}")
            return False
    
    async def update_user_mastery(self, user_id: str, concept_id: str, 
                                mastery_data: Dict[str, Any]) -> bool:
        """Update user's mastery of a concept"""
        try:
            if concept_id not in self.concepts:
                logger.error(f"Concept {concept_id} not found")
                return False
            
            # Create or update mastery record
            if concept_id in self.user_mastery[user_id]:
                mastery = self.user_mastery[user_id][concept_id]
                mastery.updated_at = datetime.now()
            else:
                mastery = UserConceptMastery(
                    user_id=user_id,
                    concept_id=concept_id,
                    mastery_level=MasteryLevel.UNKNOWN,
                    confidence_score=0.0,
                    last_practiced=datetime.now(),
                    practice_count=0,
                    learning_path=[],
                    difficulty_encountered=0.0,
                    help_needed=False,
                    review_frequency=0.0,
                    transfer_applications=[]
                )
            
            # Update fields
            for key, value in mastery_data.items():
                if hasattr(mastery, key):
                    if key == 'mastery_level' and isinstance(value, int):
                        mastery.mastery_level = MasteryLevel(value)
                    else:
                        setattr(mastery, key, value)
            
            self.user_mastery[user_id][concept_id] = mastery
            
            # Update graph analytics
            await self._update_mastery_analytics(user_id, concept_id, mastery)
            
            logger.info(f"Updated mastery for user {user_id}, concept {concept_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user mastery: {str(e)}")
            return False
    
    async def generate_learning_path(self, user_id: str, goal_concepts: List[str], 
                                   context: ContextualState) -> Optional[LearningPath]:
        """Generate optimized learning path for user"""
        try:
            # Validate goal concepts
            valid_goals = [cid for cid in goal_concepts if cid in self.concepts]
            if not valid_goals:
                logger.error("No valid goal concepts provided")
                return None
            
            # Get user's current mastery state
            user_mastery_state = await self._get_user_mastery_state(user_id)
            
            # Generate multiple path candidates
            path_candidates = []
            
            # Method 1: Graph-based shortest path
            graph_path = await self._generate_graph_based_path(user_id, valid_goals, context)
            if graph_path:
                path_candidates.append(graph_path)
            
            # Method 2: Neural network optimization
            nn_path = await self._generate_neural_path(user_id, valid_goals, context)
            if nn_path:
                path_candidates.append(nn_path)
            
            # Method 3: Prerequisite-based path
            prereq_path = await self._generate_prerequisite_path(user_id, valid_goals, context)
            if prereq_path:
                path_candidates.append(prereq_path)
            
            # Select best path
            if not path_candidates:
                logger.error("No valid paths generated")
                return None
            
            best_path = await self._select_best_path(path_candidates, user_id, context)
            
            # Create learning path object
            learning_path = LearningPath(
                path_id=str(uuid.uuid4()),
                user_id=user_id,
                goal_concepts=valid_goals,
                path_concepts=best_path['concepts'],
                estimated_duration=best_path['duration'],
                difficulty_progression=best_path['difficulty_progression'],
                personalization_factors=best_path['personalization_factors'],
                success_probability=best_path['success_probability'],
                alternative_paths=[p['id'] for p in path_candidates if p != best_path],
                checkpoints=best_path['checkpoints']
            )
            
            logger.info(f"Generated learning path for user {user_id}: {len(best_path['concepts'])} concepts")
            return learning_path
            
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
            return None
    
    async def discover_semantic_relationships(self, concepts: List[str], 
                                            context: Optional[ContextType] = None) -> List[ConceptRelationship]:
        """Discover semantic relationships between concepts"""
        try:
            relationships = []
            
            for i, concept1_id in enumerate(concepts):
                for concept2_id in concepts[i+1:]:
                    if concept1_id == concept2_id:
                        continue
                    
                    concept1 = self.concepts.get(concept1_id)
                    concept2 = self.concepts.get(concept2_id)
                    
                    if not concept1 or not concept2:
                        continue
                    
                    # Calculate semantic similarity
                    similarity = await self._calculate_semantic_similarity(concept1, concept2)
                    
                    if similarity > 0.3:  # Threshold for relationship
                        # Determine relationship type
                        rel_type = await self._determine_relationship_type(concept1, concept2, similarity)
                        
                        # Create relationship
                        relationship = ConceptRelationship(
                            relationship_id=str(uuid.uuid4()),
                            source_concept_id=concept1_id,
                            target_concept_id=concept2_id,
                            relationship_type=rel_type,
                            strength=similarity,
                            confidence=min(0.9, similarity + 0.1),
                            bidirectional=rel_type in [RelationshipType.RELATED_TO, RelationshipType.SIMILAR_TO],
                            context=[context] if context else [ContextType.ACADEMIC],
                            evidence=[f"Semantic similarity: {similarity:.3f}"]
                        )
                        
                        relationships.append(relationship)
            
            logger.info(f"Discovered {len(relationships)} semantic relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Error discovering semantic relationships: {str(e)}")
            return []
    
    async def analyze_concept_mastery_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze patterns in user's concept mastery"""
        try:
            if user_id not in self.user_mastery:
                return {}
            
            user_concepts = self.user_mastery[user_id]
            
            # Mastery distribution
            mastery_levels = [mastery.mastery_level.value for mastery in user_concepts.values()]
            mastery_distribution = {
                level.name: mastery_levels.count(level.value) 
                for level in MasteryLevel
            }
            
            # Domain analysis
            domain_mastery = defaultdict(list)
            for concept_id, mastery in user_concepts.items():
                concept = self.concepts[concept_id]
                domain_mastery[concept.domain].append(mastery.mastery_level.value)
            
            domain_avg_mastery = {
                domain: np.mean(levels) for domain, levels in domain_mastery.items()
            }
            
            # Learning velocity
            practice_counts = [mastery.practice_count for mastery in user_concepts.values()]
            avg_practice_count = np.mean(practice_counts) if practice_counts else 0
            
            # Difficulty preferences
            difficulties = [mastery.difficulty_encountered for mastery in user_concepts.values()]
            avg_difficulty = np.mean(difficulties) if difficulties else 0
            
            # Knowledge gaps
            knowledge_gaps = await self._identify_knowledge_gaps(user_id)
            
            # Strength areas
            strengths = await self._identify_strengths(user_id)
            
            return {
                'mastery_distribution': mastery_distribution,
                'domain_mastery': domain_avg_mastery,
                'average_practice_count': avg_practice_count,
                'preferred_difficulty': avg_difficulty,
                'knowledge_gaps': knowledge_gaps,
                'strength_areas': strengths,
                'total_concepts': len(user_concepts),
                'mastered_concepts': len([m for m in user_concepts.values() 
                                        if m.mastery_level.value >= 4])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing mastery patterns: {str(e)}")
            return {}
    
    async def recommend_next_concepts(self, user_id: str, context: ContextualState, 
                                    limit: int = 5) -> List[Dict[str, Any]]:
        """Recommend next concepts for user to learn"""
        try:
            if user_id not in self.user_mastery:
                # For new users, recommend fundamental concepts
                return await self._recommend_fundamental_concepts(context, limit)
            
            user_concepts = self.user_mastery[user_id]
            
            # Get candidates based on prerequisites
            candidates = []
            
            for concept_id, concept in self.concepts.items():
                if concept_id in user_concepts:
                    continue  # Skip already learned concepts
                
                # Check if prerequisites are met
                prereq_met = await self._check_prerequisites_met(user_id, concept_id)
                if not prereq_met:
                    continue
                
                # Calculate recommendation score
                score = await self._calculate_recommendation_score(user_id, concept_id, context)
                
                candidates.append({
                    'concept_id': concept_id,
                    'concept': concept,
                    'score': score,
                    'reasoning': await self._generate_recommendation_reasoning(user_id, concept_id, score)
                })
            
            # Sort by score and return top recommendations
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"Error recommending next concepts: {str(e)}")
            return []
    
    async def detect_knowledge_transfer_opportunities(self, user_id: str) -> List[Dict[str, Any]]:
        """Detect opportunities for knowledge transfer across domains"""
        try:
            if user_id not in self.user_mastery:
                return []
            
            user_concepts = self.user_mastery[user_id]
            mastered_concepts = [
                concept_id for concept_id, mastery in user_concepts.items()
                if mastery.mastery_level.value >= 4  # Proficient or above
            ]
            
            transfer_opportunities = []
            
            for mastered_id in mastered_concepts:
                mastered_concept = self.concepts[mastered_id]
                
                # Find similar concepts in different domains
                for concept_id, concept in self.concepts.items():
                    if (concept_id not in user_concepts and 
                        concept.domain != mastered_concept.domain):
                        
                        # Calculate transfer potential
                        similarity = await self._calculate_semantic_similarity(mastered_concept, concept)
                        
                        if similarity > 0.5:  # High similarity threshold
                            transfer_score = await self._calculate_transfer_score(
                                mastered_concept, concept, user_concepts
                            )
                            
                            transfer_opportunities.append({
                                'source_concept': mastered_concept.name,
                                'source_domain': mastered_concept.domain,
                                'target_concept': concept.name,
                                'target_domain': concept.domain,
                                'similarity': similarity,
                                'transfer_score': transfer_score,
                                'transfer_strategy': await self._suggest_transfer_strategy(
                                    mastered_concept, concept
                                )
                            })
            
            # Sort by transfer score
            transfer_opportunities.sort(key=lambda x: x['transfer_score'], reverse=True)
            
            return transfer_opportunities[:10]  # Top 10 opportunities
            
        except Exception as e:
            logger.error(f"Error detecting transfer opportunities: {str(e)}")
            return []
    
    async def _generate_concept_embedding(self, concept: Concept) -> List[float]:
        """Generate embedding for a concept"""
        try:
            # Create description text
            description_text = f"{concept.name} {concept.description} {' '.join(concept.keywords)}"
            
            # Tokenize (simplified - would use proper tokenizer in production)
            tokens = description_text.lower().split()[:100]  # Limit to 100 tokens
            
            # Convert to tensor (simplified)
            token_ids = torch.LongTensor([hash(token) % 10000 for token in tokens]).unsqueeze(0)
            
            # Additional features
            additional_features = torch.FloatTensor([[
                concept.difficulty_level / 6.0,
                concept.complexity_score,
                concept.importance_score,
                float(concept.concept_type == ConceptType.FUNDAMENTAL),
                float(concept.concept_type == ConceptType.SKILL),
                len(concept.prerequisites) / 10.0,  # Normalized
                len(concept.learning_objectives) / 10.0,  # Normalized
                *[0.0] * 43  # Pad to 50 features
            ]])
            
            # Generate embedding using neural network
            with torch.no_grad():
                embedding = self.concept_embedding_network(token_ids, additional_features)
            
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Error generating concept embedding: {str(e)}")
            return [0.0] * 256  # Default embedding
    
    async def _calculate_semantic_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Calculate semantic similarity between two concepts"""
        try:
            # Use embeddings if available
            if concept1.embedding and concept2.embedding:
                emb1 = np.array(concept1.embedding)
                emb2 = np.array(concept2.embedding)
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity)
            
            # Fallback to text-based similarity
            text1 = f"{concept1.name} {concept1.description} {' '.join(concept1.keywords)}"
            text2 = f"{concept2.name} {concept2.description} {' '.join(concept2.keywords)}"
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    async def _determine_relationship_type(self, concept1: Concept, concept2: Concept, 
                                         similarity: float) -> RelationshipType:
        """Determine the type of relationship between concepts"""
        # Simple heuristic-based approach (would use ML in production)
        
        # Check for prerequisite relationship
        if concept1.concept_id in concept2.prerequisites:
            return RelationshipType.PREREQUISITE
        if concept2.concept_id in concept1.prerequisites:
            return RelationshipType.PREREQUISITE
        
        # Check difficulty levels
        diff1 = concept1.difficulty_level
        diff2 = concept2.difficulty_level
        
        if abs(diff1 - diff2) > 1.5:
            if diff1 < diff2:
                return RelationshipType.BUILDS_ON
            else:
                return RelationshipType.BUILDS_ON
        
        # Check domains
        if concept1.domain == concept2.domain:
            if similarity > 0.7:
                return RelationshipType.SIMILAR_TO
            else:
                return RelationshipType.RELATED_TO
        else:
            return RelationshipType.APPLIES_TO
    
    async def _generate_graph_based_path(self, user_id: str, goal_concepts: List[str], 
                                       context: ContextualState) -> Optional[Dict[str, Any]]:
        """Generate learning path using graph algorithms"""
        try:
            user_mastery = self.user_mastery.get(user_id, {})
            
            # Find concepts user has mastered
            mastered = [
                cid for cid, mastery in user_mastery.items()
                if mastery.mastery_level.value >= 4
            ]
            
            # Create a copy of graph for path finding
            path_graph = self.graph.copy()
            
            # Add virtual start node connected to mastered concepts
            start_node = "USER_START"
            path_graph.add_node(start_node)
            
            for mastered_concept in mastered:
                if mastered_concept in path_graph:
                    path_graph.add_edge(start_node, mastered_concept, weight=0.1)
            
            # Add virtual end node connected to goal concepts
            end_node = "USER_GOAL"
            path_graph.add_node(end_node)
            
            for goal_concept in goal_concepts:
                if goal_concept in path_graph:
                    path_graph.add_edge(goal_concept, end_node, weight=0.1)
            
            # Find shortest path
            try:
                path = nx.shortest_path(path_graph, start_node, end_node, weight='weight')
                
                # Remove virtual nodes
                actual_path = [node for node in path if node not in [start_node, end_node]]
                
                # Calculate path metrics
                duration = await self._estimate_path_duration(actual_path, user_id)
                difficulty_progression = [
                    self.concepts[cid].difficulty_level for cid in actual_path
                    if cid in self.concepts
                ]
                
                return {
                    'id': str(uuid.uuid4()),
                    'type': 'graph_based',
                    'concepts': actual_path,
                    'duration': duration,
                    'difficulty_progression': difficulty_progression,
                    'success_probability': 0.8,  # Simplified
                    'personalization_factors': {'method': 'graph_shortest_path'},
                    'checkpoints': actual_path[::3]  # Every 3rd concept
                }
                
            except nx.NetworkXNoPath:
                logger.warning("No path found using graph algorithm")
                return None
            
        except Exception as e:
            logger.error(f"Error in graph-based path generation: {str(e)}")
            return None
    
    async def _generate_neural_path(self, user_id: str, goal_concepts: List[str], 
                                  context: ContextualState) -> Optional[Dict[str, Any]]:
        """Generate learning path using neural network optimization"""
        try:
            # This is a simplified version - would be more sophisticated in production
            user_mastery = self.user_mastery.get(user_id, {})
            
            # Create user state vector
            mastery_scores = []
            for concept_id in self.concepts:
                if concept_id in user_mastery:
                    mastery_scores.append(user_mastery[concept_id].mastery_level.value / 6.0)
                else:
                    mastery_scores.append(0.0)
            
            # Pad or truncate to fixed size
            user_state = mastery_scores[:256] + [0.0] * max(0, 256 - len(mastery_scores))
            
            # Add context features
            context_features = [
                context.difficulty_preference / 6.0,
                context.available_time.total_seconds() / 3600.0,  # Hours
                float(context.collaboration_available),
                context.motivation_level,
                len(context.learning_objectives) / 10.0
            ]
            
            user_state.extend(context_features)
            user_state = user_state[:384]  # Fixed size
            
            # Create concept embeddings matrix
            concept_embeddings = []
            concept_ids = []
            
            for concept_id, concept in self.concepts.items():
                if concept.embedding:
                    concept_embeddings.append(concept.embedding)
                    concept_ids.append(concept_id)
            
            if not concept_embeddings:
                return None
            
            # Use simplified neural path generation (placeholder)
            # In production, this would use the actual neural network
            selected_concepts = concept_ids[:10]  # Simplified selection
            
            duration = await self._estimate_path_duration(selected_concepts, user_id)
            difficulty_progression = [
                self.concepts[cid].difficulty_level for cid in selected_concepts
            ]
            
            return {
                'id': str(uuid.uuid4()),
                'type': 'neural_optimized',
                'concepts': selected_concepts,
                'duration': duration,
                'difficulty_progression': difficulty_progression,
                'success_probability': 0.85,
                'personalization_factors': {'method': 'neural_network'},
                'checkpoints': selected_concepts[::2]  # Every 2nd concept
            }
            
        except Exception as e:
            logger.error(f"Error in neural path generation: {str(e)}")
            return None
    
    async def _generate_prerequisite_path(self, user_id: str, goal_concepts: List[str], 
                                        context: ContextualState) -> Optional[Dict[str, Any]]:
        """Generate learning path based on prerequisite dependencies"""
        try:
            user_mastery = self.user_mastery.get(user_id, {})
            
            # Collect all required concepts and their prerequisites
            required_concepts = set(goal_concepts)
            to_process = list(goal_concepts)
            
            while to_process:
                concept_id = to_process.pop(0)
                if concept_id not in self.concepts:
                    continue
                
                concept = self.concepts[concept_id]
                for prereq_id in concept.prerequisites:
                    if prereq_id not in required_concepts:
                        required_concepts.add(prereq_id)
                        to_process.append(prereq_id)
            
            # Remove concepts already mastered
            unmastered_concepts = [
                cid for cid in required_concepts
                if cid not in user_mastery or user_mastery[cid].mastery_level.value < 4
            ]
            
            # Sort by difficulty and dependencies
            sorted_concepts = await self._topological_sort_concepts(unmastered_concepts)
            
            if not sorted_concepts:
                return None
            
            duration = await self._estimate_path_duration(sorted_concepts, user_id)
            difficulty_progression = [
                self.concepts[cid].difficulty_level for cid in sorted_concepts
                if cid in self.concepts
            ]
            
            return {
                'id': str(uuid.uuid4()),
                'type': 'prerequisite_based',
                'concepts': sorted_concepts,
                'duration': duration,
                'difficulty_progression': difficulty_progression,
                'success_probability': 0.9,  # High success due to proper prerequisites
                'personalization_factors': {'method': 'prerequisite_dependency'},
                'checkpoints': sorted_concepts[::4]  # Every 4th concept
            }
            
        except Exception as e:
            logger.error(f"Error in prerequisite-based path generation: {str(e)}")
            return None
    
    async def _topological_sort_concepts(self, concept_ids: List[str]) -> List[str]:
        """Sort concepts based on prerequisite dependencies"""
        try:
            # Create subgraph with only the specified concepts
            subgraph = self.graph.subgraph(concept_ids).copy()
            
            # Filter edges to only include prerequisite relationships
            edges_to_remove = []
            for u, v, data in subgraph.edges(data=True):
                if data.get('type') != 'prerequisite':
                    edges_to_remove.append((u, v))
            
            subgraph.remove_edges_from(edges_to_remove)
            
            # Perform topological sort
            try:
                sorted_concepts = list(nx.topological_sort(subgraph))
                return sorted_concepts
            except nx.NetworkXError:
                # If there are cycles, fall back to difficulty-based sorting
                return sorted(concept_ids, 
                            key=lambda x: self.concepts[x].difficulty_level if x in self.concepts else 0)
            
        except Exception as e:
            logger.error(f"Error in topological sort: {str(e)}")
            return concept_ids
    
    async def _select_best_path(self, path_candidates: List[Dict[str, Any]], 
                              user_id: str, context: ContextualState) -> Dict[str, Any]:
        """Select the best path from candidates"""
        if not path_candidates:
            return None
        
        if len(path_candidates) == 1:
            return path_candidates[0]
        
        # Score each path based on multiple factors
        scored_paths = []
        
        for path in path_candidates:
            score = 0.0
            
            # Success probability
            score += path['success_probability'] * 0.4
            
            # Duration efficiency (prefer shorter paths if time is limited)
            if context.available_time:
                duration_hours = path['duration'].total_seconds() / 3600
                available_hours = context.available_time.total_seconds() / 3600
                
                if duration_hours <= available_hours:
                    score += 0.3
                else:
                    score += 0.3 * (available_hours / duration_hours)
            
            # Difficulty alignment
            avg_difficulty = np.mean(path['difficulty_progression'])
            difficulty_alignment = 1.0 - abs(avg_difficulty - context.difficulty_preference) / 6.0
            score += difficulty_alignment * 0.2
            
            # Path length (prefer moderate length)
            path_length = len(path['concepts'])
            if 5 <= path_length <= 15:
                score += 0.1
            elif path_length < 5:
                score += 0.05
            
            scored_paths.append((path, score))
        
        # Return path with highest score
        best_path = max(scored_paths, key=lambda x: x[1])[0]
        return best_path
    
    async def _estimate_path_duration(self, concept_ids: List[str], user_id: str) -> timedelta:
        """Estimate duration for learning a path"""
        total_hours = 0.0
        
        for concept_id in concept_ids:
            if concept_id not in self.concepts:
                continue
            
            concept = self.concepts[concept_id]
            
            # Base time based on difficulty and complexity
            base_hours = concept.difficulty_level * concept.complexity_score * 2.0
            
            # Adjust based on user's learning velocity (if available)
            user_mastery = self.user_mastery.get(user_id, {})
            if user_mastery:
                # Calculate average learning velocity
                practice_counts = [m.practice_count for m in user_mastery.values() if m.practice_count > 0]
                if practice_counts:
                    avg_practice = np.mean(practice_counts)
                    velocity_factor = min(2.0, max(0.5, 5.0 / avg_practice))  # Faster learners need less time
                    base_hours *= velocity_factor
            
            total_hours += base_hours
        
        return timedelta(hours=total_hours)
    
    async def _get_user_mastery_state(self, user_id: str) -> Dict[str, float]:
        """Get user's current mastery state as a dictionary"""
        if user_id not in self.user_mastery:
            return {}
        
        return {
            concept_id: mastery.mastery_level.value / 6.0
            for concept_id, mastery in self.user_mastery[user_id].items()
        }
    
    async def _update_concept_relationships(self, concept: Concept):
        """Update relationships for a newly added concept"""
        # This would discover and create relationships with existing concepts
        # Simplified implementation for now
        pass
    
    async def _update_mastery_analytics(self, user_id: str, concept_id: str, 
                                      mastery: UserConceptMastery):
        """Update analytics based on mastery changes"""
        # This would update various analytics and metrics
        # Simplified implementation for now
        pass
    
    async def _recommend_fundamental_concepts(self, context: ContextualState, 
                                            limit: int) -> List[Dict[str, Any]]:
        """Recommend fundamental concepts for new users"""
        fundamentals = [
            concept for concept in self.concepts.values()
            if concept.concept_type == ConceptType.FUNDAMENTAL
            and concept.domain in context.domain_focus
        ]
        
        # Sort by importance and difficulty
        fundamentals.sort(key=lambda x: (x.importance_score, -x.difficulty_level), reverse=True)
        
        return [{
            'concept_id': concept.concept_id,
            'concept': concept,
            'score': concept.importance_score,
            'reasoning': f"Fundamental concept in {concept.domain} with high importance"
        } for concept in fundamentals[:limit]]
    
    async def _check_prerequisites_met(self, user_id: str, concept_id: str) -> bool:
        """Check if user has met prerequisites for a concept"""
        if concept_id not in self.concepts:
            return False
        
        concept = self.concepts[concept_id]
        user_mastery = self.user_mastery.get(user_id, {})
        
        for prereq_id in concept.prerequisites:
            if (prereq_id not in user_mastery or 
                user_mastery[prereq_id].mastery_level.value < 3):  # Functional knowledge required
                return False
        
        return True
    
    async def _calculate_recommendation_score(self, user_id: str, concept_id: str, 
                                            context: ContextualState) -> float:
        """Calculate recommendation score for a concept"""
        concept = self.concepts[concept_id]
        score = 0.0
        
        # Importance score
        score += concept.importance_score * 0.3
        
        # Difficulty alignment
        difficulty_diff = abs(concept.difficulty_level - context.difficulty_preference)
        difficulty_score = max(0.0, 1.0 - difficulty_diff / 6.0)
        score += difficulty_score * 0.2
        
        # Domain relevance
        if concept.domain == context.domain_focus:
            score += 0.2
        
        # Learning objectives alignment
        objective_overlap = len(set(concept.learning_objectives) & set(context.learning_objectives))
        score += min(1.0, objective_overlap / max(1, len(context.learning_objectives))) * 0.2
        
        # Novelty bonus (prefer concepts not recently studied)
        user_mastery = self.user_mastery.get(user_id, {})
        if concept_id in user_mastery:
            days_since_practice = (datetime.now() - user_mastery[concept_id].last_practiced).days
            novelty_score = min(1.0, days_since_practice / 30.0)  # Normalized to 30 days
            score += novelty_score * 0.1
        else:
            score += 0.1  # New concept bonus
        
        return score
    
    async def _generate_recommendation_reasoning(self, user_id: str, concept_id: str, 
                                               score: float) -> str:
        """Generate explanation for why a concept is recommended"""
        concept = self.concepts[concept_id]
        
        reasons = []
        
        if concept.importance_score > 0.8:
            reasons.append("high importance in the domain")
        
        if concept.concept_type == ConceptType.FUNDAMENTAL:
            reasons.append("fundamental concept")
        
        user_mastery = self.user_mastery.get(user_id, {})
        prereq_count = len([p for p in concept.prerequisites if p in user_mastery])
        if prereq_count > 0:
            reasons.append(f"builds on {prereq_count} concepts you've already learned")
        
        if score > 0.8:
            return f"Highly recommended: {', '.join(reasons)}"
        elif score > 0.6:
            return f"Recommended: {', '.join(reasons)}"
        else:
            return f"Consider learning: {', '.join(reasons)}"
    
    async def _identify_knowledge_gaps(self, user_id: str) -> List[str]:
        """Identify knowledge gaps for a user"""
        if user_id not in self.user_mastery:
            return []
        
        user_mastery = self.user_mastery[user_id]
        gaps = []
        
        for concept_id, mastery in user_mastery.items():
            if mastery.mastery_level.value < 3:  # Below functional knowledge
                concept = self.concepts[concept_id]
                if concept.importance_score > 0.7:  # Important concept
                    gaps.append(concept.name)
        
        return gaps[:10]  # Top 10 gaps
    
    async def _identify_strengths(self, user_id: str) -> List[str]:
        """Identify strength areas for a user"""
        if user_id not in self.user_mastery:
            return []
        
        user_mastery = self.user_mastery[user_id]
        
        # Group by domain
        domain_mastery = defaultdict(list)
        for concept_id, mastery in user_mastery.items():
            concept = self.concepts[concept_id]
            domain_mastery[concept.domain].append(mastery.mastery_level.value)
        
        # Calculate average mastery per domain
        domain_averages = {
            domain: np.mean(levels) for domain, levels in domain_mastery.items()
        }
        
        # Return domains with high average mastery
        strengths = [
            domain for domain, avg in domain_averages.items()
            if avg >= 4.0  # Proficient or above
        ]
        
        return strengths
    
    async def _calculate_transfer_score(self, source_concept: Concept, target_concept: Concept, 
                                      user_mastery: Dict[str, UserConceptMastery]) -> float:
        """Calculate knowledge transfer score"""
        score = 0.0
        
        # Base similarity bonus
        similarity = await self._calculate_semantic_similarity(source_concept, target_concept)
        score += similarity * 0.4
        
        # Difficulty difference penalty
        difficulty_diff = abs(source_concept.difficulty_level - target_concept.difficulty_level)
        score += max(0.0, 1.0 - difficulty_diff / 6.0) * 0.3
        
        # User's mastery of source concept
        if source_concept.concept_id in user_mastery:
            source_mastery = user_mastery[source_concept.concept_id].mastery_level.value
            score += (source_mastery / 6.0) * 0.3
        
        return score
    
    async def _suggest_transfer_strategy(self, source_concept: Concept, 
                                       target_concept: Concept) -> str:
        """Suggest strategy for knowledge transfer"""
        similarity = await self._calculate_semantic_similarity(source_concept, target_concept)
        
        if similarity > 0.8:
            return "Direct application - the concepts are very similar"
        elif similarity > 0.6:
            return "Analogical transfer - identify key similarities and differences"
        elif similarity > 0.4:
            return "Bridging - find intermediate concepts to connect the domains"
        else:
            return "Abstract principle transfer - focus on underlying principles"

# ============================================================================
# SUPPORT CLASSES
# ============================================================================

class SemanticTextProcessor:
    """Processes text for semantic understanding"""
    
    def __init__(self):
        self.tokenizer = None  # Would initialize with actual tokenizer
        self.model = None      # Would initialize with actual model
    
    async def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text"""
        # Simplified implementation
        words = text.lower().split()
        # Would use NLP techniques to identify concepts
        return [word for word in words if len(word) > 3]
    
    async def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relationships from text"""
        # Simplified implementation
        # Would use NLP techniques to identify relationships
        return []

class ConceptExtractor:
    """Extracts concepts from various content types"""
    
    async def extract_from_text(self, text: str) -> List[Concept]:
        """Extract concepts from text content"""
        # Simplified implementation
        return []
    
    async def extract_from_video(self, video_transcript: str) -> List[Concept]:
        """Extract concepts from video transcript"""
        # Simplified implementation
        return []

class RelationshipDetector:
    """Detects relationships between concepts"""
    
    async def detect_prerequisites(self, concept1: Concept, concept2: Concept) -> bool:
        """Detect if concept1 is a prerequisite for concept2"""
        # Simplified implementation
        return concept1.difficulty_level < concept2.difficulty_level

class GraphMetrics:
    """Calculates various graph metrics"""
    
    def calculate_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate centrality measures"""
        return nx.betweenness_centrality(graph)
    
    def calculate_clustering(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate clustering coefficients"""
        return nx.clustering(graph)

class LearningAnalytics:
    """Provides learning analytics based on knowledge graph"""
    
    def analyze_learning_efficiency(self, user_mastery: Dict[str, UserConceptMastery]) -> Dict[str, Any]:
        """Analyze learning efficiency metrics"""
        return {}
    
    def predict_mastery_time(self, concept: Concept, user_state: Dict[str, Any]) -> timedelta:
        """Predict time to master a concept"""
        return timedelta(hours=concept.difficulty_level * 2)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example usage of the Advanced Knowledge Graph Engine"""
    
    # Initialize engine
    kg_engine = AdvancedKnowledgeGraphEngine()
    
    # Create some sample concepts
    concepts = [
        Concept(
            concept_id="algebra_basics",
            name="Algebra Basics",
            description="Fundamental algebraic operations and equations",
            concept_type=ConceptType.FUNDAMENTAL,
            domain="Mathematics",
            subdomain="Algebra",
            difficulty_level=2.0,
            complexity_score=0.6,
            importance_score=0.9,
            prerequisites=[],
            learning_objectives=["Solve linear equations", "Understand variables"],
            keywords=["algebra", "equations", "variables", "solving"]
        ),
        Concept(
            concept_id="quadratic_equations",
            name="Quadratic Equations",
            description="Solving and graphing quadratic equations",
            concept_type=ConceptType.INTERMEDIATE,
            domain="Mathematics",
            subdomain="Algebra",
            difficulty_level=4.0,
            complexity_score=0.8,
            importance_score=0.8,
            prerequisites=["algebra_basics"],
            learning_objectives=["Solve quadratic equations", "Graph parabolas"],
            keywords=["quadratic", "parabola", "factoring", "completing square"]
        )
    ]
    
    # Add concepts to graph
    for concept in concepts:
        await kg_engine.add_concept(concept)
    
    # Create relationship
    relationship = ConceptRelationship(
        relationship_id=str(uuid.uuid4()),
        source_concept_id="algebra_basics",
        target_concept_id="quadratic_equations",
        relationship_type=RelationshipType.PREREQUISITE,
        strength=0.9,
        confidence=0.95,
        bidirectional=False,
        context=[ContextType.ACADEMIC],
        evidence=["Quadratic equations require understanding of basic algebra"]
    )
    
    await kg_engine.add_relationship(relationship)
    
    # Update user mastery
    await kg_engine.update_user_mastery("user123", "algebra_basics", {
        'mastery_level': 5,
        'confidence_score': 0.8,
        'practice_count': 10
    })
    
    # Create context
    context = ContextualState(
        user_id="user123",
        session_id="session456",
        current_context=ContextType.ACADEMIC,
        domain_focus="Mathematics",
        learning_objectives=["Master algebra"],
        available_time=timedelta(hours=2),
        difficulty_preference=3.0,
        collaboration_available=False,
        tools_available=["calculator"],
        prior_knowledge={"algebra_basics": 0.8},
        emotional_state="confident",
        motivation_level=0.8
    )
    
    # Generate learning path
    learning_path = await kg_engine.generate_learning_path(
        "user123", ["quadratic_equations"], context
    )
    
    if learning_path:
        print(f"Generated learning path: {learning_path.path_concepts}")
        print(f"Estimated duration: {learning_path.estimated_duration}")
    
    # Get recommendations
    recommendations = await kg_engine.recommend_next_concepts("user123", context)
    print(f"Recommendations: {[r['concept'].name for r in recommendations]}")
    
    # Analyze mastery patterns
    patterns = await kg_engine.analyze_concept_mastery_patterns("user123")
    print(f"Mastery patterns: {patterns}")

if __name__ == "__main__":
    asyncio.run(example_usage())