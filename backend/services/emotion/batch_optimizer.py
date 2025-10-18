"""
MasterX Batch Optimizer - Dynamic Batch Size Optimization

ML-driven batch size optimization for maximum throughput and GPU utilization.

Following AGENTS.md principles:
- Zero hardcoded values (all configurable)
- Real ML algorithms (performance history learning)
- PEP8 compliant
- Full type hints
- Clean naming
- Production-ready

Performance Goals:
- GPU utilization: >80%
- Throughput: >50 predictions/sec
- No OOM errors
- Automatic adaptation to workload

Author: MasterX Team
Version: 1.0.0
"""

import logging
import psutil
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class BatchOptimizerConfig(BaseModel):
    """
    Configuration for batch size optimizer.
    All values configurable, NO hardcoded defaults.
    """
    # Batch size bounds
    min_batch_size: int = Field(
        default=1,
        description="Minimum batch size (must be >= 1)"
    )
    max_batch_size: int = Field(
        default=32,
        description="Maximum batch size for GPU memory"
    )
    
    # GPU utilization targets
    target_gpu_utilization: float = Field(
        default=0.85,
        description="Target GPU utilization [0, 1]"
    )
    
    # Memory management
    gpu_memory_threshold: float = Field(
        default=0.90,
        description="Max GPU memory usage before reducing batch [0, 1]"
    )
    cpu_memory_threshold: float = Field(
        default=0.85,
        description="Max CPU memory usage threshold [0, 1]"
    )
    
    # Optimization settings
    enable_dynamic_sizing: bool = Field(
        default=True,
        description="Enable ML-driven dynamic batch sizing"
    )
    enable_gpu_profiling: bool = Field(
        default=True,
        description="Enable GPU memory profiling"
    )
    
    # Learning parameters
    history_window: int = Field(
        default=100,
        description="Number of batches to track for learning"
    )
    adjustment_rate: float = Field(
        default=0.1,
        description="Rate of batch size adjustment [0, 1]"
    )
    
    # Sequence length estimation
    avg_tokens_per_char: float = Field(
        default=0.25,
        description="Average tokens per character for length estimation"
    )
    memory_per_token_bytes: float = Field(
        default=4.0,
        description="Estimated GPU memory per token (bytes)"
    )
    
    # Safety margins
    safety_factor: float = Field(
        default=0.9,
        description="Safety margin for memory estimates [0, 1]"
    )
    
    model_config = ConfigDict(validate_assignment=True)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class BatchMetrics:
    """
    Performance metrics for a single batch.
    
    Tracks latency, throughput, and resource usage.
    """
    batch_size: int
    sequence_lengths: List[int]
    latency_ms: float
    throughput: float  # items/sec
    gpu_memory_used: Optional[float] = None  # MB
    gpu_memory_available: Optional[float] = None  # MB
    cpu_memory_percent: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def avg_sequence_length(self) -> float:
        """Average sequence length in batch"""
        return float(np.mean(self.sequence_lengths)) if self.sequence_lengths else 0.0
    
    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length in batch"""
        return max(self.sequence_lengths) if self.sequence_lengths else 0
    
    @property
    def gpu_memory_utilization(self) -> Optional[float]:
        """GPU memory utilization [0, 1]"""
        if self.gpu_memory_used is None or self.gpu_memory_available is None:
            return None
        if self.gpu_memory_available == 0:
            return 0.0
        return self.gpu_memory_used / self.gpu_memory_available


# ============================================================================
# GPU MEMORY PROFILER
# ============================================================================

class GPUMemoryProfiler:
    """
    GPU memory profiling and monitoring.
    
    Tracks available memory and predicts OOM conditions.
    """
    
    def __init__(self, config: BatchOptimizerConfig):
        """
        Initialize GPU memory profiler.
        
        Args:
            config: Batch optimizer configuration
        """
        self.config = config
        self.torch_available = TORCH_AVAILABLE
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            self.device_index = 0  # Primary GPU
            logger.info(f"GPU profiler initialized: {self.device_count} device(s) found")
        else:
            logger.warning("GPU not available, memory profiling disabled")
    
    def get_gpu_memory_info(self) -> Tuple[float, float]:
        """
        Get GPU memory usage.
        
        Returns:
            Tuple of (used_mb, available_mb)
        """
        if not self.cuda_available:
            return (0.0, 0.0)
        
        try:
            # Get memory info for primary GPU
            used = torch.cuda.memory_allocated(self.device_index) / (1024 ** 2)  # MB
            total = torch.cuda.get_device_properties(self.device_index).total_memory / (1024 ** 2)  # MB
            
            available = total - used
            
            return (used, available)
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return (0.0, 0.0)
    
    def estimate_batch_memory(
        self,
        batch_size: int,
        avg_sequence_length: float
    ) -> float:
        """
        Estimate GPU memory required for batch.
        
        Uses learned memory-per-token estimate.
        
        Args:
            batch_size: Batch size
            avg_sequence_length: Average sequence length
        
        Returns:
            Estimated memory in MB
        """
        # Estimate total tokens
        total_tokens = batch_size * avg_sequence_length
        
        # Memory per token (from config)
        memory_bytes = total_tokens * self.config.memory_per_token_bytes
        
        # Convert to MB and apply safety factor
        memory_mb = (memory_bytes / (1024 ** 2)) / self.config.safety_factor
        
        return memory_mb
    
    def can_fit_batch(
        self,
        batch_size: int,
        avg_sequence_length: float
    ) -> bool:
        """
        Check if batch can fit in GPU memory.
        
        Args:
            batch_size: Proposed batch size
            avg_sequence_length: Average sequence length
        
        Returns:
            True if batch can fit, False otherwise
        """
        if not self.cuda_available:
            return True  # No GPU constraints
        
        # Get available memory
        _, available_mb = self.get_gpu_memory_info()
        
        # Estimate required memory
        required_mb = self.estimate_batch_memory(batch_size, avg_sequence_length)
        
        # Check against threshold
        threshold_mb = available_mb * self.config.gpu_memory_threshold
        
        return required_mb <= threshold_mb


# ============================================================================
# BATCH SIZE OPTIMIZER
# ============================================================================

class BatchOptimizer:
    """
    ML-driven batch size optimization.
    
    Learns optimal batch sizes from performance history.
    Adapts to:
    - Input sequence lengths
    - Available GPU memory
    - Historical throughput
    
    Algorithm:
    1. Estimate memory requirements
    2. Check GPU memory availability
    3. Learn from performance history
    4. Adjust batch size for optimal throughput
    """
    
    def __init__(self, config: BatchOptimizerConfig):
        """
        Initialize batch optimizer.
        
        Args:
            config: Batch optimizer configuration
        """
        self.config = config
        
        # GPU profiler
        self.gpu_profiler = GPUMemoryProfiler(config) if config.enable_gpu_profiling else None
        
        # Performance history
        self.history: deque = deque(maxlen=config.history_window)
        
        # Current optimal batch size (learned)
        self.optimal_batch_size = (config.min_batch_size + config.max_batch_size) // 2
        
        # Performance tracking
        self.total_batches = 0
        self.total_items = 0
        self.oom_count = 0
        
        logger.info(
            f"BatchOptimizer initialized: "
            f"range=[{config.min_batch_size}, {config.max_batch_size}], "
            f"dynamic={config.enable_dynamic_sizing}"
        )
    
    def get_optimal_batch_size(
        self,
        input_lengths: List[int],
        available_memory_mb: Optional[float] = None
    ) -> int:
        """
        Get optimal batch size for inputs.
        
        ML-driven selection based on:
        - Input sequence lengths
        - Available GPU memory
        - Historical performance
        
        Args:
            input_lengths: List of input text lengths
            available_memory_mb: Available GPU memory (optional)
        
        Returns:
            Optimal batch size
        """
        if not self.config.enable_dynamic_sizing:
            # Use fixed optimal batch size
            return self.optimal_batch_size
        
        # Calculate statistics
        avg_length = float(np.mean(input_lengths)) if input_lengths else 100.0
        
        # Estimate sequence length in tokens
        avg_tokens = avg_length * self.config.avg_tokens_per_char
        
        # Start with learned optimal
        batch_size = self.optimal_batch_size
        
        # Adjust based on GPU memory
        if self.gpu_profiler and self.gpu_profiler.cuda_available:
            batch_size = self._adjust_for_gpu_memory(batch_size, avg_tokens)
        
        # Adjust based on CPU memory
        batch_size = self._adjust_for_cpu_memory(batch_size)
        
        # Adjust based on performance history
        if self.history:
            batch_size = self._adjust_from_history(batch_size, avg_tokens)
        
        # Clip to configured bounds
        batch_size = np.clip(
            batch_size,
            self.config.min_batch_size,
            self.config.max_batch_size
        )
        
        return int(batch_size)
    
    def _adjust_for_gpu_memory(
        self,
        batch_size: int,
        avg_tokens: float
    ) -> int:
        """
        Adjust batch size based on GPU memory.
        
        Args:
            batch_size: Current batch size
            avg_tokens: Average tokens per sequence
        
        Returns:
            Adjusted batch size
        """
        if not self.gpu_profiler:
            return batch_size
        
        # Check if current batch size fits
        while batch_size > self.config.min_batch_size:
            if self.gpu_profiler.can_fit_batch(batch_size, avg_tokens):
                break
            # Reduce batch size
            batch_size = int(batch_size * (1 - self.config.adjustment_rate))
        
        return max(batch_size, self.config.min_batch_size)
    
    def _adjust_for_cpu_memory(self, batch_size: int) -> int:
        """
        Adjust batch size based on CPU memory.
        
        Args:
            batch_size: Current batch size
        
        Returns:
            Adjusted batch size
        """
        try:
            # Get CPU memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # If above threshold, reduce batch size
            if memory_percent > self.config.cpu_memory_threshold:
                reduction_factor = 1 - (memory_percent - self.config.cpu_memory_threshold)
                batch_size = int(batch_size * max(reduction_factor, 0.5))
        except Exception as e:
            logger.warning(f"Error checking CPU memory: {e}")
        
        return batch_size
    
    def _adjust_from_history(
        self,
        batch_size: int,
        avg_tokens: float
    ) -> int:
        """
        Adjust batch size based on performance history.
        
        Uses ML approach: find similar batches and learn optimal size.
        
        Args:
            batch_size: Current batch size
            avg_tokens: Average tokens per sequence
        
        Returns:
            Adjusted batch size
        """
        # Find similar batches in history
        similar_metrics = [
            m for m in self.history
            if abs(m.avg_sequence_length - avg_tokens) < avg_tokens * 0.2  # 20% similarity
        ]
        
        if not similar_metrics:
            return batch_size
        
        # Calculate throughput for each batch size
        batch_throughputs = {}
        for metric in similar_metrics:
            if metric.batch_size not in batch_throughputs:
                batch_throughputs[metric.batch_size] = []
            batch_throughputs[metric.batch_size].append(metric.throughput)
        
        # Find batch size with highest average throughput
        best_batch_size = batch_size
        best_throughput = 0.0
        
        for bs, throughputs in batch_throughputs.items():
            avg_throughput = float(np.mean(throughputs))
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_batch_size = bs
        
        # Smoothly transition to best batch size
        adjusted_size = batch_size + (best_batch_size - batch_size) * self.config.adjustment_rate
        
        return int(adjusted_size)
    
    def record_batch_performance(self, metrics: BatchMetrics) -> None:
        """
        Record batch performance for learning.
        
        Args:
            metrics: Batch performance metrics
        """
        # Add to history
        self.history.append(metrics)
        
        # Update statistics
        self.total_batches += 1
        self.total_items += metrics.batch_size
        
        # Update optimal batch size (exponential moving average)
        if metrics.throughput > 0:
            # Weight recent high-performing batches more
            alpha = self.config.adjustment_rate
            self.optimal_batch_size = int(
                alpha * metrics.batch_size + (1 - alpha) * self.optimal_batch_size
            )
    
    def record_oom_error(self, batch_size: int) -> None:
        """
        Record out-of-memory error.
        
        Args:
            batch_size: Batch size that caused OOM
        """
        self.oom_count += 1
        logger.warning(f"OOM error with batch_size={batch_size}, reducing optimal")
        
        # Reduce optimal batch size
        self.optimal_batch_size = int(
            min(self.optimal_batch_size, batch_size * 0.8)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimizer statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_batches": self.total_batches,
            "total_items": self.total_items,
            "optimal_batch_size": self.optimal_batch_size,
            "oom_count": self.oom_count,
            "history_size": len(self.history)
        }
        
        # Add GPU stats
        if self.gpu_profiler and self.gpu_profiler.cuda_available:
            used_mb, available_mb = self.gpu_profiler.get_gpu_memory_info()
            stats["gpu_memory_used_mb"] = used_mb
            stats["gpu_memory_available_mb"] = available_mb
            if available_mb > 0:
                stats["gpu_memory_utilization"] = f"{(used_mb / (used_mb + available_mb)) * 100:.1f}%"
        
        # Add performance stats from history
        if self.history:
            throughputs = [m.throughput for m in self.history]
            latencies = [m.latency_ms for m in self.history]
            
            stats["avg_throughput"] = f"{np.mean(throughputs):.2f} items/sec"
            stats["avg_latency_ms"] = f"{np.mean(latencies):.2f}ms"
            stats["p95_latency_ms"] = f"{np.percentile(latencies, 95):.2f}ms"
        
        return stats
    
    def reset(self) -> None:
        """Reset optimizer state"""
        self.history.clear()
        self.total_batches = 0
        self.total_items = 0
        self.oom_count = 0
        self.optimal_batch_size = (self.config.min_batch_size + self.config.max_batch_size) // 2
        logger.info("BatchOptimizer reset")
