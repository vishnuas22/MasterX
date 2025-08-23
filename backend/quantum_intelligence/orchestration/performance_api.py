"""
🚀 PERFORMANCE MONITORING API INTEGRATION
Enterprise-grade performance API for MasterX Quantum Intelligence

BREAKTHROUGH FEATURES:
- Real-time performance dashboard API endpoints
- Sub-100ms performance metrics streaming
- Advanced performance analytics with quantum intelligence
- Enterprise-scale monitoring dashboard
- Automated performance optimization triggers
- Performance alert management system
- Real-time metrics WebSocket streaming
- Performance report generation

API ENDPOINTS:
- GET /api/performance/dashboard - Real-time performance dashboard
- GET /api/performance/metrics - Current performance metrics
- GET /api/performance/alerts - Performance alerts management
- GET /api/performance/optimizations - Optimization history
- POST /api/performance/optimize - Trigger manual optimization
- WebSocket /api/performance/stream - Real-time metrics streaming

Author: MasterX Quantum Intelligence Team
Version: 4.0 - Performance API Integration
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from .advanced_performance_monitor import (
    AdvancedPerformanceMonitor,
    PerformanceLevel,
    OptimizationStrategy,
    AlertSeverity,
    QuantumPerformanceMetrics,
    get_performance_monitor,
    get_performance_dashboard
)

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class PerformanceMetricsResponse(BaseModel):
    """Performance metrics API response model"""
    timestamp: float
    system_status: str
    performance_level: str
    key_metrics: Dict[str, float]
    quantum_intelligence: Dict[str, float]
    performance_targets: Dict[str, float]
    response_time_ms: float = Field(..., description="Average response time in milliseconds")
    cache_hit_ratio: float = Field(..., description="Cache hit ratio (0.0-1.0)")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")

class PerformanceAlertResponse(BaseModel):
    """Performance alert API response model"""
    alert_id: str
    severity: str
    metric_name: str
    current_value: float
    threshold: float
    message: str
    recommendations: List[str]
    created_at: str
    auto_remediation: Optional[str] = None

class OptimizationRequest(BaseModel):
    """Optimization request model"""
    strategy: str = Field(..., description="Optimization strategy to apply")
    force: bool = Field(default=False, description="Force optimization even if not needed")
    timeout: int = Field(default=30, description="Optimization timeout in seconds")

class OptimizationResponse(BaseModel):
    """Optimization response model"""
    success: bool
    strategy: str
    improvement_percent: float
    before_metric: float
    after_metric: float
    message: str
    applied_at: str
    error_message: Optional[str] = None

class PerformanceReportRequest(BaseModel):
    """Performance report request model"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    include_quantum_metrics: bool = True
    include_optimizations: bool = True
    include_alerts: bool = True
    format: str = Field(default="json", description="Report format: json, csv, html")

# ============================================================================
# PERFORMANCE API ROUTER
# ============================================================================

router = APIRouter(prefix="/api/performance", tags=["Performance Monitoring"])

# WebSocket connection manager for real-time streaming
class PerformanceWebSocketManager:
    """Manages WebSocket connections for real-time performance streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.streaming_task: Optional[asyncio.Task] = None
        self.streaming_active = False
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Start streaming if not already active
        if not self.streaming_active:
            self.streaming_task = asyncio.create_task(self._stream_performance_data())
            self.streaming_active = True
        
        logger.info(f"✅ WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Stop streaming if no active connections
        if not self.active_connections and self.streaming_active:
            if self.streaming_task:
                self.streaming_task.cancel()
            self.streaming_active = False
        
        logger.info(f"📡 WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast_performance_data(self, data: Dict[str, Any]):
        """Broadcast performance data to all connected clients"""
        if not self.active_connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"⚠️ Failed to send data to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def _stream_performance_data(self):
        """Background task to stream performance data"""
        try:
            while self.streaming_active and self.active_connections:
                # Get current performance data
                dashboard_data = get_performance_dashboard()
                
                # Add streaming metadata
                streaming_data = {
                    "type": "performance_update",
                    "timestamp": time.time(),
                    "data": dashboard_data
                }
                
                # Broadcast to all connected clients
                await self.broadcast_performance_data(streaming_data)
                
                # Wait for next update (1 second for real-time)
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("📡 Performance streaming task cancelled")
        except Exception as e:
            logger.error(f"❌ Error in performance streaming: {e}")
        finally:
            self.streaming_active = False

# Global WebSocket manager
websocket_manager = PerformanceWebSocketManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_performance_dashboard_api():
    """
    Get comprehensive real-time performance dashboard
    
    Returns detailed performance metrics including:
    - System performance level and status
    - Key performance metrics (response time, cache hit ratio, etc.)
    - Quantum intelligence performance metrics
    - Performance targets and thresholds
    - Recent alerts and optimizations
    - Performance trending data
    """
    try:
        dashboard_data = get_performance_dashboard()
        
        if 'error' in dashboard_data:
            raise HTTPException(status_code=500, detail=dashboard_data['error'])
        
        # Add API-specific metadata
        dashboard_data['api_metadata'] = {
            'endpoint': '/api/performance/dashboard',
            'version': '4.0',
            'generated_at': datetime.utcnow().isoformat(),
            'data_freshness': 'real-time'
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Error getting performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance dashboard: {str(e)}")

@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_current_performance_metrics():
    """
    Get current performance metrics in structured format
    
    Returns key performance indicators optimized for monitoring dashboards
    and automated systems integration.
    """
    try:
        monitor = await get_performance_monitor()
        dashboard_data = monitor.get_performance_dashboard()
        
        if 'error' in dashboard_data:
            raise HTTPException(status_code=500, detail=dashboard_data['error'])
        
        # Extract key metrics for structured response
        key_metrics = dashboard_data.get('key_metrics', {})
        quantum_metrics = dashboard_data.get('quantum_intelligence', {})
        targets = dashboard_data.get('performance_targets', {})
        
        metrics_response = PerformanceMetricsResponse(
            timestamp=dashboard_data.get('timestamp', time.time()),
            system_status=dashboard_data.get('system_status', 'unknown'),
            performance_level=dashboard_data.get('performance_level', 'unknown'),
            key_metrics=key_metrics,
            quantum_intelligence=quantum_metrics,
            performance_targets=targets,
            response_time_ms=key_metrics.get('avg_response_time_ms', 0),
            cache_hit_ratio=key_metrics.get('cache_hit_ratio', 0),
            memory_usage_percent=key_metrics.get('memory_usage_percent', 0),
            cpu_usage_percent=key_metrics.get('cpu_usage_percent', 0)
        )
        
        return metrics_response
        
    except Exception as e:
        logger.error(f"❌ Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.get("/alerts", response_model=List[PerformanceAlertResponse])
async def get_performance_alerts(
    severity: Optional[str] = None,
    limit: int = 50,
    active_only: bool = True
):
    """
    Get performance alerts with optional filtering
    
    Args:
        severity: Filter by alert severity (info, warning, critical, emergency)
        limit: Maximum number of alerts to return
        active_only: Return only active/recent alerts
    
    Returns list of performance alerts with recommendations
    """
    try:
        monitor = await get_performance_monitor()
        
        # Get alerts from monitor
        alerts = monitor.alerts
        
        # Filter by active_only (last 24 hours)
        if active_only:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            alerts = [alert for alert in alerts if alert.created_at > cutoff_time]
        
        # Filter by severity
        if severity:
            severity_enum = AlertSeverity(severity.lower())
            alerts = [alert for alert in alerts if alert.severity == severity_enum]
        
        # Apply limit
        alerts = alerts[-limit:] if limit > 0 else alerts
        
        # Convert to response models
        alert_responses = [
            PerformanceAlertResponse(
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                metric_name=alert.metric_name,
                current_value=alert.current_value,
                threshold=alert.threshold,
                message=alert.message,
                recommendations=alert.recommendations,
                created_at=alert.created_at.isoformat(),
                auto_remediation=alert.auto_remediation
            )
            for alert in alerts
        ]
        
        return alert_responses
        
    except Exception as e:
        logger.error(f"❌ Error getting performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance alerts: {str(e)}")

@router.get("/optimizations", response_model=List[OptimizationResponse])
async def get_optimization_history(limit: int = 20):
    """
    Get performance optimization history
    
    Args:
        limit: Maximum number of optimization results to return
    
    Returns list of recent performance optimizations with effectiveness metrics
    """
    try:
        monitor = await get_performance_monitor()
        
        # Get optimization results
        optimizations = monitor.optimization_results[-limit:] if limit > 0 else monitor.optimization_results
        
        # Convert to response models
        optimization_responses = [
            OptimizationResponse(
                success=opt.success,
                strategy=opt.strategy.value,
                improvement_percent=opt.improvement_percent,
                before_metric=opt.before_metric,
                after_metric=opt.after_metric,
                message=f"Applied {opt.strategy.value} with {opt.improvement_percent:.1f}% improvement" if opt.success else f"Failed to apply {opt.strategy.value}",
                applied_at=opt.applied_at.isoformat(),
                error_message=opt.error_message
            )
            for opt in optimizations
        ]
        
        return optimization_responses
        
    except Exception as e:
        logger.error(f"❌ Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization history: {str(e)}")

@router.post("/optimize", response_model=OptimizationResponse)
async def trigger_performance_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger manual performance optimization
    
    Args:
        request: Optimization request with strategy and parameters
    
    Returns optimization result with effectiveness metrics
    """
    try:
        monitor = await get_performance_monitor()
        
        # Validate optimization strategy
        try:
            strategy = OptimizationStrategy(request.strategy)
        except ValueError:
            valid_strategies = [s.value for s in OptimizationStrategy]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid optimization strategy. Valid options: {valid_strategies}"
            )
        
        # Apply optimization with timeout
        try:
            optimization_result = await asyncio.wait_for(
                monitor._apply_optimization(strategy),
                timeout=request.timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Optimization timed out after {request.timeout} seconds"
            )
        
        # Convert to response model
        response = OptimizationResponse(
            success=optimization_result.success,
            strategy=optimization_result.strategy.value,
            improvement_percent=optimization_result.improvement_percent,
            before_metric=optimization_result.before_metric,
            after_metric=optimization_result.after_metric,
            message=f"Applied {optimization_result.strategy.value} with {optimization_result.improvement_percent:.1f}% improvement" if optimization_result.success else f"Failed to apply {optimization_result.strategy.value}",
            applied_at=optimization_result.applied_at.isoformat(),
            error_message=optimization_result.error_message
        )
        
        logger.info(f"✅ Manual optimization triggered: {strategy.value} - Success: {optimization_result.success}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error triggering optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger optimization: {str(e)}")

@router.get("/health")
async def get_performance_system_health():
    """
    Get performance monitoring system health status
    
    Returns health status of the performance monitoring system itself
    """
    try:
        monitor = await get_performance_monitor()
        
        health_status = {
            "status": "healthy",
            "monitoring_active": monitor.monitoring_active,
            "optimization_enabled": monitor.optimization_enabled,
            "active_operations": len(monitor.active_operations),
            "metrics_buffer_size": len(monitor.metrics_buffer),
            "alerts_count": len(monitor.alerts),
            "optimizations_count": len(monitor.optimization_results),
            "cache_stats": monitor.cache_stats,
            "system_info": {
                "python_version": "3.11+",
                "monitoring_version": "4.0",
                "uptime_seconds": time.time() - (monitor.system_metrics.get('startup_time', time.time()))
            },
            "last_update": datetime.utcnow().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Error getting performance system health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_update": datetime.utcnow().isoformat()
        }

@router.post("/report")
async def generate_performance_report(request: PerformanceReportRequest):
    """
    Generate comprehensive performance report
    
    Args:
        request: Report generation parameters
    
    Returns comprehensive performance analysis report
    """
    try:
        monitor = await get_performance_monitor()
        
        # Set default time range if not provided
        end_time = request.end_time or datetime.utcnow()
        start_time = request.start_time or (end_time - timedelta(hours=24))
        
        # Generate comprehensive report
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "report_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600
                },
                "report_version": "4.0",
                "includes": {
                    "quantum_metrics": request.include_quantum_metrics,
                    "optimizations": request.include_optimizations,
                    "alerts": request.include_alerts
                }
            },
            
            "executive_summary": {
                "overall_performance": monitor.get_performance_dashboard().get('performance_level', 'unknown'),
                "system_status": monitor.get_performance_dashboard().get('system_status', 'unknown'),
                "key_achievements": [
                    f"Average response time: {monitor._calculate_avg_response_time() * 1000:.1f}ms",
                    f"Cache hit ratio: {monitor._calculate_cache_hit_ratio() * 100:.1f}%",
                    f"Active optimizations: {len(monitor.optimization_results)}",
                    f"Performance alerts resolved: {len([a for a in monitor.alerts if a.auto_remediation])}"
                ]
            },
            
            "performance_metrics": monitor.get_performance_dashboard(),
            
            "trend_analysis": {
                "response_time_trend": monitor._calculate_trend(list(monitor.response_times)[-50:]) if len(monitor.response_times) >= 50 else 0.0,
                "performance_stability": "stable",  # Would be calculated based on variance
                "peak_performance_periods": [],  # Would identify best performing periods
                "bottleneck_patterns": []  # Would identify recurring bottlenecks
            }
        }
        
        # Add quantum metrics if requested
        if request.include_quantum_metrics:
            report["quantum_intelligence_analysis"] = {
                "active_quantum_users": len(monitor.quantum_metrics),
                "avg_quantum_processing_time": monitor._get_avg_quantum_processing_time(),
                "quantum_coherence_trend": monitor._get_avg_quantum_coherence(),
                "optimization_effectiveness": monitor._get_avg_optimization_effectiveness()
            }
        
        # Add optimization history if requested
        if request.include_optimizations:
            report["optimization_analysis"] = {
                "total_optimizations": len(monitor.optimization_results),
                "successful_optimizations": len([opt for opt in monitor.optimization_results if opt.success]),
                "avg_improvement": sum(opt.improvement_percent for opt in monitor.optimization_results if opt.success) / max(len([opt for opt in monitor.optimization_results if opt.success]), 1),
                "optimization_history": [
                    {
                        "strategy": opt.strategy.value,
                        "improvement_percent": opt.improvement_percent,
                        "success": opt.success,
                        "applied_at": opt.applied_at.isoformat()
                    }
                    for opt in monitor.optimization_results[-20:]  # Last 20 optimizations
                ]
            }
        
        # Add alerts analysis if requested
        if request.include_alerts:
            report["alerts_analysis"] = {
                "total_alerts": len(monitor.alerts),
                "critical_alerts": len([a for a in monitor.alerts if a.severity == AlertSeverity.CRITICAL]),
                "auto_resolved_alerts": len([a for a in monitor.alerts if a.auto_remediation]),
                "recent_alerts": [
                    {
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "created_at": alert.created_at.isoformat(),
                        "auto_resolved": bool(alert.auto_remediation)
                    }
                    for alert in monitor.alerts[-10:]  # Last 10 alerts
                ]
            }
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@router.websocket("/stream")
async def websocket_performance_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time performance metrics streaming
    
    Provides real-time performance data updates for dashboard integration
    """
    try:
        await websocket_manager.connect(websocket)
        
        # Send initial performance data
        initial_data = {
            "type": "connection_established",
            "timestamp": time.time(),
            "message": "Performance monitoring stream connected",
            "data": get_performance_dashboard()
        }
        await websocket.send_text(json.dumps(initial_data))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/configuration)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client messages
                try:
                    client_data = json.loads(message)
                    
                    if client_data.get("type") == "ping":
                        pong_response = {
                            "type": "pong",
                            "timestamp": time.time()
                        }
                        await websocket.send_text(json.dumps(pong_response))
                    
                    elif client_data.get("type") == "get_current_metrics":
                        current_metrics = {
                            "type": "current_metrics",
                            "timestamp": time.time(),
                            "data": get_performance_dashboard()
                        }
                        await websocket.send_text(json.dumps(current_metrics))
                
                except json.JSONDecodeError:
                    # Invalid JSON, send error response
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON in client message",
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(error_response))
            
            except asyncio.TimeoutError:
                # Send heartbeat if no client messages
                heartbeat = {
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "connections": len(websocket_manager.active_connections)
                }
                await websocket.send_text(json.dumps(heartbeat))
    
    except WebSocketDisconnect:
        logger.info("📡 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

# ============================================================================
# STARTUP/SHUTDOWN HANDLERS
# ============================================================================

async def initialize_performance_api():
    """Initialize performance monitoring API"""
    try:
        # Start performance monitor
        monitor = await get_performance_monitor()
        logger.info("🚀 Performance Monitoring API initialized successfully")
        
        return monitor
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Performance API: {e}")
        raise

async def shutdown_performance_api():
    """Shutdown performance monitoring API"""
    try:
        global performance_monitor
        if performance_monitor:
            await performance_monitor.stop_monitoring()
        
        # Cancel streaming tasks
        if websocket_manager.streaming_task:
            websocket_manager.streaming_task.cancel()
        
        logger.info("🛑 Performance Monitoring API shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Error during Performance API shutdown: {e}")

# Export router and utilities
__all__ = [
    'router',
    'initialize_performance_api',
    'shutdown_performance_api',
    'websocket_manager',
    'PerformanceMetricsResponse',
    'PerformanceAlertResponse',
    'OptimizationRequest',
    'OptimizationResponse'
]