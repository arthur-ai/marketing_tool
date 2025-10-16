"""
Performance monitoring and metrics collection.

This module provides comprehensive performance monitoring capabilities
for the Marketing Project API.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

logger = logging.getLogger("marketing_project.performance.monitoring")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    endpoint: str
    method: str
    response_time: float
    status_code: int
    memory_usage: float
    cpu_usage: float
    request_size: int
    response_size: int
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "method": self.method,
            "response_time": self.response_time,
            "status_code": self.status_code,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "request_size": self.request_size,
            "response_size": self.response_size,
            "user_id": self.user_id,
            "ip_address": self.ip_address
        }


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0,
            "status_codes": defaultdict(int)
        })
        self.lock = asyncio.Lock()
    
    async def add_metric(self, metric: PerformanceMetrics):
        """Add a performance metric."""
        async with self.lock:
            self.metrics.append(metric)
            
            # Update endpoint statistics
            endpoint_key = f"{metric.method} {metric.endpoint}"
            stats = self.endpoint_stats[endpoint_key]
            
            stats["count"] += 1
            stats["total_time"] += metric.response_time
            stats["min_time"] = min(stats["min_time"], metric.response_time)
            stats["max_time"] = max(stats["max_time"], metric.response_time)
            stats["status_codes"][metric.status_code] += 1
            
            if metric.status_code >= 400:
                stats["error_count"] += 1
    
    async def get_metrics(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         endpoint: Optional[str] = None) -> List[PerformanceMetrics]:
        """Get metrics within time range and optional endpoint filter."""
        async with self.lock:
            filtered_metrics = []
            
            for metric in self.metrics:
                # Time filter
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                
                # Endpoint filter
                if endpoint and endpoint not in metric.endpoint:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    async def get_endpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all endpoints."""
        async with self.lock:
            stats = {}
            for endpoint, data in self.endpoint_stats.items():
                if data["count"] > 0:
                    stats[endpoint] = {
                        "count": data["count"],
                        "avg_response_time": data["total_time"] / data["count"],
                        "min_response_time": data["min_time"] if data["min_time"] != float('inf') else 0,
                        "max_response_time": data["max_time"],
                        "error_rate": data["error_count"] / data["count"],
                        "status_codes": dict(data["status_codes"])
                    }
            return stats
    
    async def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return await self.get_metrics(start_time=cutoff_time)
    
    async def clear_old_metrics(self, hours: int = 24):
        """Clear metrics older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        async with self.lock:
            # Keep only recent metrics
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            self.metrics.clear()
            self.metrics.extend(recent_metrics)


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.active_requests = 0
        self.max_active_requests = 0
        
    async def start_request(self, endpoint: str, method: str, 
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None) -> str:
        """Start monitoring a request."""
        request_id = f"{int(time.time() * 1000)}_{self.request_count}"
        self.request_count += 1
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        
        # Store request context
        if not hasattr(self, '_request_contexts'):
            self._request_contexts = {}
        
        self._request_contexts[request_id] = {
            "endpoint": endpoint,
            "method": method,
            "start_time": time.time(),
            "user_id": user_id,
            "ip_address": ip_address
        }
        
        return request_id
    
    async def end_request(self, request_id: str, status_code: int,
                         request_size: int = 0, response_size: int = 0):
        """End monitoring a request."""
        if not hasattr(self, '_request_contexts') or request_id not in self._request_contexts:
            return
        
        context = self._request_contexts[request_id]
        end_time = time.time()
        response_time = end_time - context["start_time"]
        
        # Get system metrics
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        
        # Create metric
        metric = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            endpoint=context["endpoint"],
            method=context["method"],
            response_time=response_time,
            status_code=status_code,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            request_size=request_size,
            response_size=response_size,
            user_id=context["user_id"],
            ip_address=context["ip_address"]
        )
        
        # Add to collector
        await self.metrics_collector.add_metric(metric)
        
        # Update counters
        self.active_requests -= 1
        if status_code >= 400:
            self.error_count += 1
        
        # Clean up context
        del self._request_contexts[request_id]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        uptime = time.time() - self.start_time
        recent_metrics = await self.metrics_collector.get_recent_metrics(5)
        
        if recent_metrics:
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            error_rate = sum(1 for m in recent_metrics if m.status_code >= 400) / len(recent_metrics)
        else:
            avg_response_time = 0.0
            error_rate = 0.0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "active_requests": self.active_requests,
            "max_active_requests": self.max_active_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "recent_avg_response_time": avg_response_time,
            "recent_error_rate": error_rate,
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_usage_percent": self._get_cpu_usage()
        }
    
    async def get_endpoint_performance(self) -> Dict[str, Any]:
        """Get performance metrics by endpoint."""
        return await self.metrics_collector.get_endpoint_stats()
    
    async def get_slow_requests(self, threshold: float = 1.0) -> List[PerformanceMetrics]:
        """Get requests slower than threshold seconds."""
        recent_metrics = await self.metrics_collector.get_recent_metrics(60)
        return [m for m in recent_metrics if m.response_time > threshold]
    
    async def get_error_requests(self) -> List[PerformanceMetrics]:
        """Get all error requests."""
        recent_metrics = await self.metrics_collector.get_recent_metrics(60)
        return [m for m in recent_metrics if m.status_code >= 400]
    
    @asynccontextmanager
    async def monitor_request(self, endpoint: str, method: str,
                            user_id: Optional[str] = None,
                            ip_address: Optional[str] = None):
        """Context manager for monitoring requests."""
        request_id = await self.start_request(endpoint, method, user_id, ip_address)
        try:
            yield request_id
        finally:
            # This will be called by the middleware
            pass


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
