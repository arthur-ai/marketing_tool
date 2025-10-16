"""
Performance monitoring middleware.

This middleware tracks request performance metrics and integrates
with the performance monitoring system.
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from marketing_project.performance.monitoring import performance_monitor

logger = logging.getLogger("marketing_project.middleware.performance")


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring request performance."""
    
    def __init__(self, app, monitor_health_endpoints: bool = False):
        super().__init__(app)
        self.monitor_health_endpoints = monitor_health_endpoints
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and monitor performance."""
        # Skip monitoring for certain endpoints
        if not self._should_monitor(request):
            return await call_next(request)
        
        # Extract request information
        endpoint = request.url.path
        method = request.method
        user_id = getattr(request.state, 'user', {}).get('role') if hasattr(request.state, 'user') else None
        ip_address = request.client.host if request.client else None
        
        # Start monitoring
        request_id = await performance_monitor.start_request(
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            ip_address=ip_address
        )
        
        # Store request ID for later use
        request.state.performance_request_id = request_id
        
        # Get request size
        request_size = 0
        if hasattr(request, '_body'):
            request_size = len(request._body)
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response size
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body)
            
            # End monitoring
            await performance_monitor.end_request(
                request_id=request_id,
                status_code=response.status_code,
                request_size=request_size,
                response_size=response_size
            )
            
            return response
            
        except Exception as e:
            # End monitoring with error
            await performance_monitor.end_request(
                request_id=request_id,
                status_code=500,
                request_size=request_size,
                response_size=0
            )
            
            # Re-raise the exception
            raise e
    
    def _should_monitor(self, request: Request) -> bool:
        """Determine if request should be monitored."""
        # Skip health endpoints unless explicitly enabled
        if not self.monitor_health_endpoints:
            if request.url.path in ['/health', '/ready', '/api/v1/health', '/api/v1/ready']:
                return False
        
        # Skip static files
        if request.url.path.startswith('/static/'):
            return False
        
        # Skip docs
        if request.url.path.startswith('/docs') or request.url.path.startswith('/redoc'):
            return False
        
        return True
