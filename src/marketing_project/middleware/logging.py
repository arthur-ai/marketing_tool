"""
Logging middleware for FastAPI.

This module provides request logging and request ID tracking.
"""

import logging
import time
import uuid
from typing import Dict, Any
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("marketing_project.middleware.logging")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for FastAPI."""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next):
        """Process request through logging middleware."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, process_time, request_id)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": process_time,
                    "error": str(e)
                }
            )
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get user info if available
        user_info = getattr(request.state, 'user', {})
        user_role = user_info.get('role', 'anonymous')
        
        # Prepare log data
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", ""),
            "user_role": user_role,
            "timestamp": datetime.utcnow().isoformat(),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", "0")
        }
        
        # Log request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path}",
            extra=log_data
        )
    
    async def _log_response(self, request: Request, response: Response, 
                          process_time: float, request_id: str):
        """Log outgoing response."""
        # Get user info if available
        user_info = getattr(request.state, 'user', {})
        user_role = user_info.get('role', 'anonymous')
        
        # Prepare log data
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
            "user_role": user_role,
            "timestamp": datetime.utcnow().isoformat(),
            "response_size": response.headers.get("content-length", "0")
        }
        
        # Log response
        if response.status_code >= 400:
            logger.warning(
                f"Response {request_id}: {response.status_code} - {process_time:.3f}s",
                extra=log_data
            )
        else:
            logger.info(
                f"Response {request_id}: {response.status_code} - {process_time:.3f}s",
                extra=log_data
            )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Request ID middleware for tracking requests."""
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        """Add request ID to request and response."""
        # Get request ID from header or generate new one
        request_id = request.headers.get(self.header_name, str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response
