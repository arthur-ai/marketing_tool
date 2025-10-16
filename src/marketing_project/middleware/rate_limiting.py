"""
Rate limiting middleware for FastAPI.

This module provides rate limiting functionality using in-memory storage.
For production, consider using Redis or similar distributed storage.
"""

import logging
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..models import ErrorResponse, RateLimitResponse

logger = logging.getLogger("marketing_project.middleware.rate_limiting")


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm."""
    
    def __init__(self, 
                 requests_per_minute: int = 100,
                 burst_limit: int = 20,
                 per_ip: bool = True,
                 per_user: bool = True):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.per_ip = per_ip
        self.per_user = per_user
        
        # Storage for rate limiting data
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_requests: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Cleanup interval (in seconds)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get real IP from headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return client_ip
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Get user identifier for rate limiting."""
        # Try to get API key from request state
        api_key = getattr(request.state, 'api_key', None)
        if api_key:
            return api_key
        
        # Fallback to client IP if no API key
        return self._get_client_id(request)
    
    def _cleanup_old_requests(self):
        """Clean up old request records to prevent memory leaks."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Remove requests older than 1 hour
        
        # Cleanup IP requests
        for ip in list(self.ip_requests.keys()):
            while self.ip_requests[ip] and self.ip_requests[ip][0] < cutoff_time:
                self.ip_requests[ip].popleft()
            if not self.ip_requests[ip]:
                del self.ip_requests[ip]
        
        # Cleanup user requests
        for user in list(self.user_requests.keys()):
            while self.user_requests[user] and self.user_requests[user][0] < cutoff_time:
                self.user_requests[user].popleft()
            if not self.user_requests[user]:
                del self.user_requests[user]
        
        # Cleanup burst requests
        for client in list(self.burst_requests.keys()):
            while self.burst_requests[client] and self.burst_requests[client][0] < cutoff_time:
                self.burst_requests[client].popleft()
            if not self.burst_requests[client]:
                del self.burst_requests[client]
        
        self.last_cleanup = current_time
        logger.debug("Rate limiter cleanup completed")
    
    def _check_burst_limit(self, client_id: str) -> bool:
        """Check if client has exceeded burst limit."""
        current_time = time.time()
        burst_window = 1  # 1 second window for burst limit
        
        # Remove old requests outside burst window
        while (self.burst_requests[client_id] and 
               self.burst_requests[client_id][0] < current_time - burst_window):
            self.burst_requests[client_id].popleft()
        
        # Check if burst limit exceeded
        if len(self.burst_requests[client_id]) >= self.burst_limit:
            return False
        
        # Add current request
        self.burst_requests[client_id].append(current_time)
        return True
    
    def _check_rate_limit(self, client_id: str, requests_deque: deque) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        window_size = 60  # 1 minute window
        
        # Remove old requests outside window
        while requests_deque and requests_deque[0] < current_time - window_size:
            requests_deque.popleft()
        
        # Check if rate limit exceeded
        if len(requests_deque) >= self.requests_per_minute:
            return False
        
        # Add current request
        requests_deque.append(current_time)
        return True
    
    def is_allowed(self, request: Request) -> Tuple[bool, Optional[RateLimitResponse]]:
        """
        Check if request is allowed based on rate limiting rules.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        self._cleanup_old_requests()
        
        client_id = self._get_client_id(request)
        user_id = self._get_user_id(request)
        
        # Check burst limit
        if not self._check_burst_limit(client_id):
            reset_time = datetime.utcnow() + timedelta(seconds=1)
            return False, RateLimitResponse(
                limit=self.burst_limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=1
            )
        
        # Check IP-based rate limit
        if self.per_ip:
            if not self._check_rate_limit(client_id, self.ip_requests[client_id]):
                reset_time = datetime.utcnow() + timedelta(minutes=1)
                return False, RateLimitResponse(
                    limit=self.requests_per_minute,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=60
                )
        
        # Check user-based rate limit
        if self.per_user and user_id and user_id != client_id:
            if not self._check_rate_limit(user_id, self.user_requests[user_id]):
                reset_time = datetime.utcnow() + timedelta(minutes=1)
                return False, RateLimitResponse(
                    limit=self.requests_per_minute,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=60
                )
        
        # Calculate remaining requests
        current_time = time.time()
        window_size = 60
        
        # Count remaining requests for IP
        ip_remaining = self.requests_per_minute
        if self.per_ip:
            ip_requests = self.ip_requests[client_id]
            while ip_requests and ip_requests[0] < current_time - window_size:
                ip_requests.popleft()
            ip_remaining = self.requests_per_minute - len(ip_requests)
        
        # Count remaining requests for user
        user_remaining = self.requests_per_minute
        if self.per_user and user_id and user_id != client_id:
            user_requests = self.user_requests[user_id]
            while user_requests and user_requests[0] < current_time - window_size:
                user_requests.popleft()
            user_remaining = self.requests_per_minute - len(user_requests)
        
        # Return the minimum remaining count
        remaining = min(ip_remaining, user_remaining)
        reset_time = datetime.utcnow() + timedelta(minutes=1)
        
        rate_limit_info = RateLimitResponse(
            limit=self.requests_per_minute,
            remaining=max(0, remaining),
            reset_time=reset_time
        )
        
        return True, rate_limit_info


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI."""
    
    def __init__(self, 
                 app,
                 requests_per_minute: int = 100,
                 burst_limit: int = 20,
                 per_ip: bool = True,
                 per_user: bool = True,
                 exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_limit=burst_limit,
            per_ip=per_ip,
            per_user=per_user
        )
        self.exclude_paths = exclude_paths or [
            "/health",
            "/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting middleware."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Check rate limit
        is_allowed, rate_limit_info = self.rate_limiter.is_allowed(request)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {request.client.host if request.client else 'unknown'}")
            
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=ErrorResponse(
                    success=False,
                    message="Rate limit exceeded",
                    error_code="RATE_LIMIT_EXCEEDED",
                    error_details={
                        "limit": rate_limit_info.limit,
                        "remaining": rate_limit_info.remaining,
                        "reset_time": rate_limit_info.reset_time.isoformat(),
                        "retry_after": rate_limit_info.retry_after
                    }
                ).dict()
            )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.reset_time.timestamp()))
            if rate_limit_info.retry_after:
                response.headers["Retry-After"] = str(rate_limit_info.retry_after)
            
            return response
        
        # Add rate limit headers to successful response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.reset_time.timestamp()))
        
        return response
