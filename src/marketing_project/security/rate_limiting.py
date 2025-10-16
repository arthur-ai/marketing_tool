"""
Advanced rate limiting for security.

This module provides enhanced rate limiting capabilities with security features
like IP whitelisting, user-based limiting, and attack detection.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger("marketing_project.security.rate_limiting")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 100
    burst_limit: int = 20
    window_size: int = 60  # seconds
    max_concurrent_requests: int = 10
    block_duration: int = 300  # seconds
    suspicious_threshold: int = 5  # suspicious requests before blocking


class IPWhitelist:
    """IP whitelist management."""
    
    def __init__(self):
        self.whitelisted_ips: Set[str] = set()
        self.load_whitelist()
    
    def load_whitelist(self):
        """Load whitelisted IPs from configuration."""
        # In production, this would load from a secure store
        import os
        whitelist_env = os.getenv("IP_WHITELIST", "")
        if whitelist_env:
            self.whitelisted_ips.update(whitelist_env.split(","))
    
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        return ip in self.whitelisted_ips
    
    def add_ip(self, ip: str):
        """Add IP to whitelist."""
        self.whitelisted_ips.add(ip)
    
    def remove_ip(self, ip: str):
        """Remove IP from whitelist."""
        self.whitelisted_ips.discard(ip)


class UserRateLimiter:
    """User-based rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.user_blocks: Dict[str, datetime] = {}
        self.user_suspicious: Dict[str, int] = defaultdict(int)
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is currently blocked."""
        if user_id in self.user_blocks:
            if datetime.utcnow() < self.user_blocks[user_id]:
                return True
            else:
                # Block expired
                del self.user_blocks[user_id]
        return False
    
    def block_user(self, user_id: str, duration: Optional[int] = None):
        """Block user for specified duration."""
        duration = duration or self.config.block_duration
        self.user_blocks[user_id] = datetime.utcnow() + timedelta(seconds=duration)
        logger.warning(f"User {user_id} blocked for {duration} seconds")
    
    def is_user_rate_limited(self, user_id: str) -> Tuple[bool, int]:
        """Check if user has exceeded rate limit."""
        if self.is_user_blocked(user_id):
            return True, 0
        
        current_time = time.time()
        user_requests = self.user_requests[user_id]
        
        # Remove old requests outside window
        while user_requests and user_requests[0] < current_time - self.config.window_size:
            user_requests.popleft()
        
        # Check rate limit
        if len(user_requests) >= self.config.requests_per_minute:
            # Increment suspicious counter
            self.user_suspicious[user_id] += 1
            
            # Block if too many suspicious requests
            if self.user_suspicious[user_id] >= self.config.suspicious_threshold:
                self.block_user(user_id)
                return True, 0
            
            return True, 0
        
        # Reset suspicious counter on successful request
        if self.user_suspicious[user_id] > 0:
            self.user_suspicious[user_id] = max(0, self.user_suspicious[user_id] - 1)
        
        # Add current request
        user_requests.append(current_time)
        
        remaining = self.config.requests_per_minute - len(user_requests)
        return False, remaining


class SecurityRateLimiter:
    """Enhanced rate limiter with security features."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.ip_whitelist = IPWhitelist()
        self.user_limiter = UserRateLimiter(self.config)
        
        # IP-based tracking
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.ip_blocks: Dict[str, datetime] = {}
        self.ip_suspicious: Dict[str, int] = defaultdict(int)
        
        # Attack detection
        self.attack_patterns: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Set[str] = set()
        
        # Concurrent request tracking
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old tracking data."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Remove data older than 1 hour
        
        # Cleanup IP requests
        for ip in list(self.ip_requests.keys()):
            while self.ip_requests[ip] and self.ip_requests[ip][0] < cutoff_time:
                self.ip_requests[ip].popleft()
            if not self.ip_requests[ip]:
                del self.ip_requests[ip]
        
        # Cleanup user requests
        for user in list(self.user_limiter.user_requests.keys()):
            while (self.user_limiter.user_requests[user] and 
                   self.user_limiter.user_requests[user][0] < cutoff_time):
                self.user_limiter.user_requests[user].popleft()
            if not self.user_limiter.user_requests[user]:
                del self.user_limiter.user_requests[user]
        
        # Cleanup expired blocks
        current_dt = datetime.utcnow()
        expired_ips = [
            ip for ip, block_time in self.ip_blocks.items()
            if current_dt >= block_time
        ]
        for ip in expired_ips:
            del self.ip_blocks[ip]
            self.blocked_ips.discard(ip)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if ip in self.blocked_ips:
            return True
        
        if ip in self.ip_blocks:
            if datetime.utcnow() < self.ip_blocks[ip]:
                return True
            else:
                # Block expired
                del self.ip_blocks[ip]
                self.blocked_ips.discard(ip)
        
        return False
    
    def block_ip(self, ip: str, duration: Optional[int] = None):
        """Block IP for specified duration."""
        duration = duration or self.config.block_duration
        self.ip_blocks[ip] = datetime.utcnow() + timedelta(seconds=duration)
        self.blocked_ips.add(ip)
        logger.warning(f"IP {ip} blocked for {duration} seconds")
    
    def is_ip_rate_limited(self, ip: str) -> Tuple[bool, int]:
        """Check if IP has exceeded rate limit."""
        if self.is_ip_blocked(ip):
            return True, 0
        
        # Whitelisted IPs bypass rate limiting
        if self.ip_whitelist.is_whitelisted(ip):
            return False, self.config.requests_per_minute
        
        current_time = time.time()
        ip_requests = self.ip_requests[ip]
        
        # Remove old requests outside window
        while ip_requests and ip_requests[0] < current_time - self.config.window_size:
            ip_requests.popleft()
        
        # Check rate limit
        if len(ip_requests) >= self.config.requests_per_minute:
            # Increment suspicious counter
            self.ip_suspicious[ip] += 1
            
            # Block if too many suspicious requests
            if self.ip_suspicious[ip] >= self.config.suspicious_threshold:
                self.block_ip(ip)
                return True, 0
            
            return True, 0
        
        # Reset suspicious counter on successful request
        if self.ip_suspicious[ip] > 0:
            self.ip_suspicious[ip] = max(0, self.ip_suspicious[ip] - 1)
        
        # Add current request
        ip_requests.append(current_time)
        
        remaining = self.config.requests_per_minute - len(ip_requests)
        return False, remaining
    
    def check_concurrent_requests(self, identifier: str) -> bool:
        """Check if too many concurrent requests."""
        current = self.concurrent_requests[identifier]
        if current >= self.config.max_concurrent_requests:
            return False
        
        self.concurrent_requests[identifier] += 1
        return True
    
    def release_concurrent_request(self, identifier: str):
        """Release a concurrent request."""
        if self.concurrent_requests[identifier] > 0:
            self.concurrent_requests[identifier] -= 1
    
    def detect_attack_pattern(self, ip: str, user_id: Optional[str], 
                            request_data: Dict[str, Any]) -> bool:
        """Detect potential attack patterns."""
        # Check for rapid-fire requests
        current_time = time.time()
        recent_requests = [
            req_time for req_time in self.ip_requests[ip]
            if current_time - req_time < 10  # Last 10 seconds
        ]
        
        if len(recent_requests) > 10:  # More than 10 requests in 10 seconds
            logger.warning(f"Rapid-fire requests detected from IP {ip}")
            self.block_ip(ip, 600)  # Block for 10 minutes
            return True
        
        # Check for suspicious request patterns
        if user_id:
            user_requests = self.user_limiter.user_requests[user_id]
            if len(user_requests) > 50:  # More than 50 requests in window
                logger.warning(f"High request volume detected from user {user_id}")
                self.user_limiter.block_user(user_id, 300)  # Block for 5 minutes
                return True
        
        # Check for repeated failed requests
        failed_key = f"{ip}:failed"
        if self.attack_patterns[failed_key] > 5:
            logger.warning(f"Multiple failed requests from IP {ip}")
            self.block_ip(ip, 300)  # Block for 5 minutes
            return True
        
        return False
    
    def record_failed_request(self, ip: str, user_id: Optional[str] = None):
        """Record a failed request for attack detection."""
        failed_key = f"{ip}:failed"
        self.attack_patterns[failed_key] += 1
        
        if user_id:
            user_failed_key = f"{user_id}:failed"
            self.attack_patterns[user_failed_key] += 1
    
    def is_allowed(self, ip: str, user_id: Optional[str] = None, 
                  request_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on all rate limiting rules.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Check IP blocking
        if self.is_ip_blocked(ip):
            return False, {
                "reason": "ip_blocked",
                "message": "IP address is blocked",
                "retry_after": 300
            }
        
        # Check user blocking
        if user_id and self.user_limiter.is_user_blocked(user_id):
            return False, {
                "reason": "user_blocked", 
                "message": "User is blocked",
                "retry_after": 300
            }
        
        # Check concurrent requests
        identifier = user_id or ip
        if not self.check_concurrent_requests(identifier):
            return False, {
                "reason": "too_many_concurrent",
                "message": "Too many concurrent requests",
                "retry_after": 60
            }
        
        # Check IP rate limit
        ip_limited, ip_remaining = self.is_ip_rate_limited(ip)
        if ip_limited:
            return False, {
                "reason": "ip_rate_limited",
                "message": "IP rate limit exceeded",
                "remaining": ip_remaining,
                "retry_after": 60
            }
        
        # Check user rate limit
        if user_id:
            user_limited, user_remaining = self.user_limiter.is_user_rate_limited(user_id)
            if user_limited:
                return False, {
                    "reason": "user_rate_limited",
                    "message": "User rate limit exceeded", 
                    "remaining": user_remaining,
                    "retry_after": 60
                }
        
        # Detect attack patterns
        if request_data and self.detect_attack_pattern(ip, user_id, request_data):
            return False, {
                "reason": "attack_detected",
                "message": "Attack pattern detected",
                "retry_after": 600
            }
        
        # Calculate remaining requests
        remaining = min(ip_remaining, user_remaining if user_id else ip_remaining)
        
        return True, {
            "reason": "allowed",
            "remaining": remaining,
            "limit": self.config.requests_per_minute
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "blocked_ips": len(self.blocked_ips),
            "blocked_users": len(self.user_limiter.user_blocks),
            "suspicious_ips": len(self.ip_suspicious),
            "suspicious_users": len(self.user_limiter.user_suspicious),
            "concurrent_requests": sum(self.concurrent_requests.values()),
            "attack_patterns": len(self.attack_patterns)
        }
