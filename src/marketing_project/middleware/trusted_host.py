"""
Trusted host middleware for FastAPI with health check bypass.

This module provides host validation middleware that allows health check endpoints
to bypass host validation for Kubernetes probes and load balancers.
"""

import logging
from typing import List

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("marketing_project.middleware.trusted_host")


class TrustedHostMiddlewareWithHealthBypass(BaseHTTPMiddleware):
    """
    Trusted host middleware that bypasses validation for health check endpoints.

    This is necessary because Kubernetes health check probes and load balancers
    may not send the correct Host header, but we still want to validate it for
    regular API requests.
    """

    def __init__(self, app, allowed_hosts: List[str] = None):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts or []
        # Add testserver for test environments
        if "testserver" not in self.allowed_hosts:
            self.allowed_hosts.append("testserver")
        # Health check endpoints that should bypass host validation
        self.health_check_paths = [
            "/api/v1/health",
            "/api/v1/ready",
            "/health",
            "/ready",
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request through trusted host middleware."""
        # Check if this is a health check endpoint
        path = request.url.path
        is_health_check = any(
            path.startswith(health_path) for health_path in self.health_check_paths
        )

        # Bypass host validation for health check endpoints
        if is_health_check:
            return await call_next(request)

        # For all other endpoints, validate the Host header
        host = request.headers.get("host", "").split(":")[0]  # Remove port if present

        if not self.allowed_hosts:
            # If no allowed hosts configured, allow all
            return await call_next(request)

        # Check if host matches any allowed pattern
        host_allowed = False
        for allowed_host in self.allowed_hosts:
            if allowed_host == host:
                host_allowed = True
                break
            elif allowed_host.startswith("*."):
                # Wildcard subdomain matching (e.g., *.arthur.ai)
                domain = allowed_host[2:]  # Remove "*."
                if host.endswith("." + domain) or host == domain:
                    host_allowed = True
                    break

        if not host_allowed:
            logger.warning(
                f"Rejected request from untrusted host: {host}",
                extra={
                    "host": host,
                    "path": path,
                    "method": request.method,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": f"Invalid host header: {host}. Allowed hosts: {', '.join(self.allowed_hosts)}"
                },
            )

        return await call_next(request)
