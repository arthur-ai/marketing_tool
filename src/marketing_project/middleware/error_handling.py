"""
Error handling middleware for FastAPI.

This module provides global exception handling and error response formatting.
"""

import logging
import traceback
from typing import Union

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from ..models import ErrorResponse

logger = logging.getLogger("marketing_project.middleware.error_handling")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware for FastAPI."""

    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next):
        """Process request through error handling middleware."""
        try:
            return await call_next(request)
        except Exception as e:
            return await self._handle_exception(request, e)

    async def _handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle different types of exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")

        if isinstance(exc, HTTPException):
            return await self._handle_http_exception(request, exc, request_id)
        elif isinstance(exc, RequestValidationError):
            return await self._handle_validation_error(request, exc, request_id)
        elif isinstance(exc, StarletteHTTPException):
            return await self._handle_starlette_http_exception(request, exc, request_id)
        else:
            return await self._handle_generic_exception(request, exc, request_id)

    async def _handle_http_exception(
        self, request: Request, exc: HTTPException, request_id: str
    ) -> JSONResponse:
        """Handle FastAPI HTTPException."""
        error_response = ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            error_details={"status_code": exc.status_code, "request_id": request_id},
        )

        logger.warning(
            f"HTTP exception in request {request_id}: {exc.status_code} - {exc.detail}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "url": str(request.url),
                "method": request.method,
            },
        )

        return JSONResponse(
            status_code=exc.status_code, content=error_response.model_dump(mode="json")
        )

    async def _handle_validation_error(
        self, request: Request, exc: RequestValidationError, request_id: str
    ) -> JSONResponse:
        """Handle request validation errors."""
        error_details = []
        for error in exc.errors():
            error_details.append(
                {
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }
            )

        error_response = ErrorResponse(
            success=False,
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            error_details={
                "validation_errors": error_details,
                "request_id": request_id,
            },
        )

        logger.warning(
            f"Validation error in request {request_id}: {len(error_details)} errors",
            extra={
                "request_id": request_id,
                "validation_errors": error_details,
                "url": str(request.url),
                "method": request.method,
            },
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(mode="json"),
        )

    async def _handle_starlette_http_exception(
        self, request: Request, exc: StarletteHTTPException, request_id: str
    ) -> JSONResponse:
        """Handle Starlette HTTPException."""
        error_response = ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            error_details={"status_code": exc.status_code, "request_id": request_id},
        )

        logger.warning(
            f"Starlette HTTP exception in request {request_id}: {exc.status_code} - {exc.detail}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "url": str(request.url),
                "method": request.method,
            },
        )

        return JSONResponse(
            status_code=exc.status_code, content=error_response.model_dump(mode="json")
        )

    async def _handle_generic_exception(
        self, request: Request, exc: Exception, request_id: str
    ) -> JSONResponse:
        """Handle generic exceptions."""
        # Log the full exception with traceback
        logger.error(
            f"Unhandled exception in request {request_id}: {str(exc)}",
            extra={
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "url": str(request.url),
                "method": request.method,
                "traceback": traceback.format_exc(),
            },
            exc_info=True,
        )

        # Prepare error response
        if self.debug:
            error_message = f"Internal server error: {str(exc)}"
            error_details = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc().split("\n"),
                "request_id": request_id,
            }
        else:
            error_message = "Internal server error"
            error_details = {
                "request_id": request_id,
                "error_id": request_id,  # For support purposes
            }

        error_response = ErrorResponse(
            success=False,
            message=error_message,
            error_code="INTERNAL_SERVER_ERROR",
            error_details=error_details,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode="json"),
        )


def create_error_response(
    status_code: int,
    message: str,
    error_code: str = None,
    error_details: dict = None,
    request_id: str = None,
) -> JSONResponse:
    """Create a standardized error response."""
    if error_code is None:
        error_code = f"HTTP_{status_code}"

    error_response = ErrorResponse(
        success=False,
        message=message,
        error_code=error_code,
        error_details=error_details or {},
    )

    if request_id:
        error_response.error_details["request_id"] = request_id

    return JSONResponse(
        status_code=status_code, content=error_response.model_dump(mode="json")
    )
