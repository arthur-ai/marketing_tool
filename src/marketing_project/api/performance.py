"""
Performance monitoring API endpoints.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from marketing_project.middleware.auth import get_current_user

logger = logging.getLogger("marketing_project.api.performance")

# Create router
router = APIRouter()


@router.get("/performance/summary")
async def get_performance_summary(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get performance summary and metrics.
    
    Returns comprehensive performance metrics including:
    - Request/response times
    - Memory and CPU usage
    - Error rates
    - Throughput statistics
    """
    try:
        from marketing_project.performance.monitoring import performance_monitor
        
        summary = await performance_monitor.get_performance_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance summary: {str(e)}"
        )


@router.get("/performance/endpoints")
async def get_endpoint_performance(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get performance metrics by endpoint.
    
    Returns detailed performance statistics for each API endpoint.
    """
    try:
        from marketing_project.performance.monitoring import performance_monitor
        
        endpoint_stats = await performance_monitor.get_endpoint_performance()
        return endpoint_stats
    except Exception as e:
        logger.error(f"Failed to get endpoint performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve endpoint performance: {str(e)}"
        )


@router.get("/performance/slow-requests")
async def get_slow_requests(
    threshold: float = 1.0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get requests slower than the specified threshold.
    
    Args:
        threshold: Response time threshold in seconds (default: 1.0)
    """
    try:
        from marketing_project.performance.monitoring import performance_monitor
        
        slow_requests = await performance_monitor.get_slow_requests(threshold)
        return {
            "threshold_seconds": threshold,
            "slow_requests": [req.to_dict() for req in slow_requests],
            "count": len(slow_requests)
        }
    except Exception as e:
        logger.error(f"Failed to get slow requests: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve slow requests: {str(e)}"
        )


@router.get("/performance/error-requests")
async def get_error_requests(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all error requests from the last hour.
    
    Returns requests that resulted in 4xx or 5xx status codes.
    """
    try:
        from marketing_project.performance.monitoring import performance_monitor
        
        error_requests = await performance_monitor.get_error_requests()
        return {
            "error_requests": [req.to_dict() for req in error_requests],
            "count": len(error_requests)
        }
    except Exception as e:
        logger.error(f"Failed to get error requests: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve error requests: {str(e)}"
        )
