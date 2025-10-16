"""
Security and audit API endpoints.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from marketing_project.middleware.auth import get_current_user

logger = logging.getLogger("marketing_project.api.security")

# Create router
router = APIRouter()


@router.get("/security/audit")
async def get_security_audit_logs(
    limit: int = 100,
    event_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get security audit logs.
    
    Args:
        limit: Maximum number of logs to return (default: 100)
        event_type: Filter by specific event type (optional)
    """
    try:
        from marketing_project.security.audit import security_auditor
        
        logs = await security_auditor.get_recent_logs(limit=limit, event_type=event_type)
        return {
            "logs": logs,
            "count": len(logs),
            "limit": limit,
            "event_type": event_type
        }
    except Exception as e:
        logger.error(f"Failed to get security audit logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve security audit logs: {str(e)}"
        )


@router.get("/security/stats")
async def get_security_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get security statistics and metrics.
    
    Returns security-related statistics including:
    - Authentication success/failure rates
    - Rate limiting violations
    - Attack detection events
    - Risk scores
    """
    try:
        from marketing_project.security.audit import security_auditor
        
        stats = await security_auditor.get_security_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get security stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve security statistics: {str(e)}"
        )
