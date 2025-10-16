"""
Security audit and logging.

This module provides comprehensive security auditing and logging capabilities.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("marketing_project.security.audit")


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_SUCCESS = "authz_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLOCKED = "ip_blocked"
    USER_BLOCKED = "user_blocked"
    ATTACK_DETECTED = "attack_detected"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    COMMAND_INJECTION = "command_injection"
    FILE_UPLOAD = "file_upload"
    API_KEY_USAGE = "api_key_usage"
    CONFIGURATION_CHANGE = "config_change"
    DATA_ACCESS = "data_access"
    ERROR = "error"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: SecurityEventType
    timestamp: datetime
    source_ip: str
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    risk_score: int = 0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AuditLogger:
    """Security audit logger."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or "security_audit.log"
        self.setup_logger()
    
    def setup_logger(self):
        """Setup audit logger."""
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.audit_logger.addHandler(file_handler)
    
    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        try:
            # Log to file
            self.audit_logger.info(json.dumps(event.to_dict()))
            
            # Log to console for critical events
            if event.severity in ['high', 'critical']:
                logger.warning(f"SECURITY ALERT: {event.message}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def log_authentication_success(self, source_ip: str, user_id: str, 
                                 api_key: str, endpoint: str, request_id: str):
        """Log successful authentication."""
        event = SecurityEvent(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            api_key=api_key,
            endpoint=endpoint,
            request_id=request_id,
            severity="low",
            message=f"User {user_id} authenticated successfully"
        )
        self.log_event(event)
    
    def log_authentication_failure(self, source_ip: str, api_key: str, 
                                 endpoint: str, reason: str, request_id: str):
        """Log failed authentication."""
        event = SecurityEvent(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            api_key=api_key,
            endpoint=endpoint,
            request_id=request_id,
            severity="medium",
            message=f"Authentication failed: {reason}",
            risk_score=30
        )
        self.log_event(event)
    
    def log_rate_limit_exceeded(self, source_ip: str, user_id: Optional[str],
                              endpoint: str, limit_type: str, request_id: str):
        """Log rate limit exceeded."""
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            endpoint=endpoint,
            request_id=request_id,
            severity="medium",
            message=f"Rate limit exceeded for {limit_type}",
            risk_score=20
        )
        self.log_event(event)
    
    def log_ip_blocked(self, source_ip: str, reason: str, duration: int, request_id: str):
        """Log IP blocking."""
        event = SecurityEvent(
            event_type=SecurityEventType.IP_BLOCKED,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            request_id=request_id,
            severity="high",
            message=f"IP {source_ip} blocked: {reason}",
            details={"duration": duration},
            risk_score=70
        )
        self.log_event(event)
    
    def log_attack_detected(self, source_ip: str, attack_type: str, 
                          details: Dict[str, Any], request_id: str):
        """Log detected attack."""
        event = SecurityEvent(
            event_type=SecurityEventType.ATTACK_DETECTED,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            request_id=request_id,
            severity="critical",
            message=f"Attack detected: {attack_type}",
            details=details,
            risk_score=90
        )
        self.log_event(event)
    
    def log_sql_injection(self, source_ip: str, user_id: Optional[str],
                         query: str, request_id: str):
        """Log SQL injection attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.SQL_INJECTION,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            request_id=request_id,
            severity="critical",
            message="SQL injection attempt detected",
            details={"query": query},
            risk_score=95
        )
        self.log_event(event)
    
    def log_xss_attempt(self, source_ip: str, user_id: Optional[str],
                       content: str, request_id: str):
        """Log XSS attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.XSS_ATTEMPT,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            request_id=request_id,
            severity="high",
            message="XSS attempt detected",
            details={"content": content[:100]},  # Truncate for security
            risk_score=80
        )
        self.log_event(event)
    
    def log_file_upload(self, source_ip: str, user_id: str, filename: str,
                       content_type: str, size: int, request_id: str):
        """Log file upload."""
        event = SecurityEvent(
            event_type=SecurityEventType.FILE_UPLOAD,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            request_id=request_id,
            severity="low",
            message=f"File uploaded: {filename}",
            details={
                "filename": filename,
                "content_type": content_type,
                "size": size
            }
        )
        self.log_event(event)


class SecurityAuditor:
    """Security auditor for comprehensive monitoring."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        self.event_history: List[SecurityEvent] = []
        self.max_history = 10000  # Keep last 10k events in memory
    
    def add_event(self, event: SecurityEvent):
        """Add event to history and log it."""
        # Add to history
        self.event_history.append(event)
        
        # Keep only recent events
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Log event
        self.audit_logger.log_event(event)
    
    def get_events_by_type(self, event_type: SecurityEventType, 
                          limit: int = 100) -> List[SecurityEvent]:
        """Get events by type."""
        events = [e for e in self.event_history if e.event_type == event_type]
        return events[-limit:] if limit else events
    
    def get_events_by_ip(self, source_ip: str, limit: int = 100) -> List[SecurityEvent]:
        """Get events by source IP."""
        events = [e for e in self.event_history if e.source_ip == source_ip]
        return events[-limit:] if limit else events
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[SecurityEvent]:
        """Get events by user ID."""
        events = [e for e in self.event_history if e.user_id == user_id]
        return events[-limit:] if limit else events
    
    def get_high_risk_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get high-risk events."""
        events = [e for e in self.event_history if e.risk_score >= 70]
        return events[-limit:] if limit else events
    
    def get_events_in_timeframe(self, start_time: datetime, end_time: datetime) -> List[SecurityEvent]:
        """Get events within timeframe."""
        return [
            e for e in self.event_history
            if start_time <= e.timestamp <= end_time
        ]
    
    def calculate_risk_score(self, source_ip: str, user_id: Optional[str] = None) -> int:
        """Calculate risk score for IP/user."""
        events = self.get_events_by_ip(source_ip)
        if user_id:
            user_events = self.get_events_by_user(user_id)
            events.extend(user_events)
        
        # Calculate risk based on recent events
        recent_events = [
            e for e in events
            if (datetime.utcnow() - e.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        risk_score = 0
        for event in recent_events:
            risk_score += event.risk_score
        
        # Normalize to 0-100
        return min(100, risk_score // max(1, len(recent_events)))
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect security anomalies."""
        anomalies = []
        
        # Check for high-risk IPs
        ip_risk_scores = {}
        for event in self.event_history:
            if event.source_ip not in ip_risk_scores:
                ip_risk_scores[event.source_ip] = 0
            ip_risk_scores[event.source_ip] += event.risk_score
        
        for ip, score in ip_risk_scores.items():
            if score > 200:  # High risk threshold
                anomalies.append({
                    "type": "high_risk_ip",
                    "ip": ip,
                    "risk_score": score,
                    "message": f"IP {ip} has high risk score: {score}"
                })
        
        # Check for rapid authentication failures
        auth_failures = self.get_events_by_type(SecurityEventType.AUTHENTICATION_FAILURE)
        failure_counts = {}
        for event in auth_failures:
            if event.source_ip not in failure_counts:
                failure_counts[event.source_ip] = 0
            failure_counts[event.source_ip] += 1
        
        for ip, count in failure_counts.items():
            if count > 10:  # More than 10 failures
                anomalies.append({
                    "type": "rapid_auth_failures",
                    "ip": ip,
                    "count": count,
                    "message": f"IP {ip} has {count} authentication failures"
                })
        
        # Check for attack patterns
        attack_events = self.get_events_by_type(SecurityEventType.ATTACK_DETECTED)
        if len(attack_events) > 5:
            anomalies.append({
                "type": "multiple_attacks",
                "count": len(attack_events),
                "message": f"Multiple attacks detected: {len(attack_events)}"
            })
        
        return anomalies
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics."""
        total_events = len(self.event_history)
        
        # Count by type
        event_counts = {}
        for event in self.event_history:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for event in self.event_history:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        # Recent events (last 24 hours)
        recent_cutoff = datetime.utcnow().timestamp() - 86400
        recent_events = [
            e for e in self.event_history
            if e.timestamp.timestamp() > recent_cutoff
        ]
        
        # High-risk events
        high_risk_events = [e for e in self.event_history if e.risk_score >= 70]
        
        return {
            "total_events": total_events,
            "recent_events": len(recent_events),
            "high_risk_events": len(high_risk_events),
            "event_counts": event_counts,
            "severity_counts": severity_counts,
            "anomalies": self.detect_anomalies()
        }
    
    def export_events(self, filepath: str, event_types: Optional[List[SecurityEventType]] = None):
        """Export events to JSON file."""
        events_to_export = self.event_history
        if event_types:
            events_to_export = [
                e for e in self.event_history if e.event_type in event_types
            ]
        
        export_data = [event.to_dict() for event in events_to_export]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} events to {filepath}")


# Global security auditor instance
security_auditor = SecurityAuditor()
