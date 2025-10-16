"""
Input validation and sanitization for security.

This module provides comprehensive input validation and sanitization
to prevent security vulnerabilities.
"""

import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger("marketing_project.security.input_validation")


class SecurityValidationError(Exception):
    """Exception raised when security validation fails."""
    pass


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format and security.
    
    Args:
        api_key: API key to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If API key is invalid
    """
    if not api_key:
        raise SecurityValidationError("API key cannot be empty")
    
    # Check minimum length
    if len(api_key) < 32:
        raise SecurityValidationError("API key must be at least 32 characters")
    
    # Check maximum length
    if len(api_key) > 256:
        raise SecurityValidationError("API key too long")
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9\-_]+$', api_key):
        raise SecurityValidationError("API key contains invalid characters")
    
    # Check for common weak patterns
    weak_patterns = [
        r'^test.*',
        r'^demo.*',
        r'^example.*',
        r'^admin.*',
        r'^password.*',
        r'^123456.*',
        r'^abcdef.*',
        r'^qwerty.*',
    ]
    
    for pattern in weak_patterns:
        if re.match(pattern, api_key, re.IGNORECASE):
            raise SecurityValidationError("API key appears to be weak or test key")
    
    return True


def validate_content_input(content: Union[str, Dict[str, Any]]) -> bool:
    """
    Validate content input for security issues.
    
    Args:
        content: Content to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If content is invalid
    """
    if isinstance(content, str):
        return _validate_string_content(content)
    elif isinstance(content, dict):
        return _validate_dict_content(content)
    else:
        raise SecurityValidationError("Content must be string or dictionary")


def _validate_string_content(content: str) -> bool:
    """Validate string content for security issues."""
    if not content:
        return True
    
    # Check for maximum length
    if len(content) > 1000000:  # 1MB limit
        raise SecurityValidationError("Content too large")
    
    # Check for SQL injection patterns
    sql_patterns = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\b(OR|AND)\s+\w+\s*=\s*\w+)',
        r'(\bUNION\s+SELECT\b)',
        r'(\bDROP\s+TABLE\b)',
        r'(\bINSERT\s+INTO\b)',
        r'(\bUPDATE\s+SET\b)',
        r'(\bDELETE\s+FROM\b)',
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            raise SecurityValidationError("Potential SQL injection detected")
    
    # Check for XSS patterns
    xss_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
        r'onmouseover\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
    ]
    
    for pattern in xss_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            raise SecurityValidationError("Potential XSS attack detected")
    
    # Check for command injection patterns
    cmd_patterns = [
        r'[;&|`$]',
        r'\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b',
        r'\.\./',
        r'\.\.\\',
        r'<\|',
        r'>\|',
    ]
    
    for pattern in cmd_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            raise SecurityValidationError("Potential command injection detected")
    
    return True


def _validate_dict_content(content: Dict[str, Any]) -> bool:
    """Validate dictionary content for security issues."""
    # Check for maximum depth
    if _get_dict_depth(content) > 10:
        raise SecurityValidationError("Content structure too deep")
    
    # Check for maximum size
    if len(str(content)) > 1000000:  # 1MB limit
        raise SecurityValidationError("Content too large")
    
    # Validate each string value
    for key, value in content.items():
        if isinstance(value, str):
            _validate_string_content(value)
        elif isinstance(value, dict):
            _validate_dict_content(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    _validate_string_content(item)
                elif isinstance(item, dict):
                    _validate_dict_content(item)
    
    return True


def _get_dict_depth(d: Dict[str, Any], depth: int = 0) -> int:
    """Calculate the depth of a nested dictionary."""
    if not isinstance(d, dict):
        return depth
    
    max_depth = depth
    for value in d.values():
        if isinstance(value, dict):
            max_depth = max(max_depth, _get_dict_depth(value, depth + 1))
    
    return max_depth


def sanitize_input(input_data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    """
    Sanitize input data to prevent security issues.
    
    Args:
        input_data: Data to sanitize
        
    Returns:
        Sanitized data
    """
    if isinstance(input_data, str):
        return _sanitize_string(input_data)
    elif isinstance(input_data, dict):
        return _sanitize_dict(input_data)
    else:
        return input_data


def _sanitize_string(text: str) -> str:
    """Sanitize string input."""
    if not text:
        return text
    
    # HTML escape
    text = html.escape(text, quote=True)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def _sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary input."""
    sanitized = {}
    
    for key, value in data.items():
        # Sanitize key
        sanitized_key = _sanitize_string(str(key))
        
        # Sanitize value
        if isinstance(value, str):
            sanitized[sanitized_key] = _sanitize_string(value)
        elif isinstance(value, dict):
            sanitized[sanitized_key] = _sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[sanitized_key] = [
                _sanitize_string(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            sanitized[sanitized_key] = value
    
    return sanitized


def validate_sql_query(query: str) -> bool:
    """
    Validate SQL query for security issues.
    
    Args:
        query: SQL query to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If query is invalid
    """
    if not query:
        raise SecurityValidationError("Query cannot be empty")
    
    # Check for dangerous SQL operations
    dangerous_operations = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
        'TRUNCATE', 'EXEC', 'EXECUTE', 'SP_', 'XP_'
    ]
    
    query_upper = query.upper()
    for operation in dangerous_operations:
        if operation in query_upper:
            raise SecurityValidationError(f"Dangerous SQL operation detected: {operation}")
    
    # Check for SQL injection patterns
    injection_patterns = [
        r'(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+',
        r'UNION\s+SELECT',
        r';\s*DROP',
        r';\s*DELETE',
        r';\s*INSERT',
        r';\s*UPDATE',
        r'--\s*',
        r'/\*.*?\*/',
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise SecurityValidationError("Potential SQL injection detected")
    
    return True


def validate_mongodb_query(query: Dict[str, Any]) -> bool:
    """
    Validate MongoDB query for security issues.
    
    Args:
        query: MongoDB query to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If query is invalid
    """
    if not isinstance(query, dict):
        raise SecurityValidationError("MongoDB query must be a dictionary")
    
    # Check for dangerous operators
    dangerous_operators = [
        '$where', '$expr', '$function', '$accumulator',
        '$addFields', '$lookup', '$graphLookup'
    ]
    
    def check_operators(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                if key in dangerous_operators:
                    raise SecurityValidationError(f"Dangerous MongoDB operator detected: {key} at {current_path}")
                
                if isinstance(value, (dict, list)):
                    check_operators(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_operators(item, f"{path}[{i}]")
    
    check_operators(query)
    
    return True


def validate_redis_key(key: str) -> bool:
    """
    Validate Redis key for security issues.
    
    Args:
        key: Redis key to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If key is invalid
    """
    if not key:
        raise SecurityValidationError("Redis key cannot be empty")
    
    # Check for maximum length
    if len(key) > 512:
        raise SecurityValidationError("Redis key too long")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'\.\./',
        r'\.\.\\',
        r'<\|',
        r'>\|',
        r'[;&|`$]',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, key):
            raise SecurityValidationError("Dangerous pattern in Redis key")
    
    # Check for control characters
    if re.search(r'[\x00-\x1F\x7F]', key):
        raise SecurityValidationError("Control characters in Redis key")
    
    return True


def validate_url(url: str) -> bool:
    """
    Validate URL for security issues.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If URL is invalid
    """
    if not url:
        raise SecurityValidationError("URL cannot be empty")
    
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise SecurityValidationError("Invalid URL scheme")
        
        # Check for localhost/private IPs (unless explicitly allowed)
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise SecurityValidationError("Localhost URLs not allowed")
        
        # Check for private IP ranges
        if parsed.hostname:
            import ipaddress
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private:
                    raise SecurityValidationError("Private IP addresses not allowed")
            except ValueError:
                # Not an IP address, continue
                pass
        
        return True
        
    except Exception as e:
        raise SecurityValidationError(f"Invalid URL: {str(e)}")


def validate_file_upload(filename: str, content_type: str, size: int) -> bool:
    """
    Validate file upload for security issues.
    
    Args:
        filename: Name of the file
        content_type: MIME type of the file
        size: Size of the file in bytes
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        SecurityValidationError: If file is invalid
    """
    # Check file size (10MB limit)
    if size > 10 * 1024 * 1024:
        raise SecurityValidationError("File too large")
    
    # Check filename
    if not filename:
        raise SecurityValidationError("Filename cannot be empty")
    
    # Check for dangerous extensions
    dangerous_extensions = [
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
        '.jar', '.war', '.ear', '.sh', '.ps1', '.php', '.asp', '.jsp'
    ]
    
    filename_lower = filename.lower()
    for ext in dangerous_extensions:
        if filename_lower.endswith(ext):
            raise SecurityValidationError(f"Dangerous file extension: {ext}")
    
    # Check for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise SecurityValidationError("Path traversal detected in filename")
    
    # Check content type
    allowed_types = [
        'text/plain', 'text/html', 'text/markdown', 'application/json',
        'image/jpeg', 'image/png', 'image/gif', 'application/pdf'
    ]
    
    if content_type not in allowed_types:
        raise SecurityValidationError(f"File type not allowed: {content_type}")
    
    return True
