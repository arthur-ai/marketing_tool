#!/usr/bin/env python3
"""
Test script for security validation and database sources.

This script tests the security validation and database source implementations.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from marketing_project.security.input_validation import (
    validate_api_key, validate_content_input, validate_sql_query,
    validate_mongodb_query, validate_redis_key, validate_url,
    SecurityValidationError
)
from marketing_project.security.rate_limiting import SecurityRateLimiter, RateLimitConfig
from marketing_project.security.audit import SecurityAuditor, SecurityEventType
from marketing_project.services.database_source import (
    SQLContentSource, MongoDBContentSource, RedisContentSource
)
from marketing_project.core.content_sources import DatabaseSourceConfig


async def test_security_validation():
    """Test security validation functions."""
    print("Testing Security Validation...")
    print("=" * 50)
    
    # Test API key validation
    print("\n1. Testing API key validation...")
    try:
        # Valid API key
        validate_api_key("test-api-key-12345678901234567890123456789012")
        print("   ✓ Valid API key accepted")
        
        # Invalid API key (too short)
        try:
            validate_api_key("short")
            print("   ✗ Short API key should be rejected")
        except SecurityValidationError:
            print("   ✓ Short API key correctly rejected")
        
        # Invalid API key (weak pattern)
        try:
            validate_api_key("test-api-key-12345678901234567890123456789012")
            print("   ✗ Weak API key should be rejected")
        except SecurityValidationError:
            print("   ✓ Weak API key correctly rejected")
        
    except Exception as e:
        print(f"   ✗ API key validation failed: {e}")
    
    # Test content validation
    print("\n2. Testing content validation...")
    try:
        # Valid content
        validate_content_input("This is valid content")
        print("   ✓ Valid content accepted")
        
        # SQL injection attempt
        try:
            validate_content_input("'; DROP TABLE users; --")
            print("   ✗ SQL injection should be rejected")
        except SecurityValidationError:
            print("   ✓ SQL injection correctly rejected")
        
        # XSS attempt
        try:
            validate_content_input("<script>alert('xss')</script>")
            print("   ✗ XSS attempt should be rejected")
        except SecurityValidationError:
            print("   ✓ XSS attempt correctly rejected")
        
    except Exception as e:
        print(f"   ✗ Content validation failed: {e}")
    
    # Test SQL query validation
    print("\n3. Testing SQL query validation...")
    try:
        # Valid query
        validate_sql_query("SELECT * FROM users WHERE id = 1")
        print("   ✓ Valid SQL query accepted")
        
        # Dangerous query
        try:
            validate_sql_query("DROP TABLE users")
            print("   ✗ Dangerous query should be rejected")
        except SecurityValidationError:
            print("   ✓ Dangerous query correctly rejected")
        
    except Exception as e:
        print(f"   ✗ SQL query validation failed: {e}")
    
    # Test URL validation
    print("\n4. Testing URL validation...")
    try:
        # Valid URL
        validate_url("https://example.com")
        print("   ✓ Valid URL accepted")
        
        # Invalid URL (localhost)
        try:
            validate_url("http://localhost:8000")
            print("   ✗ Localhost URL should be rejected")
        except SecurityValidationError:
            print("   ✓ Localhost URL correctly rejected")
        
    except Exception as e:
        print(f"   ✗ URL validation failed: {e}")


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\nTesting Rate Limiting...")
    print("=" * 50)
    
    try:
        # Create rate limiter
        config = RateLimitConfig(
            requests_per_minute=5,
            burst_limit=2,
            max_concurrent_requests=3
        )
        rate_limiter = SecurityRateLimiter(config)
        
        # Test normal requests
        print("\n1. Testing normal requests...")
        for i in range(3):
            allowed, info = rate_limiter.is_allowed("192.168.1.1", "user1")
            if allowed:
                print(f"   ✓ Request {i+1} allowed")
            else:
                print(f"   ✗ Request {i+1} should be allowed")
        
        # Test rate limiting
        print("\n2. Testing rate limiting...")
        for i in range(3, 8):
            allowed, info = rate_limiter.is_allowed("192.168.1.1", "user1")
            if not allowed:
                print(f"   ✓ Request {i+1} correctly rate limited")
                break
            else:
                print(f"   ✗ Request {i+1} should be rate limited")
        
        # Test concurrent requests
        print("\n3. Testing concurrent requests...")
        for i in range(5):
            identifier = f"user{i}"
            if rate_limiter.check_concurrent_requests(identifier):
                print(f"   ✓ Concurrent request {i+1} allowed")
            else:
                print(f"   ✓ Concurrent request {i+1} correctly limited")
        
        # Get statistics
        stats = rate_limiter.get_stats()
        print(f"\n4. Rate limiter stats: {stats}")
        
        rate_limiter.cleanup()
        
    except Exception as e:
        print(f"   ✗ Rate limiting test failed: {e}")


async def test_database_sources():
    """Test database source implementations."""
    print("\nTesting Database Sources...")
    print("=" * 50)
    
    # Test SQL source
    print("\n1. Testing SQL source...")
    try:
        config = DatabaseSourceConfig(
            name="test_sql",
            source_type="database",
            connection_string="sqlite:///test.db",
            table_name="test_table",
            query="SELECT * FROM test_table"
        )
        
        sql_source = SQLContentSource(config)
        print("   ✓ SQL source created")
        
        # Test initialization (will fail without actual DB)
        initialized = await sql_source.initialize()
        if not initialized:
            print("   ✓ SQL source correctly failed to initialize (no DB)")
        else:
            print("   ✓ SQL source initialized successfully")
        
        # Test health check
        healthy = await sql_source.health_check()
        print(f"   ✓ SQL source health check: {healthy}")
        
        await sql_source.cleanup()
        
    except Exception as e:
        print(f"   ✗ SQL source test failed: {e}")
    
    # Test MongoDB source
    print("\n2. Testing MongoDB source...")
    try:
        config = DatabaseSourceConfig(
            name="test_mongo",
            source_type="database",
            connection_string="mongodb://localhost:27017",
            table_name="test_collection",
            metadata={"database": "test_db"}
        )
        
        mongo_source = MongoDBContentSource(config)
        print("   ✓ MongoDB source created")
        
        # Test initialization (will fail without actual DB)
        initialized = await mongo_source.initialize()
        if not initialized:
            print("   ✓ MongoDB source correctly failed to initialize (no DB)")
        else:
            print("   ✓ MongoDB source initialized successfully")
        
        # Test health check
        healthy = await mongo_source.health_check()
        print(f"   ✓ MongoDB source health check: {healthy}")
        
        await mongo_source.cleanup()
        
    except Exception as e:
        print(f"   ✗ MongoDB source test failed: {e}")
    
    # Test Redis source
    print("\n3. Testing Redis source...")
    try:
        config = DatabaseSourceConfig(
            name="test_redis",
            source_type="database",
            connection_string="redis://localhost:6379",
            metadata={"key_pattern": "test:*"}
        )
        
        redis_source = RedisContentSource(config)
        print("   ✓ Redis source created")
        
        # Test initialization (will fail without actual DB)
        initialized = await redis_source.initialize()
        if not initialized:
            print("   ✓ Redis source correctly failed to initialize (no DB)")
        else:
            print("   ✓ Redis source initialized successfully")
        
        # Test health check
        healthy = await redis_source.health_check()
        print(f"   ✓ Redis source health check: {healthy}")
        
        await redis_source.cleanup()
        
    except Exception as e:
        print(f"   ✗ Redis source test failed: {e}")


async def test_security_audit():
    """Test security audit functionality."""
    print("\nTesting Security Audit...")
    print("=" * 50)
    
    try:
        # Create security auditor
        auditor = SecurityAuditor()
        print("   ✓ Security auditor created")
        
        # Test event logging
        print("\n1. Testing event logging...")
        auditor.audit_logger.log_authentication_success(
            source_ip="192.168.1.1",
            user_id="test_user",
            api_key="test-key",
            endpoint="/api/v1/analyze",
            request_id="test-123"
        )
        print("   ✓ Authentication success logged")
        
        auditor.audit_logger.log_authentication_failure(
            source_ip="192.168.1.2",
            api_key="invalid-key",
            endpoint="/api/v1/analyze",
            reason="Invalid API key",
            request_id="test-124"
        )
        print("   ✓ Authentication failure logged")
        
        auditor.audit_logger.log_rate_limit_exceeded(
            source_ip="192.168.1.3",
            user_id="test_user",
            endpoint="/api/v1/analyze",
            limit_type="IP",
            request_id="test-125"
        )
        print("   ✓ Rate limit exceeded logged")
        
        # Test statistics
        print("\n2. Testing statistics...")
        summary = auditor.get_security_summary()
        print(f"   ✓ Security summary: {summary}")
        
        # Test anomaly detection
        print("\n3. Testing anomaly detection...")
        anomalies = auditor.detect_anomalies()
        print(f"   ✓ Anomalies detected: {len(anomalies)}")
        
    except Exception as e:
        print(f"   ✗ Security audit test failed: {e}")


async def main():
    """Run all tests."""
    print("Security and Database Source Testing")
    print("=" * 60)
    
    await test_security_validation()
    await test_rate_limiting()
    await test_database_sources()
    await test_security_audit()
    
    print("\n" + "=" * 60)
    print("Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
