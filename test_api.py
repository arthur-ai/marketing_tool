#!/usr/bin/env python3
"""
Test script for the FastAPI implementation.

This script tests the basic functionality of the marketing project API.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import httpx
from marketing_project.models import BlogPostContext, AnalyzeRequest


async def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    api_key = os.getenv("API_KEY", "test-api-key-12345678901234567890123456789012")
    
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        print("Testing Marketing Project API...")
        print("=" * 50)
        
        # Test 1: Health check (no auth required)
        print("\n1. Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 2: API health check (with auth)
        print("\n2. Testing API health check...")
        try:
            response = await client.get(f"{base_url}/api/v1/health", headers=headers)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 3: Content analysis
        print("\n3. Testing content analysis...")
        try:
            content = BlogPostContext(
                id="test-1",
                title="Test Marketing Article",
                content="This is a test article about marketing automation and content strategy.",
                author="Test Author",
                tags=["marketing", "automation", "content"],
                category="tutorial"
            )
            
            request_data = AnalyzeRequest(content=content)
            response = await client.post(
                f"{base_url}/api/v1/analyze",
                headers=headers,
                json=request_data.dict()
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('success')}")
                print(f"   Message: {result.get('message')}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 4: Pipeline execution
        print("\n4. Testing pipeline execution...")
        try:
            content = BlogPostContext(
                id="test-2",
                title="Test Pipeline Article",
                content="This is a test article for pipeline processing.",
                author="Test Author",
                tags=["pipeline", "test"],
                category="tutorial"
            )
            
            request_data = {
                "content": content.dict(),
                "options": {"test_mode": True}
            }
            
            response = await client.post(
                f"{base_url}/api/v1/pipeline",
                headers=headers,
                json=request_data
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('success')}")
                print(f"   Message: {result.get('message')}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 5: Content sources
        print("\n5. Testing content sources...")
        try:
            response = await client.get(f"{base_url}/api/v1/content-sources", headers=headers)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('success')}")
                print(f"   Sources count: {len(result.get('sources', []))}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 6: Rate limiting (make multiple requests)
        print("\n6. Testing rate limiting...")
        try:
            for i in range(5):
                response = await client.get(f"{base_url}/api/v1/health", headers=headers)
                print(f"   Request {i+1}: Status {response.status_code}, "
                      f"Remaining: {response.headers.get('X-RateLimit-Remaining', 'N/A')}")
                if response.status_code == 429:
                    print(f"   Rate limit hit after {i+1} requests")
                    break
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "=" * 50)
        print("API testing completed!")


if __name__ == "__main__":
    print("Make sure the API server is running on http://localhost:8000")
    print("You can start it with: python -m marketing_project.server")
    print()
    
    # Set a test API key if not provided
    if not os.getenv("API_KEY"):
        os.environ["API_KEY"] = "test-api-key-12345678901234567890123456789012"
    
    asyncio.run(test_api())
