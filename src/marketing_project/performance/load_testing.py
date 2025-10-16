"""
Load testing utilities for Marketing Project API.

This module provides load testing capabilities to validate
performance under various load conditions.
"""

import asyncio
import aiohttp
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger("marketing_project.performance.load_testing")


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str
    endpoints: List[Dict[str, Any]]
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    think_time_seconds: float = 1.0
    timeout_seconds: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.endpoints:
            raise ValueError("endpoints list cannot be empty")
        if self.concurrent_users <= 0:
            raise ValueError("concurrent_users must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass
class LoadTestResult:
    """Result of a load test."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    status_codes: Dict[int, int]
    errors: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration": self.total_duration,
            "avg_response_time": self.avg_response_time,
            "min_response_time": self.min_response_time,
            "max_response_time": self.max_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate,
            "status_codes": self.status_codes,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat()
        }


class LoadTester:
    """Load testing implementation."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def run_test(self) -> LoadTestResult:
        """Run the load test."""
        logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        logger.info(f"Duration: {self.config.duration_seconds} seconds")
        
        self.start_time = time.time()
        
        # Create tasks for concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self._user_workload(user_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
        
        # Calculate results
        return self._calculate_results()
    
    async def _user_workload(self, user_id: int):
        """Simulate a single user's workload."""
        end_time = self.start_time + self.config.duration_seconds
        
        while time.time() < end_time:
            try:
                # Select random endpoint
                endpoint_config = self.config.endpoints[
                    user_id % len(self.config.endpoints)
                ]
                
                # Make request
                await self._make_request(endpoint_config)
                
                # Think time
                if self.config.think_time_seconds > 0:
                    await asyncio.sleep(self.config.think_time_seconds)
                
            except Exception as e:
                logger.error(f"Error in user {user_id} workload: {e}")
                self.results.append({
                    "user_id": user_id,
                    "endpoint": "unknown",
                    "method": "unknown",
                    "status_code": 0,
                    "response_time": 0,
                    "error": str(e),
                    "timestamp": time.time()
                })
    
    async def _make_request(self, endpoint_config: Dict[str, Any]):
        """Make a single HTTP request."""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint_config['path'].lstrip('/')}"
        method = endpoint_config.get('method', 'GET').upper()
        headers = {**self.config.headers, **endpoint_config.get('headers', {})}
        data = endpoint_config.get('data')
        
        start_time = time.time()
        
        try:
            if method == 'GET':
                async with self.session.get(url, headers=headers) as response:
                    await response.text()
                    status_code = response.status
            elif method == 'POST':
                async with self.session.post(url, headers=headers, json=data) as response:
                    await response.text()
                    status_code = response.status
            elif method == 'PUT':
                async with self.session.put(url, headers=headers, json=data) as response:
                    await response.text()
                    status_code = response.status
            elif method == 'DELETE':
                async with self.session.delete(url, headers=headers) as response:
                    await response.text()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            self.results.append({
                "endpoint": endpoint_config['path'],
                "method": method,
                "status_code": status_code,
                "response_time": response_time,
                "timestamp": time.time(),
                "error": None
            })
            
        except Exception as e:
            response_time = time.time() - start_time
            
            self.results.append({
                "endpoint": endpoint_config['path'],
                "method": method,
                "status_code": 0,
                "response_time": response_time,
                "timestamp": time.time(),
                "error": str(e)
            })
    
    def _calculate_results(self) -> LoadTestResult:
        """Calculate test results from collected data."""
        if not self.results:
            return LoadTestResult(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_duration=0,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=0,
                status_codes={},
                errors=[]
            )
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r['status_code'] < 400])
        failed_requests = total_requests - successful_requests
        
        # Response times
        response_times = [r['response_time'] for r in self.results if r['response_time'] > 0]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = self._percentile(response_times, 95)
            p99_response_time = self._percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        # Duration and throughput
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        
        # Error rate
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # Status codes
        status_codes = {}
        for result in self.results:
            status_code = result['status_code']
            status_codes[status_code] = status_codes.get(status_code, 0) + 1
        
        # Errors
        errors = [r['error'] for r in self.results if r['error']]
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=total_duration,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            status_codes=status_codes,
            errors=errors
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def save_results(self, filepath: str):
        """Save test results to file."""
        result = self._calculate_results()
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Load test results saved to {filepath}")
    
    def print_summary(self):
        """Print a summary of test results."""
        result = self._calculate_results()
        
        print("\n" + "="*60)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful Requests: {result.successful_requests}")
        print(f"Failed Requests: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate:.2%}")
        print(f"Total Duration: {result.total_duration:.2f} seconds")
        print(f"Requests/Second: {result.requests_per_second:.2f}")
        print(f"Average Response Time: {result.avg_response_time:.3f} seconds")
        print(f"Min Response Time: {result.min_response_time:.3f} seconds")
        print(f"Max Response Time: {result.max_response_time:.3f} seconds")
        print(f"95th Percentile: {result.p95_response_time:.3f} seconds")
        print(f"99th Percentile: {result.p99_response_time:.3f} seconds")
        print("\nStatus Code Distribution:")
        for status_code, count in sorted(result.status_codes.items()):
            print(f"  {status_code}: {count}")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(result.errors) > 10:
                print(f"  ... and {len(result.errors) - 10} more errors")
        
        print("="*60)


async def run_load_test(config: LoadTestConfig) -> LoadTestResult:
    """Run a load test with the given configuration."""
    async with LoadTester(config) as tester:
        return await tester.run_test()


# Example usage and predefined test configurations
def create_api_load_test_config(base_url: str) -> LoadTestConfig:
    """Create a load test configuration for the Marketing Project API."""
    return LoadTestConfig(
        base_url=base_url,
        endpoints=[
            {
                "path": "/api/v1/health",
                "method": "GET",
                "headers": {"X-API-Key": "test-key-12345678901234567890123456789012"}
            },
            {
                "path": "/api/v1/ready",
                "method": "GET",
                "headers": {"X-API-Key": "test-key-12345678901234567890123456789012"}
            },
            {
                "path": "/api/v1/content-sources",
                "method": "GET",
                "headers": {"X-API-Key": "test-key-12345678901234567890123456789012"}
            },
            {
                "path": "/api/v1/analyze",
                "method": "POST",
                "headers": {
                    "X-API-Key": "test-key-12345678901234567890123456789012",
                    "Content-Type": "application/json"
                },
                "data": {
                    "content": {
                        "id": "test-content-1",
                        "title": "Test Content",
                        "content": "This is test content for load testing",
                        "type": "blog_post"
                    }
                }
            }
        ],
        concurrent_users=20,
        duration_seconds=120,
        ramp_up_seconds=10,
        think_time_seconds=0.5,
        timeout_seconds=30
    )
