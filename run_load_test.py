#!/usr/bin/env python3
"""
Load testing script for Marketing Project API.

This script runs comprehensive load tests against the API to validate
performance under various load conditions.
"""

import asyncio
import argparse
import sys
import os
from typing import List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from marketing_project.performance.load_testing import (
    LoadTestConfig, LoadTester, create_api_load_test_config
)


async def run_basic_load_test(base_url: str):
    """Run basic load test."""
    print("🚀 Running Basic Load Test...")
    
    config = LoadTestConfig(
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
            }
        ],
        concurrent_users=10,
        duration_seconds=60,
        ramp_up_seconds=5,
        think_time_seconds=1.0
    )
    
    async with LoadTester(config) as tester:
        result = await tester.run_test()
        tester.print_summary()
        return result


async def run_stress_test(base_url: str):
    """Run stress test."""
    print("🔥 Running Stress Test...")
    
    config = LoadTestConfig(
        base_url=base_url,
        endpoints=[
            {
                "path": "/api/v1/analyze",
                "method": "POST",
                "headers": {
                    "X-API-Key": "test-key-12345678901234567890123456789012",
                    "Content-Type": "application/json"
                },
                "data": {
                    "content": {
                        "id": "stress-test-content",
                        "title": "Stress Test Content",
                        "content": "This is content for stress testing the API performance under high load conditions.",
                        "type": "blog_post"
                    }
                }
            }
        ],
        concurrent_users=50,
        duration_seconds=120,
        ramp_up_seconds=10,
        think_time_seconds=0.1
    )
    
    async with LoadTester(config) as tester:
        result = await tester.run_test()
        tester.print_summary()
        return result


async def run_spike_test(base_url: str):
    """Run spike test."""
    print("⚡ Running Spike Test...")
    
    config = LoadTestConfig(
        base_url=base_url,
        endpoints=[
            {
                "path": "/api/v1/content-sources",
                "method": "GET",
                "headers": {"X-API-Key": "test-key-12345678901234567890123456789012"}
            }
        ],
        concurrent_users=100,
        duration_seconds=30,
        ramp_up_seconds=2,
        think_time_seconds=0.0
    )
    
    async with LoadTester(config) as tester:
        result = await tester.run_test()
        tester.print_summary()
        return result


async def run_endurance_test(base_url: str):
    """Run endurance test."""
    print("⏰ Running Endurance Test...")
    
    config = LoadTestConfig(
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
            }
        ],
        concurrent_users=20,
        duration_seconds=600,  # 10 minutes
        ramp_up_seconds=30,
        think_time_seconds=2.0
    )
    
    async with LoadTester(config) as tester:
        result = await tester.run_test()
        tester.print_summary()
        return result


async def run_comprehensive_test(base_url: str):
    """Run comprehensive test with all endpoints."""
    print("🎯 Running Comprehensive Test...")
    
    config = create_api_load_test_config(base_url)
    config.concurrent_users = 30
    config.duration_seconds = 180
    config.ramp_up_seconds = 15
    
    async with LoadTester(config) as tester:
        result = await tester.run_test()
        tester.print_summary()
        return result


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Load test Marketing Project API')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='Base URL of the API (default: http://localhost:8000)')
    parser.add_argument('--test', choices=['basic', 'stress', 'spike', 'endurance', 'comprehensive', 'all'],
                       default='all', help='Type of test to run (default: all)')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"🎬 Starting load tests against {args.url}")
    print("=" * 60)
    
    results = []
    
    try:
        if args.test in ['basic', 'all']:
            result = await run_basic_load_test(args.url)
            results.append(('Basic Load Test', result))
        
        if args.test in ['stress', 'all']:
            result = await run_stress_test(args.url)
            results.append(('Stress Test', result))
        
        if args.test in ['spike', 'all']:
            result = await run_spike_test(args.url)
            results.append(('Spike Test', result))
        
        if args.test in ['endurance', 'all']:
            result = await run_endurance_test(args.url)
            results.append(('Endurance Test', result))
        
        if args.test in ['comprehensive', 'all']:
            result = await run_comprehensive_test(args.url)
            results.append(('Comprehensive Test', result))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 LOAD TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in results:
            print(f"\n{test_name}:")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  Success Rate: {(1 - result.error_rate):.2%}")
            print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
            print(f"  Requests/Second: {result.requests_per_second:.2f}")
            print(f"  95th Percentile: {result.p95_response_time:.3f}s")
        
        # Save results if requested
        if args.output:
            import json
            output_data = {
                'url': args.url,
                'test_type': args.test,
                'results': {name: result.to_dict() for name, result in results}
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
        print("\n✅ Load testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Load testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Load testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
