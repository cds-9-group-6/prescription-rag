#!/usr/bin/env python3
"""
Test script for the Prescription RAG API endpoints.

This script tests all the main API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


class APITester:
    """Test class for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.passed = 0
        self.failed = 0
    
    def test_endpoint(self, name: str, method: str, endpoint: str, data: Dict = None) -> bool:
        """Test a single endpoint."""
        print(f"\n🧪 Testing {name}...")
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            print(f"   URL: {method.upper()} {url}")
            if data:
                print(f"   Data: {json.dumps(data, indent=2)}")
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                print("   ✅ PASSED")
                self.passed += 1
                return True
            else:
                print(f"   Response: {response.text}")
                print("   ❌ FAILED")
                self.failed += 1
                return False
                
        except Exception as e:
            print(f"   Exception: {e}")
            print("   ❌ FAILED")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Run all API endpoint tests."""
        print("🚀 Starting API Endpoint Tests")
        print(f"   Base URL: {self.base_url}")
        print("="*50)
        
        # Test 1: Health Check
        self.test_endpoint(
            name="Health Check",
            method="GET",
            endpoint="/health"
        )
        
        # Test 2: Root Endpoint
        self.test_endpoint(
            name="Root Endpoint",
            method="GET", 
            endpoint="/"
        )
        
        # Test 3: Available Collections
        self.test_endpoint(
            name="Available Collections",
            method="GET",
            endpoint="/collections"
        )
        
        # Test 4: Collection Info
        self.test_endpoint(
            name="Collection Info",
            method="GET",
            endpoint="/collections/info"
        )
        
        # Test 5: Basic Query
        self.test_endpoint(
            name="Basic Query",
            method="POST",
            endpoint="/query",
            data={
                "query": "What are common diseases in tomatoes?",
                "plant_type": "Tomato",
                "season": "Summer"
            }
        )
        
        # Test 6: Query with Metrics
        self.test_endpoint(
            name="Query with Metrics",
            method="POST", 
            endpoint="/query/metrics",
            data={
                "query": "How to control potato blight?",
                "plant_type": "Potato",
                "session_id": "test-session-123"
            }
        )
        
        # Test 7: Query with Sources
        self.test_endpoint(
            name="Query with Sources",
            method="POST",
            endpoint="/query/sources", 
            data={
                "query": "Rice pest management strategies",
                "location": "Punjab",
                "disease": "Blast"
            }
        )
        
        # Test 8: Auto Plant Detection
        self.test_endpoint(
            name="Auto Plant Detection", 
            method="POST",
            endpoint="/query",
            data={
                "query": "My tomato plants have yellow leaves, what should I do?",
                "season": "Winter"
            }
        )
        
        # Test 9: Metadata Filtering
        self.test_endpoint(
            name="Metadata Filtering",
            method="POST",
            endpoint="/query",
            data={
                "query": "Disease management during monsoon",
                "season": "Kharif",
                "location": "Maharashtra"
            }
        )
        
        # Print results
        print("\n" + "="*50)
        print("📊 Test Results:")
        print(f"   ✅ Passed: {self.passed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   📈 Success Rate: {self.passed/(self.passed+self.failed)*100:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Check the API server and configuration.")
            return False


def check_server_status(base_url: str) -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test function."""
    base_url = "http://localhost:8080"
    
    # Check if server is running
    print("🔍 Checking if API server is running...")
    if not check_server_status(base_url):
        print(f"❌ API server is not responding at {base_url}")
        print("\nTo start the server:")
        print("   ./run_api.sh")
        print("   # or")
        print("   python -m api.start_server")
        sys.exit(1)
    
    print("✅ API server is running")
    
    # Wait a moment for server to be fully ready
    print("⏳ Waiting for server to be fully ready...")
    time.sleep(2)
    
    # Run tests
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
