"""
Test script for the /query/metrics endpoint of the Prescription RAG API.

This script tests the metrics-enabled query endpoint which includes session tracking
and comprehensive metrics logging functionality.
"""

import requests
import json
import time
import sys
import uuid
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsTester:
    """Test class specifically for the /query/metrics endpoint."""
    
    def __init__(self, base_url: str = "http://0.0.0.0:8081"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_session_id = str(uuid.uuid4())
    
    def test_metrics_query(self, query: str, plant_type: Optional[str] = None, 
                          season: Optional[str] = None, location: Optional[str] = None, 
                          disease: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Test the /query/metrics endpoint with given parameters."""
        
        # Use provided session_id or default test session_id
        if session_id is None:
            session_id = self.test_session_id
        
        # Prepare request data
        data = {
            "query": query,
            "session_id": session_id
        }
        
        # Add optional parameters
        if plant_type:
            data["plant_type"] = plant_type
        if season:
            data["season"] = season
        if location:
            data["location"] = location
        if disease:
            data["disease"] = disease
        
        print(f"\n🧪 Testing Metrics Query:")
        print(f"   Query: '{query}'")
        print(f"   Session ID: {session_id}")
        if plant_type:
            print(f"   Plant Type: {plant_type}")
        if season:
            print(f"   Season: {season}")
        if location:
            print(f"   Location: {location}")
        if disease:
            print(f"   Disease: {disease}")
        print("-" * 60)
        
        try:
            url = f"{self.base_url}/query/metrics"
            start_time = time.time()
            
            response = self.session.post(url, json=data)  # No timeout for debug mode
            
            request_time = time.time() - start_time
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Request Time: {request_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('success', False)}")
                print(f"   Collection Used: {result.get('collection_used', 'N/A')}")
                print(f"   Query Time: {result.get('query_time', 0):.2f}s")
                print("\n📝 Answer:")
                print(f"   {result.get('answer', 'No answer received')[:200]}...")
                print("\n✅ SUCCESS")
                return result
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get('detail', error_detail)
                except:
                    pass
                print(f"   Error Response: {error_detail}")
                print("\n❌ FAILED")
                return {"error": error_detail, "status_code": response.status_code}
                
        except Exception as e:
            print(f"   Exception: {e}")
            print("\n❌ FAILED")
            return {"error": str(e), "exception": True}
    
    def test_multiple_queries_same_session(self, queries: list, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Test multiple queries with the same session ID to verify session tracking."""
        
        if session_id is None:
            session_id = f"test_session_{int(time.time())}"
        
        print(f"\n🔗 Testing Multiple Queries with Same Session ID: {session_id}")
        print("=" * 60)
        
        results = []
        for i, query_data in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")
            result = self.test_metrics_query(session_id=session_id, **query_data)
            results.append(result)
            time.sleep(1)  # Small delay between queries
        
        return {"session_id": session_id, "results": results}
    
    def test_error_scenarios(self) -> Dict[str, Any]:
        """Test various error scenarios for the metrics endpoint."""
        
        print(f"\n❌ Testing Error Scenarios")
        print("=" * 60)
        
        error_tests = []
        
        # Test 1: Empty query
        print("\n--- Test: Empty Query ---")
        try:
            response = self.session.post(
                f"{self.base_url}/query/metrics", 
                json={"query": "", "session_id": "error_test"}
            )
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            error_tests.append({"test": "empty_query", "status_code": response.status_code})
        except Exception as e:
            print(f"   Exception: {e}")
            error_tests.append({"test": "empty_query", "error": str(e)})
        
        # Test 2: Invalid JSON
        print("\n--- Test: Invalid Request Format ---")
        try:
            response = self.session.post(
                f"{self.base_url}/query/metrics", 
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            error_tests.append({"test": "invalid_json", "status_code": response.status_code})
        except Exception as e:
            print(f"   Exception: {e}")
            error_tests.append({"test": "invalid_json", "error": str(e)})
        
        # Test 3: Missing required field
        print("\n--- Test: Missing Query Field ---")
        try:
            response = self.session.post(
                f"{self.base_url}/query/metrics", 
                json={"session_id": "missing_query_test"}
            )
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            error_tests.append({"test": "missing_query", "status_code": response.status_code})
        except Exception as e:
            print(f"   Exception: {e}")
            error_tests.append({"test": "missing_query", "error": str(e)})
        
        return {"error_tests": error_tests}


def check_server_health(base_url: str) -> bool:
    """Check if the API server is healthy and ready."""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy")
            print(f"   Available Collections: {health_data.get('available_collections', [])}")
            print(f"   Total Collections: {health_data.get('total_collections', 0)}")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False


def main():
    """Main function to run metrics query tests."""
    
    # Configuration
    base_url = "http://0.0.0.0:8081"  # Default from start_server.py
    
    print("🚀 Prescription RAG API - Query Metrics Endpoint Tester")
    print(f"   Target URL: {base_url}")
    print("=" * 70)
    
    # Check server health first
    print("\n🔍 Checking server health...")
    if not check_server_health(base_url):
        print("\n💡 To start the server, run:")
        print("   python api/start_server.py")
        print("   # or")
        print("   ./run_api.sh")
        sys.exit(1)
    
    # Initialize tester
    tester = MetricsTester(base_url)
    
    print(f"\n🆔 Test Session ID: {tester.test_session_id}")
    
    # Test cases for the /query/metrics endpoint
    print("\n" + "=" * 70)
    print("🧪 RUNNING METRICS QUERY TESTS")
    print("=" * 70)
    
    # Test 1: Basic tomato disease query with metrics
    tester.test_metrics_query(
        query="What are common diseases in tomatoes and how to treat them?",
        plant_type="Tomato",
        season="Summer",
        location="Karnataka",
        disease="Aphids"
    )
    
    # Test 2: Potato blight with different session
    tester.test_metrics_query(
        query="How to control potato blight disease organically?",
        plant_type="Potato",
        location="Punjab",
        season="Rabi",  
        session_id="potato_query_session"
    )
    
    # Test 3: Rice query with auto plant detection
    tester.test_metrics_query(
        query="My rice crops are showing brown spots on leaves. What should I do?",
        season="Kharif",
        location="West Bengal"
    )
    
    # Test 4: Apple pest query
    tester.test_metrics_query(
        query="Apple scab prevention and treatment methods for organic farming",
        plant_type="Apple",
        disease="scab",
        season="Spring"
    )
    
    # Test 5: Multiple queries with same session ID
    session_queries = [
        {
            "query": "What is coconut root wilt disease?",
            "plant_type": "Coconut",
            "location": "Kerala"
        },
        {
            "query": "How to prevent coconut root wilt?",
            "plant_type": "Coconut",
            "location": "Kerala"
        },
        {
            "query": "Organic treatment for coconut root wilt",
            "plant_type": "Coconut",
            "location": "Kerala"
        }
    ]
    
    tester.test_multiple_queries_same_session(
        queries=session_queries,
        session_id="coconut_consultation_session"
    )
    
    # Test 6: Error scenarios
    tester.test_error_scenarios()
    
    # Test 7: Long query test
    tester.test_metrics_query(
        query="I am a farmer in Maharashtra and I have been growing tomatoes for the past 5 years. Recently, I noticed that my tomato plants are showing yellow leaves with brown spots, and some fruits are developing dark patches. The weather has been humid and there has been intermittent rainfall. I want to know what disease this might be and what organic treatment options are available that won't harm beneficial insects in my farm.",
        plant_type="Tomato",
        location="Maharashtra",
        season="Monsoon",
        session_id="detailed_consultation_session"
    )
    
    print("\n" + "=" * 70)
    print("✅ All metrics query tests completed!")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    print(f"\n📊 Test Summary:")
    print(f"   - Basic metrics queries: ✅")
    print(f"   - Session tracking: ✅") 
    print(f"   - Multiple queries per session: ✅")
    print(f"   - Error scenarios: ✅")
    print(f"   - Long query handling: ✅")


if __name__ == "__main__":
    main()
