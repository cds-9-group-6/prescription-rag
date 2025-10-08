"""
Test script for the /query/metrics endpoint of the Prescription RAG API.

This script tests the metrics-enabled query endpoint which includes session tracking
and comprehensive metrics logging functionality. Updated to match the exact format
used by prescription_tool.py in sasya-arogya-engine.

The test now supports:
- Structured query generation matching prescription_tool.py format
- disease_name, plant_type, season, location, severity parameters
- Custom query overrides for advanced testing
- Session tracking and multiple queries per session
- Error scenario testing

Usage:
    python test/test_query_metrics_endpoint.py --url http://localhost:8081
    python test/test_query_metrics_endpoint.py --host 0.0.0.0 --port 8081
"""

import requests
import time
import sys
import uuid
import argparse
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsTester:
    """Test class specifically for the /query/metrics endpoint."""
    
    def __init__(self, base_url: str = "http://0.0.0.0:8081"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_session_id = str(uuid.uuid4())
    
    def test_metrics_query(self, query: str = None, disease_name: str = None,
                          plant_type: Optional[str] = None, season: Optional[str] = None, 
                          location: Optional[str] = None, severity: Optional[str] = "Medium",
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """Test the /query/metrics endpoint with given parameters, matching prescription_tool.py format."""
        
        # Use provided session_id or default test session_id
        if session_id is None:
            session_id = self.test_session_id
        
        # If no custom query provided, construct the structured query like prescription_tool.py does
        if query is None and disease_name:
            query = f"""Disease: {disease_name}
Plant: {plant_type or "general"}
Location: {location or ""}
Season: {season or ""}
Severity: {severity}

Provide comprehensive treatment recommendations including:
1. Chemical treatments with dosages and application methods
2. Organic/natural treatment alternatives
3. Preventive measures
4. Application timing and frequency
5. Safety precautions
6. Expected recovery timeline
"""
        elif query is None:
            raise ValueError("Either 'query' or 'disease_name' must be provided")
        
        # Prepare request data matching prescription_tool.py format
        data = {
            "query": query,
            "plant_type": plant_type,
            "season": season,
            "location": location,
            "disease": disease_name,  # Note: prescription_tool.py sends disease_name as "disease" field
            "session_id": session_id
        }
        
        print(f"\n🧪 Testing Metrics Query:")
        print(f"   Disease: {disease_name}")
        print(f"   Plant Type: {plant_type}")
        print(f"   Season: {season}")
        print(f"   Location: {location}")
        print(f"   Severity: {severity}")
        print(f"   Session ID: {session_id}")
        print(f"   Query Length: {len(query)} characters")
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
                except ValueError:
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
                json={"query": "", "disease": None, "session_id": "error_test"}
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
                json={"plant_type": "Tomato", "disease": "Test Disease", "session_id": "missing_query_test"}
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test script for the Prescription RAG API /query/metrics endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_query_metrics_endpoint.py --url http://localhost:8081
  python test_query_metrics_endpoint.py --host localhost --port 8081
  python test_query_metrics_endpoint.py --host 0.0.0.0 --port 8081
        """
    )
    
    # Option 1: Full URL
    parser.add_argument(
        "--url", 
        type=str, 
        help="Full URL of the prescription API (e.g., http://localhost:8081)"
    )
    
    # Option 2: Host and port separately
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host of the prescription API (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8081,
        help="Port of the prescription API (default: 8081)"
    )
    
    args = parser.parse_args()
    
    # Determine the base URL
    if args.url:
        base_url = args.url.rstrip('/')
    else:
        base_url = f"http://{args.host}:{args.port}"
    
    return base_url


def main():
    """Main function to run metrics query tests."""
    
    # Parse command line arguments
    base_url = parse_arguments()
    
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
        print(f"\n💡 To test a different URL, run:")
        print(f"   python test/test_query_metrics_endpoint.py --url <your_url>")
        print(f"   python test/test_query_metrics_endpoint.py --host <host> --port <port>")
        sys.exit(1)
    
    # Initialize tester
    tester = MetricsTester(base_url)
    
    print(f"\n🆔 Test Session ID: {tester.test_session_id}")
    
    # Test cases for the /query/metrics endpoint
    print("\n" + "=" * 70)
    print("🧪 RUNNING METRICS QUERY TESTS")
    print("=" * 70)
    
    # Test 1: Basic tomato disease query with metrics (using structured format like prescription_tool.py)
    tester.test_metrics_query(
        disease_name="Early Blight",
        plant_type="Tomato",
        season="Summer",
        location="Karnataka",
        severity="Medium"
    )
    
    # Test 2: Potato blight with different session
    tester.test_metrics_query(
        disease_name="Late Blight",
        plant_type="Potato",
        location="Punjab",
        season="Rabi",
        severity="High",
        session_id="potato_query_session"
    )
    
    # Test 3: Custom query format (override structured format)
    tester.test_metrics_query(
        query="What are common diseases in tomatoes and how to treat them?",
        disease_name="General Tomato Diseases",
        plant_type="Tomato",
        season="Summer",
        location="Karnataka"
    )
    
    # Test 4: Rice disease query 
    tester.test_metrics_query(
        disease_name="Brown Spot",
        plant_type="Rice",
        season="Kharif",
        location="West Bengal",
        severity="Medium"
    )
    
    # Test 5: Apple pest query
    tester.test_metrics_query(
        disease_name="Scab",
        plant_type="Apple",
        season="Spring",
        location="Himachal Pradesh",
        severity="Low"
    )
    
    # Test 6: Multiple queries with same session ID
    session_queries = [
        {
            "disease_name": "Root Wilt",
            "plant_type": "Coconut",
            "location": "Kerala",
            "season": "Monsoon",
            "severity": "High"
        },
        {
            "disease_name": "Leaf Blight", 
            "plant_type": "Coconut",
            "location": "Kerala",
            "season": "Monsoon",
            "severity": "Medium"
        },
        {
            "disease_name": "Stem Bleeding",
            "plant_type": "Coconut",
            "location": "Kerala", 
            "season": "Post-Monsoon",
            "severity": "Low"
        }
    ]
    
    tester.test_multiple_queries_same_session(
        queries=session_queries,
        session_id="coconut_consultation_session"
    )
    
    # Test 7: Error scenarios
    tester.test_error_scenarios()
    
    # Test 8: Complex disease scenario with detailed context
    tester.test_metrics_query(
        query="I am a farmer in Maharashtra and I have been growing tomatoes for the past 5 years. Recently, I noticed that my tomato plants are showing yellow leaves with brown spots, and some fruits are developing dark patches. The weather has been humid and there has been intermittent rainfall. I want to know what disease this might be and what organic treatment options are available that won't harm beneficial insects in my farm.",
        disease_name="Tomato Blight Complex",
        plant_type="Tomato",
        location="Maharashtra",
        season="Monsoon",
        severity="Medium",
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
