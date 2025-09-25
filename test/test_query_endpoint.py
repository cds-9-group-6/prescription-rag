"""
Simple test script for the /query endpoint of the Prescription RAG API.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional


class QueryTester:
    """Simple test class for the /query endpoint."""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def test_query(self, query: str, plant_type: Optional[str] = None, 
                   season: Optional[str] = None, location: Optional[str] = None, 
                   disease: Optional[str] = None) -> Dict[str, Any]:
        """Test the /query endpoint with given parameters."""
        
        # Prepare request data
        data = {"query": query}
        if plant_type:
            data["plant_type"] = plant_type
        if season:
            data["season"] = season
        if location:
            data["location"] = location
        if disease:
            data["disease"] = disease
        
        print(f"\n🧪 Testing Query:")
        print(f"   Query: '{query}'")
        if plant_type:
            print(f"   Plant Type: {plant_type}")
        if season:
            print(f"   Season: {season}")
        if location:
            print(f"   Location: {location}")
        if disease:
            print(f"   Disease: {disease}")
        print("-" * 50)
        
        try:
            url = f"{self.base_url}/query"
            start_time = time.time()
            
            response = self.session.post(url, json=data, timeout=30)
            
            request_time = time.time() - start_time
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Request Time: {request_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('success', False)}")
                print(f"   Collection Used: {result.get('collection_used', 'N/A')}")
                print(f"   Query Time: {result.get('query_time', 0):.2f}s")
                print("\n📝 Answer:")
                print(f"   {result.get('answer', 'No answer received')}")
                print("\n✅ SUCCESS")
                return result
            else:
                print(f"   Error Response: {response.text}")
                print("\n❌ FAILED")
                return {"error": response.text, "status_code": response.status_code}
                
        except Exception as e:
            print(f"   Exception: {e}")
            print("\n❌ FAILED")
            return {"error": str(e), "exception": True}


def check_server_health(base_url: str) -> bool:
    """Check if the API server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy")
            print(f"   Available Collections: {health_data.get('available_collections', [])}")
            print(f"   Total Collections: {health_data.get('total_collections', 0)}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False


def main():
    """Main function to run query tests."""
    
    # Configuration
    base_url = "http://localhost:8081"  # Default from start_server.py
    
    print("🚀 Prescription RAG API - Query Endpoint Tester")
    print(f"   Target URL: {base_url}")
    print("=" * 60)
    
    # Check server health first
    print("\n🔍 Checking server health...")
    if not check_server_health(base_url):
        print("\n💡 To start the server, run:")
        print("   python api/start_server.py")
        print("   # or")
        print("   ./run_api.sh")
        sys.exit(1)
    
    # Initialize tester
    tester = QueryTester(base_url)
    
    # Test cases for the /query endpoint
    print("\n" + "=" * 60)
    print("🧪 RUNNING QUERY TESTS")
    print("=" * 60)
    
    # Test 1: Basic tomato disease query
    tester.test_query(
        query="What are common diseases in tomatoes and how to treat them?",
        plant_type="Tomato",
        season="Summer",
        location="Karnataka",
        # disease="Bacterial Leaf Spot and Aphids"
        disease="Aphids"
    )
    
    tester.test_query(
        query="What are common diseases in tomatoes and how to treat them?",
        plant_type="Tomato",
        season="Summer",
        # location="Karnataka",
        # # disease="Bacterial Leaf Spot and Aphids"
        # disease="Aphids"
    )

    tester.test_query(
        query="What are common diseases in tomatoes and how to treat them?",
        plant_type="Tomato",
        season="Summer",
        location="Karnataka",
        # disease="Bacterial Leaf Spot and Aphids"
        # disease="Aphids"
    )


    # # Test 2: Potato pest management
    # tester.test_query(
    #     query="How to control potato blight disease?",
    #     plant_type="Potato",
    #     location="Punjab"
    # )
    
    # # Test 3: Rice with auto plant detection
    # tester.test_query(
    #     query="My rice crops are showing brown spots on leaves. What should I do?",
    #     season="Kharif"
    # )
    
    # # Test 4: Apple pest query
    # tester.test_query(
    #     query="Apple scab prevention and treatment methods",
    #     plant_type="Apple",
    #     disease="scab"
    # )
    
    # # Test 5: Coconut disease with location
    # tester.test_query(
    #     query="Coconut palm diseases in coastal areas",
    #     plant_type="Coconut",
    #     location="Kerala",
    #     season="Monsoon"
    # )
    
    # # Test 6: General agricultural query
    # tester.test_query(
    #     query="Organic farming pest management strategies"
    # )
    
    print("\n" + "=" * 60)
    print("✅ All query tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()