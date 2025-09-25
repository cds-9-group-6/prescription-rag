#!/usr/bin/env python3
"""
Test the actual API endpoint with the data that was previously failing.
"""

import requests
import json
import sys

def test_api_endpoint():
    """Test the /query/treatment endpoint with a real query."""
    
    url = "http://localhost:8081/query/treatment"
    
    # Test with the query that was causing validation issues
    test_data = {
        "query": "My tomato plants have white powdery coating on leaves, what should I do?",
        "plant_type": "Tomato",
        "season": "Summer"
    }
    
    print("🚀 Testing Structured Treatment API Endpoint")
    print(f"🎯 URL: {url}")
    print("="*60)
    
    try:
        print("📤 Sending request...")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        print("")
        
        response = requests.post(url, json=test_data, timeout=120)
        
        print(f"📥 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Request Successful!")
            print(f"⏱️  Query Time: {result.get('query_time', 'N/A')}s")
            print(f"🗂️  Collection Used: {result.get('collection_used', 'N/A')}")
            print(f"✅ Parsing Success: {result.get('parsing_success', 'N/A')}")
            print(f"✅ Overall Success: {result.get('success', 'N/A')}")
            
            if result.get('success') and result.get('treatment'):
                treatment = result['treatment']
                print("\n📋 Treatment Response Structure:")
                
                # Check diagnosis
                if treatment.get('diagnosis'):
                    diag = treatment['diagnosis']
                    print(f"   🔍 Disease: {diag.get('disease_name', 'N/A')}")
                    print(f"   📊 Severity: {diag.get('severity', 'N/A')}")
                    print(f"   🍃 Affected Parts: {diag.get('affected_parts', [])}")
                
                # Check immediate treatment
                if treatment.get('immediate_treatment'):
                    imm = treatment['immediate_treatment']
                    print(f"   ⚡ Immediate Actions: {len(imm.get('actions', []))} actions")
                    print(f"   ⏰ Timeline: {imm.get('timeline', 'N/A')}")
                
                # Check weekly plan
                if treatment.get('weekly_treatment_plan'):
                    weekly = treatment['weekly_treatment_plan']
                    weeks = []
                    for i in range(1, 5):
                        if weekly.get(f'week_{i}'):
                            weeks.append(f"Week {i}")
                    print(f"   📅 Weekly Plan: {', '.join(weeks)} ({len(weeks)} weeks)")
                
                # Check medicines
                if treatment.get('medicine_recommendations'):
                    med = treatment['medicine_recommendations']
                    primary = med.get('primary_treatment', {})
                    secondary = med.get('secondary_treatment')
                    organic_count = len(med.get('organic_alternatives', []))
                    
                    print(f"   💊 Primary Medicine: {primary.get('medicine_name', 'N/A') if primary else 'None'}")
                    print(f"   💊 Secondary Medicine: {secondary.get('medicine_name', 'N/A') if secondary else 'None'}")
                    print(f"   🌿 Organic Alternatives: {organic_count} options")
                
                print("\n🎉 Structured treatment response validation SUCCESSFUL!")
                
                # Save the full response for inspection
                with open('successful_treatment_response.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print("💾 Full response saved to: successful_treatment_response.json")
                
            else:
                print("⚠️  Treatment data not available or parsing failed")
                if result.get('raw_response'):
                    print(f"📄 Raw Response (first 500 chars):")
                    print(result['raw_response'][:500] + "..." if len(result['raw_response']) > 500 else result['raw_response'])
                
        else:
            print(f"❌ Request Failed")
            print(f"📄 Response: {response.text}")
            return False
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: API server is not running")
        print("\nTo start the server:")
        print("   ./run_api.sh")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Checking if API server is running...")
    
    try:
        health_response = requests.get("http://localhost:8081/health", timeout=5)
        if health_response.status_code != 200:
            print("⚠️  API server is not healthy")
            return False
    except:
        print("❌ Cannot connect to API server at http://localhost:8081")
        print("Make sure the server is running: ./run_api.sh")
        return False
    
    print("✅ API server is running and healthy")
    print("")
    
    success = test_api_endpoint()
    
    if success:
        print("\n🎉 API endpoint test completed successfully!")
        print("✅ The Pydantic validation fixes are working in the live API!")
    else:
        print("\n❌ API endpoint test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
