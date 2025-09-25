#!/usr/bin/env python3
"""
Test script for the structured treatment endpoint.

This script tests the new /query/treatment endpoint to ensure it returns
properly structured JSON responses with treatment recommendations.
"""

import requests
import json
import sys


def test_structured_treatment(base_url="http://localhost:8081"):
    """Test the structured treatment endpoint."""
    
    endpoint = f"{base_url}/query/treatment"
    
    # Test cases for different scenarios
    test_cases = [
        {
            "name": "Tomato Blight Treatment",
            "data": {
                "query": "My tomato plants have brown spots on leaves and they are wilting. What should I do?",
                "plant_type": "Tomato",
                "season": "Summer",
                "location": "Maharashtra",
                "disease": "Blight"
            }
        },
        {
            "name": "Potato Disease Auto-Detection",
            "data": {
                "query": "My potato plants have black spots and the leaves are turning yellow. The plants look sick.",
                "season": "Kharif",
                "location": "Punjab"
            }
        },
        {
            "name": "Rice Pest Management",
            "data": {
                "query": "Rice plants are being attacked by brown hoppers and the leaves are drying. Need immediate help.",
                "plant_type": "Rice",
                "season": "Kharif",
                "location": "West Bengal"
            }
        }
    ]
    
    print("🚀 Testing Structured Treatment Endpoint")
    print(f"🎯 URL: {endpoint}")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Make the request
            response = requests.post(endpoint, json=test_case['data'], timeout=60)
            
            print(f"📤 Request: {json.dumps(test_case['data'], indent=2)}")
            print(f"📥 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Request Successful")
                print(f"⏱️  Query Time: {result.get('query_time', 'N/A')}s")
                print(f"🗂️  Collection Used: {result.get('collection_used', 'N/A')}")
                print(f"✅ Parsing Success: {result.get('parsing_success', 'N/A')}")
                print(f"✅ Overall Success: {result.get('success', 'N/A')}")
                
                if result.get('success') and result.get('treatment'):
                    treatment = result['treatment']
                    
                    print("\n📋 Treatment Structure:")
                    print(f"   🔍 Diagnosis: {treatment.get('diagnosis', {}).get('disease_name', 'N/A')}")
                    print(f"   ⚡ Immediate Actions: {len(treatment.get('immediate_treatment', {}).get('actions', []))} actions")
                    print(f"   📅 Weekly Plan: {len([k for k in treatment.get('weekly_treatment_plan', {}).keys() if k.startswith('week_')])} weeks")
                    print(f"   💊 Primary Medicine: {treatment.get('medicine_recommendations', {}).get('primary_treatment', {}).get('medicine_name', 'N/A')}")
                    
                    # Show a sample of the structured data
                    print(f"\n📝 Sample Treatment Data:")
                    if 'diagnosis' in treatment:
                        print(f"   Disease: {treatment['diagnosis'].get('disease_name', 'N/A')}")
                        print(f"   Severity: {treatment['diagnosis'].get('severity', 'N/A')}")
                    
                    if 'immediate_treatment' in treatment:
                        actions = treatment['immediate_treatment'].get('actions', [])
                        print(f"   First Action: {actions[0] if actions else 'N/A'}")
                        
                    if 'medicine_recommendations' in treatment:
                        primary = treatment['medicine_recommendations'].get('primary_treatment', {})
                        if primary:
                            print(f"   Primary Medicine: {primary.get('medicine_name', 'N/A')}")
                            print(f"   Dosage: {primary.get('dosage', 'N/A')}")
                
                else:
                    print("⚠️  Treatment parsing failed or no treatment data")
                    if result.get('raw_response'):
                        print(f"📄 Raw Response (first 200 chars): {result['raw_response'][:200]}...")
                
                # Save full response for detailed inspection
                filename = f"treatment_response_{i}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Full response saved to: {filename}")
                
            else:
                print(f"❌ Request Failed")
                print(f"📄 Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n📊 Test Summary")
    print("✅ All structured treatment tests completed!")
    print("💡 Check the saved JSON files for detailed responses")
    print("\n🔍 To inspect the JSON structure:")
    print("   cat treatment_response_*.json | jq '.'")


def main():
    """Main function."""
    base_url = "http://localhost:8081"
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API server is not healthy at {base_url}")
            sys.exit(1)
    except:
        print(f"❌ Cannot connect to API server at {base_url}")
        print("Make sure the server is running: ./run_api.sh")
        sys.exit(1)
    
    print("✅ API server is running")
    test_structured_treatment(base_url)


if __name__ == "__main__":
    main()
