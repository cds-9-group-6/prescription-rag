#!/bin/bash

# Quick test script for the /query/metrics endpoint
# Simple version for fast testing

API_URL="http://localhost:8081/query/metrics"

echo "🚀 Quick test of /query/metrics endpoint"
echo "🎯 URL: $API_URL"
echo ""

# Test with a simple tomato query
echo "📤 Testing with tomato disease query..."
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What causes tomato leaf curl?",
    "plant_type": "Tomato",
    "season": "Summer",
    "session_id": "quick-test-001"
  }' \
  | jq '.' 2>/dev/null || cat

echo -e "\n\n📤 Testing with auto plant detection..."
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My potato plants have black spots, help!",
    "location": "Punjab",
    "session_id": "quick-test-002"
  }' \
  | jq '.' 2>/dev/null || cat

echo -e "\n\n✅ Quick test completed!"

