#!/bin/bash

# Shell script to test the /query/metrics endpoint
# This script tests various scenarios for the metrics endpoint

# Configuration
API_BASE_URL="http://localhost:8081"
ENDPOINT="/query/metrics"
FULL_URL="${API_BASE_URL}${ENDPOINT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check if API server is running
check_server() {
    echo "🔍 Checking if API server is running at ${API_BASE_URL}..."
    
    if curl -s -f "${API_BASE_URL}/health" > /dev/null 2>&1; then
        print_success "API server is running"
        return 0
    else
        print_error "API server is not responding at ${API_BASE_URL}"
        echo -e "\nTo start the server:"
        echo "   ./run_api.sh"
        echo "   # or"
        echo "   python -m api.start_server"
        exit 1
    fi
}

# Function to test an endpoint
test_query_metrics() {
    local test_name="$1"
    local json_data="$2"
    local session_id="$3"
    
    print_header "Test: $test_name"
    echo "📤 Request Data:"
    echo "$json_data" | jq '.' 2>/dev/null || echo "$json_data"
    echo ""
    echo "🌐 Sending POST request to: $FULL_URL"
    
    # Make the request and capture response
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_data" \
        "$FULL_URL")
    
    # Extract HTTP status code and response body
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    echo "📥 HTTP Status: $http_code"
    
    if [ "$http_code" = "200" ]; then
        print_success "Request successful"
        echo "📊 Response:"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
        
        # Extract key metrics from response
        if command -v jq > /dev/null 2>&1; then
            answer_length=$(echo "$response_body" | jq -r '.answer | length' 2>/dev/null)
            query_time=$(echo "$response_body" | jq -r '.query_time' 2>/dev/null)
            collection_used=$(echo "$response_body" | jq -r '.collection_used' 2>/dev/null)
            
            echo ""
            echo "📈 Key Metrics:"
            echo "   Collection Used: $collection_used"
            echo "   Query Time: ${query_time}s"
            echo "   Answer Length: $answer_length characters"
            echo "   Session ID: $session_id"
        fi
    else
        print_error "Request failed"
        echo "📥 Response:"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    fi
    
    echo -e "\n" + "-"*60
}

# Main test execution
main() {
    echo "🚀 Starting /query/metrics Endpoint Tests"
    echo "🎯 Target URL: $FULL_URL"
    
    # Check if server is running
    check_server
    
    # Check if jq is available for JSON formatting
    if ! command -v jq > /dev/null 2>&1; then
        print_warning "jq not found. Install it for better JSON formatting: brew install jq"
    fi
    
    print_header "Starting Tests"
    
    # Test 1: Basic tomato disease query
    test_query_metrics "Basic Tomato Disease Query" '{
        "query": "What are common diseases in tomatoes during summer?",
        "plant_type": "Tomato",
        "season": "Summer",
        "session_id": "test-session-001"
    }' "test-session-001"
    
    # Test 2: Auto plant type detection
    test_query_metrics "Auto Plant Type Detection" '{
        "query": "My potato plants have black spots on leaves, what should I do?",
        "season": "Kharif",
        "location": "Punjab",
        "session_id": "test-session-002"
    }' "test-session-002"
    
    # Test 3: Metadata filtering with disease
    test_query_metrics "Metadata Filtering with Disease" '{
        "query": "How to control blight effectively?",
        "plant_type": "Potato",
        "season": "Winter",
        "location": "Maharashtra",
        "disease": "Blight",
        "session_id": "test-session-003"
    }' "test-session-003"
    
    # Test 4: Rice farming query
    test_query_metrics "Rice Farming Query" '{
        "query": "Best practices for rice cultivation during monsoon season",
        "plant_type": "Rice",
        "season": "Kharif",
        "location": "West Bengal",
        "session_id": "test-session-004"
    }' "test-session-004"
    
    # Test 5: General agricultural query
    test_query_metrics "General Agricultural Query" '{
        "query": "Organic pest control methods for vegetables",
        "season": "Summer",
        "session_id": "test-session-005"
    }' "test-session-005"
    
    # Test 6: Complex query with all filters
    test_query_metrics "Complex Query with All Filters" '{
        "query": "What fungicides are recommended for powdery mildew in tomatoes?",
        "plant_type": "Tomato",
        "season": "Summer",
        "location": "Karnataka",
        "disease": "Powdery_mildew",
        "session_id": "test-session-006"
    }' "test-session-006"
    
    # Test 7: Minimal query (just query text)
    test_query_metrics "Minimal Query" '{
        "query": "How to improve soil fertility?"
    }' "default"
    
    # Test 8: Performance test with longer query
    test_query_metrics "Performance Test - Longer Query" '{
        "query": "I am a farmer from Tamil Nadu and I am facing severe issues with my tomato crop. The leaves are turning yellow and there are brown spots appearing. This started happening after the recent rains. What could be the cause and how can I treat this problem? Also suggest preventive measures for future.",
        "plant_type": "Tomato",
        "season": "Monsoon",
        "location": "Tamil Nadu",
        "session_id": "test-session-performance"
    }' "test-session-performance"
    
    # Test 9: Test structured treatment endpoint
    print_header "Testing Structured Treatment Endpoint"
    echo "🧪 Testing POST /query/treatment for structured JSON response..."
    
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{
            "query": "My tomato plants have blight, give me detailed treatment plan",
            "plant_type": "Tomato",
            "season": "Summer"
        }' \
        "${API_BASE_URL}/query/treatment")
    
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    echo "📥 HTTP Status: $http_code"
    
    if [ "$http_code" = "200" ]; then
        print_success "Structured treatment endpoint working"
        if command -v jq > /dev/null 2>&1; then
            parsing_success=$(echo "$response_body" | jq -r '.parsing_success' 2>/dev/null)
            treatment_available=$(echo "$response_body" | jq -r '.treatment != null' 2>/dev/null)
            echo "📊 Parsing Success: $parsing_success"
            echo "📋 Treatment Data Available: $treatment_available"
            
            if [ "$treatment_available" = "true" ]; then
                disease_name=$(echo "$response_body" | jq -r '.treatment.diagnosis.disease_name' 2>/dev/null)
                primary_medicine=$(echo "$response_body" | jq -r '.treatment.medicine_recommendations.primary_treatment.medicine_name' 2>/dev/null)
                echo "🔍 Disease: $disease_name"
                echo "💊 Primary Medicine: $primary_medicine"
            fi
        fi
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    else
        print_error "Structured treatment endpoint failed"
        echo "$response_body"
    fi
    
    print_header "Test Summary"
    print_success "All tests completed!"
    echo "📊 Check the API server logs for detailed metrics information"
    echo "🎯 Endpoint tested: $FULL_URL"
    echo ""
    echo "💡 Tips:"
    echo "   - Check query_time values to monitor performance"
    echo "   - Verify collection_used matches expected plant types"
    echo "   - Session IDs help track individual queries"
    echo "   - Different filters test metadata search capabilities"
}

# Function to test error scenarios
test_error_scenarios() {
    print_header "Testing Error Scenarios"
    
    # Test with empty query
    test_query_metrics "Empty Query Test" '{
        "query": "",
        "session_id": "error-test-001"
    }' "error-test-001"
    
    # Test with invalid JSON
    echo -e "\n${BLUE}=== Test: Invalid JSON ===${NC}"
    echo "📤 Sending invalid JSON..."
    
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "test", invalid}' \
        "$FULL_URL")
    
    http_code=$(echo "$response" | tail -n1)
    echo "📥 HTTP Status: $http_code"
    
    if [ "$http_code" != "200" ]; then
        print_success "Correctly handled invalid JSON (status: $http_code)"
    else
        print_error "Should have failed with invalid JSON"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -e, --errors        Also test error scenarios"  
    echo "  -u, --url URL       Use custom API base URL (default: $API_BASE_URL)"
    echo ""
    echo "Examples:"
    echo "  $0                  Run basic tests"
    echo "  $0 --errors         Run basic tests + error scenarios"
    echo "  $0 -u http://localhost:8080  Use different port"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -e|--errors)
            TEST_ERRORS=true
            shift
            ;;
        -u|--url)
            API_BASE_URL="$2"
            FULL_URL="${API_BASE_URL}${ENDPOINT}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run tests
main

# Run error tests if requested
if [ "$TEST_ERRORS" = true ]; then
    test_error_scenarios
fi

echo -e "\n🎉 Testing completed!"

