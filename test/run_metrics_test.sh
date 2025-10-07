#!/bin/bash

# Shell script to run the Python-based metrics endpoint test
# This script provides a convenient way to run test_query_metrics_endpoint.py

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_query_metrics_endpoint.py"
API_BASE_URL="http://0.0.0.0:8081"

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script runs the Python-based test suite for the /query/metrics endpoint."
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -u, --url URL       Use custom API base URL (default: $API_BASE_URL)"
    echo "  -p, --python PATH   Use specific Python interpreter"
    echo ""
    echo "Examples:"
    echo "  $0                  Run tests with default settings"
    echo "  $0 -u http://0.0.0.0:8080  Use different port"
    echo "  $0 -p python3.9     Use specific Python version"
    echo ""
    echo "Note: This runs test_query_metrics_endpoint.py in the current directory"
}

# Check if Python test file exists
check_test_file() {
    if [ ! -f "$TEST_SCRIPT" ]; then
        print_error "Test script not found: $TEST_SCRIPT"
        echo "Expected location: test_query_metrics_endpoint.py (in test directory)"
        exit 1
    fi
}

# Check Python installation
check_python() {
    if ! command -v "$PYTHON_CMD" > /dev/null 2>&1; then
        print_error "Python interpreter not found: $PYTHON_CMD"
        echo "Please install Python or specify a different interpreter with -p"
        exit 1
    fi
    
    # Check if required packages are available
    if ! "$PYTHON_CMD" -c "import requests" 2>/dev/null; then
        print_warning "requests package not found. Installing..."
        "$PYTHON_CMD" -m pip install requests
    fi
}

# Main function
main() {
    print_header "Python-based Metrics Endpoint Test Runner"
    echo "🎯 Target URL: $API_BASE_URL"
    echo "🐍 Python: $PYTHON_CMD"
    echo "📄 Test Script: $TEST_SCRIPT"
    
    # Checks
    check_test_file
    check_python
    
    print_header "Running Tests"
    
    # Set environment variable for the test script if needed
    export API_BASE_URL="$API_BASE_URL"
    
    # Run the Python test script
    if "$PYTHON_CMD" "$TEST_SCRIPT"; then
        print_success "Test execution completed"
    else
        print_error "Test execution failed"
        exit 1
    fi
}

# Default values
PYTHON_CMD="python3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -u|--url)
            API_BASE_URL="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run main function
main

echo -e "\n🎉 Test runner completed!"
echo ""
echo "💡 Tips:"
echo "   - Use the existing ../test_query_metrics.sh for curl-based testing"
echo "   - Use this script (test/run_metrics_test.sh) for Python-based testing"  
echo "   - Both test the same /query/metrics endpoint with different approaches"
