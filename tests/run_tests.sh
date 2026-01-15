#!/bin/bash
# Test runner script for Feature-SST tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default test category
TEST_CATEGORY="${1:-all}"

echo -e "${GREEN}Running Feature-SST tests: ${TEST_CATEGORY}${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install pytest"
    exit 1
fi

# Function to run tests
run_tests() {
    local test_file="$1"
    local test_name="$2"
    
    echo -e "\n${YELLOW}Running ${test_name}...${NC}"
    pytest "$test_file" -v --tb=short
}

# Run tests based on category
case "$TEST_CATEGORY" in
    all)
        echo "Running all Feature-SST tests..."
        pytest test_feature_sst.py test_sst_cuda.py test_sst_validation.py -v --tb=short
        ;;
    unit|main)
        run_tests "test_feature_sst.py" "main Feature-SST tests"
        ;;
    cuda)
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo -e "${YELLOW}Warning: CUDA not available, skipping CUDA tests${NC}"
            exit 0
        fi
        run_tests "test_sst_cuda.py" "CUDA-specific tests"
        ;;
    validation)
        run_tests "test_sst_validation.py" "validation tests"
        ;;
    forward)
        pytest test_feature_sst.py -k "forward" -v --tb=short
        ;;
    backward)
        pytest test_feature_sst.py -k "backward" -v --tb=short
        ;;
    performance)
        pytest test_feature_sst.py::TestPerformanceBenchmarks -v --tb=short
        ;;
    *)
        echo -e "${RED}Unknown test category: ${TEST_CATEGORY}${NC}"
        echo "Usage: $0 [all|unit|main|cuda|validation|forward|backward|performance]"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Tests completed!${NC}"
