#!/bin/bash
# Test runner script for Stability Enforcement tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running Stability Enforcement Test Suite${NC}"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install pytest"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:--v}"

case "$TEST_TYPE" in
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        pytest tests/test_feature_stab_enforce.py tests/test_stab_enforce_cuda.py tests/test_stab_enforce_validation.py $VERBOSE
        ;;
    unit)
        echo -e "${YELLOW}Running unit tests...${NC}"
        pytest tests/test_feature_stab_enforce.py::TestStabilityEnforcementFunctions $VERBOSE
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        pytest tests/test_feature_stab_enforce.py::TestMambaStabilityEnforcement $VERBOSE
        ;;
    cuda)
        echo -e "${YELLOW}Running CUDA tests...${NC}"
        pytest tests/test_stab_enforce_cuda.py -m cuda $VERBOSE
        ;;
    validation)
        echo -e "${YELLOW}Running validation tests...${NC}"
        pytest tests/test_stab_enforce_validation.py $VERBOSE
        ;;
    performance)
        echo -e "${YELLOW}Running performance benchmarks...${NC}"
        pytest tests/test_feature_stab_enforce.py::TestPerformanceBenchmarks $VERBOSE
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [all|unit|integration|cuda|validation|performance] [verbose]"
        exit 1
        ;;
esac

echo -e "${GREEN}Tests completed!${NC}"
