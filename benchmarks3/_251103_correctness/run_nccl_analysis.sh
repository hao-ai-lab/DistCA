#!/bin/bash

# Script to analyze NCCL AllGather kernel timing from nsys profile files
# This script provides an easy way to run the analysis with different options

CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="logs.v0-big"
KERNEL_NAME="ncclDevKernel_AllGather_RING_LL"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}NCCL Kernel Timing Analysis${NC}"
echo -e "${BLUE}===========================${NC}"

cd "$CURDIR"

# Check if logs directory exists
if [ ! -d "$LOGS_DIR" ]; then
    echo -e "${RED}Error: Directory $LOGS_DIR not found!${NC}"
    echo "Available directories:"
    ls -la | grep "^d"
    exit 1
fi

# Check for nsys files
NSYS_COUNT=$(find "$LOGS_DIR" -name "*.nsys-rep" | wc -l)
if [ "$NSYS_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No .nsys-rep files found in $LOGS_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found $NSYS_COUNT nsys profile files${NC}"

# Check if nsys command is available
if ! command -v nsys &> /dev/null; then
    echo -e "${YELLOW}Warning: nsys command not found. Analysis may fail.${NC}"
    echo "Please make sure NVIDIA Nsight Systems is installed and in PATH"
fi

echo -e "\n${BLUE}Running simple analysis...${NC}"
echo "=========================================="

# Run the simple analyzer
python3 simple_nccl_analyzer.py --logs-dir "$LOGS_DIR" --kernel-name "$KERNEL_NAME"

SIMPLE_EXIT_CODE=$?

echo -e "\n${BLUE}Analysis Summary${NC}"
echo "================"
echo "Logs directory: $CURDIR/$LOGS_DIR"
echo "Kernel searched: $KERNEL_NAME"
echo "nsys files analyzed: $NSYS_COUNT"
echo "Simple analyzer exit code: $SIMPLE_EXIT_CODE"

if [ $SIMPLE_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Simple analysis completed successfully!${NC}"
else
    echo -e "${YELLOW}Simple analysis completed with warnings/errors${NC}"
fi

echo -e "\n${BLUE}Advanced Analysis Options:${NC}"
echo "To run the full-featured analyzer:"
echo "  python3 analyze_nccl_kernels.py --logs-dir $LOGS_DIR --kernel-name '$KERNEL_NAME'"
echo ""
echo "To save results to files:"
echo "  python3 analyze_nccl_kernels.py --output-csv results.csv --output-report report.txt"
echo ""
echo "To analyze different kernel:"
echo "  python3 simple_nccl_analyzer.py --kernel-name 'your_kernel_name'"

# Optional: Try to run a quick test with one file
echo -e "\n${BLUE}Testing with one file...${NC}"
FIRST_NSYS_FILE=$(find "$LOGS_DIR" -name "*.nsys-rep" | head -1)
if [ -n "$FIRST_NSYS_FILE" ]; then
    echo "Testing nsys stats on: $FIRST_NSYS_FILE"
    if command -v nsys &> /dev/null; then
        nsys stats --help 2>/dev/null | head -5
        echo ""
        echo "Running: nsys stats '$FIRST_NSYS_FILE' | grep -i allgather | head -3"
        nsys stats "$FIRST_NSYS_FILE" 2>/dev/null | grep -i allgather | head -3
    fi
fi

echo -e "\n${GREEN}Analysis script completed!${NC}"







