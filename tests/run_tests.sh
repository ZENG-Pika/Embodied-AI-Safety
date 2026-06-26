#!/bin/bash
# Integration test runner for SimBox DataEngine

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="tests/test_results_${TIMESTAMP}.log"
SUMMARY_FILE="tests/test_summary_${TIMESTAMP}.txt"
TEMP_LOG="tests/temp_test_output.log"

declare -a TEST_SUITES=()
TEST_SUITES+=("SimBox Tests (Isaac Sim):3:/isaac-sim/python.sh tests/integration/simbox/test_simbox.py")

TOTAL_SUITES=${#TEST_SUITES[@]}

echo "Starting SimBox DataEngine Integration Tests..."
echo "=============================================="
echo -e "${BOLD}${CYAN}TEST EXECUTION PLAN:${NC}"
echo -e "  ${BOLD}Total Test Suites: ${TOTAL_SUITES}${NC}"
echo -e "  ${BOLD}SimBox Scenarios Covered:${NC}"
echo -e "    - Pipeline: Full end-to-end workflow"
echo -e "    - Plan: Trajectory planning generation"
echo -e "    - Render: Scene rendering with validation"
echo ""

echo -e "${BLUE}Detailed logs: $LOG_FILE${NC}"
echo "=============================================="

TOTAL_TEST_SUITES=0
PASSED_TEST_SUITES=0
FAILED_TEST_SUITES=0

run_test_suite() {
    local suite_name="$1"
    local expected_sessions="$2"
    local test_command="$3"
    local current_suite=$4

    TOTAL_TEST_SUITES=$((TOTAL_TEST_SUITES + 1))

    echo -e "${BOLD}${BLUE}[$current_suite/$TOTAL_SUITES] Starting: $suite_name${NC}"
    echo -e "  ${CYAN}-> Running ${expected_sessions} SimBox tests with Isaac Sim${NC}"

    echo "Test Suite: $suite_name" >> "$LOG_FILE"
    echo "Expected Test Functions: $expected_sessions" >> "$LOG_FILE"
    echo "Command: $test_command" >> "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"

    echo -e "${CYAN}Executing: $test_command${NC}"
    eval "$test_command" > "$TEMP_LOG" 2>&1
    local exit_status=$?

    cat "$TEMP_LOG" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    if [ $exit_status -eq 0 ]; then
        echo -e "${GREEN}✓ [$current_suite/$TOTAL_SUITES] PASSED: $suite_name${NC}"
        PASSED_TEST_SUITES=$((PASSED_TEST_SUITES + 1))
    else
        echo -e "${RED}✗ [$current_suite/$TOTAL_SUITES] FAILED: $suite_name${NC}"
        FAILED_TEST_SUITES=$((FAILED_TEST_SUITES + 1))
        echo -e "${YELLOW}  Check $LOG_FILE for error details${NC}"
        echo -e "${YELLOW}Recent output:${NC}"
        tail -20 "$TEMP_LOG"
    fi

    echo ""
    rm -f "$TEMP_LOG"
}

> "$SUMMARY_FILE"
> "$LOG_FILE"

echo "Test execution started at: $(date)" > "$SUMMARY_FILE"
echo "Planned suites: $TOTAL_SUITES" >> "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"

echo "Test execution started at: $(date)" >> "$LOG_FILE"
echo "================================================" >> "$LOG_FILE"

for i in "${!TEST_SUITES[@]}"; do
    suite_info="${TEST_SUITES[$i]}"
    suite_name=$(echo "$suite_info" | cut -d':' -f1)
    expected_sessions=$(echo "$suite_info" | cut -d':' -f2)
    test_command=$(echo "$suite_info" | cut -d':' -f3-)
    current_suite=$((i + 1))

    run_test_suite "$suite_name" "$expected_sessions" "$test_command" $current_suite
done

echo "=============================================="
echo "TEST EXECUTION SUMMARY"
echo "=============================================="
echo -e "${CYAN}Test Suites:${NC}"
echo -e "  Total: $TOTAL_TEST_SUITES"
echo -e "  ${GREEN}Passed: $PASSED_TEST_SUITES${NC}"
echo -e "  ${RED}Failed: $FAILED_TEST_SUITES${NC}"
echo ""

echo "" >> "$SUMMARY_FILE"
echo "FINAL SUMMARY:" >> "$SUMMARY_FILE"
echo "Suites - Total: $TOTAL_TEST_SUITES, Passed: $PASSED_TEST_SUITES, Failed: $FAILED_TEST_SUITES" >> "$SUMMARY_FILE"

if [ $FAILED_TEST_SUITES -eq 0 ]; then
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    echo "ALL TESTS PASSED at $(date)" >> "$SUMMARY_FILE"
else
    echo -e "${RED}SOME TEST SUITES FAILED${NC}"
    echo "SOME TEST SUITES FAILED at $(date)" >> "$SUMMARY_FILE"
fi

echo ""
echo -e "${BLUE}Results: $SUMMARY_FILE${NC}"
echo -e "${BLUE}Logs: $LOG_FILE${NC}"

if [ $FAILED_TEST_SUITES -eq 0 ]; then
    exit 0
else
    exit 1
fi
