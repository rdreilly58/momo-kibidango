#!/bin/bash
# Health check script for Speculative Decoding service

set -euo pipefail

# Configuration
API_URL="${API_URL:-http://localhost:5000}"
METRICS_URL="${METRICS_URL:-http://localhost:8080/metrics}"
TIMEOUT=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Exit codes
EXIT_SUCCESS=0
EXIT_WARNING=1
EXIT_CRITICAL=2

print_status() {
    echo -e "${YELLOW}[CHECK] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

# Track overall status
OVERALL_STATUS=$EXIT_SUCCESS

# Function to update overall status
update_status() {
    if [ $1 -gt $OVERALL_STATUS ]; then
        OVERALL_STATUS=$1
    fi
}

echo "=== Speculative Decoding Health Check ==="
echo "Timestamp: $(date)"
echo "API URL: $API_URL"
echo ""

# 1. Check if service is running
print_status "Checking service status..."
if systemctl is-active --quiet speculative-decoding; then
    print_success "Service is running"
else
    print_error "Service is not running"
    update_status $EXIT_CRITICAL
    systemctl status speculative-decoding --no-pager || true
fi

# 2. Check API health endpoint
print_status "Checking API health..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$API_URL/health" 2>/dev/null || echo "000")

if [ "$HTTP_STATUS" = "200" ]; then
    print_success "API health check passed"
else
    print_error "API health check failed (HTTP $HTTP_STATUS)"
    update_status $EXIT_CRITICAL
fi

# 3. Check API readiness
print_status "Checking API readiness..."
READY_RESPONSE=$(curl -s --max-time $TIMEOUT "$API_URL/ready" 2>/dev/null || echo "{}")
READY_STATUS=$(echo "$READY_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$READY_STATUS" = "ready" ]; then
    print_success "API is ready"
else
    print_warning "API not ready: $READY_STATUS"
    update_status $EXIT_WARNING
    echo "$READY_RESPONSE" | jq . 2>/dev/null || echo "$READY_RESPONSE"
fi

# 4. Check system metrics
print_status "Checking system metrics..."
STATUS_RESPONSE=$(curl -s --max-time $TIMEOUT "$API_URL/status" 2>/dev/null || echo "{}")

if [ -n "$STATUS_RESPONSE" ] && [ "$STATUS_RESPONSE" != "{}" ]; then
    # Extract key metrics
    MEMORY_GB=$(echo "$STATUS_RESPONSE" | grep -o '"memory_gb":[0-9.]*' | cut -d: -f2)
    GPU_MEMORY_GB=$(echo "$STATUS_RESPONSE" | grep -o '"gpu_memory_gb":[0-9.]*' | cut -d: -f2)
    MODEL_MODE=$(echo "$STATUS_RESPONSE" | grep -o '"model_mode":"[^"]*"' | cut -d'"' -f4)
    
    echo "  Model mode: $MODEL_MODE"
    echo "  Memory usage: ${MEMORY_GB:-0} GB"
    echo "  GPU memory: ${GPU_MEMORY_GB:-0} GB"
    
    # Check memory thresholds
    if [ -n "$MEMORY_GB" ]; then
        TOTAL_MEM=$(echo "$MEMORY_GB + ${GPU_MEMORY_GB:-0}" | bc)
        if (( $(echo "$TOTAL_MEM > 11.5" | bc -l) )); then
            print_warning "Memory usage critical: ${TOTAL_MEM}GB"
            update_status $EXIT_WARNING
        else
            print_success "Memory usage normal: ${TOTAL_MEM}GB"
        fi
    fi
else
    print_error "Failed to fetch system metrics"
    update_status $EXIT_WARNING
fi

# 5. Check Prometheus metrics
print_status "Checking metrics endpoint..."
METRICS_CHECK=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$METRICS_URL" 2>/dev/null || echo "000")

if [ "$METRICS_CHECK" = "200" ]; then
    print_success "Metrics endpoint accessible"
    
    # Get key metrics
    METRICS=$(curl -s --max-time $TIMEOUT "$METRICS_URL" 2>/dev/null || echo "")
    
    if [ -n "$METRICS" ]; then
        INFERENCE_TOTAL=$(echo "$METRICS" | grep -E '^speculative_decoding_inference_total' | grep 'status="success"' | awk '{print $2}' | head -1)
        ERROR_TOTAL=$(echo "$METRICS" | grep -E '^speculative_decoding_errors_total' | awk '{sum += $2} END {print sum}')
        
        echo "  Total successful inferences: ${INFERENCE_TOTAL:-0}"
        echo "  Total errors: ${ERROR_TOTAL:-0}"
        
        # Calculate error rate if we have data
        if [ -n "$INFERENCE_TOTAL" ] && [ -n "$ERROR_TOTAL" ] && [ "$INFERENCE_TOTAL" != "0" ]; then
            ERROR_RATE=$(echo "scale=2; $ERROR_TOTAL / ($INFERENCE_TOTAL + $ERROR_TOTAL) * 100" | bc)
            if (( $(echo "$ERROR_RATE > 1" | bc -l) )); then
                print_warning "High error rate: ${ERROR_RATE}%"
                update_status $EXIT_WARNING
            fi
        fi
    fi
else
    print_warning "Metrics endpoint not accessible (HTTP $METRICS_CHECK)"
    update_status $EXIT_WARNING
fi

# 6. Test inference capability
print_status "Testing inference capability..."
INFERENCE_RESPONSE=$(curl -s -X POST "$API_URL/infer" \
    -H "Content-Type: application/json" \
    --max-time 30 \
    -d '{"prompt": "Health check test", "max_length": 10}' 2>/dev/null || echo "{}")

if echo "$INFERENCE_RESPONSE" | grep -q '"success":true'; then
    print_success "Inference test passed"
    TOKENS=$(echo "$INFERENCE_RESPONSE" | grep -o '"tokens_generated":[0-9]*' | cut -d: -f2)
    TIME=$(echo "$INFERENCE_RESPONSE" | grep -o '"generation_time":[0-9.]*' | cut -d: -f2)
    if [ -n "$TOKENS" ] && [ -n "$TIME" ]; then
        echo "  Tokens generated: $TOKENS"
        echo "  Generation time: ${TIME}s"
    fi
else
    print_error "Inference test failed"
    update_status $EXIT_CRITICAL
    echo "$INFERENCE_RESPONSE" | jq . 2>/dev/null || echo "$INFERENCE_RESPONSE"
fi

# 7. Check disk space
print_status "Checking disk space..."
DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$DISK_USAGE" -gt 90 ]; then
    print_error "Disk usage critical: ${DISK_USAGE}%"
    update_status $EXIT_CRITICAL
elif [ "$DISK_USAGE" -gt 80 ]; then
    print_warning "Disk usage high: ${DISK_USAGE}%"
    update_status $EXIT_WARNING
else
    print_success "Disk usage normal: ${DISK_USAGE}%"
fi

# 8. Check logs for recent errors
print_status "Checking recent logs..."
if [ -f ~/.openclaw/workspace/skills/speculative-decoding/logs/error.log ]; then
    RECENT_ERRORS=$(tail -100 ~/.openclaw/workspace/skills/speculative-decoding/logs/error.log 2>/dev/null | grep -c ERROR || echo 0)
    if [ "$RECENT_ERRORS" -gt 10 ]; then
        print_warning "Found $RECENT_ERRORS errors in recent logs"
        update_status $EXIT_WARNING
    else
        print_success "Log check passed"
    fi
else
    print_status "No error log found (may be normal)"
fi

# Summary
echo ""
echo "=== Health Check Summary ==="

case $OVERALL_STATUS in
    $EXIT_SUCCESS)
        print_success "All checks passed - Service is healthy"
        ;;
    $EXIT_WARNING)
        print_warning "Some checks failed - Service degraded"
        ;;
    $EXIT_CRITICAL)
        print_error "Critical checks failed - Service unhealthy"
        ;;
esac

exit $OVERALL_STATUS