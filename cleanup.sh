#!/bin/bash
#
# CodeRunner Cleanup Script
# Gracefully stops containers and cleans up resources
#

echo "CodeRunner Cleanup"
echo "=================="

# Step 1: Try graceful stop of coderunner container
echo "→ Stopping coderunner container gracefully..."
container stop coderunner 2>/dev/null && echo "  Stopped coderunner" || echo "  coderunner not running"

# Wait for graceful shutdown
sleep 2

# Step 2: Remove coderunner container if it still exists
echo "→ Removing coderunner container..."
container rm coderunner 2>/dev/null || true

# Step 3: Stop the container system
echo "→ Stopping container system..."
container system stop 2>/dev/null || true

# Step 4: Clean up buildkit if requested
if [ "$1" = "--full" ]; then
    echo "→ Full cleanup: removing buildkit..."
    container rm buildkit 2>/dev/null || true
fi

echo ""
echo "✅ Cleanup complete"
echo ""
echo "To restart CodeRunner:"
echo "  ./install.sh"
echo ""
echo "For full cleanup (including buildkit):"
echo "  ./cleanup.sh --full"
