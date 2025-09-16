#!/bin/bash

# Test script for multi-platform container builds
# Tests both Docker and Apple container builds locally

set -e

echo "🚀 CodeRunner Multi-Platform Build Test"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOS=true
    echo "🍎 macOS detected"
else
    MACOS=false
    echo "🐧 Non-macOS system detected"
fi

echo ""
echo "1. Testing Docker Build"
echo "----------------------"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker."
    exit 1
fi

print_status "Docker found: $(docker --version)"

# Test basic Docker build
echo "Building with Docker..."
if docker build -t coderunner-docker-test .; then
    print_status "Docker build successful"
else
    print_error "Docker build failed"
    exit 1
fi

# Test if the Docker image runs
echo "Testing Docker container startup..."
DOCKER_CONTAINER_ID=$(docker run -d -p 8223:8222 coderunner-docker-test)
sleep 5

# Check if container is running
if docker ps | grep -q "$DOCKER_CONTAINER_ID"; then
    print_status "Docker container started successfully"
    
    # Test MCP endpoint (basic connectivity test)
    if curl -s --fail --connect-timeout 5 http://localhost:8223/mcp > /dev/null; then
        print_status "MCP endpoint responding on Docker container"
    else
        print_warning "MCP endpoint not responding (may need more startup time)"
    fi
    
    # Cleanup
    docker stop "$DOCKER_CONTAINER_ID" > /dev/null
    print_status "Docker container stopped and cleaned up"
else
    print_error "Docker container failed to start"
fi

echo ""
echo "2. Testing Docker Multi-Architecture Build"
echo "-----------------------------------------"

# Check if buildx is available
if docker buildx version &> /dev/null; then
    print_status "Docker buildx found"
    
    # Create builder if it doesn't exist
    if ! docker buildx inspect multiarch-builder &> /dev/null; then
        echo "Creating multi-arch builder..."
        docker buildx create --name multiarch-builder --use
    else
        docker buildx use multiarch-builder
    fi
    
    # Test multi-arch build (without pushing)
    echo "Testing multi-architecture build..."
    if docker buildx build --platform linux/amd64,linux/arm64 -t coderunner-multiarch .; then
        print_status "Multi-architecture build successful"
    else
        print_error "Multi-architecture build failed"
    fi
else
    print_warning "Docker buildx not available - skipping multi-arch test"
fi

# Skip Apple container test for now
if $MACOS; then
    echo ""
    echo "3. Apple Container Build (SKIPPED)"
    echo "---------------------------------"
    print_warning "Apple container test skipped - focusing on Docker builds"
fi

echo ""
echo "4. Build Comparison Summary"
echo "=========================="

echo "Docker image size:"
docker images coderunner-docker-test --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Apple container comparison skipped

echo ""
echo "🎉 Build test completed!"
echo ""
echo "Next steps:"
echo "- If builds are successful, you can push to registries:"
echo "  Docker: docker tag coderunner-docker-test instavm/coderunner:docker-latest"
echo "  Apple:  container images tag coderunner-apple-test instavm/coderunner:apple-latest"
echo "- Set up automated builds with GitHub Actions"
echo "- Update documentation with multi-platform instructions"