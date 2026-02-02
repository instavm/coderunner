#!/bin/bash

# Function to get current macOS version
get_macos_version() {
  sw_vers -productVersion | awk -F. '{print $1 "." $2}'
}

# Check the system type
if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "❌ This script is intended for macOS systems only. Exiting."
  exit 1
fi

# Check macOS version
macos_version=$(get_macos_version)
if (( $(echo "$macos_version < 26.0" | bc -l) )); then
  echo "Warning: Your macOS version is $macos_version. Version 26.0 or later is recommended. Some features of 'container' might not work properly."
else
  echo "✅ macOS system detected."
fi

download_url="https://github.com/apple/container/releases/download/0.8.0/container-installer-signed.pkg"

# Check if container is installed and display its version
if command -v container &> /dev/null
then
    echo "Apple 'container' tool detected. Current version:"
    container --version
    current_version=$(container --version | awk '{print $4}')
    echo $current_version
    target_version=$(echo $download_url | awk -F'/' '{print $8}')


    if [ "$current_version" != "$target_version" ]; then
        echo "Consider updating to version $target_version. Download it here: $download_url"
    fi

    echo "Stopping any running Apple 'container' processes..."
    container system stop 2>/dev/null || true
else
    echo "Apple 'container' tool not detected. Proceeding with installation..."

    # Download and install the Apple 'container' tool
    echo "Downloading Apple 'container' tool..."
    curl -Lo container-installer.pkg "$download_url"

    echo "Installing Apple 'container' tool..."
    sudo installer -pkg container-installer.pkg -target /
fi

# Stop any existing container system to clean up stale connections
echo "Stopping any existing container system..."
container system stop 2>/dev/null || true

# Wait a moment for cleanup
sleep 2

# Start the container system (this is blocking and will wait for kernel download if needed)
echo "Starting the Sandbox Container system (this may take a few minutes if downloading kernel)..."
echo "Note: First run may take 5+ minutes to download the kernel image."

# Use timeout to prevent indefinite blocking (10 minutes max)
if command -v timeout &> /dev/null; then
    if ! timeout 600 container system start; then
        if [ $? -eq 124 ]; then
            echo "❌ Container system start timed out after 10 minutes."
            echo "This usually means the kernel download is taking too long."
            echo "Try running manually: container system start"
        else
            echo "❌ Failed to start container system."
        fi
        exit 1
    fi
else
    # timeout command not available (older macOS), run without timeout
    if ! container system start; then
        echo "❌ Failed to start container system."
        exit 1
    fi
fi

# Quick verification that system is ready
echo "Verifying container system is ready..."
if container system status &>/dev/null; then
    echo "✅ Container system is ready."
else
    echo "❌ Container system started but status check failed."
    echo "Try running: container system stop && container system start"
    exit 1
fi

echo "Setting up local network domain..."

# Run the commands for setting up the local network
echo "Running: sudo container system dns create local"
sudo container system dns create local 2>/dev/null || echo "DNS domain 'local' already exists (this is fine)"

echo "Running: container system property set dns.domain local"
container system property set dns.domain local


echo "Pulling the latest image: instavm/coderunner"
if ! container image pull instavm/coderunner; then
    echo "❌ Failed to pull image. Please check your internet connection and try again."
    exit 1
fi

echo "→ Ensuring coderunner assets directories…"
ASSETS_SRC="$HOME/.coderunner/assets"
mkdir -p "$ASSETS_SRC/skills/user"
mkdir -p "$ASSETS_SRC/outputs"

# Stop any existing coderunner container
echo "Stopping any existing coderunner container..."
container stop coderunner 2>/dev/null || true
sleep 2

# Run the command to start the sandbox container
echo "Running: container run --name coderunner --detach --rm --cpus 8 --memory 4g instavm/coderunner"
if container run \
  --volume "$ASSETS_SRC/skills/user:/app/uploads/skills/user" \
  --volume "$ASSETS_SRC/outputs:/app/uploads/outputs" \
  --name coderunner \
  --detach \
  --rm \
  --cpus 8 \
  --memory 4g \
  instavm/coderunner; then
    echo "✅ Setup complete. MCP server is available at http://coderunner.local:8222/mcp"
else
    echo "❌ Failed to start coderunner container. Please check the logs with: container logs coderunner"
    exit 1
fi