#!/bin/bash

# Function to get current macOS version
get_macos_version() {
  sw_vers -productVersion | awk -F. '{print $1 "." $2}'
}

# Check the system type
if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "This script is intended for macOS systems only. Exiting."
  exit 1
fi

# Check macOS version
macos_version=$(get_macos_version)
if (( $(echo "$macos_version < 26.0" | bc -l) )); then
  echo "Warning: Your macOS version is $macos_version. Version 26.0 or later is recommended. Some features of 'container' might not work properly."
fi

# Download and install the Apple 'container' tool
echo "--------------------------------"
echo "Downloading Apple 'container' tool..."
echo "--------------------------------"
curl -Lo container-installer.pkg https://github.com/apple/container/releases/download/0.1.0/container-0.1.0-installer-signed.pkg

echo "--------------------------------"
echo "Installing Apple 'container' tool..."
echo "--------------------------------"
sudo installer -pkg container-installer.pkg -target /

echo "Setting up local network domain..."

# Run the commands for setting up the local network
echo "--------------------------------"
echo "Running: sudo container system dns create local"
echo "--------------------------------"
sudo container system dns create local

echo "Running: container system dns default set local"
container system dns default set local

echo "--------------------------------"
echo "Starting the Sandbox Container..."
echo "--------------------------------"

# Run the command to start the sandbox container
echo "--------------------------------"
echo "Running: container run --name coderunner --detach --rm --cpus 8 --memory 4g instavm/coderunner"
echo "--------------------------------"

container run --name coderunner --detach --rm --cpus 8 --memory 4g instavm/coderunner

echo "--------------------------------"
echo "Setup complete. MCP server is available at http://coderunner.local:8222/sse"
echo "--------------------------------"