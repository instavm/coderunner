#!/bin/bash

# --- Helper Functions ---

# Function to check if a command exists
command_exists() {
  command -v "$1" &> /dev/null
}

# Function to get current macOS version
get_macos_version() {
  sw_vers -productVersion | awk -F. '{print $1 "." $2}'
}

# --- Main Installation Logic ---

echo "Starting CodeRunner Setup..."

# --- macOS Specific Setup ---
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "✅ macOS system detected."

  # Check macOS version
  macos_version=$(get_macos_version)
  if (( $(echo "$macos_version < 26.0" | bc -l) )); then
    echo "⚠️ Warning: Your macOS version is $macos_version. Version 26.0 or later is recommended for Apple Container."
  fi

  # Check for Apple Container tool
  if command_exists container; then
    echo "✅ Apple 'container' tool detected."
    container --version
  else
    echo "❌ Apple 'container' tool not found."
    echo "Please install it from: https://github.com/apple/container/releases"
    exit 1
  fi

  echo "Starting Apple Container services..."
  container system start
  sudo container system dns create local
  container system dns default set local

  echo "Pulling the latest image for Apple Container..."
  container image pull instavm/coderunner

  echo "→ Ensuring coderunner assets directory exists..."
  ASSETS_SRC="$HOME/.coderunner/assets"
  mkdir -p "$ASSETS_SRC"

  echo "🚀 Starting CodeRunner container..."
  container run --volume "$ASSETS_SRC:/app/uploads" --name coderunner --detach --rm --cpus 8 --memory 4g instavm/coderunner

  echo "✅ Setup complete! MCP server is available at http://coderunner.local:8222/mcp"

# --- Docker-based Setup for Linux/Other ---
else
  echo "✅ Non-macOS system detected. Setting up with Docker."

  # Check for Docker
  if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker to continue."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
  fi

  echo "✅ Docker is installed."

  # Check if Docker daemon is running
  if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker and re-run this script."
    exit 1
  fi

  echo "Pulling the latest image from Docker Hub..."
  docker pull instavm/coderunner

  echo "→ Ensuring coderunner assets directory exists..."
  ASSETS_SRC="$HOME/.coderunner/assets"
  mkdir -p "$ASSETS_SRC"

  echo "🚀 Starting CodeRunner container using Docker..."
  docker run -d --rm --name coderunner \
    -p 8222:8222 \
    -v "$ASSETS_SRC:/app/uploads" \
    instavm/coderunner

  echo "✅ Setup complete! MCP server is available at http://localhost:8222/mcp"
fi
