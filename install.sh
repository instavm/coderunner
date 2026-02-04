#!/bin/bash

# CodeRunner Install Script
# Installs and starts the CodeRunner sandbox container

set -e

# --- Configuration ---
CONFIG_FILE="${CODERUNNER_CONFIG:-$HOME/.coderunner.config}"
CODERUNNER_HOME="${CODERUNNER_HOME:-$HOME/.coderunner}"

# --- Default options ---
WITH_SSH_AGENT=false
WITH_CREDENTIALS=false
ENV_FILE=""

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-ssh-agent)
            WITH_SSH_AGENT=true
            shift
            ;;
        --with-credentials)
            WITH_CREDENTIALS=true
            shift
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-ssh-agent    Forward SSH agent for git operations"
            echo "  --with-credentials  Mount config directories (~/.claude, ~/.config/gh, etc.)"
            echo "  --env-file FILE     Load environment variables from file"
            echo "  --config FILE       Use custom config file (default: ~/.coderunner.config)"
            echo ""
            echo "Config file format (~/.coderunner.config):"
            echo "  ANTHROPIC_API_KEY=sk-xxx"
            echo "  OPENAI_API_KEY=sk-xxx"
            echo "  GITHUB_TOKEN=ghp_xxx"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Load config file if exists ---
declare -a ENV_VARS
if [[ -f "$CONFIG_FILE" ]]; then
    echo "Loading config from $CONFIG_FILE"
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        if [[ -n "$key" && -n "$value" ]]; then
            ENV_VARS+=("--env" "$key=$value")
        fi
    done < "$CONFIG_FILE"
fi

# --- Load env file if specified ---
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    echo "Loading environment from $ENV_FILE"
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        if [[ -n "$key" && -n "$value" ]]; then
            ENV_VARS+=("--env" "$key=$value")
        fi
    done < "$ENV_FILE"
fi

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

echo "→ Ensuring coderunner directories..."
ASSETS_SRC="$CODERUNNER_HOME/assets"
mkdir -p "$ASSETS_SRC/skills/user"
mkdir -p "$ASSETS_SRC/outputs"

# Stop any existing coderunner container
echo "Stopping any existing coderunner container..."
container stop coderunner 2>/dev/null || true
sleep 2

# --- Build volume mounts ---
declare -a VOLUME_MOUNTS
VOLUME_MOUNTS+=("--volume" "$ASSETS_SRC/skills/user:/app/uploads/skills/user")
VOLUME_MOUNTS+=("--volume" "$ASSETS_SRC/outputs:/app/uploads/outputs")

# SSH Agent forwarding
if [[ "$WITH_SSH_AGENT" == true && -n "$SSH_AUTH_SOCK" ]]; then
    echo "Enabling SSH agent forwarding..."
    VOLUME_MOUNTS+=("--volume" "$SSH_AUTH_SOCK:/ssh-agent")
    ENV_VARS+=("--env" "SSH_AUTH_SOCK=/ssh-agent")
fi

# Credential directory mounts (read-only)
if [[ "$WITH_CREDENTIALS" == true ]]; then
    echo "Mounting credential directories..."
    # AI CLIs
    [[ -d "$HOME/.claude" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.claude:/root/.claude:ro")
    [[ -d "$HOME/.cursor" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.cursor:/root/.cursor:ro")
    [[ -d "$HOME/.codex" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.codex:/root/.codex:ro")
    # Cloud CLIs
    [[ -d "$HOME/.config/gh" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.config/gh:/root/.config/gh:ro")
    [[ -d "$HOME/.config/gcloud" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.config/gcloud:/root/.config/gcloud:ro")
    [[ -d "$HOME/.aws" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.aws:/root/.aws:ro")
    [[ -d "$HOME/.azure" ]] && VOLUME_MOUNTS+=("--volume" "$HOME/.azure:/root/.azure:ro")
fi

# Run the container
echo "Starting coderunner container..."
if container run \
  "${VOLUME_MOUNTS[@]}" \
  "${ENV_VARS[@]}" \
  --name coderunner \
  --detach \
  --rm \
  --cpus 8 \
  --memory 4g \
  instavm/coderunner; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "CodeRunner is running"
    echo ""
    echo "  Server:  http://coderunner.local:8222"
    echo "  MCP:     http://coderunner.local:8222/mcp"
    echo "  Health:  http://coderunner.local:8222/health"
    echo ""
    echo "Commands:"
    echo "  coderunner status    Check server status"
    echo "  coderunner stop      Stop the server"
    echo "  coderunner run       Run AI agents (claude, codex, cursor)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "Failed to start container. Check logs: container logs coderunner"
    exit 1
fi