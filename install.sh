#!/bin/bash

CPUS="${CODERUNNER_CPUS:-8}"
MEMORY="${CODERUNNER_MEMORY:-4g}"
MOUNT_SPECS=()
VOLUME_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --cpus N                  CPU count (default: 8)
  --memory SIZE             Memory limit, e.g. 4g or 8192m (default: 4g)
  --mount HOST:TARGET[:ro|rw]
                            Mount a host directory under /workspace.
                            Mounts are read-only unless :rw is specified.
  -h, --help                Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus)
            [ "$#" -ge 2 ] || { echo "Error: --cpus requires a value." >&2; exit 2; }
            CPUS="$2"
            shift 2
            ;;
        --memory)
            [ "$#" -ge 2 ] || { echo "Error: --memory requires a value." >&2; exit 2; }
            MEMORY="$2"
            shift 2
            ;;
        --mount)
            [ "$#" -ge 2 ] || { echo "Error: --mount requires a value." >&2; exit 2; }
            MOUNT_SPECS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: CPU count must be a positive integer." >&2
    exit 2
fi
if ! [[ "$MEMORY" =~ ^[1-9][0-9]*([KkMmGgTtPp][Bb]?)?$ ]]; then
    echo "Error: memory must be a positive size such as 4g or 8192m." >&2
    exit 2
fi

NORMALIZED_MOUNTS=()
for spec in "${MOUNT_SPECS[@]}"; do
    source_path=${spec%%:*}
    remainder=${spec#*:}
    if [ "$remainder" = "$spec" ] || [ -z "$source_path" ] || [ -z "$remainder" ]; then
        echo "Error: mount must use HOST:TARGET[:ro|rw]." >&2
        exit 2
    fi
    target_path=${remainder%%:*}
    if [ "$target_path" = "$remainder" ]; then
        mode="ro"
    else
        mode=${remainder#*:}
        if [[ "$mode" == *:* ]]; then
            echo "Error: mount must use HOST:TARGET[:ro|rw]." >&2
            exit 2
        fi
    fi
    if [ "$mode" != "ro" ] && [ "$mode" != "rw" ]; then
        echo "Error: mount mode must be 'ro' or 'rw'." >&2
        exit 2
    fi
    if [ ! -d "$source_path" ]; then
        echo "Error: mount source is not a directory: $source_path" >&2
        exit 2
    fi
    source_path=$(cd "$source_path" && pwd -P)
    case "$target_path" in
        /workspace|/workspace/*) ;;
        *)
            echo "Error: mount targets must be /workspace or a path below it." >&2
            exit 2
            ;;
    esac
    if [[ "$target_path" == *"/../"* ]] || [[ "$target_path" == */.. ]] || [[ "$target_path" == *"//"* ]]; then
        echo "Error: mount target contains an invalid path segment." >&2
        exit 2
    fi
    for existing in "${NORMALIZED_MOUNTS[@]}"; do
        existing_target=${existing#*:}
        existing_target=${existing_target%:*}
        if [ "$existing_target" = "$target_path" ]; then
            echo "Error: duplicate mount target: $target_path" >&2
            exit 2
        fi
    done
    NORMALIZED_MOUNTS+=("$source_path:$target_path:$mode")
    if [ "$mode" = "ro" ]; then
        VOLUME_ARGS+=(--volume "$source_path:$target_path:ro")
    else
        VOLUME_ARGS+=(--volume "$source_path:$target_path")
    fi
done

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
if ! container system start; then
    echo "❌ Failed to start container system."
    exit 1
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


echo "→ Ensuring coderunner assets directories…"
ASSETS_SRC="$HOME/.coderunner/assets"
mkdir -p "$ASSETS_SRC/skills/user"
mkdir -p "$ASSETS_SRC/outputs"

CONFIG_FILE="$HOME/.coderunner/container-config"
desired_config="cpus=$CPUS"$'\n'"memory=$MEMORY"
for mount in "${NORMALIZED_MOUNTS[@]}"; do
    desired_config="${desired_config}"$'\n'"mount=$mount"
done
if container inspect coderunner &>/dev/null; then
    if [ -f "$CONFIG_FILE" ]; then
        if [ "$(cat "$CONFIG_FILE")" != "$desired_config" ]; then
            echo "❌ Existing container was created with different resource or mount settings."
            echo "   Recreate it with: container delete coderunner && ./install.sh [options]"
            exit 1
        fi
    elif [ "$desired_config" != $'cpus=8\nmemory=4g' ]; then
        echo "❌ Existing container predates configurable resources and mounts."
        echo "   Recreate it with: container delete coderunner && ./install.sh [options]"
        exit 1
    fi
fi

# Stop any existing coderunner container
echo "Stopping any existing coderunner container..."
container stop coderunner 2>/dev/null || true
sleep 2

echo "Trying to resume any existing coderunner container..."
if container start coderunner 2>/dev/null; then
    echo "✅ Setup complete. MCP server is available at http://coderunner.local:8222/mcp"
    exit 0
fi

echo "Pulling the latest image: instavm/coderunner"
if ! container image pull instavm/coderunner; then
    echo "❌ Failed to pull image. Please check your internet connection and try again."
    exit 1
fi

# Run the command to start the sandbox container
echo "Starting coderunner with $CPUS CPUs and $MEMORY memory..."
if container run \
  --volume "$ASSETS_SRC/skills/user:/app/uploads/skills/user" \
  --volume "$ASSETS_SRC/outputs:/app/uploads/outputs" \
  "${VOLUME_ARGS[@]}" \
  --name coderunner \
  --detach \
  --cpus "$CPUS" \
  --memory "$MEMORY" \
  instavm/coderunner; then
    printf '%s' "$desired_config" > "$CONFIG_FILE"
    echo "✅ Setup complete. MCP server is available at http://coderunner.local:8222/mcp"
else
    echo "❌ Failed to start coderunner container. Please check the logs with: container logs coderunner"
    exit 1
fi
