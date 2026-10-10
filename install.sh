#!/bin/bash
set -u

CONTAINER_VERSION="1.2.2"
CONTAINER_INSTALLER="container-${CONTAINER_VERSION}-installer-signed.pkg"
download_url="https://github.com/apple/container/releases/download/${CONTAINER_VERSION}/${CONTAINER_INSTALLER}"

install_container_tool() {
    echo "Downloading Apple 'container' $CONTAINER_VERSION..."
    if ! curl -fL -o "$CONTAINER_INSTALLER" "$download_url"; then
        echo "❌ Failed to download $download_url"
        exit 1
    fi
    if ! sudo installer -pkg "$CONTAINER_INSTALLER" -target /; then
        rm -f "$CONTAINER_INSTALLER"
        echo "❌ Failed to install Apple 'container'."
        exit 1
    fi
    rm -f "$CONTAINER_INSTALLER"
}

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
MEMORY=$(printf '%s' "$MEMORY" | tr '[:upper:]' '[:lower:]')
MEMORY=${MEMORY%b}

NORMALIZED_MOUNTS=()
for spec in ${MOUNT_SPECS[@]+"${MOUNT_SPECS[@]}"}; do
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
    for existing in ${NORMALIZED_MOUNTS[@]+"${NORMALIZED_MOUNTS[@]}"}; do
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
        # Apple's --volume syntax represents writable mounts by omitting :ro.
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
macos_major=${macos_version%%.*}
if [ "$macos_major" -lt 26 ]; then
  echo "Warning: Your macOS version is $macos_version. Version 26.0 or later is recommended. Some features of 'container' might not work properly."
else
  echo "✅ macOS system detected."
fi

# Check if container is installed and display its version
if command -v container &> /dev/null
then
    echo "Apple 'container' tool detected. Current version:"
    container --version
    current_version=$(container --version | awk '{print $4}')
    echo $current_version
    if [ "$current_version" != "$CONTAINER_VERSION" ]; then
        echo "Updating Apple 'container' to version $CONTAINER_VERSION..."
        install_container_tool
    fi

    echo "Stopping any running Apple 'container' processes..."
    container system stop 2>/dev/null || true
else
    echo "Apple 'container' tool not detected. Proceeding with installation..."
    install_container_tool
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

# Since container 1.0, the default DNS domain is set in the service config
# file instead of `container system property set dns.domain`.
CONTAINER_CONFIG="$HOME/.config/container/config.toml"
mkdir -p "$(dirname "$CONTAINER_CONFIG")"
touch "$CONTAINER_CONFIG"
dns_changed=$(python3 - "$CONTAINER_CONFIG" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
match = re.search(r"(?ms)^\[dns\]\s*\n(.*?)(?=^\[|\Z)", text)
changed = not (
    match and re.search(r'(?m)^domain\s*=\s*"local"\s*$', match.group(1))
)
if not changed:
    print(0)
    raise SystemExit
if match:
    body = match.group(1)
    if re.search(r"(?m)^domain\s*=", body):
        body = re.sub(r'(?m)^domain\s*=.*$', 'domain = "local"', body)
    else:
        body = 'domain = "local"\n' + body
    text = text[:match.start(1)] + body + text[match.end(1):]
else:
    text = text.rstrip() + '\n\n[dns]\ndomain = "local"\n'
path.write_text(text)
print(1)
PY
)
if [ "$dns_changed" = "1" ]; then
    echo "Configured DNS domain 'local' in $CONTAINER_CONFIG; restarting container system..."
    container system stop 2>/dev/null || true
    if ! container system start; then
        echo "❌ Failed to restart container system after DNS configuration."
        exit 1
    fi
fi

# Set up a dedicated virtualenv for the Claude Desktop stdio proxy so that
# it does not depend on the state of the system Python installation.
setup_proxy_venv() {
    local venv_dir="$HOME/.coderunner/venv"
    if ! command -v python3 &> /dev/null; then
        echo "python3 not found; skipping proxy setup. Install Python 3.10+ to use the Claude Desktop proxy."
        return 0
    fi
    if [ ! -x "$venv_dir/bin/python" ]; then
        echo "Creating Python virtualenv for the MCP proxy at $venv_dir ..."
        if ! python3 -m venv "$venv_dir"; then
            echo "Warning: could not create virtualenv; skipping proxy setup."
            return 0
        fi
    fi
    if "$venv_dir/bin/pip" install --quiet --upgrade "fastmcp==3.4.7"; then
        echo "MCP proxy dependencies installed."
        echo "For Claude Desktop, use this python in claude_desktop_config.json:"
        echo "  $venv_dir/bin/python"
    else
        echo "Warning: failed to install proxy dependencies into $venv_dir."
    fi
}
setup_proxy_venv

echo "→ Ensuring coderunner assets directories…"
ASSETS_SRC="$HOME/.coderunner/assets"
mkdir -p "$ASSETS_SRC/skills/user"
mkdir -p "$ASSETS_SRC/outputs"

NETWORK_MODE="${CODERUNNER_NETWORK:-default}"
NETWORK_ARGS=()
MCP_HOST="coderunner.local"
case "$NETWORK_MODE" in
    default)
        EXPECTED_NETWORK="default"
        ;;
    none)
        EXPECTED_NETWORK="coderunner-offline"
        if ! container network inspect "$EXPECTED_NETWORK" &>/dev/null; then
            container network create --internal "$EXPECTED_NETWORK"
        fi
        NETWORK_ARGS=(--network "$EXPECTED_NETWORK" --publish 127.0.0.1:8222:8222)
        MCP_HOST="127.0.0.1"
        ;;
    *)
        echo "❌ CODERUNNER_NETWORK must be 'default' or 'none'."
        exit 1
        ;;
esac

# Wait until the MCP server inside the container answers over HTTP.
wait_for_server() {
    echo "Waiting for the MCP server to become ready (this can take a minute)..."
    local i code
    for i in $(seq 1 60); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 "http://$MCP_HOST:8222/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
            echo "✅ Setup complete. MCP server is available at http://$MCP_HOST:8222/mcp"
            return 0
        fi
        sleep 2
    done
    echo "❌ The container started but the MCP server did not respond within 120 seconds."
    echo "   Check the container logs with: container logs coderunner"
    echo "   If coderunner.local does not resolve, verify domain = \"local\" in ~/.config/container/config.toml"
    echo "   and check the DNS service with: container system dns list"
    return 1
}

CONFIG_FILE="$HOME/.coderunner/container-config"
desired_config="cpus=$CPUS"$'\n'"memory=$MEMORY"
if [ -n "${NORMALIZED_MOUNTS[*]-}" ]; then
    sorted_mounts=$(printf '%s\n' ${NORMALIZED_MOUNTS[@]+"${NORMALIZED_MOUNTS[@]}"} | LC_ALL=C sort)
    desired_config="${desired_config}"$'\n'"$(printf '%s\n' "$sorted_mounts" | sed 's/^/mount=/')"
fi
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
if container inspect coderunner &>/dev/null; then
    existing_network=$(container inspect coderunner | python3 -c '
import json, sys
networks = json.load(sys.stdin)[0]["configuration"]["networks"]
print(networks[0]["network"] if networks else "default")
')
    if [ "$existing_network" != "$EXPECTED_NETWORK" ]; then
        container stop coderunner 2>/dev/null || true
        echo "❌ Existing container uses network '$existing_network', but '$EXPECTED_NETWORK' was requested."
        echo "   Recreate it with: container delete coderunner && CODERUNNER_NETWORK=$NETWORK_MODE ./install.sh"
        exit 1
    fi
    if container start coderunner; then
        wait_for_server
        exit $?
    fi
fi
if container inspect coderunner &>/dev/null; then
    echo "❌ Existing coderunner container could not be started."
    echo "   Check its logs, or recreate it with: container delete coderunner && ./install.sh [options]"
    exit 1
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
  ${VOLUME_ARGS[@]+"${VOLUME_ARGS[@]}"} \
  --name coderunner \
  --detach \
  --cpus "$CPUS" \
  --memory "$MEMORY" \
  ${NETWORK_ARGS[@]+"${NETWORK_ARGS[@]}"} \
  instavm/coderunner; then
    printf '%s' "$desired_config" > "$CONFIG_FILE"
    wait_for_server
    exit $?
else
    echo "❌ Failed to start coderunner container. Please check the logs with: container logs coderunner"
    exit 1
fi
