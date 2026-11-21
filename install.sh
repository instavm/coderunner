#!/bin/bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODERUNNER_HOME="${CODERUNNER_HOME:-$HOME/.coderunner}"

# 1.3.1 and 1.4.1 fix image-load path traversal and symlink escapes.
CONTAINER_VERSION="1.5.0"
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

CPUS="${CODERUNNER_CPUS:-4}"
MEMORY="${CODERUNNER_MEMORY:-}"
PROFILE="${CODERUNNER_PROFILE:-auto}"
SEARCH="${CODERUNNER_SEARCH:-0}"
MOUNT_SPECS=()
VOLUME_ARGS=()
NETWORK_MODE="${CODERUNNER_NETWORK:-default}"
IMAGE="${CODERUNNER_IMAGE:-}"
BUILD_IMAGE=0
# Published sandbox images for this checkout (see .github/workflows/docker-build.yml). Until a tag is
# published, the installer builds the image from the checkout instead.
SANDBOX_TAG="0.2.0"
ASSISTANT=1
MODEL=""
ASSUME_YES=0

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --cpus N                  CPU count (default: 4)
  --memory SIZE             Memory limit, e.g. 2g or 8192m (default: 2g lite, 4g full)
  --profile lite|full       lite: code runs as plain processes, no Jupyter or browser in the
                            sandbox (the assistant uses your own browser), about 1 GB.
                            full: Jupyter kernels and a headless browser for MCP clients.
                            Default: lite with the assistant, full with --no-assistant
  --mount HOST:TARGET[:ro|rw]
                            Mount a host directory under /workspace.
                            Mounts are read-only unless :rw is specified.
  --network MODE            default (internet), none (no network) or proxy
                            (only owner-approved hosts; macOS). Default: default
  --build                   Build the sandbox image from this checkout
  --image NAME              Use a specific sandbox image
  --model ID                Use this model instead of the hardware-based pick
  --no-assistant            Install the sandbox and the coderunner CLI only (no model setup or login service)
  --search                  Search mode: the file index and the search tools for Claude Code, Codex and other
                            agents over MCP. No sandbox VM, no chat model (about 1.5 GB, a few minutes).
  -y, --yes                 Answer yes to installing a model runtime
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
        --network)
            [ "$#" -ge 2 ] || { echo "Error: --network requires a value." >&2; exit 2; }
            NETWORK_MODE="$2"
            shift 2
            ;;
        --image)
            [ "$#" -ge 2 ] || { echo "Error: --image requires a value." >&2; exit 2; }
            IMAGE="$2"
            shift 2
            ;;
        --model)
            [ "$#" -ge 2 ] || { echo "Error: --model requires a value." >&2; exit 2; }
            MODEL="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --build)
            BUILD_IMAGE=1
            shift
            ;;
        --search)
            SEARCH=1
            shift
            ;;
        --no-assistant)
            ASSISTANT=0
            shift
            ;;
        -y|--yes)
            ASSUME_YES=1
            shift
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
if [ "$PROFILE" = "auto" ]; then
    if [ "$ASSISTANT" = "1" ]; then PROFILE=lite; else PROFILE=full; fi
fi
if [ "$PROFILE" != "lite" ] && [ "$PROFILE" != "full" ]; then
    echo "Error: --profile must be lite or full." >&2
    exit 2
fi
if [ -z "$MEMORY" ]; then
    # lite: the assistant's general Linux VM (a shell first); 2 GB leaves room for builds and small servers.
    if [ "$PROFILE" = "lite" ]; then MEMORY=2g; else MEMORY=4g; fi
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
    # Never hand the sandbox the host's tokens, secrets or assistant state.
    mkdir -p "$CODERUNNER_HOME"
    state_path=$(cd "$CODERUNNER_HOME" && pwd -P)
    if [ "$source_path" = "$state_path" ] || [[ "$state_path" == "$source_path"/* ]] \
        || [[ "$source_path" == "$state_path"/* ]] || [ "$source_path" = "/" ]; then
        echo "Error: mount source must not contain or be inside $CODERUNNER_HOME (tokens and secrets)." >&2
        exit 2
    fi
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

case "$NETWORK_MODE" in
    default|none|proxy) ;;
    *)
        echo "Error: --network must be 'default', 'none' or 'proxy'." >&2
        exit 2
        ;;
esac

# The sandbox credential is created once and shared with host tools through a
# 0600 file. An exported CODERUNNER_API_TOKEN takes precedence and is saved.
ensure_token() {
    local token_file="$CODERUNNER_HOME/api-token" token
    mkdir -p "$CODERUNNER_HOME"
    if [ -n "${CODERUNNER_API_TOKEN:-}" ]; then
        token="$CODERUNNER_API_TOKEN"
    elif [ -s "$token_file" ]; then
        token=$(tr -d '[:space:]' < "$token_file")
    else
        token=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))' 2>/dev/null \
            || openssl rand -base64 48 | tr '+/' '-_' | tr -d '=\n')
    fi
    if [ "${#token}" -lt 32 ] || [[ "$token" =~ [[:space:]] ]]; then
        echo "❌ CODERUNNER_API_TOKEN must be at least 32 characters without whitespace." >&2
        exit 1
    fi
    (umask 077 && printf '%s\n' "$token" > "$token_file")
    CODERUNNER_API_TOKEN="$token"
    export CODERUNNER_API_TOKEN
}
ensure_token

echo "→ Ensuring coderunner assets directories…"
ASSETS_SRC="$CODERUNNER_HOME/assets"
mkdir -p "$ASSETS_SRC/skills/user"
mkdir -p "$ASSETS_SRC/outputs"

MCP_HOST="127.0.0.1"

# Wait until the MCP server inside the container answers over HTTP.
wait_for_server() {
    echo "Waiting for the MCP server to become ready (this can take a minute)..."
    local code auth_code unauth_code
    for _ in $(seq 1 60); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 "http://$MCP_HOST:8222/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
            unauth_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 \
                "http://$MCP_HOST:8222/v1/sessions/session")
            auth_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 \
                -H @<(printf 'Authorization: Bearer %s\n' "$CODERUNNER_API_TOKEN") \
                "http://$MCP_HOST:8222/v1/sessions/session")
            if [ "$auth_code" != "200" ] || [ "$unauth_code" != "401" ]; then
                echo "❌ The server must reject anonymous requests and accept the configured credential."
                echo "   Published images built before authentication ignore the token. Recreate the sandbox"
                echo "   from this checkout: container delete coderunner && ./install.sh --build"
                return 1
            fi
            echo "✅ Sandbox ready. MCP server: http://$MCP_HOST:8222/mcp (token: coderunner token)"
            return 0
        fi
        sleep 2
    done
    echo "❌ The container started but the MCP server did not respond within 120 seconds."
    echo "   Check the container logs with: container logs coderunner (or docker logs coderunner)"
    echo "   If coderunner.local does not resolve, verify domain = \"local\" in ~/.config/container/config.toml"
    echo "   and check the DNS service with: container system dns list"
    return 1
}

# Values the sandbox uses to reach the host gateway (local model, secret
# injection) and, in proxy mode, the egress proxy. Call this only inside the
# subshell that starts the container: the values are exported there and passed
# by name so the credential never appears in a process list, and the proxy
# settings never leak into this installer's own pip/brew/model downloads.
export_gateway_env() {
    local gateway="$1"
    export CODERUNNER_GATEWAY_URL="http://$gateway:8223"
    export OPENAI_BASE_URL="$CODERUNNER_GATEWAY_URL/v1"
    export OPENAI_API_KEY="$CODERUNNER_API_TOKEN"
    GATEWAY_ENV_ARGS=(--env CODERUNNER_GATEWAY_URL --env OPENAI_BASE_URL --env OPENAI_API_KEY)
    if [ "$NETWORK_MODE" = "proxy" ]; then
        local proxy="http://coderunner:$CODERUNNER_API_TOKEN@$gateway:8225"
        export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" http_proxy="$proxy" https_proxy="$proxy"
        export NO_PROXY="localhost,127.0.0.1,$gateway" no_proxy="localhost,127.0.0.1,$gateway"
        GATEWAY_ENV_ARGS+=(--env HTTP_PROXY --env HTTPS_PROXY --env http_proxy --env https_proxy
                           --env NO_PROXY --env no_proxy)
    fi
}

desired_config() {
    local config="cpus=$CPUS"$'\n'"memory=$MEMORY"$'\n'"profile=$PROFILE"
    if [ -n "${NORMALIZED_MOUNTS[*]-}" ]; then
        local sorted_mounts
        sorted_mounts=$(printf '%s\n' ${NORMALIZED_MOUNTS[@]+"${NORMALIZED_MOUNTS[@]}"} | LC_ALL=C sort)
        config="${config}"$'\n'"$(printf '%s\n' "$sorted_mounts" | sed 's/^/mount=/')"
    fi
    printf '%s' "$config"
}

# The builder VM kept ~13 GB of disk and 2 GB of memory after building the image; on an 18 GB Mac with a
# full disk the build itself failed while loading the result.
BUILD_SPACE_GB=20

check_build_space() {
    local free_gb
    free_gb=$(df -g "$SCRIPT_DIR" | awk 'NR == 2 {print $4}')
    if [ -n "$free_gb" ] && [ "$free_gb" -lt "$BUILD_SPACE_GB" ]; then
        echo "❌ Building the sandbox image needs about $BUILD_SPACE_GB GB free; $free_gb GB is free."
        exit 1
    fi
}

remove_builder() {
    container builder stop >/dev/null 2>&1 || true
    container builder delete >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------- macOS ----

install_sandbox_macos() {
    macos_version=$(sw_vers -productVersion | awk -F. '{print $1 "." $2}')
    macos_major=${macos_version%%.*}
    if [ "$macos_major" -lt 26 ]; then
        echo "Warning: Your macOS version is $macos_version. Version 26.0 or later is recommended. Some features of 'container' might not work properly."
    else
        echo "✅ macOS system detected."
    fi

    if command -v container &> /dev/null; then
        echo "Apple 'container' tool detected. Current version:"
        container --version
        current_version=$(container --version | awk '{print $4}')
        if [ "$current_version" != "$CONTAINER_VERSION" ]; then
            echo "Updating Apple 'container' to version $CONTAINER_VERSION..."
            install_container_tool
        fi
    else
        echo "Apple 'container' tool not detected. Proceeding with installation..."
        install_container_tool
    fi

    # Restart the container system to clean up stale connections.
    echo "Stopping any existing container system..."
    container system stop 2>/dev/null || true
    sleep 2
    echo "Starting the Sandbox Container system (this may take a few minutes if downloading kernel)..."
    if ! container system start; then
        echo "❌ Failed to start container system."
        exit 1
    fi
    if container system status &>/dev/null; then
        echo "✅ Container system is ready."
    else
        echo "❌ Container system started but status check failed."
        echo "Try running: container system stop && container system start"
        exit 1
    fi

    echo "Setting up local network domain..."
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

    NETWORK_ARGS=()
    MCP_HOST="coderunner.local"
    case "$NETWORK_MODE" in
        default)
            EXPECTED_NETWORK="default"
            ;;
        none|proxy)
            # Host-only network. In proxy mode the only way out is the host's egress proxy.
            EXPECTED_NETWORK=$([ "$NETWORK_MODE" = "none" ] && echo "coderunner-offline" || echo "coderunner-egress")
            if ! container network inspect "$EXPECTED_NETWORK" &>/dev/null; then
                container network create --internal "$EXPECTED_NETWORK"
            fi
            NETWORK_ARGS=(--network "$EXPECTED_NETWORK" --publish 127.0.0.1:8222:8222)
            MCP_HOST="127.0.0.1"
            ;;
    esac
    gateway=$(container network inspect "$EXPECTED_NETWORK" 2>/dev/null | python3 -c '
import ipaddress, json, sys

def find(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if "gateway" in key.lower() and isinstance(value, str):
                try:
                    return str(ipaddress.ip_address(value.split("/")[0]))
                except ValueError:
                    pass
        for value in node.values():
            found = find(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find(value)
            if found:
                return found
    return None

try:
    print(find(json.load(sys.stdin)) or "")
except Exception:
    print("")
')
    gateway=${gateway:-192.168.64.1}

    CONFIG_FILE="$CODERUNNER_HOME/container-config"
    config_now=$(desired_config)
    if container inspect coderunner &>/dev/null; then
        if [ -f "$CONFIG_FILE" ]; then
            if [ "$(cat "$CONFIG_FILE")" != "$config_now" ]; then
                echo "❌ Existing container was created with different resource or mount settings."
                echo "   Recreate it with: container delete coderunner && ./install.sh [options]"
                exit 1
            fi
        elif [ "$config_now" != $'cpus=8\nmemory=4g' ]; then
            echo "❌ Existing container predates configurable resources and mounts."
            echo "   Recreate it with: container delete coderunner && ./install.sh [options]"
            exit 1
        fi
    fi

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
            echo "   Recreate it with: container delete coderunner && ./install.sh --network $NETWORK_MODE"
            exit 1
        fi
        if container start coderunner; then
            wait_for_server || exit 1
            return 0
        fi
        echo "❌ Existing coderunner container could not be started."
        echo "   Check its logs, or recreate it with: container delete coderunner && ./install.sh [options]"
        exit 1
    fi

    if [ "$BUILD_IMAGE" = "1" ]; then
        IMAGE="${IMAGE:-coderunner-local:latest}"
        check_build_space
        echo "Building the sandbox image $IMAGE from $SCRIPT_DIR (this takes a while)..."
        # After an install or upgrade the builder image is downloaded on first use; inside
        # `container build` that download can outlast the API timeout, so start the builder first.
        container builder start >/dev/null 2>&1 || true
        if ! container build -t "$IMAGE" "$SCRIPT_DIR"; then
            remove_builder
            echo "❌ Image build failed."
            exit 1
        fi
        remove_builder
    else
        IMAGE="${IMAGE:-instavm/coderunner:$SANDBOX_TAG}"
        if container image inspect "$IMAGE" &>/dev/null; then
            echo "Using the image already on this Mac: $IMAGE"
        elif echo "Pulling the image: $IMAGE" && ! container image pull "$IMAGE"; then
            if [ "$IMAGE" != "instavm/coderunner:$SANDBOX_TAG" ] || [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
                echo "❌ Failed to pull image. Please check your internet connection and try again."
                exit 1
            fi
            echo "The $SANDBOX_TAG sandbox image is not published yet; building it from this checkout instead."
            IMAGE="coderunner-local:latest"
            check_build_space
            container builder start >/dev/null 2>&1 || true
            container build -t "$IMAGE" "$SCRIPT_DIR" || { remove_builder; echo "❌ Image build failed."; exit 1; }
            remove_builder
        fi
    fi

    echo "Starting coderunner ($PROFILE profile) with $CPUS CPUs and $MEMORY memory..."
    if (export_gateway_env "$gateway" && container run \
      --volume "$ASSETS_SRC/skills/user:/app/uploads/skills/user" \
      --volume "$ASSETS_SRC/outputs:/app/uploads/outputs" \
      ${VOLUME_ARGS[@]+"${VOLUME_ARGS[@]}"} \
      --name coderunner \
      --detach \
      --cpus "$CPUS" \
      --memory "$MEMORY" \
      --env CODERUNNER_API_TOKEN \
      --env CODERUNNER_PROFILE="$PROFILE" \
      "${GATEWAY_ENV_ARGS[@]}" \
      ${NETWORK_ARGS[@]+"${NETWORK_ARGS[@]}"} \
      "$IMAGE"); then
        printf '%s' "$config_now" > "$CONFIG_FILE"
        wait_for_server || exit 1
    else
        echo "❌ Failed to start coderunner container. Please check the logs with: container logs coderunner"
        exit 1
    fi
}

# ---------------------------------------------------------------- Linux ----

install_sandbox_linux() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo "✅ WSL2 detected; using Docker inside WSL."
    fi
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is required: https://docs.docker.com/engine/install/ (or Docker Desktop with WSL integration)."
        exit 1
    fi
    if ! docker info &> /dev/null; then
        echo "❌ Cannot reach the Docker daemon. Start it, or add your user to the docker group."
        exit 1
    fi
    if [ "$NETWORK_MODE" != "default" ]; then
        echo "❌ --network $NETWORK_MODE currently needs Apple container on macOS. Use the default mode on Linux."
        exit 2
    fi
    if docker info --format '{{.OperatingSystem}}' 2>/dev/null | grep -qi "docker desktop" \
        && [ "${CODERUNNER_ALLOW_DOCKER_DESKTOP:-0}" != "1" ]; then
        echo "❌ Docker Desktop lets containers reach every service listening on this computer's loopback"
        echo "   (model runtimes, signal-cli, development servers) through host.docker.internal."
        echo "   Use Docker Engine, or accept the risk with CODERUNNER_ALLOW_DOCKER_DESKTOP=1 ./install.sh"
        exit 1
    fi
    local nproc_count
    nproc_count=$(nproc 2>/dev/null || echo "$CPUS")
    if [ "$CPUS" -gt "$nproc_count" ]; then
        CPUS="$nproc_count"
    fi
    RUNTIME_ARGS=()
    runtime="${CODERUNNER_DOCKER_RUNTIME:-}"
    if [ -z "$runtime" ] && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"runsc"'; then
        runtime="runsc"
    fi
    if [ -n "$runtime" ]; then
        echo "Using the $runtime container runtime."
        RUNTIME_ARGS=(--runtime "$runtime")
    fi
    MCP_HOST="127.0.0.1"

    if docker inspect coderunner &> /dev/null; then
        echo "Resuming the existing coderunner container..."
        if docker start coderunner > /dev/null; then
            wait_for_server || exit 1
            return 0
        fi
        echo "❌ Existing coderunner container could not be started: docker logs coderunner"
        exit 1
    fi

    if [ "$BUILD_IMAGE" = "1" ]; then
        IMAGE="${IMAGE:-coderunner-local:latest}"
        echo "Building the sandbox image $IMAGE from $SCRIPT_DIR (this takes a while)..."
        docker build -t "$IMAGE" "$SCRIPT_DIR" || { echo "❌ Image build failed."; exit 1; }
    else
        IMAGE="${IMAGE:-instavm/coderunner:$SANDBOX_TAG}"
        if ! docker pull "$IMAGE"; then
            if [ "$IMAGE" != "instavm/coderunner:$SANDBOX_TAG" ] || [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
                echo "❌ Failed to pull $IMAGE."
                exit 1
            fi
            echo "The $SANDBOX_TAG sandbox image is not published yet; building it from this checkout instead."
            IMAGE="coderunner-local:latest"
            docker build -t "$IMAGE" "$SCRIPT_DIR" || { echo "❌ Image build failed."; exit 1; }
        fi
    fi
    DOCKER_VOLUMES=()
    for spec in ${NORMALIZED_MOUNTS[@]+"${NORMALIZED_MOUNTS[@]}"}; do
        DOCKER_VOLUMES+=(--volume "$spec")
    done
    if (export_gateway_env "host.docker.internal" && docker run --detach --name coderunner --restart unless-stopped \
      --publish 127.0.0.1:8222:8222 \
      --add-host host.docker.internal:host-gateway \
      --cpus "$CPUS" --memory "$MEMORY" --env CODERUNNER_PROFILE="$PROFILE" \
      --volume "$ASSETS_SRC/skills/user:/app/uploads/skills/user" \
      --volume "$ASSETS_SRC/outputs:/app/uploads/outputs" \
      ${DOCKER_VOLUMES[@]+"${DOCKER_VOLUMES[@]}"} \
      --env CODERUNNER_API_TOKEN \
      "${GATEWAY_ENV_ARGS[@]}" \
      ${RUNTIME_ARGS[@]+"${RUNTIME_ARGS[@]}"} \
      "$IMAGE" > /dev/null); then
        wait_for_server || exit 1
    else
        echo "❌ Failed to start the coderunner container: docker logs coderunner"
        exit 1
    fi
}

# --------------------------------------------------------- host runtime ----

# EmbeddingGemma 2 needs Ollama 0.36.0 and decision models 0.35.0. Homebrew installs are upgraded
# here; otherwise `coderunner setup` fetches its own pinned copy on macOS (leaving the Ollama app
# alone) or explains how to update.
MIN_OLLAMA="0.36.0"
check_ollama_version() {
    command -v ollama &> /dev/null || return 0
    local found
    found=$(ollama --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | tail -1)
    [ -n "$found" ] || return 0
    if [ "$(printf '%s\n%s\n' "$MIN_OLLAMA" "$found" | sort -V | head -1)" = "$MIN_OLLAMA" ]; then
        return 0
    fi
    echo "Ollama $found is older than $MIN_OLLAMA, which EmbeddingGemma 2 and decision models need."
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null && brew list --formula ollama &> /dev/null; then
        brew upgrade ollama || echo "Warning: brew upgrade ollama failed."
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "coderunner setup will download its own copy of Ollama into $CODERUNNER_HOME/runtime."
        echo "If the Ollama app is running, update it from its menu (Restart to Update) or quit it first."
    else
        echo "Update it with the official script after reviewing it: curl -fsSL https://ollama.com/install.sh | sh"
    fi
}

# A dedicated virtualenv holds the host assistant (`coderunner` CLI and
# daemon) and the Claude Desktop stdio proxy, independent of system Python.
setup_host_runtime() {
    local venv_dir="$CODERUNNER_HOME/venv" bin
    if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        echo "Python 3.10+ is needed for the host assistant; the sandbox is ready on its own."
        return 0
    fi
    if [ ! -x "$venv_dir/bin/python" ]; then
        echo "Creating Python virtualenv at $venv_dir ..."
        python3 -m venv "$venv_dir" || { echo "Warning: could not create virtualenv."; return 0; }
    fi
    if ! "$venv_dir/bin/pip" install --quiet --upgrade "fastmcp==3.4.7" -e "${SCRIPT_DIR}[mcp]"; then
        echo "Warning: failed to install the host assistant into $venv_dir."
        return 0
    fi
    bin="$HOME/.local/bin"
    mkdir -p "$bin"
    ln -sf "$venv_dir/bin/coderunner" "$bin/coderunner"
    case ":$PATH:" in
        *":$bin:"*) ;;
        *) echo "Add $bin to your PATH to use the 'coderunner' command (or run $venv_dir/bin/coderunner)." ;;
    esac
    [ "$ASSISTANT" = "1" ] || return 0

    if [ "$SEARCH" = "1" ]; then
        if ! "$venv_dir/bin/coderunner" setup --search --no-voice; then
            echo "⚠️  Search setup did not finish. Fix the issue above and run: coderunner setup --search"
            return 0
        fi
        if "$venv_dir/bin/coderunner" service install; then
            echo "✅ CodeRunner installed as a login service (search mode)."
        else
            echo "⚠️  Could not install the login service; start it with: coderunner up"
        fi
        echo
        echo "Next steps:"
        echo "  coderunner files add ~/Documents                  index a folder (more with further adds)"
        echo "  coderunner files sources add mail                 also messages, notes, calendar"
        echo "  claude mcp add coderunner-search -- coderunner mcp --tools search"
        echo "  coderunner find \"the lease for the apartment\"      search from the terminal"
        return 0
    fi

    if ! command -v ollama &> /dev/null && ! command -v llama-server &> /dev/null; then
        local answer="n"
        if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
            if [ "$ASSUME_YES" = "1" ]; then
                answer="y"
            elif [ -t 0 ]; then
                read -r -p "Install Ollama with Homebrew to run local models? [Y/n] " answer
                answer=${answer:-y}
            fi
            if [[ "$answer" =~ ^[Yy] ]]; then
                brew install ollama || echo "Warning: Ollama installation failed."
            fi
        else
            echo "No local model runtime found. Install Ollama (https://ollama.com/download) or llama.cpp,"
            echo "then run: coderunner setup"
        fi
    fi
    check_ollama_version

    setup_args=(setup --network "$NETWORK_MODE")
    [ -n "$MODEL" ] && setup_args+=(--model "$MODEL")
    if ! "$venv_dir/bin/coderunner" "${setup_args[@]}"; then
        echo "⚠️  Model setup did not finish. Fix the issue above and run: coderunner setup"
    fi
    if "$venv_dir/bin/coderunner" service install; then
        echo "✅ Assistant daemon installed as a login service."
    else
        echo "⚠️  Could not install the login service; start the assistant with: coderunner up"
    fi
    echo
    echo "Next steps:"
    echo "  coderunner chat                     talk to your assistant (approvals are asked in the terminal)"
    echo "  coderunner status                   check the sandbox, model and daemon"
    echo "  coderunner remote enable            reach it privately from your phone over Tailscale"
    echo "  See ASSISTANT.md for channels, connectors and the safety model."
}

if [ "$SEARCH" = "1" ]; then
    ASSISTANT=1
    echo "Search mode: no sandbox VM is installed."
else
    case "$OSTYPE" in
        darwin*) install_sandbox_macos ;;
        linux*) install_sandbox_linux ;;
        *)
            echo "❌ Unsupported system: $OSTYPE. Use macOS, Linux or Windows with WSL2."
            exit 1
            ;;
    esac
fi
setup_host_runtime
