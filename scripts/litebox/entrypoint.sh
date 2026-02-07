#!/usr/bin/env bash
set -euo pipefail

RUNNER=${LITEBOX_RUNNER:-/usr/local/bin/litebox_runner_linux_userland}
ROOTFS_TAR=${LITEBOX_ROOTFS_TAR:-/opt/litebox/rootfs.tar}
NODE_BIN=${LITEBOX_NODE_BIN:-/usr/bin/node}
CLAUDE_REL_PATH_FILE=${LITEBOX_CLAUDE_REL_PATH_FILE:-/opt/litebox/claude_entrypoint_rel.txt}
WORKSPACE_DIR=${LITEBOX_WORKSPACE_DIR:-/workspace}
INCLUDE_WORKSPACE=${LITEBOX_INCLUDE_WORKSPACE:-1}
EXCLUDE_GIT=${LITEBOX_WORKSPACE_EXCLUDE_GIT:-1}

if [ ! -x "$RUNNER" ]; then
  echo "LiteBox runner not found at $RUNNER" >&2
  exit 1
fi
if [ ! -f "$ROOTFS_TAR" ]; then
  echo "Rootfs tar not found at $ROOTFS_TAR" >&2
  exit 1
fi
if [ ! -x "$NODE_BIN" ]; then
  echo "Node binary not found at $NODE_BIN" >&2
  exit 1
fi
if [ ! -f "$CLAUDE_REL_PATH_FILE" ]; then
  echo "Claude entrypoint rel path file not found at $CLAUDE_REL_PATH_FILE" >&2
  exit 1
fi

CLAUDE_REL_PATH=$(cat "$CLAUDE_REL_PATH_FILE")
CLAUDE_ENTRYPOINT="/usr/local/lib/node_modules/@anthropic-ai/claude-code/${CLAUDE_REL_PATH}"

if [ "$INCLUDE_WORKSPACE" = "1" ] && [ -d "$WORKSPACE_DIR" ]; then
  tmp_dir=$(mktemp -d)
  cleanup() {
    rm -rf "$tmp_dir"
    if [ -n "${COMBINED_TAR:-}" ] && [ -f "$COMBINED_TAR" ]; then
      rm -f "$COMBINED_TAR"
    fi
  }
  trap cleanup EXIT

  tar -xf "$ROOTFS_TAR" -C "$tmp_dir"
  mkdir -p "$tmp_dir/workspace"

  TAR_EXCLUDES=()
  if [ "$EXCLUDE_GIT" = "1" ]; then
    TAR_EXCLUDES+=(--exclude=.git)
  fi

  tar -C "$WORKSPACE_DIR" "${TAR_EXCLUDES[@]}" -cf "$tmp_dir/workspace.tar" .
  tar -xf "$tmp_dir/workspace.tar" -C "$tmp_dir/workspace"

  COMBINED_TAR=$(mktemp)
  tar --format=ustar -C "$tmp_dir" -cf "$COMBINED_TAR" .
  ROOTFS_TAR="$COMBINED_TAR"
fi

ARGS=(
  --unstable
  --interception-backend seccomp
  --env "HOME=/"
  --env "NODE_PATH=/usr/local/lib/node_modules"
  --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
  --env "SSL_CERT_DIR=/etc/ssl/certs"
  --env "LD_LIBRARY_PATH=/lib:/lib64:/usr/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu"
  --initial-files "$ROOTFS_TAR"
)

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  ARGS+=(--env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
fi

exec "$RUNNER" "${ARGS[@]}" "$NODE_BIN" "$CLAUDE_ENTRYPOINT" "$@"
