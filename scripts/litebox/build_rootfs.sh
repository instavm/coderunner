#!/usr/bin/env bash
set -euo pipefail

OUT_TAR=${1:-/opt/litebox/rootfs.tar}
NODE_BIN=${NODE_BIN:-$(command -v node)}
CLAUDE_PKG_DIR=${CLAUDE_PKG_DIR:-/usr/local/lib/node_modules/claude-code}

if [ ! -x "$NODE_BIN" ]; then
  echo "node binary not found at $NODE_BIN" >&2
  exit 1
fi
if [ ! -d "$CLAUDE_PKG_DIR" ]; then
  echo "Claude package not found at $CLAUDE_PKG_DIR" >&2
  exit 1
fi

ROOTFS_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$ROOTFS_DIR"
}
trap cleanup EXIT

mkdir -p "$ROOTFS_DIR"

# Copy shared libraries (including the ELF interpreter) needed by node
while IFS= read -r lib; do
  dest="$ROOTFS_DIR${lib}"
  mkdir -p "$(dirname "$dest")"
  cp -L "$lib" "$dest"
done < <(ldd "$NODE_BIN" | awk '{for (i=1;i<=NF;i++) if (index($i, "/") == 1) print $i}' | sort -u)

# Include the Node binary itself
node_dest="$ROOTFS_DIR${NODE_BIN}"
mkdir -p "$(dirname "$node_dest")"
cp -L "$NODE_BIN" "$node_dest"

# CA certificates and basic network config
mkdir -p "$ROOTFS_DIR/etc/ssl/certs"
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  cp -L /etc/ssl/certs/ca-certificates.crt "$ROOTFS_DIR/etc/ssl/certs/"
fi
for f in /etc/hosts /etc/resolv.conf /etc/nsswitch.conf; do
  if [ -f "$f" ]; then
    mkdir -p "$ROOTFS_DIR/etc"
    cp -L "$f" "$ROOTFS_DIR/etc/"
  fi
done

# Claude package (includes its node_modules)
mkdir -p "$ROOTFS_DIR/usr/local/lib/node_modules"
cp -a "$CLAUDE_PKG_DIR" "$ROOTFS_DIR/usr/local/lib/node_modules/"

# Build tar (ustar allows longer names)
rm -f "$OUT_TAR"
tar --format=ustar -C "$ROOTFS_DIR" -cf "$OUT_TAR" .

echo "Rootfs tar created at $OUT_TAR"
