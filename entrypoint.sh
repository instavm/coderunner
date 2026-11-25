#!/bin/bash
set -u

# What the owner would miss if the VM were created again (an image upgrade, `coderunner sandbox reset`) lives on
# the data volume mounted at /data, which outlives the VM: the desktop Chrome's profile (its sign-ins), its
# Downloads, and /workspace when nothing from the host is mounted there. Each path becomes a link into /data; what
# was already there moves over the first time. Without a data volume (a sandbox created before it) nothing changes.
DATA="${CODERUNNER_DATA:-/data}"
keep() {    # keep <path> <name under $DATA> <owner or empty>
    local path="$1" store="$DATA/$2" owner="$3" copied=""
    mkdir -p "$store" || return 0
    if [ -L "$path" ]; then
        [ "$(readlink "$path")" = "$store" ] && return 0
        rm -f "$path"
    elif [ -d "$path" ]; then
        if [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
            if [ -z "$(ls -A "$store" 2>/dev/null)" ]; then
                cp -a "$path/." "$store/" || return 0          # a full disk: leave it where it is
                rm -rf "$path"
                copied=1
            else
                mv "$path" "$path.before-data-volume-$(date +%s)" || return 0
            fi
        else
            rmdir "$path" || return 0
        fi
    fi
    mkdir -p "$(dirname "$path")"
    ln -s "$store" "$path" || return 0
    if [ -n "$owner" ]; then
        chown -h "$owner:$owner" "$path" 2>/dev/null
        chown ${copied:+-R} "$owner:$owner" "$store" 2>/dev/null
    fi
    return 0
}
if [ -d "$DATA" ] && { [ -n "${CODERUNNER_DATA:-}" ] || awk -v d="$DATA" '$2 == d {found=1} END {exit !found}' /proc/mounts 2>/dev/null; }; then
    desktop_user="${CODERUNNER_DESKTOP_USER:-box}"
    home="${CODERUNNER_DESKTOP_HOME:-$(getent passwd "$desktop_user" 2>/dev/null | cut -d: -f6)}"
    if [ -n "$home" ]; then
        keep "$home/.config/coderunner-chrome" chrome-profile "$desktop_user"
        keep "$home/Downloads" downloads "$desktop_user"
        workspace_owner="$desktop_user"
    else
        workspace_owner=""
    fi
    # Host folders mounted at or under /workspace stay as they are: they already live on the host.
    workspace="${CODERUNNER_WORKSPACE:-/workspace}"
    if ! awk -v w="$workspace" '$2 == w || index($2, w "/") == 1 {found=1} END {exit !found}' /proc/mounts 2>/dev/null; then
        keep "$workspace" workspace "$workspace_owner"
    fi
fi

# Caches go at every start. Kept, they only grow the VM's disk, a file on the owner's Mac whose blocks are never
# given back; whatever needs them builds them again. /tmp, the pip, npm and apt caches, Chrome's disk caches and
# the desktop's logs; never the Chrome profile itself (its sign-ins), Downloads or /workspace. Run only inside the
# sandbox, or on a test's CODERUNNER_TIDY_ROOT.
tidy_caches() {
    local root="$1" home="${CODERUNNER_DESKTOP_HOME:-}" dir
    if [ -z "$home" ]; then
        home="$(getent passwd "${CODERUNNER_DESKTOP_USER:-box}" 2>/dev/null | cut -d: -f6)"
        [ -n "$home" ] && home="$root$home"
    fi
    find "$root/tmp" -xdev -mindepth 1 -delete 2>/dev/null
    rm -rf "$root/root/.cache/pip" "$root/root/.npm/_cacache" "$root/var/lib/apt/lists/"* \
        "$root/var/cache/apt/archives/"*.deb "$root/var/cache/apt/"*.bin
    if [ -n "$home" ] && [ -d "$home" ]; then
        rm -rf "$home/.cache/pip" "$home/.cache/coderunner-chrome" "$home/.cache/google-chrome" "$home/.cache/"*.log
        for dir in "$home/.config/coderunner-chrome/"*/; do
            [ -d "$dir" ] && rm -rf "${dir}Cache" "${dir}Code Cache" "${dir}GPUCache"
        done
    fi
    return 0
}
TIDY_ROOT="${CODERUNNER_TIDY_ROOT:-}"
if [ -n "$TIDY_ROOT" ] || [ -f /app/server.py ]; then
    tidy_caches "$TIDY_ROOT"
fi

# Lite profile: no Jupyter server and no browser. server.py runs code as plain processes, so the
# sandbox fits in about 1 GB. The assistant browses with the owner's browser on the host.
if [ "${CODERUNNER_PROFILE:-full}" = "lite" ]; then
    mkdir -p /app/uploads/work
    exec uvicorn server:app --host 0.0.0.0 --port 8222 --workers 1 --no-access-log --ws-ping-interval 30 --ws-ping-timeout 600
fi

# Start Jupyter server. It has no authentication, so it must only listen on
# loopback: it is used exclusively by server.py inside this container.
# XSRF checks stay disabled because the API is called without cookies.
jupyter server \
  --ip=127.0.0.1 \
  --port=8888 \
  --no-browser \
  --IdentityProvider.token='' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.notebook_dir='/app/uploads' \
  --ServerApp.log_level='INFO' \
  --ServerApp.allow_root=True &

echo "Waiting for Jupyter Server to become available..."

max_wait=30
count=0

while ! curl -s --fail http://localhost:8888/api/status > /dev/null; do
    count=$((count + 1))

    if [ "$count" -gt "$max_wait" ]; then
        echo "Error: Jupyter Server did not start within ${max_wait} seconds."
        exit 1
    fi

    echo -n "."
    sleep 1
done

echo
echo "Jupyter Server is ready!"


# Start a Python3 kernel session and store the kernel ID
response=$(curl -s -X POST "http://localhost:8888/api/kernels" -H "Content-Type: application/json" -d '{"name":"python3"}')
kernel_id=$(echo "$response" | jq -r '.id')
echo "Python3 kernel started with ID: $kernel_id"

# Write the kernel ID to a file for later use
echo "$kernel_id" > /app/uploads/python_kernel_id.txt

# Playwright is used exclusively by server.py inside this container, so it
# must only listen on loopback.
python -m playwright run-server --port 3000 --host 127.0.0.1 &

# Start FastAPI application
exec uvicorn server:app --host 0.0.0.0 --port 8222 --workers 1 --no-access-log --ws-ping-interval 30 --ws-ping-timeout 600
