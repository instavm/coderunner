#!/bin/bash
# End-to-end smoke test for a running CodeRunner container.
#
# Usage: ./test-e2e.sh [host]
#   host defaults to coderunner.local
#
# Set EGRESS_BLOCKED=1 when testing CODERUNNER_NETWORK=none.

set -u

HOST="${1:-coderunner.local}"
BASE="http://$HOST:8222"
SKILLS_DIR="${SKILLS_DIR:-$HOME/.coderunner/assets/skills/user}"
PASS=0
FAIL=0

check() {
    local name="$1" expected="$2" actual="$3"
    if [ "$actual" = "$expected" ]; then
        echo "ok   $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL $name (expected: $expected, got: $actual)"
        FAIL=$((FAIL + 1))
    fi
}

echo "Testing CodeRunner at $BASE"
echo

# 1. Health endpoint
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")
check "health endpoint returns 200" "200" "$code"

# 2. Requests with a foreign Origin header are rejected
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Origin: http://evil.example" "$BASE/health")
check "foreign Origin is rejected" "403" "$code"

# 3. Requests with a foreign Host header are rejected
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Host: evil.example" "$BASE/health")
check "foreign Host is rejected" "403" "$code"

# 4. Code execution via the REST API
out=$(curl -s -X POST "$BASE/execute" -H "Content-Type: application/json" \
    -d '{"code": "print(6 * 7)"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["stdout"].strip())')
check "python execution works" "42" "$out"

# 5. Browser navigation works when outbound networking is enabled
if [ -z "${EGRESS_BLOCKED:-}" ]; then
    out=$(curl -s -X POST "$BASE/v1/browser/interactions/navigate" \
        -H "Content-Type: application/json" -d '{"url":"https://pypi.org"}' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    check "browser navigation works" "success" "$out"
fi

# 6. MCP endpoint accepts an initialize request
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
    -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}')
check "MCP initialize returns 200" "200" "$code"

# 7. Path traversal in skill file tools is rejected
sid=$(curl -s -D - -o /dev/null -X POST "$BASE/mcp" \
    -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}' \
    | awk 'tolower($1) == "mcp-session-id:" {print $2}' | tr -d '\r')
if [ -n "$sid" ]; then
    curl -s -o /dev/null -X POST "$BASE/mcp" \
        -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
        -H "mcp-session-id: $sid" \
        -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    out=$(curl -s -X POST "$BASE/mcp" \
        -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
        -H "mcp-session-id: $sid" \
        -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_skill_file","arguments":{"skill_name":"..","filename":"../../../../etc/passwd"}}}')
    case "$out" in
        *"root:"*) check "skill path traversal is rejected" "rejected" "leaked" ;;
        *)         check "skill path traversal is rejected" "rejected" "rejected" ;;
    esac
else
    check "skill path traversal is rejected (no MCP session)" "session" "none"
fi

# 8. Zip archives cannot write outside the user skills directory
if [ -d "$SKILLS_DIR" ]; then
    if [ -z "$sid" ]; then
        check "skill archive traversal is rejected (no MCP session)" "session" "none"
    else
        marker="$(dirname "$SKILLS_DIR")/zip-slip-check"
        rm -f "$marker"
        python3 - "$SKILLS_DIR/unsafe.zip" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr("../zip-slip-check", "unsafe")
PY
        curl -s -o /dev/null -X POST "$BASE/mcp" \
            -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
            -H "mcp-session-id: $sid" \
            -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_skills","arguments":{}}}'
        if [ -e "$marker" ]; then
            check "skill archive traversal is rejected" "rejected" "leaked"
            rm -f "$marker"
        else
            check "skill archive traversal is rejected" "rejected" "rejected"
        fi
        rm -f "$SKILLS_DIR/unsafe.zip"
    fi
fi

# 9. Jupyter must not be reachable from outside the container
code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 "http://$HOST:8888/api/status" 2>/dev/null)
check "jupyter is not exposed" "000" "$code"

# 10. Playwright server must not be reachable from outside the container
code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 "http://$HOST:3000/" 2>/dev/null)
check "playwright is not exposed" "000" "$code"

# Optional offline-network checks
if [ -n "${EGRESS_BLOCKED:-}" ]; then
    out=$(curl -s -X POST "$BASE/execute" -H "Content-Type: application/json" \
        -d '{"code": "import urllib.request\ntry:\n    urllib.request.urlopen(\"https://example.com\", timeout=10)\n    print(\"reachable\")\nexcept Exception:\n    print(\"blocked\")"}' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["stdout"].strip())')
    check "disallowed domain is blocked" "blocked" "$out"

    out=$(curl -s -X POST "$BASE/execute" -H "Content-Type: application/json" \
        -d '{"code": "import socket\ntry:\n    socket.gethostbyname(\"example.com\")\n    print(\"reachable\")\nexcept OSError:\n    print(\"blocked\")"}' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["stdout"].strip())')
    check "external DNS is unavailable" "blocked" "$out"
fi

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
