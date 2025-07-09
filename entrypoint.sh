#!/bin/bash
set -euo pipefail

# Trap handlers for cleanup
cleanup() {
    echo "Shutting down services..."
    pkill -f "jupyter server" || true
    pkill -f "uvicorn" || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Configuration
JUPYTER_PORT=${JUPYTER_PORT:-8888}
JUPYTER_HOST=${JUPYTER_HOST:-0.0.0.0}
MAX_WAIT=${MAX_WAIT:-30}
SHARED_DIR=${SHARED_DIR:-/app/uploads}
FASTMCP_PORT=${FASTMCP_PORT:-8222}
FASTMCP_HOST=${FASTMCP_HOST:-0.0.0.0}

echo "Starting Jupyter server on ${JUPYTER_HOST}:${JUPYTER_PORT}..."

# Start Jupyter server
jupyter server \
  --ip="${JUPYTER_HOST}" \
  --port="${JUPYTER_PORT}" \
  --no-browser \
  --IdentityProvider.token='' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.notebook_dir="${SHARED_DIR}" \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_credentials=True \
  --ServerApp.allow_remote_access=True \
  --ServerApp.log_level='INFO' \
  --ServerApp.allow_root=True &

JUPYTER_PID=$!

echo "Waiting for Jupyter Server to become available..."

count=0
while ! curl -s --fail "http://localhost:${JUPYTER_PORT}/api/status" > /dev/null; do
    count=$((count + 1))
    
    if [ "$count" -gt "$MAX_WAIT" ]; then
        echo "Error: Jupyter Server did not start within ${MAX_WAIT} seconds."
        kill $JUPYTER_PID 2>/dev/null || true
        exit 1
    fi

    echo -n "."
    sleep 1
done

echo
echo "Jupyter Server is ready!"

# Start Python3 kernel session and store the kernel ID
echo "Starting Python3 kernel..."
python_response=$(curl -s -X POST "http://localhost:${JUPYTER_PORT}/api/kernels" \
    -H "Content-Type: application/json" \
    -d '{"name":"python3"}')

if [ $? -ne 0 ]; then
    echo "Error: Failed to start Python3 kernel"
    kill $JUPYTER_PID 2>/dev/null || true
    exit 1
fi

python_kernel_id=$(echo "$python_response" | jq -r '.id')
if [ "$python_kernel_id" == "null" ] || [ -z "$python_kernel_id" ]; then
    echo "Error: Failed to get Python kernel ID from response: $python_response"
    kill $JUPYTER_PID 2>/dev/null || true
    exit 1
fi

echo "Python3 kernel started with ID: $python_kernel_id"
echo "$python_kernel_id" > "${SHARED_DIR}/python_kernel_id.txt"

# Start Bash kernel session and store the kernel ID
echo "Starting Bash kernel..."
bash_response=$(curl -s -X POST "http://localhost:${JUPYTER_PORT}/api/kernels" \
    -H "Content-Type: application/json" \
    -d '{"name":"bash"}')

if [ $? -ne 0 ]; then
    echo "Warning: Failed to start Bash kernel (continuing without it)"
else
    bash_kernel_id=$(echo "$bash_response" | jq -r '.id')
    if [ "$bash_kernel_id" == "null" ] || [ -z "$bash_kernel_id" ]; then
        echo "Warning: Failed to get Bash kernel ID from response: $bash_response"
    else
        echo "Bash kernel started with ID: $bash_kernel_id"
        echo "$bash_kernel_id" > "${SHARED_DIR}/bash_kernel_id.txt"
    fi
fi

echo "Starting FastAPI application on ${FASTMCP_HOST}:${FASTMCP_PORT}..."

# Start FastAPI application
exec uvicorn server:app --host "$FASTMCP_HOST" --port "$FASTMCP_PORT" --workers 1 --no-access-log