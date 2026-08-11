#!/bin/bash
set -u

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
node /usr/lib/node_modules/playwright/cli.js run-server --port 3000 --host 127.0.0.1 &

# Start FastAPI application
exec uvicorn server:app --host 0.0.0.0 --port 8222 --workers 1 --no-access-log
