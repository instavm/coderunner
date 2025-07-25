#!/bin/bash
# Start Jupyter server
jupyter server \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --IdentityProvider.token='' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.notebook_dir='/app/uploads' \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_credentials=True \
  --ServerApp.allow_remote_access=True \
  --ServerApp.log_level='INFO' \
  --ServerApp.allow_root=True &

echo "Waiting for Jupyter Server to become available..."

max_wait=30

# This while loop is a great pattern. The logic here is correct.
while ! curl -s --fail http://localhost:8888/api/status > /dev/null; do
    count=$((count + 1)) # More readable spacing

    # This 'if' statement will now work because $max_wait has a value
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
kernel_id=$(echo $response | jq -r '.id')
echo "Python3 kernel started with ID: $kernel_id"

# Write the kernel ID to a file for later use
echo $kernel_id > /app/uploads/python_kernel_id.txt

npx -y playwright@1.53.0 run-server --port 3000 --host 0.0.0.0 &

# exec python mcp_main.py

# Start FastAPI application
exec uvicorn server:app --host 0.0.0.0 --port 8222 --workers 1 --no-access-log