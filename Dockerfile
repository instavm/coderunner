# Multi-stage build for optimized image size
FROM python:3.13.3 as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Runtime stage
FROM python:3.13.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Install runtime dependencies and tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Install the bash kernel spec for Jupyter
RUN python -m bash_kernel.install

# Copy application code
COPY server.py config.py jupyter_client.py /app/

# Create application directories
RUN mkdir -p /app/uploads /app/jupyter_runtime

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables for the application
ENV CODERUNNER_FASTMCP_HOST="0.0.0.0"
ENV CODERUNNER_FASTMCP_PORT="8222"
ENV CODERUNNER_JUPYTER_HOST="0.0.0.0"
ENV CODERUNNER_JUPYTER_PORT="8888"

# Expose the FastAPI port
EXPOSE 8222 8888

# Use tini for proper signal handling
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
