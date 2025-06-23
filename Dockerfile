# Use the specified standard Python 3.13.3 base image (Debian-based)
FROM python:3.13.3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies INCLUDING systemd
RUN apt-get update && apt-get install -y --no-install-recommends \
    systemd \
    sudo \
    curl \
    iproute2 \
    ffmpeg \
    bash \
    build-essential \
    procps \
    openssh-client \
    openssh-server \
    jq \
    kmod \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Install the bash kernel spec for Jupyter (not working with uv)
RUN python -m bash_kernel.install


# Copy the application code (server.py)
COPY ./server.py /app/server.py

# Create application/jupyter directories
RUN mkdir -p /app/uploads /app/jupyter_runtime

# # Generate SSH host keys
# RUN ssh-keygen -A

# Clean systemd machine-id
RUN rm -f /etc/machine-id && touch /etc/machine-id

# --- Set environment variables for the application ---
ENV FASTMCP_HOST="0.0.0.0"
ENV FASTMCP_PORT="8222"


# Expose the FastAPI port
EXPOSE 8222

# Start the FastAPI application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--no-access-log"]


# Copy the entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
