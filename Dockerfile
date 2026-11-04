# Node.js for the sandbox, taken from the official image. Images are pinned by
# digest; Dependabot keeps the digests current.
FROM node:22.23.4-bookworm-slim@sha256:4b150b48b4116e2a22f2c39cc27025fbf4f153e2078fd3b14d159fdb5c7279da AS node

# Use the specified standard Python 3.13.3 base image (Debian-based)
FROM python:3.13.3@sha256:0bc836167214f98aca9c9bca7b4c6dc2c2a77f4a29d5029e6561a14706335102

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
    cargo \
    xvfb \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxshmfence1 \
    libasound2 \
    unzip \
    p7zip-full \
    bc \
    ripgrep \
    fd-find \
    sqlite3 \
    libsqlite3-dev \
    wkhtmltopdf \
    poppler-utils \
    default-jre \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


# Copy requirements file (generated from requirements.in, every package pinned with hashes)
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --require-hashes -r requirements.txt


# Install the bash kernel spec for Jupyter (not working with uv)
RUN python -m bash_kernel.install


# Copy the application code (server.py)
COPY ./server.py /app/server.py

# Create application/jupyter directories
RUN mkdir -p /app/uploads /app/jupyter_runtime

# Copy skills directory structure into the container
# Public skills are baked into the image
# User skills directory is created as mount point for user-added skills
COPY ./skills/public /app/uploads/skills/public
RUN mkdir -p /app/uploads/skills/user

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

# Ensure Node.js, npm (and npx) are set up
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx



ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Browsers for the Playwright version in requirements.txt. The same package runs
# the Playwright server (entrypoint.sh), so client and server versions always match.
RUN python -m playwright install

# Copy the entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh






# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
