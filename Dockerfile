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


# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt


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
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get update && apt-get install -y nodejs



ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RUN npm install playwright@1.53.0 -g
RUN npx playwright@1.53.0 install

# --- AI Coding Agents ---
# Claude Code CLI (pinned to major version 1.x)
RUN npm install -g @anthropic-ai/claude-code@1

# OpenAI Codex CLI
RUN npm install -g @openai/codex

# Gemini CLI
RUN npm install -g @anthropic-ai/claude-code@1 && \
    pip install --no-cache-dir google-generativeai

# Cursor CLI (installs as 'agent' at ~/.local/bin)
RUN curl -fsSL https://cursor.com/install | bash && \
    ln -sf /root/.local/bin/agent /usr/local/bin/cursor

# --- Cloud CLIs ---
# AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# GitHub CLI (gh)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
    tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Copy the entrypoint script into the image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh






# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
