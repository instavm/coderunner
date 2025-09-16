# Docker Multi-Platform Setup

This document explains how to set up Docker builds and deployment for CodeRunner across multiple platforms.

## Required Secrets for GitHub Actions

To enable automatic Docker Hub publishing, add these secrets to your GitHub repository:

### Setting up Docker Hub Secrets

1. **Go to your Docker Hub account**:
   - Visit [hub.docker.com](https://hub.docker.com)
   - Go to Account Settings → Security → New Access Token
   - Create a token with Read, Write, Delete permissions

2. **Add secrets to GitHub repository**:
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Add these repository secrets:

```
DOCKER_USERNAME: your-dockerhub-username
DOCKER_TOKEN: your-dockerhub-access-token
```

## Workflow Triggers

The Docker build workflow runs on:
- **Push to main/develop**: Builds and pushes `docker-latest` tag
- **Pull Requests**: Builds only (no push) for testing
- **Releases**: Builds and pushes version tags like `v1.0.0-docker`

## Docker Usage

### Quick Start with Docker

```bash
# Run CodeRunner with Docker
docker run -d \
  --name coderunner \
  -p 8222:8222 \
  -v ./uploads:/app/uploads \
  instavm/coderunner:docker-latest

# Access MCP server
curl http://localhost:8222/mcp
```

### Docker Compose (Recommended)

```bash
# Use the provided docker-compose.yml
docker compose up -d

# View logs
docker compose logs -f

# Stop service
docker compose down
```

### Environment Variables

- `FASTMCP_HOST`: Server host (default: 0.0.0.0)
- `FASTMCP_PORT`: Server port (default: 8222)

## Platform Support

| Platform | Container Tool | Image Tag | Registry |
|----------|----------------|-----------|----------|
| macOS (Apple Silicon) | Apple `container` | `apple-latest` | OCI-compatible |
| Linux (x64/ARM64) | Docker | `docker-latest` | Docker Hub |
| Windows | Docker Desktop | `docker-latest` | Docker Hub |

## Integration Examples

### With Claude Desktop (Docker)
```json
{
  "mcpServers": {
    "coderunner": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-p", "8222:8222",
        "-v", "./uploads:/app/uploads",
        "instavm/coderunner:docker-latest"
      ]
    }
  }
}
```

### With OpenAI Agents
```python
# Update openai_client.py to use Docker endpoint
hostname = "localhost"  # Instead of coderunner.local
address = "127.0.0.1"
url = f"http://{address}:8222/mcp"
```

## Building Locally

```bash
# Standard Docker build
docker build -t coderunner-local .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 -t coderunner-multiarch .
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change port mapping `-p 8223:8222`
2. **Volume permissions**: Ensure `./uploads` directory exists and is writable
3. **Memory issues**: Add memory limits `--memory 4g`

### Health Checks

```bash
# Check container status
docker ps

# Check logs
docker logs coderunner

# Test MCP endpoint
curl -f http://localhost:8222/mcp || echo "MCP not ready"
```