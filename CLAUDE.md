# CodeRunner Codebase Analysis

## Project Overview
CodeRunner is an MCP (Model Context Protocol) server that executes AI-generated code in a sandboxed Docker container environment. It provides secure, isolated code execution for AI models like Claude and ChatGPT without requiring file uploads to the cloud.

## Project Structure

```
coderunner/
├── server.py                    # Main MCP server with kernel pool management
├── Dockerfile                   # Container build configuration
├── entrypoint.sh               # Container initialization script
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker orchestration
├── install.sh                  # macOS installation script
├── cleanup.sh                  # Cleanup utilities
├── test-build.sh              # Build testing script
├── config.py                   # Configuration utilities
├── examples/
│   ├── claude_desktop/
│   │   ├── mcpproxy.py        # HTTP-to-stdio MCP proxy
│   │   └── *.example.json     # Configuration examples
│   └── openai_agents/
│       └── openai_client.py   # OpenAI agents integration
├── tests/
│   └── test_kernel_manager.py # Test suite
├── uploads/                    # Shared container directory
├── images/                     # Documentation assets
└── htmlcov/                   # Test coverage reports
```

## Architecture

### Core Components

1. **Enhanced MCP Server** (`server.py:31`)
   - FastMCP-based server with kernel pool management
   - Two primary tools: `execute_python_code` and `navigate_and_get_all_visible_text`
   - Advanced WebSocket handling with retry logic and health monitoring
   - Progress streaming for long-running operations

2. **Kernel Pool Management** (`server.py:106-308`)
   - Dynamic kernel pool with configurable min/max limits (2-5 kernels)
   - Health monitoring with automatic recovery
   - Exponential backoff retry mechanisms
   - Concurrent execution support

3. **Containerized Execution Environment**
   - Docker-based sandboxing with Debian base (Python 3.13.3)
   - Pre-configured Jupyter server on port 8888
   - Playwright automation server on port 3000
   - Shared volume at `/app/uploads` for file exchange

4. **Integration Adapters**
   - `mcpproxy.py`: HTTP-to-stdio MCP bridge for Claude Desktop
   - `openai_client.py`: SSE transport for OpenAI agents

## Key Code Paths

### Enhanced Python Code Execution Flow
1. **Entry Point**: `execute_python_code()` function (`server.py:476`)
2. **Kernel Pool Management**: `kernel_pool.get_available_kernel()` (`server.py:143`)
3. **Retry Logic**: `execute_with_retry()` with exponential backoff (`server.py:346`)
4. **Kernel Execution**: `_execute_on_kernel()` with WebSocket communication (`server.py:378`)
5. **Message Protocol**: Enhanced Jupyter execute_request creation (`server.py:313`)
6. **Progress Streaming**: Real-time progress reporting via MCP context (`server.py:401-452`)
7. **Health Monitoring**: Background health checks and recovery (`server.py:279-305`)
8. **Resource Management**: Automatic kernel release and cleanup (`server.py:171-194`)

### Web Scraping Flow
1. **Entry Point**: `navigate_and_get_all_visible_text()` function (`server.py:501`)
2. **Playwright Connection**: Connects to Chromium via WebSocket at `ws://127.0.0.1:3000/` (`server.py:514`)
3. **Page Navigation**: Automated browser navigation to target URL (`server.py:516`)
4. **Content Extraction**: BeautifulSoup processing for visible text extraction (`server.py:519`)
5. **Resource Cleanup**: Proper browser instance closure (`server.py:522`)

### Container Initialization Flow
1. **Jupyter Server Startup**: Configures and launches server on port 8888 (`entrypoint.sh:3-14`)
   - Disables XSRF protection for container environment
   - Allows all origins and remote access
   - Sets notebook directory to `/app/uploads`
2. **Health Check Loop**: Waits up to 30 seconds for Jupyter readiness (`entrypoint.sh:21-32`)
3. **Python Kernel Bootstrap**: Creates initial Python3 kernel session (`entrypoint.sh:39-44`)
4. **Kernel ID Persistence**: Stores kernel ID in `/app/uploads/python_kernel_id.txt` (`entrypoint.sh:44`)
5. **Playwright Server**: Launches automation server on port 3000 (`entrypoint.sh:46`)
6. **MCP Server Launch**: Starts FastAPI application on port 8222 (`entrypoint.sh:51`)

## Configuration

### System Dependencies (`Dockerfile:13-44`)
- **Base Image**: Python 3.13.3 on Debian
- **System Packages**: systemd, ffmpeg, build-essential, curl, sudo
- **Browser Dependencies**: Chromium, X11 libraries, audio libraries
- **Development Tools**: Node.js 22.x, cargo, openssh-client/server
- **Security**: Machine-id cleanup, SSH host key generation (commented)

### Python Dependencies (`requirements.txt`)
- **MCP Framework**: `fastmcp`, `mcp[cli]` for protocol implementation
- **Web Framework**: `fastapi`, `uvicorn[standard]` for HTTP server
- **Jupyter Stack**: `jupyter-server`, `bash_kernel` for code execution
- **Browser Automation**: `playwright==1.53.0` for web scraping
- **HTTP/WebSocket**: `websockets`, `httpx`, `aiofiles` for async communication
- **AI Integration**: `openai`, `openai-agents` for LLM connectivity
- **Data Processing**: `beautifulsoup4` for HTML parsing

### Runtime Configuration
- **MCP Server**: Host `0.0.0.0`, Port `8222` (`Dockerfile:74-75`)
- **Jupyter Server**: Port `8888`, token-less authentication (`entrypoint.sh:7-8`)
- **Playwright Server**: Port `3000`, headless browser automation (`entrypoint.sh:46`)
- **Kernel Pool**: 2-5 concurrent kernels, 5-minute timeout (`server.py:34-38`)
- **WebSocket Settings**: 10-minute timeout, 30-second ping interval (`server.py:46-48`)

### Integration Examples
- **Claude Desktop**: HTTP-to-stdio MCP proxy with DNS resolution (`examples/claude_desktop/mcpproxy.py`)
- **OpenAI Agents**: Server-Sent Events transport with request tracing (`examples/openai_agents/openai_client.py`)
- **Configuration Templates**: JSON examples for various integration scenarios

## Advanced Features

### Kernel Pool Management
- **Dynamic Scaling**: Automatic kernel creation/destruction based on demand
- **Health Monitoring**: Background health checks every 30 seconds (`server.py:279`)
- **Failure Recovery**: Exponential backoff retry with kernel replacement
- **Concurrent Execution**: Support for multiple simultaneous code executions
- **Resource Limits**: Configurable pool size (2-5 kernels) and timeouts

### Error Handling & Resilience
- **Custom Exception Hierarchy**: `KernelError`, `NoKernelAvailableError`, `KernelExecutionError`, `KernelTimeoutError` (`server.py:67-81`)
- **Retry Logic**: Up to 3 attempts with exponential backoff (`server.py:346`)
- **WebSocket Resilience**: Connection recovery and timeout handling
- **Progress Reporting**: Real-time execution status via MCP context

### Performance Optimizations
- **Adaptive Timeouts**: Dynamic WebSocket timeouts based on activity (`server.py:406-418`)
- **Connection Reuse**: Persistent kernel sessions across requests
- **Async Architecture**: Full async/await implementation for concurrency
- **Resource Monitoring**: Kernel state tracking and cleanup

## Potential Improvements

### Performance Enhancements
1. **Browser Connection Pooling** (`server.py:513`)
   - Current: New Playwright instance per web scraping request
   - Suggested: Persistent browser pool with session reuse

2. **Configuration Management**
   - Current: Hardcoded URLs and timeouts (`server.py:42-48`)
   - Suggested: Environment-based configuration with `config.py` integration

3. **Shared Utility Functions**
   - Current: DNS resolution duplicated across integration files
   - Suggested: Extract common utilities to shared module

### Security Architecture

1. **Container Isolation**
   - Docker-based sandboxing with resource limits
   - Disabled machine-id for container security (`Dockerfile:71`)
   - No persistent state outside shared volume

2. **Jupyter Security Model** (`entrypoint.sh:7-10`)
   - Token-less authentication (acceptable in isolated environment)
   - XSRF protection disabled for container-internal communication
   - All-origins access mitigated by container network isolation

3. **Network Security**
   - Services bound to container-internal interfaces
   - No direct external exposure without explicit port mapping
   - WebSocket connections limited to localhost addresses

### Monitoring & Observability

1. **Comprehensive Logging**
   - Structured logging with timestamps and levels (`server.py:25-28`)
   - Kernel operation tracking and error reporting
   - WebSocket connection state monitoring

2. **Health Monitoring**
   - Periodic kernel health checks (`server.py:247-277`)
   - Automatic unhealthy kernel replacement
   - Resource utilization tracking

3. **Error Tracking**
   - Detailed exception logging with stack traces
   - Kernel failure count and recovery metrics
   - WebSocket connection error categorization

### Testing Infrastructure

1. **Current Test Suite**
   - Kernel manager tests in `tests/test_kernel_manager.py`
   - Coverage reports generated in `htmlcov/` directory
   - Build validation script: `test-build.sh`

2. **Coverage Areas**
   - Kernel pool management and lifecycle
   - MCP tool functionality validation
   - WebSocket communication reliability
   - Integration testing with AI platforms

### Documentation & Examples

1. **Integration Documentation**
   - Claude Desktop configuration examples (`examples/claude_desktop/`)
   - OpenAI agents integration patterns (`examples/openai_agents/`)
   - Docker setup and deployment guides (`DOCKER_SETUP.md`)

2. **Visual Documentation**
   - Architecture diagrams and screenshots (`images/` directory)
   - Demo workflows and UI examples
   - Banner and branding assets

## Development Workflow

### Container Management
```bash
# Build and start container
sudo ./install.sh

# Test build without deployment
./test-build.sh

# Check service status
curl -s http://localhost:8888/api/status   # Jupyter health
curl -s http://localhost:8222/health       # MCP server health

# View logs
docker logs coderunner

# Stop and cleanup
./cleanup.sh
```

### Integration Setup
```bash
# Claude Desktop integration
cp examples/claude_desktop/claude_desktop_config.example.json \
   ~/Library/Application\ Support/Claude/claude_desktop_config.json

# OpenAI agents integration
export OPENAI_API_KEY="your-key"
python examples/openai_agents/openai_client.py

# Docker Compose deployment
docker-compose up -d
```

### Testing & Validation
```bash
# Run test suite
python -m pytest tests/

# Generate coverage report
python -m pytest --cov=. --cov-report=html

# View coverage results
open htmlcov/index.html
```

## Key Files Reference

### Core Components
- `server.py`: Enhanced MCP server with kernel pool management
- `Dockerfile`: Multi-stage container build with Python 3.13.3
- `entrypoint.sh`: Container initialization and service orchestration
- `requirements.txt`: Python dependency specification
- `docker-compose.yml`: Multi-container orchestration

### Integration Modules
- `examples/claude_desktop/mcpproxy.py`: HTTP-to-stdio MCP bridge
- `examples/openai_agents/openai_client.py`: SSE transport for OpenAI
- `config.py`: Configuration management utilities

### Deployment & Testing
- `install.sh`: macOS container deployment script
- `cleanup.sh`: Environment cleanup utilities
- `test-build.sh`: Build validation and testing
- `tests/test_kernel_manager.py`: Kernel management test suite

### Documentation
- `README.md`: Project overview and quick start
- `CONTRIBUTING.md`: Development guidelines
- `DOCKER_SETUP.md`: Detailed deployment instructions
- `OpenCode.md`: Additional project documentation

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Container                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Jupyter Server │  │   MCP Server     │  │ Playwright      │ │
│  │   Port: 8888    │  │   Port: 8222     │  │  Port: 3000     │ │
│  │                 │  │                  │  │                 │ │
│  │ ┌─────────────┐ │  │ ┌──────────────┐ │  │ ┌─────────────┐ │ │
│  │ │   Kernel    │ │  │ │ Kernel Pool  │ │  │ │  Chromium   │ │ │
│  │ │   Pool      │ │  │ │ Manager      │ │  │ │  Browser    │ │ │
│  │ │  (2-5 max)  │ │  │ │              │ │  │ │             │ │ │
│  │ └─────────────┘ │  │ └──────────────┘ │  │ └─────────────┘ │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
│           │                     │                     │         │
│           └─────────────────────┼─────────────────────┘         │
│                                 │                               │
└─────────────────────────────────┼───────────────────────────────┘
                                  │
                          ┌───────▼────────┐
                          │   Host System  │
                          │                │
                          │ ┌────────────┐ │
                          │ │    AI      │ │
                          │ │  Clients   │ │
                          │ │            │ │
                          │ │ • Claude   │ │
                          │ │ • ChatGPT  │ │
                          │ │ • Others   │ │
                          │ └────────────┘ │
                          └────────────────┘
```

## Execution Flow Diagram

```
AI Client Request
       │
       ▼
┌─────────────┐    HTTP/WebSocket    ┌─────────────┐
│ MCP Server  │◄────────────────────►│ Kernel Pool │
│ (FastMCP)   │                      │ Manager     │
└─────────────┘                      └─────────────┘
       │                                     │
       │ Get Available Kernel               │
       ▼                                     ▼
┌─────────────┐                     ┌─────────────┐
│ Retry Logic │                     │   Jupyter   │
│ (3 attempts)│                     │   Kernel    │
└─────────────┘                     └─────────────┘
       │                                     │
       │ Execute Code                        │
       ▼                                     ▼
┌─────────────┐    WebSocket Msgs    ┌─────────────┐
│  Progress   │◄────────────────────►│   Python    │
│  Reporting  │                      │ Execution   │
└─────────────┘                      └─────────────┘
       │                                     │
       │ Stream Results                      │
       ▼                                     ▼
┌─────────────┐                     ┌─────────────┐
│  Response   │                     │   Output    │
│ to Client   │◄────────────────────│ Collection  │
└─────────────┘                     └─────────────┘
```