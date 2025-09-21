# CodeRunner REST API Implementation

This document describes the new REST API implementation that provides InstaVM-compatible code execution endpoints alongside the existing MCP server.

## Overview

This PR implements a comprehensive REST API that enables:

1. **Zero-config local code execution** - No API keys or setup required
2. **InstaVM-compatible interface** - Seamless migration path to cloud
3. **Session management** - Persistent execution contexts
4. **Multi-language support** - Python, bash, JavaScript execution
5. **Auto-container management** - Automatic Docker container lifecycle

## New Components

### 1. REST API Server (`api/rest_server.py`)

FastAPI server running on port 8223 alongside the existing MCP server (port 8222).

**Key Endpoints:**
- `POST /execute` - Synchronous code execution (InstaVM compatible)
- `POST /execute_async` - Asynchronous code execution 
- `POST /sessions` - Create execution session
- `GET /sessions/{id}` - Get session info
- `DELETE /sessions/{id}` - Close session
- `GET /health` - Health check
- `GET /languages` - List supported languages

### 2. Session Management (`core/session_manager.py`)

Manages persistent execution contexts with:
- Dedicated kernel allocation per session
- Automatic session cleanup (24-hour timeout)
- Session state tracking and health monitoring
- Integration with existing kernel pool

### 3. Language Processing (`core/language_processor.py`)

Multi-language execution support:
- **Python**: Default execution environment
- **Bash/Shell**: Commands prefixed with `%%bash`
- **JavaScript**: Node.js execution via `%%javascript`
- Automatic language detection and command preprocessing

### 4. Container Management (`container_manager.py`)

Auto-management of Docker containers:
- Automatic container startup and health checking
- Docker availability validation
- Platform-specific installation guidance
- Container lifecycle management (start/stop/remove)

### 5. CodeRunner Client (`__init__.py`)

Zero-configuration Python client:
- Auto-starts Docker container if needed
- InstaVM-compatible method signatures
- Comprehensive error handling
- Session management and health monitoring

### 6. Cloud Migration Support (`cloud/__init__.py`)

Seamless migration to InstaVM cloud:
- Import alias pattern: `from coderunner.cloud import InstaVM as CodeRunner`
- Identical interface between local and cloud execution
- Graceful fallback when InstaVM package not installed

## Usage Examples

### Local Execution (Zero Config)
```python
from coderunner import CodeRunner

# Auto-starts container, no setup required
runner = CodeRunner()

# Execute Python code
result = runner.execute("print('Hello World!')")
print(result['stdout'])  # "Hello World!"

# Execute bash commands  
result = runner.execute("echo 'Hello Bash!'", language="bash")
print(result['stdout'])  # "Hello Bash!"
```

### Cloud Migration (1-2 Line Change)
```python
# Change import and add API key - everything else stays the same!
from coderunner.cloud import InstaVM as CodeRunner

runner = CodeRunner(api_key="your-key")
result = runner.execute("print('Hello Cloud!')")
```

### Session Management
```python
with CodeRunner() as runner:
    # Persistent session across executions
    runner.execute("x = 5")
    result = runner.execute("print(x)")
    print(result['stdout'])  # "5"
```

## API Compatibility

The REST API implements the same interface as InstaVM for seamless migration:

| Endpoint | Method | InstaVM Compatible | Description |
|----------|--------|-------------------|-------------|
| `/execute` | POST | ✅ | Synchronous execution |
| `/execute_async` | POST | ✅ | Asynchronous execution |
| `/sessions` | POST | ✅ | Create session |
| `/sessions/{id}` | GET/DELETE | ✅ | Session management |
| `/health` | GET | ✅ | Health check |

## Architecture Changes

### Container Configuration
- **Port 8223**: New REST API server
- **Port 8222**: Existing MCP server (unchanged)
- **Port 8888**: Jupyter server (unchanged)
- **Port 3000**: Playwright server (unchanged)

### Process Flow
```
entrypoint.sh
├── Start Jupyter Server (port 8888)
├── Start Playwright Server (port 3000) 
├── Start REST API Server (port 8223) [NEW]
└── Start MCP Server (port 8222) [EXISTING]
```

### Integration Points
- REST API uses existing `kernel_pool` for execution
- Session manager allocates dedicated kernels
- Language processor handles multi-language support
- All existing MCP functionality remains unchanged

## Testing

Comprehensive test suite covering:
- **Language processor**: Normalization, detection, preprocessing
- **Container manager**: Docker integration, health checks, lifecycle
- **CodeRunner client**: Execution, sessions, error handling
- **API endpoints**: Request/response validation, error cases

Run tests:
```bash
pytest tests/ -v
```

## Deployment

### Docker Updates
- `Dockerfile`: Expose port 8223
- `entrypoint.sh`: Start REST API server alongside MCP
- `requirements.txt`: Add new dependencies

### Local Development
```bash
# Start container with new API
./install.sh

# Test REST API
curl http://localhost:8223/health

# Test code execution
curl -X POST http://localhost:8223/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "print(\"Hello API!\")", "language": "python"}'
```

## Migration Benefits

### For New Users
- **Zero friction**: No signups, API keys, or setup
- **Instant start**: `pip install coderunner` and go
- **Full functionality**: Python, bash, session management

### For Existing InstaVM Users  
- **Easy testing**: Test workflows locally before cloud deployment
- **Cost optimization**: Local development reduces cloud usage
- **Simple migration**: 1-2 line import change

### For Development Workflows
- **Hybrid execution**: Local dev, cloud production
- **Consistent interface**: Same code works everywhere
- **Gradual adoption**: Migrate at your own pace

## Backward Compatibility

- ✅ **Existing MCP server unchanged** - All current functionality preserved
- ✅ **Container ports unchanged** - MCP still on 8222, Jupyter on 8888  
- ✅ **Dockerfile compatible** - New port 8223 added alongside existing
- ✅ **Dependencies minimal** - Only adds client-side packages

## Next Steps

This PR provides the foundation for:

1. **Browser automation integration** (Phase 2)
   - Enhanced Playwright API matching InstaVM
   - Browser session management
   - Screenshot and interaction capabilities

2. **Advanced features**
   - File upload/download endpoints
   - Real-time execution streaming
   - Enhanced async task management

3. **Integration improvements**
   - OpenAI function calling support
   - LangChain/LlamaIndex integrations
   - Enhanced error reporting

## Review Focus Areas

1. **REST API design** - InstaVM compatibility and error handling
2. **Session management** - Kernel allocation and cleanup logic  
3. **Container management** - Docker integration and platform support
4. **Client interface** - Method signatures and error handling
5. **Test coverage** - Core functionality and edge cases