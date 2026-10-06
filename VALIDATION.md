# Branch validation and limits

The bearer-authentication middleware has four isolated ASGI tests covering
anonymous/incorrect credentials, weak/missing server configuration, duplicate
headers, credential rotation, and unauthenticated WebSocket rejection. Shell
syntax, Python syntax, focused lint and diff whitespace checks were also run.
These checks do not start a guest, Jupyter, browser, installer or Docker image.

Docker host publication defaults to loopback in Compose and the documented
commands. The existing Apple container network modes remain unchanged.
Previously published images have not been rebuilt or deployed by this branch;
build and validate this checkout before using the new authentication contract.
Real MCP client, installer and image compatibility remain unverified.

A separate standalone local-inference experiment used cached llama.cpp b11146
and Qwen3-1.7B Q8_0 on Apple Silicon with disabled reasoning, one request slot,
4096 context and four CPU/batch threads. It became ready in approximately
61 seconds and reported approximately 66 generated tokens per second on short
synthetic responses. A plain arithmetic answer contained the correct sum 10
with extra formatting; the synthetic `report_sum` call returned 13 instead of
10. No tool was executed. The experiment therefore failed correctness/format
acceptance. A different model has not been tested or shown to fix this result.

Those measurements are not CodeRunner guest, agent-loop, browser, messaging,
GPU-passthrough or end-to-end assistant validation. No model weights or local
runtime receipts are included in this repository change.
