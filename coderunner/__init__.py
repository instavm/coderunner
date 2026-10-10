"""Host-side runtime for CodeRunner.

Everything in this package runs on the user's machine, outside the sandbox VM.
It owns the trusted pieces: model runtime, policy and approvals, audit log,
secrets, the local browser broker, connectors and messaging channels. Code in
the sandbox can reach only the narrow gateway API and never holds the admin
credential.
"""

__version__ = "0.2.0"
