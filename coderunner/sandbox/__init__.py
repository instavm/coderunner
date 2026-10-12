"""The isolated VM where the assistant runs code.

  drivers   start, stop and locate the VM with the platform's tool (Apple container, Docker)
  client    run code and manage sessions through the server inside the VM
"""

from .client import SandboxClient
from .drivers import AppleDriver, DockerDriver, Driver, driver_for

__all__ = ["AppleDriver", "DockerDriver", "Driver", "SandboxClient", "driver_for"]
