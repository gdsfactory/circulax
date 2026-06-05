"""Backward-compatibility shim — re-exports ``va_component`` from ``bosdi.circulax``.

The Verilog-A / JAX-component bridge lives in the ``bosdi`` package.
Existing imports keep working through this shim:

  - ``from circulax.components.va_component import va_component``  ✓

For new code, import from ``bosdi.circulax`` directly.
"""

from bosdi.circulax.va_component import (
    JacobianReturn,
    PhysicsReturn,
    _install_custom_jvp,
    va_component,
)

__all__ = ["JacobianReturn", "PhysicsReturn", "_install_custom_jvp", "va_component"]
