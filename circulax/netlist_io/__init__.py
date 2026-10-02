"""File-input adapter: parse and evaluate model cards, then build kfnetlist topology.

NetlistParse supplies syntax; kfnetlist owns circuit connectivity. Library
resolution retains card scopes and OSDI metadata until compilation provisions
models and passes a kfnetlist.Netlist to compile_circuit.
"""

from circulax.netlist_io.library import Library, ResolvedCircuit, ResolvedInstance
from circulax.netlist_io.syntax import NetlistError

__all__ = ["Library", "NetlistError", "ResolvedCircuit", "ResolvedInstance"]
