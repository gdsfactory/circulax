"""File-input adapter: parse and evaluate model cards, then build kfnetlist topology.

NetlistParse supplies syntax; kfnetlist owns circuit connectivity. Library
resolution retains card scopes and OSDI metadata until compilation provisions
models and passes a kfnetlist.Netlist to compile_circuit.
"""

from circulax.netlist_io.library import Library, ResolvedCircuit, ResolvedInstance
from circulax.netlist_io.model import LibraryModel, NetlistDefinition
from circulax.netlist_io.sources import parse_source, parse_waveform
from circulax.netlist_io.syntax import NetlistError

__all__ = [
    "Library",
    "LibraryModel",
    "NetlistDefinition",
    "NetlistError",
    "ResolvedCircuit",
    "ResolvedInstance",
    "parse_source",
    "parse_waveform",
]
