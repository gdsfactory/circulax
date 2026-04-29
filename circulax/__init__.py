"""circulax: A differentiable, JAX based circuit simulator."""

from circulax._version import __version__
from circulax.circuit import Circuit, compile_circuit
from circulax.compiler import compile_netlist
from circulax.netlist import build_net_map, netlist
from circulax.netlist import circulaxNetlist as Netlist
from circulax.s_transforms import fdomain_component
from circulax.solvers import analyze_circuit, setup_ac_sweep, setup_harmonic_balance, setup_transient
from circulax.testbench import attach_testbench
from circulax.utils import apply_global_params, update_group_params, update_params_dict
