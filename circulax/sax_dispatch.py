"""Detect purely linear, S-parameter circuits and dispatch them to SAX.

A netlist whose every model is a plain SAX model function describes a linear,
memoryless, frequency-domain circuit — exactly the class SAX composes
directly, without ever assembling a nodal system. Circulax can simulate those
too (``sax_component`` stamps each S-matrix as an admittance and the global
sparse solve does the composition), but going through the DAE machinery to get
an answer SAX reaches by matrix composition costs roughly two orders of
magnitude in evaluation time, plus an XLA compile per call shape.

This module is the detection half of that shortcut. :func:`check_sax_dispatch`
decides whether a netlist qualifies and, when it does not, says which instance
disqualified it; :func:`build_sax_circuit` produces the SAX evaluator.
:meth:`circulax.Circuit.sdict` consumes both, so callers keep one API and get
the faster backend automatically.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import kfnetlist as kfnl

# Instances circulax adds for nodal analysis that have no SAX counterpart. A
# netlist containing one is describing a *driven* circuit, not a scattering
# object, so there is no S-matrix for SAX to compose.
_NON_SAX_COMPONENTS = frozenset({"ground"})


@dataclass(frozen=True)
class SaxDispatch:
    """Whether a netlist can be evaluated by SAX, and why not if it cannot.

    Attributes:
        dispatchable: ``True`` when every model is a SAX model and the netlist
            declares external ports.
        reasons: Human-readable blockers, empty when *dispatchable*. Surfaced
            by :attr:`circulax.Circuit.sax_dispatch` so a user who expected the
            fast path can see which instance cost them it.

    """

    dispatchable: bool
    reasons: tuple[str, ...] = field(default=())

    def __bool__(self) -> bool:
        """Allow ``if check_sax_dispatch(...):``."""
        return self.dispatchable


def _sax_model_of(obj: Any) -> Any | None:
    """Return the underlying SAX model function for *obj*, else ``None``.

    Accepts both entries as the user wrote them (a raw SAX model function) and
    entries already normalised by the compiler (a ``CircuitComponent`` subclass
    built by :func:`~circulax.s_transforms.sax_component`, which records the
    function it wrapped on ``_sax_model_fn``). Checking both means dispatch
    works whether it is asked before or after :func:`compile_netlist` has run.
    """
    from circulax.s_transforms import _is_sax_model

    if inspect.isclass(obj):
        return getattr(obj, "_sax_model_fn", None)
    return obj if _is_sax_model(obj) else None


def _netlist_as_dict(net: dict | kfnl.Netlist) -> dict:
    return net.to_dict() if isinstance(net, kfnl.Netlist) else net


def check_sax_dispatch(net_dict: dict | kfnl.Netlist | None, models_map: dict | None) -> SaxDispatch:
    """Decide whether *net_dict* can be evaluated by SAX instead of the nodal solver.

    Three conditions, all necessary:

    1. **Every instantiated model is a SAX model.** One nonlinear, stateful or
       Verilog-A component makes the circuit a DAE system that has no
       S-parameter representation at all, so the whole netlist falls back.
       Only models actually instantiated are checked — an unused nonlinear
       entry in *models_map* is not a blocker.
    2. **No nodal-only instances** (``ground``/``GND``). Their presence means
       the netlist is a driven test bench rather than a scattering object.
    3. **External ports are declared.** SAX composes an S-matrix *between
       declared ports*; without a ``ports`` mapping there is nothing to return.

    Args:
        net_dict: Netlist to inspect (SAX-format dict or ``kfnetlist.Netlist``).
            ``None`` — a circuit compiled without a retained source netlist —
            is not dispatchable.
        models_map: Model mapping as passed to :func:`compile_circuit`.

    Returns:
        A :class:`SaxDispatch` verdict carrying the blocking reasons.

    """
    if net_dict is None or models_map is None:
        return SaxDispatch(False, ("Circuit has no retained source netlist or models.",))

    net = _netlist_as_dict(net_dict)
    instances = net.get("instances", {})
    if not instances:
        return SaxDispatch(False, ("Netlist declares no instances.",))

    reasons: list[str] = []
    if not net.get("ports"):
        reasons.append("Netlist declares no external 'ports'; SAX has no port set to build an S-matrix over.")

    for inst_name, inst in instances.items():
        comp_type = inst.get("component") if isinstance(inst, dict) else getattr(inst, "component", None)
        if comp_type in _NON_SAX_COMPONENTS or inst_name == "GND":
            reasons.append(f"Instance '{inst_name}' is a nodal-analysis component ('{comp_type}') with no SAX equivalent.")
            continue
        model = models_map.get(comp_type)
        if model is None:
            reasons.append(f"Instance '{inst_name}' references unknown model '{comp_type}'.")
        elif _sax_model_of(model) is None:
            reasons.append(f"Instance '{inst_name}' uses model '{comp_type}', which is not a SAX model.")

    return SaxDispatch(not reasons, tuple(reasons))


def build_sax_circuit(net_dict: dict | kfnl.Netlist, models_map: dict) -> Any:
    """Build the SAX evaluator for a netlist that passed :func:`check_sax_dispatch`.

    Args:
        net_dict: Netlist (SAX-format dict or ``kfnetlist.Netlist``).
        models_map: Model mapping; entries already wrapped as circulax
            components are unwrapped back to the SAX functions they came from,
            so a ``Circuit``'s post-compile models work here unchanged.

    Returns:
        The callable returned by :func:`sax.circuit`, mapping parameter
        keywords to a SAX S-dict.

    Raises:
        ValueError: If the netlist is not SAX-dispatchable.

    """
    verdict = check_sax_dispatch(net_dict, models_map)
    if not verdict:
        msg = "Netlist is not SAX-dispatchable: " + "; ".join(verdict.reasons)
        raise ValueError(msg)

    import sax

    sax_models = {}
    for name, model in models_map.items():
        fn = _sax_model_of(model)
        if fn is not None:
            sax_models[name] = fn

    circuit_fn, _ = sax.circuit(netlist=_netlist_as_dict(net_dict), models=sax_models)
    return circuit_fn
