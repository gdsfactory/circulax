# Disciplines and physical tolerances (proposal)

Circulax can associate physical quantities with unknowns and equations. Opt into
this proposal with `compile_circuit(..., tolerances=TolerancePolicy())`. The
scalar `rtol` / `atol` interface remains available for existing models. Combining
scalar tolerances with a policy is an error.

```python
from circulax import TolerancePolicy, compile_circuit

circuit = compile_circuit(netlist, models, tolerances=TolerancePolicy(
    rtol=1e-6,
    atol={"voltage": 1e-6, "current": 1e-12},
    local_error_factor=10,
))
y, converged = circuit.dc_checked()
report = circuit.dc_diagnostics(y)
```

A successful physical solve requires both a small **undamped Newton correction**
and a small **physical equation residual** at the returned candidate. This avoids
accepting a tiny damped step while the equations remain unsatisfied. Updates use
`atol + rtol * max(abs(y_new), abs(y_old))`. Residuals use
`atol + rtol * nature.nominal`; `nominal=0` means absolute-only residual checking.
No relative residual magnitude is inferred from a nearly cancelled net residual.

The units of the unknown and its equation often differ. An electrical node is a
voltage unknown governed by a current-balance equation; an inductor's current
state is governed by a voltage equation. Real and imaginary parts are paired
before computing norms, including HB collocation samples.

## User-defined models

Definitions are immutable values and need no registry. Units document the model's
contract; no unit conversion or symbolic dimensional checking is performed.

```python
from circulax import Nature, Discipline
from circulax.components.base_component import component

velocity = Nature("velocity", units="m/s", atol=1e-8)
force = Nature("force", units="N", atol=1e-10)
mechanical = Discipline("mechanical", potential=velocity, flow=force)

@component(ports=("p", "n"), discipline=mechanical)
def Damper(signals, drag=2.0):
    force = drag * (signals.p - signals.n)
    return {"p": force, "n": -force}, {}
```

`discipline=` supplies a default for all ports. `port_disciplines=` overrides
individual ports, for example electrical and thermal ports on the same heater.
`state_natures=` declares each internal unknown; `equation_natures=` independently
declares each state equation. These keywords also work on `@source`.

```python
@component(
    ports=("p", "n"), states=("i",), discipline=electrical,
    state_natures={"i": current}, equation_natures={"i": voltage},
)
def Inductor(signals, L=1e-3):
    return {"p": signals.i, "n": -signals.i, "i": signals.p-signals.n}, {"i": -L*signals.i}
```

Connected non-reference ports must declare the same discipline. The shared zero
reference can serve different domains. Diagnostics replace its penalty residual
with its potential error. A thermal zero reference represents temperature rise;
absolute ambient temperature must be represented explicitly in the model.

Nature names identify policy overrides. Different definitions sharing a name
are rejected; use distinct names for specializations. All states and ports must
be annotated in physical mode. Unannotated models remain usable in scalar mode.

## Overrides and diagnostics

Nature overrides in `TolerancePolicy(atol={...})` apply to both unknowns and
equations of that nature. `unknown_atol={"L1,i_L": ...}` and
`residual_atol={"L1,i_L": ...}` override the corresponding exact instance-local
name separately. All connected names refer to the same node; contradictory
node overrides and unknown names are errors. Top-level netlist aliases are not
accepted as policy override keys. Ground tolerances come from the strictest
connected potential nature.

`circuit.convergence_layout` exposes resolved arrays, canonical diagnostic names,
and separate unknown/equation natures. `circuit.dc_diagnostics(y, params=...)`
recomputes normalized residuals and correction error; use the same parameters as
the solve. A finite estimate is not evidence of convergence: use `dc_checked`.

## Transient and harmonic balance

All native BE, BDF2, trapezoidal, and SDIRK3 variants use the same policy for
nonlinear termination, with each method's discretized equations. Initialization
uses the circuit policy too. Rejected nonlinear steps retain existing solver
failure handling and delay-history behavior.

For adaptive local-error control, use:

```python
solution = circuit.transient(
    t0=0, t1=1e-3, dt0=1e-6,
    stepsize_controller=circuit.adaptive_controller(),
)
```

This PID controller uses unknown-nature tolerances and a maximum norm. Complex
unknowns share a magnitude-based denominator across their real/imaginary pair.
`local_error_factor` independently scales its absolute and relative bounds.
Explicit controllers retain their own settings. The default remains constant
stepping. Neither nonlinear convergence nor local-error acceptance establishes
global waveform accuracy: check timestep refinement and the error estimator.

HB receives the same physical checks across time samples and raises on failed
physical convergence. This does not measure harmonic truncation error; use
independent harmonic-refinement checks before relying on a periodic solution.

## Coverage and migration

The prototype annotates native electrical primitives, optical passive devices,
photonic sources/waveguides/couplers, and photodetectors, including their normalized
carrier states. Optical pseudo-potential, pseudo-flow, and wave-amplitude natures
are distinct from electrical quantities. Their metadata follows the existing
power-wave stamps; `z0` changes the physical solution coordinates, not the units.

OSDI adapters, SAX wrappers, generated rational factories, and some composite
models do not yet supply metadata and are rejected in policy mode. No guessed
fallback is used. Potential-only signal-flow disciplines, contribution-based
relative residual references, domain-aware damping, exact ground elimination,
and additional integration-error estimators remain follow-up work. Jacobian-only
`g_leak` regularization remains numerical and is excluded from physical residuals.

This is an opt-in proposal rather than an immediate major-version switch.
Switching the default later would be breaking: user models would need metadata,
connections could fail discipline validation, and formerly accepted estimates
could fail the stronger checks. The two notebooks under `examples/disciplines/`
show the proposed migration and custom-domain workflow.

When supplying transient `y0`, satisfy algebraic source constraints at `t0`. An
ideal-source discontinuity in an inconsistent initial state can prevent adaptive
error control from accepting any step. The notebooks show consistent startup.


## Relationship to issue #67

This proposal implements the metadata and weighted-convergence portion of
[issue #67](https://github.com/gdsfactory/circulax/issues/67). Component-local
state unknown natures and equation natures are independent, so adapters can
identify electrical branch currents directly from their declared current nature.
The opt-in policy changes convergence criteria, not the existing damping or
regularization algorithm.

The remaining issue scope is intentionally separate: establish DC iteration
baselines for direct, GMIN, and source-stepping solves; implement physical GMIN
stamps in both F and the Jacobian on eligible potential nodes; and measure
per-unknown limiting independently. This change makes no iteration-count or
speedup claim. Tighter current bounds may increase the number of iterations.
Issue #67 should remain open for those follow-ups.
