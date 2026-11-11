# Optical wave boundaries and observables

Current follow-up: [notebook observations and small-signal traveling-power derivatives](notebook_power_wave_observations.md). The scope and validation below describe this earlier milestone.

Explicit wave boundaries are available alongside the existing optical sources.
This describes the source/observable milestone of issue #112. The subsequent
[native passive migration](native_passive_power_waves.md) adds five algebraic
passives to this contract. The [delay milestone](bidirectional_optical_delays.md)
adds bidirectional delay support. The [mixed phase-shifter milestone](mixed_electro_optic_power_waves.md)
adds optical-terminal-specific observables for `phase_shifter_factory`. The
[detector milestone](detector_power_waves.md) adds reusable absorbed-power detectors
and notebook integration. Other mixed models and opt-in convergence work remain separate.

For each component port, with flow **positive into that component**:

```text
v = sqrt(r0) * (a + b)       a, b: complex envelopes in sqrt(W)
i = (a - b) / sqrt(r0)       r0: finite positive real coordinate normalization
a = (v/sqrt(r0) + sqrt(r0)*i) / 2
b = (v/sqrt(r0) - sqrt(r0)*i) / 2
P_in = abs(a)**2; P_out = abs(b)**2; P_net_into_component = P_in - P_out
```

`abs(v)**2` is generally **not** traveling optical power. Interference between
incident and returning light changes the potential even for `r0=1`.

## Choosing a source

`OpticalSource(power=P, phase=phi)` remains an ideal **potential difference**:
`v_p1 - v_p2 = sqrt(P)*exp(1j*phi)`. `OpticalSourcePulse` keeps its sigmoid
potential-difference waveform. Their existing `power` parameters and two-port
orientation are unchanged; neither promises launched power P into a reflecting
load. For example, at `r0=1`, a prescribed potential `E` facing a reflector gamma
implies launched wave `E/(1+gamma)` wherever that loaded problem is solvable.

```python
from circulax.components.photonic import OpticalSource
from circulax.components import OpticalWaveSource, OpticalWaveTermination

legacy = OpticalSource(power=4.0, phase=0.3)  # fixes potential difference
source = OpticalWaveSource(amplitude=2.0 * complex(0.955336489, 0.295520207))
load = OpticalWaveTermination()
```

`OpticalWaveSource` has a single circuit-facing terminal `p`. Its `amplitude`
parameter directly sets the emitted envelope in sqrt(W), so launched power is
`abs(amplitude)**2`. Relative to the **source**, emitted light is `b`, returning
light is `a`. Relative to the attached device, these names exchange. The source
absorbs all returning light without changing its emitted wave:

```text
b_source = amplitude
F_p = v/r0 - 2*amplitude/sqrt(r0); Q_p = 0
```

`OpticalWaveTermination` also has just `p`, emits zero, and absorbs arbitrary
incident light: `F_p = v/r0; Q_p = 0`. These equations justify eliminating each
boundary's wave unknowns for **every** return amplitude. Nothing about a connected
device's matching, coupling, or numerical defaults enters that reduction.
Neither boundary uses a ground terminal or adds an auxiliary unknown.

## Reflecting-load example

Use factories for a different static reference. Supply the same reference to
all components in this wave network; changing coordinates consistently does not
change physical waves, powers, or the model's scattering law.

```python
import jax.numpy as jnp
from circulax import compile_circuit, sax_component
from circulax.components import optical_wave_source, optical_wave_termination

r0 = 50.0

def through(phase=0.4):
    t = jnp.exp(1j * phase)
    return {("in", "out"): t, ("out", "in"): t}

def reflector(gamma=0.5j):
    return {("p", "p"): gamma}

net = {
    "instances": {
        "S": {"component": "source", "settings": {"amplitude": 1.0 + 0j}},
        "D": {"component": "device"},
        "L": {"component": "reflector"},
    },
    "connections": {"S,p": "D,in", "D,out": "L,p"},
}
circuit = compile_circuit(net, {
    "source": optical_wave_source(r0=r0),
    "device": sax_component(through, r0=r0),
    "reflector": sax_component(reflector, r0=r0),
}, backend="dense", g_leak=0.0)
y = circuit.dc()
source_waves = circuit.port_waves(y, "S,p")
received = circuit.port_waves(y, "L,p")
assert jnp.allclose(source_waves.outgoing, 1.0)
assert jnp.allclose(source_waves.incident, 0.5j * jnp.exp(0.8j))
assert jnp.allclose(received.incident_power, 1.0)
assert jnp.allclose(received.net_power, 0.75)
assert jnp.allclose(source_waves.net_power, -0.75)
```

Here the source emits 1 W, absorbs 0.25 W of returning light, and supplies 0.75 W
net. The potential at `D,in` is `sqrt(r0)*(1 + 0.5j*exp(0.8j))`, including
interference. Replace the reflector with `optical_wave_termination(r0=r0)` for
zero returning wave and 1 W absorbed at the load. A lossy device instead receives
power according to its scattering law and dissipates the difference.

## Time dependence, results, and analysis limits

`optical_wave_source(r0=..., waveform=...)` builds a source class with a
JAX-compatible callable of time returning a dimensionless complex multiplier.
The emitted envelope is `amplitude * waveform(t)`. As for existing `@source`
components, DC evaluates at time zero; transient and HB evaluate their sample
times. There is no independent DC override in this boundary API.

```python
Source = optical_wave_source(
    r0=r0, waveform=lambda t: 0.8 + 0.2*jnp.exp(2j*jnp.pi*1e9*t)
)
# Register Source under "source" and recompile the example netlist.
# sol = circuit.transient(t0=0, t1=1e-9, dt0=1e-11,
#                         saveat=jnp.linspace(0, 1e-9, 11))
# waves = circuit.port_waves(sol.ys, "L,p", analysis="tran", t=sol.ts)
```

`Circuit.port_waves(y, "Instance,port", ...)` returns a JAX-compatible `PortWaves`
pair with `incident` and `outgoing` complex amplitudes, and `incident_power`,
`outgoing_power`, and signed `net_power` properties. It reconstructs from local
potential and terminal flow, without reading private wave-state names. Pass the
same scalar `params` updates used for the solve, and the sample times for a
time-varying boundary. Leading sample dimensions are preserved; parameter batches
can use external `jax.vmap` around solving and reading waves together.

The accessor supports static SAX components, the five migrated native algebraic
passives, and these explicit boundaries, whose
terminal flows are algebraic. Frequency-dependent SAX wrappers support DC only,
using their zero-frequency local operator. It rejects their time-domain
observables, legacy sources, and components without the wave contract.
`OpticalDelayLine` supports DC, transient and HB time samples because its
complete terminal F flows are history independent. The same applies to the optical
`o1`/`o2` terminals of `phase_shifter_factory`, whose electrical terminals are
explicitly rejected even when the group declares a wave reference. Other delayed
models remain rejected unless they explicitly provide that contract. For other models,
`circulax.power_waves.reconstruct_waves(potential, flow, r0=...)` is reusable once
the caller has the **complete** signed terminal flow, including any dQ/dt.

AC source metadata accepts an independent emitted-wave **perturbation** in
sqrt(W), through `ac_response(sources={"S": delta}, ...)` or the existing
quadrature small-signal API. It does not scale that perturbation by the DC
amplitude or waveform. AC results are perturbations, with their existing
complex-coordinate/sideband conventions; they are not total physical fields.
`port_waves` rejects AC labels and harmonic-coefficient labels. For HB, pass the
first return value of `circuit.hb(...)` (time samples in the usual real-block
layout), `analysis="hb_time"`, and the HB sample times. Compute power on those
reconstructed samples; individual coefficient magnitudes omit interference.
Raw arrays do not carry analysis tags, so callers must supply the correct layout
and analysis label. This milestone does not add AC traveling-power observables.

## Connection and compatibility assumptions

Use a common compatible carrier frequency, carrier phase/time convention, mode,
and positive real reference at a direct connection. These boundaries operate on
complex carrier-frame envelopes, not real optical-frequency sinusoids. They do
not translate carrier frames, convert modes, or renormalize S-parameters. The
same numerical S law may be reused under a consistent coordinate change of the
whole abstract wave network; changing only one component's reference changes its
connection physics. Use explicit adapters for physical reference/mode changes.

Connect exactly two compatible optical component terminals per node. Use an
explicit physical splitter for branching; an electrical-style shared node is
not a substitute for an optical splitter. The compiler does not yet enforce
optical disciplines or these connection assumptions. The five native passives and the two migrated delay models share the
documented contract and their respective analysis limits;
do not assume detectors or other mixed models do.
Existing `Circuit.port` continues to return the potential, with no new power
interpretation. No existing source settings or convergence defaults change. The native passive
milestone changes raw state layouts as described in its migration guide.
