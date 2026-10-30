# Angelov GaN

`circulax.components.angelov_gan.AngelovGaN` implements the deterministic
Angelov GaNFET model with the upstream channel, gate leakage, nonlinear charge,
finite parasitic RLC, delayed channel-current, input/output dispersion and
self-heating equations. It is ordinary native JAX F/Q code and requires neither
OSDI nor bosdi at runtime.

The reference is [VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/angelov/vacode/angelov_gan.va),
model revision 2.22.24052023. Copyright (c) 2002–2015 Tiburon design Automation,
Inc.; the full MIT notice is preserved in the generated module, handwritten
case-formula helpers and vendored VA source.

```python
from circulax.components.angelov_gan import AngelovGaN

fet = AngelovGaN(Idsmod=4, Capmod=4, Cgs0=1e-12, Cgd0=3e-13,
                P222=.2, Selft=1, Rth=10., Cth=1e-5,
                Tau=1e-9, Lg=1e-9, Ls=.5e-9, Ld=2e-9,
                Crf=2e-12, Crfin=1e-12)
```

Ports, in reference order, are **`d,g,s,t,rf`**. Drain, gate and source use volts
and amperes. `t` is temperature rise in K, represented by a voltage; its flow is
W and storage is J. With `Selft=1`, its equation includes the reference thermal
power generation, temperature-dependent resistance and `Cth*d(t)/dt`. With
`Selft=0`, it has only the reference `1e-12` shunt. The reference uses `abs(t)`
when calculating local device temperature; its thermal balance uses signed `t`.

`rf` is the auxiliary dispersion/backgate node, measured relative to the
intrinsic source. It connects through the model's nonlinear `Rc`/`Ldc` branch
and to the intrinsic drain through `Crf`. **Keep `rf` electrically unclamped
to retain that internal network, and include it in the netlist's `ports`
mapping as in the example below.** Keep `t` in that mapping too when solving
its thermal balance. This allocates those nodes even when they have no
external connections. Connecting `rf` to a voltage source or ground changes
the model's dispersion/backgate condition. Both `t` and `rf` remain available
for external network connections.

The retained topology contains 25 local variables: five ports, twelve internal
voltage/relaxation states and eight branch-current states. Internal variables
include gate/source/drain parasitic nodes, the input-dispersion node, `xt1/xt2`
for channel delay, and currents through the delay, dispersion and parasitic
inductor/voltage branches. Circulax allocates them automatically. Zero parasitic
inductance and zero Tau retain algebraic constraints without removing branches.
Two unused noise-only coordinates `ia/ib` are removed; they have no coupling to
this deterministic model.

Supported source selections and important parameters:

| Group | Parameters |
|---|---|
| Channel | All `Idsmod=0…4`, `Ipk0`, `Vpks`, `Dvpks`, `P1,P2,P3`, `Alphar,Alphas`, `Lambda,Lvg`, breakdown/backgate coefficients |
| Gate leakage | Both `Igmod=0,1`, `Ij,Pg,Ne,Vjg`, gate-breakdown coefficients |
| Charge | All `Capmod=0…4`, `Cgspi,Cgs0,Cgdpi,Cgd0,Cgdpe,Cds` (F), `P10…P41`, `P111,P222,P10pk,m` |
| Parasitics | `Rg,Rd,Rs,Ri,Rgd,Rd2` (Ω), `Lg,Ls,Ld,Ldc` (H) |
| Delay/dispersion | `Tau` (s), `Rcmin,Rc,Rcin,Rdel` (Ω), `Crf,Crfin,Cdel` (F), `KRFDC,Kbgate` |
| Thermal | `Selft=0,1`, `Rth` (K/W), `Cth` (J/K), `Temp,Tnom` (°C), `Trise` (K), source temperature coefficients |
| Simulator | `_temperature` (ambient K, default 300.15), `_mfactor` (positive, default 1) |

Capmod 0 is linear capacitance, 1 bias-dependent capacitance, 2 charge-based,
3 bias-dependent capacitance with GaN peaking, and 4 charge-based GaN peaking.
The capacitance modes preserve the reference's stored-charge stamps, including
`C(V)*V` for modes 1 and 3. No fdomain-only approximations are used.

The finite-resistance topology requires positive `Rg,Rd,Rs,Ri,Rgd,Rc,Rcin,Rdel,
Cdel,Rth`; concrete scalar zero/negative values are rejected by the constructor.
Other parameters should obey source bounds and physical domains (in particular
positive peaking denominators). Zero-resistance topology alternatives and noise
analysis are not supported. `Noise=0` and `Noimod=0` are specialized and cannot
be changed through the constructor. Noise-only coefficient fields retained from
the source have no deterministic effect. Simulator diagnostics/operating-point
metadata are omitted; `SPICE_GMIN` is fixed at `1e-12`.

`Temp,Tnom,Pg,Ne` use NaN defaults to preserve `$param_given` behavior. Without
explicit Temp, temperature is `_temperature + Trise`; explicit Temp overrides
Trise. If Ne is explicit and Pg absent, the reference derives the gate emission
coefficient from Ne and local thermal voltage. Explicit Pg takes precedence.
Presence flags are inferred at runtime, so changing a module/PyTree parameter
from NaN to a finite value remains effective. All continuous physical parameters,
including temperature-dependent setup calculations, are JAX inputs; gradients
are tested for Ipk0 and for a temperature supplied after a PyTree update.
Discrete model selections and the presence/absence of a parameter are not
differentiable choices.

## Circuit use

```python
import jax.numpy as jnp
from circulax import compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.angelov_gan import AngelovGaN

net = {
    "instances": {
        "M": {"component": "gan", "settings": {
            "Selft": 1, "Rth": 10., "Cth": 1e-5,
            "Cgs0": 1e-12, "Cgd0": 3e-13, "Tau": 1e-9,
        }},
        "R": {"component": "R", "settings": {"R": 10.}},
        "VD": {"component": "V", "settings": {"V": 2.}},
        "VG": {"component": "V", "settings": {"V": -.5}},
        "GND": {"component": "ground"},
    },
    "connections": {
        "VD,p1": "R,p1", "R,p2": "M,d", "VG,p1": "M,g",
        "GND,p1": ("VD,p2", "VG,p2", "M,s"),
    },
    "ports": {"out": "M,d", "thermal": "M,t", "rf": "M,rf"},
}
circuit = compile_circuit(net, {"gan": AngelovGaN, "R": Resistor, "V": VoltageSource})
y_dc = circuit.dc()
ac = circuit.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.})
```

The coupled current/charge and parasitic network stays in one compact component;
it can be embedded as a leaf in larger recursive subcircuits. The internal
voltage states preserve parasitic poles and should not be replaced by external
terminal voltages.

## Verification and provenance

Run `.venv/bin/python -m pytest tests/test_angelov_gan.py -q`. The 34 tests cover
all five channel and capacitance modes with nonzero nonlinear coefficients,
both gate models, conservation of electrical terminal/internal charge and KCL,
eager/JIT agreement, finite full Jacobians, parameter gradients, runtime
parameter-presence updates, delay/inductor/thermal storage, and prescribed
`F + dQ/dy * dy/dt` time-domain currents.

Optional tests compile the vendored VA with OpenVAF and evaluate OSDI using
`osdi_loader`/`osdi_jax`. They compare physical F/Q and G/C across three bias
vectors and thirteen parameter sets. The reference's grounded noise-only `ia/ib`
coordinates are projected out; all physical electrical, thermal, dispersion and
branch-current rows/columns are compared. Actual resistor-loaded circuit DC and
AC results are also compared with OSDI, with self-heating both enabled and
disabled and nonzero delay, RLC and dispersion. Reference tests skip when tools
are absent. Full solved transient trajectories and HB parity have not been
measured.

`tests/data/va_models/angelov_gan/regenerate.py` records its validated bosdi
source revision. It compiles unoptimized MIR, restores unnamed branch-current
identities, preserves runtime parameter-presence flags, normalizes boolean
casts, compacts setup-cache references and disables stale phi batching metadata.
Short source-exact case helpers replace dropped multiway channel/charge
assignments, including the mode-dependent tanh term used by resistance branches.
`native_helpers.py` and `native_surface.py` are embedded into the generated
module and are not runtime dependencies. After regeneration, run `ruff check
--fix` and `ruff format` on the module.

The supplied `compact.vams` is a clean minimal compatibility header containing
only the numerical constants `DEFAULT_TNOM=27` and `SPICE_GMIN=1e-12`. It replaces
an unrelated legacy utility header; the device source itself is unmodified.
Standard discipline/constant headers are vendored alongside it. Source `Ldc=100m`
is represented by its equivalent `0.1 H` default in generation.

On the development CPU (JAX 0.7.2/Python 3.12.14), a 25-variable instance with
self-heating, charge, delay and dispersion took **113 ms to trace and 170 ms to
compile F/Q**. Its voltage/state Jacobian took **467 ms to trace and 362 ms to
compile**. Warm calls including dispatch and synchronization averaged 251 µs
and 292 µs respectively over 100 repeats. These are component measurements,
not full circuit solve timings. Reproduce with:

```sh
.venv/bin/python -m benchmarks.sample_components circulax.components.angelov_gan AngelovGaN \
  --bias '[1,-0.2,0,10,0.95,0.98,0.99,-0.19,0.01,0,-0.2,-0.18,-0.19,-0.19,0,0.03,0.03,0.03,0.00001,0.000001,0.000001,0.03,0.03,0.03,0.03]' \
  --settings '{"Selft":1,"Tau":1e-9,"Cgs0":1e-12,"Cgd0":3e-13,"Crf":2e-12,"Crfin":1e-12}'
```
