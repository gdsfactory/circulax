# Waveform Sources

`WaveformVoltageSource` and `WaveformCurrentSource` (in `circulax.components.electronic`) are
independent sources with a **DC bias** and an optional **SIN, PULSE or PWL** transient waveform.
One class handles every waveform: the waveform kind is a numeric per-instance setting, so
sources of different kinds and parameters share one batched component group.
The older `VoltageSource`, `VoltageSourceAC`, `PulseVoltageSource` and `CurrentSource` are unchanged.

## Binding SPICE text

`circulax.netlist_io.parse_source` returns keyword settings named exactly like the component
fields, so binding is `Cls(**settings)` (or the `settings` of a netlist instance):

```python
from circulax import compile_circuit
from circulax.components.electronic import WaveformVoltageSource
from circulax.netlist_io import parse_source

settings = parse_source("DC 0.5 SIN(0.9 1 1k)")   # {"kind": 1.0, "dc": 0.5, "offset": 0.9, ...}
net = {
    "instances": {
        "GND": {"component": "ground"},
        "V1": {"component": "vsrc", "settings": settings},
        "R1": {"component": "resistor", "settings": {"R": 1e3}},
    },
    "connections": {"V1,p1": "R1,p1", "V1,p2": "GND,p1", "R1,p2": "GND,p1"},
}
```

Supported syntax: `[DC] value`, `SIN(offset amplitude freq [delay damping phase])`,
`PULSE(v1 v2 [td tr tf pw per])` and `PWL(t1 v1 t2 v2 ... [r=time] [td=delay])`, with a `DC`
clause on either side of the waveform. Numbers use engineering suffixes with SPICE meaning
(`M` is milli, `meg` is mega); `SIN` phase is in degrees on input. `AC` clauses, other waveform
kinds, non-finite or out-of-range values and malformed argument lists raise
`circulax.netlist_io.NetlistError`. `parse_waveform` parses just the waveform clause.

Values SPICE takes from the analysis can be passed as `tstep` (default PULSE `tr`/`tf`) and `tstop`
(default PULSE `pw` and SIN `freq = 1/tstop`). Without them omitted PULSE edges are ideal, the pulse
stays high, and an omitted SIN frequency is an error.

To batch different waveform kinds in one group, give `pwl_points=N` to every call: all instances
then carry `pwl_t`/`pwl_v` of length `N` (PWL padded by repeating its last point). PWL instances of
*different* lengths are still correct; they simply compile into separate groups.

## Initialization and DC semantics

- `dc` is the bias. DC solves, source stepping and the transient initial state use `dc` only.
  When the spec has no `DC` clause, `dc` is `0.0` (the SPICE default); it is never inferred
  from the waveform.
- The waveform applies for `t > 0`. It need not equal `dc` at `t = 0`; a difference is a step at
  `t = 0+`. To start a transient *on* the waveform, set `dc` to its `t = 0` value, or pass `y0`.
  (ngspice differs here: it starts `.tran` from the waveform's time-zero value.)
- Harmonic balance samples `t = 0` as an ordinary time point, where these sources return `dc`. Set `dc`
  equal to the waveform's `t = 0` value (e.g. `DC 0 SIN(0 1 1k)`), which then matches `VoltageSourceAC`;
  any other `dc` corrupts only that first sample.
- `amplitude_param="dc"`: source stepping (`solve_dc_source`, `solve_dc_auto`) ramps `dc`
  and never touches the waveform.

## Waveform boundaries

| Kind | Before start (`t <= delay`) | After |
|------|-----------------------------|-------|
| SIN  | `offset + amplitude * sin(phase)` | `offset + amplitude * exp(-damping*(t-delay)) * sin(2*pi*freq*(t-delay) + phase)` |
| PULSE | `v1` | linear `tr` rise to `v2`, hold `pw`, linear `tf` fall; repeats every `per` (`per = 0`: single pulse) |
| PWL  | first value | linear between points, last value held; `r >= 0` loops `[r, t_last)` forever |

These match ngspice to ~1e-8 for delayed, damped, phase-shifted SIN, periodic PULSE and
repeating PWL. Source-driven AC analysis is not part of these components. Instances bind identically from a plain
netlist dict or a `kfnetlist.Netlist` (`settings=parse_source(...)`).
