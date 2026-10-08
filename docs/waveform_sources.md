# Waveform Sources

`WaveformVoltageSource` and `WaveformCurrentSource` (in `circulax.components.electronic`) are
SPICE independent sources with an optional **DC override** and a **SIN, PULSE or PWL** transient waveform.
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
`PULSE(v1 v2 [td tr tf pw per])` and `PWL(t1 v1 t2 v2 ...) [r=time] [td=delay]`, with a `DC`
clause on either side of the waveform. Numbers use engineering suffixes with SPICE meaning
(`M` is milli, `meg` is mega); `SIN` phase is in degrees on input. `AC` clauses, other waveform
kinds, non-finite or out-of-range values and malformed argument lists raise
`circulax.netlist_io.NetlistError`. `parse_waveform` parses just the waveform clause.

Zero and omitted waveform arguments retain finite zero placeholders until the analysis
resolves them. PULSE rise/fall times default to `TSTEP`; width and period default to `TSTOP`.
SIN frequency defaults to `1/TSTOP`. Set `tstep` and `tstop` in the parser to supply those
settings explicitly. Otherwise `Circuit.transient` uses `dt0` for `TSTEP` and `t1` for `TSTOP`;
its optional `tstep` argument supplies a printing interval distinct from the initial solver step.
Harmonic balance uses its sample interval and fundamental period for these defaults.

To batch different waveform kinds in one group, give `pwl_points=N` to every call: all instances
then carry `pwl_t`/`pwl_v` of length `N` (PWL padded by repeating its last point). PWL instances of
*different* lengths are still correct; they simply compile into separate groups.

## Initialization and DC semantics

- Without an explicit DC value, `.dc()` uses the waveform's time-zero value.
  For example, `SIN(0.9 1 1k)` gives a 0.9 V operating point.
- An explicit DC value overrides `.dc()` only: `DC 0.5 SIN(0.9 1 1k)` gives a
  0.5 V operating point, while the transient starts at 0.9 V. An explicit `DC 0`
  is distinct from an omitted value. Parsed settings record this in `dc_given`;
  direct component constructors also remember whether `dc` was supplied.
- Transient initialization solves the circuit with the waveform evaluated at
  time zero, including capacitor and inductor initial states. Passing `y0` overrides
  this automatic initialization.
- Harmonic balance evaluates the waveform at every sample, including time zero.
  The DC override only supplies its operating-point seed.
- Source stepping scales the full operating-point source value, including one
  inferred from the waveform, through `source_scale`.

The solvers select DC or waveform evaluation through a numeric `source_mode` field.
Time zero is an ordinary waveform sample, so evaluation no longer infers the analysis
from the sign of time. For direct component evaluation, set `source_mode=1` and provide
`tstep`/`tstop` when the waveform uses analysis defaults.

## Waveform boundaries

| Kind | Before start (`t <= delay`) | After |
|------|-----------------------------|-------|
| SIN  | `offset + amplitude * sin(phase)` | `offset + amplitude * exp(-damping*(t-delay)) * sin(2*pi*freq*(t-delay) + phase)` |
| PULSE | `v1` | linear `tr` rise to `v2`, hold `pw`, linear `tf` fall; repeats every `per` (zero/omitted `per` uses `TSTOP`) |
| PWL  | first value | linear between points, last value held; `r >= 0` loops `[r, t_last)` forever |

Parity tests compare operating points and transient samples against ngspice for SIN,
PULSE and repeating PWL, including explicit DC overrides and omitted/zero arguments.
PWL repeat time `r` must match a supplied point before the last point; `r=-1` disables repetition.
Both standard options after the parentheses and the previous options inside them are accepted.
Repeating PWL is also supported for current sources as an extension; ngspice supports
`r`/`td` only for voltage sources, so current repeats are checked against the equivalent
voltage waveform. Source-driven AC analysis is not part of these components. Instances bind identically from a plain
netlist dict or a `kfnetlist.Netlist` (`settings=parse_source(...)`).
