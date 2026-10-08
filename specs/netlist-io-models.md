# Schematic library models

`circulax.netlist_io.LibraryModel` is a non-callable, reusable registration of
a model-library subcircuit. `LibraryModel.from_file` stores a path, subcircuit,
corner sections and provisioning options without parsing or loading native code
until simulation compilation.
Original library and include files decode as UTF-8 first, with a lossless Latin-1
fallback for legacy card comments such as the micro sign in IHP libraries.

`compile_circuit` accepts registrations in its model map. Each wrapper instance
binds its own settings, evaluates original model cards and produces canonical
`kfnetlist.Netlist` topology with shared leaf model definitions. The existing
`kfnetlist.Netlist.flatten` path preserves internal devices and connections.
Distinct parameter rows share the same OSDI descriptor and compile into native
device batches; registration does not create a new OSDI model per instance.

- `defaults` supplies schematic parameter values, overridden by instance settings.
- `params_map` maps source setting names to library parameter names.
- `parameter_values` explicitly translates categorical source values to numeric
  library values before unit conversion and expression evaluation. Unknown enum
  values raise an error; defaults and per-instance overrides use the same mapping.
- `parameter_units` maps source names to explicit `(source_unit, target_unit)`
  pairs. Conversion precedes card evaluation and uses declared dimensions;
  parameter names do not imply units. Supported base units are metres, seconds,
  volts, amperes, farads, henries and ohms, with engineering prefixes and positive
  integer powers. Netlist numeric suffixes retain their existing parser semantics.
- `expressions` maps library parameter names to safe expressions over converted
  source settings. When present, only expression targets and explicitly renamed
  settings reach the library; unused factory settings remain outside model cards.
- `port_map` maps public schematic aliases to the wrapper's original terminals.
  Inner leaf ports and connections remain intact.
- Instance `corner` selects an allowed section, with `section` as the default.
  Parsed corner libraries are cached; model-card values remain per instance.

OSDI descriptors are shared by module path, terminal order, temperature, native
analysis mode, state policy and simulator parameters. Incompatible definitions
using the same leaf name raise a model-conflict error instead of silently
combining native registrations.

`osdi_modules` supplies compiled modules explicitly. If they cover every native
leaf module, unused library load declarations are not provisioned. Missing
declared `.osdi` files may fall back to matching `.va` source files, using the
existing content-addressed compiler cache. The metadata reader validates its
supported OSDI ABI independently of solver compilation.

`ResolvedCircuit.compile` and reusable registrations share the topology builder.
Inline SPICE resistor, capacitor and inductor statements retain their evaluated
instance values and map to `sp_resistor`, `sp_capacitor` and `sp_inductor` native
modules respectively. RF wrappers retain internal R/L/C connections.
Standalone resolved compilation exposes node probes; wrapper registrations expose
only their public terminals. This API does not change simulation-window behavior.

`netlist_io.expressions.parse_sine_waveform` reads SPICE-style
`SIN(offset amplitude frequency [delay damping phase])` specifications using the
same safe numeric expression and engineering-suffix evaluator as model cards.
It returns numeric parameters; phase is converted from degrees to radians.
Arguments must be finite, and frequency, delay and damping must be nonnegative.
Unknown waveform kinds and invalid argument counts raise `NetlistError`.
The caller selects the analysis and evaluates the waveform; this helper does
not replace a separately declared DC operating-point value.

`netlist_io.sources.parse_source` / `parse_waveform` read the value part of a SPICE V/I
source card (`[DC] value`, `SIN(...)`, `PULSE(...)`, `PWL(... [r=] [td=])`) into keyword
settings named like the fields of `components.electronic.WaveformVoltageSource` and
`WaveformCurrentSource`, so clients bind with `Cls(**settings)`. Numbers use SPICE suffix
semantics (uppercase `M` is milli). `AC` clauses, unknown waveforms and invalid values raise
`NetlistError`; the DC bias is never inferred from the waveform (`dc` is `0.0` when absent).
`pwl_points` pads every kind to a common PWL length so mixed kinds batch in one group.
The components return `dc` for `t <= 0` (DC solve, source stepping, default transient
initial state) and the waveform for `t > 0`; source stepping scales `dc` only. See
`docs/waveform_sources.md`.
Harmonic balance samples `t = 0`, where the sources return `dc`; it matches `VoltageSourceAC` only when
`dc` equals the waveform's `t = 0` value. An omitted SIN frequency defaults to `1 / tstop` when the
caller supplies `tstop` (as in SPICE) and is otherwise an error.
