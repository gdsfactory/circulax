# IHP / VACASK parity development

The standalone path is: original IHP native VACASK libraries → NetlistParse CST
→ Circulax lexical parameter scopes and wrapper elaboration → OpenVAF OSDI
modules → bosdi → Circulax solvers. VACASK independently parses the original
library for every reference run. The benchmark-only runner in
`benchmarks/utils/vacask_reference.py` uses InSpice to decode reference results.
The runner and reader dependency are outside the installed Circulax package.

## Development installation

The existing `benchmark` environment uses Python 3.13 and contains the
native integration dependencies plus the reference reader on Linux. Its own solve group keeps
InSpice out of development and ordinary CI environments. Rust, OpenVAF and a built
VACASK installation must be available separately.

```sh
pixi install -e benchmark
```

Run the native integration checks with `pixi run -e benchmark native-tests`.
Set `VACASK_MODULE_PATH` to the native module directory to include module-dependent checks.

The NetlistParse dependency is temporarily pinned to the exact fork commit in
[parser PR #6](https://github.com/NyanCAD/NetlistParse.rs/pull/6). The Verilog-A
extra pins bosdi's setup, temperature, collapse and integral evaluation fixes to an exact development commit.
These are source builds and require Rust and a C++ compiler. Installing the
benchmark environment includes the fixes; no manual patch or local checkout override is
needed. Replace these development pins with upstream releases once available.

This exposes `parse_spectre(source)` and the explicit extension mode
`parse_spectre(source, vacask=True)`. No regex conversion through SPICE is used.
The patch leaves the default Spectre parser behavior unchanged. Native VACASK
loads, sections, grouped card parameters and @if/@else/@end are opt-in syntax.
Include resolution, parameter evaluation and topology elaboration belong to
Circulax, following NetlistParse's documented syntax/semantics split. Arbitrary
Python evaluation is never used for card/property expressions.

The SPICE-compatible capacitor requires the simulator-parameter setup fix in
[bosdi PR #19](https://github.com/gdsfactory/bosdi/pull/19). Temperature support is
in [bosdi PR #20](https://github.com/gdsfactory/bosdi/pull/20). Temperature is an
immutable configuration of each loaded model id, so cached handles, parameter
updates and uncached evaluation all preserve the selected kelvin temperature.
Separate registrations of the same binary may use different temperatures.

## Running comparisons

From the Circulax checkout:

```sh
pixi run -e benchmark ihp-parity \
  --pdk /path/to/ihp-parity-pdk \
  --vacask /path/to/VACASK/build/simulator/vacask \
  --module-path /path/to/VACASK/build/lib/vacask/mod \
  --shared-library-path /path/to/VACASK/.pixi/envs/default/lib \
  --compiler /path/to/openvaf-r \
  --output benchmarks/ihp_parity/results.json
```

`--only SUBSTRING` selects a subset. Paths are configurable; no simulator or PDK
checkout path is embedded in library code. Verilog-A compilation uses a cache
under `~/.cache/circulax/osdi`, keyed by source and include contents, compiler
version and architecture. VACASK receives the same binaries in its temporary run
folder, retaining its own card/wrapper evaluation. Binaries are not written to
the PDK checkout. These are OSDI **ABI 0.4** artifacts for VACASK/bosdi, not
ngspice binaries. The source directory name `ngspice/va` is only the PDK's
storage location for shared Verilog-A sources. Ngspice libraries and compiled
artifacts are left untouched; a separate compiler/ABI path is required for
ngspice compatibility.

Tests cover MOS corner/finger/multiplicity drain currents, RF NQS and HV MOS,
resistor corners, inverter/NAND/NOR operating points, an inverter DC sweep,
CMIM/RF CMIM driven AC and CMIM pulse transient. Transient runs bound VACASK's
`maxstep` to 1 ps; `step` alone is its initial/output step, not an upper bound.
Circulax uses fixed 1 ps BDF2. VACASK transient solver tolerances are `reltol=1e-6`,
`vntol=1e-8`, `abstol=1e-12`; tighter tolerances caused its inverter transient to
abort with "Timestep too small". These remain below the comparison tolerance.
Waveform tolerances are 0.3% plus 100 µV; DC/AC
use 1e-5 relative plus 1 nA or 1 nV absolute according to the measured vector.
Transient input waveforms and successful solver completion must also agree.

The original development run passed all 51 comparisons. Temperature checks at
−40 °C, 27 °C and 125 °C add six passing MOS/resistor comparisons, and a loaded
CMOS inverter pulse transient adds another passing comparison. Two additional
MOS AC checks (QS and RF/NQS, 1 MHz–1 GHz) now pass after the native collapse
and integral evaluation fixes. All 60 comparisons pass the original acceptance
criteria; `results.json` records the complete run.

The HV corner exposed duplicate common declarations in VACASK itself, reported
in [IHP issue #258](https://github.com/gdsfactory/IHP/issues/258). The shared
common library now only loads modules; `swsoa=0` is local to each MOS wrapper,
and primitive default cards have library-specific aliases. VACASK rejects
duplicate card names in one scope before flattening. The public IHP device names
remain unchanged. Aliases resolving to the same OSDI module still share a bosdi
batch; `test_card_aliases_share_an_osdi_batch` verifies this explicitly.

## PDK API

The PDK adds a Circulax entry derived from each native VACASK metadata entry.
Schematic geometry equations, port order and deterministic corner choices remain
identical. The optional `ihp.models.circulax.resolve_component` adapter accepts
schematic model metadata, cell properties and a terminal-to-node mapping:

```python
from ihp.cells.fet_transistors import nmos_schematic
from ihp.models.circulax import resolve_component

resolved = resolve_component(
    nmos_schematic().info['models'],
    {'width': 2.0, 'length': 0.13, 'nf': 2, 'm': 3},
    {'D': 'out', 'G': 'in', 'S': '0', 'B': '0'},
    corner='mos_tt',
)
# Inspect resolved.instances without loading native code.
circuit = resolved.compile(module_paths=(module_directory,), compiler=compiler)
```

A biased circuit can be loaded with `Library.from_file(path).resolve().compile(...)`.
The pure adapter does not import Circulax until invoked. Invalid corners and
unknown card parameters fail explicitly.

## Scope and remaining runtime work

These changes establish the first three integration steps, not complete VACASK
feature parity. Temperature is supplied explicitly to both Circulax and VACASK:
300 K (26.85 °C) for the baseline, with −40 °C, 27 °C and 125 °C checks added.
The library and PDK adapter now default to VACASK's usual 27 °C (300.15 K);
this is not silently treated as 300 K. The benchmark selects its temperatures
explicitly.
Both MOS AC comparisons now pass ([bosdi #23](https://github.com/gdsfactory/bosdi/pull/23)). Native OSDI collapse follows each instance's
setup flags ([bosdi #22](https://github.com/gdsfactory/bosdi/pull/22)). Integral
equations use explicit DC/AC evaluation modes. Following VACASK, the harness
retains DC conductance and obtains capacitance from a separate AC evaluation.
Maximum QS and RF/NQS AC discrepancies are 1.11e-16 V and 1.29e-12 V.

`resolved.compile(analysis="dc")` is the default. Use separate `analysis="ac"`
or `analysis="tran"` registrations for their stamps, evaluated at the DC initial
point. Analysis mode is fixed for a compiled circuit; the benchmark explicitly
coordinates these stages. General automatic mode switching inside Circulax's
public analysis methods is still future work.

VBIC HBT loads expose eight OSDI states and remain explicitly rejected by bosdi's
component descriptor. They need state-history/limiting/thermal runtime work before
HBT simulation parity can be claimed. Model-card expressions that depend on
circuit voltages, such as the HV varactor's `v(...)` parasitic expressions, also
need a runtime expression representation; the static loader rejects them.
Statistical/mismatch sections referring to absent converted libraries fail with
the missing include path. HB differentiation and general Verilog-A `$abstime`
are outside this initial integration.

[Circulax issue #63](https://github.com/gdsfactory/circulax/issues/63) and
[IHP issue #257](https://github.com/gdsfactory/IHP/issues/257) identify real
integration gaps. One assumption needs correction: the actual IHP PSP103 NQS
binary tested here has zero OSDI states; it does not have the VBIC state blocker.
The common native libraries and wrapper metadata should remain the source of
truth instead of manually copied compact-model defaults. Generic parsing and
elaboration live in Circulax; only PDK metadata and its thin adapter live in IHP.

## Local PDK validation

Run `make dev` in the PDK checkout to install its development dependencies and
fetch the centrally managed, gitignored `.pre-commit-config.yaml`. The complete
`uv run pre-commit run --all-files` suite then passes. Metadata/library tests
also pass (18 passed, one skipped without `vacask-bin`).

The PDK changes are committed separately: [IHP PR #259](https://github.com/gdsfactory/IHP/pull/259)
fixes the native VACASK common declarations, and the Circulax metadata adapter
is a draft PR stacked on that branch. The native-library fix can be reviewed
independently of Circulax.

## Native mode orchestration and VBIC follow-up

Public `Circuit.dc`, `sp`/`ac`, and `transient` now select immutable DC, AC and transient OSDI registrations automatically, including parameter updates. AC uses `G_dc + j*omega*C_ac`. The fixed raw-node and scatter layouts are checked between modes. Native harmonic balance remains explicitly unsupported.

For the audited OpenVAF binaries, the harness opts into `state_policy="limiting_only"`. OpenVAF's OSDI state count represents `$limit` Newton buffers, not physical NQS history. `ENABLE_LIM` stays disabled; physical DDT charges and IDT unknowns remain in the circuit DAE. Generic history-dependent binaries and `$abstime` still need runtime support.

The original 60 comparisons pass after public orchestration changes. `vbic-results.json` adds eight passing IHP VBIC comparisons: DC collector current and AC collector-current response, one/four fingers, NQS enabled/disabled, self-heating disabled, 0.8 V base and 1.2 V collector. Four ten-finger reference cases fail to converge in VACASK itself, even with self-heating disabled; their errors are retained. Running the entire suite currently returns failure for these four explicit reference errors (68 passes out of 72 attempted comparisons). Earlier self-heated one/four-finger checks also passed; ten-finger self-heated references failed. These checks do not establish general HBT transient parity.

A separate compiled `$limit`/DDT RC regression verifies public JIT DC/SP and exponential transient decay. The nine BSIM4 and eight VBIC limiting slots do not require delayed circuit unknowns.
