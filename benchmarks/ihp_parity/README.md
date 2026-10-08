# IHP parity benchmark

Circulax and ngspice read `gdsfactory/IHP`'s native
`ihp/models/ngspice/models/*.lib` files. VACASK reads the corresponding
preconverted `ihp/models/vacask/models/*.lib` files from the same checkout.
VACASK's ngspice importer is not used and needs no local adapter patch.
The harness renders its fixed testbench vocabulary into VACASK root syntax,
while ngspice and Circulax use native ngspice testbench statements.

## PDK revision and required corrections

Validation uses IHP PR #259 revision
`fee0b94b051d822737d959f2b7ae049db533f217`, which fixes duplicate `swsoa`
and primitive model declarations in the converted libraries, plus
[toolchain/ihp-resistor-parity.patch](toolchain/ihp-resistor-parity.patch).
That supplemental patch corrects the three converted resistor instances from
`sw_mman=0` to `sw_mman=1`, matching the shipped native cards. The switch enables
deterministic manual offsets; it does not draw random samples. Without this
correction, 12 resistor comparisons fail. The existing converter preserves the
native value when regenerating these cards, as verified independently.

These are IHP model-library corrections, not VACASK simulator modifications.
PR #259 is currently unmerged, so current IHP main is not the validated input.
The supplemental patch includes an independent dialect-consistency test.

## Running

Run the same locked Linux task used in GitHub Actions:

```sh
pixi run --locked -e ihp-parity ihp-parity-ci
```

This installs IHP PR #259 from `fix/vacask-common-declarations`, pinned by
`pixi.lock` to `fee0b94`, and `vacask-bin==0.3.3.dev2`, which bundles VACASK,
OpenVAF and compatible primitive modules. The task builds ngspice 45.2 with
OSDI support from a pinned source revision, compiles the shipped IHP models
for a generic CPU, then requires **60/60 comparisons for each simulator**.
Builds and caches live under `.pixi/ihp-parity`; results, provenance and
AC/transient diagnostics are written to `parity-results/`.

The task copies installed IHP model resources into its build directory and
applies the guarded three-setting resistor correction there. The installed
PR dependency remains unchanged. No external checkout or locally modified
VACASK build is needed. First builds require Git, a C toolchain, Autoconf,
Automake, Libtool, Bison and Flex; the CI job installs these prerequisites.

The Linux `IHP parity / ngspice + VACASK` job in
[ci.yaml](../../.github/workflows/ci.yaml) runs this task on pull requests and
main/development pushes and uploads diagnostics even when parity fails.
Its Python 3.12 environment is isolated from the Python 3.13 benchmark
environment because IHP requires Python `<3.13`. Model resources are located
without importing IHP's geometry package.

For manual or selected runs, `pixi run -e ihp-parity ihp-parity --help` exposes
the underlying harness. `--pdk` defaults to the installed package; use the
staged corrected tree for VACASK resistor comparisons.

The parser is pinned to NetlistParse revision `d83031d`. Circulax requires
OSDI ABI 0.4 modules. Compile PSP QS/NQS and r3_cmc from the selected IHP
checkout's `ihp/models/ngspice/va` sources, plus VACASK's SPICE-port
resistor/capacitor/inductor sources, using a compatible compiler. Native ngspice
has a separate `--ngspice-osdi-module` option. VACASK loads its modules through
the converted PDK's common include; Circulax's explicit modules are not injected
into the VACASK deck. `--vacask-compiler` configures the reference independently
of Circulax's `--compiler`. Compiled reference VA sources are cached and staged
in the temporary run directory, avoiding repeated compilation for every case.

The portable helper accepts `IHP_ROOT`, `OSDI_DIR` and optional `PYTHON`:

```sh
export IHP_ROOT=/path/to/validated/IHP
export OSDI_DIR=/path/to/abi04/modules
# Run inside the benchmark environment, e.g. pixi run -e benchmark bash ...
bash benchmarks/ihp_parity/toolchain/run-local-parity.sh \
  --simulator ngspice --output /tmp/ihp-ngspice-results.json
bash benchmarks/ihp_parity/toolchain/run-local-parity.sh \
  --simulator vacask --vacask /path/to/vacask \
  --module-path /path/to/vacask/modules \
  --vacask-compiler /path/to/openvaf-r \
  --shared-library-path /path/to/shared/libraries \
  --output /tmp/ihp-vacask-results.json
```

`OSDI_DIR` contains `psp103.osdi`, `psp103_nqs.osdi`, `r3_cmc.osdi`,
`sp_resistor.osdi`, `sp_capacitor.osdi` and `sp_inductor.osdi`. See
[toolchain/README.md](toolchain/README.md) for the recorded toolchain.
`--pdk` also accepts an upstream IHP-Open-PDK root or native models directory
for ngspice. That layout requires an explicit `--vacask-models` converted tree
when selecting VACASK; there is no fallback to native foreign imports.

`--only SUBSTRING` selects comparisons; matching none is an error. Failures are
recorded in JSON and remaining cases continue. Any failure returns nonzero.
AC/transient `.npz` diagnostics are saved beside the results. HBT/VBIC cases
remain excluded from this matrix, per issue #69.

## Acceptance and analysis semantics

The 60 cases cover LV/HV MOS corners, finger counts/multiplicity, RF MOS,
resistor corners, MOS/resistor temperature sweeps, NAND/NOR/inverter DC,
inverter DC sweep, QS/NQS MOS AC, CMIM/RF CMIM AC, and loaded inverter/CMIM
pulse transients. Acceptance tolerances remain `rtol=1e-5`, `atol=1e-9` for
DC/AC and `rtol=3e-3`, `atol=1e-4` for transients. Input waveforms separately
require `rtol=atol=1e-9`. Transient steps remain bounded to 1 ps.

For VACASK AC, Circulax assembles `G_dc + j*omega*C_ac`; for ngspice it uses
`G_ac + j*omega*C_ac`, matching each simulator's small-signal evaluation.
Circulax DC solves use `rtol=1e-10`, `atol=1e-12`; reference solver tolerances
are unchanged. The VACASK backend supplies Circulax with `gmin=1e-12`, matching
VACASK's device option. The shipped IHP PSP sources do not use that parameter.

Circulax requires released bosdi 0.1.8 or later in the 0.1 series. Native
harmonic balance evaluates transient-mode F/Q for devices without ABI state
slots and uses a DC starting point. Devices with ABI state slots and gradients
through a converged native HB solve remain unsupported. HB is covered by
separate RC-divider regressions rather than this parity matrix.

The harness explicitly selects `statistical_mode="nominal"`: `agauss` evaluates
to its nominal argument. The default library loader rejects random functions.
Sampling and statistical/mismatch parity are outside this benchmark. Ngspice
uses a fixed seed for reproducibility. Explicit deterministic model-card
settings such as the resistor manual-offset switch are preserved.

## Recorded validation

`ihp-ngspice-results.json` and `ihp-vacask-results.json` each record **60/60
passing** on 2026-10-05, using the same IHP source revision and unchanged
acceptance tolerances. `ihp-provenance.json` records input revisions, patches,
binary hashes and toolchain versions. Validation uses the same locked
packages and pinned ngspice build as the CI task, without simulator patches.

Run the regression selection with:

```sh
PATH=/path/to/abi04/compiler/directory:$PATH \
VACASK_EXECUTABLE=/path/to/vacask \
VACASK_MODULE_PATH=/path/to/abi04/modules \
VACASK_SHARED_LIBRARY_PATH=/path/to/shared/libraries \
  pixi run -e benchmark python -m pytest \
  tests/netlist_io benchmarks/utils/test_reference.py \
  tests/test_circuit.py tests/test_ac_sweep.py tests/test_subcircuit.py -q
```

The reference tests run real ngspice OP/DC/AC/transient analyses and analytic
VACASK converted-wrapper cases for both branches and multiplicity. The helper
module remains benchmark-only; InSpice is not a Circulax runtime dependency.
