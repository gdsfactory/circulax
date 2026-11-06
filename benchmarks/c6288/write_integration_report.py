"""Regenerate the measured comparison section in the existing benchmark README."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent


def spread(data):
    return f"{data['median']:.2f} ({data['min']:.2f}–{data['max']:.2f}; σ={data['sample_std']:.2f})"


def main():
    data = json.loads((ROOT / "integration-summary.json").read_text())
    full = data["circulax"]["full"]
    red = data["circulax"]["reduced"]
    native = data["vacask"]
    speedup = full["wall_s"]["median"] / red["wall_s"]["median"]
    text = f"""<!-- BEGIN MEASURED COLLAPSE COMPARISON -->
## Measured collapse comparison, 2026-11-05

All three full and three reduced Circulax runs completed **2 ns at fixed 2 ps**
with BDF2/full Newton (1,000 steps), finite saved signals, valid final input and
output logic levels, and product **`0xfffe0001`**. All six native VACASK runs
also passed. These measurements use the same experimental BOSDI build for
both Circulax representations and identical OSDI binaries across simulators.

Each timing is a median over three sequential runs. Parentheses give min–max
and sample standard deviation, in seconds. **Circulax simulation time excludes
setup and JIT/warmup; VACASK time includes its whole process.**

| Configuration | Unknowns | Setup (s) | JIT/warmup (s) | Simulation or process time (s) | Accepted steps/points |
|---|---:|---:|---:|---:|---:|
| Circulax full, fixed BDF2 | 96,165 | {full["setup_s"]["median"]:.2f} | {full["compile_s"]["median"]:.2f} | {spread(full["wall_s"])} | 1,000 |
| Circulax reduced, fixed BDF2 | **25,381** | {red["setup_s"]["median"]:.2f} | {red["compile_s"]["median"]:.2f} | **{spread(red["wall_s"])}** | 1,000 |
| VACASK upstream, adaptive trap | 25,380 | included | — | {spread(native["upstream"]["process_wall_s"])} | 1,022 |
| VACASK adaptive BDF2, maxstep=2 ps | 25,380 | included | — | {spread(native["bdf2-capped"]["process_wall_s"])} | 1,524 |

The reduced timed transient is **{speedup:.2f}× faster than the matched full
representation**. The unknown-count reduction is 3.79×, so a linear count-based
explanation is insufficient. Reduced setup ranges from {red["setup_s"]["min"]:.2f}
to {red["setup_s"]["max"]:.2f} s; JIT/warmup ranges from {red["compile_s"]["min"]:.2f}
to {red["compile_s"]["max"]:.2f} s. The two later reduced runs that also record
parent whole-process time took **83.21–84.34 s**, exceeding upstream VACASK's
whole-process time. Python import/startup and final waveform serialization are
outside Circulax's three individually measured phases. Thread budgets and
numerical tolerances also differ across simulators, as detailed below.

### Actual allocation and correctness

The configured PSP103 map is identical for all 10,112 instances, despite the
NMOS/PMOS parameter differences:

```text
raw slot:       0 1 2 3 4 5 6 7 8 9 10 11 12
representative: 0 1 2 3 4 1 2 0 3 3  3  3 12
```

Four terminals plus **both genuine noise auxiliaries** survive. Seven slots per
device are removed, eliminating **70,784 unknowns**. The remaining 20,224 OSDI
internals plus 5,157 other circuit variables give **25,381**, exactly one more
than VACASK because Circulax retains an explicit ground slot. The maps come
from setup at **300.15 K**, transient analysis, the actual per-instance arrays,
and the registration's simulator settings; the later full/reduced records also
have identical parameter-array SHA-256 hashes.

The maximum difference between full/reduced saved c6288 outputs is
**2.403×10⁻¹² V**, RMS **4.094×10⁻¹⁴ V**. Repeated waveforms are deterministic.
DC, transient, and AC projection/solve tests cover mixed collapse patterns,
ground/inactive slots, terminal unions, repeated indices, charge history,
parameter updates, and analysis/settings registrations. The final expanded
Circulax test suite passed **{data["test_runs"]["circulax"]["passed"]} tests with no skips**. The separate released
BOSDI 0.1.8 compatibility run passed **22 tests**, and a real c6288 preparation
with that package fell back to **96,165 unknowns** with no topology metadata.
BOSDI's own native tests passed **53**, with **30 optional compiler cases
skipped**; its independent OSDI 0.5 header fixture was exercised. Circulax's
Library loader remains OSDI **0.4**.

Compilation logs show exactly **one `circuit_diffeqsolve` compilation per
Circulax process**, across warmup and timed calls, for all six c6288 and all
18 ring runs. The standalone short-call JIT audit is retained as well.

### Cross-simulator waveforms and limitations

Identical final digital output does **not** establish analog waveform agreement.
Relative to reduced Circulax, native adaptive trap has **1.2305 V peak / 75.68 mV
RMS** output error; capped adaptive BDF2 has **1.2344 V peak / 75.86 mV RMS**.
The differences include a native `p30` glitch absent from the fixed-step
Circulax trace. Final output differences are much smaller: **2.10 µV** and
**73.34 nV**, respectively. Native signals are linearly interpolated onto the
1,001 Circulax saved times for these comparisons. The complete cause of the
switching/glitch discrepancy has not been isolated; changing tolerances,
refining the fixed step, and auditing startup would be separate accuracy work.
The original dialect-normalized gate topology and ordered subcircuit ports
match across all **2,437 source rows**; this does not independently validate
every step of either simulator's elaboration.

![Full/reduced and native waveforms, including the worst native discrepancy](collapse-waveforms.png)

### Ring regression measurements

The existing 1 µs / fixed 50 ps, trapezoidal frozen-Newton OSDI benchmark ran at
3, 9, and 15 stages, three repetitions per representation. Every recorded run
completed with finite states and oscillation. These times exclude DC and JIT.

| Stages | Unknowns full → reduced | Full median (range), s | Reduced median (range), s | Frequency full → reduced, MHz |
|---|---:|---:|---:|---:|
"""
    for n in (3, 9, 15):
        a = data["rings"][f"{n}-full"]
        b = data["rings"][f"{n}-reduced"]
        ta = a["wall_s"]
        tb = b["wall_s"]
        text += f"| {n} | {a['unknowns'][0]} → {b['unknowns'][0]} | {ta['median']:.3f} ({ta['min']:.3f}–{ta['max']:.3f}) | {tb['median']:.3f} ({tb['min']:.3f}–{tb['max']:.3f}) | {a['frequencies_MHz'][0]:.9f} → {b['frequencies_MHz'][0]:.9f} |\n"
    text += """
The largest frequency difference is **1.11 ppm** at 15 stages. Its original
homotopy starts differ by **44.7 mV**, and its unaligned transient difference
is **0.177 V peak / 39.7 mV RMS**; the 3-/9-stage maximum differences are
1.19×10⁻⁸ / 4.59×10⁻⁸ V. The startup audit below distinguishes this initialization
limitation from the reduced assembly. Performance improved at all three sizes;
these measurements do not establish pointwise equivalence of separately
initialized 15-stage trajectories.

### Remaining cost and kernel diagnostic

Stored unique COO coordinates fall from **1,453,457 to 216,753**. VACASK reports
158,904 initial structural nonzeros. Native BOSDI buffers stay **(10,112, 13)**
for voltages/F/Q and **(10,112, 169)** for G/C in both representations.

The separate zero-iterate diagnostic uses three repetitions of ten synchronized
calls, after each kernel's own JIT warmup. Median milliseconds per call:

| Kernel | Full | Reduced |
|---|---:|---:|
"""
    for key, label in [
        ("raw_model_eval", "Raw PSP103 BOSDI evaluation"),
        ("full_system_assembly_including_eval", "System assembly including evaluation"),
        ("coalesce_numeric_klu_factor_and_solve", "COO coalescing + numeric KLU factor/solve"),
    ]:
        a = data["kernel_profiles"]["full"][key]["median_s"] * 1000
        b = data["kernel_profiles"]["reduced"][key]["median_s"] * 1000
        text += f"| {label} | {a:.2f} | {b:.2f} |\n"
    text += """
All diagnostic outputs were finite. Raw evaluation did not become faster;
standalone assembly became slower, while the numeric sparse path improved
substantially. Initial diagnostic batches are retained separately to show
short-kernel variability. These are **zero-bias kernel measurements, not a
whole-transient decomposition**: Newton iteration counts, matrix values, and
loop-invariant hoisting also affect runtime. The full-Newton Circulax path calls
`solve_with_symbol`, while native logs report one factorization followed by
3,502 (trap) or 4,638 (BDF2) refactorizations. That API/algorithm difference,
remaining excess COO structure, dense device-stamp traffic, and approximately
19 s of setup deserve separate profiling; unknown count alone does not explain
all differences. Native timings are dominated by eval/load (about 63.6 s trap,
81.4 s BDF2 in the first runs), and its continuous-bypass behavior also differs.

Machine-readable details are in [integration-summary.json](integration-summary.json).
Commands, revisions, source/runtime/model hashes, logs, `.npz` waveforms,
[tests](integration-tests-complete.log), [released compatibility](released-bosdi-tests.log),
and [kernel profiles](profile-reduced.json) are retained alongside this README.
There are no outstanding c6288 transient failures. The original failed frozen-
Newton warmup remains historical context below; no new result is presented as
an exhaustive multiplier functional test.
<!-- END MEASURED COLLAPSE COMPARISON -->
"""
    audit_path = ROOT / "ring-startup-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        a, b = audit["full"], audit["reduced"]
        comparison = audit["common_uic_comparison"]
        section = f"""### 15-stage startup failure and controlled comparison

The original source/gmin homotopy did **not** reach a DC operating point at
15 stages. Its output starts at {a["initial_output_V"]:.6f} V (full) versus
{b["initial_output_V"]:.6f} V (reduced), and the maximum absolute non-ground
mixed residuals are {a["homotopy_max_residual_excluding_ground"]:.5g} and
{b["homotopy_max_residual_excluding_ground"]:.5g}. A separate checked Newton
refinement with 500 iterations and tighter tolerances also failed in both
representations. The original ring harness only checked finiteness and a loose
magnitude bound; its transient `status=ok` does **not** certify DC convergence.
The harness now records the initial output and DC residual explicitly. The
public homotopy solver's discarded convergence status is an existing limitation
left unchanged here; use a checked solve when a true operating point is required.

A controlled additional full/reduced pair uses **the same raw-consistent UIC**:
all ring nodes at 0.6 V, VDD at 1.2 V, zero source-current/noise auxiliaries,
with full raw aliases expanded using each handle's actual mapping. Both
complete the existing 1 µs / 50 ps trapezoidal run with finite states. Output
error is **{comparison["max_abs_V"]:.3g} V maximum / {comparison["rms_V"]:.3g} V
RMS**, and frequencies are {a["common_uic_freq_MHz"]:.9f} /
{b["common_uic_freq_MHz"]:.9f} MHz. Thus the large default-start phase difference
is not evidence of lost internal dynamics or a reduced stamp mismatch.
See [ring-startup-audit.json](ring-startup-audit.json) and reproduce with
`.venv/bin/python -m benchmarks.c6288.audit_ring_startup`.

"""
        text = text.replace("### Remaining cost and kernel diagnostic", section + "### Remaining cost and kernel diagnostic")
    path = ROOT / "README.md"
    old = path.read_text()
    start = "<!-- BEGIN MEASURED COLLAPSE COMPARISON -->"
    end = "<!-- END MEASURED COLLAPSE COMPARISON -->"
    if start in old:
        a = old.index(start)
        b = old.index(end, a) + len(end)
        old = old[:a] + old[b:]
    position = old.index("\n```sh")
    path.write_text(re.sub(r"\n{3,}", "\n\n", old[:position] + "\n\n" + text + "\n" + old[position:]))


if __name__ == "__main__":
    main()
