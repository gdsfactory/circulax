# GF180 compact-model parity

This checks native OSDI evaluation with a second PDK/model family after IHP's
PSP103 tests. It does **not** establish full GF180 SPICE-wrapper compatibility.

NetlistParse reads the original `sm141064.ngspice` sections, expressions and
model cards. The benchmark selects explicit bins and resolves their parameters,
then supplies identical numeric cards to bosdi and standalone VACASK. VACASK
independently builds and solves its circuit. This validates evaluation and
assembly, but does not independently validate the SPICE expression interpreter,
automatic bin selection, schematic metadata, or hierarchical wrapper conversion.

## Results

- `results.json`: 70/70 configurations match native VACASK through both low-level bosdi and full Circulax assembly for DC drain current
  and complex AC drain current at seven frequencies from 1 kHz to 1 GHz, using
  VACASK's own `sp_bsim4v8` binary. Five corners, 3.3 V NMOS/PMOS, 6 V NMOS/PMOS,
  native NMOS, representative bins, and disabled/enabled gate, body and
  source/drain resistance networks are covered.
- `cogenda-results.json`: 25 configurations match in low-level bosdi **and** full
  Circulax DC/AC circuit assembly, using Photonflux's Cogenda `bsim4va` binary.
  The other 45 cases remain explicit errors: 20 lack the `lintnoi` parameter;
  20 enabled body-network cases fail in native VACASK; five additional body-network
  cases do not converge in the low-level internal solve. No unsupported parameter
  was discarded to make those cases pass.
- Every comparison uses `rtol=1e-5`, `atol=1e-12 A`; failures are retained.
- The existing IHP suite still passes 60/60 after the ground-collapse fix.

Both runtimes use the **same implementation within each comparison**. GF180
cards request BSIM4 4.5 or 4.6; VACASK's supplied model implements 4.8.3 and the
Cogenda model implements 4.8. Their version selectors are recorded but omitted
from the supplied numeric card. This does not establish equivalence with
ngspice's older BSIM4 implementation. VACASK's `rgeomod` spelling is mapped to
its Verilog-A parameter `instance_rgeomod`; declared OSDI aliases are also honored.

## What GF180 exposed

Cogenda declares zero OSDI state slots; VACASK's BSIM4 declares nine. The current
OpenVAF-Reloaded executable compiled Cogenda successfully and its default-topology
cases work through native OSDI. Photonflux's older setup-crash/compiler warning
therefore does not apply universally to this tested toolchain; the enabled
body-network failures are still unresolved.

The full Circulax Cogenda test exposed an unhandled OSDI collapse-to-ground
sentinel (`UINT32_MAX`) for inactive branch-current unknowns. Keeping those raw
slots without constraints left zero rows and a singular matrix. Bosdi now honors
selected ground-collapse flags, uses a valid private ground scratch slot for
native residual loading, and constrains the retained raw unknown to zero. The
active/inactive branch regression covers cached/uncached and residual-only
paths, and DC/AC/transient modes in the integration branch. See bosdi PRs
[#22](https://github.com/gdsfactory/bosdi/pull/22) and
[#23](https://github.com/gdsfactory/bosdi/pull/23).

## Reproduce

From the Circulax integration checkout, using an environment with its Verilog-A
extra installed, and the separate benchmark reader dependencies:

```bash
uv pip install -r benchmarks/requirements-reference.txt
```

Then run:

```bash
PYTHONPATH=. python benchmarks/gf180_parity/run.py \
  --pdk /home/cdaunt/code/gdsfactory/pdks/gf180mcu \
  --vacask-root /home/cdaunt/code/vacask/VACASK \
  --output benchmarks/gf180_parity/results.json
```

For the alternate implementation, compile `vendor/BSIM4/bsim4.va` from
[Photonflux](https://github.com/alexsludds/photonflux) at the commit recorded in
`environment.json`, then pass `--model-osdi /absolute/path/to/bsim4.osdi`. Do not
compare one implementation against a different reference implementation.

```bash
openvaf-r /path/to/photonflux/vendor/BSIM4/bsim4.va -o /tmp/cogenda-bsim4.osdi
PYTHONPATH=. python benchmarks/gf180_parity/run.py \
  --pdk /home/cdaunt/code/gdsfactory/pdks/gf180mcu \
  --vacask-root /home/cdaunt/code/vacask/VACASK \
  --model-osdi /tmp/cogenda-bsim4.osdi \
  --output benchmarks/gf180_parity/cogenda-results.json
```

`--only` filters by case-name substring. Per-case errors are recorded and the
runner continues; inspect every result rather than treating exit code zero as
suite success. Binary and PDK source hashes are saved alongside the results.

## Remaining integration work

- Circulax's `osdi_component` rejects unaudited models declaring state slots by default. The
  comparisons now explicitly opt into `state_policy="limiting_only"` for the audited OpenVAF BSIM4 binary. Its nine slots are `$limit` Newton buffers; `ENABLE_LIM` is disabled. Physical charges remain in Q. Generic state-history, `$abstime`, and BSIM4 transient parity remain unvalidated.
- The installed VACASK SPICE converter needs level-54/version handling and lacks
  complete MOS/bin/wrapper conversion for this library. The benchmark does not
  use the temporary converter experiments or modify the PDK.
- Full wrapper loading needs automatic bin selection, parameter-default semantics,
  deterministic statistical settings, and runtime voltage-dependent passive
  expressions. GF180's existing `docs/simulation.md` and simulation tests record
  several VACASK adapter limitations in those areas.
- GF180 passive/BJT circuits and transient comparisons remain outside this matrix.
- Photonflux's reported bosdi 0.1.5 defaults, branch names, collapse allow-list,
  division and keyword issues were already integrated via
  [bosdi PR #16](https://github.com/gdsfactory/bosdi/pull/16). Those fixes are
  distinct from the newly discovered native ground-collapse case.
