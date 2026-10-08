# Reproducible parity toolchain

Run `pixi run --locked -e ihp-parity ihp-parity-ci` from the repository root.
The GitHub Actions parity job runs this exact command and uploads both
60-case results, provenance, waveform diagnostics and ngspice build logs.

| Component | Pinned input |
| --- | --- |
| IHP | PR #259, `fee0b94b051d822737d959f2b7ae049db533f217`, installed through Pixi |
| VACASK / OpenVAF / primitive modules | `vacask-bin==0.3.3.dev2`, locked wheel |
| ngspice | 45.2, `724dc77b9153dc75eaec474b1f7a44f1fcf5f362`, built with `--enable-osdi` |
| NetlistParse | Locked Python extension at `d83031dd9b75ab8688dea134033eaa14f9107bfb` |
| IHP compact modules | Compiled from shipped VA sources with `--target_cpu generic`, OSDI ABI 0.4 |

The Conda ngspice package lacks the required OSDI support. The
`ihp-build-ngspice` dependency task therefore builds the pinned official
mirror tag, verifies its commit and caches the installation under
`.pixi/ihp-parity/toolchain/ngspice`. No workstation paths are required. The locked Pixi C++ runtime takes
precedence over the wheel's older runtime so ngspice built by current
system compilers can run alongside VACASK.
`../ci.py` provisions the model resources and runs both matrices; hashes and
the IHP Git revision are saved in `parity-results/provenance.json`.

VACASK reads IHP's preconverted libraries; its native ngspice importer is
not used. No VACASK source modification is required. IHP PR #259 fixes
duplicate common declarations. One additional converted-library correction
is still required: the three resistor cards must preserve native
`sw_mman=1`. The task checks both dialects before correcting a copy under
`.pixi/ihp-parity/pdk`; changed upstream settings require review.

`ihp-resistor-parity.patch` records that supplemental IHP fix and an
independent dialect-consistency test. The existing converter already
preserves the native settings when regenerating these cards. Apply this
patch to an IHP checkout when testing the correction upstream:

```sh
git -C /path/to/IHP apply /path/to/ihp-resistor-parity.patch
python -m pytest tests/test_vacask_resistor_parity.py -q
```

`run-local-parity.sh` remains available for manually supplied compatible
toolchains; see the parent README for its options. CI uses the locked task.
