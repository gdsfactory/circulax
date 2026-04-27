# ngspice ring oscillator setup

Files:

- `runme.sim` (symlink) — N=9 template, canonical upstream input.
- `runme_N.sp` — generated for other N values by `ngspice_gen.py`.
- `ngspice_gen.py` — emits `runme_N.sp` for arbitrary N.
- `models.inc` (symlink) — PSP103 NMOS/PMOS subcircuits.
- `prepare.py` (symlink) — upstream's hook that compiles `psp103v4.osdi`.

The harness compiles `psp103v4.osdi` here on demand from
`/home/cdaunt/code/vacask/VACASK/devices/psp103v4/psp103.va` via
`openvaf-r`; ngspice 45.2 loads OSDI 0.4 binaries fine.

`psp103v4.osdi` and the .raw output files are gitignored.
