# VACASK ring oscillator setup

Files:

- `runme.sim` (symlink) — N=9 template, the canonical upstream input.
- `runme_N.sim` — generated for other N values by `vacask_gen.py`.
- `vacask_gen.py` — emits `runme_N.sim` based on the N=9 template.
- `models.inc` (symlink) — PSP103 NMOS/PMOS subcircuits.
- `prepare.py` (symlink) — upstream's pre-build hook (compiles `psp103v4.osdi`).

The symlinks resolve to upstream's `VACASK/benchmark/ring/vacask/`.
`tran1.raw` and the compiled `psp103v4.osdi` are written here at run
time; both are gitignored.
