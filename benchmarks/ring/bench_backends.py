"""KLU backend scaling comparison for the PSP103 ring oscillator.

Runs the OSDI variant across N stages for each klujax solver backend,
producing a µs/step table that shows how each backend scales with system
size (number of ring stages → number of circuit unknowns).

Backends compared:
  klu_split_linear   — symbolic analysis once at build time; numeric
                       factorisation every Newton step via klujax.factor +
                       klujax.solve_with_symbol.  Symbolic handle is held in
                       KLUHandleManager so KLU never re-analyses the sparsity
                       pattern.  O(nnz) per step.
  klu_split_refactor — same symbolic handle + klujax.refactor (reuses numeric
                       fill-order from previous step as warm start).  Faster
                       for mildly nonlinear circuits.
  klu_rs_linear      — same as klu_split_linear but routing through klujax_rs
                       (Rust/Rayon backend at /home/cdaunt/code/klujax_rs-static).
                       klujax_rs parallelises the vmap batch dimension (N device
                       instances per group) using a Rayon thread pool.  Break-even
                       vs klujax is n_lhs ≈ 32 for solve; below that klujax is
                       faster due to Rayon overhead.
  klu_rs_refactor    — klujax_rs + refactor (combines Rayon parallelism with
                       numeric warm-starting).
  ngspice            — external reference (runs full symbolic+numeric KLU every
                       Newton step, no handle reuse, single-threaded).

Usage:
    pixi run python benchmarks/ring/bench_backends.py
    pixi run python benchmarks/ring/bench_backends.py 3 9 15 21 33
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path


# Patch KLUHandleManager.__del__ in both klujax and klujax_rs to be a no-op.
# Both implementations try to call jax.core.Tracer inside __del__ after JAX has
# been torn down during interpreter shutdown, producing flood of
# "Exception ignored in: <__del__>" AttributeError messages.  The memory is
# freed by the OS anyway; swapping __del__ for a no-op is safe.
def _patch_klu_del(mod_name: str) -> None:
    mod = sys.modules.get(mod_name)
    if mod and hasattr(mod, "KLUHandleManager"):
        try:
            mod.KLUHandleManager.__del__ = lambda self: None
        except (AttributeError, TypeError):
            pass

# Applied after first _run_circulax call (which triggers the lazy klujax import).
# Also re-applied after klujax_rs load inside _activate_klujax_rs.


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/cdaunt/code/vacask/VACASK/python")

RESULTS_PATH = HERE / "backend_results.csv"
README = HERE / "README.md"

# Location of the klujax_rs Rayon-backed build.  Override via env var when
# the build lives somewhere other than the default local dev path.
_RS_DEFAULT = Path("/home/cdaunt/code/klujax_rs-static")
KLUJAX_RS_PATH = Path(os.environ.get("KLUJAX_RS_PATH", _RS_DEFAULT))

DEFAULT_N = [3, 9, 15, 21, 27, 33]

BACKENDS = [
    ("klu_split_linear",   "klu_split_linear",   False),
    ("klu_split_refactor", "klu_split_refactor",  False),
    ("klu_rs_linear",      "klu_split_linear",    True),
    ("klu_rs_refactor",    "klu_split_refactor",  True),
]


def _activate_klujax_rs() -> bool:
    """Inject klujax_rs as the 'klujax' module in sys.modules.

    Returns True if klujax_rs was found and activated, False otherwise.
    Falls back gracefully: regular klujax stays if the RS build is missing
    or KLUJAX_RS_PATH env var / default path doesn't exist.
    """
    if not KLUJAX_RS_PATH.exists():
        return False
    rs_str = str(KLUJAX_RS_PATH)
    if rs_str not in sys.path:
        sys.path.insert(0, rs_str)
    try:
        # klujax_rs.py has a bool("0")==True bug — DEBUG is always True,
        # printing "KLUJAX_RS DEBUG MODE." at import and op names on every call.
        # Suppress import-time chatter by redirecting stderr; then patch the
        # module-level debug() to a no-op so per-call prints never appear.
        import io
        _old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            import klujax_rs
        finally:
            sys.stderr = _old_stderr
        klujax_rs.debug = lambda _s: None  # silence per-call op prints
        sys.modules["klujax"] = klujax_rs
        _patch_klu_del("klujax_rs")  # silence __del__ noise at shutdown
        # Flush any circulax solver module that already captured the old klujax
        for mod_name in list(sys.modules):
            if "circulax" in mod_name and "solvers" in mod_name:
                del sys.modules[mod_name]
        return True
    except ImportError:
        return False


def _deactivate_klujax_rs() -> None:
    """Restore the original klujax in sys.modules.

    Also flushes cached circulax solver modules so the next bench_circulax
    import picks up the restored klujax rather than the RS version.
    """
    for mod_name in list(sys.modules):
        if ("circulax" in mod_name and "solvers" in mod_name) or mod_name == "bench_circulax":
            del sys.modules[mod_name]
    try:
        sys.path = [p for p in sys.path if p != str(KLUJAX_RS_PATH)]
        # Re-import from site-packages (original klujax).
        if "klujax_rs" in sys.modules:
            del sys.modules["klujax_rs"]
        import importlib

        import klujax as _real
        sys.modules["klujax"] = _real
    except ImportError:
        pass


def _run_circulax(n: int, backend_key: str, use_rs: bool = False) -> dict:
    if use_rs:
        active = _activate_klujax_rs()
        if not active:
            return {"simulator": f"circulax_rs_{backend_key}", "backend": backend_key,
                    "n_stages": n, "status": "skipped", "notes": "klujax_rs not built"}
    if "bench_circulax" in sys.modules:
        del sys.modules["bench_circulax"]
    import bench_circulax as cx
    r = cx.run(n_stages=n, variant="osdi", backend=backend_key)
    if use_rs:
        _deactivate_klujax_rs()
    # Patch KLUHandleManager.__del__ to silence shutdown noise now that klujax
    # is definitely imported (it's a lazy import triggered by bench_circulax).
    _patch_klu_del("klujax")
    _patch_klu_del("klujax_rs")
    r["simulator"] = f"circulax_{backend_key}{'_rs' if use_rs else ''}"
    r["backend"] = backend_key
    return r


def _run_ngspice(n: int) -> dict:
    """Delegate to run.py's ngspice runner for a consistent reference."""
    import run as r
    return r.run_ngspice(n)


def _write_csv(rows: list[dict]) -> None:
    fields = ["simulator", "backend", "n_stages", "status", "wall_s",
              "compile_s", "dc_s", "n_steps", "us_per_step", "sys_size"]
    with RESULTS_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _render_table(rows: list[dict]) -> str:
    """Table: rows=N, cols=backend µs/step."""
    by_n: dict[int, dict[str, dict]] = {}
    for r in rows:
        n = r.get("n_stages")
        if n is None:
            continue
        # Key by backend label for klujax variants; by "ngspice" for ngspice
        sim = str(r.get("simulator", ""))
        key = str(r.get("backend", sim))
        if "ngspice" in sim:
            key = "ngspice"
        by_n.setdefault(int(n), {})[key] = r

    def _fmt(r: dict) -> str:
        if not r or r.get("status") not in ("ok",):
            return r.get("status", "—") if r else "—"
        u = r.get("us_per_step")
        try:
            return f"{float(u):.1f}"
        except (TypeError, ValueError):
            return "—"

    header = (
        "| N | sys_size | klu_split_linear | klu_split_refactor"
        " | klu_rs_linear | klu_rs_refactor | ngspice | best_cx/ngspice |\n"
        "|---|----------|-----------------|--------------------"
        "|---------------|-----------------|---------|-----------------|"
    )
    lines = [header]
    for n in sorted(by_n):
        r_lin   = by_n[n].get("klu_split_linear", {})
        r_ref   = by_n[n].get("klu_split_refactor", {})
        r_rs_l  = by_n[n].get("klu_rs_linear", {})
        r_rs_r  = by_n[n].get("klu_rs_refactor", {})
        r_ng    = by_n[n].get("ngspice", {})
        sys_size = next((r.get("sys_size", "?") for r in [r_lin, r_ref, r_rs_l] if r), "?")
        # Speed ratio: best circulax vs ngspice
        try:
            all_us = [float(r["us_per_step"]) for r in [r_lin, r_ref, r_rs_l, r_rs_r]
                      if r and r.get("status") == "ok"]
            best_cx = min(all_us)
            ng_us = float(r_ng["us_per_step"])
            ratio = f"{ng_us / best_cx:.1f}×"
        except (TypeError, ValueError, KeyError):
            ratio = "—"
        lines.append(
            f"| {n} | {sys_size} | {_fmt(r_lin)} | {_fmt(r_ref)}"
            f" | {_fmt(r_rs_l)} | {_fmt(r_rs_r)} | {_fmt(r_ng)} | {ratio} |"
        )
    return "\n".join(lines)


def _update_readme(table: str) -> None:
    text = README.read_text()
    start = "<!-- BACKEND_RESULTS -->"
    end = "<!-- /BACKEND_RESULTS -->"
    block = f"{start}\n{table}\n_{time.strftime('%Y-%m-%d')}_\n{end}"
    if start in text and end in text:
        import re as _re
        text = _re.sub(
            rf"{re.escape(start)}.*?{re.escape(end)}", block, text, flags=re.DOTALL
        )
    else:
        text += f"\n## KLU backend scaling\n\n{block}\n"
    README.write_text(text)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    n_list = [int(x) for x in argv] if argv else DEFAULT_N

    rows: list[dict] = []
    for n in n_list:
        # ngspice reference
        print(f"[N={n}] ngspice…", flush=True)
        try:
            r = _run_ngspice(n)
            r["backend"] = "ngspice"
        except Exception as e:
            r = {"simulator": "ngspice", "backend": "ngspice", "n_stages": n,
                 "status": f"EXC_{type(e).__name__}", "notes": str(e)[:60]}
        rows.append(r)
        print(f"    → {r.get('status', '?')}  us/step={r.get('us_per_step', '—')}")

        for label, backend_key, use_rs in BACKENDS:
            print(f"[N={n}] circulax {label}…", flush=True)
            try:
                r = _run_circulax(n, backend_key, use_rs=use_rs)
                r["backend"] = label
            except Exception as e:
                r = {"simulator": f"circulax_{label}", "backend": label,
                     "n_stages": n, "status": f"EXC_{type(e).__name__}",
                     "notes": str(e)[:60]}
            rows.append(r)
            w = r.get("wall_s")
            print(f"    → {r.get('status', '?')}  us/step={r.get('us_per_step', '—')!r}  "
                  f"wall={w if w is None else f'{w:.2f}s'}")

    _write_csv(rows)
    table = _render_table(rows)
    _update_readme(table)
    print(f"\n{table}")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
