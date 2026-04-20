"""PSP103 DC op-point dump + Schur reduction, targeted at the bosdi author.

Produces, at NMOS Vds=1.0, Vgs=0.7, Vbs=0 with the VACASK psp103n card:

1. The full 6x6 post-collapse (cond, cap) and the 6-vectors (cur, chg).
2. bosdi.osdi_debug.classify_rows() output per row.
3. format_jacobian_table() dump (constraint rows flagged).
4. schur_reduce at alpha=0 (DC-only) -> 4x4 G_eff and 4-vector r_eff.
5. schur_reduce at alpha=1/dt for dt in {10 ps, 1 ns} -> 4x4 j_eff.
6. Finite-differenced C_eff = (schur(alpha=alpha_1).j_eff - schur(alpha=alpha_0).j_eff)
                               / (alpha_1 - alpha_0)   for a simple alpha-linear readout.

Run from the circulax-osdi worktree:

    pixi run python scripts/psp103_schur_dump.py > reports/psp103_schur_dump.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from fixtures.psp103_models import PSP103_OSDI, geom_settings, make_psp103_descriptors
from osdi_debug import classify_rows, format_jacobian_table, schur_reduce
from osdi_jax import osdi_eval
from osdi_loader import load_osdi_model


BIAS = {"V_D": 1.0, "V_G": 0.7, "V_S": 0.0, "V_B": 0.0}
V_INT_GUESS = 0.5   # intermediate guess, doesn't affect row classification


def pack_params(desc, settings):
    merged = desc.make_instance(settings)
    return jnp.array([[merged[k] for k in desc.param_names]], dtype=jnp.float64)


def evaluate_nmos():
    model = load_osdi_model(PSP103_OSDI)
    nmos, _ = make_psp103_descriptors()
    params = pack_params(nmos, geom_settings(10e-6, 1e-6))

    n = model.num_nodes
    v = jnp.zeros((1, n))
    v = v.at[0, 0].set(BIAS["V_D"])
    v = v.at[0, 1].set(BIAS["V_G"])
    v = v.at[0, 2].set(BIAS["V_S"])
    v = v.at[0, 3].set(BIAS["V_B"])
    v = v.at[0, 4].set(V_INT_GUESS)
    v = v.at[0, 5].set(V_INT_GUESS)

    states0 = jnp.zeros((1, 0))
    cur, cond, chg, cap, _ = osdi_eval(model.id, v, params, states0)
    return (
        model,
        np.asarray(cur)[0],
        np.asarray(cond).reshape(n, n),
        np.asarray(chg)[0],
        np.asarray(cap).reshape(n, n),
    )


def format_matrix(name: str, M: np.ndarray, labels: list[str]) -> str:
    n = M.shape[0]
    lines = [f"{name}  (rows: KCL equation, cols: d/dV)"]
    header = "          " + " ".join(f"{lbl:>11}" for lbl in labels[:n])
    lines.append(header)
    for i in range(n):
        row = f"  {labels[i]:>7} " + " ".join(f"{M[i, j]:>11.4e}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    LABELS = ["D", "G", "S", "B", "int0", "int1"]

    model, cur, cond, chg, cap = evaluate_nmos()
    n_pins = model.num_pins

    print("=" * 78)
    print(f"PSP103 NMOS W=10 um L=1 um | VACASK psp103n card")
    print(
        f"Bias: V_D={BIAS['V_D']}, V_G={BIAS['V_G']}, V_S={BIAS['V_S']}, "
        f"V_B={BIAS['V_B']}, V_int0=V_int1={V_INT_GUESS} (guess)"
    )
    print(f"Node order: D, G, S, B, int0, int1  (post-collapse)")
    print(f"num_pins = {n_pins}, num_nodes = {model.num_nodes}, num_params = {model.num_params}")
    print(f"collapsible_pairs (raw OSDI indices): {model.collapsible_pairs}")
    print("=" * 78)

    print("\n--- cur (resistive residual, A) ---")
    for i, lbl in enumerate(LABELS):
        print(f"  cur[{lbl:>4}] = {cur[i]:+.6e}")

    print("\n--- chg (reactive residual, C) ---")
    for i, lbl in enumerate(LABELS):
        print(f"  chg[{lbl:>4}] = {chg[i]:+.6e}")

    print("\n" + format_matrix("cond (G = dcur/dV, S)", cond, LABELS))
    print("\n" + format_matrix("cap  (C = dchg/dV, F)", cap, LABELS))

    print("\n--- bosdi.osdi_debug.classify_rows ---")
    for c in classify_rows(cond, cap):
        print(f"  row {c.row} ({LABELS[c.row]:>4}): {c.kind:<14}  {c.detail}")

    print("\n--- bosdi.osdi_debug.format_jacobian_table ---")
    print(format_jacobian_table(cond, cap, threshold=1e-18))

    print("\n" + "=" * 78)
    print("SCHUR REDUCTIONS (internal nodes 4, 5 eliminated)")
    print("=" * 78)

    # Wrap to the batched-leading-axis layout schur_reduce expects? The dataclass
    # docstring allows (..., num_nodes, num_nodes); our single-device arrays are
    # 2D (no leading batch) and 1D, which schur_reduce handles fine.
    for alpha, label in [
        (0.0, "alpha=0  (DC-only)"),
        (1.0e11, "alpha=1e11 (dt=10 ps, Backward Euler)"),
        (1.0e9, "alpha=1e9  (dt=1 ns,  Backward Euler)"),
    ]:
        res = schur_reduce(cur, cond, chg, cap, num_pins=n_pins, alpha=alpha)
        j = np.asarray(res.j_eff)
        r = np.asarray(res.r_eff)
        print(f"\n--- {label} --- singular={res.singular}")
        print(format_matrix(f"j_eff (4x4)", j, LABELS[:n_pins]))
        for i, lbl in enumerate(LABELS[:n_pins]):
            print(f"  r_eff[{lbl:>4}] = {r[i]:+.6e}")

    # Effective terminal capacitance by finite difference across alpha.
    alpha_lo, alpha_hi = 1.0e9, 1.0e10
    res_lo = schur_reduce(cur, cond, chg, cap, num_pins=n_pins, alpha=alpha_lo)
    res_hi = schur_reduce(cur, cond, chg, cap, num_pins=n_pins, alpha=alpha_hi)
    c_eff_fd = (np.asarray(res_hi.j_eff) - np.asarray(res_lo.j_eff)) / (alpha_hi - alpha_lo)
    print("\n--- effective terminal capacitance (finite-difference) ---")
    print(f"  dC/d(alpha) between alpha={alpha_lo:.0e} and {alpha_hi:.0e}")
    print(format_matrix("C_eff (F) at terminals", c_eff_fd, LABELS[:n_pins]))
    print(f"\n  C_eff[D,D] = {c_eff_fd[0, 0]*1e15:.2f} fF")


if __name__ == "__main__":
    main()
