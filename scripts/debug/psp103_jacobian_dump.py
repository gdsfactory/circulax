"""Dump PSP103 Jacobian at the bosdi author's reference bias (Vds=1.0, Vgs=0.7).

Shows every row of cond/cap plus cur/chg for a single NMOS, classifies each
row (constraint-row vs physics-row) using the +/-1 symmetry test from the
bosdi author's feedback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cdaunt/code/bosdi/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from osdi_jax import osdi_eval  # noqa: E402
from osdi_loader import load_osdi_model  # noqa: E402

from fixtures.psp103_models import (  # noqa: E402
    PSP103_OSDI,
    geom_settings,
    make_psp103_descriptors,
)


def classify_row(g_row: np.ndarray, c_row: np.ndarray, tol: float = 1e-9) -> str:
    """Use the bosdi author's heuristic: a constraint row has exactly +/-1 entries
    with flipped signs on two slots and everything else below tol.
    """
    g_abs_max = np.abs(g_row).max()
    c_abs_max = np.abs(c_row).max()

    # Indices of entries larger than tol
    big = [int(i) for i, v in enumerate(g_row) if abs(v) > tol]
    big_vals = [float(g_row[i]) for i in big]
    is_constraint = (
        len(big) == 2
        and abs(abs(big_vals[0]) - 1.0) < 1e-9
        and abs(abs(big_vals[1]) - 1.0) < 1e-9
        and big_vals[0] * big_vals[1] < 0  # opposite signs
        and c_abs_max < tol
    )
    return (
        f"CONSTRAINT(V[{big[0]}] - V[{big[1]}] = residual)"
        if is_constraint
        else f"physics (|G|_max={g_abs_max:.3e}, |C|_max={c_abs_max:.3e})"
    )


def main() -> None:
    model = load_osdi_model(PSP103_OSDI)
    print(
        f"PSP103 loaded: num_pins={model.num_pins}, num_nodes={model.num_nodes}, "
        f"num_params={model.num_params}"
    )
    print(f"collapsible_pairs (raw pre-collapse indices): {model.collapsible_pairs}")
    print(f"resistive_mask: {list(model.resistive_mask)}")
    print(f"num_resist_jac={model.num_resist_jac}, num_react_jac={model.num_react_jac}")
    print(f"resist_jac_pairs: {model.resist_jac_pairs}")
    print(f"react_jac_pairs: {model.react_jac_pairs}")

    nmos, _ = make_psp103_descriptors()

    def pack(desc, settings):
        merged = desc.make_instance(settings)
        return jnp.array(
            [[merged[k] for k in desc.param_names]], dtype=jnp.float64
        )

    params_n = pack(nmos, geom_settings(10e-6, 1e-6))

    # Bias at the bosdi author's suggested reference: Vds=1.0, Vgs=0.7.
    # Node order in bosdi: D, G, S, B, int0, int1  (terminals, then post-collapse internals).
    # At Vds=1.0, Vgs=0.7, Vbs=0:  V = [1.0, 0.7, 0.0, 0.0, unknown, unknown]
    # We don't know the right V_int values a priori, so we sweep a few guesses
    # and dump the one closest to the DC equilibrium — but for stamping patterns
    # the guesses shouldn't change the constraint-vs-physics classification.
    labels = ("D", "G", "S", "B", "int0", "int1")

    for v_int_guess in (0.0, 0.5, 1.0):
        v = jnp.zeros((1, model.num_nodes))
        v = v.at[0, 0].set(1.0)  # D
        v = v.at[0, 1].set(0.7)  # G
        v = v.at[0, 2].set(0.0)  # S
        v = v.at[0, 3].set(0.0)  # B
        v = v.at[0, 4].set(v_int_guess)
        v = v.at[0, 5].set(v_int_guess)

        states0 = jnp.zeros((1, 0))
        cur, cond, chg, cap, _ = osdi_eval(model.id, v, params_n, states0)
        G = np.asarray(cond).reshape(model.num_nodes, model.num_nodes)
        C = np.asarray(cap).reshape(model.num_nodes, model.num_nodes)
        cur = np.asarray(cur)[0]
        chg = np.asarray(chg)[0]

        print(f"\n==== NMOS at Vds=1.0, Vgs=0.7, V_int0=V_int1={v_int_guess} ====")
        print(f"cur : {cur}")
        print(f"chg : {chg}")
        print("\nPer-row classification (G row | C row):")
        for i in range(model.num_nodes):
            cls = classify_row(G[i], C[i])
            print(f"  row {i:>2} ({labels[i]:>4}): {cls}")
            nz_g = [(j, float(G[i, j])) for j in range(model.num_nodes) if abs(G[i, j]) > 1e-12]
            nz_c = [(j, float(C[i, j])) for j in range(model.num_nodes) if abs(C[i, j]) > 1e-12]
            print(f"     G nonzeros: {nz_g}")
            print(f"     C nonzeros: {nz_c}")


if __name__ == "__main__":
    main()
