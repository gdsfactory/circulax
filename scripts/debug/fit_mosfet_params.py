"""Calibrate MosfetSimple params against PSP103 at known bias points.

Uses the DC cross-check fixtures we already generated for VACASK
(tests/fixtures/vacask/*.raw) as ground truth.  Fits Vt, KP, LAMBDA,
THETA (mobility degradation) to minimise RMS log-error across biases.

Model form (NMOS; PMOS via type=-1):

    Vgs_eff = N_SMOOTH * softplus((Vgs - Vt) / N_SMOOTH)
    Id_sat  = KP*(W/L) * Vgs_eff^2 / (2*(1 + THETA*Vgs_eff)) * (1 + LAMBDA*|Vds|)
    I_ds    = type * Id_sat * tanh(Vds / (Vgs_eff + eps))
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))


# ── Reference data (from PSP103 OSDI via VACASK, single device) ──────────
# (W_um, L_um, Vds, Vgs, |Id|) — type=+1 for NMOS, -1 for PMOS
# Source: tests/fixtures/vacask/*.raw (earlier measurements)
NMOS_REF = [
    # W  L   Vds  Vgs     Id
    (10, 1, 1.0, 0.7, 626.33e-6),   # saturation
    (10, 1, 0.6, 1.2, 1629.86e-6),  # triode / strong inversion
    (10, 1, 0.6, 0.3, 46.15e-6),    # near threshold
]
PMOS_REF = [
    # W=20 µm PMOS, Vsg=1.0, Vsd=0.6 → |Id| 2522.94 µA
    (20, 1, 0.6, 1.0, 2522.94e-6),  # saturation-ish
]


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)  # numerically stable


def id_model(Vds, Vgs, W_um, L_um, Vt, KP, LAMBDA, THETA, N_SMOOTH=0.04) -> float:
    """Same formula as MosfetSimple (type=+1 here; magnitude is type-agnostic)."""
    beta = KP * (W_um * 1e-6) / (L_um * 1e-6)
    Vgs_eff = N_SMOOTH * softplus((Vgs - Vt) / N_SMOOTH)
    Vdsat_plus = Vgs_eff + 1e-3
    sat_ramp = np.tanh(Vds / Vdsat_plus)
    mob_factor = 1.0 / (1.0 + THETA * Vgs_eff)
    Id_sat = 0.5 * beta * Vgs_eff**2 * mob_factor * (1.0 + LAMBDA * np.abs(Vds))
    return Id_sat * sat_ramp


def log_rms_error(params: np.ndarray, ref: list) -> float:
    Vt, KP, LAMBDA, THETA = params
    errs = []
    for W, L, Vds, Vgs, Id_ref in ref:
        Id_pred = id_model(Vds, Vgs, W, L, Vt, KP, LAMBDA, THETA)
        # Log-space so near-threshold (small current) gets equal weight to saturation.
        errs.append(np.log(max(Id_pred, 1e-12)) - np.log(Id_ref))
    return float(np.sqrt(np.mean(np.asarray(errs) ** 2)))


def fit(ref: list, label: str, x0=None) -> tuple[float, float, float, float]:
    x0 = x0 or [0.3, 500e-6, 0.05, 0.5]   # Vt, KP, LAMBDA, THETA
    bounds = [
        (0.0, 0.8),     # Vt
        (1e-6, 5e-3),   # KP
        (0.0, 0.3),     # LAMBDA
        (0.0, 5.0),     # THETA
    ]
    res = minimize(
        log_rms_error, x0, args=(ref,),
        method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
    )
    Vt, KP, LAMBDA, THETA = res.x
    print(f"\n{label}:  Vt={Vt:.4f}  KP={KP * 1e6:.1f} µA/V²  "
          f"LAMBDA={LAMBDA:.4f}  THETA={THETA:.4f}")
    print(f"  log-RMS error = {res.fun:.4f}")
    print(f"  bias table:")
    print(f"    {'W':>4s} {'L':>4s} {'Vds':>5s} {'Vgs':>5s} {'Id_ref (µA)':>12s} "
          f"{'Id_fit (µA)':>12s} {'ratio':>7s}")
    for W, L, Vds, Vgs, Id_ref in ref:
        Id_fit = id_model(Vds, Vgs, W, L, Vt, KP, LAMBDA, THETA)
        print(f"    {W:>4d} {L:>4d} {Vds:>5.2f} {Vgs:>5.2f} "
              f"{Id_ref * 1e6:>12.1f} {Id_fit * 1e6:>12.1f} {Id_fit / Id_ref:>7.3f}")
    return tuple(res.x)


if __name__ == "__main__":
    print("Fitting MosfetSimple params against PSP103 OSDI reference.")
    nmos_params = fit(NMOS_REF, "NMOS")
    pmos_params = fit(PMOS_REF, "PMOS (single-point, type=-1)",
                      x0=list(nmos_params))

    print("\n" + "=" * 70)
    print("Recommended MosfetSimple defaults (plug into tests/fixtures/mosfet_simple.py):")
    print("=" * 70)
    nVt, nKP, nL, nT = nmos_params
    print(f"  NMOS: Vt={nVt:.4f}  KP={nKP:.3e}  LAMBDA={nL:.4f}  THETA={nT:.4f}")
    pVt, pKP, pL, pT = pmos_params
    print(f"  PMOS: Vt={pVt:.4f}  KP={pKP:.3e}  LAMBDA={pL:.4f}  THETA={pT:.4f}")
