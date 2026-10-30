"""Native JAX Angelov GaN channel, charge and thermal/dispersion equations.

Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,
code/angelov/vacode/angelov_gan.va. Regenerate using tests/data/va_models/angelov_gan/regenerate.py.
No OSDI or bosdi runtime is required. Original notice follows verbatim.

The MIT License (MIT)

  Copyright (c) 2002-2015 Tiburon design Automation, Inc

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
 Verilog-A definition of Angelov GaNFET with Thermal Node, Backgate Node and high voltage  E-mode

        based on angelov_gan.va Rev. 2.22.24052023
"""

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, E402, F841, FBT003, PLR0915, PLR0917

from __future__ import annotations

import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _gan_limexp(x):
    limit = 69.07755278982137
    return jnp.where(x > limit, 1e30 * (1.0 + x - limit), jnp.exp(jnp.minimum(x, limit)))


def _gan_channel(*, mode, vgs, vgd, vds, vrf, ipk, vpk, p1, p2, p3, alphar, alphas, lam, lvg, lsb, vtr, ebd, krfdc):
    """Return Ids0 and the source's mode-dependent tanh_psi1."""
    x = vgs - vpk
    xn = vgd - vpk
    psi = p1 * x + p2 * x * x + p3 * x * x * x
    psi_n = p1 * xn + p2 * xn * xn + p3 * xn * xn * xn
    psi_new = p1 * (x + p2 * x * x + p3 * x * x * x)
    psi_n_new = p1 * (xn + p2 * xn * xn + p3 * xn * xn * xn)
    t = 1.0 + jnp.tanh(psi)
    tn = 1.0 + jnp.tanh(psi_n)
    t1_original = 1.0 + jnp.tanh(0.5 * (_gan_limexp(psi) - _gan_limexp(-psi)))
    t1 = 1.0 + jnp.tanh(0.5 * (_gan_limexp(psi_new) - _gan_limexp(-psi_new)))
    t1n = 1.0 + jnp.tanh(0.5 * (_gan_limexp(psi_n_new) - _gan_limexp(-psi_n_new)))
    sat = jnp.tanh((alphar + alphas * t) * vds)
    satn = jnp.tanh((alphar + alphas * tn) * vds)
    sat1 = jnp.tanh((alphar + alphas * t1) * vds)
    sat1n = jnp.tanh((alphar + alphas * t1n) * vds)
    mode0 = ipk * t * sat * (1.0 + lam * vds + lsb * _gan_limexp(-vgd - vtr))
    pos1 = ipk * t * (1.0 + sat) * (1.0 + (lam + lvg * t) * vds + lsb * _gan_limexp(ebd * (vds - vtr)))
    neg1 = ipk * tn * (1.0 - satn) * (1.0 - (lam + lvg * tn) * vds)
    mode1 = 0.5 * (pos1 - neg1)
    mode2 = ipk * t1 * sat1 * (1.0 + (lam + lvg * t1) * vds + lsb * _gan_limexp(ebd * (-vgd - vtr)))
    pos3 = ipk * t1 * (1.0 + sat1) * (1.0 + (lam + lvg * t1) * vds + lsb * _gan_limexp(ebd * (vds - vtr)))
    neg3 = ipk * t1n * (1.0 - sat1n) * (1.0 - (lam + lvg * t1n) * vds)
    mode3 = 0.5 * (pos3 - neg3)
    sat4 = jnp.tanh((alphar + alphas * t1_original) * vds)
    sat4rf = jnp.tanh((alphar + alphas * t1_original) * vrf)
    mode4 = ipk * t * (sat4 + krfdc * sat4rf) * (1.0 + (lam + lvg * t) * (vds + krfdc * vrf) + lsb * _gan_limexp(ebd * (vds - vtr)))
    current = jnp.where(
        mode == 0, mode0, jnp.where(mode == 1, mode1, jnp.where(mode == 2, mode2, jnp.where(mode == 3, mode3, mode4)))
    )
    return current, jnp.where((mode == 2) | (mode == 3), t1, t1_original)


def _gan_logcosh(x):
    return jnp.logaddexp(x, -x) - jnp.log(2.0)


def _gan_charge(
    *,
    mode,
    vgs,
    vgd,
    vds,
    delta,
    cgspi,
    cgs0,
    cgdpi,
    cgd0,
    p10,
    p11,
    p20,
    p21,
    p30,
    p31,
    p40,
    p41,
    p111,
    p222,
    p10pk,
    m,
    tccgs0,
    tccgd0,
    tcp10,
):
    """Return source-equivalent Qgs/Qgd for all five Capmod choices."""
    cgs0 = cgs0 * (1.0 + tccgs0 * delta)
    cgd0 = cgd0 * (1.0 + tccgd0 * delta)
    p10 = p10 * (1.0 + tcp10 * delta)
    p40 = p40 * (1.0 + tcp10 * delta)
    psi1 = p10 + p11 * vgs + p111 * vds
    psi2 = p20 + p21 * vds
    psi3 = p30 - p31 * vds
    psi4 = p40 + p41 * vgd - p111 * vds
    tanh1 = 1.0 + jnp.tanh(psi1)
    tanh2 = 1.0 + jnp.tanh(psi2)
    tanh3 = 1.0 + jnp.tanh(psi3) - p111
    tanh4 = 1.0 + jnp.tanh(psi4)
    gs0, gd0 = cgspi * vgs, cgdpi * vgd
    gs1 = (cgspi + cgs0 * tanh1 * tanh2) * vgs
    gd1 = (cgdpi + cgd0 * (tanh3 * tanh4 + 2.0 * p111)) * vgd
    gs_offset = p10 + p111 * vds
    gd_offset = p40 - p111 * vds
    gs_shape = psi1 + _gan_logcosh(psi1) - gs_offset - _gan_logcosh(gs_offset)
    gd_shape = psi4 + _gan_logcosh(psi4) - gd_offset - _gan_logcosh(gd_offset)
    gs2 = cgs0 * (gs_shape * (tanh2 - p111) / p11 + 2.0 * p111 * vgs) + gs0
    gd2 = cgd0 * (gd_shape * tanh3 / p41 + 2.0 * p111 * vgd) + gd0
    y = vgs / p10pk - 1.0
    cgsdepl = m * (m + y * y) ** (-1.5)
    gs3 = (cgs0 * (1.0 + jnp.tanh(p10 + p11 * (vgs + p111 * vds)) + p222 * cgsdepl) * tanh2 + cgspi) * vgs
    gd3 = (cgd0 * (tanh3 * (1.0 + jnp.tanh(p40 + p41 * (vgd + vds * (1.0 - p111)))) + 2.0 * p111) + cgdpi) * vgd
    gsdepl = p222 * (p10pk + vgs) * (m + y * y) ** (-0.5)
    gsdepl0 = p222 * p10pk * (m + 1.0) ** (-0.5)
    gs4 = cgs0 * ((gs_shape + gsdepl - gsdepl0) * (tanh2 - p111) / p11 + 2.0 * p111 * vgs) + gs0
    gs = jnp.where(mode == 0, gs0, jnp.where(mode == 1, gs1, jnp.where(mode == 2, gs2, jnp.where(mode == 3, gs3, gs4))))
    gd = jnp.where(mode == 0, gd0, jnp.where(mode == 1, gd1, jnp.where(mode == 3, gd3, gd2)))
    return gs, gd


def _AngelovGaN_setup(
    Selft: int = 0,
    Trise: float = 0.0,
    Temp: float = float("nan"),
    Idsmod: int = 0,
    Igmod: int = 0,
    Capmod: int = 2,
    Ipk0: float = 0.05,
    Vpks: float = -0.2,
    Dvpks: float = 0.2,
    P1: float = 0.8,
    P2: float = 0.0,
    P3: float = 0.0,
    Alphar: float = 0.1,
    Alphas: float = 1.0,
    Lambda: float = 0.001,
    Lvg: float = 0.0,
    B1: float = 0.1,
    B2: float = 4.0,
    Lsb0: float = 0.0,
    Vtr: float = 50.0,
    Vsb2: float = 0.0,
    Ebd: float = 0.2,
    Cds: float = 0.0,
    Cgspi: float = 0.0,
    Cgs0: float = 0.0,
    Cgdpi: float = 0.0,
    Cgdpe: float = 0.0,
    Cgd0: float = 0.0,
    P10: float = 0.0,
    P11: float = 1.0,
    P20: float = 0.0,
    P21: float = 0.2,
    P30: float = 0.0,
    P31: float = 0.2,
    P40: float = 0.0,
    P41: float = 1.0,
    P111: float = 0.0,
    P222: float = 0.0,
    P10pk: float = 1.0,
    m: float = 0.5,
    Ij: float = 0.00005,
    Pg: float = float("nan"),
    Ne: float = float("nan"),
    Vjg: float = 0.8,
    Rg: float = 0.05,
    Rd: float = 0.05,
    Rd2: float = 0.0,
    Ri: float = 0.05,
    Rs: float = 0.05,
    Rgd: float = 0.05,
    Ld: float = 0.0,
    Ls: float = 0.0,
    Lg: float = 0.0,
    Ldc: float = 0.1,
    Tau: float = 0.0,
    Rcmin: float = 1.0e3,
    Rc: float = 10.0e3,
    Crf: float = 0.0,
    Rcin: float = 100.0e3,
    Crfin: float = 0.0,
    Rdel: float = 1.0,
    Cdel: float = 1.0e-15,
    Kbgate: float = 0.0,
    KRFDC: float = 0.0,
    Rth: float = 0.001,
    Cth: float = 1e-6,
    Tcipk0: float = -0.002,
    Tcp1: float = -0.002,
    Tcp3: float = 0.00,
    Tcp10: float = 0.00,
    Tccgs0: float = 0.001,
    Tccgd0: float = 0.001,
    Tcrc: float = 0.0,
    Tccrf: float = 0.0,
    Tcrs: float = 0.002,
    TcRtherm: float = 0.001,
    TcVpk: float = 0.001,
    TcVjg: float = -0.001,
    Tclsb0: float = 0.0,
    TcVtr: float = 0.000,
    Kbdgate: float = 0.0,
    Vbdgs: float = 10.0,
    Vbdgd: float = 100.0,
    Pbdg: float = 0.5,
    NoiseR: float = 0.5,
    NoiseP: float = 1.0,
    NoiseC: float = 0.9,
    Fnc: float = 0.0,
    Kf: float = 0.0,
    Af: float = 1.0,
    Ffe: float = 1.0,
    Td: float = 25.0,
    Td1: float = 0.1,
    Tmn: float = 1.0,
    Klf: float = 1.0e-14,
    Fgr: float = 60.0e3,
    Np: float = 0.3,
    Lw: float = 0.1,
    Tnom: float = float("nan"),
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    """Evaluate native Angelov GaN reference equations with parameter-presence flags."""
    given_Ne = jnp.logical_not(jnp.isnan(Ne))
    Ne = jnp.where(given_Ne, Ne, 1.0)
    given_Pg = jnp.logical_not(jnp.isnan(Pg))
    Pg = jnp.where(given_Pg, Pg, 15.0)
    given_Temp = jnp.logical_not(jnp.isnan(Temp))
    Temp = jnp.where(given_Temp, Temp, 25.0)
    given_Tnom = jnp.logical_not(jnp.isnan(Tnom))
    Tnom = jnp.where(given_Tnom, Tnom, 27.0)
    i_v12612 = Selft == 1
    i_v54 = Selft != 0
    i_v58 = jnp.where(given_Temp, (Temp + 273.15), (_temperature + Trise))
    i_v72 = jnp.where(given_Tnom, (Tnom + 273.15), 300.15)
    i_v86 = Rth > 0.0
    i_v260 = jnp.where((~given_Pg), given_Ne, False)
    i_v263 = jnp.divide(0.5, jnp.where((Ne == 0.0) | ~jnp.isfinite(Ne), 1e-300, Ne))
    i_v1793 = Igmod == 0
    i_v2318 = Igmod == 1
    i_v2564 = 0.001 * Kbdgate
    i_v2302 = -Pbdg
    i_v2303 = i_v2302 * Vbdgs
    i_v2376 = jnp.where(
        i_v1793,
        0.0,
        (
            jnp.where(
                (i_v2303 > 69.07755278982137),
                (1e30 + (1e30 * (i_v2303 - 69.07755278982137))),
                (jnp.exp(jnp.clip(i_v2303, -709.0, 709.0))),
            )
        ),
    )
    i_v5198 = jnp.where((Capmod == 2), True, (Capmod == 4))
    i_v2311 = i_v2302 * Vbdgd
    i_v2609 = jnp.where(
        i_v1793,
        0.0,
        (
            jnp.where(
                (i_v2311 > 69.07755278982137),
                (1e30 + (1e30 * (i_v2311 - 69.07755278982137))),
                (jnp.exp(jnp.clip(i_v2311, -709.0, 709.0))),
            )
        ),
    )
    i_v3516 = 2.0 * P111
    i_v6768 = jnp.where((Cdel > 0.0), True, (Rdel > 0.0))
    i_v7646 = Rcin > 0.0
    i_v217 = jnp.where((jnp.where((Idsmod == 1), True, (Idsmod == 4))), (Capmod == 4), False)
    i_v7848 = Rgd > 0.0
    i_v8255 = Ri > 0.0
    i_v4256 = jnp.divide(Tau, 3.0)
    i_v6516 = Rc > 0.0
    i_v1511 = jnp.where((jnp.where((Idsmod == 0), True, (Idsmod == 1))), True, (Idsmod == 4))
    i_v8466 = Rg > 0.0
    i_v8912 = Rs > 0.0
    i_v9453 = jnp.where((Rd > 0.0), True, (Rd2 > 0.0))
    return jnp.array(
        [
            i_v12612,
            i_v54,
            i_v58,
            i_v72,
            i_v86,
            i_v260,
            i_v263,
            i_v1793,
            i_v2318,
            i_v2564,
            i_v2376,
            i_v5198,
            i_v2609,
            i_v3516,
            i_v6768,
            i_v7646,
            i_v217,
            i_v7848,
            i_v8255,
            i_v4256,
            i_v6516,
            i_v1511,
            i_v8466,
            i_v8912,
            i_v9453,
        ]
    )


@component(
    ports=("d", "g", "s", "t", "rf"),
    states=(
        "v_di",
        "v_dii",
        "v_gi",
        "v_si",
        "v_sii",
        "v_gdi",
        "v_gsi",
        "v_gii",
        "v_ggi",
        "v_bi",
        "v_xt1",
        "v_xt2",
        "i_un_xt1_xt2_19",
        "i_un_rf_si_20",
        "i_un_gi_ggi_21",
        "i_un_g_gi_22",
        "i_un_si_sii_23",
        "i_un_sii_s_24",
        "i_un_di_dii_25",
        "i_un_dii_d_26",
    ),
)
def AngelovGaN(
    signals: Signals,
    s: States,
    init,
    Selft: int = 0,
    Trise: float = 0.0,
    Temp: float = float("nan"),
    Idsmod: int = 0,
    Igmod: int = 0,
    Capmod: int = 2,
    Ipk0: float = 0.05,
    Vpks: float = -0.2,
    Dvpks: float = 0.2,
    P1: float = 0.8,
    P2: float = 0.0,
    P3: float = 0.0,
    Alphar: float = 0.1,
    Alphas: float = 1.0,
    Lambda: float = 0.001,
    Lvg: float = 0.0,
    B1: float = 0.1,
    B2: float = 4.0,
    Lsb0: float = 0.0,
    Vtr: float = 50.0,
    Vsb2: float = 0.0,
    Ebd: float = 0.2,
    Cds: float = 0.0,
    Cgspi: float = 0.0,
    Cgs0: float = 0.0,
    Cgdpi: float = 0.0,
    Cgdpe: float = 0.0,
    Cgd0: float = 0.0,
    P10: float = 0.0,
    P11: float = 1.0,
    P20: float = 0.0,
    P21: float = 0.2,
    P30: float = 0.0,
    P31: float = 0.2,
    P40: float = 0.0,
    P41: float = 1.0,
    P111: float = 0.0,
    P222: float = 0.0,
    P10pk: float = 1.0,
    m: float = 0.5,
    Ij: float = 0.00005,
    Pg: float = float("nan"),
    Ne: float = float("nan"),
    Vjg: float = 0.8,
    Rg: float = 0.05,
    Rd: float = 0.05,
    Rd2: float = 0.0,
    Ri: float = 0.05,
    Rs: float = 0.05,
    Rgd: float = 0.05,
    Ld: float = 0.0,
    Ls: float = 0.0,
    Lg: float = 0.0,
    Ldc: float = 0.1,
    Tau: float = 0.0,
    Rcmin: float = 1.0e3,
    Rc: float = 10.0e3,
    Crf: float = 0.0,
    Rcin: float = 100.0e3,
    Crfin: float = 0.0,
    Rdel: float = 1.0,
    Cdel: float = 1.0e-15,
    Kbgate: float = 0.0,
    KRFDC: float = 0.0,
    Rth: float = 0.001,
    Cth: float = 1e-6,
    Tcipk0: float = -0.002,
    Tcp1: float = -0.002,
    Tcp3: float = 0.00,
    Tcp10: float = 0.00,
    Tccgs0: float = 0.001,
    Tccgd0: float = 0.001,
    Tcrc: float = 0.0,
    Tccrf: float = 0.0,
    Tcrs: float = 0.002,
    TcRtherm: float = 0.001,
    TcVpk: float = 0.001,
    TcVjg: float = -0.001,
    Tclsb0: float = 0.0,
    TcVtr: float = 0.000,
    Kbdgate: float = 0.0,
    Vbdgs: float = 10.0,
    Vbdgd: float = 100.0,
    Pbdg: float = 0.5,
    NoiseR: float = 0.5,
    NoiseP: float = 1.0,
    NoiseC: float = 0.9,
    Fnc: float = 0.0,
    Kf: float = 0.0,
    Af: float = 1.0,
    Ffe: float = 1.0,
    Td: float = 25.0,
    Td1: float = 0.1,
    Tmn: float = 1.0,
    Klf: float = 1.0e-14,
    Fgr: float = 60.0e3,
    Np: float = 0.3,
    Lw: float = 0.1,
    Tnom: float = float("nan"),
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """GaN RF transistor with external thermal/dispersion ports.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 2.22.24052023, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `angelov_gan.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/angelov/vacode/angelov_gan.va).
    Attribution: Tiburon design Automation, Inc. (MIT notice).
    The retained source and includes are under `tests/data/va_models/angelov_gan/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/angelov/vacode/angelov_gan.va),
    with comparison cases in `tests/test_angelov_gan.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        angelov_gan.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.angelov_gan import AngelovGaN

        jax.config.update("jax_enable_x64", True)
        model = AngelovGaN()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(AngelovGaN.__name__, "ports:", model.ports)
        ```

    """
    given_Ne = jnp.logical_not(jnp.isnan(Ne))
    Ne = jnp.where(given_Ne, Ne, 1.0)
    given_Pg = jnp.logical_not(jnp.isnan(Pg))
    Pg = jnp.where(given_Pg, Pg, 15.0)
    given_Temp = jnp.logical_not(jnp.isnan(Temp))
    Temp = jnp.where(given_Temp, Temp, 25.0)
    given_Tnom = jnp.logical_not(jnp.isnan(Tnom))
    Tnom = jnp.where(given_Tnom, Tnom, 27.0)
    """Auto-generated from Verilog-A."""
    i_v12612 = init[0]
    i_v54 = init[1]
    i_v58 = init[2]
    i_v72 = init[3]
    i_v86 = init[4]
    i_v260 = init[5]
    i_v263 = init[6]
    i_v1793 = init[7]
    i_v2318 = init[8]
    i_v2564 = init[9]
    i_v2376 = init[10]
    i_v5198 = init[11]
    i_v2609 = init[12]
    i_v3516 = init[13]
    i_v6768 = init[14]
    i_v7646 = init[15]
    i_v217 = init[16]
    i_v7848 = init[17]
    i_v8255 = init[18]
    i_v4256 = init[19]
    i_v6516 = init[20]
    i_v1511 = init[21]
    i_v8466 = init[22]
    i_v8912 = init[23]
    i_v9453 = init[24]
    v28370 = -s.i_un_dii_d_26
    v28371 = v28370 + 1e-15
    v31027 = _mfactor * v28371
    v31028 = _mfactor * s.i_un_g_gi_22
    v28360 = -s.i_un_sii_s_24
    v28372 = v28360 - 1e-15
    v28374 = v28372 - 1e-12
    v10484 = (s.v_gii - signals.s) * 1e-12
    v28376 = v28374 - v10484
    v31030 = _mfactor * v28376
    v61 = signals.t < 0.0
    v62 = -signals.t
    v63 = jnp.where(v61, v62, signals.t)
    v64 = i_v58 + v63
    v66 = jnp.where(i_v54, v64, i_v58)
    v77 = v66 - i_v72
    v78 = v77 < 0.0
    v79 = -v77
    v80 = jnp.where(v78, v79, v77)
    v83 = v80 > 0.0
    v87 = jnp.where(v83, True, i_v86)
    v94 = v80 < 0.0
    v108 = -v80
    v109 = jnp.where(v94, v108, v80)
    v110 = Tcipk0 * v109
    v111 = 1.0 + v110
    v112 = Ipk0 * v111
    v815 = jnp.where(v87, v112, Ipk0)
    v380 = B2 * (s.v_di - s.v_si)
    v381 = jnp.cosh(v380)
    v453 = v381 * v381
    v454 = 1e-12 + v453
    v455 = jnp.divide(B1, jnp.where((v454 == 0.0) | ~jnp.isfinite(v454), 1e-300, v454))
    v456 = 1.0 + v455
    v457 = P1 * v456
    v473 = v80 < 0.0
    v474 = -v80
    v475 = jnp.where(v473, v474, v80)
    v476 = Tcp1 * v475
    v477 = 1.0 + v476
    v478 = v457 * v477
    v184 = TcVpk * v80
    v185 = Vpks + v184
    v501 = jnp.where(v87, v185, Vpks)
    v522 = v501 - Dvpks
    v530 = Alphas * (s.v_di - s.v_si)
    v531 = jnp.tanh(v530)
    v532 = Dvpks * v531
    v533 = v522 + v532
    v447 = Kbgate * (signals.rf - s.v_si)
    v540 = v533 - v447
    v20 = -(s.v_gdi - s.v_di)
    v207 = TcVtr * v80
    v208 = Vtr + v207
    v619 = jnp.where(v87, v208, Vtr)
    v639 = v20 - v619
    v640 = Vsb2 * v639
    v642 = v640 * v639
    v643 = v540 - v642
    v650 = -v80
    v651 = jnp.where(v473, v650, v80)
    v652 = TcVpk * v651
    v653 = 1.0 + v652
    v654 = v643 * v653
    v728 = s.v_gii - s.v_si - v654
    v738 = v478 * v728
    v730 = v728 * v728
    v740 = P2 * v730
    v741 = v738 + v740
    v487 = -v80
    v488 = jnp.where(v473, v487, v80)
    v489 = Tcp3 * v488
    v490 = 1.0 + v489
    v491 = P3 * v490
    v745 = v491 * v728
    v746 = v745 * v730
    v747 = v741 + v746
    v750 = jnp.tanh(v747)
    v751 = 1.0 + v750
    v854 = v815 * v751
    v784 = Alphas * v751
    v785 = Alphar + v784
    v796 = v785 * (s.v_di - s.v_si)
    v797 = jnp.tanh(v796)
    v856 = v854 * v797
    v860 = Lambda * (s.v_di - s.v_si)
    v861 = 1.0 + v860
    v121 = -v80
    v122 = jnp.where(v94, v121, v80)
    v123 = Tclsb0 * v122
    v124 = 1.0 + v123
    v125 = Lsb0 * v124
    v877 = jnp.where(v87, v125, Lsb0)
    v933 = v639 > 69.07755278982137
    v934 = v639 - 69.07755278982137
    v935 = 1e30 * v934
    v936 = 1e30 + v935
    v937 = jnp.exp(jnp.clip(v639, -709.0, 709.0))
    v938 = jnp.where(v933, v936, v937)
    v939 = v877 * v938
    v940 = v861 + v939
    v941, _mode_tanh1 = _gan_channel(
        mode=Idsmod,
        vgs=s.v_gii - s.v_si,
        vgd=s.v_gdi - s.v_di,
        vds=s.v_di - s.v_si,
        vrf=signals.rf - s.v_si,
        ipk=v815,
        vpk=v654,
        p1=v478,
        p2=P2,
        p3=v491,
        alphar=Alphar,
        alphas=Alphas,
        lam=Lambda,
        lvg=Lvg,
        lsb=v877,
        vtr=v619,
        ebd=Ebd,
        krfdc=KRFDC,
    )
    v756 = v747 > 69.07755278982137
    v757 = v747 - 69.07755278982137
    v758 = 1e30 * v757
    v759 = 1e30 + v758
    v760 = jnp.exp(jnp.clip(v747, -709.0, 709.0))
    v761 = jnp.where(v756, v759, v760)
    v765 = -v747
    v766 = v765 > 69.07755278982137
    v767 = v765 - 69.07755278982137
    v768 = 1e30 * v767
    v769 = 1e30 + v768
    v770 = jnp.exp(jnp.clip(v765, -709.0, 709.0))
    v771 = jnp.where(v766, v769, v770)
    v772 = v761 - v771
    v773 = 0.5 * v772
    v774 = jnp.tanh(v773)
    v775 = _mode_tanh1
    v12705 = v941 * (s.v_di - s.v_si)
    v12706 = v12705 < 0.0
    v12707 = -v12705
    v12708 = jnp.where(v12706, v12707, v12705)
    v68 = 8.617332384960955e-05 * v66
    v313 = jnp.divide(i_v263, jnp.where((v68 == 0.0) | ~jnp.isfinite(v68), 1e-300, v68))
    v1821 = jnp.where(i_v260, v313, Pg)
    v202 = TcVjg * v80
    v203 = Vjg + v202
    v1906 = jnp.where(v87, v203, Vjg)
    v2132 = s.v_gsi - s.v_si - v1906
    v2340 = s.v_gsi - s.v_si - v1906
    v2341 = jnp.tanh(v2340)
    v2357 = s.v_gsi - s.v_si - v1906
    v2553 = jnp.where(i_v2318, v2341, v2357)
    v2552 = jnp.where(i_v1793, v2132, v2553)
    v2555 = v1821 * v2552
    v2556 = v2555 > 69.07755278982137
    v2557 = v2555 - 69.07755278982137
    v2558 = 1e30 * v2557
    v2559 = 1e30 + v2558
    v2560 = jnp.exp(jnp.clip(v2555, -709.0, 709.0))
    v2561 = jnp.where(v2556, v2559, v2560)
    v2135 = -(s.v_gsi - s.v_si)
    v2137 = v2135 - Vbdgs
    v2361 = -(s.v_gsi - s.v_si)
    v2362 = v2361 - Vbdgs
    v2366 = jnp.where(i_v1793, v2137, v2362)
    v2367 = Pbdg * v2366
    v2368 = v2367 > 69.07755278982137
    v2369 = v2367 - 69.07755278982137
    v2370 = 1e30 * v2369
    v2371 = 1e30 + v2370
    v2372 = jnp.exp(jnp.clip(v2367, -709.0, 709.0))
    v2373 = jnp.where(v2368, v2371, v2372)
    v2528 = v2373 - i_v2376
    v2568 = i_v2564 * v2528
    v2569 = v2561 - v2568
    v1975 = -1.0 * v1906
    v1976 = jnp.tanh(v1975)
    v1977 = v1821 * v1976
    v1978 = v1977 > 69.07755278982137
    v1979 = v1977 - 69.07755278982137
    v1980 = 1e30 * v1979
    v1981 = 1e30 + v1980
    v1982 = jnp.exp(jnp.clip(v1977, -709.0, 709.0))
    v1983 = jnp.where(v1978, v1981, v1982)
    v2292 = -v1821
    v2294 = v2292 * v1906
    v2295 = v2294 > 69.07755278982137
    v2296 = v2294 - 69.07755278982137
    v2297 = 1e30 * v2296
    v2298 = 1e30 + v2297
    v2299 = jnp.exp(jnp.clip(v2294, -709.0, 709.0))
    v2300 = jnp.where(v2295, v2298, v2299)
    v2574 = jnp.where(i_v1793, v1983, v2300)
    v2586 = v2569 - v2574
    v2587 = Ij * v2586
    v12820 = v2587 * (s.v_gsi - s.v_si)
    v12821 = v12820 < 0.0
    v12822 = -v12820
    v12823 = jnp.where(v12821, v12822, v12820)
    v12824 = v12708 + v12823
    v12825 = -1.0 * v12824
    v1787 = -v80
    v1788 = jnp.where(v473, v1787, v80)
    v1789 = TcRtherm * v1788
    v1790 = 1.0 + v1789
    v1791 = Rth * v1790
    v13256 = jnp.divide(signals.t, jnp.where((v1791 == 0.0) | ~jnp.isfinite(v1791), 1e-300, v1791))
    v13257 = v12825 + v13256
    v13261 = v13257 + 0.0
    v13262 = signals.t * 1e-12
    v21886 = jnp.where(i_v12612, v13261, v13262)
    v31031 = _mfactor * v21886
    v13259 = Cth * signals.t
    v21918 = jnp.where(i_v12612, v13259, 0.0)
    v31032 = _mfactor * v21918
    v6510 = 1e-12 * (s.v_dii - signals.rf)
    v6511 = 0.0 + v6510
    v28320 = -v6511
    v28325 = v28320 + s.i_un_rf_si_20
    v31034 = _mfactor * v28325
    v173 = -v80
    v174 = jnp.where(v94, v173, v80)
    v175 = Tccrf * v174
    v176 = 1.0 + v175
    v177 = Crf * v176
    v6222 = jnp.where(v87, v177, Crf)
    v6321 = v6222 * (s.v_dii - signals.rf)
    v28321 = -v6321
    v31036 = _mfactor * v28321
    v6172 = s.v_xt2 + 0.0
    v2284 = s.v_gdi - s.v_di - v1906
    v2353 = s.v_gdi - s.v_di - v1906
    v2354 = jnp.tanh(v2353)
    v2359 = s.v_gdi - s.v_di - v1906
    v2775 = jnp.where(i_v2318, v2354, v2359)
    v2774 = jnp.where(i_v1793, v2284, v2775)
    v2779 = v1821 * v2774
    v2780 = v2779 > 69.07755278982137
    v2781 = v2779 - 69.07755278982137
    v2782 = 1e30 * v2781
    v2783 = 1e30 + v2782
    v2784 = jnp.exp(jnp.clip(v2779, -709.0, 709.0))
    v2785 = jnp.where(v2780, v2783, v2784)
    v2289 = v20 - Vbdgd
    v2365 = v20 - Vbdgd
    v2593 = jnp.where(i_v1793, v2289, v2365)
    v2596 = Pbdg * v2593
    v2597 = v2596 > 69.07755278982137
    v2598 = v2596 - 69.07755278982137
    v2599 = 1e30 * v2598
    v2600 = 1e30 + v2599
    v2601 = jnp.exp(jnp.clip(v2596, -709.0, 709.0))
    v2602 = jnp.where(v2597, v2600, v2601)
    v2760 = v2602 - i_v2609
    v2790 = i_v2564 * v2760
    v2791 = v2785 - v2790
    v2798 = v2791 - v2574
    v2799 = Ij * v2798
    v5387 = v2799 + 0.0
    v5768 = v2799 + 0.0
    v12525 = jnp.where(i_v5198, v5387, v5768)
    v28310 = v6172 - v12525
    v28315 = v28310 - 0.0
    v28365 = v28315 + s.i_un_di_dii_25
    v31038 = _mfactor * v28365
    v6164 = Cds * (s.v_di - s.v_si)
    v147 = -v80
    v148 = jnp.where(v94, v147, v80)
    v149 = Tccgd0 * v148
    v150 = 1.0 + v149
    v151 = Cgd0 * v150
    v3379 = jnp.where(v87, v151, Cgd0)
    v190 = Tcp10 * v80
    v191 = 1.0 + v190
    v198 = P40 * v191
    v3074 = jnp.where(v87, v198, P40)
    v3175 = P41 * (s.v_gdi - s.v_di)
    v3176 = v3074 + v3175
    v3006 = P111 * (s.v_di - s.v_si)
    v3178 = v3176 - v3006
    v3536 = jnp.cosh(v3178)
    v3537 = jnp.log(jnp.maximum(v3536, 1e-300))
    v3545 = v3178 + v3537
    v3529 = v3074 - v3006
    v3530 = jnp.cosh(v3529)
    v3531 = jnp.log(jnp.maximum(v3530, 1e-300))
    v3541 = v3529 + v3531
    v3546 = v3545 - v3541
    v3024 = P31 * (s.v_di - s.v_si)
    v3025 = P30 - v3024
    v3028 = jnp.tanh(v3025)
    v3029 = 1.0 + v3028
    v3030 = v3029 - P111
    v3549 = v3546 * v3030
    v3550 = jnp.divide(v3549, jnp.where((P41 == 0.0) | ~jnp.isfinite(P41), 1e-300, P41))
    v3556 = i_v3516 * (s.v_gdi - s.v_di)
    v3557 = v3550 + v3556
    v3558 = v3379 * v3557
    v3559 = Cgdpi * (s.v_gdi - s.v_di)
    v3560 = v3558 + v3559
    v21894 = jnp.where(i_v5198, v3560, 0.0)
    v5765 = Cgdpi * (s.v_gdi - s.v_di)
    v21901 = jnp.where(i_v5198, 0.0, v5765)
    _mode_qgs, v21957 = _gan_charge(
        mode=Capmod,
        vgs=s.v_gsi - s.v_si,
        vgd=s.v_gdi - s.v_di,
        vds=s.v_di - s.v_si,
        delta=v80,
        cgspi=Cgspi,
        cgs0=Cgs0,
        cgdpi=Cgdpi,
        cgd0=Cgd0,
        p10=P10,
        p11=P11,
        p20=P20,
        p21=P21,
        p30=P30,
        p31=P31,
        p40=P40,
        p41=P41,
        p111=P111,
        p222=P222,
        p10pk=P10pk,
        m=m,
        tccgs0=Tccgs0,
        tccgd0=Tccgd0,
        tcp10=Tcp10,
    )
    v28311 = v6164 - v21957
    v5964 = Cgdpe * (s.v_gi - s.v_di)
    v21968 = v5964 + 0.0
    v28316 = v28311 - v21968
    v31040 = _mfactor * v28316
    v28366 = v6511 - s.i_un_di_dii_25
    v28369 = v28366 + s.i_un_dii_d_26
    v31044 = _mfactor * v28369
    v31045 = _mfactor * v6321
    v28345 = 0.0 + s.i_un_gi_ggi_21
    v28350 = v28345 - s.i_un_g_gi_22
    v28387 = v28350 + 0.0
    v31047 = _mfactor * v28387
    v31048 = _mfactor * v21968
    v28294 = -v6172
    v5576 = v2587 + 0.0
    v5960 = v2587 + 0.0
    v12407 = jnp.where(i_v5198, v5576, v5960)
    v12401 = jnp.where(i_v6768, v12407, 0.0)
    v28302 = v28294 - v12401
    v28326 = v28302 - s.i_un_rf_si_20
    v7649 = jnp.divide(s.v_bi - s.v_si, jnp.where((Rcin == 0.0) | ~jnp.isfinite(Rcin), 1e-300, Rcin))
    v18279 = jnp.where(i_v7646, v7649, 0.0)
    v28334 = v28326 - v18279
    v28355 = v28334 + s.i_un_si_sii_23
    v28389 = v28355 - 0.0
    v31052 = _mfactor * v28389
    v28295 = -v6164
    v134 = -v80
    v135 = jnp.where(v94, v134, v80)
    v136 = Tccgs0 * v135
    v137 = 1.0 + v136
    v138 = Cgs0 * v137
    v3232 = jnp.where(v87, v138, Cgs0)
    v192 = P10 * v191
    v2843 = jnp.where(v87, v192, P10)
    v2944 = P11 * (s.v_gsi - s.v_si)
    v2945 = v2843 + v2944
    v3007 = v2945 + v3006
    v3501 = jnp.cosh(v3007)
    v3503 = jnp.log(jnp.maximum(v3501, 1e-300))
    v3511 = v3007 + v3503
    v3493 = v2843 + v3006
    v3494 = jnp.cosh(v3493)
    v3496 = jnp.log(jnp.maximum(v3494, 1e-300))
    v3507 = v3493 + v3496
    v3512 = v3511 - v3507
    v3015 = P21 * (s.v_di - s.v_si)
    v3016 = P20 + v3015
    v3019 = jnp.tanh(v3016)
    v3020 = 1.0 + v3019
    v3485 = v3020 - P111
    v3513 = v3512 * v3485
    v3514 = jnp.divide(v3513, jnp.where((P11 == 0.0) | ~jnp.isfinite(P11), 1e-300, P11))
    v3520 = i_v3516 * (s.v_gsi - s.v_si)
    v3521 = v3514 + v3520
    v3522 = v3232 * v3521
    v3523 = Cgspi * (s.v_gsi - s.v_si)
    v3524 = v3522 + v3523
    v21897 = jnp.where(i_v5198, v3524, 0.0)
    v21898 = jnp.where(i_v6768, v21897, 0.0)
    v5957 = Cgspi * (s.v_gsi - s.v_si)
    v21904 = jnp.where(i_v5198, 0.0, v5957)
    v21905 = jnp.where(i_v6768, v21904, 0.0)
    v21958 = _mode_qgs
    v28303 = v28295 - v21958
    v227 = v80 * v80
    v228 = Tccrf * v227
    v229 = 1.0 + v228
    v237 = Cdel * v229
    v252 = -v80
    v253 = jnp.where(v94, v252, v80)
    v254 = Tccrf * v253
    v255 = 1.0 + v254
    v256 = Cdel * v255
    v7163 = jnp.where(i_v217, v237, v256)
    v7162 = jnp.where(v87, v7163, Cdel)
    v7248 = v7162 * (s.v_gii - s.v_si)
    v21915 = jnp.where(i_v6768, v7248, 0.0)
    v28330 = v28303 - v21915
    v31054 = _mfactor * v28330
    v28356 = -s.i_un_si_sii_23
    v28359 = v28356 + s.i_un_sii_s_24
    v31058 = _mfactor * v28359
    v7851 = jnp.divide(s.v_ggi - s.v_gdi, jnp.where((Rgd == 0.0) | ~jnp.isfinite(Rgd), 1e-300, Rgd))
    v8252 = jnp.where(i_v7848, v7851, 0.0)
    v28338 = v12525 - v8252
    v31060 = _mfactor * v28338
    v31061 = _mfactor * v21957
    v230 = Rdel * v229
    v242 = -v80
    v243 = jnp.where(v94, v242, v80)
    v244 = Tccrf * v243
    v245 = 1.0 + v244
    v246 = Rdel * v245
    v6825 = jnp.where(i_v217, v230, v246)
    v6824 = jnp.where(v87, v6825, Rdel)
    v6913 = jnp.divide(s.v_gsi - s.v_gii, jnp.where((v6824 == 0.0) | ~jnp.isfinite(v6824), 1e-300, v6824))
    v16871 = jnp.where(i_v6768, v6913, 0.0)
    v28327 = v12401 + v16871
    v8258 = jnp.divide(s.v_ggi - s.v_gsi, jnp.where((Ri == 0.0) | ~jnp.isfinite(Ri), 1e-300, Ri))
    v19186 = jnp.where(i_v8255, v8258, 0.0)
    v28340 = v28327 - v19186
    v31063 = _mfactor * v28340
    v7447 = Crfin * (s.v_gsi - s.v_bi)
    v28331 = v21958 + v7447
    v31065 = _mfactor * v28331
    v28328 = -v16871
    v28375 = v28328 + v10484
    v31067 = _mfactor * v28375
    v31068 = _mfactor * v21915
    v28339 = v8252 + v19186
    v28346 = v28339 - s.i_un_gi_ggi_21
    v31070 = _mfactor * v28346
    v28373 = v18279 + 1e-12
    v31072 = _mfactor * v28373
    v28332 = -v7447
    v31074 = _mfactor * v28332
    v3886 = -v941
    v4072 = v3886 + 0.0
    v28280 = v4072 + s.i_un_xt1_xt2_19
    v31076 = _mfactor * v28280
    v4070 = Tau * s.v_xt1
    v31077 = _mfactor * v4070
    v28281 = s.v_xt2 - s.i_un_xt1_xt2_19
    v31079 = _mfactor * v28281
    v28279 = -(s.v_xt1 - s.v_xt2)
    v4258 = i_v4256 * s.i_un_xt1_xt2_19
    v160 = -v80
    v161 = jnp.where(v94, v160, v80)
    v162 = Tcrc * v161
    v163 = 1.0 + v162
    v164 = Rc * v163
    v1536 = jnp.where(v87, v164, Rc)
    v1648 = 1.0 + v751
    v1649 = jnp.divide(v1536, jnp.where((v1648 == 0.0) | ~jnp.isfinite(v1648), 1e-300, v1648))
    v1650 = Rcmin + v1649
    v1688 = 1.0 + v775
    v1689 = jnp.divide(v1536, jnp.where((v1688 == 0.0) | ~jnp.isfinite(v1688), 1e-300, v1688))
    v1690 = Rcmin + v1689
    v6542 = jnp.where(i_v1511, v1650, v1690)
    v6575 = s.i_un_rf_si_20 * v6542
    v6577 = v6575 + 0.0
    v16203 = jnp.where(i_v6516, v6577, 0.0)
    v28324 = v16203 - (signals.rf - s.v_si)
    v6514 = Ldc * s.i_un_rf_si_20
    v21914 = jnp.where(i_v6516, v6514, 0.0)
    v8469 = s.i_un_gi_ggi_21 * Rg
    v19417 = jnp.where(i_v8466, v8469, 0.0)
    v28344 = v19417 - (s.v_gi - s.v_ggi)
    v28348 = -(signals.g - s.v_gi)
    v8696 = Lg * s.i_un_g_gi_22
    v1654 = Rd2 * v751
    v1659 = Rs + v1654
    v1691 = Rd2 * v775
    v1694 = Rs + v1691
    v1695 = jnp.where(i_v1511, v1659, v1694)
    v1761 = -v80
    v1762 = jnp.where(v473, v1761, v80)
    v1763 = Tcrs * v1762
    v1764 = 1.0 + v1763
    v1765 = v1695 * v1764
    v8996 = s.i_un_si_sii_23 * v1765
    v19853 = jnp.where(i_v8912, v8996, 0.0)
    v28354 = v19853 - (s.v_si - s.v_sii)
    v28358 = -(s.v_sii - signals.s)
    v9229 = Ls * s.i_un_sii_s_24
    v1655 = Rd + v1654
    v1692 = Rd + v1691
    v1769 = jnp.where(i_v1511, v1655, v1692)
    v1776 = -v80
    v1777 = jnp.where(v473, v1776, v80)
    v1778 = Tcrs * v1777
    v1779 = 1.0 + v1778
    v1780 = v1769 * v1779
    v9543 = s.i_un_di_dii_25 * v1780
    v20245 = jnp.where(i_v9453, v9543, 0.0)
    v28364 = v20245 - (s.v_di - s.v_dii)
    v28368 = -(s.v_dii - signals.d)
    v9790 = Ld * s.i_un_dii_d_26
    return {
        "d": v31027,
        "g": v31028,
        "s": v31030,
        "t": v31031,
        "rf": v31034,
        "v_di": v31038,
        "v_dii": v31044,
        "v_gi": v31047,
        "v_si": v31052,
        "v_sii": v31058,
        "v_gdi": v31060,
        "v_gsi": v31063,
        "v_gii": v31067,
        "v_ggi": v31070,
        "v_bi": v31072,
        "v_xt1": v31076,
        "v_xt2": v31079,
        "i_un_xt1_xt2_19": v28279,
        "i_un_rf_si_20": v28324,
        "i_un_gi_ggi_21": v28344,
        "i_un_g_gi_22": v28348,
        "i_un_si_sii_23": v28354,
        "i_un_sii_s_24": v28358,
        "i_un_di_dii_25": v28364,
        "i_un_dii_d_26": v28368,
    }, {
        "t": v31032,
        "rf": v31036,
        "v_di": v31040,
        "v_dii": v31045,
        "v_gi": v31048,
        "v_si": v31054,
        "v_gdi": v31061,
        "v_gsi": v31065,
        "v_gii": v31068,
        "v_bi": v31074,
        "v_xt1": v31077,
        "i_un_xt1_xt2_19": v4258,
        "i_un_rf_si_20": v21914,
        "i_un_g_gi_22": v8696,
        "i_un_sii_s_24": v9229,
        "i_un_dii_d_26": v9790,
    }


@AngelovGaN.setup
def _AngelovGaN_register_setup(*_a, **_kw):
    return _AngelovGaN_setup(*_a, **_kw)


import inspect
from numbers import Real

_gan_init = AngelovGaN.__init__
_gan_signature = inspect.signature(_gan_init)


def _gan_checked_init(self, *args, **kwargs):
    bound = _gan_signature.bind(self, *args, **kwargs)
    for name in ("Rg", "Rd", "Rs", "Ri", "Rgd", "Rc", "Rcin", "Rdel", "Cdel", "Rth"):
        value = bound.arguments.get(name, _gan_signature.parameters[name].default)
        if isinstance(value, Real) and value <= 0:
            message = f"AngelovGaN requires {name} > 0 for its retained finite-resistance topology"
            raise ValueError(message)
    for name, allowed in (("Idsmod", range(5)), ("Capmod", range(5)), ("Igmod", range(2)), ("Selft", range(2))):
        value = bound.arguments.get(name, _gan_signature.parameters[name].default)
        if isinstance(value, Real) and value not in allowed:
            message = f"AngelovGaN {name} must be one of {tuple(allowed)}"
            raise ValueError(message)
    _gan_init(self, *args, **kwargs)


_gan_checked_init.__signature__ = _gan_signature
AngelovGaN.__init__ = _gan_checked_init
