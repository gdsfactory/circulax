# ruff: noqa: ANN001, ANN002, ANN003, ANN202, E402, EM102, FBT003, PLR0915, PLR0917
# Native Angelov: upstream6099b4e12e6f66ff530f59b8d01fb909094be85a; bosdi159907ddf29a0225614cf1b7999214401b0b5a2d.
#
#
#  The MIT License (MIT)
#
#  Copyright (c) 2002-2015 Tiburon design Automation, Inc
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#  Verilog-A definition of Angelov GaNFET
#
#  Rev. 2.10.06813
#
"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _Angelov_setup(
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
    Vtr: float = 20.0,
    Vsb2: float = 0.0,
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
    Ij: float = 5e-05,
    Pg: float = float("nan"),
    Ne: float = float("nan"),
    Vjg: float = 0.7,
    Rg: float = 0.05,
    Rd: float = 0.05,
    Rd2: float = 0.0,
    Ri: float = 0.05,
    Rs: float = 0.05,
    Rgd: float = 0.05,
    Ld: float = 0.0,
    Ls: float = 0.0,
    Lg: float = 0.0,
    Tau: float = 0.0,
    Rcmin: float = 1000.0,
    Rc: float = 10000.0,
    Crf: float = 0.0,
    Rcin: float = 100000.0,
    Crfin: float = 0.0,
    Rth: float = 0.001,
    Cth: float = 0.0001,
    Tcipk0: float = -0.002,
    Tcp1: float = -0.002,
    Tccgs0: float = 0.002,
    Tccgd0: float = 0.002,
    Tclsb0: float = 0.0,
    Tcrc: float = 0.0,
    Tccrf: float = 0.0,
    Tcrs: float = 0.003,
    TcRtherm: float = 0.001,
    TcVpk: float = 0.001,
    TcVjg: float = -0.001,
    TcVtr: float = 0.0,
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
    Klf: float = 1e-14,
    Fgr: float = 60000.0,
    Np: float = 0.3,
    Lw: float = 0.1,
    Tnom: float = float("nan"),
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    _given_Temp = jnp.logical_not(jnp.isnan(Temp))
    Temp = jnp.where(_given_Temp, Temp, 25.0)
    _given_Tnom = jnp.logical_not(jnp.isnan(Tnom))
    Tnom = jnp.where(_given_Tnom, Tnom, 25.0)
    _given_Pg = jnp.logical_not(jnp.isnan(Pg))
    Pg = jnp.where(_given_Pg, Pg, 15.0)
    _given_Ne = jnp.logical_not(jnp.isnan(Ne))
    Ne = jnp.where(_given_Ne, Ne, 1.0)
    i_v2925 = Capmod == 2
    i_v173 = jnp.where(jnp.logical_not(_given_Pg), _given_Ne, False)
    i_v176 = jnp.divide(0.5, jnp.where((Ne == 0.0) | jnp.logical_not(jnp.isfinite(Ne)), 1e-300, Ne))
    i_v50 = Selft != 0
    i_v54 = jnp.where(_given_Temp, Temp + 273.15, _temperature + Trise)
    i_v1118 = Igmod == 0
    i_v68 = jnp.where(_given_Tnom, Tnom + 273.15, 300.15)
    i_v82 = Rth > 0.0
    i_v1451 = Igmod == 1
    i_v156 = P41 * TcVpk
    i_v2006 = 2.0 * P111
    i_v4116 = Rgd > 0.0
    i_v4358 = Ri > 0.0
    i_v3721 = Rc > 0.0
    i_v881 = jnp.where(Idsmod == 0, True, Idsmod == 1)
    i_v3995 = Rcin > 0.0
    i_v150 = P11 * TcVpk
    i_v6914 = jnp.where(i_v50, Rth, False)
    i_v2356 = jnp.divide(Tau, 3.0)
    i_v4488 = Rg > 0.0
    i_v4641 = Lg > 0.0
    i_v4646 = Rs > 0.0
    i_v1084 = Rth > 0.0
    i_v5007 = jnp.where(Rd > 0.0, True, Rd2 > 0.0)
    i_v5304 = Ld > 0.0
    return jnp.array(
        [
            i_v2925,
            i_v173,
            i_v176,
            i_v50,
            i_v54,
            i_v1118,
            i_v68,
            i_v82,
            i_v1451,
            i_v156,
            i_v2006,
            i_v4116,
            i_v4358,
            i_v3721,
            i_v881,
            i_v3995,
            i_v150,
            i_v6914,
            i_v2356,
            i_v4488,
            i_v4641,
            i_v4646,
            i_v1084,
            i_v5007,
            i_v5304,
        ]
    )


@component(
    ports=("d", "g", "s"),
    states=(
        "v_di",
        "v_gi",
        "v_si",
        "v_sii",
        "v_gdi",
        "v_gsi",
        "v_bi",
        "v_rf",
        "v_t",
        "v_xt1",
        "v_xt2",
        "i_un_xt1_xt2_16",
        "i_un_g_gi_17",
        "i_un_si_sii_18",
        "i_un_sii_s_19",
        "i_un_di_d_20",
    ),
)
def Angelov(
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
    Vtr: float = 20.0,
    Vsb2: float = 0.0,
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
    Ij: float = 5e-05,
    Pg: float = float("nan"),
    Ne: float = float("nan"),
    Vjg: float = 0.7,
    Rg: float = 0.05,
    Rd: float = 0.05,
    Rd2: float = 0.0,
    Ri: float = 0.05,
    Rs: float = 0.05,
    Rgd: float = 0.05,
    Ld: float = 0.0,
    Ls: float = 0.0,
    Lg: float = 0.0,
    Tau: float = 0.0,
    Rcmin: float = 1000.0,
    Rc: float = 10000.0,
    Crf: float = 0.0,
    Rcin: float = 100000.0,
    Crfin: float = 0.0,
    Rth: float = 0.001,
    Cth: float = 0.0001,
    Tcipk0: float = -0.002,
    Tcp1: float = -0.002,
    Tccgs0: float = 0.002,
    Tccgd0: float = 0.002,
    Tclsb0: float = 0.0,
    Tcrc: float = 0.0,
    Tccrf: float = 0.0,
    Tcrs: float = 0.003,
    TcRtherm: float = 0.001,
    TcVpk: float = 0.001,
    TcVjg: float = -0.001,
    TcVtr: float = 0.0,
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
    Klf: float = 1e-14,
    Fgr: float = 60000.0,
    Np: float = 0.3,
    Lw: float = 0.1,
    Tnom: float = float("nan"),
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """RF transistor with coupled nonlinear conduction, parasitic RLC, dispersion and heating.

    Use finite-resistance parasitic networks; model parameters and port conventions follow the retained reference.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 2.10.06813, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `angelov.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/angelov/vacode/angelov.va).
    Attribution: Tiburon design Automation, Inc. (MIT notice).
    The retained source and includes are under `tests/data/va_models/angelov/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/angelov/vacode/angelov.va),
    with comparison cases in `tests/test_angelov.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        angelov.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.angelov import Angelov

        jax.config.update("jax_enable_x64", True)
        model = Angelov()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(Angelov.__name__, "ports:", model.ports)
        ```

    """
    _given_Temp = jnp.logical_not(jnp.isnan(Temp))
    Temp = jnp.where(_given_Temp, Temp, 25.0)
    _given_Tnom = jnp.logical_not(jnp.isnan(Tnom))
    Tnom = jnp.where(_given_Tnom, Tnom, 25.0)
    _given_Pg = jnp.logical_not(jnp.isnan(Pg))
    Pg = jnp.where(_given_Pg, Pg, 15.0)
    _given_Ne = jnp.logical_not(jnp.isnan(Ne))
    Ne = jnp.where(_given_Ne, Ne, 1.0)
    i_v2925 = init[0]
    i_v173 = init[1]
    i_v176 = init[2]
    i_v50 = init[3]
    i_v54 = init[4]
    i_v1118 = init[5]
    i_v68 = init[6]
    i_v82 = init[7]
    i_v1451 = init[8]
    i_v156 = init[9]
    i_v2006 = init[10]
    i_v4116 = init[11]
    i_v4358 = init[12]
    i_v3721 = init[13]
    i_v881 = init[14]
    i_v3995 = init[15]
    i_v150 = init[16]
    i_v6914 = init[17]
    i_v2356 = init[18]
    i_v4488 = init[19]
    i_v4641 = init[20]
    i_v4646 = init[21]
    i_v1084 = init[22]
    i_v5007 = init[23]
    i_v5304 = init[24]
    v18205 = -s.i_un_di_d_20
    v20035 = _mfactor * v18205
    v20036 = _mfactor * s.i_un_g_gi_17
    v18198 = -s.i_un_sii_s_19
    v20038 = _mfactor * v18198
    v3512 = s.v_xt2 + 0.0
    v57 = s.v_t < 0.0
    v58 = -s.v_t
    v59 = jnp.where(v57, v58, s.v_t)
    v60 = i_v54 + v59
    v62 = jnp.where(i_v50, v60, i_v54)
    v64 = 8.617332384960955e-05 * v62
    v190 = jnp.divide(i_v176, jnp.where((v64 == 0.0) | jnp.logical_not(jnp.isfinite(v64)), 1e-300, v64))
    v1136 = jnp.where(i_v173, v190, Pg)
    v73 = v62 - i_v68
    v74 = v73 < 0.0
    v75 = -v73
    v76 = jnp.where(v74, v75, v73)
    v79 = v76 > 0.0
    v83 = jnp.where(v79, True, i_v82)
    v162 = TcVjg * v76
    v163 = Vjg + v162
    v1201 = jnp.where(v83, v163, Vjg)
    v1438 = s.v_gdi - s.v_di - v1201
    v1468 = s.v_gdi - s.v_di - v1201
    v1469 = jnp.tanh(v1468)
    v1474 = s.v_gdi - s.v_di - v1201
    v1511 = jnp.where(i_v1451, v1469, v1474)
    v1510 = jnp.where(i_v1118, v1438, v1511)
    v1513 = v1136 * v1510
    v1514 = v1513 > 69.07755278982137
    v1515 = v1513 - 69.07755278982137
    v1516 = 1e30 * v1515
    v1517 = 1e30 + v1516
    v1518 = jnp.exp(jnp.clip(v1513, -709.0, 709.0))
    v1519 = jnp.where(v1514, v1517, v1518)
    v1245 = -1.0 * v1201
    v1246 = jnp.tanh(v1245)
    v1247 = v1136 * v1246
    v1248 = v1247 > 69.07755278982137
    v1249 = v1247 - 69.07755278982137
    v1250 = 1e30 * v1249
    v1251 = 1e30 + v1250
    v1252 = jnp.exp(jnp.clip(v1247, -709.0, 709.0))
    v1253 = jnp.where(v1248, v1251, v1252)
    v1442 = -v1136
    v1444 = v1442 * v1201
    v1445 = v1444 > 69.07755278982137
    v1446 = v1444 - 69.07755278982137
    v1447 = 1e30 * v1446
    v1448 = 1e30 + v1447
    v1449 = jnp.exp(jnp.clip(v1444, -709.0, 709.0))
    v1450 = jnp.where(v1445, v1448, v1449)
    v1497 = jnp.where(i_v1118, v1253, v1450)
    v1523 = v1519 - v1497
    v1524 = Ij * v1523
    v3037 = v1524 + 0.0
    v3265 = v1524 + 0.0
    v6823 = jnp.where(i_v2925, v3037, v3265)
    v18160 = v3512 - v6823
    v18165 = v18160 - 0.0
    v18204 = v18165 + s.i_un_di_d_20
    v20040 = _mfactor * v18204
    v3507 = Cds * (s.v_di - s.v_si)
    v125 = Tccgd0 * v76
    v126 = 1.0 + v125
    v127 = Cgd0 * v126
    v1908 = jnp.where(v83, v127, Cgd0)
    v157 = i_v156 * v76
    v158 = P40 + v157
    v1711 = jnp.where(v83, v158, P40)
    v1773 = P41 * (s.v_gdi - s.v_di)
    v1774 = v1711 + v1773
    v1657 = P111 * (s.v_di - s.v_si)
    v1776 = v1774 - v1657
    v2026 = jnp.cosh(v1776)
    v2027 = jnp.log(jnp.maximum(v2026, 1e-300))
    v2035 = v1776 + v2027
    v2019 = v1711 - v1657
    v2020 = jnp.cosh(v2019)
    v2021 = jnp.log(jnp.maximum(v2020, 1e-300))
    v2031 = v2019 + v2021
    v2036 = v2035 - v2031
    v1675 = P31 * (s.v_di - s.v_si)
    v1676 = P30 - v1675
    v1679 = jnp.tanh(v1676)
    v1680 = 1.0 + v1679
    v1681 = v1680 - P111
    v2039 = v2036 * v1681
    v2040 = jnp.divide(v2039, jnp.where((P41 == 0.0) | jnp.logical_not(jnp.isfinite(P41)), 1e-300, P41))
    v2046 = i_v2006 * (s.v_gdi - s.v_di)
    v2047 = v2040 + v2046
    v2048 = v1908 * v2047
    v2049 = Cgdpi * (s.v_gdi - s.v_di)
    v2050 = v2048 + v2049
    v13389 = jnp.where(i_v2925, v2050, 0.0)
    v3262 = Cgdpi * (s.v_gdi - s.v_di)
    v13395 = jnp.where(i_v2925, 0.0, v3262)
    v13457 = jnp.where(
        Capmod == 1, (Cgdpi + v1908 * (v1681 * (1.0 + jnp.tanh(v1776)) + 2.0 * P111)) * (s.v_gdi - s.v_di), v13389 + v13395
    )
    v18161 = v3507 - v13457
    v3385 = Cgdpe * (signals.g - s.v_di)
    v13468 = v3385 + 0.0
    v18166 = v18161 - v13468
    v139 = Tccrf * v76
    v140 = 1.0 + v139
    v141 = Crf * v140
    v3546 = jnp.where(v83, v141, Crf)
    v3607 = v3546 * (s.v_di - s.v_rf)
    v18168 = v18166 + v3607
    v20042 = _mfactor * v18168
    v4119 = jnp.divide(s.v_gi - s.v_gdi, jnp.where((Rgd == 0.0) | jnp.logical_not(jnp.isfinite(Rgd)), 1e-300, Rgd))
    v11441 = jnp.where(i_v4116, v4119, 0.0)
    v18178 = 0.0 + v11441
    v4361 = jnp.divide(s.v_gi - s.v_gsi, jnp.where((Ri == 0.0) | jnp.logical_not(jnp.isfinite(Ri)), 1e-300, Ri))
    v11829 = jnp.where(i_v4358, v4361, 0.0)
    v18180 = v18178 + v11829
    v18188 = v18180 - s.i_un_g_gi_17
    v18216 = v18188 + 0.0
    v20046 = _mfactor * v18216
    v20047 = _mfactor * v13468
    v18144 = -v3512
    v1346 = s.v_gsi - s.v_si - v1201
    v1461 = s.v_gsi - s.v_si - v1201
    v1462 = jnp.tanh(v1461)
    v1472 = s.v_gsi - s.v_si - v1201
    v1487 = jnp.where(i_v1451, v1462, v1472)
    v1486 = jnp.where(i_v1118, v1346, v1487)
    v1488 = v1136 * v1486
    v1489 = v1488 > 69.07755278982137
    v1490 = v1488 - 69.07755278982137
    v1491 = 1e30 * v1490
    v1492 = 1e30 + v1491
    v1493 = jnp.exp(jnp.clip(v1488, -709.0, 709.0))
    v1494 = jnp.where(v1489, v1492, v1493)
    v1502 = v1494 - v1497
    v1503 = Ij * v1502
    v3149 = v1503 + 0.0
    v3381 = v1503 + 0.0
    v6704 = jnp.where(i_v2925, v3149, v3381)
    v18152 = v18144 - v6704
    v132 = Tcrc * v76
    v133 = 1.0 + v132
    v134 = Rc * v133
    v898 = jnp.where(v83, v134, Rc)
    v104 = Tcp1 * v76
    v105 = 1.0 + v104
    v106 = P1 * v105
    v228 = jnp.where(v83, v106, P1)
    v221 = B2 * (s.v_di - s.v_si)
    v222 = jnp.cosh(v221)
    v233 = v222 * v222
    v234 = jnp.divide(B1, jnp.where((v233 == 0.0) | jnp.logical_not(jnp.isfinite(v233)), 1e-300, v233))
    v235 = 1.0 + v234
    v236 = v228 * v235
    v145 = TcVpk * v76
    v146 = Vpks + v145
    v242 = jnp.where(v83, v146, Vpks)
    v246 = v242 - Dvpks
    v248 = Alphas * (s.v_di - s.v_si)
    v249 = jnp.tanh(v248)
    v250 = Dvpks * v249
    v251 = v246 + v250
    v20 = -(s.v_gi - s.v_di)
    v280 = v20 - Vtr
    v281 = Vsb2 * v280
    v167 = TcVtr * v76
    v168 = Vtr + v167
    v286 = jnp.where(v83, v168, Vtr)
    v289 = v20 - v286
    v290 = v281 * v289
    v291 = v251 - v290
    v320 = s.v_gsi - s.v_si - v291
    v324 = v236 * v320
    v322 = v320 * v320
    v326 = P2 * v322
    v327 = v324 + v326
    v329 = P3 * v320
    v330 = v329 * v322
    v331 = v327 + v330
    v334 = jnp.tanh(v331)
    v335 = 1.0 + v334
    v979 = 1.0 + v335
    v980 = jnp.divide(v898, jnp.where((v979 == 0.0) | jnp.logical_not(jnp.isfinite(v979)), 1e-300, v979))
    v981 = Rcmin + v980
    v340 = v331 > 69.07755278982137
    v341 = v331 - 69.07755278982137
    v342 = 1e30 * v341
    v343 = 1e30 + v342
    v344 = jnp.exp(jnp.clip(v331, -709.0, 709.0))
    v345 = jnp.where(v340, v343, v344)
    v349 = -v331
    v350 = v349 > 69.07755278982137
    v351 = v349 - 69.07755278982137
    v352 = 1e30 * v351
    v353 = 1e30 + v352
    v354 = jnp.exp(jnp.clip(v349, -709.0, 709.0))
    v355 = jnp.where(v350, v353, v354)
    v356 = v345 - v355
    v357 = 0.5 * v356
    v358 = jnp.tanh(v357)
    _psi_nested = v236 * (v320 + P2 * v322 + P3 * v320 * v322)
    v359 = jnp.where(Idsmod >= 2, 1.0 + jnp.tanh(0.5 * (_limexp(_psi_nested) - _limexp(-_psi_nested))), 1.0 + v358)
    v1016 = 1.0 + v359
    v1017 = jnp.divide(v898, jnp.where((v1016 == 0.0) | jnp.logical_not(jnp.isfinite(v1016)), 1e-300, v1016))
    v1018 = Rcmin + v1017
    v3741 = jnp.where(i_v881, v981, v1018)
    v3760 = jnp.divide(s.v_rf - s.v_si, jnp.where((v3741 == 0.0) | jnp.logical_not(jnp.isfinite(v3741)), 1e-300, v3741))
    v10349 = jnp.where(i_v3721, v3760, 0.0)
    v18171 = v18152 - v10349
    v3998 = jnp.divide(s.v_bi - s.v_si, jnp.where((Rcin == 0.0) | jnp.logical_not(jnp.isfinite(Rcin)), 1e-300, Rcin))
    v11045 = jnp.where(i_v3995, v3998, 0.0)
    v18175 = v18171 - v11045
    v18193 = v18175 + s.i_un_si_sii_18
    v18218 = v18193 - 0.0
    v20051 = _mfactor * v18218
    v18145 = -v3507
    v118 = Tccgs0 * v76
    v119 = 1.0 + v118
    v120 = Cgs0 * v119
    v1817 = jnp.where(v83, v120, Cgs0)
    v151 = i_v150 * v76
    v152 = P10 + v151
    v1554 = jnp.where(v83, v152, P10)
    v1616 = P11 * (s.v_gsi - s.v_si)
    v1617 = v1554 + v1616
    v1658 = v1617 + v1657
    v1991 = jnp.cosh(v1658)
    v1993 = jnp.log(jnp.maximum(v1991, 1e-300))
    v2001 = v1658 + v1993
    v1983 = v1554 + v1657
    v1984 = jnp.cosh(v1983)
    v1986 = jnp.log(jnp.maximum(v1984, 1e-300))
    v1997 = v1983 + v1986
    v2002 = v2001 - v1997
    v1666 = P21 * (s.v_di - s.v_si)
    v1667 = P20 + v1666
    v1670 = jnp.tanh(v1667)
    v1671 = 1.0 + v1670
    v1975 = v1671 - P111
    v2003 = v2002 * v1975
    v2004 = jnp.divide(v2003, jnp.where((P11 == 0.0) | jnp.logical_not(jnp.isfinite(P11)), 1e-300, P11))
    v2010 = i_v2006 * (s.v_gsi - s.v_si)
    v2011 = v2004 + v2010
    v2012 = v1817 * v2011
    v2013 = Cgspi * (s.v_gsi - s.v_si)
    v2014 = v2012 + v2013
    v13392 = jnp.where(i_v2925, v2014, 0.0)
    v3378 = Cgspi * (s.v_gsi - s.v_si)
    v13398 = jnp.where(i_v2925, 0.0, v3378)
    v13458 = jnp.where(Capmod == 1, (Cgspi + v1817 * (1.0 + jnp.tanh(v1658)) * v1671) * (s.v_gsi - s.v_si), v13392 + v13398)
    v18153 = v18145 - v13458
    v20053 = _mfactor * v18153
    v18194 = -s.i_un_si_sii_18
    v18197 = v18194 + s.i_un_sii_s_19
    v20057 = _mfactor * v18197
    v18179 = v6823 - v11441
    v20059 = _mfactor * v18179
    v20060 = _mfactor * v13457
    v18181 = v6704 - v11829
    v20062 = _mfactor * v18181
    v3877 = Crfin * (s.v_bi - s.v_gsi)
    v18173 = v13458 - v3877
    v20064 = _mfactor * v18173
    v20065 = _mfactor * v11045
    v20066 = _mfactor * v3877
    v20067 = _mfactor * v10349
    v18169 = -v3607
    v20069 = _mfactor * v18169
    v7172 = -s.v_xt2
    v7216 = v7172 * (s.v_di - s.v_si)
    v7328 = v1503 * (s.v_gsi - s.v_si)
    v7329 = v7216 + v7328
    v7330 = v7329 < 0.0
    v7331 = -v7329
    v7332 = jnp.where(v7330, v7331, v7329)
    v7333 = -1.0 * v7332
    v7337 = 0.0 + v7333
    v7338 = jnp.divide(s.v_t, jnp.where((Rth == 0.0) | jnp.logical_not(jnp.isfinite(Rth)), 1e-300, Rth))
    v7339 = v7337 + v7338
    v7341 = s.v_t * 1e-12
    v13383 = jnp.where(i_v6914, v7339, v7341)
    v20070 = _mfactor * v13383
    v6916 = Cth * s.v_t
    v13417 = jnp.where(i_v6914, v6916, 0.0)
    v20071 = _mfactor * v13417
    v97 = Tcipk0 * v76
    v98 = 1.0 + v97
    v99 = Ipk0 * v98
    v391 = jnp.where(v83, v99, Ipk0)
    v397 = v391 * v335
    v368 = Alphas * v335
    v369 = Alphar + v368
    v377 = v369 * (s.v_di - s.v_si)
    v378 = jnp.tanh(v377)
    v399 = v397 * v378
    v403 = Lambda * (s.v_di - s.v_si)
    v404 = 1.0 + v403
    v111 = Tclsb0 * v76
    v112 = 1.0 + v111
    v113 = Lsb0 * v112
    v414 = jnp.where(v83, v113, Lsb0)
    v434 = v289 > 69.07755278982137
    v435 = v289 - 69.07755278982137
    v436 = 1e30 * v435
    v437 = 1e30 + v436
    v438 = jnp.exp(jnp.clip(v289, -709.0, 709.0))
    v439 = jnp.where(v434, v437, v438)
    v440 = v414 * v439
    v441 = v404 + v440
    v442 = _channel_current(
        Idsmod,
        v399 * v441,
        s.v_di - s.v_si,
        s.v_gi - s.v_di,
        v291,
        v236,
        P2,
        P3,
        v391,
        v335,
        v359,
        Alphar,
        Alphas,
        Lambda,
        Lvg,
        v414,
        v286,
    )
    v2130 = -v442
    v2244 = v2130 + 0.0
    v18130 = v2244 + s.i_un_xt1_xt2_16
    v20073 = _mfactor * v18130
    v2242 = Tau * s.v_xt1
    v20074 = _mfactor * v2242
    v18131 = s.v_xt2 - s.i_un_xt1_xt2_16
    v20076 = _mfactor * v18131
    v18129 = -(s.v_xt1 - s.v_xt2)
    v2358 = i_v2356 * s.i_un_xt1_xt2_16
    v4491 = s.i_un_g_gi_17 * Rg
    v4623 = v4491 + 0.0
    v12072 = jnp.where(i_v4488, v4623, 0.0)
    v18186 = v12072 - (signals.g - s.v_gi)
    v4621 = Lg * s.i_un_g_gi_17
    v13408 = jnp.where(i_v4488, v4621, 0.0)
    v4642 = Lg * s.i_un_g_gi_17
    v13409 = jnp.where(i_v4641, v4642, 0.0)
    v13410 = jnp.where(i_v4488, 0.0, v13409)
    v13464 = v13408 + v13410
    v1085 = jnp.where(v76, True, i_v1084)
    v985 = Rd2 * v335
    v990 = Rs + v985
    v1019 = Rd2 * v359
    v1022 = Rs + v1019
    v1089 = jnp.where(i_v881, v990, v1022)
    v1097 = Tcrs * v76
    v1098 = 1.0 + v1097
    v1099 = v1089 * v1098
    v4674 = jnp.where(v1085, v1099, v1089)
    v4703 = s.i_un_si_sii_18 * v4674
    v12298 = jnp.where(i_v4646, v4703, 0.0)
    v18192 = v12298 - (s.v_si - s.v_sii)
    v18196 = -(s.v_sii - signals.s)
    v4861 = Ls * s.i_un_sii_s_19
    v986 = Rd + v985
    v1020 = Rd + v1019
    v1104 = jnp.where(i_v881, v986, v1020)
    v1109 = v1104 * v1098
    v5039 = jnp.where(v1085, v1109, v1104)
    v5073 = s.i_un_di_d_20 * v5039
    v5223 = v5073 + 0.0
    v12702 = jnp.where(i_v5007, v5223, 0.0)
    v18203 = v12702 - (s.v_di - signals.d)
    v5221 = Ld * s.i_un_di_d_20
    v13412 = jnp.where(i_v5007, v5221, 0.0)
    v5305 = Ld * s.i_un_di_d_20
    v13413 = jnp.where(i_v5304, v5305, 0.0)
    v13414 = jnp.where(i_v5007, 0.0, v13413)
    v13467 = v13412 + v13414
    return (
        {
            "d": v20035,
            "g": v20036,
            "s": v20038,
            "v_di": v20040,
            "v_gi": v20046,
            "v_si": v20051,
            "v_sii": v20057,
            "v_gdi": v20059,
            "v_gsi": v20062,
            "v_bi": v20065,
            "v_rf": v20067,
            "v_t": v20070,
            "v_xt1": v20073,
            "v_xt2": v20076,
            "i_un_xt1_xt2_16": v18129,
            "i_un_g_gi_17": v18186,
            "i_un_si_sii_18": v18192,
            "i_un_sii_s_19": v18196,
            "i_un_di_d_20": v18203,
        },
        {
            "v_di": v20042,
            "v_gi": v20047,
            "v_si": v20053,
            "v_gdi": v20060,
            "v_gsi": v20064,
            "v_bi": v20066,
            "v_rf": v20069,
            "v_t": v20071,
            "v_xt1": v20074,
            "i_un_xt1_xt2_16": v2358,
            "i_un_g_gi_17": v13464,
            "i_un_sii_s_19": v4861,
            "i_un_di_d_20": v13467,
        },
    )


@Angelov.setup
def _Angelov_register_setup(*_a, **_kw):
    return _Angelov_setup(*_a, **_kw)


def _limexp(value):
    threshold = 69.07755278982137  # OpenVAF limexp linearization at log(1e30).
    return jnp.where(value > threshold, 1e30 * (1.0 + value - threshold), jnp.exp(jnp.minimum(value, threshold)))


def _channel_current(
    mode, original, vds, vgd, vpk, p1m, p2, p3, ipk, positive, positive_nested, alphar, alphas, lam, lvg, lsb, vtr
):
    offset = vgd - vpk
    plain = p1m * offset + p2 * offset**2 + p3 * offset**3
    nested = p1m * (offset + p2 * offset**2 + p3 * offset**3)
    negative = 1.0 + jnp.tanh(plain)
    negative_nested = 1.0 + jnp.tanh(0.5 * (_limexp(nested) - _limexp(-nested)))
    branch_positive = ipk * positive * (1.0 + jnp.tanh((alphar + alphas * positive) * vds))
    branch_positive *= 1.0 + (lam + lvg * positive) * vds + lsb * _limexp(vds - vtr)
    branch_negative = ipk * negative * (1.0 - jnp.tanh((alphar + alphas * negative) * vds))
    branch_negative *= 1.0 - (lam + lvg * negative) * vds
    model1 = 0.5 * (branch_positive - branch_negative)
    model2 = ipk * positive_nested * jnp.tanh((alphar + alphas * positive_nested) * vds)
    model2 *= 1.0 + (lam + lvg * positive_nested) * vds + lsb * _limexp(-vgd - vtr)
    branch_positive = ipk * positive_nested * (1.0 + jnp.tanh((alphar + alphas * positive_nested) * vds))
    branch_positive *= 1.0 + (lam + lvg * positive_nested) * vds + lsb * _limexp(vds - vtr)
    branch_negative = ipk * negative_nested * (1.0 - jnp.tanh((alphar + alphas * negative_nested) * vds))
    branch_negative *= 1.0 - (lam + lvg * negative_nested) * vds
    model3 = 0.5 * (branch_positive - branch_negative)
    return jnp.select((mode == 0, mode == 1, mode == 2), (original, model1, model2), default=model3)


# Preserve Verilog-A $param_given semantics at the Python constructor boundary.
from functools import wraps as _wraps
from inspect import signature as _signature

import numpy as np

_Angelov_init = Angelov.__init__
_Angelov_signature = _signature(_Angelov_init)


@_wraps(_Angelov_init)
def _angelov_init(self, *args, **kwargs):
    bound = _Angelov_signature.bind_partial(self, *args, **kwargs)
    # Fixed finite-resistance topology: zero-valued branches need distinct
    # collapsed KCL. Reject them instead of silently leaving singular rows.
    for name in ("Rg", "Rd", "Rs", "Ri", "Rgd", "Rc", "Rcin"):
        value = bound.arguments.get(name, _Angelov_signature.parameters[name].default)
        if not isinstance(value, jax.core.Tracer) and bool(np.any(np.asarray(value) <= 0)):
            raise ValueError(f"Angelov requires {name}>0; zero-resistance collapsed topology is unsupported")
    for name, allowed in (("Selft", (0, 1)), ("Idsmod", (0, 1, 2, 3)), ("Igmod", (0, 1)), ("Capmod", (0, 1, 2))):
        value = bound.arguments.get(name, _Angelov_signature.parameters[name].default)
        if not isinstance(value, jax.core.Tracer) and bool(np.any(~np.isin(np.asarray(value), np.asarray(allowed)))):
            raise ValueError(f"Angelov {name} must be one of {allowed}")
    _Angelov_init(self, *args, **kwargs)


Angelov.__init__ = _angelov_init
