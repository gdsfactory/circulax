"""Native JAX MOSVAR 1.3 current, charge and internal relaxation equations.

Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,
code/MOSVAR/vacode/mosvar.va and its includes. Regenerate with
tests/data/va_models/mosvar/regenerate.py using the bosdi version recorded
there. LEVEL=1000 and SWRES=1 are specialized; all finite parasitic resistances
and internal charge states are retained. Noise is excluded.
No bosdi or OSDI runtime is needed.

(c) Copyright 2008, All Rights Reserved, Arizona State University and
Freescale Semiconductor

(c) Copyright 2015, All Rights Reserved, Silicon Integration Initiative, Inc.


Version: 1.3.0, July 2015

======================================================================================
======================================================================================

Software is distributed as is, completely without expressed or implied warranty or
service support.  ASU, Si2, their members and employees are not liable for the
condition or performance of the software.

ASU and Si2 own the copyright and grants users a perpetual, irrevocable, worldwide,
non-exclusive, royalty-free license with respect to the software as set forth below.

ASU and Si2 hereby disclaims all implied warranties.

ASU and Si2 grant the users the right to modify, copy, and redistribute the
software and documentation, both within the user's organization and externally,
subject to the following restrictions:

1. The users agree not to charge for the code itself but may charge for additions,
extensions, or support.

2. In any product based on the software, the users agree to acknowledge developers
of the software. This acknowledgment shall appear in the product documentation.

3. The users agree to obey all government restrictions governing redistribution or
export of the software.

4. The users agree to reproduce any copyright notice which appears on the software
and documentation on any copy or modification of such made available to others.

======================================================================================
======================================================================================

"""

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, FBT003, PLR0915, PLR0917

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _MOSVAR_setup(
    W: float = 1.0e-06,
    L: float = 1.0e-06,
    NGCON: int = 1,
    DTA: float = 0.0,
    VERSION: float = 1.3,
    SUBVERSION: float = 0.0,
    REVISION: float = 0.0,
    TMIN: float = -100.0,
    TMAX: float = 500.0,
    VMAX: float = 10000.0,
    TR: float = 21.0,
    LMIN: float = 1.0e-08,
    LMAX: float = 9.9e09,
    WMIN: float = 1.0e-08,
    WMAX: float = 9.9e09,
    TYPE: int = -1,
    TYPEP: int = -1,
    TOXO: float = 20.0e-10,
    EPSROXO: float = 3.9,
    TAU: float = 0.1,
    VFBO: float = 0.0,
    NSUBO: float = 3.0e23,
    MNSUBO: float = 1.0,
    DNSUBO: float = 0.0,
    VNSUBO: float = 0.0,
    NSLPO: float = 0.1,
    NPO: float = 1.0e27,
    QMC: float = 1.0,
    DLQ: float = 0.0,
    DWQ: float = 0.0,
    DWR: float = 0.0,
    CFRL: float = 0.0,
    CFRW: float = 0.0,
    RSHG: float = 1.0,
    RPV: float = 0.0,
    REND: float = 1.0e-04,
    RSHS: float = 1000.0,
    UAC: float = 5.0e-02,
    UACRED: float = 0.0,
    STVFB: float = 0.0,
    STRSHG: float = 0.0,
    STRPV: float = 0.0,
    STREND: float = 0.0,
    STRSHS: float = 0.0,
    STUAC: float = 0.0,
    FETA: float = 1.0,
    SWIGATE: int = 0,
    CHIBO: float = 3.1,
    CHIBPO: float = 4.5,
    STIG: float = 2.0,
    LOV: float = 0.0,
    NOVO: float = 5.0e25,
    IGINVLW: float = 0.0,
    IGOVW: float = 0.0,
    GCOO: float = 0.0,
    GC2O: float = 0.375,
    GC3O: float = 0.063,
    IGCHVLW: float = 0.0,
    IGOVHVW: float = 0.0,
    GCOHVO: float = 0.0,
    GC2HVO: float = 0.375,
    GC3HVO: float = 0.063,
    IGMAX: float = 1.0e-05,
    RACNOISE: int = 1,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    i_v101 = 273.15 + (jnp.where((TR > -273.0), TR, -273.0))
    i_v105 = _temperature + DTA
    i_v132 = jnp.divide(i_v101, jnp.where((i_v105 == 0.0) | ~jnp.isfinite(i_v105), 1e-300, i_v105))
    i_v637 = W + DWR
    i_v669 = 12.0 * i_v637
    i_v670 = jnp.divide(
        ((RSHS * (jnp.power(jnp.maximum(i_v132, 0.0), STRSHS))) * L),
        jnp.where((i_v669 == 0.0) | ~jnp.isfinite(i_v669), 1e-300, i_v669),
    )
    i_v732 = jnp.where((i_v670 > 0.001), (jnp.where((i_v670 < 1000.0), i_v670, 1000.0)), 0.001)
    i_v855 = jnp.divide(1.0, jnp.where((i_v732 == 0.0) | ~jnp.isfinite(i_v732), 1e-300, i_v732))
    i_v130 = jnp.divide(i_v105, jnp.where((i_v101 == 0.0) | ~jnp.isfinite(i_v101), 1e-300, i_v101))
    i_v173 = UAC * (jnp.power(jnp.maximum(i_v130, 0.0), STUAC))
    i_v919 = jnp.divide(
        ((12.0 * (jnp.where((i_v173 > 0.001), (jnp.where((i_v173 < 20.0), i_v173, 20.0)), 0.001))) * W),
        jnp.where((L == 0.0) | ~jnp.isfinite(L), 1e-300, L),
    )
    i_v53 = QMC > 0.0
    i_v27 = jnp.divide((3.453e-11 * (jnp.divide(EPSROXO, 3.9))), jnp.where((TOXO == 0.0) | ~jnp.isfinite(TOXO), 1e-300, TOXO))
    i_v61 = (2.3807972 * QMC) * (jnp.power(jnp.maximum(i_v27, 0.0), 0.6666666666666666))
    i_v1677 = jnp.where(i_v53, (jnp.where((TYPE < 0), (1.2514650134837189 * i_v61), i_v61)), 0.0)
    i_v1733 = 0.75 * i_v1677
    i_v257 = 1.179 - (i_v105 * (9.025e-05 + (i_v105 * 3.05e-07)))
    i_v136 = jnp.divide((i_v105 * 1.3806505e-23), 1.6021918e-19)
    i_v330 = 2.0 * i_v136
    i_v120 = i_v105 * i_v105
    i_v291 = jnp.divide((((1.045 + (0.00045 * i_v105)) * ((0.523 + (0.0014 * i_v105)) - (1.48e-06 * i_v120))) * i_v120), 90000.0)
    i_v297 = jnp.sqrt(jnp.maximum((jnp.where((i_v291 > 0.001), i_v291, 0.001)), 1e-300))
    i_v304 = jnp.divide(1.0, ((2.5e25 * i_v297) * (jnp.sqrt(jnp.maximum(i_v297, 1e-300)))))
    i_v9289 = i_v1677 > 0.0
    i_v2506 = NPO < 1e27
    i_v147 = VFBO + ((i_v105 - i_v101) * STVFB)
    i_v5592 = (-TYPE) * TYPEP
    i_v142 = jnp.divide(1.0, jnp.where((i_v136 == 0.0) | ~jnp.isfinite(i_v136), 1e-300, i_v136))
    i_v367 = jnp.sqrt(jnp.maximum(i_v142, 1e-300))
    i_v414 = (
        jnp.divide(
            (jnp.sqrt(jnp.maximum((3.348580862e-29 * NPO), 1e-300))),
            jnp.where((i_v27 == 0.0) | ~jnp.isfinite(i_v27), 1e-300, i_v27),
        )
    ) * i_v367
    i_v422 = 1.0 + (i_v414 * 0.7071067811865475)
    i_v427 = 1e-05 * i_v422
    i_v424 = jnp.divide(1.0, jnp.where((i_v422 == 0.0) | ~jnp.isfinite(i_v422), 1e-300, i_v422))
    i_v429 = (i_v257 + (i_v330 * (jnp.log(jnp.maximum((NPO * i_v304), 1e-300))))) * i_v142
    i_v502 = i_v429 - 460.51701859880916
    i_v2826 = jnp.where(
        (i_v429 < 460.51701859880916),
        (jnp.exp(jnp.clip((-i_v429), -709.0, 709.0))),
        (
            jnp.divide(
                1e-200,
                jnp.where(
                    ((1.0 + (i_v502 * (1.0 + ((0.5 * i_v502) * (1.0 + (i_v502 * 0.3333333333333333)))))) == 0.0)
                    | ~jnp.isfinite(1.0 + (i_v502 * (1.0 + ((0.5 * i_v502) * (1.0 + (i_v502 * 0.3333333333333333)))))),
                    1e-300,
                    (1.0 + (i_v502 * (1.0 + ((0.5 * i_v502) * (1.0 + (i_v502 * 0.3333333333333333)))))),
                ),
            )
        ),
    )
    i_v5933 = 1.0 - i_v2826
    i_v5851 = ((i_v424 * i_v424) * 0.1666666666666667) * 0.7071067811865475
    i_v6036 = -i_v427
    i_v416 = i_v414 * i_v414
    i_v418 = jnp.divide(1.0, i_v416)
    i_v6487 = i_v416 * 0.5
    i_v6492 = i_v416 * 0.25
    i_v6339 = 1.25 + (i_v414 * 0.7324648775608221)
    i_v6340 = jnp.divide(1.0, jnp.where((i_v6339 == 0.0) | ~jnp.isfinite(i_v6339), 1e-300, i_v6339))
    i_v6444 = (((i_v422 * 1.25) * i_v6340) - 1.0) * i_v6340
    i_v6586 = i_v429 + 3.0
    i_v6600 = 0.5 * (i_v6586 - (jnp.sqrt(jnp.maximum(((i_v6586 * i_v6586) + 5.0), 1e-300))))
    i_v6725 = i_v429 - 230.25850929940458
    i_v8271 = 1.0 + (0.37 * (jnp.divide(TOXO, 1e-09)))
    i_v8553 = jnp.sqrt(jnp.maximum(i_v132, 1e-300))
    i_v8623 = jnp.where((TYPE < 0), (0.3333333333333333 * FETA), (0.5 * FETA))
    i_v140 = (100.0 * i_v136) * i_v136
    i_v2509 = (-TYPE) * TYPEP
    i_v2895 = 1.0 - i_v2826
    i_v2801 = ((i_v424 * i_v424) * 0.1666666666666667) * 0.7071067811865475
    i_v2993 = -i_v427
    i_v3439 = i_v416 * 0.5
    i_v3444 = i_v416 * 0.25
    i_v3300 = 1.25 + (i_v414 * 0.7324648775608221)
    i_v3301 = jnp.divide(1.0, jnp.where((i_v3300 == 0.0) | ~jnp.isfinite(i_v3300), 1e-300, i_v3300))
    i_v3396 = (((i_v422 * 1.25) * i_v3301) - 1.0) * i_v3301
    i_v3542 = i_v429 + 3.0
    i_v3556 = 0.5 * (i_v3542 - (jnp.sqrt(jnp.maximum(((i_v3542 * i_v3542) + 5.0), 1e-300))))
    i_v3681 = i_v429 - 230.25850929940458
    i_v638 = 2.0 * i_v637
    i_v639 = jnp.divide(
        (REND * (jnp.power(jnp.maximum(i_v132, 0.0), STREND))), jnp.where((i_v638 == 0.0) | ~jnp.isfinite(i_v638), 1e-300, i_v638)
    )
    i_v708 = jnp.where((i_v639 > 0.001), (jnp.where((i_v639 < 1000.0), i_v639, 1000.0)), 0.001)
    i_v847 = jnp.divide(1.0, jnp.where((i_v708 == 0.0) | ~jnp.isfinite(i_v708), 1e-300, i_v708))
    i_v1002 = SWIGATE != 0
    i_v224 = W + DWQ
    i_v1184 = jnp.power(jnp.maximum(i_v130, 0.0), STIG)
    i_v10754 = jnp.where(i_v1002, (((((2.0 * IGOVW) * LOV) * i_v224) * 1000000000000.0) * i_v1184), 0.0)
    i_v12804 = i_v10754 > 0.0
    i_v10981 = jnp.where(i_v1002, (((((2.0 * IGOVHVW) * LOV) * i_v224) * 1000000000000.0) * i_v1184), 0.0)
    i_v12840 = jnp.where(i_v12804, True, (i_v10981 > 0.0))
    i_v12850 = jnp.where((TYPEP == 1), (i_v10981 > 0.0), False)
    i_v17721 = jnp.where(
        i_v1002,
        (
            (jnp.divide((1.3333333333333333 * (jnp.sqrt(jnp.maximum((2.918995620956536e-49 * CHIBPO), 1e-300)))), 1.05457168e-34))
            * TOXO
        ),
        0.0,
    )
    i_v13404 = GC3HVO < 0.0
    i_v11179 = jnp.where((SWIGATE != 0), (jnp.where((i_v10754 > 0.0), True, (i_v10981 > 0.0))), False)
    i_v10638 = jnp.where(((TYPEP * TYPE) == -1), (TYPEP * i_v257), 0.0)
    i_v476 = (
        jnp.divide(
            (jnp.sqrt(jnp.maximum((3.348580862e-29 * NOVO), 1e-300))),
            jnp.where((i_v27 == 0.0) | ~jnp.isfinite(i_v27), 1e-300, i_v27),
        )
    ) * i_v367
    i_v481 = 1.0 + (i_v476 * 0.7071067811865475)
    i_v483 = 1e-05 * i_v481
    i_v478 = i_v476 * i_v476
    i_v12286 = 0.5 * i_v478
    i_v12513 = 0.25 * i_v478
    i_v1902 = 1.25 + (i_v476 * 0.7324648775608221)
    i_v11976 = jnp.divide(
        ((jnp.divide((i_v481 * 1.25), jnp.where((i_v1902 == 0.0) | ~jnp.isfinite(i_v1902), 1e-300, i_v1902))) - 1.0),
        jnp.where((i_v1902 == 0.0) | ~jnp.isfinite(i_v1902), 1e-300, i_v1902),
    )
    i_v12676 = 0.5 * i_v478
    (i_v13175, i_v13446, i_v13782) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1002, _t, _f),
        (
            (jnp.divide(1.0, jnp.where((CHIBPO == 0.0) | ~jnp.isfinite(CHIBPO), 1e-300, CHIBPO))),
            (
                jnp.where(
                    (GC3HVO < 0.0),
                    (jnp.divide((-0.495 * GC2HVO), jnp.where((GC3HVO == 0.0) | ~jnp.isfinite(GC3HVO), 1e-300, GC3HVO))),
                    0.0,
                )
            ),
            (0.5 * ((TYPE * (i_v257 + (6.0 * i_v136))) + i_v257)),
        ),
        (0.1, 0.0, 0.0),
    )
    i_v14003 = i_v257 - i_v13782
    (i_v12902, i_v18526) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1002, _t, _f),
        (
            (GCOHVO * i_v136),
            (
                (
                    jnp.divide(
                        (1.3333333333333333 * (jnp.sqrt(jnp.maximum((2.918995620956536e-49 * CHIBO), 1e-300)))), 1.05457168e-34
                    )
                )
                * TOXO
            ),
        ),
        (0.0, 0.0),
    )
    i_v15376 = GC3O < 0.0
    (i_v15129, i_v15420, i_v14836) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1002, _t, _f),
        (
            (jnp.divide(1.0, jnp.where((CHIBO == 0.0) | ~jnp.isfinite(CHIBO), 1e-300, CHIBO))),
            (
                jnp.where(
                    (GC3O < 0.0), (jnp.divide((-0.495 * GC2O), jnp.where((GC3O == 0.0) | ~jnp.isfinite(GC3O), 1e-300, GC3O))), 0.0
                )
            ),
            (GCOO * i_v136),
        ),
        (0.1, 0.0, 0.0),
    )
    i_v542 = 2.0 * ((CFRW * W) + (CFRL * L))
    i_v608 = W * L
    i_v609 = jnp.divide(
        (RPV * (jnp.power(jnp.maximum(i_v132, 0.0), STRPV))), jnp.where((i_v608 == 0.0) | ~jnp.isfinite(i_v608), 1e-300, i_v608)
    )
    i_v690 = jnp.where((i_v609 > 0.001), (jnp.where((i_v609 < 100.0), i_v609, 100.0)), 0.001)
    i_v833 = jnp.divide(1.0, jnp.where((i_v690 == 0.0) | ~jnp.isfinite(i_v690), 1e-300, i_v690))
    i_v218 = L + DLQ
    i_v16281 = jnp.where(i_v1002, ((((IGINVLW * i_v224) * i_v218) * 1000000000000.0) * i_v1184), 0.0)
    i_v16556 = i_v16281 > 0.0
    i_v16595 = jnp.where(i_v1002, ((((IGCHVLW * i_v224) * i_v218) * 1000000000000.0) * i_v1184), 0.0)
    i_v16871 = jnp.where(i_v16556, True, (i_v16595 > 0.0))
    i_v17257 = jnp.where((TYPEP == 1), (i_v16595 > 0.0), False)
    i_v17389 = GC3HVO < 0.0
    i_v14105 = jnp.where(
        i_v1002, (0.5 * ((TYPE * (i_v257 + (i_v330 * (jnp.log(jnp.maximum((NSUBO * i_v304), 1e-300)))))) + i_v257)), 0.0
    )
    i_v17625 = i_v257 - i_v14105
    i_v18270 = GC3O < 0.0
    i_v580 = (3.0 + ((NGCON - 1.0) * 9.0)) * L
    i_v581 = jnp.divide(
        ((RSHG * (jnp.power(jnp.maximum(i_v132, 0.0), STRSHG))) * W),
        jnp.where((i_v580 == 0.0) | ~jnp.isfinite(i_v580), 1e-300, i_v580),
    )
    i_v678 = jnp.where((i_v581 > 0.001), (jnp.where((i_v581 < 1000.0), i_v581, 1000.0)), 0.001)
    i_v813 = jnp.divide(1.0, jnp.where((i_v678 == 0.0) | ~jnp.isfinite(i_v678), 1e-300, i_v678))
    return jnp.array(
        [
            i_v132,
            i_v855,
            i_v919,
            i_v53,
            i_v27,
            i_v1677,
            i_v1733,
            i_v257,
            i_v136,
            i_v330,
            i_v304,
            i_v9289,
            i_v2506,
            i_v147,
            i_v5592,
            i_v142,
            i_v367,
            i_v414,
            i_v427,
            i_v424,
            i_v429,
            i_v2826,
            i_v5933,
            i_v5851,
            i_v6036,
            i_v416,
            i_v418,
            i_v6487,
            i_v6492,
            i_v6444,
            i_v6586,
            i_v6600,
            i_v6725,
            i_v8271,
            i_v8553,
            i_v8623,
            i_v140,
            i_v2509,
            i_v2895,
            i_v2801,
            i_v2993,
            i_v3439,
            i_v3444,
            i_v3396,
            i_v3542,
            i_v3556,
            i_v3681,
            i_v847,
            i_v1002,
            i_v224,
            i_v10754,
            i_v12804,
            i_v10981,
            i_v12840,
            i_v12850,
            i_v17721,
            i_v13404,
            i_v11179,
            i_v10638,
            i_v476,
            i_v481,
            i_v483,
            i_v478,
            i_v12286,
            i_v12513,
            i_v11976,
            i_v12676,
            i_v13175,
            i_v13446,
            i_v13782,
            i_v14003,
            i_v12902,
            i_v18526,
            i_v15376,
            i_v15129,
            i_v15420,
            i_v14836,
            i_v542,
            i_v833,
            i_v218,
            i_v16281,
            i_v16556,
            i_v16595,
            i_v16871,
            i_v17257,
            i_v17389,
            i_v14105,
            i_v17625,
            i_v18270,
            i_v813,
        ]
    )


@component(ports=("g", "bi", "b"), states=("v_gii", "v_gi", "v_ci", "v_n", "i_b_rgs", "i_s_rgs"))
def MOSVAR(
    signals: Signals,
    s: States,
    init,
    W: float = 1.0e-06,
    L: float = 1.0e-06,
    NGCON: int = 1,
    DTA: float = 0.0,
    VERSION: float = 1.3,
    SUBVERSION: float = 0.0,
    REVISION: float = 0.0,
    TMIN: float = -100.0,
    TMAX: float = 500.0,
    VMAX: float = 10000.0,
    TR: float = 21.0,
    LMIN: float = 1.0e-08,
    LMAX: float = 9.9e09,
    WMIN: float = 1.0e-08,
    WMAX: float = 9.9e09,
    TYPE: int = -1,
    TYPEP: int = -1,
    TOXO: float = 20.0e-10,
    EPSROXO: float = 3.9,
    TAU: float = 0.1,
    VFBO: float = 0.0,
    NSUBO: float = 3.0e23,
    MNSUBO: float = 1.0,
    DNSUBO: float = 0.0,
    VNSUBO: float = 0.0,
    NSLPO: float = 0.1,
    NPO: float = 1.0e27,
    QMC: float = 1.0,
    DLQ: float = 0.0,
    DWQ: float = 0.0,
    DWR: float = 0.0,
    CFRL: float = 0.0,
    CFRW: float = 0.0,
    RSHG: float = 1.0,
    RPV: float = 0.0,
    REND: float = 1.0e-04,
    RSHS: float = 1000.0,
    UAC: float = 5.0e-02,
    UACRED: float = 0.0,
    STVFB: float = 0.0,
    STRSHG: float = 0.0,
    STRPV: float = 0.0,
    STREND: float = 0.0,
    STRSHS: float = 0.0,
    STUAC: float = 0.0,
    FETA: float = 1.0,
    SWIGATE: int = 0,
    CHIBO: float = 3.1,
    CHIBPO: float = 4.5,
    STIG: float = 2.0,
    LOV: float = 0.0,
    NOVO: float = 5.0e25,
    IGINVLW: float = 0.0,
    IGOVW: float = 0.0,
    GCOO: float = 0.0,
    GC2O: float = 0.375,
    GC3O: float = 0.063,
    IGCHVLW: float = 0.0,
    IGOVHVW: float = 0.0,
    GCOHVO: float = 0.0,
    GC2HVO: float = 0.375,
    GC3HVO: float = 0.063,
    IGMAX: float = 1.0e-05,
    RACNOISE: int = 1,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """MOS varactor including its finite-resistance network.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 1.3, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `mosvar.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/MOSVAR/vacode/mosvar.va).
    Attribution: Arizona State University, Freescale Semiconductor and Silicon Integration Initiative; original
    redistribution terms are retained.
    The retained source and includes are under `tests/data/va_models/mosvar/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/MOSVAR/vacode/mosvar.va),
    with comparison cases in `tests/test_mosvar.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        mosvar.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.mosvar import MOSVAR

        jax.config.update("jax_enable_x64", True)
        model = MOSVAR()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(MOSVAR.__name__, "ports:", model.ports)
        ```

    """
    i_v132 = init[0]
    i_v855 = init[1]
    i_v919 = init[2]
    i_v53 = init[3]
    i_v27 = init[4]
    i_v1677 = init[5]
    i_v1733 = init[6]
    i_v257 = init[7]
    i_v136 = init[8]
    i_v330 = init[9]
    i_v304 = init[10]
    i_v9289 = init[11]
    i_v2506 = init[12]
    i_v147 = init[13]
    i_v5592 = init[14]
    i_v142 = init[15]
    i_v367 = init[16]
    i_v414 = init[17]
    i_v427 = init[18]
    i_v424 = init[19]
    i_v429 = init[20]
    i_v2826 = init[21]
    i_v5933 = init[22]
    i_v5851 = init[23]
    i_v6036 = init[24]
    i_v416 = init[25]
    i_v418 = init[26]
    i_v6487 = init[27]
    i_v6492 = init[28]
    i_v6444 = init[29]
    i_v6586 = init[30]
    i_v6600 = init[31]
    i_v6725 = init[32]
    i_v8271 = init[33]
    i_v8553 = init[34]
    i_v8623 = init[35]
    i_v140 = init[36]
    i_v2509 = init[37]
    i_v2895 = init[38]
    i_v2801 = init[39]
    i_v2993 = init[40]
    i_v3439 = init[41]
    i_v3444 = init[42]
    i_v3396 = init[43]
    i_v3542 = init[44]
    i_v3556 = init[45]
    i_v3681 = init[46]
    i_v847 = init[47]
    i_v1002 = init[48]
    i_v224 = init[49]
    i_v10754 = init[50]
    i_v12804 = init[51]
    i_v10981 = init[52]
    i_v12840 = init[53]
    i_v12850 = init[54]
    i_v17721 = init[55]
    i_v13404 = init[56]
    i_v11179 = init[57]
    i_v10638 = init[58]
    i_v476 = init[59]
    i_v481 = init[60]
    i_v483 = init[61]
    i_v478 = init[62]
    i_v12286 = init[63]
    i_v12513 = init[64]
    i_v11976 = init[65]
    i_v12676 = init[66]
    i_v13175 = init[67]
    i_v13446 = init[68]
    i_v13782 = init[69]
    i_v14003 = init[70]
    i_v12902 = init[71]
    i_v18526 = init[72]
    i_v15376 = init[73]
    i_v15129 = init[74]
    i_v15420 = init[75]
    i_v14836 = init[76]
    i_v542 = init[77]
    i_v833 = init[78]
    i_v218 = init[79]
    i_v16281 = init[80]
    i_v16556 = init[81]
    i_v16595 = init[82]
    i_v16871 = init[83]
    i_v17257 = init[84]
    i_v17389 = init[85]
    i_v14105 = init[86]
    i_v17625 = init[87]
    i_v18270 = init[88]
    i_v813 = init[89]
    v40712 = s.i_b_rgs + s.i_s_rgs
    v44693 = _mfactor * v40712
    v1445 = s.v_gi - s.v_ci - VNSUBO
    v1446 = TYPE * v1445
    v1456 = v1446 * v1446
    v1458 = v1456 + NSLPO
    v1459 = jnp.sqrt(jnp.maximum(v1458, 1e-300))
    v1460 = v1446 + v1459
    v1461 = 0.5 * v1460
    v1462 = DNSUBO * v1461
    v1463 = 1.0 + v1462
    v1466 = v1463 + MNSUBO
    v1467 = v1463 - MNSUBO
    v1469 = v1467 * v1467
    v1471 = v1469 + 1e-06
    v1472 = jnp.sqrt(jnp.maximum(v1471, 1e-300))
    v1473 = v1466 - v1472
    v1474 = 0.5 * v1473
    v1475 = NSUBO * v1474
    v1539 = 3.348580862e-29 * v1475
    v1540 = jnp.sqrt(jnp.maximum(v1539, 1e-300))
    v1633 = jnp.divide(v1540, jnp.where((i_v27 == 0.0) | ~jnp.isfinite(i_v27), 1e-300, i_v27))
    v1636 = v1633 * v1633
    v1533 = v1475 * i_v304
    v1534 = jnp.log(jnp.maximum(v1533, 1e-300))
    v1535 = i_v330 * v1534
    v1536 = i_v257 + v1535
    v1638 = v1636 * v1536
    v1639 = jnp.sqrt(jnp.maximum(v1638, 1e-300))
    v1734 = jnp.power(jnp.maximum(v1639, 0.0), 0.6666666666666666)
    v1735 = i_v1733 * v1734
    v1739 = 1.3333333333333333 * v1735
    v1740 = jnp.divide(v1739, v1639)
    v1741 = 1.0 + v1740
    v1742 = v1633 * v1741
    v1795 = jnp.where(i_v53, v1742, v1633)
    v1979 = s.v_gi - s.v_ci - i_v147
    v1980 = TYPE * v1979
    v5015 = v1980 + s.v_n
    v5090 = v5015 * i_v142
    v5092 = v5090 < 0.0
    v5093 = -v5090
    v5094 = jnp.where(v5092, v5093, v5090)
    v1797 = v1795 * i_v367
    v1803 = v1797 * 0.7071067811865475
    v1804 = 1.0 + v1803
    v1808 = 1e-05 * v1804
    v5138 = v5094 <= v1808
    v5186 = jnp.divide(v5090, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v5190 = v5090 > v1808
    v1799 = v1797 * v1797
    v5347 = 0.5 * v1799
    v5348 = v5090 + v5347
    v5359 = 0.25 * v1799
    v5360 = v5090 + v5359
    v5309 = jnp.divide(v5090, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v5193 = v1804 * 1.25
    v1839 = v1797 * 0.7324648775608221
    v1840 = 1.25 + v1839
    v5304 = jnp.divide(v5193, jnp.where((v1840 == 0.0) | ~jnp.isfinite(v1840), 1e-300, v1840))
    v5305 = v5304 - 1.0
    v5306 = jnp.divide(v5305, jnp.where((v1840 == 0.0) | ~jnp.isfinite(v1840), 1e-300, v1840))
    v5310 = v5306 * v5090
    v5311 = 1.0 + v5310
    v5312 = v5309 * v5311
    v5314 = v5312 < 460.51701859880916
    v5316 = -v5312
    v5317 = jnp.exp(jnp.clip(v5316, -709.0, 709.0))
    v5320 = v5312 - 460.51701859880916
    v5322 = 0.5 * v5320
    v5324 = v5320 * 0.3333333333333333
    v5325 = 1.0 + v5324
    v5326 = v5322 * v5325
    v5327 = 1.0 + v5326
    v5328 = v5320 * v5327
    v5329 = 1.0 + v5328
    v5330 = jnp.divide(1e-200, jnp.where((v5329 == 0.0) | ~jnp.isfinite(v5329), 1e-300, v5329))
    v5331 = jnp.where(v5314, v5317, v5330)
    v5332 = 1.0 - v5331
    v5361 = v5360 - v5332
    v5362 = jnp.sqrt(jnp.maximum(v5361, 1e-300))
    v5363 = v1797 * v5362
    v5364 = v5348 - v5363
    v5395 = v5090 - v5364
    v5403 = v5395 * v5395
    v5404 = v5364 - 1.0
    v5366 = v5364 < 460.51701859880916
    v5368 = -v5364
    v5369 = jnp.exp(jnp.clip(v5368, -709.0, 709.0))
    v5372 = v5364 - 460.51701859880916
    v5374 = 0.5 * v5372
    v5376 = v5372 * 0.3333333333333333
    v5377 = 1.0 + v5376
    v5378 = v5374 * v5377
    v5379 = 1.0 + v5378
    v5380 = v5372 * v5379
    v5381 = 1.0 + v5380
    v5382 = jnp.divide(1e-200, jnp.where((v5381 == 0.0) | ~jnp.isfinite(v5381), 1e-300, v5381))
    v5387 = jnp.where(v5366, v5369, v5382)
    v5405 = v5404 + v5387
    v5406 = v1799 * v5405
    v5407 = v5403 - v5406
    v5413 = 2.0 * v5407
    v5396 = 2.0 * v5395
    v5397 = 1.0 - v5387
    v5398 = v1799 * v5397
    v5399 = v5396 + v5398
    v5409 = v5399 * v5399
    v5388 = v5347 * v5387
    v5389 = 1.0 - v5388
    v5410 = 4.0 * v5389
    v5411 = v5410 * v5407
    v5412 = v5409 - v5411
    v5414 = jnp.sqrt(jnp.maximum(v5412, 1e-300))
    v5415 = v5399 + v5414
    v5416 = jnp.divide(v5413, jnp.where((v5415 == 0.0) | ~jnp.isfinite(v5415), 1e-300, v5415))
    v5418 = v5364 + v5416
    v5420 = -v5090
    v5422 = 1.25 * v5420
    v5424 = jnp.divide(v5422, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v5426 = v5424 + 10.0
    v5427 = v5424 - 6.0
    v5429 = v5427 * v5427
    v5430 = v5429 + 64.0
    v5431 = jnp.sqrt(jnp.maximum(v5430, 1e-300))
    v5432 = v5426 - v5431
    v5433 = 0.5 * v5432
    v5435 = v5420 - v5433
    v5437 = v5435 * v5435
    v5439 = v5433 + 1.0
    v5440 = v1799 * v5439
    v5441 = v5437 + v5440
    v5444 = 2.0 * v5435
    v5445 = v5444 - v1799
    v5451 = v5441 + v5445
    v5460 = v5441 * v5451
    v5447 = jnp.divide(v5441, jnp.where((v1799 == 0.0) | ~jnp.isfinite(v1799), 1e-300, v1799))
    v5448 = jnp.log(jnp.maximum(v5447, 1e-300))
    v5449 = v5448 - v5433
    v5461 = v5460 * v5449
    v5453 = v5451 * v5451
    v5454 = 0.5 * v5445
    v5455 = v5454 * v5445
    v5456 = v5455 - v5441
    v5457 = v5456 * v5449
    v5458 = v5453 + v5457
    v5462 = v5451 * v5449
    v5463 = v5462 * v5449
    v5464 = jnp.divide(v5463, jnp.where((v5458 == 0.0) | ~jnp.isfinite(v5458), 1e-300, v5458))
    v5465 = v5464 * v5445
    v5466 = v5445 * v5445
    v5467 = v5466 * 0.3333333333333333
    v5468 = v5467 - v5441
    v5469 = v5465 * v5468
    v5470 = v5458 + v5469
    v5471 = jnp.divide(v5461, jnp.where((v5470 == 0.0) | ~jnp.isfinite(v5470), 1e-300, v5470))
    v5472 = v5433 + v5471
    v5538 = v5420 - v5472
    v5545 = v5538 * v5538
    v5546 = v5472 + 1.0
    v5474 = v5472 < 0.0
    v5475 = -v5472
    v5476 = jnp.where(v5474, v5475, v5472)
    v5477 = v5476 < 230.25850929940458
    v5482 = jnp.exp(jnp.clip(v5472, -709.0, 709.0))
    v5485 = v5472 < -230.25850929940458
    v5488 = -230.25850929940458 - v5472
    v5491 = 0.5 * v5488
    v5494 = v5488 * 0.3333333333333333
    v5495 = 1.0 + v5494
    v5496 = v5491 * v5495
    v5497 = 1.0 + v5496
    v5498 = v5488 * v5497
    v5499 = 1.0 + v5498
    v5500 = jnp.divide(1e-100, jnp.where((v5499 == 0.0) | ~jnp.isfinite(v5499), 1e-300, v5499))
    v5502 = v5472 - 230.25850929940458
    v5504 = 0.5 * v5502
    v5506 = v5502 * 0.3333333333333333
    v5507 = 1.0 + v5506
    v5508 = v5504 * v5507
    v5509 = 1.0 + v5508
    v5510 = v5502 * v5509
    v5511 = 1.0 + v5510
    v5512 = 1e100 * v5511
    v5524 = jnp.where(v5485, v5500, v5512)
    v5523 = jnp.where(v5477, v5482, v5524)
    v5547 = v5546 - v5523
    v5548 = v1799 * v5547
    v5549 = v5545 + v5548
    v5554 = 2.0 * v5549
    v5539 = 2.0 * v5538
    v5540 = v5523 - 1.0
    v5541 = v1799 * v5540
    v5542 = v5539 + v5541
    v5550 = v5542 * v5542
    v5522 = 0.5 * v1799
    v5525 = v5522 * v5523
    v5526 = 1.0 - v5525
    v5551 = 4.0 * v5526
    v5552 = v5551 * v5549
    v5553 = v5550 - v5552
    v5555 = jnp.sqrt(jnp.maximum(v5553, 1e-300))
    v5556 = v5542 + v5555
    v5557 = jnp.divide(v5554, jnp.where((v5556 == 0.0) | ~jnp.isfinite(v5556), 1e-300, v5556))
    v5558 = v5472 + v5557
    v5559 = -v5558
    v5561 = jnp.where(v5190, v5418, v5559)
    v5560 = jnp.where(v5138, v5186, v5561)
    v5588 = v5560 * i_v136
    v5622 = v1980 - v5588
    v5623 = i_v5592 * v5622
    v5649 = v5623 * i_v142
    v5651 = v5649 < 0.0
    v5652 = -v5649
    v5653 = jnp.where(v5651, v5652, v5649)
    v5752 = v5653 <= i_v427
    v5857 = v5649 * i_v424
    v5934 = v5649 * i_v5933
    v6029 = v5934 * i_v414
    v6030 = v6029 * i_v5851
    v6031 = 1.0 + v6030
    v6032 = v5857 * v6031
    v6037 = v5649 < i_v6036
    v6039 = -v5649
    v6041 = 1.25 * v6039
    v6044 = v6041 * i_v424
    v6047 = v6044 + 10.0
    v6048 = v6044 - 6.0
    v6050 = v6048 * v6048
    v6051 = v6050 + 64.0
    v6052 = jnp.sqrt(jnp.maximum(v6051, 1e-300))
    v6053 = v6047 - v6052
    v6054 = 0.5 * v6053
    v6056 = v6039 - v6054
    v6058 = v6056 * v6056
    v6136 = v6054 + 1.0
    v6137 = i_v416 * v6136
    v6138 = v6058 + v6137
    v6140 = 2.0 * v6056
    v6141 = v6140 - i_v416
    v6247 = v6138 + v6141
    v6256 = v6138 * v6247
    v6143 = -v6054
    v6243 = v6138 * i_v418
    v6244 = jnp.log(jnp.maximum(v6243, 1e-300))
    v6245 = v6143 + v6244
    v6257 = v6256 * v6245
    v6249 = v6247 * v6247
    v6250 = 0.5 * v6141
    v6251 = v6250 * v6141
    v6252 = v6251 - v6138
    v6253 = v6252 * v6245
    v6254 = v6249 + v6253
    v6258 = v6247 * v6245
    v6259 = v6258 * v6245
    v6260 = jnp.divide(v6259, jnp.where((v6254 == 0.0) | ~jnp.isfinite(v6254), 1e-300, v6254))
    v6261 = v6260 * v6141
    v6262 = v6141 * v6141
    v6263 = v6262 * 0.3333333333333333
    v6264 = v6263 - v6138
    v6265 = v6261 * v6264
    v6266 = v6254 + v6265
    v6267 = jnp.divide(v6257, jnp.where((v6266 == 0.0) | ~jnp.isfinite(v6266), 1e-300, v6266))
    v6268 = v6054 + v6267
    v6330 = -v6268
    v6296 = v6039 - v6268
    v6313 = v6296 * v6296
    v6270 = v6268 < 230.25850929940458
    v6272 = jnp.exp(jnp.clip(v6268, -709.0, 709.0))
    v6275 = v6268 - 230.25850929940458
    v6277 = 0.5 * v6275
    v6279 = v6275 * 0.3333333333333333
    v6280 = 1.0 + v6279
    v6281 = v6277 * v6280
    v6282 = 1.0 + v6281
    v6283 = v6275 * v6282
    v6284 = 1.0 + v6283
    v6285 = 1e100 * v6284
    v6286 = jnp.where(v6270, v6272, v6285)
    v6314 = v6286 - v6268
    v6315 = v6314 - 1.0
    v6287 = jnp.divide(1.0, jnp.where((v6286 == 0.0) | ~jnp.isfinite(v6286), 1e-300, v6286))
    v6302 = i_v2826 * v6287
    v6316 = v6315 + v6302
    v6317 = v6268 - 1.0
    v6318 = i_v2826 * v6317
    v6319 = v6316 + v6318
    v6320 = i_v416 * v6319
    v6321 = v6313 - v6320
    v6327 = 2.0 * v6321
    v6303 = 2.0 * v6296
    v6307 = v6286 - 1.0
    v6308 = v6307 - v6302
    v6309 = v6308 + i_v2826
    v6310 = i_v416 * v6309
    v6311 = v6303 + v6310
    v6326 = v6311 * v6311
    v6323 = v6286 + v6302
    v6324 = i_v416 * v6323
    v6325 = 2.0 - v6324
    v6328 = v6327 * v6325
    v6329 = v6326 - v6328
    v6332 = jnp.sqrt(jnp.maximum(v6329, 1e-300))
    v6333 = v6311 + v6332
    v6334 = jnp.divide(v6327, jnp.where((v6333 == 0.0) | ~jnp.isfinite(v6333), 1e-300, v6333))
    v6335 = v6330 - v6334
    v6488 = v5649 + i_v6487
    v6493 = v5649 + i_v6492
    v6448 = v5649 * i_v424
    v6449 = i_v6444 * v5649
    v6450 = 1.0 + v6449
    v6451 = v6448 * v6450
    v6453 = -v6451
    v6455 = v6453 > -230.25850929940458
    v6458 = jnp.exp(jnp.clip(v6453, -709.0, 709.0))
    v6462 = -230.25850929940458 - v6453
    v6466 = 0.5 * v6462
    v6470 = v6462 * 0.3333333333333333
    v6471 = 1.0 + v6470
    v6472 = v6466 * v6471
    v6473 = 1.0 + v6472
    v6474 = v6462 * v6473
    v6475 = 1.0 + v6474
    v6476 = jnp.divide(1e-100, jnp.where((v6475 == 0.0) | ~jnp.isfinite(v6475), 1e-300, v6475))
    v6477 = jnp.where(v6455, v6458, v6476)
    v6478 = 1.0 - v6477
    v6494 = v6493 - v6478
    v6495 = jnp.sqrt(jnp.maximum(v6494, 1e-300))
    v6496 = i_v414 * v6495
    v6497 = v6488 - v6496
    v6588 = v6497 + i_v6586
    v6589 = v6497 - i_v6586
    v6591 = v6589 * v6589
    v6592 = v6591 + 5.0
    v6593 = jnp.sqrt(jnp.maximum(v6592, 1e-300))
    v6594 = v6588 - v6593
    v6595 = 0.5 * v6594
    v6601 = v6595 - i_v6600
    v6646 = i_v429 - v6601
    v6602 = v5649 - v6601
    v6605 = v6602 * v6602
    v6603 = -v6601
    v6604 = jnp.exp(jnp.clip(v6603, -709.0, 709.0))
    v6606 = v6604 + v6601
    v6607 = v6606 - 1.0
    v6612 = v6601 + 1.0
    v6613 = i_v2826 * v6612
    v6614 = v6607 - v6613
    v6615 = i_v416 * v6614
    v6616 = v6605 - v6615
    v6617 = v6616 < 1e-40
    v6618 = jnp.where(v6617, 1e-40, v6616)
    v6647 = jnp.divide(v6618, i_v416)
    v6648 = jnp.log(jnp.maximum(v6647, 1e-300))
    v6649 = v6646 + v6648
    v6652 = v6649 < 0.0
    v6653 = -v6649
    v6654 = jnp.where(v6652, v6653, v6649)
    v6655 = v6654 < 1e-120
    v6632 = 2.0 * v6602
    v6633 = 1.0 - v6604
    v6637 = v6633 - i_v2826
    v6638 = i_v416 * v6637
    v6639 = v6632 + v6638
    v6650 = v6618 + v6639
    v6690 = v6618 * v6650
    v6691 = v6690 * v6649
    v6665 = v6650 * v6650
    v6670 = 0.5 * v6639
    v6671 = v6670 * v6639
    v6626 = i_v6487 * v6604
    v6627 = 1.0 - v6626
    v6680 = v6618 * v6627
    v6681 = v6671 - v6680
    v6686 = v6681 * v6649
    v6687 = v6665 + v6686
    v6692 = v6650 * v6649
    v6693 = v6692 * v6649
    v6694 = jnp.divide(v6693, jnp.where((v6687 == 0.0) | ~jnp.isfinite(v6687), 1e-300, v6687))
    v6695 = v6694 * v6639
    v6696 = v6639 * v6639
    v6697 = v6696 * 0.3333333333333333
    v6699 = v6697 - v6680
    v6700 = v6695 * v6699
    v6701 = v6687 + v6700
    v6702 = jnp.divide(v6691, jnp.where((v6701 == 0.0) | ~jnp.isfinite(v6701), 1e-300, v6701))
    v6703 = v6601 + v6702
    v6704 = jnp.where(v6655, v6601, v6703)
    v6781 = v5649 - v6704
    v6807 = v6781 * v6781
    v6705 = v6704 < 230.25850929940458
    v6707 = jnp.exp(jnp.clip(v6704, -709.0, 709.0))
    v6708 = jnp.divide(1.0, v6707)
    v6726 = v6704 > i_v6725
    v6729 = v6704 - i_v429
    v6730 = jnp.exp(jnp.clip(v6729, -709.0, 709.0))
    v6733 = jnp.divide(i_v2826, v6730)
    v6750 = v6704 - 230.25850929940458
    v6752 = 0.5 * v6750
    v6754 = v6750 * 0.3333333333333333
    v6755 = 1.0 + v6754
    v6756 = v6752 * v6755
    v6757 = 1.0 + v6756
    v6758 = v6750 * v6757
    v6759 = 1.0 + v6758
    v6760 = jnp.divide(1e-100, jnp.where((v6759 == 0.0) | ~jnp.isfinite(v6759), 1e-300, v6759))
    v6796 = jnp.where(v6726, v6733, v6760)
    v6795 = jnp.where(v6705, v6708, v6796)
    v6808 = v6795 + v6704
    v6809 = v6808 - 1.0
    v6716 = i_v2826 * v6707
    v6736 = i_v429 - v6704
    v6737 = v6736 - 230.25850929940458
    v6740 = 0.5 * v6737
    v6743 = v6737 * 0.3333333333333333
    v6744 = 1.0 + v6743
    v6745 = v6740 * v6744
    v6746 = 1.0 + v6745
    v6747 = v6737 * v6746
    v6748 = 1.0 + v6747
    v6749 = jnp.divide(1e-100, jnp.where((v6748 == 0.0) | ~jnp.isfinite(v6748), 1e-300, v6748))
    v6799 = jnp.where(v6726, v6730, v6749)
    v6798 = jnp.where(v6705, v6716, v6799)
    v6810 = v6809 + v6798
    v6811 = v6704 + 1.0
    v6812 = i_v2826 * v6811
    v6813 = v6810 - v6812
    v6814 = i_v416 * v6813
    v6815 = v6807 - v6814
    v6820 = 2.0 * v6815
    v6782 = 2.0 * v6781
    v6797 = 1.0 - v6795
    v6800 = v6797 + v6798
    v6804 = v6800 - i_v2826
    v6805 = i_v416 * v6804
    v6806 = v6782 + v6805
    v6819 = v6806 * v6806
    v6816 = v6795 + v6798
    v6817 = i_v416 * v6816
    v6818 = 2.0 - v6817
    v6821 = v6820 * v6818
    v6822 = v6819 - v6821
    v6824 = jnp.sqrt(jnp.maximum(v6822, 1e-300))
    v6825 = v6806 + v6824
    v6826 = jnp.divide(v6820, jnp.where((v6825 == 0.0) | ~jnp.isfinite(v6825), 1e-300, v6825))
    v6827 = v6704 + v6826
    v6832 = jnp.where(v6037, v6335, v6827)
    v6831 = jnp.where(v5752, v6032, v6832)
    v6833 = i_v5592 * v6831
    v6864 = v6833 * i_v136
    v6897 = v5015 - v6864
    v6898 = jnp.divide(v6897, jnp.where((i_v136 == 0.0) | ~jnp.isfinite(i_v136), 1e-300, i_v136))
    v6899 = v6898 < 0.0
    v6900 = -v6898
    v6901 = jnp.where(v6899, v6900, v6898)
    v6956 = v6901 <= v1808
    v7013 = jnp.divide(v6898, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v7016 = v6898 > v1808
    v7150 = 0.5 * v1799
    v7151 = v6898 + v7150
    v7207 = 0.25 * v1799
    v7208 = v6898 + v7207
    v7080 = jnp.divide(v6898, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v7019 = v1804 * 1.25
    v7075 = jnp.divide(v7019, jnp.where((v1840 == 0.0) | ~jnp.isfinite(v1840), 1e-300, v1840))
    v7076 = v7075 - 1.0
    v7077 = jnp.divide(v7076, jnp.where((v1840 == 0.0) | ~jnp.isfinite(v1840), 1e-300, v1840))
    v7081 = v7077 * v6898
    v7082 = 1.0 + v7081
    v7083 = v7080 * v7082
    v7085 = v7083 < 460.51701859880916
    v7087 = -v7083
    v7088 = jnp.exp(jnp.clip(v7087, -709.0, 709.0))
    v7091 = v7083 - 460.51701859880916
    v7093 = 0.5 * v7091
    v7095 = v7091 * 0.3333333333333333
    v7096 = 1.0 + v7095
    v7097 = v7093 * v7096
    v7098 = 1.0 + v7097
    v7099 = v7091 * v7098
    v7100 = 1.0 + v7099
    v7101 = jnp.divide(1e-200, jnp.where((v7100 == 0.0) | ~jnp.isfinite(v7100), 1e-300, v7100))
    v7102 = jnp.where(v7085, v7088, v7101)
    v7103 = 1.0 - v7102
    v7209 = v7208 - v7103
    v7210 = jnp.sqrt(jnp.maximum(v7209, 1e-300))
    v7211 = v1797 * v7210
    v7212 = v7151 - v7211
    v7243 = v6898 - v7212
    v7251 = v7243 * v7243
    v7252 = v7212 - 1.0
    v7214 = v7212 < 460.51701859880916
    v7216 = -v7212
    v7217 = jnp.exp(jnp.clip(v7216, -709.0, 709.0))
    v7220 = v7212 - 460.51701859880916
    v7222 = 0.5 * v7220
    v7224 = v7220 * 0.3333333333333333
    v7225 = 1.0 + v7224
    v7226 = v7222 * v7225
    v7227 = 1.0 + v7226
    v7228 = v7220 * v7227
    v7229 = 1.0 + v7228
    v7230 = jnp.divide(1e-200, jnp.where((v7229 == 0.0) | ~jnp.isfinite(v7229), 1e-300, v7229))
    v7235 = jnp.where(v7214, v7217, v7230)
    v7253 = v7252 + v7235
    v7254 = v1799 * v7253
    v7255 = v7251 - v7254
    v7261 = 2.0 * v7255
    v7244 = 2.0 * v7243
    v7245 = 1.0 - v7235
    v7246 = v1799 * v7245
    v7247 = v7244 + v7246
    v7257 = v7247 * v7247
    v7236 = v7150 * v7235
    v7237 = 1.0 - v7236
    v7258 = 4.0 * v7237
    v7259 = v7258 * v7255
    v7260 = v7257 - v7259
    v7262 = jnp.sqrt(jnp.maximum(v7260, 1e-300))
    v7263 = v7247 + v7262
    v7264 = jnp.divide(v7261, jnp.where((v7263 == 0.0) | ~jnp.isfinite(v7263), 1e-300, v7263))
    v7266 = v7212 + v7264
    v7268 = -v6898
    v7270 = 1.25 * v7268
    v7272 = jnp.divide(v7270, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v7274 = v7272 + 10.0
    v7275 = v7272 - 6.0
    v7277 = v7275 * v7275
    v7278 = v7277 + 64.0
    v7279 = jnp.sqrt(jnp.maximum(v7278, 1e-300))
    v7280 = v7274 - v7279
    v7281 = 0.5 * v7280
    v7283 = v7268 - v7281
    v7285 = v7283 * v7283
    v7287 = v7281 + 1.0
    v7288 = v1799 * v7287
    v7289 = v7285 + v7288
    v7292 = 2.0 * v7283
    v7293 = v7292 - v1799
    v7299 = v7289 + v7293
    v7308 = v7289 * v7299
    v7295 = jnp.divide(v7289, jnp.where((v1799 == 0.0) | ~jnp.isfinite(v1799), 1e-300, v1799))
    v7296 = jnp.log(jnp.maximum(v7295, 1e-300))
    v7297 = v7296 - v7281
    v7309 = v7308 * v7297
    v7301 = v7299 * v7299
    v7302 = 0.5 * v7293
    v7303 = v7302 * v7293
    v7304 = v7303 - v7289
    v7305 = v7304 * v7297
    v7306 = v7301 + v7305
    v7310 = v7299 * v7297
    v7311 = v7310 * v7297
    v7312 = jnp.divide(v7311, jnp.where((v7306 == 0.0) | ~jnp.isfinite(v7306), 1e-300, v7306))
    v7313 = v7312 * v7293
    v7314 = v7293 * v7293
    v7315 = v7314 * 0.3333333333333333
    v7316 = v7315 - v7289
    v7317 = v7313 * v7316
    v7318 = v7306 + v7317
    v7319 = jnp.divide(v7309, jnp.where((v7318 == 0.0) | ~jnp.isfinite(v7318), 1e-300, v7318))
    v7320 = v7281 + v7319
    v7386 = v7268 - v7320
    v7393 = v7386 * v7386
    v7394 = v7320 + 1.0
    v7322 = v7320 < 0.0
    v7323 = -v7320
    v7324 = jnp.where(v7322, v7323, v7320)
    v7325 = v7324 < 230.25850929940458
    v7330 = jnp.exp(jnp.clip(v7320, -709.0, 709.0))
    v7333 = v7320 < -230.25850929940458
    v7336 = -230.25850929940458 - v7320
    v7339 = 0.5 * v7336
    v7342 = v7336 * 0.3333333333333333
    v7343 = 1.0 + v7342
    v7344 = v7339 * v7343
    v7345 = 1.0 + v7344
    v7346 = v7336 * v7345
    v7347 = 1.0 + v7346
    v7348 = jnp.divide(1e-100, jnp.where((v7347 == 0.0) | ~jnp.isfinite(v7347), 1e-300, v7347))
    v7350 = v7320 - 230.25850929940458
    v7352 = 0.5 * v7350
    v7354 = v7350 * 0.3333333333333333
    v7355 = 1.0 + v7354
    v7356 = v7352 * v7355
    v7357 = 1.0 + v7356
    v7358 = v7350 * v7357
    v7359 = 1.0 + v7358
    v7360 = 1e100 * v7359
    v7372 = jnp.where(v7333, v7348, v7360)
    v7371 = jnp.where(v7325, v7330, v7372)
    v7395 = v7394 - v7371
    v7396 = v1799 * v7395
    v7397 = v7393 + v7396
    v7402 = 2.0 * v7397
    v7387 = 2.0 * v7386
    v7388 = v7371 - 1.0
    v7389 = v1799 * v7388
    v7390 = v7387 + v7389
    v7398 = v7390 * v7390
    v7370 = 0.5 * v1799
    v7373 = v7370 * v7371
    v7374 = 1.0 - v7373
    v7399 = 4.0 * v7374
    v7400 = v7399 * v7397
    v7401 = v7398 - v7400
    v7403 = jnp.sqrt(jnp.maximum(v7401, 1e-300))
    v7404 = v7390 + v7403
    v7405 = jnp.divide(v7402, jnp.where((v7404 == 0.0) | ~jnp.isfinite(v7404), 1e-300, v7404))
    v7406 = v7320 + v7405
    v7407 = -v7406
    v7409 = jnp.where(v7016, v7266, v7407)
    v7408 = jnp.where(v6956, v7013, v7409)
    v7435 = jnp.where(i_v2506, v7408, v5560)
    v7668 = -v1808
    v7669 = v7435 < v7668
    v7437 = v7435 < 230.25850929940458
    v7439 = jnp.exp(jnp.clip(v7435, -709.0, 709.0))
    v7440 = jnp.divide(1.0, v7439)
    v1737 = v1536 + v1735
    v1810 = jnp.where(i_v53, v1737, v1536)
    v1812 = v1810 * i_v142
    v7532 = v1812 - 230.25850929940458
    v7533 = v7435 > v7532
    v1814 = v1812 < 460.51701859880916
    v1816 = -v1812
    v1817 = jnp.exp(jnp.clip(v1816, -709.0, 709.0))
    v1820 = v1812 - 460.51701859880916
    v1822 = 0.5 * v1820
    v1824 = v1820 * 0.3333333333333333
    v1825 = 1.0 + v1824
    v1826 = v1822 * v1825
    v1827 = 1.0 + v1826
    v1828 = v1820 * v1827
    v1829 = 1.0 + v1828
    v1830 = jnp.divide(1e-200, jnp.where((v1829 == 0.0) | ~jnp.isfinite(v1829), 1e-300, v1829))
    v2017 = jnp.where(v1814, v1817, v1830)
    v7536 = v1812 - v7435
    v7537 = jnp.exp(jnp.clip(v7536, -709.0, 709.0))
    v7625 = v2017 * v7537
    v7627 = v7435 - 230.25850929940458
    v7629 = 0.5 * v7627
    v7631 = v7627 * 0.3333333333333333
    v7632 = 1.0 + v7631
    v7633 = v7629 * v7632
    v7634 = 1.0 + v7633
    v7635 = v7627 * v7634
    v7636 = 1.0 + v7635
    v7637 = jnp.divide(1e-100, jnp.where((v7636 == 0.0) | ~jnp.isfinite(v7636), 1e-300, v7636))
    v7672 = jnp.where(v7533, v7625, v7637)
    v7671 = jnp.where(v7437, v7440, v7672)
    v7674 = v7671 + v7435
    v7675 = v7674 - 1.0
    v7676 = jnp.sqrt(jnp.maximum(v7675, 1e-300))
    v7677 = -v7676
    v7679 = v7435 < 0.0
    v7680 = -v7435
    v7681 = jnp.where(v7679, v7680, v7435)
    v7686 = v7681 <= v1808
    v7699 = 0.7071067811865475 * v7435
    v7691 = 0.3333333333333333 * v7435
    v7692 = 0.25 * v7435
    v7693 = 1.0 - v7692
    v7694 = v7691 * v7693
    v7695 = 1.0 - v7694
    v7700 = jnp.sqrt(jnp.maximum(v7695, 1e-300))
    v7701 = v7699 * v7700
    v7703 = v7435 - 1.0
    v7709 = v7703 + v7671
    v7710 = jnp.sqrt(jnp.maximum(v7709, 1e-300))
    v7729 = jnp.where(v7686, v7701, v7710)
    v7728 = jnp.where(v7669, v7677, v7729)
    v7730 = i_v136 * v7728
    v7764 = v7730 * v1797
    v8558 = -v7764
    v8561 = v7764 - v8558
    v8564 = v8561 * v8561
    v1478 = jnp.divide(v1475, 1e23)
    v7974 = 1.0 + v1478
    v7975 = 1.62 * v7974
    v7977 = v7975 * v7974
    v8272 = v7977 * i_v8271
    v8275 = v8272 * i_v8271
    v8552 = v8275 * i_v132
    v8554 = v8552 * i_v8553
    v8555 = v8554 * i_v136
    v8556 = v8555 * i_v136
    v8565 = v8564 + v8556
    v8566 = jnp.sqrt(jnp.maximum(v8565, 1e-300))
    v8568 = 0.5 * v8566
    v8861 = -s.v_n
    v8864 = v8861 - s.v_n
    v8867 = v8864 * v8864
    v8868 = v8867 + v8556
    v8869 = jnp.sqrt(jnp.maximum(v8868, 1e-300))
    v8871 = 0.5 * v8869
    v8872 = i_v8623 * v8871
    v8873 = v8568 + v8872
    v9293 = v8873 * v8873
    v9571 = v9293 + i_v140
    v9574 = jnp.power(jnp.maximum(v9571, 0.0), -0.1666666666666667)
    v9575 = i_v1677 * v9574
    v9576 = 1.0 + v9575
    v9577 = jnp.divide(i_v27, jnp.where((v9576 == 0.0) | ~jnp.isfinite(v9576), 1e-300, v9576))
    v9906 = jnp.where(i_v9289, v9577, i_v27)
    v9908 = v1795 * v9906
    v1985 = v1980 * i_v142
    v1987 = v1985 < 0.0
    v1988 = -v1985
    v1989 = jnp.where(v1987, v1988, v1985)
    v1996 = v1989 <= v1808
    v1806 = jnp.divide(1.0, jnp.where((v1804 == 0.0) | ~jnp.isfinite(v1804), 1e-300, v1804))
    v2013 = v1985 * v1806
    v2019 = 1.0 - v2017
    v2020 = v1985 * v2019
    v2025 = v2020 * v1797
    v2004 = v1806 * v1806
    v2006 = v2004 * 0.1666666666666667
    v2007 = v2006 * 0.7071067811865475
    v2026 = v2025 * v2007
    v2027 = 1.0 + v2026
    v2028 = v2013 * v2027
    v2032 = -v1808
    v2033 = v1985 < v2032
    v2035 = -v1985
    v2037 = 1.25 * v2035
    v2040 = v2037 * v1806
    v2044 = v2040 + 10.0
    v2045 = v2040 - 6.0
    v2047 = v2045 * v2045
    v2049 = v2047 + 64.0
    v2050 = jnp.sqrt(jnp.maximum(v2049, 1e-300))
    v2051 = v2044 - v2050
    v2052 = 0.5 * v2051
    v2054 = v2035 - v2052
    v2056 = v2054 * v2054
    v2065 = v2052 + 1.0
    v2066 = v1799 * v2065
    v2067 = v2056 + v2066
    v2069 = 2.0 * v2054
    v2070 = v2069 - v1799
    v2085 = v2067 + v2070
    v2094 = v2067 * v2085
    v2072 = -v2052
    v1801 = jnp.divide(1.0, jnp.where((v1799 == 0.0) | ~jnp.isfinite(v1799), 1e-300, v1799))
    v2081 = v2067 * v1801
    v2082 = jnp.log(jnp.maximum(v2081, 1e-300))
    v2083 = v2072 + v2082
    v2095 = v2094 * v2083
    v2087 = v2085 * v2085
    v2088 = 0.5 * v2070
    v2089 = v2088 * v2070
    v2090 = v2089 - v2067
    v2091 = v2090 * v2083
    v2092 = v2087 + v2091
    v2096 = v2085 * v2083
    v2097 = v2096 * v2083
    v2098 = jnp.divide(v2097, jnp.where((v2092 == 0.0) | ~jnp.isfinite(v2092), 1e-300, v2092))
    v2099 = v2098 * v2070
    v2100 = v2070 * v2070
    v2101 = v2100 * 0.3333333333333333
    v2102 = v2101 - v2067
    v2103 = v2099 * v2102
    v2104 = v2092 + v2103
    v2105 = jnp.divide(v2095, jnp.where((v2104 == 0.0) | ~jnp.isfinite(v2104), 1e-300, v2104))
    v2106 = v2052 + v2105
    v2170 = -v2106
    v2136 = v2035 - v2106
    v2153 = v2136 * v2136
    v2109 = v2106 < 230.25850929940458
    v2111 = jnp.exp(jnp.clip(v2106, -709.0, 709.0))
    v2115 = v2106 - 230.25850929940458
    v2117 = 0.5 * v2115
    v2119 = v2115 * 0.3333333333333333
    v2120 = 1.0 + v2119
    v2121 = v2117 * v2120
    v2122 = 1.0 + v2121
    v2123 = v2115 * v2122
    v2124 = 1.0 + v2123
    v2125 = 1e100 * v2124
    v2126 = jnp.where(v2109, v2111, v2125)
    v2154 = v2126 - v2106
    v2155 = v2154 - 1.0
    v2127 = jnp.divide(1.0, jnp.where((v2126 == 0.0) | ~jnp.isfinite(v2126), 1e-300, v2126))
    v2142 = v2017 * v2127
    v2156 = v2155 + v2142
    v2157 = v2106 - 1.0
    v2158 = v2017 * v2157
    v2159 = v2156 + v2158
    v2160 = v1799 * v2159
    v2161 = v2153 - v2160
    v2167 = 2.0 * v2161
    v2143 = 2.0 * v2136
    v2147 = v2126 - 1.0
    v2148 = v2147 - v2142
    v2149 = v2148 + v2017
    v2150 = v1799 * v2149
    v2151 = v2143 + v2150
    v2166 = v2151 * v2151
    v2163 = v2126 + v2142
    v2164 = v1799 * v2163
    v2165 = 2.0 - v2164
    v2168 = v2167 * v2165
    v2169 = v2166 - v2168
    v2172 = jnp.sqrt(jnp.maximum(v2169, 1e-300))
    v2173 = v2151 + v2172
    v2174 = jnp.divide(v2167, jnp.where((v2173 == 0.0) | ~jnp.isfinite(v2173), 1e-300, v2173))
    v2175 = v2170 - v2174
    v2238 = v1799 * 0.5
    v2239 = v1985 + v2238
    v2244 = v1799 * 0.25
    v2245 = v1985 + v2244
    v2198 = v1985 * v1806
    v2191 = v1804 * 1.25
    v2181 = jnp.divide(1.0, jnp.where((v1840 == 0.0) | ~jnp.isfinite(v1840), 1e-300, v1840))
    v2192 = v2191 * v2181
    v2193 = v2192 - 1.0
    v2194 = v2193 * v2181
    v2199 = v2194 * v1985
    v2200 = 1.0 + v2199
    v2201 = v2198 * v2200
    v2203 = -v2201
    v2205 = v2203 > -230.25850929940458
    v2208 = jnp.exp(jnp.clip(v2203, -709.0, 709.0))
    v2213 = -230.25850929940458 - v2203
    v2217 = 0.5 * v2213
    v2221 = v2213 * 0.3333333333333333
    v2222 = 1.0 + v2221
    v2223 = v2217 * v2222
    v2224 = 1.0 + v2223
    v2225 = v2213 * v2224
    v2226 = 1.0 + v2225
    v2227 = jnp.divide(1e-100, jnp.where((v2226 == 0.0) | ~jnp.isfinite(v2226), 1e-300, v2226))
    v2228 = jnp.where(v2205, v2208, v2227)
    v2229 = 1.0 - v2228
    v2246 = v2245 - v2229
    v2247 = jnp.sqrt(jnp.maximum(v2246, 1e-300))
    v2248 = v1797 * v2247
    v2249 = v2239 - v2248
    v2260 = v1812 + 3.0
    v2262 = v2249 + v2260
    v2263 = v2249 - v2260
    v2265 = v2263 * v2263
    v2267 = v2265 + 5.0
    v2268 = jnp.sqrt(jnp.maximum(v2267, 1e-300))
    v2269 = v2262 - v2268
    v2270 = 0.5 * v2269
    v2271 = v2260 * v2260
    v2272 = v2271 + 5.0
    v2273 = jnp.sqrt(jnp.maximum(v2272, 1e-300))
    v2274 = v2260 - v2273
    v2275 = 0.5 * v2274
    v2276 = v2270 - v2275
    v2322 = v1812 - v2276
    v2277 = v1985 - v2276
    v2281 = v2277 * v2277
    v2278 = -v2276
    v2279 = jnp.exp(jnp.clip(v2278, -709.0, 709.0))
    v2282 = v2279 + v2276
    v2283 = v2282 - 1.0
    v2288 = v2276 + 1.0
    v2289 = v2017 * v2288
    v2290 = v2283 - v2289
    v2291 = v1799 * v2290
    v2292 = v2281 - v2291
    v2293 = v2292 < 1e-40
    v2294 = jnp.where(v2293, 1e-40, v2292)
    v2323 = jnp.divide(v2294, jnp.where((v1799 == 0.0) | ~jnp.isfinite(v1799), 1e-300, v1799))
    v2324 = jnp.log(jnp.maximum(v2323, 1e-300))
    v2325 = v2322 + v2324
    v2328 = v2325 < 0.0
    v2329 = -v2325
    v2330 = jnp.where(v2328, v2329, v2325)
    v2332 = v2330 < 1e-120
    v2308 = 2.0 * v2277
    v2309 = 1.0 - v2279
    v2313 = v2309 - v2017
    v2314 = v1799 * v2313
    v2315 = v2308 + v2314
    v2326 = v2294 + v2315
    v2367 = v2294 * v2326
    v2368 = v2367 * v2325
    v2342 = v2326 * v2326
    v2347 = 0.5 * v2315
    v2348 = v2347 * v2315
    v2302 = v2238 * v2279
    v2303 = 1.0 - v2302
    v2357 = v2294 * v2303
    v2358 = v2348 - v2357
    v2363 = v2358 * v2325
    v2364 = v2342 + v2363
    v2369 = v2326 * v2325
    v2370 = v2369 * v2325
    v2371 = jnp.divide(v2370, jnp.where((v2364 == 0.0) | ~jnp.isfinite(v2364), 1e-300, v2364))
    v2372 = v2371 * v2315
    v2373 = v2315 * v2315
    v2374 = v2373 * 0.3333333333333333
    v2376 = v2374 - v2357
    v2377 = v2372 * v2376
    v2378 = v2364 + v2377
    v2379 = jnp.divide(v2368, jnp.where((v2378 == 0.0) | ~jnp.isfinite(v2378), 1e-300, v2378))
    v2380 = v2276 + v2379
    v2381 = jnp.where(v2332, v2276, v2380)
    v2458 = v1985 - v2381
    v2484 = v2458 * v2458
    v2382 = v2381 < 230.25850929940458
    v2384 = jnp.exp(jnp.clip(v2381, -709.0, 709.0))
    v2385 = jnp.divide(1.0, v2384)
    v2402 = v1812 - 230.25850929940458
    v2403 = v2381 > v2402
    v2406 = v2381 - v1812
    v2407 = jnp.exp(jnp.clip(v2406, -709.0, 709.0))
    v2410 = jnp.divide(v2017, v2407)
    v2427 = v2381 - 230.25850929940458
    v2429 = 0.5 * v2427
    v2431 = v2427 * 0.3333333333333333
    v2432 = 1.0 + v2431
    v2433 = v2429 * v2432
    v2434 = 1.0 + v2433
    v2435 = v2427 * v2434
    v2436 = 1.0 + v2435
    v2437 = jnp.divide(1e-100, jnp.where((v2436 == 0.0) | ~jnp.isfinite(v2436), 1e-300, v2436))
    v2473 = jnp.where(v2403, v2410, v2437)
    v2472 = jnp.where(v2382, v2385, v2473)
    v2485 = v2472 + v2381
    v2486 = v2485 - 1.0
    v2393 = v2017 * v2384
    v2413 = v1812 - v2381
    v2414 = v2413 - 230.25850929940458
    v2417 = 0.5 * v2414
    v2420 = v2414 * 0.3333333333333333
    v2421 = 1.0 + v2420
    v2422 = v2417 * v2421
    v2423 = 1.0 + v2422
    v2424 = v2414 * v2423
    v2425 = 1.0 + v2424
    v2426 = jnp.divide(1e-100, jnp.where((v2425 == 0.0) | ~jnp.isfinite(v2425), 1e-300, v2425))
    v2476 = jnp.where(v2403, v2407, v2426)
    v2475 = jnp.where(v2382, v2393, v2476)
    v2487 = v2486 + v2475
    v2488 = v2381 + 1.0
    v2489 = v2017 * v2488
    v2490 = v2487 - v2489
    v2491 = v1799 * v2490
    v2492 = v2484 - v2491
    v2497 = 2.0 * v2492
    v2459 = 2.0 * v2458
    v2474 = 1.0 - v2472
    v2477 = v2474 + v2475
    v2481 = v2477 - v2017
    v2482 = v1799 * v2481
    v2483 = v2459 + v2482
    v2496 = v2483 * v2483
    v2493 = v2472 + v2475
    v2494 = v1799 * v2493
    v2495 = 2.0 - v2494
    v2498 = v2497 * v2495
    v2499 = v2496 - v2498
    v2501 = jnp.sqrt(jnp.maximum(v2499, 1e-300))
    v2502 = v2483 + v2501
    v2503 = jnp.divide(v2497, jnp.where((v2502 == 0.0) | ~jnp.isfinite(v2502), 1e-300, v2502))
    v2504 = v2381 + v2503
    v2544 = jnp.where(v2033, v2175, v2504)
    v2543 = jnp.where(v1996, v2028, v2544)
    v2582 = v2543 * i_v136
    v2583 = v1980 - v2582
    v2584 = i_v2509 * v2583
    v2616 = v2584 * i_v142
    v2618 = v2616 < 0.0
    v2619 = -v2616
    v2620 = jnp.where(v2618, v2619, v2616)
    v2709 = v2620 <= i_v427
    v2807 = v2616 * i_v424
    v2896 = v2616 * i_v2895
    v2986 = v2896 * i_v414
    v2987 = v2986 * i_v2801
    v2988 = 1.0 + v2987
    v2989 = v2807 * v2988
    v2994 = v2616 < i_v2993
    v2996 = -v2616
    v2998 = 1.25 * v2996
    v3001 = v2998 * i_v424
    v3004 = v3001 + 10.0
    v3005 = v3001 - 6.0
    v3007 = v3005 * v3005
    v3008 = v3007 + 64.0
    v3009 = jnp.sqrt(jnp.maximum(v3008, 1e-300))
    v3010 = v3004 - v3009
    v3011 = 0.5 * v3010
    v3013 = v2996 - v3011
    v3015 = v3013 * v3013
    v3106 = v3011 + 1.0
    v3107 = i_v416 * v3106
    v3108 = v3015 + v3107
    v3110 = 2.0 * v3013
    v3111 = v3110 - i_v416
    v3208 = v3108 + v3111
    v3217 = v3108 * v3208
    v3113 = -v3011
    v3204 = v3108 * i_v418
    v3205 = jnp.log(jnp.maximum(v3204, 1e-300))
    v3206 = v3113 + v3205
    v3218 = v3217 * v3206
    v3210 = v3208 * v3208
    v3211 = 0.5 * v3111
    v3212 = v3211 * v3111
    v3213 = v3212 - v3108
    v3214 = v3213 * v3206
    v3215 = v3210 + v3214
    v3219 = v3208 * v3206
    v3220 = v3219 * v3206
    v3221 = jnp.divide(v3220, jnp.where((v3215 == 0.0) | ~jnp.isfinite(v3215), 1e-300, v3215))
    v3222 = v3221 * v3111
    v3223 = v3111 * v3111
    v3224 = v3223 * 0.3333333333333333
    v3225 = v3224 - v3108
    v3226 = v3222 * v3225
    v3227 = v3215 + v3226
    v3228 = jnp.divide(v3218, jnp.where((v3227 == 0.0) | ~jnp.isfinite(v3227), 1e-300, v3227))
    v3229 = v3011 + v3228
    v3291 = -v3229
    v3257 = v2996 - v3229
    v3274 = v3257 * v3257
    v3231 = v3229 < 230.25850929940458
    v3233 = jnp.exp(jnp.clip(v3229, -709.0, 709.0))
    v3236 = v3229 - 230.25850929940458
    v3238 = 0.5 * v3236
    v3240 = v3236 * 0.3333333333333333
    v3241 = 1.0 + v3240
    v3242 = v3238 * v3241
    v3243 = 1.0 + v3242
    v3244 = v3236 * v3243
    v3245 = 1.0 + v3244
    v3246 = 1e100 * v3245
    v3247 = jnp.where(v3231, v3233, v3246)
    v3275 = v3247 - v3229
    v3276 = v3275 - 1.0
    v3248 = jnp.divide(1.0, jnp.where((v3247 == 0.0) | ~jnp.isfinite(v3247), 1e-300, v3247))
    v3263 = i_v2826 * v3248
    v3277 = v3276 + v3263
    v3278 = v3229 - 1.0
    v3279 = i_v2826 * v3278
    v3280 = v3277 + v3279
    v3281 = i_v416 * v3280
    v3282 = v3274 - v3281
    v3288 = 2.0 * v3282
    v3264 = 2.0 * v3257
    v3268 = v3247 - 1.0
    v3269 = v3268 - v3263
    v3270 = v3269 + i_v2826
    v3271 = i_v416 * v3270
    v3272 = v3264 + v3271
    v3287 = v3272 * v3272
    v3284 = v3247 + v3263
    v3285 = i_v416 * v3284
    v3286 = 2.0 - v3285
    v3289 = v3288 * v3286
    v3290 = v3287 - v3289
    v3293 = jnp.sqrt(jnp.maximum(v3290, 1e-300))
    v3294 = v3272 + v3293
    v3295 = jnp.divide(v3288, jnp.where((v3294 == 0.0) | ~jnp.isfinite(v3294), 1e-300, v3294))
    v3296 = v3291 - v3295
    v3440 = v2616 + i_v3439
    v3445 = v2616 + i_v3444
    v3400 = v2616 * i_v424
    v3401 = i_v3396 * v2616
    v3402 = 1.0 + v3401
    v3403 = v3400 * v3402
    v3405 = -v3403
    v3407 = v3405 > -230.25850929940458
    v3410 = jnp.exp(jnp.clip(v3405, -709.0, 709.0))
    v3414 = -230.25850929940458 - v3405
    v3418 = 0.5 * v3414
    v3422 = v3414 * 0.3333333333333333
    v3423 = 1.0 + v3422
    v3424 = v3418 * v3423
    v3425 = 1.0 + v3424
    v3426 = v3414 * v3425
    v3427 = 1.0 + v3426
    v3428 = jnp.divide(1e-100, jnp.where((v3427 == 0.0) | ~jnp.isfinite(v3427), 1e-300, v3427))
    v3429 = jnp.where(v3407, v3410, v3428)
    v3430 = 1.0 - v3429
    v3446 = v3445 - v3430
    v3447 = jnp.sqrt(jnp.maximum(v3446, 1e-300))
    v3448 = i_v414 * v3447
    v3449 = v3440 - v3448
    v3544 = v3449 + i_v3542
    v3545 = v3449 - i_v3542
    v3547 = v3545 * v3545
    v3548 = v3547 + 5.0
    v3549 = jnp.sqrt(jnp.maximum(v3548, 1e-300))
    v3550 = v3544 - v3549
    v3551 = 0.5 * v3550
    v3557 = v3551 - i_v3556
    v3602 = i_v429 - v3557
    v3558 = v2616 - v3557
    v3561 = v3558 * v3558
    v3559 = -v3557
    v3560 = jnp.exp(jnp.clip(v3559, -709.0, 709.0))
    v3562 = v3560 + v3557
    v3563 = v3562 - 1.0
    v3568 = v3557 + 1.0
    v3569 = i_v2826 * v3568
    v3570 = v3563 - v3569
    v3571 = i_v416 * v3570
    v3572 = v3561 - v3571
    v3573 = v3572 < 1e-40
    v3574 = jnp.where(v3573, 1e-40, v3572)
    v3603 = jnp.divide(v3574, i_v416)
    v3604 = jnp.log(jnp.maximum(v3603, 1e-300))
    v3605 = v3602 + v3604
    v3608 = v3605 < 0.0
    v3609 = -v3605
    v3610 = jnp.where(v3608, v3609, v3605)
    v3611 = v3610 < 1e-120
    v3588 = 2.0 * v3558
    v3589 = 1.0 - v3560
    v3593 = v3589 - i_v2826
    v3594 = i_v416 * v3593
    v3595 = v3588 + v3594
    v3606 = v3574 + v3595
    v3646 = v3574 * v3606
    v3647 = v3646 * v3605
    v3621 = v3606 * v3606
    v3626 = 0.5 * v3595
    v3627 = v3626 * v3595
    v3582 = i_v3439 * v3560
    v3583 = 1.0 - v3582
    v3636 = v3574 * v3583
    v3637 = v3627 - v3636
    v3642 = v3637 * v3605
    v3643 = v3621 + v3642
    v3648 = v3606 * v3605
    v3649 = v3648 * v3605
    v3650 = jnp.divide(v3649, jnp.where((v3643 == 0.0) | ~jnp.isfinite(v3643), 1e-300, v3643))
    v3651 = v3650 * v3595
    v3652 = v3595 * v3595
    v3653 = v3652 * 0.3333333333333333
    v3655 = v3653 - v3636
    v3656 = v3651 * v3655
    v3657 = v3643 + v3656
    v3658 = jnp.divide(v3647, jnp.where((v3657 == 0.0) | ~jnp.isfinite(v3657), 1e-300, v3657))
    v3659 = v3557 + v3658
    v3660 = jnp.where(v3611, v3557, v3659)
    v3737 = v2616 - v3660
    v3763 = v3737 * v3737
    v3661 = v3660 < 230.25850929940458
    v3663 = jnp.exp(jnp.clip(v3660, -709.0, 709.0))
    v3664 = jnp.divide(1.0, v3663)
    v3682 = v3660 > i_v3681
    v3685 = v3660 - i_v429
    v3686 = jnp.exp(jnp.clip(v3685, -709.0, 709.0))
    v3689 = jnp.divide(i_v2826, v3686)
    v3706 = v3660 - 230.25850929940458
    v3708 = 0.5 * v3706
    v3710 = v3706 * 0.3333333333333333
    v3711 = 1.0 + v3710
    v3712 = v3708 * v3711
    v3713 = 1.0 + v3712
    v3714 = v3706 * v3713
    v3715 = 1.0 + v3714
    v3716 = jnp.divide(1e-100, jnp.where((v3715 == 0.0) | ~jnp.isfinite(v3715), 1e-300, v3715))
    v3752 = jnp.where(v3682, v3689, v3716)
    v3751 = jnp.where(v3661, v3664, v3752)
    v3764 = v3751 + v3660
    v3765 = v3764 - 1.0
    v3672 = i_v2826 * v3663
    v3692 = i_v429 - v3660
    v3693 = v3692 - 230.25850929940458
    v3696 = 0.5 * v3693
    v3699 = v3693 * 0.3333333333333333
    v3700 = 1.0 + v3699
    v3701 = v3696 * v3700
    v3702 = 1.0 + v3701
    v3703 = v3693 * v3702
    v3704 = 1.0 + v3703
    v3705 = jnp.divide(1e-100, jnp.where((v3704 == 0.0) | ~jnp.isfinite(v3704), 1e-300, v3704))
    v3755 = jnp.where(v3682, v3686, v3705)
    v3754 = jnp.where(v3661, v3672, v3755)
    v3766 = v3765 + v3754
    v3767 = v3660 + 1.0
    v3768 = i_v2826 * v3767
    v3769 = v3766 - v3768
    v3770 = i_v416 * v3769
    v3771 = v3763 - v3770
    v3776 = 2.0 * v3771
    v3738 = 2.0 * v3737
    v3753 = 1.0 - v3751
    v3756 = v3753 + v3754
    v3760 = v3756 - i_v2826
    v3761 = i_v416 * v3760
    v3762 = v3738 + v3761
    v3775 = v3762 * v3762
    v3772 = v3751 + v3754
    v3773 = i_v416 * v3772
    v3774 = 2.0 - v3773
    v3777 = v3776 * v3774
    v3778 = v3775 - v3777
    v3780 = jnp.sqrt(jnp.maximum(v3778, 1e-300))
    v3781 = v3762 + v3780
    v3782 = jnp.divide(v3776, jnp.where((v3781 == 0.0) | ~jnp.isfinite(v3781), 1e-300, v3781))
    v3783 = v3660 + v3782
    v3788 = jnp.where(v2994, v3296, v3783)
    v3787 = jnp.where(v2709, v2989, v3788)
    v3789 = i_v2509 * v3787
    v3820 = v3789 * i_v136
    v3852 = v1980 - v3820
    v3853 = jnp.divide(v3852, jnp.where((i_v136 == 0.0) | ~jnp.isfinite(i_v136), 1e-300, i_v136))
    v3854 = v3853 < 0.0
    v3855 = -v3853
    v3856 = jnp.where(v3854, v3855, v3853)
    v3917 = v3856 <= v1808
    v3984 = v3853 * v1806
    v4022 = 1.0 - v2017
    v4023 = v3853 * v4022
    v4080 = v4023 * v1797
    v3976 = v1806 * v1806
    v3977 = v3976 * 0.1666666666666667
    v3978 = v3977 * 0.7071067811865475
    v4081 = v4080 * v3978
    v4082 = 1.0 + v4081
    v4083 = v3984 * v4082
    v4086 = -v1808
    v4087 = v3853 < v4086
    v4089 = -v3853
    v4091 = 1.25 * v4089
    v4094 = v4091 * v1806
    v4097 = v4094 + 10.0
    v4098 = v4094 - 6.0
    v4100 = v4098 * v4098
    v4101 = v4100 + 64.0
    v4102 = jnp.sqrt(jnp.maximum(v4101, 1e-300))
    v4103 = v4097 - v4102
    v4104 = 0.5 * v4103
    v4106 = v4089 - v4104
    v4108 = v4106 * v4106
    v4148 = v4104 + 1.0
    v4149 = v1799 * v4148
    v4150 = v4108 + v4149
    v4152 = 2.0 * v4106
    v4153 = v4152 - v1799
    v4221 = v4150 + v4153
    v4230 = v4150 * v4221
    v4155 = -v4104
    v4217 = v4150 * v1801
    v4218 = jnp.log(jnp.maximum(v4217, 1e-300))
    v4219 = v4155 + v4218
    v4231 = v4230 * v4219
    v4223 = v4221 * v4221
    v4224 = 0.5 * v4153
    v4225 = v4224 * v4153
    v4226 = v4225 - v4150
    v4227 = v4226 * v4219
    v4228 = v4223 + v4227
    v4232 = v4221 * v4219
    v4233 = v4232 * v4219
    v4234 = jnp.divide(v4233, jnp.where((v4228 == 0.0) | ~jnp.isfinite(v4228), 1e-300, v4228))
    v4235 = v4234 * v4153
    v4236 = v4153 * v4153
    v4237 = v4236 * 0.3333333333333333
    v4238 = v4237 - v4150
    v4239 = v4235 * v4238
    v4240 = v4228 + v4239
    v4241 = jnp.divide(v4231, jnp.where((v4240 == 0.0) | ~jnp.isfinite(v4240), 1e-300, v4240))
    v4242 = v4104 + v4241
    v4304 = -v4242
    v4270 = v4089 - v4242
    v4287 = v4270 * v4270
    v4244 = v4242 < 230.25850929940458
    v4246 = jnp.exp(jnp.clip(v4242, -709.0, 709.0))
    v4249 = v4242 - 230.25850929940458
    v4251 = 0.5 * v4249
    v4253 = v4249 * 0.3333333333333333
    v4254 = 1.0 + v4253
    v4255 = v4251 * v4254
    v4256 = 1.0 + v4255
    v4257 = v4249 * v4256
    v4258 = 1.0 + v4257
    v4259 = 1e100 * v4258
    v4260 = jnp.where(v4244, v4246, v4259)
    v4288 = v4260 - v4242
    v4289 = v4288 - 1.0
    v4261 = jnp.divide(1.0, jnp.where((v4260 == 0.0) | ~jnp.isfinite(v4260), 1e-300, v4260))
    v4276 = v2017 * v4261
    v4290 = v4289 + v4276
    v4291 = v4242 - 1.0
    v4292 = v2017 * v4291
    v4293 = v4290 + v4292
    v4294 = v1799 * v4293
    v4295 = v4287 - v4294
    v4301 = 2.0 * v4295
    v4277 = 2.0 * v4270
    v4281 = v4260 - 1.0
    v4282 = v4281 - v4276
    v4283 = v4282 + v2017
    v4284 = v1799 * v4283
    v4285 = v4277 + v4284
    v4300 = v4285 * v4285
    v4297 = v4260 + v4276
    v4298 = v1799 * v4297
    v4299 = 2.0 - v4298
    v4302 = v4301 * v4299
    v4303 = v4300 - v4302
    v4306 = jnp.sqrt(jnp.maximum(v4303, 1e-300))
    v4307 = v4285 + v4306
    v4308 = jnp.divide(v4301, jnp.where((v4307 == 0.0) | ~jnp.isfinite(v4307), 1e-300, v4307))
    v4309 = v4304 - v4308
    v4423 = v1799 * 0.5
    v4424 = v3853 + v4423
    v4428 = v1799 * 0.25
    v4429 = v3853 + v4428
    v4384 = v3853 * v1806
    v4377 = v1804 * 1.25
    v4314 = jnp.divide(1.0, jnp.where((v1840 == 0.0) | ~jnp.isfinite(v1840), 1e-300, v1840))
    v4378 = v4377 * v4314
    v4379 = v4378 - 1.0
    v4380 = v4379 * v4314
    v4385 = v4380 * v3853
    v4386 = 1.0 + v4385
    v4387 = v4384 * v4386
    v4389 = -v4387
    v4391 = v4389 > -230.25850929940458
    v4394 = jnp.exp(jnp.clip(v4389, -709.0, 709.0))
    v4398 = -230.25850929940458 - v4389
    v4402 = 0.5 * v4398
    v4406 = v4398 * 0.3333333333333333
    v4407 = 1.0 + v4406
    v4408 = v4402 * v4407
    v4409 = 1.0 + v4408
    v4410 = v4398 * v4409
    v4411 = 1.0 + v4410
    v4412 = jnp.divide(1e-100, jnp.where((v4411 == 0.0) | ~jnp.isfinite(v4411), 1e-300, v4411))
    v4413 = jnp.where(v4391, v4394, v4412)
    v4414 = 1.0 - v4413
    v4430 = v4429 - v4414
    v4431 = jnp.sqrt(jnp.maximum(v4430, 1e-300))
    v4432 = v1797 * v4431
    v4433 = v4424 - v4432
    v4484 = v1812 + 3.0
    v4486 = v4433 + v4484
    v4487 = v4433 - v4484
    v4489 = v4487 * v4487
    v4490 = v4489 + 5.0
    v4491 = jnp.sqrt(jnp.maximum(v4490, 1e-300))
    v4492 = v4486 - v4491
    v4493 = 0.5 * v4492
    v4494 = v4484 * v4484
    v4495 = v4494 + 5.0
    v4496 = jnp.sqrt(jnp.maximum(v4495, 1e-300))
    v4497 = v4484 - v4496
    v4498 = 0.5 * v4497
    v4499 = v4493 - v4498
    v4544 = v1812 - v4499
    v4500 = v3853 - v4499
    v4503 = v4500 * v4500
    v4501 = -v4499
    v4502 = jnp.exp(jnp.clip(v4501, -709.0, 709.0))
    v4504 = v4502 + v4499
    v4505 = v4504 - 1.0
    v4510 = v4499 + 1.0
    v4511 = v2017 * v4510
    v4512 = v4505 - v4511
    v4513 = v1799 * v4512
    v4514 = v4503 - v4513
    v4515 = v4514 < 1e-40
    v4516 = jnp.where(v4515, 1e-40, v4514)
    v4545 = jnp.divide(v4516, jnp.where((v1799 == 0.0) | ~jnp.isfinite(v1799), 1e-300, v1799))
    v4546 = jnp.log(jnp.maximum(v4545, 1e-300))
    v4547 = v4544 + v4546
    v4550 = v4547 < 0.0
    v4551 = -v4547
    v4552 = jnp.where(v4550, v4551, v4547)
    v4553 = v4552 < 1e-120
    v4530 = 2.0 * v4500
    v4531 = 1.0 - v4502
    v4535 = v4531 - v2017
    v4536 = v1799 * v4535
    v4537 = v4530 + v4536
    v4548 = v4516 + v4537
    v4588 = v4516 * v4548
    v4589 = v4588 * v4547
    v4563 = v4548 * v4548
    v4568 = 0.5 * v4537
    v4569 = v4568 * v4537
    v4524 = v4423 * v4502
    v4525 = 1.0 - v4524
    v4578 = v4516 * v4525
    v4579 = v4569 - v4578
    v4584 = v4579 * v4547
    v4585 = v4563 + v4584
    v4590 = v4548 * v4547
    v4591 = v4590 * v4547
    v4592 = jnp.divide(v4591, jnp.where((v4585 == 0.0) | ~jnp.isfinite(v4585), 1e-300, v4585))
    v4593 = v4592 * v4537
    v4594 = v4537 * v4537
    v4595 = v4594 * 0.3333333333333333
    v4597 = v4595 - v4578
    v4598 = v4593 * v4597
    v4599 = v4585 + v4598
    v4600 = jnp.divide(v4589, jnp.where((v4599 == 0.0) | ~jnp.isfinite(v4599), 1e-300, v4599))
    v4601 = v4499 + v4600
    v4602 = jnp.where(v4553, v4499, v4601)
    v4679 = v3853 - v4602
    v4705 = v4679 * v4679
    v4603 = v4602 < 230.25850929940458
    v4605 = jnp.exp(jnp.clip(v4602, -709.0, 709.0))
    v4606 = jnp.divide(1.0, v4605)
    v4623 = v1812 - 230.25850929940458
    v4624 = v4602 > v4623
    v4627 = v4602 - v1812
    v4628 = jnp.exp(jnp.clip(v4627, -709.0, 709.0))
    v4631 = jnp.divide(v2017, v4628)
    v4648 = v4602 - 230.25850929940458
    v4650 = 0.5 * v4648
    v4652 = v4648 * 0.3333333333333333
    v4653 = 1.0 + v4652
    v4654 = v4650 * v4653
    v4655 = 1.0 + v4654
    v4656 = v4648 * v4655
    v4657 = 1.0 + v4656
    v4658 = jnp.divide(1e-100, jnp.where((v4657 == 0.0) | ~jnp.isfinite(v4657), 1e-300, v4657))
    v4694 = jnp.where(v4624, v4631, v4658)
    v4693 = jnp.where(v4603, v4606, v4694)
    v4706 = v4693 + v4602
    v4707 = v4706 - 1.0
    v4614 = v2017 * v4605
    v4634 = v1812 - v4602
    v4635 = v4634 - 230.25850929940458
    v4638 = 0.5 * v4635
    v4641 = v4635 * 0.3333333333333333
    v4642 = 1.0 + v4641
    v4643 = v4638 * v4642
    v4644 = 1.0 + v4643
    v4645 = v4635 * v4644
    v4646 = 1.0 + v4645
    v4647 = jnp.divide(1e-100, jnp.where((v4646 == 0.0) | ~jnp.isfinite(v4646), 1e-300, v4646))
    v4697 = jnp.where(v4624, v4628, v4647)
    v4696 = jnp.where(v4603, v4614, v4697)
    v4708 = v4707 + v4696
    v4709 = v4602 + 1.0
    v4710 = v2017 * v4709
    v4711 = v4708 - v4710
    v4712 = v1799 * v4711
    v4713 = v4705 - v4712
    v4718 = 2.0 * v4713
    v4680 = 2.0 * v4679
    v4695 = 1.0 - v4693
    v4698 = v4695 + v4696
    v4702 = v4698 - v2017
    v4703 = v1799 * v4702
    v4704 = v4680 + v4703
    v4717 = v4704 * v4704
    v4714 = v4693 + v4696
    v4715 = v1799 * v4714
    v4716 = 2.0 - v4715
    v4719 = v4718 * v4716
    v4720 = v4717 - v4719
    v4722 = jnp.sqrt(jnp.maximum(v4720, 1e-300))
    v4723 = v4704 + v4722
    v4724 = jnp.divide(v4718, jnp.where((v4723 == 0.0) | ~jnp.isfinite(v4723), 1e-300, v4723))
    v4725 = v4602 + v4724
    v4746 = jnp.where(v4087, v4309, v4725)
    v4745 = jnp.where(v3917, v4083, v4746)
    v4743 = jnp.where(i_v2506, v4745, v2543)
    v9684 = v4743 + 10.0
    v9685 = v4743 - 10.0
    v9687 = v9685 * v9685
    v9689 = v9687 + 0.01
    v9690 = jnp.sqrt(jnp.maximum(v9689, 1e-300))
    v9691 = v9684 - v9690
    v9692 = 0.5 * v9691
    v9693 = -1.0 * v9692
    v9694 = jnp.exp(jnp.clip(v9693, -709.0, 709.0))
    v9695 = i_v136 * v9694
    v9697 = jnp.sqrt(jnp.maximum(v9695, 1e-300))
    v9909 = v9908 * v9697
    v10184 = i_v919 * v9909
    v9955 = -v1980
    v9956 = v1980 * v1980
    v9958 = v9956 + 0.04
    v9959 = jnp.sqrt(jnp.maximum(v9958, 1e-300))
    v9960 = v9955 + v9959
    v9961 = 0.5 * v9960
    v10186 = UACRED * v9961
    v10187 = 1.0 + v10186
    v10188 = jnp.divide(v10184, jnp.where((v10187 == 0.0) | ~jnp.isfinite(v10187), 1e-300, v10187))
    v23424 = i_v855 + v10188
    v23425 = (s.v_ci - signals.bi) * v23424
    v28707 = v23425 + 0.0
    v29097 = v28707 + 0.0
    v40705 = -v29097
    v24257 = (signals.bi - signals.b) * i_v847
    v28296 = v24257 + 0.0
    v40708 = v40705 + v28296
    v10640 = s.v_gi - signals.bi - i_v10638
    v10641 = TYPE * v10640
    v10722 = v10641 * i_v142
    v11187 = v10722 < 0.0
    v11188 = -v10722
    v11189 = jnp.where(v11187, v11188, v10722)
    v11464 = v11189 <= i_v483
    v11744 = jnp.divide(v10722, jnp.where((i_v481 == 0.0) | ~jnp.isfinite(i_v481), 1e-300, i_v481))
    v11748 = v10722 > i_v483
    v12287 = v10722 + i_v12286
    v12514 = v10722 + i_v12513
    v11979 = jnp.divide(v10722, jnp.where((i_v481 == 0.0) | ~jnp.isfinite(i_v481), 1e-300, i_v481))
    v11980 = i_v11976 * v10722
    v11981 = 1.0 + v11980
    v11982 = v11979 * v11981
    v11984 = v11982 < 460.51701859880916
    v11986 = -v11982
    v11987 = jnp.exp(jnp.clip(v11986, -709.0, 709.0))
    v11990 = v11982 - 460.51701859880916
    v11992 = 0.5 * v11990
    v11994 = v11990 * 0.3333333333333333
    v11995 = 1.0 + v11994
    v11996 = v11992 * v11995
    v11997 = 1.0 + v11996
    v11998 = v11990 * v11997
    v11999 = 1.0 + v11998
    v12000 = jnp.divide(1e-200, jnp.where((v11999 == 0.0) | ~jnp.isfinite(v11999), 1e-300, v11999))
    v12001 = jnp.where(v11984, v11987, v12000)
    v12002 = 1.0 - v12001
    v12515 = v12514 - v12002
    v12516 = jnp.sqrt(jnp.maximum(v12515, 1e-300))
    v12517 = i_v476 * v12516
    v12518 = v12287 - v12517
    v12549 = v10722 - v12518
    v12557 = v12549 * v12549
    v12558 = v12518 - 1.0
    v12520 = v12518 < 460.51701859880916
    v12522 = -v12518
    v12523 = jnp.exp(jnp.clip(v12522, -709.0, 709.0))
    v12526 = v12518 - 460.51701859880916
    v12528 = 0.5 * v12526
    v12530 = v12526 * 0.3333333333333333
    v12531 = 1.0 + v12530
    v12532 = v12528 * v12531
    v12533 = 1.0 + v12532
    v12534 = v12526 * v12533
    v12535 = 1.0 + v12534
    v12536 = jnp.divide(1e-200, jnp.where((v12535 == 0.0) | ~jnp.isfinite(v12535), 1e-300, v12535))
    v12541 = jnp.where(v12520, v12523, v12536)
    v12559 = v12558 + v12541
    v12560 = i_v478 * v12559
    v12561 = v12557 - v12560
    v12567 = 2.0 * v12561
    v12550 = 2.0 * v12549
    v12551 = 1.0 - v12541
    v12552 = i_v478 * v12551
    v12553 = v12550 + v12552
    v12563 = v12553 * v12553
    v12542 = i_v12286 * v12541
    v12543 = 1.0 - v12542
    v12564 = 4.0 * v12543
    v12565 = v12564 * v12561
    v12566 = v12563 - v12565
    v12568 = jnp.sqrt(jnp.maximum(v12566, 1e-300))
    v12569 = v12553 + v12568
    v12570 = jnp.divide(v12567, jnp.where((v12569 == 0.0) | ~jnp.isfinite(v12569), 1e-300, v12569))
    v12572 = v12518 + v12570
    v12574 = -v10722
    v12576 = 1.25 * v12574
    v12578 = jnp.divide(v12576, jnp.where((i_v481 == 0.0) | ~jnp.isfinite(i_v481), 1e-300, i_v481))
    v12580 = v12578 + 10.0
    v12581 = v12578 - 6.0
    v12583 = v12581 * v12581
    v12584 = v12583 + 64.0
    v12585 = jnp.sqrt(jnp.maximum(v12584, 1e-300))
    v12586 = v12580 - v12585
    v12587 = 0.5 * v12586
    v12589 = v12574 - v12587
    v12591 = v12589 * v12589
    v12593 = v12587 + 1.0
    v12594 = i_v478 * v12593
    v12595 = v12591 + v12594
    v12598 = 2.0 * v12589
    v12599 = v12598 - i_v478
    v12605 = v12595 + v12599
    v12614 = v12595 * v12605
    v12601 = jnp.divide(v12595, i_v478)
    v12602 = jnp.log(jnp.maximum(v12601, 1e-300))
    v12603 = v12602 - v12587
    v12615 = v12614 * v12603
    v12607 = v12605 * v12605
    v12608 = 0.5 * v12599
    v12609 = v12608 * v12599
    v12610 = v12609 - v12595
    v12611 = v12610 * v12603
    v12612 = v12607 + v12611
    v12616 = v12605 * v12603
    v12617 = v12616 * v12603
    v12618 = jnp.divide(v12617, jnp.where((v12612 == 0.0) | ~jnp.isfinite(v12612), 1e-300, v12612))
    v12619 = v12618 * v12599
    v12620 = v12599 * v12599
    v12621 = v12620 * 0.3333333333333333
    v12622 = v12621 - v12595
    v12623 = v12619 * v12622
    v12624 = v12612 + v12623
    v12625 = jnp.divide(v12615, jnp.where((v12624 == 0.0) | ~jnp.isfinite(v12624), 1e-300, v12624))
    v12626 = v12587 + v12625
    v12692 = v12574 - v12626
    v12699 = v12692 * v12692
    v12700 = v12626 + 1.0
    v12628 = v12626 < 0.0
    v12629 = -v12626
    v12630 = jnp.where(v12628, v12629, v12626)
    v12631 = v12630 < 230.25850929940458
    v12636 = jnp.exp(jnp.clip(v12626, -709.0, 709.0))
    v12639 = v12626 < -230.25850929940458
    v12642 = -230.25850929940458 - v12626
    v12645 = 0.5 * v12642
    v12648 = v12642 * 0.3333333333333333
    v12649 = 1.0 + v12648
    v12650 = v12645 * v12649
    v12651 = 1.0 + v12650
    v12652 = v12642 * v12651
    v12653 = 1.0 + v12652
    v12654 = jnp.divide(1e-100, jnp.where((v12653 == 0.0) | ~jnp.isfinite(v12653), 1e-300, v12653))
    v12656 = v12626 - 230.25850929940458
    v12658 = 0.5 * v12656
    v12660 = v12656 * 0.3333333333333333
    v12661 = 1.0 + v12660
    v12662 = v12658 * v12661
    v12663 = 1.0 + v12662
    v12664 = v12656 * v12663
    v12665 = 1.0 + v12664
    v12666 = 1e100 * v12665
    v12678 = jnp.where(v12639, v12654, v12666)
    v12677 = jnp.where(v12631, v12636, v12678)
    v12701 = v12700 - v12677
    v12702 = i_v478 * v12701
    v12703 = v12699 + v12702
    v12708 = 2.0 * v12703
    v12693 = 2.0 * v12692
    v12694 = v12677 - 1.0
    v12695 = i_v478 * v12694
    v12696 = v12693 + v12695
    v12704 = v12696 * v12696
    v12679 = i_v12676 * v12677
    v12680 = 1.0 - v12679
    v12705 = 4.0 * v12680
    v12706 = v12705 * v12703
    v12707 = v12704 - v12706
    v12709 = jnp.sqrt(jnp.maximum(v12707, 1e-300))
    v12710 = v12696 + v12709
    v12711 = jnp.divide(v12708, jnp.where((v12710 == 0.0) | ~jnp.isfinite(v12710), 1e-300, v12710))
    v12712 = v12626 + v12711
    v12713 = -v12712
    v12763 = jnp.where(v11748, v12572, v12713)
    v12762 = jnp.where(v11464, v11744, v12763)
    v12764 = v10722 - v12762
    v12765 = i_v136 * v12764
    v12859 = jnp.where(i_v11179, v12765, 0.0)
    v13133 = v12859 * v12859
    v13134 = v13133 + 1e-06
    v13135 = jnp.sqrt(jnp.maximum(v13134, 1e-300))
    v13401 = v13135 * i_v13175
    v13667 = v13401 + i_v13446
    v13668 = v13401 - i_v13446
    v13670 = v13668 * v13668
    v13671 = v13670 + 1e-06
    v13672 = jnp.sqrt(jnp.maximum(v13671, 1e-300))
    v13673 = v13667 - v13672
    v13674 = 0.5 * v13673
    v14656 = jnp.where(i_v13404, v13674, v13401)
    v14661 = GC3HVO * v14656
    v14662 = GC2HVO + v14661
    v14663 = v14656 * v14662
    v14664 = -1.5 + v14663
    v14665 = i_v17721 * v14664
    v14668 = v14665 > 0.0
    v14670 = 0.5 * v14665
    v14671 = v14665 * 0.3333333333333333
    v14672 = 1.0 + v14671
    v14673 = v14670 * v14672
    v14674 = 1.0 + v14673
    v14675 = v14665 * v14674
    v14676 = 1.0 + v14675
    v14680 = v14665 > -230.25850929940458
    v14682 = jnp.exp(jnp.clip(v14665, -709.0, 709.0))
    v14685 = -230.25850929940458 - v14665
    v14688 = 0.5 * v14685
    v14691 = v14685 * 0.3333333333333333
    v14692 = 1.0 + v14691
    v14693 = v14688 * v14692
    v14694 = 1.0 + v14693
    v14695 = v14685 * v14694
    v14696 = 1.0 + v14695
    v14697 = jnp.divide(1e-100, jnp.where((v14696 == 0.0) | ~jnp.isfinite(v14696), 1e-300, v14696))
    v14720 = jnp.where(v14680, v14682, v14697)
    v14719 = jnp.where(v14668, v14676, v14720)
    v14721 = i_v10981 * v14719
    v14723 = v14721 * TYPE
    v13687 = jnp.where(i_v11179, v12762, 0.0)
    v13691 = TYPE * v13687
    v12862 = TYPE * v12859
    v13122 = v12862 + i_v12902
    v13125 = 0.0 - v13122
    v13127 = v13125 * v13125
    v13128 = v13127 + 0.01
    v13129 = jnp.sqrt(jnp.maximum(v13128, 1e-300))
    v13130 = v13122 + v13129
    v13131 = 0.5 * v13130
    v14008 = i_v14003 + v13131
    v14055 = v14008 * i_v142
    v14056 = v13691 + v14055
    v14057 = -v14056
    v12842 = TYPE * (s.v_gi - signals.bi)
    v14356 = TYPE * v12842
    v14361 = v14356 * i_v142
    v14362 = v14057 + v14361
    v14364 = v14362 < 230.25850929940458
    v14366 = jnp.exp(jnp.clip(v14362, -709.0, 709.0))
    v14367 = 1.0 + v14366
    v14368 = jnp.log(jnp.maximum(v14367, 1e-300))
    v14726 = jnp.where(v14364, v14368, v14362)
    v14334 = v14057 < 230.25850929940458
    v14336 = jnp.exp(jnp.clip(v14057, -709.0, 709.0))
    v14337 = 1.0 + v14336
    v14338 = jnp.log(jnp.maximum(v14337, 1e-300))
    v14735 = jnp.where(v14334, v14338, v14057)
    v14741 = v14726 - v14735
    v14742 = v14723 * v14741
    v16177 = jnp.where(i_v12850, v14742, 0.0)
    v15085 = v12859 * v12859
    v15086 = v15085 + 1e-06
    v15087 = jnp.sqrt(jnp.maximum(v15086, 1e-300))
    v15374 = v15087 * i_v15129
    v15663 = v15374 + i_v15420
    v15664 = v15374 - i_v15420
    v15666 = v15664 * v15664
    v15667 = v15666 + 1e-06
    v15668 = jnp.sqrt(jnp.maximum(v15667, 1e-300))
    v15669 = v15663 - v15668
    v15670 = 0.5 * v15669
    v16115 = jnp.where(i_v15376, v15670, v15374)
    v16120 = GC3O * v16115
    v16121 = GC2O + v16120
    v16122 = v16115 * v16121
    v16123 = -1.5 + v16122
    v16124 = i_v18526 * v16123
    v16125 = v16124 < 0.0
    v16126 = -v16124
    v16127 = jnp.where(v16125, v16126, v16124)
    v16128 = v16127 < 230.25850929940458
    v16133 = jnp.exp(jnp.clip(v16124, -709.0, 709.0))
    v16136 = v16124 < -230.25850929940458
    v16139 = -230.25850929940458 - v16124
    v16142 = 0.5 * v16139
    v16145 = v16139 * 0.3333333333333333
    v16146 = 1.0 + v16145
    v16147 = v16142 * v16146
    v16148 = 1.0 + v16147
    v16149 = v16139 * v16148
    v16150 = 1.0 + v16149
    v16151 = jnp.divide(1e-100, jnp.where((v16150 == 0.0) | ~jnp.isfinite(v16150), 1e-300, v16150))
    v16153 = v16124 - 230.25850929940458
    v16155 = 0.5 * v16153
    v16157 = v16153 * 0.3333333333333333
    v16158 = 1.0 + v16157
    v16159 = v16155 * v16158
    v16160 = 1.0 + v16159
    v16161 = v16153 * v16160
    v16162 = 1.0 + v16161
    v16163 = 1e100 * v16162
    v16214 = jnp.where(v16136, v16151, v16163)
    v16213 = jnp.where(v16128, v16133, v16214)
    v16215 = i_v10754 * v16213
    v16217 = v16215 * TYPE
    v15693 = TYPE * v13687
    v14794 = TYPE * v12859
    v15075 = v14794 + i_v14836
    v15078 = 0.0 - v15075
    v15080 = v15078 * v15078
    v15081 = v15080 + 0.01
    v15082 = jnp.sqrt(jnp.maximum(v15081, 1e-300))
    v15083 = v15075 - v15082
    v15084 = 0.5 * v15083
    v15719 = v15084 - i_v13782
    v15736 = v15719 * i_v142
    v15737 = v15693 + v15736
    v15768 = v15737 < 230.25850929940458
    v15770 = jnp.exp(jnp.clip(v15737, -709.0, 709.0))
    v15771 = 1.0 + v15770
    v15772 = jnp.log(jnp.maximum(v15771, 1e-300))
    v16224 = jnp.where(v15768, v15772, v15737)
    v15797 = TYPE * v12842
    v15802 = v15797 * i_v142
    v15803 = v15737 - v15802
    v15804 = v15803 < 230.25850929940458
    v15806 = jnp.exp(jnp.clip(v15803, -709.0, 709.0))
    v15807 = 1.0 + v15806
    v15808 = jnp.log(jnp.maximum(v15807, 1e-300))
    v16235 = jnp.where(v15804, v15808, v15803)
    v16241 = v16224 - v16235
    v16242 = v16217 * v16241
    v16243 = v16177 + v16242
    v25258 = jnp.where(i_v12804, v16243, v16177)
    v25252 = jnp.where(i_v12840, v25258, 0.0)
    v25246 = jnp.where(i_v1002, v25252, 0.0)
    v25311 = TYPE * v25246
    v27062 = v25311 + 0.0
    v37181 = jnp.where(i_v1002, v27062, v25311)
    v40723 = v40708 - v37181
    v44695 = _mfactor * v40723
    v20777 = i_v542 * (s.v_gii - signals.bi)
    v40726 = -v20777
    v44697 = _mfactor * v40726
    v40709 = -v28296
    v44699 = _mfactor * v40709
    v40695 = -s.i_b_rgs
    v22442 = (s.v_gii - s.v_gi) * i_v833
    v27885 = v22442 + 0.0
    v40698 = v40695 + v27885
    v40713 = v40698 - s.i_s_rgs
    v44701 = _mfactor * v40713
    v44702 = _mfactor * v20777
    v40699 = -v27885
    v4726 = jnp.where(i_v2506, v3853, v1985)
    v17186 = v4726 - v7435
    v17248 = v17186 * i_v136
    v17329 = v17248 * v17248
    v17330 = v17329 + 1e-06
    v17331 = jnp.sqrt(jnp.maximum(v17330, 1e-300))
    v17386 = v17331 * i_v13175
    v17445 = v17386 + i_v13446
    v17446 = v17386 - i_v13446
    v17448 = v17446 * v17446
    v17449 = v17448 + 1e-06
    v17450 = jnp.sqrt(jnp.maximum(v17449, 1e-300))
    v17451 = v17445 - v17450
    v17452 = 0.5 * v17451
    v18008 = jnp.where(i_v17389, v17452, v17386)
    v18013 = GC3HVO * v18008
    v18014 = GC2HVO + v18013
    v18015 = v18008 * v18014
    v18016 = -1.5 + v18015
    v18017 = i_v17721 * v18016
    v18020 = v18017 > 0.0
    v18022 = 0.5 * v18017
    v18023 = v18017 * 0.3333333333333333
    v18024 = 1.0 + v18023
    v18025 = v18022 * v18024
    v18026 = 1.0 + v18025
    v18027 = v18017 * v18026
    v18028 = 1.0 + v18027
    v18032 = v18017 > -230.25850929940458
    v18034 = jnp.exp(jnp.clip(v18017, -709.0, 709.0))
    v18037 = -230.25850929940458 - v18017
    v18040 = 0.5 * v18037
    v18043 = v18037 * 0.3333333333333333
    v18044 = 1.0 + v18043
    v18045 = v18040 * v18044
    v18046 = 1.0 + v18045
    v18047 = v18037 * v18046
    v18048 = 1.0 + v18047
    v18049 = jnp.divide(1e-100, jnp.where((v18048 == 0.0) | ~jnp.isfinite(v18048), 1e-300, v18048))
    v18072 = jnp.where(v18032, v18034, v18049)
    v18071 = jnp.where(v18020, v18028, v18072)
    v18073 = i_v16595 * v18071
    v18075 = v18073 * TYPE
    v17590 = TYPE * v7435
    v17263 = TYPE * v17248
    v17318 = v17263 + i_v12902
    v17321 = 0.0 - v17318
    v17323 = v17321 * v17321
    v17324 = v17323 + 0.01
    v17325 = jnp.sqrt(jnp.maximum(v17324, 1e-300))
    v17326 = v17318 + v17325
    v17327 = 0.5 * v17326
    v17627 = i_v17625 + v17327
    v17629 = v17627 * i_v142
    v17630 = v17590 + v17629
    v17631 = -v17630
    v16873 = TYPE * (s.v_gi - s.v_ci)
    v17655 = TYPE * v16873
    v17660 = v17655 * i_v142
    v17661 = v17631 + v17660
    v17663 = v17661 < 230.25850929940458
    v17665 = jnp.exp(jnp.clip(v17661, -709.0, 709.0))
    v17666 = 1.0 + v17665
    v17667 = jnp.log(jnp.maximum(v17666, 1e-300))
    v18078 = jnp.where(v17663, v17667, v17661)
    v17633 = v17631 < 230.25850929940458
    v17635 = jnp.exp(jnp.clip(v17631, -709.0, 709.0))
    v17636 = 1.0 + v17635
    v17637 = jnp.log(jnp.maximum(v17636, 1e-300))
    v18087 = jnp.where(v17633, v17637, v17631)
    v18093 = v18078 - v18087
    v18094 = v18075 * v18093
    v18894 = jnp.where(i_v17257, v18094, 0.0)
    v18211 = v17248 * v17248
    v18212 = v18211 + 1e-06
    v18213 = jnp.sqrt(jnp.maximum(v18212, 1e-300))
    v18268 = v18213 * i_v15129
    v18326 = v18268 + i_v15420
    v18327 = v18268 - i_v15420
    v18329 = v18327 * v18327
    v18330 = v18329 + 1e-06
    v18331 = jnp.sqrt(jnp.maximum(v18330, 1e-300))
    v18332 = v18326 - v18331
    v18333 = 0.5 * v18332
    v18832 = jnp.where(i_v18270, v18333, v18268)
    v18837 = GC3O * v18832
    v18838 = GC2O + v18837
    v18839 = v18832 * v18838
    v18840 = -1.5 + v18839
    v18841 = i_v18526 * v18840
    v18842 = v18841 < 0.0
    v18843 = -v18841
    v18844 = jnp.where(v18842, v18843, v18841)
    v18845 = v18844 < 230.25850929940458
    v18850 = jnp.exp(jnp.clip(v18841, -709.0, 709.0))
    v18853 = v18841 < -230.25850929940458
    v18856 = -230.25850929940458 - v18841
    v18859 = 0.5 * v18856
    v18862 = v18856 * 0.3333333333333333
    v18863 = 1.0 + v18862
    v18864 = v18859 * v18863
    v18865 = 1.0 + v18864
    v18866 = v18856 * v18865
    v18867 = 1.0 + v18866
    v18868 = jnp.divide(1e-100, jnp.where((v18867 == 0.0) | ~jnp.isfinite(v18867), 1e-300, v18867))
    v18870 = v18841 - 230.25850929940458
    v18872 = 0.5 * v18870
    v18874 = v18870 * 0.3333333333333333
    v18875 = 1.0 + v18874
    v18876 = v18872 * v18875
    v18877 = 1.0 + v18876
    v18878 = v18870 * v18877
    v18879 = 1.0 + v18878
    v18880 = 1e100 * v18879
    v18931 = jnp.where(v18853, v18868, v18880)
    v18930 = jnp.where(v18845, v18850, v18931)
    v18932 = i_v16281 * v18930
    v18934 = v18932 * TYPE
    v18403 = TYPE * v7435
    v18146 = TYPE * v17248
    v18201 = v18146 + i_v14836
    v18204 = 0.0 - v18201
    v18206 = v18204 * v18204
    v18207 = v18206 + 0.01
    v18208 = jnp.sqrt(jnp.maximum(v18207, 1e-300))
    v18209 = v18201 - v18208
    v18210 = 0.5 * v18209
    v18426 = v18210 - i_v14105
    v18428 = v18426 * i_v142
    v18429 = v18403 + v18428
    v18431 = v18429 < 230.25850929940458
    v18433 = jnp.exp(jnp.clip(v18429, -709.0, 709.0))
    v18434 = 1.0 + v18433
    v18435 = jnp.log(jnp.maximum(v18434, 1e-300))
    v18941 = jnp.where(v18431, v18435, v18429)
    v18460 = TYPE * v16873
    v18465 = v18460 * i_v142
    v18466 = v18429 - v18465
    v18467 = v18466 < 230.25850929940458
    v18469 = jnp.exp(jnp.clip(v18466, -709.0, 709.0))
    v18470 = 1.0 + v18469
    v18471 = jnp.log(jnp.maximum(v18470, 1e-300))
    v18952 = jnp.where(v18467, v18471, v18466)
    v18958 = v18941 - v18952
    v18959 = v18934 * v18958
    v18960 = v18894 + v18959
    v24780 = jnp.where(i_v16556, v18960, v18894)
    v24720 = jnp.where(i_v16871, v24780, 0.0)
    v24718 = jnp.where(i_v1002, v24720, 0.0)
    v24783 = TYPE * v24718
    v25794 = v24783 + 0.0
    v27032 = v25794 + 0.0
    v37117 = jnp.where(i_v1002, v27032, v25794)
    v40716 = v40699 + v37117
    v40722 = v40716 + v37181
    v44704 = _mfactor * v40722
    v7434 = v7408 * i_v136
    v19127 = jnp.where(i_v2506, v7434, v5588)
    v19281 = v1980 - v19127
    v19298 = jnp.where(i_v2506, v6864, 0.0)
    v19475 = v19281 - v19298
    v19844 = v19475 * i_v218
    v20213 = v19844 * i_v224
    v20364 = v20213 * v9906
    v20366 = v20364 * TYPE
    v44705 = _mfactor * v20366
    v40718 = v29097 - v37117
    v44707 = _mfactor * v40718
    v40719 = -v20366
    v44709 = _mfactor * v40719
    v26538 = s.v_n + 0.0
    v4739 = v4726 <= 0.0
    v4961 = i_v136 * v1799
    v4824 = v4743 < 1e-05
    v4838 = 0.1666666666666667 * v2017
    v4839 = v4838 * v4743
    v4840 = v4839 * v4743
    v4841 = v4840 * v4743
    v4843 = 1.75 * v4743
    v4844 = 1.0 + v4843
    v4845 = v4841 * v4844
    v4747 = v4743 < 230.25850929940458
    v4749 = jnp.exp(jnp.clip(v4743, -709.0, 709.0))
    v4750 = jnp.divide(1.0, v4749)
    v4759 = jnp.divide(1.0, v4750)
    v4760 = v4759 - v4743
    v4761 = v4760 - 1.0
    v4762 = v2017 * v4761
    v4779 = v1812 - 230.25850929940458
    v4780 = v4743 > v4779
    v4783 = v4743 - v1812
    v4784 = jnp.exp(jnp.clip(v4783, -709.0, 709.0))
    v4788 = v4743 + 1.0
    v4789 = v2017 * v4788
    v4790 = v4784 - v4789
    v4793 = v1812 - v4743
    v4794 = v4793 - 230.25850929940458
    v4797 = 0.5 * v4794
    v4800 = v4794 * 0.3333333333333333
    v4801 = 1.0 + v4800
    v4802 = v4797 * v4801
    v4803 = 1.0 + v4802
    v4804 = v4794 * v4803
    v4805 = 1.0 + v4804
    v4806 = jnp.divide(1e-100, jnp.where((v4805 == 0.0) | ~jnp.isfinite(v4805), 1e-300, v4805))
    v4819 = v4743 + 1.0
    v4820 = v2017 * v4819
    v4821 = v4806 - v4820
    v4899 = jnp.where(v4780, v4790, v4821)
    v4898 = jnp.where(v4747, v4762, v4899)
    v4896 = jnp.where(v4824, v4845, v4898)
    v4962 = v4961 * v4896
    v4826 = 0.5 * v4743
    v4827 = v4826 * v4743
    v4828 = 0.3333333333333333 * v4743
    v4829 = 0.25 * v4743
    v4830 = 1.0 - v4829
    v4831 = v4828 * v4830
    v4832 = 1.0 - v4831
    v4833 = v4827 * v4832
    v4856 = v4743 - 1.0
    v4787 = jnp.divide(v2017, v4784)
    v4807 = v4743 - 230.25850929940458
    v4809 = 0.5 * v4807
    v4811 = v4807 * 0.3333333333333333
    v4812 = 1.0 + v4811
    v4813 = v4809 * v4812
    v4814 = 1.0 + v4813
    v4815 = v4807 * v4814
    v4816 = 1.0 + v4815
    v4817 = jnp.divide(1e-100, jnp.where((v4816 == 0.0) | ~jnp.isfinite(v4816), 1e-300, v4816))
    v4859 = jnp.where(v4780, v4787, v4817)
    v4858 = jnp.where(v4747, v4750, v4859)
    v4860 = v4856 + v4858
    v4895 = jnp.where(v4824, v4833, v4860)
    v4900 = v4895 + v4896
    v4901 = jnp.sqrt(jnp.maximum(v4900, 1e-300))
    v4902 = v1797 * v4901
    v4852 = 0.7071067811865475 * v4743
    v4851 = jnp.sqrt(jnp.maximum(v4832, 1e-300))
    v4853 = v4852 * v4851
    v4861 = jnp.sqrt(jnp.maximum(v4860, 1e-300))
    v4963 = jnp.where(v4824, v4853, v4861)
    v4964 = v1797 * v4963
    v4965 = v4902 + v4964
    v4966 = jnp.divide(v4962, jnp.where((v4965 == 0.0) | ~jnp.isfinite(v4965), 1e-300, v4965))
    v4967 = jnp.where(v4739, 0.0, v4966)
    v40724 = v26538 + v4967
    v44711 = _mfactor * v40724
    v20369 = TAU * s.v_n
    v44712 = _mfactor * v20369
    v21610 = (signals.g - s.v_gii) * i_v813
    v27474 = v21610 + 0.0
    v40693 = v27474 - s.i_b_rgs
    v40711 = -s.i_s_rgs
    return {
        "g": v44693,
        "bi": v44695,
        "b": v44699,
        "v_gii": v44701,
        "v_gi": v44704,
        "v_ci": v44707,
        "v_n": v44711,
        "i_b_rgs": v40693,
        "i_s_rgs": v40711,
    }, {"bi": v44697, "v_gii": v44702, "v_gi": v44705, "v_ci": v44709, "v_n": v44712}


@MOSVAR.setup
def _MOSVAR_register_setup(*_a, **_kw):
    return _MOSVAR_setup(*_a, **_kw)
