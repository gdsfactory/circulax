# ruff: noqa: ANN001, ANN002, ANN003, ANN202, FBT003, PLR0915, PLR0917
# Native equations generated from VA-Models 6099b4e12e6f66ff530f59b8d01fb909094be85a.
# Generator: bosdi 159907ddf29a0225614cf1b7999214401b0b5a2d, unoptimized MIR.
# ======================================================================================
# ======================================================================================
# Filename: diode_cmc.va
# ======================================================================================
#
#
# Silicon Integration Initiative (Si2)
# Compact Model Coalition In-Code Statement
#
# Software is distributed as is, completely without warranty or service support.  NXP
# Semiconductors, Hiroshima University, and Silicon Integration Initiative, Inc. (Si2),
# along with their employees are not liable for the condition or performance of the software.
#
# NXP Semiconductors, Hiroshima University, and Si2 own the copyright and grant users
# a perpetual, irrevocable, worldwide, non-exclusive, royalty-free license with respect to the
# software as set forth below.
#
# NXP Semiconductors, Hiroshima University, and Si2 hereby disclaim all implied
# warranties.
#
# NXP Semiconductors, Hiroshima University, and Si2 grant the users the right to modify,
# copy, and redistribute the software and documentation, both within the user's organization
# and externally, subject to the following restrictions
#
# 1. The users agree not to charge for the NXP Semiconductors, Hiroshima University,
# and Si2 -developed code itself but may charge for additions, extensions, or support.
#
# 2. In any product based on the software, the users agree to acknowledge NXP
# Semiconductors, Hiroshima University, and Si2 that developed the software.  This
# acknowledgment shall appear in the product documentation.
#
# 3. Redistributions to others of source code and documentation must retain the copyright
# notice, disclaimer, and list of conditions.
#
# 4. Redistributions to others in binary form must reproduce the copyright notice,
# disclaimer, and list of conditions in the documentation and/or other materials provided
# with the distribution.
#
# CMC In-Code Statement Revision 1.1 6/9/2016
#
#
#
# The diode_cmc is based on Juncap2 200.3.3 (test) from NXP Semiconductors.
#
# Portions commented as diode_cmc code: additional models since 2009.
#
# Portions commented as diode_cmc recovery and high-injection equations code: recovery model by Hiroshima University.
#
# All other parts are taken from Juncap2 200.3.3. Copyright 2008 NXP Semiconductors, All Rights Reserved.
# ======================================================================================
# ======================================================================================
#
#
#
"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _DIODE_CMC_setup(
    TYPE: float = 1.0,
    DTA: float = 0.0,
    IMAX: float = 1000.0,
    TRJ: float = 21.0,
    CJORBOT: float = 0.001,
    CJORSTI: float = 1e-09,
    CJORGAT: float = 1e-09,
    VBIRBOT: float = 1.0,
    VBIRSTI: float = 1.0,
    VBIRGAT: float = 1.0,
    PBOT: float = 0.5,
    PSTI: float = 0.5,
    PGAT: float = 0.5,
    PHIGBOT: float = 1.16,
    PHIGSTI: float = 1.16,
    PHIGGAT: float = 1.16,
    IDSATRBOT: float = 1e-12,
    IDSATRSTI: float = 1e-18,
    IDSATRGAT: float = 1e-18,
    CSRHBOT: float = 100.0,
    CSRHSTI: float = 0.0001,
    CSRHGAT: float = 0.0001,
    XJUNSTI: float = 1e-07,
    XJUNGAT: float = 1e-07,
    CTATBOT: float = 100.0,
    CTATSTI: float = 0.0001,
    CTATGAT: float = 0.0001,
    MEFFTATBOT: float = 0.25,
    MEFFTATSTI: float = 0.25,
    MEFFTATGAT: float = 0.25,
    CBBTBOT: float = 1e-12,
    CBBTSTI: float = 1e-18,
    CBBTGAT: float = 1e-18,
    FBBTRBOT: float = 1000000000.0,
    FBBTRSTI: float = 1000000000.0,
    FBBTRGAT: float = 1000000000.0,
    STFBBTBOT: float = -0.001,
    STFBBTSTI: float = -0.001,
    STFBBTGAT: float = -0.001,
    VBRBOT: float = 10.0,
    VBRSTI: float = 10.0,
    VBRGAT: float = 10.0,
    PBRBOT: float = 4.0,
    PBRSTI: float = 4.0,
    PBRGAT: float = 4.0,
    FREV: float = 1000.0,
    RSBOT: float = 0.0,
    RSSTI: float = 0.0,
    RSGAT: float = 0.0,
    RSCOM: float = 0.0,
    STRS: float = 0.0,
    KF: float = 0.0,
    AF: float = 1.0,
    TT: float = 0.0,
    STVBRBOT1: float = 0.0,
    STVBRBOT2: float = 0.0,
    STVBRSTI1: float = 0.0,
    STVBRSTI2: float = 0.0,
    STVBRGAT1: float = 0.0,
    STVBRGAT2: float = 0.0,
    NFABOT: float = 1.0,
    NFASTI: float = 1.0,
    NFAGAT: float = 1.0,
    ABMIN: float = 0.0,
    ABMAX: float = 1.0,
    LSMIN: float = 0.0,
    LSMAX: float = 1.0,
    LGMIN: float = 0.0,
    LGMAX: float = 1.0,
    TEMPMIN: float = -55.0,
    TEMPMAX: float = 155.0,
    VFMAX: float = 0.0,
    VRMAX: float = 0.0,
    XTI: float = 3.0,
    SCALE: float = 1.0,
    SHRINK: float = 0.0,
    SWJUNEXP: float = 0.0,
    VJUNREF: float = 2.5,
    FJUNQ: float = 0.03,
    CORECOVERY: float = 0.0,
    NJH: float = 1.0,
    NJDV: float = 0.1,
    NDIBOT: float = 1e16,
    NDIGAT: float = 1e16,
    NDISTI: float = 1e16,
    INJ1: float = 1.0,
    INJ2: float = 10.0,
    NQS: float = 5e-09,
    TAU: float = 2e-07,
    WI: float = 5e-06,
    DEPNQS: float = 0.0,
    TNOM: float = 21.0,
    TAUT: float = 0.0,
    INJT: float = 0.0,
    AB: float = 1e-12,
    LS: float = 1e-06,
    LG: float = 0.0,
    MULT: float = 1.0,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    i_v7358 = jnp.where(MULT > 0.0, MULT, 0.0)
    i_v374514 = TYPE * i_v7358
    i_v11625 = jnp.where(SWJUNEXP > 0.5, 1.0, 0.0)
    i_v11696 = i_v11625 > 0.9
    i_v277 = jnp.where(NFABOT > 0.1, NFABOT, 0.1)
    i_v285 = jnp.where(NFAGAT > 0.1, NFAGAT, 0.1)
    i_v11943 = i_v277 - i_v285
    i_v355 = jnp.where(SCALE > 0.0, SCALE, 0.0)
    i_v6474 = 1.0 - 0.01 * jnp.where(SHRINK > 0.0, SHRINK, 0.0)
    i_v7324 = jnp.where(AB > 0.0, AB, 0.0) * i_v355 * i_v355 * i_v6474 * i_v6474
    i_v11404 = i_v7324 > 0.0
    i_v7352 = jnp.where(LG > 0.0, LG, 0.0) * i_v355 * i_v6474
    i_v281 = jnp.where(NFASTI > 0.1, NFASTI, 0.1)
    i_v12141 = i_v277 - i_v281
    i_v7338 = jnp.where(LS > 0.0, LS, 0.0) * i_v355 * i_v6474
    i_v12231 = i_v285 - i_v281
    i_v12273 = jnp.where(
        jnp.where(
            jnp.where(jnp.where(jnp.where(i_v11943 < 0.0, -i_v11943, i_v11943) > 1e-06, i_v11404, False), i_v7352 > 0.0, False),
            True,
            jnp.where(jnp.where(jnp.where(i_v12141 < 0.0, -i_v12141, i_v12141) > 1e-06, i_v11404, False), i_v7338 > 0.0, False),
        ),
        True,
        jnp.where(jnp.where(jnp.where(i_v12231 < 0.0, -i_v12231, i_v12231) > 1e-06, i_v7352 > 0.0, False), i_v7338 > 0.0, False),
    )
    i_v12433 = jnp.where(i_v11696, jnp.where(i_v12273, 0.0, i_v11625), i_v11625) == 1.0
    i_v574 = _temperature + DTA
    i_v579 = jnp.where(i_v574 > 23.149999999999977, i_v574, 23.149999999999977)
    i_v570 = 273.15 + jnp.where(
        jnp.where(jnp.logical_not(True), True, False), jnp.where(TNOM > -250.0, TNOM, -250.0), jnp.where(TRJ > -250.0, TRJ, -250.0)
    )
    i_v584 = jnp.divide(i_v579, jnp.where((i_v570 == 0.0) | jnp.logical_not(jnp.isfinite(i_v570)), 1e-300, i_v570))
    i_v1101 = jnp.divide(jnp.where(XTI > 0.1, XTI, 0.1), 2.0)
    i_v603 = 1108.0 + i_v570
    i_v604 = jnp.divide(
        -(0.000702 * i_v570 * i_v570), jnp.where((i_v603 == 0.0) | jnp.logical_not(jnp.isfinite(i_v603)), 1e-300, i_v603)
    )
    i_v590 = 8.61726105451295e-05 * i_v570
    i_v592 = jnp.divide(1.0, jnp.where((i_v590 == 0.0) | jnp.logical_not(jnp.isfinite(i_v590)), 1e-300, i_v590))
    i_v1074 = 1108.0 + i_v579
    i_v1075 = jnp.divide(
        -(0.000702 * i_v579 * i_v579), jnp.where((i_v1074 == 0.0) | jnp.logical_not(jnp.isfinite(i_v1074)), 1e-300, i_v1074)
    )
    i_v1077 = PHIGBOT + i_v1075
    i_v594 = 8.61726105451295e-05 * i_v579
    i_v596 = jnp.divide(1.0, jnp.where((i_v594 == 0.0) | jnp.logical_not(jnp.isfinite(i_v594)), 1e-300, i_v594))
    i_v1106 = 0.5 * ((PHIGBOT + i_v604) * i_v592 - i_v1077 * i_v596)
    i_v1194 = jnp.power(
        jnp.maximum(i_v584, 0.0),
        jnp.divide(i_v1101, jnp.where((i_v277 == 0.0) | jnp.logical_not(jnp.isfinite(i_v277)), 1e-300, i_v277)),
    ) * jnp.exp(
        jnp.clip(
            jnp.divide(i_v1106, jnp.where((i_v277 == 0.0) | jnp.logical_not(jnp.isfinite(i_v277)), 1e-300, i_v277)), -709.0, 709.0
        )
    )
    i_v1474 = jnp.where(IDSATRBOT > 0.0, IDSATRBOT, 0.0) * i_v1194 * i_v1194
    i_v7412 = i_v1474 * i_v7324
    i_v1079 = PHIGSTI + i_v1075
    i_v1115 = 0.5 * ((PHIGSTI + i_v604) * i_v592 - i_v1079 * i_v596)
    i_v1259 = jnp.power(
        jnp.maximum(i_v584, 0.0),
        jnp.divide(i_v1101, jnp.where((i_v281 == 0.0) | jnp.logical_not(jnp.isfinite(i_v281)), 1e-300, i_v281)),
    ) * jnp.exp(
        jnp.clip(
            jnp.divide(i_v1115, jnp.where((i_v281 == 0.0) | jnp.logical_not(jnp.isfinite(i_v281)), 1e-300, i_v281)), -709.0, 709.0
        )
    )
    i_v1624 = jnp.where(IDSATRSTI > 0.0, IDSATRSTI, 0.0) * i_v1259 * i_v1259
    i_v7753 = i_v1624 * i_v7338
    i_v1081 = PHIGGAT + i_v1075
    i_v1124 = 0.5 * ((PHIGGAT + i_v604) * i_v592 - i_v1081 * i_v596)
    i_v1321 = jnp.power(
        jnp.maximum(i_v584, 0.0),
        jnp.divide(i_v1101, jnp.where((i_v285 == 0.0) | jnp.logical_not(jnp.isfinite(i_v285)), 1e-300, i_v285)),
    ) * jnp.exp(
        jnp.clip(
            jnp.divide(i_v1124, jnp.where((i_v285 == 0.0) | jnp.logical_not(jnp.isfinite(i_v285)), 1e-300, i_v285)), -709.0, 709.0
        )
    )
    i_v1771 = jnp.where(IDSATRGAT > 0.0, IDSATRGAT, 0.0) * i_v1321 * i_v1321
    i_v7869 = i_v1771 * i_v7352
    i_v10873 = i_v7412 + i_v7753 + i_v7869
    i_v259383 = jnp.where(i_v12433, i_v10873, 0.0)
    i_v257200 = TYPE == -1.0
    i_v224964 = jnp.where(
        i_v11696,
        jnp.where(
            i_v12273, 1.0, jnp.where(i_v7338 > 0.0, i_v281, jnp.where(i_v7352 > 0.0, i_v285, jnp.where(i_v11404, i_v277, 1.0)))
        ),
        1.0,
    )
    i_v8640 = i_v7324 == 0.0
    i_v8686 = i_v7338 == 0.0
    i_v8764 = i_v7352 == 0.0
    i_v247031 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v43 = jnp.where(IMAX > 1e-12, IMAX, 1e-12)
    i_v249106 = 0.5 * i_v43
    i_v251206 = jnp.where(
        i_v247031,
        jnp.log(
            jnp.maximum(
                jnp.divide(
                    i_v249106,
                    jnp.where(
                        (i_v10873 + 1e-21 == 0.0) | jnp.logical_not(jnp.isfinite(i_v10873 + 1e-21)), 1e-300, i_v10873 + 1e-21
                    ),
                ),
                1e-300,
            )
        ),
        0.0,
    )
    i_v253199 = jnp.where(i_v251206 < 230.25850929940458, i_v251206, 230.25850929940458)
    i_v259262, i_v259277 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v12433, _t, _f), (i_v253199, jnp.exp(jnp.clip(i_v253199, -709.0, 709.0))), (0.0, 0.0)
    )
    i_v226864 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v138867 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v7938 = jnp.where(
        i_v7412 > 0.0,
        i_v594
        * jnp.log(
            jnp.maximum(
                jnp.divide(i_v43, jnp.where((i_v7412 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7412)), 1e-300, i_v7412)) + 1.0,
                1e-300,
            )
        )
        * i_v277,
        100000000.0,
    )
    i_v7943 = jnp.where(
        i_v7753 > 0.0,
        i_v594
        * jnp.log(
            jnp.maximum(
                jnp.divide(i_v43, jnp.where((i_v7753 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7753)), 1e-300, i_v7753)) + 1.0,
                1e-300,
            )
        )
        * i_v281,
        100000000.0,
    )
    i_v7946 = jnp.where(i_v7938 < i_v7943, i_v7938, i_v7943)
    i_v7949 = jnp.where(
        i_v7869 > 0.0,
        i_v594
        * jnp.log(
            jnp.maximum(
                jnp.divide(i_v43, jnp.where((i_v7869 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7869)), 1e-300, i_v7869)) + 1.0,
                1e-300,
            )
        )
        * i_v285,
        100000000.0,
    )
    i_v7952 = jnp.where(i_v7946 < i_v7949, i_v7946, i_v7949)
    i_v141030 = i_v7952 > 0.1
    i_v7255 = jnp.divide(i_v277, i_v596)
    i_v7129 = NDIBOT * 1000000.0
    i_v7207 = 1.45e16 * i_v1194
    i_v7209 = i_v7207 * i_v7207
    i_v7210 = jnp.divide(i_v7209, jnp.where((i_v7129 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7129)), 1e-300, i_v7129))
    i_v7257 = jnp.log(
        jnp.maximum(
            jnp.divide(i_v7129, jnp.where((i_v7210 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7210)), 1e-300, i_v7210)), 1e-300
        )
    )
    i_v7258 = i_v7255 * i_v7257
    i_v142353 = NJH - (NJDV * (0.1 - i_v7258) + i_v277) - 0.01
    i_v142355 = 4.0 * NJH * 0.01
    i_v142374 = (
        NJH
        - 0.5
        * (i_v142353 + jnp.sqrt(jnp.maximum(i_v142353 * i_v142353 + jnp.where(i_v142355 > 0.0, i_v142355, -i_v142355), 1e-300)))
        - i_v277
        - 0.01
    )
    i_v142376 = 4.0 * i_v277 * 0.01
    i_v142412 = jnp.where(
        i_v277 < NJH,
        i_v277
        + 0.5
        * (i_v142374 + jnp.sqrt(jnp.maximum(i_v142374 * i_v142374 + jnp.where(i_v142376 > 0.0, i_v142376, -i_v142376), 1e-300))),
        i_v277,
    )
    i_v142428 = i_v596 * (
        jnp.divide(0.1, jnp.where((i_v142412 == 0.0) | jnp.logical_not(jnp.isfinite(i_v142412)), 1e-300, i_v142412))
        + jnp.divide(
            i_v7258 * (i_v142412 - i_v277),
            jnp.where((i_v277 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v277 * NJH)), 1e-300, i_v277 * NJH),
        )
    )
    i_v142488 = -230.25850929940458 - i_v142428
    i_v142527 = i_v142428 - 230.25850929940458
    i_v144309 = 0.1 - i_v7952
    i_v144690 = i_v277 < NJH
    i_v144698 = NJH - (NJDV * (i_v7952 - i_v7258) + i_v277) - 0.01
    i_v144700 = 4.0 * NJH * 0.01
    i_v144711 = jnp.sqrt(jnp.maximum(i_v144698 * i_v144698 + jnp.where(i_v144700 > 0.0, i_v144700, -i_v144700), 1e-300))
    i_v144722 = NJH - 0.5 * (i_v144698 + i_v144711) - i_v277 - 0.01
    i_v144724 = 4.0 * i_v277 * 0.01
    i_v144735 = jnp.sqrt(jnp.maximum(i_v144722 * i_v144722 + jnp.where(i_v144724 > 0.0, i_v144724, -i_v144724), 1e-300))
    i_v144768, i_v144918 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v144690, _t, _f),
        (
            i_v277 + 0.5 * (i_v144722 + i_v144735),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v144698, i_v144711))) * (0.5 * (1.0 + jnp.divide(i_v144722, i_v144735))),
        ),
        (i_v277, 0.0),
    )
    i_v144781 = i_v277 * NJH
    i_v144784 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v144768 == 0.0) | jnp.logical_not(jnp.isfinite(i_v144768)), 1e-300, i_v144768))
        + jnp.divide(
            i_v7258 * (i_v144768 - i_v277),
            jnp.where((i_v144781 == 0.0) | jnp.logical_not(jnp.isfinite(i_v144781)), 1e-300, i_v144781),
        )
    )
    i_v144844 = -230.25850929940458 - i_v144784
    i_v144883 = i_v144784 - 230.25850929940458
    i_v118 = jnp.where(CSRHBOT > 0.0, CSRHBOT, 0.0)
    i_v145 = jnp.where(CTATBOT > 0.0, CTATBOT, 0.0)
    i_v150254 = jnp.where(i_v118 == 0.0, i_v145 == 0.0, False)
    i_v1102 = jnp.power(jnp.maximum(i_v584, 0.0), i_v1101)
    i_v1108 = i_v1102 * jnp.exp(jnp.clip(i_v1106, -709.0, 709.0))
    i_v141283 = 0.5 * (0.1 * i_v596)
    i_v141310 = -230.25850929940458 - i_v141283
    i_v141331 = i_v141283 - 230.25850929940458
    i_v7981 = i_v7952 * i_v596
    i_v8007 = -230.25850929940458 - i_v7981
    i_v8027 = i_v7981 - 230.25850929940458
    i_v15213 = jnp.where(
        jnp.where(i_v7981 < 0.0, -i_v7981, i_v7981) < 230.25850929940458,
        jnp.exp(jnp.clip(i_v7981, -709.0, 709.0)),
        jnp.where(
            i_v7981 < -230.25850929940458,
            jnp.divide(
                1e-100,
                jnp.where(
                    (1.0 + i_v8007 * (1.0 + 0.5 * (i_v8007 * (1.0 + i_v8007 * 0.3333333333333333))) == 0.0)
                    | jnp.logical_not(jnp.isfinite(1.0 + i_v8007 * (1.0 + 0.5 * (i_v8007 * (1.0 + i_v8007 * 0.3333333333333333))))),
                    1e-300,
                    1.0 + i_v8007 * (1.0 + 0.5 * (i_v8007 * (1.0 + i_v8007 * 0.3333333333333333))),
                ),
            ),
            1e100 * (1.0 + i_v8027 * (1.0 + 0.5 * (i_v8027 * (1.0 + i_v8027 * 0.3333333333333333)))),
        ),
    )
    i_v145787 = jnp.where(
        i_v141030,
        jnp.where(
            jnp.where(i_v141283 < 0.0, -i_v141283, i_v141283) < 230.25850929940458,
            jnp.exp(jnp.clip(i_v141283, -709.0, 709.0)),
            jnp.where(
                i_v141283 < -230.25850929940458,
                jnp.divide(
                    1e-100,
                    jnp.where(
                        (1.0 + i_v141310 * (1.0 + 0.5 * (i_v141310 * (1.0 + i_v141310 * 0.3333333333333333))) == 0.0)
                        | jnp.logical_not(
                            jnp.isfinite(1.0 + i_v141310 * (1.0 + 0.5 * (i_v141310 * (1.0 + i_v141310 * 0.3333333333333333))))
                        ),
                        1e-300,
                        1.0 + i_v141310 * (1.0 + 0.5 * (i_v141310 * (1.0 + i_v141310 * 0.3333333333333333))),
                    ),
                ),
                1e100 * (1.0 + i_v141331 * (1.0 + 0.5 * (i_v141331 * (1.0 + i_v141331 * 0.3333333333333333)))),
            ),
        ),
        jnp.sqrt(jnp.maximum((1.0 + i_v144309 * i_v596) * i_v15213, 1e-300)),
    )
    i_v152102 = jnp.where(i_v138867, i_v145787, 0.0)
    i_v47 = jnp.where(CJORBOT > 1e-12, CJORBOT, 1e-12)
    i_v3628 = jnp.divide(1.0447941624768001e-10, jnp.where((i_v47 == 0.0) | jnp.logical_not(jnp.isfinite(i_v47)), 1e-300, i_v47))
    i_v76 = jnp.where(PBOT > 0.05, jnp.where(PBOT < 0.95, PBOT, 0.95), 0.05)
    i_v150986 = i_v76 == 0.5
    i_v61 = jnp.where(VBIRBOT > 0.05, VBIRBOT, 0.05)
    i_v1953 = 2.0 * i_v594
    i_v1956 = i_v61 * i_v584 - i_v1953 * jnp.log(jnp.maximum(i_v1108, 1e-300))
    i_v2324 = i_v1956 + i_v594 * jnp.log(jnp.maximum(1.0 + jnp.exp(jnp.clip((0.05 - i_v1956) * i_v596, -709.0, 709.0)), 1e-300))
    i_v65 = jnp.where(VBIRSTI > 0.05, VBIRSTI, 0.05)
    i_v1117 = i_v1102 * jnp.exp(jnp.clip(i_v1115, -709.0, 709.0))
    i_v2137 = i_v65 * i_v584 - i_v1953 * jnp.log(jnp.maximum(i_v1117, 1e-300))
    i_v2333 = i_v2137 + i_v594 * jnp.log(jnp.maximum(1.0 + jnp.exp(jnp.clip((0.05 - i_v2137) * i_v596, -709.0, 709.0)), 1e-300))
    i_v69 = jnp.where(VBIRGAT > 0.05, VBIRGAT, 0.05)
    i_v1126 = i_v1102 * jnp.exp(jnp.clip(i_v1124, -709.0, 709.0))
    i_v2315 = i_v69 * i_v584 - i_v1953 * jnp.log(jnp.maximum(i_v1126, 1e-300))
    i_v2342 = i_v2315 + i_v594 * jnp.log(jnp.maximum(1.0 + jnp.exp(jnp.clip((0.05 - i_v2315) * i_v596, -709.0, 709.0)), 1e-300))
    i_v8828 = jnp.where(i_v8640, i_v2333 + i_v2342, i_v2324)
    i_v8843 = jnp.where(i_v8686, i_v2324 + i_v2342, i_v2333)
    i_v8859 = jnp.where(i_v8828 < i_v8843, i_v8828, i_v8843)
    i_v8862 = jnp.where(i_v8764, i_v2324 + i_v2333, i_v2342)
    i_v8881 = jnp.where(i_v8859 < i_v8862, i_v8859, i_v8862)
    i_v145898 = jnp.divide(1.0, jnp.where((i_v145787 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145787)), 1e-300, i_v145787))
    i_v146277 = 2.0 * (
        i_v594
        * jnp.log(jnp.maximum(2.0 + i_v145898 + jnp.sqrt(jnp.maximum((i_v145898 + 1.0) * (i_v145898 + 3.0), 1e-300)), 1e-300))
    )
    i_v146666 = i_v8881 - i_v146277
    i_v150632 = jnp.where(
        i_v138867,
        0.5 * (0.1 + i_v146666 - jnp.sqrt(jnp.maximum((0.1 - i_v146666) * (0.1 - i_v146666) + 4.0 * i_v594 * i_v594, 1e-300))),
        0.0,
    )
    i_v150635 = i_v2324 - i_v150632
    i_v3907 = jnp.divide(1.0, jnp.where((i_v61 == 0.0) | jnp.logical_not(jnp.isfinite(i_v61)), 1e-300, i_v61))
    i_v151725 = i_v3628 * jnp.where(
        i_v150986, jnp.sqrt(jnp.maximum(i_v150635 * i_v3907, 1e-300)), jnp.power(jnp.maximum(i_v150635 * i_v3907, 0.0), i_v76)
    )
    i_v152112 = i_v1108 * ((i_v152102 - 1.0) * i_v151725)
    i_v150644 = jnp.where(i_v138867, i_v146277, 0.0)
    i_v150649 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v150644, jnp.where((i_v150635 == 0.0) | jnp.logical_not(jnp.isfinite(i_v150635)), 1e-300, i_v150635)),
            1e-300,
        )
    )
    i_v150650 = 1.0 - i_v150649
    i_v151005 = i_v150650 + jnp.where(
        i_v150986,
        0.0,
        (jnp.divide(i_v150650 * i_v150650 * jnp.log(jnp.maximum(i_v150650, 1e-300)), i_v150649) + i_v150650) * (1.0 - 2.0 * i_v76),
    )
    i_v3937 = 0.5 * i_v1077
    i_v3939 = jnp.where(i_v3937 > i_v594, i_v3937, i_v594)
    i_v3979 = i_v3939 * i_v596
    i_v4127 = jnp.divide(
        jnp.sqrt(
            jnp.maximum(
                32.0
                * jnp.where(MEFFTATBOT > 0.01, MEFFTATBOT, 0.01)
                * 9.1093826e-31
                * 1.6021918e-19
                * (i_v3939 * i_v3939 * i_v3939),
                1e-300,
            )
        ),
        3.1637150399999996e-34,
    )
    i_v2516 = 1.0 - i_v76
    i_v152530 = i_v4127 * jnp.divide(
        jnp.where(i_v150254, 0.0, i_v151725) * i_v2516,
        jnp.where(
            (jnp.where(i_v150254, 0.0, i_v150635) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v150254, 0.0, i_v150635))),
            1e-300,
            jnp.where(i_v150254, 0.0, i_v150635),
        ),
    )
    i_v152880 = jnp.divide(
        0.666666666666667 * i_v3979, jnp.where((i_v152530 == 0.0) | jnp.logical_not(jnp.isfinite(i_v152530)), 1e-300, i_v152530)
    )
    i_v152881 = i_v152880 * i_v152880
    i_v152882 = i_v152881 * i_v152881
    i_v152887 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v152882,
                jnp.where((i_v152882 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v152882 + 1.0)), 1e-300, i_v152882 + 1.0),
            ),
            1e-300,
        )
    )
    i_v152891 = jnp.sqrt(jnp.maximum(jnp.where(i_v152887 < 0.0, -i_v152887, i_v152887), 1e-300))
    i_v153292 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v152530, i_v152891), 1e-300))
    i_v153322 = (2.0 * (i_v152880 * i_v152891) - i_v152887 - 1.0) * i_v153292
    i_v153325 = i_v153322 > 0.0
    i_v153587 = jnp.where(
        i_v153325,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v153322 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v153322)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v153322,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v153322 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v153322)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v153322,
            ),
        ),
    )
    i_v153800 = i_v153587 * i_v153587
    i_v152895 = i_v152887 * i_v152891
    i_v153319 = i_v3979 * i_v152880 * i_v152891 - i_v3979 * i_v152887 + 0.5 * (i_v152530 * i_v152895)
    i_v153555 = -(i_v153322 * i_v153322) + i_v153319
    i_v153568 = -230.25850929940458 - i_v153555
    i_v154018 = (
        0.29214664 * i_v153587 + 0.26992878119627894 * i_v153800 + 0.43792457880372104 * (i_v153800 * i_v153587)
    ) * jnp.where(
        i_v153555 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v153555, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v153568 * (1.0 + 0.5 * (i_v153568 * (1.0 + i_v153568 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v153568 * (1.0 + 0.5 * (i_v153568 * (1.0 + i_v153568 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v153568 * (1.0 + 0.5 * (i_v153568 * (1.0 + i_v153568 * 0.3333333333333333))),
            ),
        ),
    )
    i_v154034 = -230.25850929940458 - i_v153319
    i_v153279 = jnp.where(i_v150254, 0.0, i_v151005)
    i_v2837 = jnp.divide(1.0, jnp.where((i_v2516 == 0.0) | jnp.logical_not(jnp.isfinite(i_v2516)), 1e-300, i_v2516))
    i_v153250 = -i_v76 * i_v2837
    i_v153282 = jnp.where(
        i_v153250 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v152530 * i_v152895 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v152530 * i_v152895)),
                1e-300,
                1.0 + i_v152530 * i_v152895,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v152530 * i_v152895, 0.0), i_v153250),
    )
    i_v176 = jnp.where(CBBTBOT > 0.0, CBBTBOT, 0.0)
    i_v8979 = jnp.where(i_v8640, i_v65 + i_v69, i_v61)
    i_v9003 = jnp.where(i_v8686, i_v61 + i_v69, i_v65)
    i_v9020 = jnp.where(i_v8979 < i_v9003, i_v8979, i_v9003)
    i_v9031 = jnp.where(i_v8764, i_v61 + i_v65, i_v69)
    i_v9052 = jnp.where(i_v9020 < i_v9031, i_v9020, i_v9031) - 0.05
    i_v154887 = jnp.where(
        i_v138867,
        0.5 * (0.1 + i_v9052 - jnp.sqrt(jnp.maximum((0.1 - i_v9052) * (0.1 - i_v9052) + 4.0 * i_v590 * i_v590, 1e-300))),
        0.0,
    )
    i_v3898 = jnp.divide(1.0, i_v3628)
    i_v155344 = i_v2837 * jnp.divide(
        (i_v61 - i_v154887) * i_v3898,
        jnp.where(
            (
                jnp.where(
                    i_v76 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v61 - i_v154887) * i_v3907, 1e-300)),
                    jnp.power(jnp.maximum((i_v61 - i_v154887) * i_v3907, 0.0), i_v76),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v76 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v61 - i_v154887) * i_v3907, 1e-300)),
                        jnp.power(jnp.maximum((i_v61 - i_v154887) * i_v3907, 0.0), i_v76),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v76 == 0.5,
                jnp.sqrt(jnp.maximum((i_v61 - i_v154887) * i_v3907, 1e-300)),
                jnp.power(jnp.maximum((i_v61 - i_v154887) * i_v3907, 0.0), i_v76),
            ),
        ),
    )
    i_v4637 = i_v579 - i_v570
    i_v4640 = FBBTRBOT * (1.0 + STFBBTBOT * i_v4637)
    i_v5102 = jnp.where(i_v4640 > 0.0, i_v4640, 0.0)
    i_v155705 = jnp.divide(-i_v5102, jnp.where((i_v155344 == 0.0) | jnp.logical_not(jnp.isfinite(i_v155344)), 1e-300, i_v155344))
    i_v155732 = -230.25850929940458 - i_v155705
    i_v155753 = i_v155705 - 230.25850929940458
    i_v5859 = jnp.where(VBRBOT > 0.1, VBRBOT, 0.1) * (1.0 + i_v4637 * (STVBRBOT1 + i_v4637 * STVBRBOT2))
    i_v6413 = i_v5859 <= 0.1
    i_v26813 = jnp.where(i_v6413, 0.1, i_v5859)
    i_v156214 = jnp.where(i_v138867, -1.000000082740371e-11, 0.0)
    i_v5226 = jnp.divide(FREV - 1.0, jnp.where((FREV == 0.0) | jnp.logical_not(jnp.isfinite(FREV)), 1e-300, FREV))
    i_v217 = jnp.where(PBRBOT > 0.1, PBRBOT, 0.1)
    i_v6514 = jnp.where(
        i_v6413, 10.0, jnp.divide(1.0, jnp.where((i_v5859 == 0.0) | jnp.logical_not(jnp.isfinite(i_v5859)), 1e-300, i_v5859))
    )
    i_v157191 = i_v156214 * i_v6514
    i_v157192 = i_v157191 < 0.0
    i_v157230 = i_v156214 * i_v6514
    i_v5339 = 1.0 - jnp.power(jnp.maximum(i_v5226, 0.0), i_v217)
    i_v5340 = jnp.divide(1.0, jnp.where((i_v5339 == 0.0) | jnp.logical_not(jnp.isfinite(i_v5339)), 1e-300, i_v5339))
    i_v6517 = -(i_v5340 * i_v5340 * jnp.power(jnp.maximum(i_v5226, 0.0), i_v217 - 1.0)) * i_v217 * i_v6514
    i_v7132 = NDISTI * 1000000.0
    i_v143237 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v143248 = NJH - (NJDV * (0.1 - i_v143237) + i_v281) - 0.01
    i_v143250 = 4.0 * NJH * 0.01
    i_v143269 = (
        NJH
        - 0.5
        * (i_v143248 + jnp.sqrt(jnp.maximum(i_v143248 * i_v143248 + jnp.where(i_v143250 > 0.0, i_v143250, -i_v143250), 1e-300)))
        - i_v281
        - 0.01
    )
    i_v143271 = 4.0 * i_v281 * 0.01
    i_v143307 = jnp.where(
        i_v281 < NJH,
        i_v281
        + 0.5
        * (i_v143269 + jnp.sqrt(jnp.maximum(i_v143269 * i_v143269 + jnp.where(i_v143271 > 0.0, i_v143271, -i_v143271), 1e-300))),
        i_v281,
    )
    i_v143323 = i_v596 * (
        jnp.divide(0.1, jnp.where((i_v143307 == 0.0) | jnp.logical_not(jnp.isfinite(i_v143307)), 1e-300, i_v143307))
        + jnp.divide(
            i_v143237 * (i_v143307 - i_v281),
            jnp.where((i_v281 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v281 * NJH)), 1e-300, i_v281 * NJH),
        )
    )
    i_v143383 = -230.25850929940458 - i_v143323
    i_v143422 = i_v143323 - 230.25850929940458
    i_v145024 = i_v281 < NJH
    i_v145023 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v145032 = NJH - (NJDV * (i_v7952 - i_v145023) + i_v281) - 0.01
    i_v145034 = 4.0 * NJH * 0.01
    i_v145045 = jnp.sqrt(jnp.maximum(i_v145032 * i_v145032 + jnp.where(i_v145034 > 0.0, i_v145034, -i_v145034), 1e-300))
    i_v145056 = NJH - 0.5 * (i_v145032 + i_v145045) - i_v281 - 0.01
    i_v145058 = 4.0 * i_v281 * 0.01
    i_v145069 = jnp.sqrt(jnp.maximum(i_v145056 * i_v145056 + jnp.where(i_v145058 > 0.0, i_v145058, -i_v145058), 1e-300))
    i_v145102, i_v145252 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v145024, _t, _f),
        (
            i_v281 + 0.5 * (i_v145056 + i_v145069),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v145032, i_v145045))) * (0.5 * (1.0 + jnp.divide(i_v145056, i_v145069))),
        ),
        (i_v281, 0.0),
    )
    i_v145115 = i_v281 * NJH
    i_v145118 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v145102 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145102)), 1e-300, i_v145102))
        + jnp.divide(
            i_v145023 * (i_v145102 - i_v281),
            jnp.where((i_v145115 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145115)), 1e-300, i_v145115),
        )
    )
    i_v145178 = -230.25850929940458 - i_v145118
    i_v145217 = i_v145118 - 230.25850929940458
    i_v124 = jnp.where(CSRHSTI > 0.0, CSRHSTI, 0.0)
    i_v151 = jnp.where(CTATSTI > 0.0, CTATSTI, 0.0)
    i_v161128 = jnp.where(i_v124 == 0.0, i_v151 == 0.0, False)
    i_v52 = jnp.where(CJORSTI > 1e-18, CJORSTI, 1e-18)
    i_v3763 = jnp.divide(
        jnp.where(XJUNSTI > 1e-09, XJUNSTI, 1e-09) * 1.0447941624768001e-10,
        jnp.where((i_v52 == 0.0) | jnp.logical_not(jnp.isfinite(i_v52)), 1e-300, i_v52),
    )
    i_v82 = jnp.where(PSTI > 0.05, jnp.where(PSTI < 0.95, PSTI, 0.95), 0.05)
    i_v161998 = i_v82 == 0.5
    i_v161578 = i_v2333 - i_v150632
    i_v3910 = jnp.divide(1.0, jnp.where((i_v65 == 0.0) | jnp.logical_not(jnp.isfinite(i_v65)), 1e-300, i_v65))
    i_v162737 = i_v3763 * jnp.where(
        i_v161998, jnp.sqrt(jnp.maximum(i_v161578 * i_v3910, 1e-300)), jnp.power(jnp.maximum(i_v161578 * i_v3910, 0.0), i_v82)
    )
    i_v163185 = i_v1117 * ((i_v152102 - 1.0) * i_v162737)
    i_v161661 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v150644, jnp.where((i_v161578 == 0.0) | jnp.logical_not(jnp.isfinite(i_v161578)), 1e-300, i_v161578)),
            1e-300,
        )
    )
    i_v161662 = 1.0 - i_v161661
    i_v162017 = i_v161662 + jnp.where(
        i_v161998,
        0.0,
        (jnp.divide(i_v161662 * i_v161662 * jnp.log(jnp.maximum(i_v161662, 1e-300)), i_v161661) + i_v161662) * (1.0 - 2.0 * i_v82),
    )
    i_v3944 = 0.5 * i_v1079
    i_v3949 = jnp.where(i_v3944 > i_v594, i_v3944, i_v594)
    i_v3984 = i_v3949 * i_v596
    i_v4262 = jnp.divide(
        jnp.sqrt(
            jnp.maximum(
                32.0
                * jnp.where(MEFFTATSTI > 0.01, MEFFTATSTI, 0.01)
                * 9.1093826e-31
                * 1.6021918e-19
                * (i_v3949 * i_v3949 * i_v3949),
                1e-300,
            )
        ),
        3.1637150399999996e-34,
    )
    i_v2678 = 1.0 - i_v82
    i_v163603 = i_v4262 * jnp.divide(
        jnp.where(i_v161128, 0.0, i_v162737) * i_v2678,
        jnp.where(
            (jnp.where(i_v161128, 0.0, i_v161578) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v161128, 0.0, i_v161578))),
            1e-300,
            jnp.where(i_v161128, 0.0, i_v161578),
        ),
    )
    i_v163953 = jnp.divide(
        0.666666666666667 * i_v3984, jnp.where((i_v163603 == 0.0) | jnp.logical_not(jnp.isfinite(i_v163603)), 1e-300, i_v163603)
    )
    i_v163954 = i_v163953 * i_v163953
    i_v163955 = i_v163954 * i_v163954
    i_v163960 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v163955,
                jnp.where((i_v163955 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v163955 + 1.0)), 1e-300, i_v163955 + 1.0),
            ),
            1e-300,
        )
    )
    i_v163964 = jnp.sqrt(jnp.maximum(jnp.where(i_v163960 < 0.0, -i_v163960, i_v163960), 1e-300))
    i_v164365 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v163603, i_v163964), 1e-300))
    i_v164395 = (2.0 * (i_v163953 * i_v163964) - i_v163960 - 1.0) * i_v164365
    i_v164398 = i_v164395 > 0.0
    i_v164528 = jnp.where(
        i_v164398,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v164395 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v164395)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v164395,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v164395 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v164395)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v164395,
            ),
        ),
    )
    i_v164609 = i_v164528 * i_v164528
    i_v163968 = i_v163960 * i_v163964
    i_v164392 = i_v3984 * i_v163953 * i_v163964 - i_v3984 * i_v163960 + 0.5 * (i_v163603 * i_v163968)
    i_v164496 = -(i_v164395 * i_v164395) + i_v164392
    i_v164509 = -230.25850929940458 - i_v164496
    i_v164695 = (
        0.29214664 * i_v164528 + 0.26992878119627894 * i_v164609 + 0.43792457880372104 * (i_v164609 * i_v164528)
    ) * jnp.where(
        i_v164496 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v164496, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v164509 * (1.0 + 0.5 * (i_v164509 * (1.0 + i_v164509 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v164509 * (1.0 + 0.5 * (i_v164509 * (1.0 + i_v164509 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v164509 * (1.0 + 0.5 * (i_v164509 * (1.0 + i_v164509 * 0.3333333333333333))),
            ),
        ),
    )
    i_v164711 = -230.25850929940458 - i_v164392
    i_v164352 = jnp.where(i_v161128, 0.0, i_v162017)
    i_v2840 = jnp.divide(1.0, jnp.where((i_v2678 == 0.0) | jnp.logical_not(jnp.isfinite(i_v2678)), 1e-300, i_v2678))
    i_v164323 = -i_v82 * i_v2840
    i_v164355 = jnp.where(
        i_v164323 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v163603 * i_v163968 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v163603 * i_v163968)),
                1e-300,
                1.0 + i_v163603 * i_v163968,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v163603 * i_v163968, 0.0), i_v164323),
    )
    i_v182 = jnp.where(CBBTSTI > 0.0, CBBTSTI, 0.0)
    i_v3901 = jnp.divide(1.0, jnp.where((i_v3763 == 0.0) | jnp.logical_not(jnp.isfinite(i_v3763)), 1e-300, i_v3763))
    i_v166057 = i_v2840 * jnp.divide(
        (i_v65 - i_v154887) * i_v3901,
        jnp.where(
            (
                jnp.where(
                    i_v82 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v65 - i_v154887) * i_v3910, 1e-300)),
                    jnp.power(jnp.maximum((i_v65 - i_v154887) * i_v3910, 0.0), i_v82),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v82 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v65 - i_v154887) * i_v3910, 1e-300)),
                        jnp.power(jnp.maximum((i_v65 - i_v154887) * i_v3910, 0.0), i_v82),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v82 == 0.5,
                jnp.sqrt(jnp.maximum((i_v65 - i_v154887) * i_v3910, 1e-300)),
                jnp.power(jnp.maximum((i_v65 - i_v154887) * i_v3910, 0.0), i_v82),
            ),
        ),
    )
    i_v4868 = FBBTRSTI * (1.0 + STFBBTSTI * i_v4637)
    i_v5110 = jnp.where(i_v4868 > 0.0, i_v4868, 0.0)
    i_v166418 = jnp.divide(-i_v5110, jnp.where((i_v166057 == 0.0) | jnp.logical_not(jnp.isfinite(i_v166057)), 1e-300, i_v166057))
    i_v166445 = -230.25850929940458 - i_v166418
    i_v166466 = i_v166418 - 230.25850929940458
    i_v6137 = jnp.where(VBRSTI > 0.1, VBRSTI, 0.1) * (1.0 + i_v4637 * (STVBRSTI1 + i_v4637 * STVBRSTI2))
    i_v6420 = i_v6137 <= 0.1
    i_v39291 = jnp.where(i_v6420, 0.1, i_v6137)
    i_v221 = jnp.where(PBRSTI > 0.1, PBRSTI, 0.1)
    i_v6546 = jnp.where(
        i_v6420, 10.0, jnp.divide(1.0, jnp.where((i_v6137 == 0.0) | jnp.logical_not(jnp.isfinite(i_v6137)), 1e-300, i_v6137))
    )
    i_v167783 = i_v156214 * i_v6546
    i_v167784 = i_v167783 < 0.0
    i_v167822 = i_v156214 * i_v6546
    i_v5450 = 1.0 - jnp.power(jnp.maximum(i_v5226, 0.0), i_v221)
    i_v5451 = jnp.divide(1.0, jnp.where((i_v5450 == 0.0) | jnp.logical_not(jnp.isfinite(i_v5450)), 1e-300, i_v5450))
    i_v6548 = -(i_v5451 * i_v5451 * jnp.power(jnp.maximum(i_v5226, 0.0), i_v221 - 1.0)) * i_v221 * i_v6546
    i_v7135 = NDIGAT * 1000000.0
    i_v144096 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v144107 = NJH - (NJDV * (0.1 - i_v144096) + i_v285) - 0.01
    i_v144109 = 4.0 * NJH * 0.01
    i_v144128 = (
        NJH
        - 0.5
        * (i_v144107 + jnp.sqrt(jnp.maximum(i_v144107 * i_v144107 + jnp.where(i_v144109 > 0.0, i_v144109, -i_v144109), 1e-300)))
        - i_v285
        - 0.01
    )
    i_v144130 = 4.0 * i_v285 * 0.01
    i_v144166 = jnp.where(
        i_v285 < NJH,
        i_v285
        + 0.5
        * (i_v144128 + jnp.sqrt(jnp.maximum(i_v144128 * i_v144128 + jnp.where(i_v144130 > 0.0, i_v144130, -i_v144130), 1e-300))),
        i_v285,
    )
    i_v144182 = i_v596 * (
        jnp.divide(0.1, jnp.where((i_v144166 == 0.0) | jnp.logical_not(jnp.isfinite(i_v144166)), 1e-300, i_v144166))
        + jnp.divide(
            i_v144096 * (i_v144166 - i_v285),
            jnp.where((i_v285 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v285 * NJH)), 1e-300, i_v285 * NJH),
        )
    )
    i_v144242 = -230.25850929940458 - i_v144182
    i_v144281 = i_v144182 - 230.25850929940458
    i_v145394 = i_v285 < NJH
    i_v145393 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v145402 = NJH - (NJDV * (i_v7952 - i_v145393) + i_v285) - 0.01
    i_v145404 = 4.0 * NJH * 0.01
    i_v145415 = jnp.sqrt(jnp.maximum(i_v145402 * i_v145402 + jnp.where(i_v145404 > 0.0, i_v145404, -i_v145404), 1e-300))
    i_v145426 = NJH - 0.5 * (i_v145402 + i_v145415) - i_v285 - 0.01
    i_v145428 = 4.0 * i_v285 * 0.01
    i_v145439 = jnp.sqrt(jnp.maximum(i_v145426 * i_v145426 + jnp.where(i_v145428 > 0.0, i_v145428, -i_v145428), 1e-300))
    i_v145472, i_v145622 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v145394, _t, _f),
        (
            i_v285 + 0.5 * (i_v145426 + i_v145439),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v145402, i_v145415))) * (0.5 * (1.0 + jnp.divide(i_v145426, i_v145439))),
        ),
        (i_v285, 0.0),
    )
    i_v145485 = i_v285 * NJH
    i_v145488 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v145472 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145472)), 1e-300, i_v145472))
        + jnp.divide(
            i_v145393 * (i_v145472 - i_v285),
            jnp.where((i_v145485 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145485)), 1e-300, i_v145485),
        )
    )
    i_v145548 = -230.25850929940458 - i_v145488
    i_v145587 = i_v145488 - 230.25850929940458
    i_v130 = jnp.where(CSRHGAT > 0.0, CSRHGAT, 0.0)
    i_v157 = jnp.where(CTATGAT > 0.0, CTATGAT, 0.0)
    i_v171873 = jnp.where(i_v130 == 0.0, i_v157 == 0.0, False)
    i_v56 = jnp.where(CJORGAT > 1e-18, CJORGAT, 1e-18)
    i_v3895 = jnp.divide(
        jnp.where(XJUNGAT > 1e-09, XJUNGAT, 1e-09) * 1.0447941624768001e-10,
        jnp.where((i_v56 == 0.0) | jnp.logical_not(jnp.isfinite(i_v56)), 1e-300, i_v56),
    )
    i_v88 = jnp.where(PGAT > 0.05, jnp.where(PGAT < 0.95, PGAT, 0.95), 0.05)
    i_v172743 = i_v88 == 0.5
    i_v172323 = i_v2342 - i_v150632
    i_v3913 = jnp.divide(1.0, jnp.where((i_v69 == 0.0) | jnp.logical_not(jnp.isfinite(i_v69)), 1e-300, i_v69))
    i_v173482 = i_v3895 * jnp.where(
        i_v172743, jnp.sqrt(jnp.maximum(i_v172323 * i_v3913, 1e-300)), jnp.power(jnp.maximum(i_v172323 * i_v3913, 0.0), i_v88)
    )
    i_v173930 = i_v1126 * ((i_v152102 - 1.0) * i_v173482)
    i_v172406 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v150644, jnp.where((i_v172323 == 0.0) | jnp.logical_not(jnp.isfinite(i_v172323)), 1e-300, i_v172323)),
            1e-300,
        )
    )
    i_v172407 = 1.0 - i_v172406
    i_v172762 = i_v172407 + jnp.where(
        i_v172743,
        0.0,
        (jnp.divide(i_v172407 * i_v172407 * jnp.log(jnp.maximum(i_v172407, 1e-300)), i_v172406) + i_v172407) * (1.0 - 2.0 * i_v88),
    )
    i_v3957 = 0.5 * i_v1081
    i_v3962 = jnp.where(i_v3957 > i_v594, i_v3957, i_v594)
    i_v3986 = i_v3962 * i_v596
    i_v4394 = jnp.divide(
        jnp.sqrt(
            jnp.maximum(
                32.0
                * jnp.where(MEFFTATGAT > 0.01, MEFFTATGAT, 0.01)
                * 9.1093826e-31
                * 1.6021918e-19
                * (i_v3962 * i_v3962 * i_v3962),
                1e-300,
            )
        ),
        3.1637150399999996e-34,
    )
    i_v2834 = 1.0 - i_v88
    i_v174348 = i_v4394 * jnp.divide(
        jnp.where(i_v171873, 0.0, i_v173482) * i_v2834,
        jnp.where(
            (jnp.where(i_v171873, 0.0, i_v172323) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v171873, 0.0, i_v172323))),
            1e-300,
            jnp.where(i_v171873, 0.0, i_v172323),
        ),
    )
    i_v174698 = jnp.divide(
        0.666666666666667 * i_v3986, jnp.where((i_v174348 == 0.0) | jnp.logical_not(jnp.isfinite(i_v174348)), 1e-300, i_v174348)
    )
    i_v174699 = i_v174698 * i_v174698
    i_v174700 = i_v174699 * i_v174699
    i_v174705 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v174700,
                jnp.where((i_v174700 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v174700 + 1.0)), 1e-300, i_v174700 + 1.0),
            ),
            1e-300,
        )
    )
    i_v174709 = jnp.sqrt(jnp.maximum(jnp.where(i_v174705 < 0.0, -i_v174705, i_v174705), 1e-300))
    i_v175110 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v174348, i_v174709), 1e-300))
    i_v175140 = (2.0 * (i_v174698 * i_v174709) - i_v174705 - 1.0) * i_v175110
    i_v175143 = i_v175140 > 0.0
    i_v175273 = jnp.where(
        i_v175143,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v175140 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v175140)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v175140,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v175140 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v175140)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v175140,
            ),
        ),
    )
    i_v175354 = i_v175273 * i_v175273
    i_v174713 = i_v174705 * i_v174709
    i_v175137 = i_v3986 * i_v174698 * i_v174709 - i_v3986 * i_v174705 + 0.5 * (i_v174348 * i_v174713)
    i_v175241 = -(i_v175140 * i_v175140) + i_v175137
    i_v175254 = -230.25850929940458 - i_v175241
    i_v175440 = (
        0.29214664 * i_v175273 + 0.26992878119627894 * i_v175354 + 0.43792457880372104 * (i_v175354 * i_v175273)
    ) * jnp.where(
        i_v175241 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v175241, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v175254 * (1.0 + 0.5 * (i_v175254 * (1.0 + i_v175254 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v175254 * (1.0 + 0.5 * (i_v175254 * (1.0 + i_v175254 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v175254 * (1.0 + 0.5 * (i_v175254 * (1.0 + i_v175254 * 0.3333333333333333))),
            ),
        ),
    )
    i_v175456 = -230.25850929940458 - i_v175137
    i_v175097 = jnp.where(i_v171873, 0.0, i_v172762)
    i_v2843 = jnp.divide(1.0, jnp.where((i_v2834 == 0.0) | jnp.logical_not(jnp.isfinite(i_v2834)), 1e-300, i_v2834))
    i_v175068 = -i_v88 * i_v2843
    i_v175100 = jnp.where(
        i_v175068 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v174348 * i_v174713 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v174348 * i_v174713)),
                1e-300,
                1.0 + i_v174348 * i_v174713,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v174348 * i_v174713, 0.0), i_v175068),
    )
    i_v188 = jnp.where(CBBTGAT > 0.0, CBBTGAT, 0.0)
    i_v3904 = jnp.divide(1.0, jnp.where((i_v3895 == 0.0) | jnp.logical_not(jnp.isfinite(i_v3895)), 1e-300, i_v3895))
    i_v176802 = i_v2843 * jnp.divide(
        (i_v69 - i_v154887) * i_v3904,
        jnp.where(
            (
                jnp.where(
                    i_v88 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v69 - i_v154887) * i_v3913, 1e-300)),
                    jnp.power(jnp.maximum((i_v69 - i_v154887) * i_v3913, 0.0), i_v88),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v88 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v69 - i_v154887) * i_v3913, 1e-300)),
                        jnp.power(jnp.maximum((i_v69 - i_v154887) * i_v3913, 0.0), i_v88),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v88 == 0.5,
                jnp.sqrt(jnp.maximum((i_v69 - i_v154887) * i_v3913, 1e-300)),
                jnp.power(jnp.maximum((i_v69 - i_v154887) * i_v3913, 0.0), i_v88),
            ),
        ),
    )
    i_v5096 = FBBTRGAT * (1.0 + STFBBTGAT * i_v4637)
    i_v5121 = jnp.where(i_v5096 > 0.0, i_v5096, 0.0)
    i_v177163 = jnp.divide(-i_v5121, jnp.where((i_v176802 == 0.0) | jnp.logical_not(jnp.isfinite(i_v176802)), 1e-300, i_v176802))
    i_v177190 = -230.25850929940458 - i_v177163
    i_v177211 = i_v177163 - 230.25850929940458
    i_v6412 = jnp.where(VBRGAT > 0.1, VBRGAT, 0.1) * (1.0 + i_v4637 * (STVBRGAT1 + i_v4637 * STVBRGAT2))
    i_v6430 = i_v6412 <= 0.1
    i_v53512 = jnp.where(i_v6430, 0.1, i_v6412)
    i_v225 = jnp.where(PBRGAT > 0.1, PBRGAT, 0.1)
    i_v6575 = jnp.where(
        i_v6430, 10.0, jnp.divide(1.0, jnp.where((i_v6412 == 0.0) | jnp.logical_not(jnp.isfinite(i_v6412)), 1e-300, i_v6412))
    )
    i_v178528 = i_v156214 * i_v6575
    i_v178529 = i_v178528 < 0.0
    i_v178567 = i_v156214 * i_v6575
    i_v5558 = 1.0 - jnp.power(jnp.maximum(i_v5226, 0.0), i_v225)
    i_v5559 = jnp.divide(1.0, jnp.where((i_v5558 == 0.0) | jnp.logical_not(jnp.isfinite(i_v5558)), 1e-300, i_v5558))
    i_v6576 = -(i_v5559 * i_v5559 * jnp.power(jnp.maximum(i_v5226, 0.0), i_v225 - 1.0)) * i_v225 * i_v6575
    i_v180236 = (
        i_v7324
        * jnp.where(
            i_v8640,
            0.0,
            (
                i_v1474
                * jnp.where(
                    i_v138867,
                    jnp.where(
                        i_v141030,
                        jnp.where(
                            jnp.where(i_v142428 < 0.0, -i_v142428, i_v142428) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v142428, -709.0, 709.0)),
                            jnp.where(
                                i_v142428 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v142488 * (1.0 + 0.5 * (i_v142488 * (1.0 + i_v142488 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v142488 * (1.0 + 0.5 * (i_v142488 * (1.0 + i_v142488 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v142488 * (1.0 + 0.5 * (i_v142488 * (1.0 + i_v142488 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v142527 * (1.0 + 0.5 * (i_v142527 * (1.0 + i_v142527 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v144309
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v144768 - i_v7952 * i_v144918,
                                        jnp.where(
                                            (i_v144768 * i_v144768 == 0.0) | jnp.logical_not(jnp.isfinite(i_v144768 * i_v144768)),
                                            1e-300,
                                            i_v144768 * i_v144768,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v7258 * i_v144918,
                                        jnp.where((i_v144781 == 0.0) | jnp.logical_not(jnp.isfinite(i_v144781)), 1e-300, i_v144781),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v144784 < 0.0, -i_v144784, i_v144784) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v144784, -709.0, 709.0)),
                            jnp.where(
                                i_v144784 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v144844 * (1.0 + 0.5 * (i_v144844 * (1.0 + i_v144844 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v144844 * (1.0 + 0.5 * (i_v144844 * (1.0 + i_v144844 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v144844 * (1.0 + 0.5 * (i_v144844 * (1.0 + i_v144844 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v144883 * (1.0 + 0.5 * (i_v144883 * (1.0 + i_v144883 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v150254, 0.0, i_v118 * (i_v152112 * i_v151005))
                + jnp.where(
                    i_v145 == 0.0,
                    0.0,
                    i_v145
                    * (
                        jnp.where(i_v150254, 0.0, i_v152112)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3979
                                * jnp.where(
                                    i_v153325,
                                    i_v154018,
                                    2.0
                                    * jnp.where(
                                        i_v153319 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v153319, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v154034 * (1.0 + 0.5 * (i_v154034 * (1.0 + i_v154034 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v154034
                                                        * (1.0 + 0.5 * (i_v154034 * (1.0 + i_v154034 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v154034 * (1.0 + 0.5 * (i_v154034 * (1.0 + i_v154034 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v154018,
                                ),
                                i_v153292,
                            )
                        )
                        * jnp.divide(
                            i_v153279 * i_v153282,
                            jnp.where(
                                (i_v153279 + i_v153282 == 0.0) | jnp.logical_not(jnp.isfinite(i_v153279 + i_v153282)),
                                1e-300,
                                i_v153279 + i_v153282,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v176 == 0.0,
                    0.0,
                    i_v176
                    * (
                        0.1
                        * i_v155344
                        * i_v155344
                        * jnp.where(
                            jnp.where(i_v155705 < 0.0, -i_v155705, i_v155705) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v155705, -709.0, 709.0)),
                            jnp.where(
                                i_v155705 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v155732 * (1.0 + 0.5 * (i_v155732 * (1.0 + i_v155732 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v155732 * (1.0 + 0.5 * (i_v155732 * (1.0 + i_v155732 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v155732 * (1.0 + 0.5 * (i_v155732 * (1.0 + i_v155732 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v155753 * (1.0 + 0.5 * (i_v155753 * (1.0 + i_v155753 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v26813 > 1000.0,
                1.0,
                jnp.where(
                    i_v156214 > -i_v5226 * i_v26813,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v217 == 4.0,
                                    jnp.where(i_v157192, -i_v157191, i_v157191)
                                    * jnp.where(i_v157192, -i_v157191, i_v157191)
                                    * jnp.where(i_v157192, -i_v157191, i_v157191)
                                    * jnp.where(i_v157192, -i_v157191, i_v157191),
                                    jnp.power(jnp.maximum(jnp.where(i_v157230 < 0.0, -i_v157230, i_v157230), 0.0), i_v217),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v217 == 4.0,
                                        jnp.where(i_v157192, -i_v157191, i_v157191)
                                        * jnp.where(i_v157192, -i_v157191, i_v157191)
                                        * jnp.where(i_v157192, -i_v157191, i_v157191)
                                        * jnp.where(i_v157192, -i_v157191, i_v157191),
                                        jnp.power(jnp.maximum(jnp.where(i_v157230 < 0.0, -i_v157230, i_v157230), 0.0), i_v217),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v217 == 4.0,
                                jnp.where(i_v157192, -i_v157191, i_v157191)
                                * jnp.where(i_v157192, -i_v157191, i_v157191)
                                * jnp.where(i_v157192, -i_v157191, i_v157191)
                                * jnp.where(i_v157192, -i_v157191, i_v157191),
                                jnp.power(jnp.maximum(jnp.where(i_v157230 < 0.0, -i_v157230, i_v157230), 0.0), i_v217),
                            ),
                        ),
                    ),
                    i_v5340 + (i_v156214 + i_v5226 * i_v26813) * i_v6517,
                ),
            ),
        )
        + i_v7338
        * jnp.where(
            i_v8686,
            0.0,
            (
                i_v1624
                * jnp.where(
                    i_v138867,
                    jnp.where(
                        i_v141030,
                        jnp.where(
                            jnp.where(i_v143323 < 0.0, -i_v143323, i_v143323) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v143323, -709.0, 709.0)),
                            jnp.where(
                                i_v143323 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v143383 * (1.0 + 0.5 * (i_v143383 * (1.0 + i_v143383 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v143383 * (1.0 + 0.5 * (i_v143383 * (1.0 + i_v143383 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v143383 * (1.0 + 0.5 * (i_v143383 * (1.0 + i_v143383 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v143422 * (1.0 + 0.5 * (i_v143422 * (1.0 + i_v143422 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v144309
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v145102 - i_v7952 * i_v145252,
                                        jnp.where(
                                            (i_v145102 * i_v145102 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145102 * i_v145102)),
                                            1e-300,
                                            i_v145102 * i_v145102,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v145023 * i_v145252,
                                        jnp.where((i_v145115 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145115)), 1e-300, i_v145115),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v145118 < 0.0, -i_v145118, i_v145118) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v145118, -709.0, 709.0)),
                            jnp.where(
                                i_v145118 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v145178 * (1.0 + 0.5 * (i_v145178 * (1.0 + i_v145178 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v145178 * (1.0 + 0.5 * (i_v145178 * (1.0 + i_v145178 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v145178 * (1.0 + 0.5 * (i_v145178 * (1.0 + i_v145178 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v145217 * (1.0 + 0.5 * (i_v145217 * (1.0 + i_v145217 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v161128, 0.0, i_v124 * (i_v163185 * i_v162017))
                + jnp.where(
                    i_v151 == 0.0,
                    0.0,
                    i_v151
                    * (
                        jnp.where(i_v161128, 0.0, i_v163185)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3984
                                * jnp.where(
                                    i_v164398,
                                    i_v164695,
                                    2.0
                                    * jnp.where(
                                        i_v164392 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v164392, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v164711 * (1.0 + 0.5 * (i_v164711 * (1.0 + i_v164711 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v164711
                                                        * (1.0 + 0.5 * (i_v164711 * (1.0 + i_v164711 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v164711 * (1.0 + 0.5 * (i_v164711 * (1.0 + i_v164711 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v164695,
                                ),
                                i_v164365,
                            )
                        )
                        * jnp.divide(
                            i_v164352 * i_v164355,
                            jnp.where(
                                (i_v164352 + i_v164355 == 0.0) | jnp.logical_not(jnp.isfinite(i_v164352 + i_v164355)),
                                1e-300,
                                i_v164352 + i_v164355,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v182 == 0.0,
                    0.0,
                    i_v182
                    * (
                        0.1
                        * i_v166057
                        * i_v166057
                        * jnp.where(
                            jnp.where(i_v166418 < 0.0, -i_v166418, i_v166418) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v166418, -709.0, 709.0)),
                            jnp.where(
                                i_v166418 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v166445 * (1.0 + 0.5 * (i_v166445 * (1.0 + i_v166445 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v166445 * (1.0 + 0.5 * (i_v166445 * (1.0 + i_v166445 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v166445 * (1.0 + 0.5 * (i_v166445 * (1.0 + i_v166445 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v166466 * (1.0 + 0.5 * (i_v166466 * (1.0 + i_v166466 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v39291 > 1000.0,
                1.0,
                jnp.where(
                    i_v156214 > -i_v5226 * i_v39291,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v221 == 4.0,
                                    jnp.where(i_v167784, -i_v167783, i_v167783)
                                    * jnp.where(i_v167784, -i_v167783, i_v167783)
                                    * jnp.where(i_v167784, -i_v167783, i_v167783)
                                    * jnp.where(i_v167784, -i_v167783, i_v167783),
                                    jnp.power(jnp.maximum(jnp.where(i_v167822 < 0.0, -i_v167822, i_v167822), 0.0), i_v221),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v221 == 4.0,
                                        jnp.where(i_v167784, -i_v167783, i_v167783)
                                        * jnp.where(i_v167784, -i_v167783, i_v167783)
                                        * jnp.where(i_v167784, -i_v167783, i_v167783)
                                        * jnp.where(i_v167784, -i_v167783, i_v167783),
                                        jnp.power(jnp.maximum(jnp.where(i_v167822 < 0.0, -i_v167822, i_v167822), 0.0), i_v221),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v221 == 4.0,
                                jnp.where(i_v167784, -i_v167783, i_v167783)
                                * jnp.where(i_v167784, -i_v167783, i_v167783)
                                * jnp.where(i_v167784, -i_v167783, i_v167783)
                                * jnp.where(i_v167784, -i_v167783, i_v167783),
                                jnp.power(jnp.maximum(jnp.where(i_v167822 < 0.0, -i_v167822, i_v167822), 0.0), i_v221),
                            ),
                        ),
                    ),
                    i_v5451 + (i_v156214 + i_v5226 * i_v39291) * i_v6548,
                ),
            ),
        )
        + i_v7352
        * jnp.where(
            i_v8764,
            0.0,
            (
                i_v1771
                * jnp.where(
                    i_v138867,
                    jnp.where(
                        i_v141030,
                        jnp.where(
                            jnp.where(i_v144182 < 0.0, -i_v144182, i_v144182) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v144182, -709.0, 709.0)),
                            jnp.where(
                                i_v144182 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v144242 * (1.0 + 0.5 * (i_v144242 * (1.0 + i_v144242 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v144242 * (1.0 + 0.5 * (i_v144242 * (1.0 + i_v144242 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v144242 * (1.0 + 0.5 * (i_v144242 * (1.0 + i_v144242 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v144281 * (1.0 + 0.5 * (i_v144281 * (1.0 + i_v144281 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v144309
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v145472 - i_v7952 * i_v145622,
                                        jnp.where(
                                            (i_v145472 * i_v145472 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145472 * i_v145472)),
                                            1e-300,
                                            i_v145472 * i_v145472,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v145393 * i_v145622,
                                        jnp.where((i_v145485 == 0.0) | jnp.logical_not(jnp.isfinite(i_v145485)), 1e-300, i_v145485),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v145488 < 0.0, -i_v145488, i_v145488) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v145488, -709.0, 709.0)),
                            jnp.where(
                                i_v145488 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v145548 * (1.0 + 0.5 * (i_v145548 * (1.0 + i_v145548 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v145548 * (1.0 + 0.5 * (i_v145548 * (1.0 + i_v145548 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v145548 * (1.0 + 0.5 * (i_v145548 * (1.0 + i_v145548 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v145587 * (1.0 + 0.5 * (i_v145587 * (1.0 + i_v145587 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v171873, 0.0, i_v130 * (i_v173930 * i_v172762))
                + jnp.where(
                    i_v157 == 0.0,
                    0.0,
                    i_v157
                    * (
                        jnp.where(i_v171873, 0.0, i_v173930)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3986
                                * jnp.where(
                                    i_v175143,
                                    i_v175440,
                                    2.0
                                    * jnp.where(
                                        i_v175137 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v175137, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v175456 * (1.0 + 0.5 * (i_v175456 * (1.0 + i_v175456 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v175456
                                                        * (1.0 + 0.5 * (i_v175456 * (1.0 + i_v175456 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v175456 * (1.0 + 0.5 * (i_v175456 * (1.0 + i_v175456 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v175440,
                                ),
                                i_v175110,
                            )
                        )
                        * jnp.divide(
                            i_v175097 * i_v175100,
                            jnp.where(
                                (i_v175097 + i_v175100 == 0.0) | jnp.logical_not(jnp.isfinite(i_v175097 + i_v175100)),
                                1e-300,
                                i_v175097 + i_v175100,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v188 == 0.0,
                    0.0,
                    i_v188
                    * (
                        0.1
                        * i_v176802
                        * i_v176802
                        * jnp.where(
                            jnp.where(i_v177163 < 0.0, -i_v177163, i_v177163) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v177163, -709.0, 709.0)),
                            jnp.where(
                                i_v177163 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v177190 * (1.0 + 0.5 * (i_v177190 * (1.0 + i_v177190 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v177190 * (1.0 + 0.5 * (i_v177190 * (1.0 + i_v177190 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v177190 * (1.0 + 0.5 * (i_v177190 * (1.0 + i_v177190 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v177211 * (1.0 + 0.5 * (i_v177211 * (1.0 + i_v177211 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v53512 > 1000.0,
                1.0,
                jnp.where(
                    i_v156214 > -i_v5226 * i_v53512,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v225 == 4.0,
                                    jnp.where(i_v178529, -i_v178528, i_v178528)
                                    * jnp.where(i_v178529, -i_v178528, i_v178528)
                                    * jnp.where(i_v178529, -i_v178528, i_v178528)
                                    * jnp.where(i_v178529, -i_v178528, i_v178528),
                                    jnp.power(jnp.maximum(jnp.where(i_v178567 < 0.0, -i_v178567, i_v178567), 0.0), i_v225),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v225 == 4.0,
                                        jnp.where(i_v178529, -i_v178528, i_v178528)
                                        * jnp.where(i_v178529, -i_v178528, i_v178528)
                                        * jnp.where(i_v178529, -i_v178528, i_v178528)
                                        * jnp.where(i_v178529, -i_v178528, i_v178528),
                                        jnp.power(jnp.maximum(jnp.where(i_v178567 < 0.0, -i_v178567, i_v178567), 0.0), i_v225),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v225 == 4.0,
                                jnp.where(i_v178529, -i_v178528, i_v178528)
                                * jnp.where(i_v178529, -i_v178528, i_v178528)
                                * jnp.where(i_v178529, -i_v178528, i_v178528)
                                * jnp.where(i_v178529, -i_v178528, i_v178528),
                                jnp.power(jnp.maximum(jnp.where(i_v178567 < 0.0, -i_v178567, i_v178567), 0.0), i_v225),
                            ),
                        ),
                    ),
                    i_v5559 + (i_v156214 + i_v5226 * i_v53512) * i_v6576,
                ),
            ),
        )
    )
    i_v180251 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v182780 = i_v7952 > 0.2
    i_v184103 = NJH - (NJDV * (0.2 - i_v7258) + i_v277) - 0.01
    i_v184105 = 4.0 * NJH * 0.01
    i_v184124 = (
        NJH
        - 0.5
        * (i_v184103 + jnp.sqrt(jnp.maximum(i_v184103 * i_v184103 + jnp.where(i_v184105 > 0.0, i_v184105, -i_v184105), 1e-300)))
        - i_v277
        - 0.01
    )
    i_v184126 = 4.0 * i_v277 * 0.01
    i_v184162 = jnp.where(
        i_v277 < NJH,
        i_v277
        + 0.5
        * (i_v184124 + jnp.sqrt(jnp.maximum(i_v184124 * i_v184124 + jnp.where(i_v184126 > 0.0, i_v184126, -i_v184126), 1e-300))),
        i_v277,
    )
    i_v184178 = i_v596 * (
        jnp.divide(0.2, jnp.where((i_v184162 == 0.0) | jnp.logical_not(jnp.isfinite(i_v184162)), 1e-300, i_v184162))
        + jnp.divide(
            i_v7258 * (i_v184162 - i_v277),
            jnp.where((i_v277 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v277 * NJH)), 1e-300, i_v277 * NJH),
        )
    )
    i_v184238 = -230.25850929940458 - i_v184178
    i_v184277 = i_v184178 - 230.25850929940458
    i_v186059 = 0.2 - i_v7952
    i_v186440 = i_v277 < NJH
    i_v186448 = NJH - (NJDV * (i_v7952 - i_v7258) + i_v277) - 0.01
    i_v186450 = 4.0 * NJH * 0.01
    i_v186461 = jnp.sqrt(jnp.maximum(i_v186448 * i_v186448 + jnp.where(i_v186450 > 0.0, i_v186450, -i_v186450), 1e-300))
    i_v186472 = NJH - 0.5 * (i_v186448 + i_v186461) - i_v277 - 0.01
    i_v186474 = 4.0 * i_v277 * 0.01
    i_v186485 = jnp.sqrt(jnp.maximum(i_v186472 * i_v186472 + jnp.where(i_v186474 > 0.0, i_v186474, -i_v186474), 1e-300))
    i_v186518, i_v186668 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v186440, _t, _f),
        (
            i_v277 + 0.5 * (i_v186472 + i_v186485),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v186448, i_v186461))) * (0.5 * (1.0 + jnp.divide(i_v186472, i_v186485))),
        ),
        (i_v277, 0.0),
    )
    i_v186531 = i_v277 * NJH
    i_v186534 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v186518 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186518)), 1e-300, i_v186518))
        + jnp.divide(
            i_v7258 * (i_v186518 - i_v277),
            jnp.where((i_v186531 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186531)), 1e-300, i_v186531),
        )
    )
    i_v186594 = -230.25850929940458 - i_v186534
    i_v186633 = i_v186534 - 230.25850929940458
    i_v191302 = jnp.where(
        i_v180251,
        jnp.where(
            i_v182780,
            jnp.where(
                jnp.where(i_v184178 < 0.0, -i_v184178, i_v184178) < 230.25850929940458,
                jnp.exp(jnp.clip(i_v184178, -709.0, 709.0)),
                jnp.where(
                    i_v184178 < -230.25850929940458,
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            (1.0 + i_v184238 * (1.0 + 0.5 * (i_v184238 * (1.0 + i_v184238 * 0.3333333333333333))) == 0.0)
                            | jnp.logical_not(
                                jnp.isfinite(1.0 + i_v184238 * (1.0 + 0.5 * (i_v184238 * (1.0 + i_v184238 * 0.3333333333333333))))
                            ),
                            1e-300,
                            1.0 + i_v184238 * (1.0 + 0.5 * (i_v184238 * (1.0 + i_v184238 * 0.3333333333333333))),
                        ),
                    ),
                    1e100 * (1.0 + i_v184277 * (1.0 + 0.5 * (i_v184277 * (1.0 + i_v184277 * 0.3333333333333333)))),
                ),
            ),
            (
                1.0
                + i_v186059
                * (
                    i_v596
                    * (
                        jnp.divide(
                            i_v186518 - i_v7952 * i_v186668,
                            jnp.where(
                                (i_v186518 * i_v186518 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186518 * i_v186518)),
                                1e-300,
                                i_v186518 * i_v186518,
                            ),
                        )
                        + jnp.divide(
                            i_v7258 * i_v186668,
                            jnp.where((i_v186531 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186531)), 1e-300, i_v186531),
                        )
                    )
                )
            )
            * jnp.where(
                jnp.where(i_v186534 < 0.0, -i_v186534, i_v186534) < 230.25850929940458,
                jnp.exp(jnp.clip(i_v186534, -709.0, 709.0)),
                jnp.where(
                    i_v186534 < -230.25850929940458,
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            (1.0 + i_v186594 * (1.0 + 0.5 * (i_v186594 * (1.0 + i_v186594 * 0.3333333333333333))) == 0.0)
                            | jnp.logical_not(
                                jnp.isfinite(1.0 + i_v186594 * (1.0 + 0.5 * (i_v186594 * (1.0 + i_v186594 * 0.3333333333333333))))
                            ),
                            1e-300,
                            1.0 + i_v186594 * (1.0 + 0.5 * (i_v186594 * (1.0 + i_v186594 * 0.3333333333333333))),
                        ),
                    ),
                    1e100 * (1.0 + i_v186633 * (1.0 + 0.5 * (i_v186633 * (1.0 + i_v186633 * 0.3333333333333333)))),
                ),
            ),
        )
        - 1.0,
        0.0,
    )
    i_v192004 = jnp.where(i_v118 == 0.0, i_v145 == 0.0, False)
    i_v183033 = 0.5 * (0.2 * i_v596)
    i_v183060 = -230.25850929940458 - i_v183033
    i_v183081 = i_v183033 - 230.25850929940458
    i_v187537 = jnp.where(
        i_v182780,
        jnp.where(
            jnp.where(i_v183033 < 0.0, -i_v183033, i_v183033) < 230.25850929940458,
            jnp.exp(jnp.clip(i_v183033, -709.0, 709.0)),
            jnp.where(
                i_v183033 < -230.25850929940458,
                jnp.divide(
                    1e-100,
                    jnp.where(
                        (1.0 + i_v183060 * (1.0 + 0.5 * (i_v183060 * (1.0 + i_v183060 * 0.3333333333333333))) == 0.0)
                        | jnp.logical_not(
                            jnp.isfinite(1.0 + i_v183060 * (1.0 + 0.5 * (i_v183060 * (1.0 + i_v183060 * 0.3333333333333333))))
                        ),
                        1e-300,
                        1.0 + i_v183060 * (1.0 + 0.5 * (i_v183060 * (1.0 + i_v183060 * 0.3333333333333333))),
                    ),
                ),
                1e100 * (1.0 + i_v183081 * (1.0 + 0.5 * (i_v183081 * (1.0 + i_v183081 * 0.3333333333333333)))),
            ),
        ),
        jnp.sqrt(jnp.maximum((1.0 + i_v186059 * i_v596) * i_v15213, 1e-300)),
    )
    i_v193852 = jnp.where(i_v180251, i_v187537, 0.0)
    i_v192736 = i_v76 == 0.5
    i_v187648 = jnp.divide(1.0, jnp.where((i_v187537 == 0.0) | jnp.logical_not(jnp.isfinite(i_v187537)), 1e-300, i_v187537))
    i_v188027 = 2.0 * (
        i_v594
        * jnp.log(jnp.maximum(2.0 + i_v187648 + jnp.sqrt(jnp.maximum((i_v187648 + 1.0) * (i_v187648 + 3.0), 1e-300)), 1e-300))
    )
    i_v188416 = i_v8881 - i_v188027
    i_v192382 = jnp.where(
        i_v180251,
        0.5 * (0.2 + i_v188416 - jnp.sqrt(jnp.maximum((0.2 - i_v188416) * (0.2 - i_v188416) + 4.0 * i_v594 * i_v594, 1e-300))),
        0.0,
    )
    i_v192385 = i_v2324 - i_v192382
    i_v193475 = i_v3628 * jnp.where(
        i_v192736, jnp.sqrt(jnp.maximum(i_v192385 * i_v3907, 1e-300)), jnp.power(jnp.maximum(i_v192385 * i_v3907, 0.0), i_v76)
    )
    i_v193862 = i_v1108 * ((i_v193852 - 1.0) * i_v193475)
    i_v192394 = jnp.where(i_v180251, i_v188027, 0.0)
    i_v192399 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v192394, jnp.where((i_v192385 == 0.0) | jnp.logical_not(jnp.isfinite(i_v192385)), 1e-300, i_v192385)),
            1e-300,
        )
    )
    i_v192400 = 1.0 - i_v192399
    i_v192755 = i_v192400 + jnp.where(
        i_v192736,
        0.0,
        (jnp.divide(i_v192400 * i_v192400 * jnp.log(jnp.maximum(i_v192400, 1e-300)), i_v192399) + i_v192400) * (1.0 - 2.0 * i_v76),
    )
    i_v194280 = i_v4127 * jnp.divide(
        jnp.where(i_v192004, 0.0, i_v193475) * i_v2516,
        jnp.where(
            (jnp.where(i_v192004, 0.0, i_v192385) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v192004, 0.0, i_v192385))),
            1e-300,
            jnp.where(i_v192004, 0.0, i_v192385),
        ),
    )
    i_v194630 = jnp.divide(
        0.666666666666667 * i_v3979, jnp.where((i_v194280 == 0.0) | jnp.logical_not(jnp.isfinite(i_v194280)), 1e-300, i_v194280)
    )
    i_v194631 = i_v194630 * i_v194630
    i_v194632 = i_v194631 * i_v194631
    i_v194637 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v194632,
                jnp.where((i_v194632 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v194632 + 1.0)), 1e-300, i_v194632 + 1.0),
            ),
            1e-300,
        )
    )
    i_v194641 = jnp.sqrt(jnp.maximum(jnp.where(i_v194637 < 0.0, -i_v194637, i_v194637), 1e-300))
    i_v195042 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v194280, i_v194641), 1e-300))
    i_v195072 = (2.0 * (i_v194630 * i_v194641) - i_v194637 - 1.0) * i_v195042
    i_v195075 = i_v195072 > 0.0
    i_v195337 = jnp.where(
        i_v195075,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v195072 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v195072)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v195072,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v195072 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v195072)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v195072,
            ),
        ),
    )
    i_v195550 = i_v195337 * i_v195337
    i_v194645 = i_v194637 * i_v194641
    i_v195069 = i_v3979 * i_v194630 * i_v194641 - i_v3979 * i_v194637 + 0.5 * (i_v194280 * i_v194645)
    i_v195305 = -(i_v195072 * i_v195072) + i_v195069
    i_v195318 = -230.25850929940458 - i_v195305
    i_v195768 = (
        0.29214664 * i_v195337 + 0.26992878119627894 * i_v195550 + 0.43792457880372104 * (i_v195550 * i_v195337)
    ) * jnp.where(
        i_v195305 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v195305, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v195318 * (1.0 + 0.5 * (i_v195318 * (1.0 + i_v195318 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v195318 * (1.0 + 0.5 * (i_v195318 * (1.0 + i_v195318 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v195318 * (1.0 + 0.5 * (i_v195318 * (1.0 + i_v195318 * 0.3333333333333333))),
            ),
        ),
    )
    i_v195784 = -230.25850929940458 - i_v195069
    i_v195029 = jnp.where(i_v192004, 0.0, i_v192755)
    i_v195000 = -i_v76 * i_v2837
    i_v195032 = jnp.where(
        i_v195000 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v194280 * i_v194645 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v194280 * i_v194645)),
                1e-300,
                1.0 + i_v194280 * i_v194645,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v194280 * i_v194645, 0.0), i_v195000),
    )
    i_v196637 = jnp.where(
        i_v180251,
        0.5 * (0.2 + i_v9052 - jnp.sqrt(jnp.maximum((0.2 - i_v9052) * (0.2 - i_v9052) + 4.0 * i_v590 * i_v590, 1e-300))),
        0.0,
    )
    i_v197094 = i_v2837 * jnp.divide(
        (i_v61 - i_v196637) * i_v3898,
        jnp.where(
            (
                jnp.where(
                    i_v76 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v61 - i_v196637) * i_v3907, 1e-300)),
                    jnp.power(jnp.maximum((i_v61 - i_v196637) * i_v3907, 0.0), i_v76),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v76 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v61 - i_v196637) * i_v3907, 1e-300)),
                        jnp.power(jnp.maximum((i_v61 - i_v196637) * i_v3907, 0.0), i_v76),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v76 == 0.5,
                jnp.sqrt(jnp.maximum((i_v61 - i_v196637) * i_v3907, 1e-300)),
                jnp.power(jnp.maximum((i_v61 - i_v196637) * i_v3907, 0.0), i_v76),
            ),
        ),
    )
    i_v197455 = jnp.divide(-i_v5102, jnp.where((i_v197094 == 0.0) | jnp.logical_not(jnp.isfinite(i_v197094)), 1e-300, i_v197094))
    i_v197482 = -230.25850929940458 - i_v197455
    i_v197503 = i_v197455 - 230.25850929940458
    i_v197964 = jnp.where(i_v180251, -5.000000413701855e-12, 0.0)
    i_v198941 = i_v197964 * i_v6514
    i_v198942 = i_v198941 < 0.0
    i_v198980 = i_v197964 * i_v6514
    i_v184987 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v184998 = NJH - (NJDV * (0.2 - i_v184987) + i_v281) - 0.01
    i_v185000 = 4.0 * NJH * 0.01
    i_v185019 = (
        NJH
        - 0.5
        * (i_v184998 + jnp.sqrt(jnp.maximum(i_v184998 * i_v184998 + jnp.where(i_v185000 > 0.0, i_v185000, -i_v185000), 1e-300)))
        - i_v281
        - 0.01
    )
    i_v185021 = 4.0 * i_v281 * 0.01
    i_v185057 = jnp.where(
        i_v281 < NJH,
        i_v281
        + 0.5
        * (i_v185019 + jnp.sqrt(jnp.maximum(i_v185019 * i_v185019 + jnp.where(i_v185021 > 0.0, i_v185021, -i_v185021), 1e-300))),
        i_v281,
    )
    i_v185073 = i_v596 * (
        jnp.divide(0.2, jnp.where((i_v185057 == 0.0) | jnp.logical_not(jnp.isfinite(i_v185057)), 1e-300, i_v185057))
        + jnp.divide(
            i_v184987 * (i_v185057 - i_v281),
            jnp.where((i_v281 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v281 * NJH)), 1e-300, i_v281 * NJH),
        )
    )
    i_v185133 = -230.25850929940458 - i_v185073
    i_v185172 = i_v185073 - 230.25850929940458
    i_v186774 = i_v281 < NJH
    i_v186773 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v186782 = NJH - (NJDV * (i_v7952 - i_v186773) + i_v281) - 0.01
    i_v186784 = 4.0 * NJH * 0.01
    i_v186795 = jnp.sqrt(jnp.maximum(i_v186782 * i_v186782 + jnp.where(i_v186784 > 0.0, i_v186784, -i_v186784), 1e-300))
    i_v186806 = NJH - 0.5 * (i_v186782 + i_v186795) - i_v281 - 0.01
    i_v186808 = 4.0 * i_v281 * 0.01
    i_v186819 = jnp.sqrt(jnp.maximum(i_v186806 * i_v186806 + jnp.where(i_v186808 > 0.0, i_v186808, -i_v186808), 1e-300))
    i_v186852, i_v187002 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v186774, _t, _f),
        (
            i_v281 + 0.5 * (i_v186806 + i_v186819),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v186782, i_v186795))) * (0.5 * (1.0 + jnp.divide(i_v186806, i_v186819))),
        ),
        (i_v281, 0.0),
    )
    i_v186865 = i_v281 * NJH
    i_v186868 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v186852 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186852)), 1e-300, i_v186852))
        + jnp.divide(
            i_v186773 * (i_v186852 - i_v281),
            jnp.where((i_v186865 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186865)), 1e-300, i_v186865),
        )
    )
    i_v186928 = -230.25850929940458 - i_v186868
    i_v186967 = i_v186868 - 230.25850929940458
    i_v202878 = jnp.where(i_v124 == 0.0, i_v151 == 0.0, False)
    i_v203748 = i_v82 == 0.5
    i_v203328 = i_v2333 - i_v192382
    i_v204487 = i_v3763 * jnp.where(
        i_v203748, jnp.sqrt(jnp.maximum(i_v203328 * i_v3910, 1e-300)), jnp.power(jnp.maximum(i_v203328 * i_v3910, 0.0), i_v82)
    )
    i_v204935 = i_v1117 * ((i_v193852 - 1.0) * i_v204487)
    i_v203411 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v192394, jnp.where((i_v203328 == 0.0) | jnp.logical_not(jnp.isfinite(i_v203328)), 1e-300, i_v203328)),
            1e-300,
        )
    )
    i_v203412 = 1.0 - i_v203411
    i_v203767 = i_v203412 + jnp.where(
        i_v203748,
        0.0,
        (jnp.divide(i_v203412 * i_v203412 * jnp.log(jnp.maximum(i_v203412, 1e-300)), i_v203411) + i_v203412) * (1.0 - 2.0 * i_v82),
    )
    i_v205353 = i_v4262 * jnp.divide(
        jnp.where(i_v202878, 0.0, i_v204487) * i_v2678,
        jnp.where(
            (jnp.where(i_v202878, 0.0, i_v203328) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v202878, 0.0, i_v203328))),
            1e-300,
            jnp.where(i_v202878, 0.0, i_v203328),
        ),
    )
    i_v205703 = jnp.divide(
        0.666666666666667 * i_v3984, jnp.where((i_v205353 == 0.0) | jnp.logical_not(jnp.isfinite(i_v205353)), 1e-300, i_v205353)
    )
    i_v205704 = i_v205703 * i_v205703
    i_v205705 = i_v205704 * i_v205704
    i_v205710 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v205705,
                jnp.where((i_v205705 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v205705 + 1.0)), 1e-300, i_v205705 + 1.0),
            ),
            1e-300,
        )
    )
    i_v205714 = jnp.sqrt(jnp.maximum(jnp.where(i_v205710 < 0.0, -i_v205710, i_v205710), 1e-300))
    i_v206115 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v205353, i_v205714), 1e-300))
    i_v206145 = (2.0 * (i_v205703 * i_v205714) - i_v205710 - 1.0) * i_v206115
    i_v206148 = i_v206145 > 0.0
    i_v206278 = jnp.where(
        i_v206148,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v206145 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v206145)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v206145,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v206145 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v206145)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v206145,
            ),
        ),
    )
    i_v206359 = i_v206278 * i_v206278
    i_v205718 = i_v205710 * i_v205714
    i_v206142 = i_v3984 * i_v205703 * i_v205714 - i_v3984 * i_v205710 + 0.5 * (i_v205353 * i_v205718)
    i_v206246 = -(i_v206145 * i_v206145) + i_v206142
    i_v206259 = -230.25850929940458 - i_v206246
    i_v206445 = (
        0.29214664 * i_v206278 + 0.26992878119627894 * i_v206359 + 0.43792457880372104 * (i_v206359 * i_v206278)
    ) * jnp.where(
        i_v206246 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v206246, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v206259 * (1.0 + 0.5 * (i_v206259 * (1.0 + i_v206259 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v206259 * (1.0 + 0.5 * (i_v206259 * (1.0 + i_v206259 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v206259 * (1.0 + 0.5 * (i_v206259 * (1.0 + i_v206259 * 0.3333333333333333))),
            ),
        ),
    )
    i_v206461 = -230.25850929940458 - i_v206142
    i_v206102 = jnp.where(i_v202878, 0.0, i_v203767)
    i_v206073 = -i_v82 * i_v2840
    i_v206105 = jnp.where(
        i_v206073 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v205353 * i_v205718 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v205353 * i_v205718)),
                1e-300,
                1.0 + i_v205353 * i_v205718,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v205353 * i_v205718, 0.0), i_v206073),
    )
    i_v207807 = i_v2840 * jnp.divide(
        (i_v65 - i_v196637) * i_v3901,
        jnp.where(
            (
                jnp.where(
                    i_v82 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v65 - i_v196637) * i_v3910, 1e-300)),
                    jnp.power(jnp.maximum((i_v65 - i_v196637) * i_v3910, 0.0), i_v82),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v82 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v65 - i_v196637) * i_v3910, 1e-300)),
                        jnp.power(jnp.maximum((i_v65 - i_v196637) * i_v3910, 0.0), i_v82),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v82 == 0.5,
                jnp.sqrt(jnp.maximum((i_v65 - i_v196637) * i_v3910, 1e-300)),
                jnp.power(jnp.maximum((i_v65 - i_v196637) * i_v3910, 0.0), i_v82),
            ),
        ),
    )
    i_v208168 = jnp.divide(-i_v5110, jnp.where((i_v207807 == 0.0) | jnp.logical_not(jnp.isfinite(i_v207807)), 1e-300, i_v207807))
    i_v208195 = -230.25850929940458 - i_v208168
    i_v208216 = i_v208168 - 230.25850929940458
    i_v209533 = i_v197964 * i_v6546
    i_v209534 = i_v209533 < 0.0
    i_v209572 = i_v197964 * i_v6546
    i_v185846 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v185857 = NJH - (NJDV * (0.2 - i_v185846) + i_v285) - 0.01
    i_v185859 = 4.0 * NJH * 0.01
    i_v185878 = (
        NJH
        - 0.5
        * (i_v185857 + jnp.sqrt(jnp.maximum(i_v185857 * i_v185857 + jnp.where(i_v185859 > 0.0, i_v185859, -i_v185859), 1e-300)))
        - i_v285
        - 0.01
    )
    i_v185880 = 4.0 * i_v285 * 0.01
    i_v185916 = jnp.where(
        i_v285 < NJH,
        i_v285
        + 0.5
        * (i_v185878 + jnp.sqrt(jnp.maximum(i_v185878 * i_v185878 + jnp.where(i_v185880 > 0.0, i_v185880, -i_v185880), 1e-300))),
        i_v285,
    )
    i_v185932 = i_v596 * (
        jnp.divide(0.2, jnp.where((i_v185916 == 0.0) | jnp.logical_not(jnp.isfinite(i_v185916)), 1e-300, i_v185916))
        + jnp.divide(
            i_v185846 * (i_v185916 - i_v285),
            jnp.where((i_v285 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v285 * NJH)), 1e-300, i_v285 * NJH),
        )
    )
    i_v185992 = -230.25850929940458 - i_v185932
    i_v186031 = i_v185932 - 230.25850929940458
    i_v187144 = i_v285 < NJH
    i_v187143 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v187152 = NJH - (NJDV * (i_v7952 - i_v187143) + i_v285) - 0.01
    i_v187154 = 4.0 * NJH * 0.01
    i_v187165 = jnp.sqrt(jnp.maximum(i_v187152 * i_v187152 + jnp.where(i_v187154 > 0.0, i_v187154, -i_v187154), 1e-300))
    i_v187176 = NJH - 0.5 * (i_v187152 + i_v187165) - i_v285 - 0.01
    i_v187178 = 4.0 * i_v285 * 0.01
    i_v187189 = jnp.sqrt(jnp.maximum(i_v187176 * i_v187176 + jnp.where(i_v187178 > 0.0, i_v187178, -i_v187178), 1e-300))
    i_v187222, i_v187372 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v187144, _t, _f),
        (
            i_v285 + 0.5 * (i_v187176 + i_v187189),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v187152, i_v187165))) * (0.5 * (1.0 + jnp.divide(i_v187176, i_v187189))),
        ),
        (i_v285, 0.0),
    )
    i_v187235 = i_v285 * NJH
    i_v187238 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v187222 == 0.0) | jnp.logical_not(jnp.isfinite(i_v187222)), 1e-300, i_v187222))
        + jnp.divide(
            i_v187143 * (i_v187222 - i_v285),
            jnp.where((i_v187235 == 0.0) | jnp.logical_not(jnp.isfinite(i_v187235)), 1e-300, i_v187235),
        )
    )
    i_v187298 = -230.25850929940458 - i_v187238
    i_v187337 = i_v187238 - 230.25850929940458
    i_v213623 = jnp.where(i_v130 == 0.0, i_v157 == 0.0, False)
    i_v214493 = i_v88 == 0.5
    i_v214073 = i_v2342 - i_v192382
    i_v215232 = i_v3895 * jnp.where(
        i_v214493, jnp.sqrt(jnp.maximum(i_v214073 * i_v3913, 1e-300)), jnp.power(jnp.maximum(i_v214073 * i_v3913, 0.0), i_v88)
    )
    i_v215680 = i_v1126 * ((i_v193852 - 1.0) * i_v215232)
    i_v214156 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v192394, jnp.where((i_v214073 == 0.0) | jnp.logical_not(jnp.isfinite(i_v214073)), 1e-300, i_v214073)),
            1e-300,
        )
    )
    i_v214157 = 1.0 - i_v214156
    i_v214512 = i_v214157 + jnp.where(
        i_v214493,
        0.0,
        (jnp.divide(i_v214157 * i_v214157 * jnp.log(jnp.maximum(i_v214157, 1e-300)), i_v214156) + i_v214157) * (1.0 - 2.0 * i_v88),
    )
    i_v216098 = i_v4394 * jnp.divide(
        jnp.where(i_v213623, 0.0, i_v215232) * i_v2834,
        jnp.where(
            (jnp.where(i_v213623, 0.0, i_v214073) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v213623, 0.0, i_v214073))),
            1e-300,
            jnp.where(i_v213623, 0.0, i_v214073),
        ),
    )
    i_v216448 = jnp.divide(
        0.666666666666667 * i_v3986, jnp.where((i_v216098 == 0.0) | jnp.logical_not(jnp.isfinite(i_v216098)), 1e-300, i_v216098)
    )
    i_v216449 = i_v216448 * i_v216448
    i_v216450 = i_v216449 * i_v216449
    i_v216455 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v216450,
                jnp.where((i_v216450 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v216450 + 1.0)), 1e-300, i_v216450 + 1.0),
            ),
            1e-300,
        )
    )
    i_v216459 = jnp.sqrt(jnp.maximum(jnp.where(i_v216455 < 0.0, -i_v216455, i_v216455), 1e-300))
    i_v216860 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v216098, i_v216459), 1e-300))
    i_v216890 = (2.0 * (i_v216448 * i_v216459) - i_v216455 - 1.0) * i_v216860
    i_v216893 = i_v216890 > 0.0
    i_v217023 = jnp.where(
        i_v216893,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v216890 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v216890)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v216890,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v216890 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v216890)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v216890,
            ),
        ),
    )
    i_v217104 = i_v217023 * i_v217023
    i_v216463 = i_v216455 * i_v216459
    i_v216887 = i_v3986 * i_v216448 * i_v216459 - i_v3986 * i_v216455 + 0.5 * (i_v216098 * i_v216463)
    i_v216991 = -(i_v216890 * i_v216890) + i_v216887
    i_v217004 = -230.25850929940458 - i_v216991
    i_v217190 = (
        0.29214664 * i_v217023 + 0.26992878119627894 * i_v217104 + 0.43792457880372104 * (i_v217104 * i_v217023)
    ) * jnp.where(
        i_v216991 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v216991, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v217004 * (1.0 + 0.5 * (i_v217004 * (1.0 + i_v217004 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v217004 * (1.0 + 0.5 * (i_v217004 * (1.0 + i_v217004 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v217004 * (1.0 + 0.5 * (i_v217004 * (1.0 + i_v217004 * 0.3333333333333333))),
            ),
        ),
    )
    i_v217206 = -230.25850929940458 - i_v216887
    i_v216847 = jnp.where(i_v213623, 0.0, i_v214512)
    i_v216818 = -i_v88 * i_v2843
    i_v216850 = jnp.where(
        i_v216818 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v216098 * i_v216463 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v216098 * i_v216463)),
                1e-300,
                1.0 + i_v216098 * i_v216463,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v216098 * i_v216463, 0.0), i_v216818),
    )
    i_v218552 = i_v2843 * jnp.divide(
        (i_v69 - i_v196637) * i_v3904,
        jnp.where(
            (
                jnp.where(
                    i_v88 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v69 - i_v196637) * i_v3913, 1e-300)),
                    jnp.power(jnp.maximum((i_v69 - i_v196637) * i_v3913, 0.0), i_v88),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v88 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v69 - i_v196637) * i_v3913, 1e-300)),
                        jnp.power(jnp.maximum((i_v69 - i_v196637) * i_v3913, 0.0), i_v88),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v88 == 0.5,
                jnp.sqrt(jnp.maximum((i_v69 - i_v196637) * i_v3913, 1e-300)),
                jnp.power(jnp.maximum((i_v69 - i_v196637) * i_v3913, 0.0), i_v88),
            ),
        ),
    )
    i_v218913 = jnp.divide(-i_v5121, jnp.where((i_v218552 == 0.0) | jnp.logical_not(jnp.isfinite(i_v218552)), 1e-300, i_v218552))
    i_v218940 = -230.25850929940458 - i_v218913
    i_v218961 = i_v218913 - 230.25850929940458
    i_v220278 = i_v197964 * i_v6575
    i_v220279 = i_v220278 < 0.0
    i_v220317 = i_v197964 * i_v6575
    i_v221986 = (
        i_v7324
        * jnp.where(
            i_v8640,
            0.0,
            (
                i_v1474 * i_v191302
                + jnp.where(i_v192004, 0.0, i_v118 * (i_v193862 * i_v192755))
                + jnp.where(
                    i_v145 == 0.0,
                    0.0,
                    i_v145
                    * (
                        jnp.where(i_v192004, 0.0, i_v193862)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3979
                                * jnp.where(
                                    i_v195075,
                                    i_v195768,
                                    2.0
                                    * jnp.where(
                                        i_v195069 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v195069, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v195784 * (1.0 + 0.5 * (i_v195784 * (1.0 + i_v195784 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v195784
                                                        * (1.0 + 0.5 * (i_v195784 * (1.0 + i_v195784 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v195784 * (1.0 + 0.5 * (i_v195784 * (1.0 + i_v195784 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v195768,
                                ),
                                i_v195042,
                            )
                        )
                        * jnp.divide(
                            i_v195029 * i_v195032,
                            jnp.where(
                                (i_v195029 + i_v195032 == 0.0) | jnp.logical_not(jnp.isfinite(i_v195029 + i_v195032)),
                                1e-300,
                                i_v195029 + i_v195032,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v176 == 0.0,
                    0.0,
                    i_v176
                    * (
                        0.2
                        * i_v197094
                        * i_v197094
                        * jnp.where(
                            jnp.where(i_v197455 < 0.0, -i_v197455, i_v197455) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v197455, -709.0, 709.0)),
                            jnp.where(
                                i_v197455 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v197482 * (1.0 + 0.5 * (i_v197482 * (1.0 + i_v197482 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v197482 * (1.0 + 0.5 * (i_v197482 * (1.0 + i_v197482 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v197482 * (1.0 + 0.5 * (i_v197482 * (1.0 + i_v197482 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v197503 * (1.0 + 0.5 * (i_v197503 * (1.0 + i_v197503 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v26813 > 1000.0,
                1.0,
                jnp.where(
                    i_v197964 > -i_v5226 * i_v26813,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v217 == 4.0,
                                    jnp.where(i_v198942, -i_v198941, i_v198941)
                                    * jnp.where(i_v198942, -i_v198941, i_v198941)
                                    * jnp.where(i_v198942, -i_v198941, i_v198941)
                                    * jnp.where(i_v198942, -i_v198941, i_v198941),
                                    jnp.power(jnp.maximum(jnp.where(i_v198980 < 0.0, -i_v198980, i_v198980), 0.0), i_v217),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v217 == 4.0,
                                        jnp.where(i_v198942, -i_v198941, i_v198941)
                                        * jnp.where(i_v198942, -i_v198941, i_v198941)
                                        * jnp.where(i_v198942, -i_v198941, i_v198941)
                                        * jnp.where(i_v198942, -i_v198941, i_v198941),
                                        jnp.power(jnp.maximum(jnp.where(i_v198980 < 0.0, -i_v198980, i_v198980), 0.0), i_v217),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v217 == 4.0,
                                jnp.where(i_v198942, -i_v198941, i_v198941)
                                * jnp.where(i_v198942, -i_v198941, i_v198941)
                                * jnp.where(i_v198942, -i_v198941, i_v198941)
                                * jnp.where(i_v198942, -i_v198941, i_v198941),
                                jnp.power(jnp.maximum(jnp.where(i_v198980 < 0.0, -i_v198980, i_v198980), 0.0), i_v217),
                            ),
                        ),
                    ),
                    i_v5340 + (i_v197964 + i_v5226 * i_v26813) * i_v6517,
                ),
            ),
        )
        + i_v7338
        * jnp.where(
            i_v8686,
            0.0,
            (
                i_v1624
                * jnp.where(
                    i_v180251,
                    jnp.where(
                        i_v182780,
                        jnp.where(
                            jnp.where(i_v185073 < 0.0, -i_v185073, i_v185073) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v185073, -709.0, 709.0)),
                            jnp.where(
                                i_v185073 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v185133 * (1.0 + 0.5 * (i_v185133 * (1.0 + i_v185133 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v185133 * (1.0 + 0.5 * (i_v185133 * (1.0 + i_v185133 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v185133 * (1.0 + 0.5 * (i_v185133 * (1.0 + i_v185133 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v185172 * (1.0 + 0.5 * (i_v185172 * (1.0 + i_v185172 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v186059
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v186852 - i_v7952 * i_v187002,
                                        jnp.where(
                                            (i_v186852 * i_v186852 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186852 * i_v186852)),
                                            1e-300,
                                            i_v186852 * i_v186852,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v186773 * i_v187002,
                                        jnp.where((i_v186865 == 0.0) | jnp.logical_not(jnp.isfinite(i_v186865)), 1e-300, i_v186865),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v186868 < 0.0, -i_v186868, i_v186868) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v186868, -709.0, 709.0)),
                            jnp.where(
                                i_v186868 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v186928 * (1.0 + 0.5 * (i_v186928 * (1.0 + i_v186928 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v186928 * (1.0 + 0.5 * (i_v186928 * (1.0 + i_v186928 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v186928 * (1.0 + 0.5 * (i_v186928 * (1.0 + i_v186928 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v186967 * (1.0 + 0.5 * (i_v186967 * (1.0 + i_v186967 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v202878, 0.0, i_v124 * (i_v204935 * i_v203767))
                + jnp.where(
                    i_v151 == 0.0,
                    0.0,
                    i_v151
                    * (
                        jnp.where(i_v202878, 0.0, i_v204935)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3984
                                * jnp.where(
                                    i_v206148,
                                    i_v206445,
                                    2.0
                                    * jnp.where(
                                        i_v206142 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v206142, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v206461 * (1.0 + 0.5 * (i_v206461 * (1.0 + i_v206461 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v206461
                                                        * (1.0 + 0.5 * (i_v206461 * (1.0 + i_v206461 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v206461 * (1.0 + 0.5 * (i_v206461 * (1.0 + i_v206461 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v206445,
                                ),
                                i_v206115,
                            )
                        )
                        * jnp.divide(
                            i_v206102 * i_v206105,
                            jnp.where(
                                (i_v206102 + i_v206105 == 0.0) | jnp.logical_not(jnp.isfinite(i_v206102 + i_v206105)),
                                1e-300,
                                i_v206102 + i_v206105,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v182 == 0.0,
                    0.0,
                    i_v182
                    * (
                        0.2
                        * i_v207807
                        * i_v207807
                        * jnp.where(
                            jnp.where(i_v208168 < 0.0, -i_v208168, i_v208168) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v208168, -709.0, 709.0)),
                            jnp.where(
                                i_v208168 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v208195 * (1.0 + 0.5 * (i_v208195 * (1.0 + i_v208195 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v208195 * (1.0 + 0.5 * (i_v208195 * (1.0 + i_v208195 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v208195 * (1.0 + 0.5 * (i_v208195 * (1.0 + i_v208195 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v208216 * (1.0 + 0.5 * (i_v208216 * (1.0 + i_v208216 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v39291 > 1000.0,
                1.0,
                jnp.where(
                    i_v197964 > -i_v5226 * i_v39291,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v221 == 4.0,
                                    jnp.where(i_v209534, -i_v209533, i_v209533)
                                    * jnp.where(i_v209534, -i_v209533, i_v209533)
                                    * jnp.where(i_v209534, -i_v209533, i_v209533)
                                    * jnp.where(i_v209534, -i_v209533, i_v209533),
                                    jnp.power(jnp.maximum(jnp.where(i_v209572 < 0.0, -i_v209572, i_v209572), 0.0), i_v221),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v221 == 4.0,
                                        jnp.where(i_v209534, -i_v209533, i_v209533)
                                        * jnp.where(i_v209534, -i_v209533, i_v209533)
                                        * jnp.where(i_v209534, -i_v209533, i_v209533)
                                        * jnp.where(i_v209534, -i_v209533, i_v209533),
                                        jnp.power(jnp.maximum(jnp.where(i_v209572 < 0.0, -i_v209572, i_v209572), 0.0), i_v221),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v221 == 4.0,
                                jnp.where(i_v209534, -i_v209533, i_v209533)
                                * jnp.where(i_v209534, -i_v209533, i_v209533)
                                * jnp.where(i_v209534, -i_v209533, i_v209533)
                                * jnp.where(i_v209534, -i_v209533, i_v209533),
                                jnp.power(jnp.maximum(jnp.where(i_v209572 < 0.0, -i_v209572, i_v209572), 0.0), i_v221),
                            ),
                        ),
                    ),
                    i_v5451 + (i_v197964 + i_v5226 * i_v39291) * i_v6548,
                ),
            ),
        )
        + i_v7352
        * jnp.where(
            i_v8764,
            0.0,
            (
                i_v1771
                * jnp.where(
                    i_v180251,
                    jnp.where(
                        i_v182780,
                        jnp.where(
                            jnp.where(i_v185932 < 0.0, -i_v185932, i_v185932) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v185932, -709.0, 709.0)),
                            jnp.where(
                                i_v185932 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v185992 * (1.0 + 0.5 * (i_v185992 * (1.0 + i_v185992 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v185992 * (1.0 + 0.5 * (i_v185992 * (1.0 + i_v185992 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v185992 * (1.0 + 0.5 * (i_v185992 * (1.0 + i_v185992 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v186031 * (1.0 + 0.5 * (i_v186031 * (1.0 + i_v186031 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v186059
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v187222 - i_v7952 * i_v187372,
                                        jnp.where(
                                            (i_v187222 * i_v187222 == 0.0) | jnp.logical_not(jnp.isfinite(i_v187222 * i_v187222)),
                                            1e-300,
                                            i_v187222 * i_v187222,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v187143 * i_v187372,
                                        jnp.where((i_v187235 == 0.0) | jnp.logical_not(jnp.isfinite(i_v187235)), 1e-300, i_v187235),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v187238 < 0.0, -i_v187238, i_v187238) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v187238, -709.0, 709.0)),
                            jnp.where(
                                i_v187238 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v187298 * (1.0 + 0.5 * (i_v187298 * (1.0 + i_v187298 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v187298 * (1.0 + 0.5 * (i_v187298 * (1.0 + i_v187298 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v187298 * (1.0 + 0.5 * (i_v187298 * (1.0 + i_v187298 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v187337 * (1.0 + 0.5 * (i_v187337 * (1.0 + i_v187337 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v213623, 0.0, i_v130 * (i_v215680 * i_v214512))
                + jnp.where(
                    i_v157 == 0.0,
                    0.0,
                    i_v157
                    * (
                        jnp.where(i_v213623, 0.0, i_v215680)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3986
                                * jnp.where(
                                    i_v216893,
                                    i_v217190,
                                    2.0
                                    * jnp.where(
                                        i_v216887 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v216887, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v217206 * (1.0 + 0.5 * (i_v217206 * (1.0 + i_v217206 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v217206
                                                        * (1.0 + 0.5 * (i_v217206 * (1.0 + i_v217206 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v217206 * (1.0 + 0.5 * (i_v217206 * (1.0 + i_v217206 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v217190,
                                ),
                                i_v216860,
                            )
                        )
                        * jnp.divide(
                            i_v216847 * i_v216850,
                            jnp.where(
                                (i_v216847 + i_v216850 == 0.0) | jnp.logical_not(jnp.isfinite(i_v216847 + i_v216850)),
                                1e-300,
                                i_v216847 + i_v216850,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v188 == 0.0,
                    0.0,
                    i_v188
                    * (
                        0.2
                        * i_v218552
                        * i_v218552
                        * jnp.where(
                            jnp.where(i_v218913 < 0.0, -i_v218913, i_v218913) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v218913, -709.0, 709.0)),
                            jnp.where(
                                i_v218913 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v218940 * (1.0 + 0.5 * (i_v218940 * (1.0 + i_v218940 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v218940 * (1.0 + 0.5 * (i_v218940 * (1.0 + i_v218940 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v218940 * (1.0 + 0.5 * (i_v218940 * (1.0 + i_v218940 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v218961 * (1.0 + 0.5 * (i_v218961 * (1.0 + i_v218961 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v53512 > 1000.0,
                1.0,
                jnp.where(
                    i_v197964 > -i_v5226 * i_v53512,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v225 == 4.0,
                                    jnp.where(i_v220279, -i_v220278, i_v220278)
                                    * jnp.where(i_v220279, -i_v220278, i_v220278)
                                    * jnp.where(i_v220279, -i_v220278, i_v220278)
                                    * jnp.where(i_v220279, -i_v220278, i_v220278),
                                    jnp.power(jnp.maximum(jnp.where(i_v220317 < 0.0, -i_v220317, i_v220317), 0.0), i_v225),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v225 == 4.0,
                                        jnp.where(i_v220279, -i_v220278, i_v220278)
                                        * jnp.where(i_v220279, -i_v220278, i_v220278)
                                        * jnp.where(i_v220279, -i_v220278, i_v220278)
                                        * jnp.where(i_v220279, -i_v220278, i_v220278),
                                        jnp.power(jnp.maximum(jnp.where(i_v220317 < 0.0, -i_v220317, i_v220317), 0.0), i_v225),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v225 == 4.0,
                                jnp.where(i_v220279, -i_v220278, i_v220278)
                                * jnp.where(i_v220279, -i_v220278, i_v220278)
                                * jnp.where(i_v220279, -i_v220278, i_v220278)
                                * jnp.where(i_v220279, -i_v220278, i_v220278),
                                jnp.power(jnp.maximum(jnp.where(i_v220317 < 0.0, -i_v220317, i_v220317), 0.0), i_v225),
                            ),
                        ),
                    ),
                    i_v5559 + (i_v197964 + i_v5226 * i_v53512) * i_v6576,
                ),
            ),
        )
    )
    i_v226884 = jnp.where(i_v180236 > 0.0, i_v221986 > 0.0, False)
    i_v224902 = 0.1 * i_v596
    i_v226816 = i_v180236 - i_v10873 * (jnp.exp(jnp.clip(i_v224902 * i_v224964, -709.0, 709.0)) - 1.0)
    i_v226850 = i_v221986 - i_v10873 * (jnp.exp(jnp.clip(0.2 * i_v596 * i_v224964, -709.0, 709.0)) - 1.0)
    i_v226936 = jnp.where(
        jnp.where(
            jnp.where(
                jnp.divide(i_v226816, jnp.where((i_v180236 == 0.0) | jnp.logical_not(jnp.isfinite(i_v180236)), 1e-300, i_v180236))
                > 0.001,
                True,
                jnp.divide(i_v226850, jnp.where((i_v221986 == 0.0) | jnp.logical_not(jnp.isfinite(i_v221986)), 1e-300, i_v221986))
                > 0.001,
            ),
            i_v226816 > 0.0,
            False,
        ),
        i_v226850 > 0.0,
        False,
    )
    i_v227249 = jnp.divide(
        i_v594
        * jnp.log(
            jnp.maximum(
                jnp.divide(i_v226816, jnp.where((i_v226850 == 0.0) | jnp.logical_not(jnp.isfinite(i_v226850)), 1e-300, i_v226850)),
                1e-300,
            )
        ),
        -0.1,
    )
    i_v230343 = jnp.where(
        i_v226884,
        jnp.where(
            i_v226936,
            jnp.divide(
                i_v226816,
                jnp.where(
                    (jnp.exp(jnp.clip(i_v224902 * i_v227249, -709.0, 709.0)) - 1.0 == 0.0)
                    | jnp.logical_not(jnp.isfinite(jnp.exp(jnp.clip(i_v224902 * i_v227249, -709.0, 709.0)) - 1.0)),
                    1e-300,
                    jnp.exp(jnp.clip(i_v224902 * i_v227249, -709.0, 709.0)) - 1.0,
                ),
            ),
            0.0,
        ),
        0.0,
    )
    i_v249174 = jnp.where(i_v226864, i_v230343, 0.0)
    i_v259720 = jnp.where(i_v12433, i_v249174, 0.0)
    i_v232280 = jnp.where(i_v226884, jnp.where(i_v226936, i_v227249, 1.0), 1.0)
    i_v259506 = jnp.where(i_v12433, jnp.where(i_v226864, i_v232280, 1.0), 1.0)
    i_v253203 = jnp.where(
        i_v247031,
        jnp.log(
            jnp.maximum(
                jnp.divide(
                    i_v249106,
                    jnp.where(
                        (i_v249174 + 1e-21 == 0.0) | jnp.logical_not(jnp.isfinite(i_v249174 + 1e-21)), 1e-300, i_v249174 + 1e-21
                    ),
                ),
                1e-300,
            )
        ),
        0.0,
    )
    i_v255197 = jnp.where(i_v253203 < 230.25850929940458, i_v253203, 230.25850929940458)
    i_v259594, i_v259612 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v12433, _t, _f), (i_v255197, jnp.exp(jnp.clip(i_v255197, -709.0, 709.0))), (0.0, 0.0)
    )
    i_v12766 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v369 = jnp.where(VJUNREF > 0.5, VJUNREF, 0.5)
    i_v12709 = -0.4 * i_v369
    i_v13211 = i_v12709 < i_v7952
    i_v13899 = NJH - (NJDV * (i_v12709 - i_v7258) + i_v277) - 0.01
    i_v13902 = 4.0 * NJH * 0.01
    i_v13923 = (
        NJH
        - 0.5 * (i_v13899 + jnp.sqrt(jnp.maximum(i_v13899 * i_v13899 + jnp.where(i_v13902 > 0.0, i_v13902, -i_v13902), 1e-300)))
        - i_v277
        - 0.01
    )
    i_v13925 = 4.0 * i_v277 * 0.01
    i_v13962 = jnp.where(
        i_v277 < NJH,
        i_v277
        + 0.5 * (i_v13923 + jnp.sqrt(jnp.maximum(i_v13923 * i_v13923 + jnp.where(i_v13925 > 0.0, i_v13925, -i_v13925), 1e-300))),
        i_v277,
    )
    i_v13978 = i_v596 * (
        jnp.divide(i_v12709, jnp.where((i_v13962 == 0.0) | jnp.logical_not(jnp.isfinite(i_v13962)), 1e-300, i_v13962))
        + jnp.divide(
            i_v7258 * (i_v13962 - i_v277),
            jnp.where((i_v277 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v277 * NJH)), 1e-300, i_v277 * NJH),
        )
    )
    i_v14038 = -230.25850929940458 - i_v13978
    i_v14077 = i_v13978 - 230.25850929940458
    i_v15142 = i_v12709 - i_v7952
    i_v15318 = i_v277 < NJH
    i_v15326 = NJH - (NJDV * (i_v7952 - i_v7258) + i_v277) - 0.01
    i_v15328 = 4.0 * NJH * 0.01
    i_v15339 = jnp.sqrt(jnp.maximum(i_v15326 * i_v15326 + jnp.where(i_v15328 > 0.0, i_v15328, -i_v15328), 1e-300))
    i_v15351 = NJH - 0.5 * (i_v15326 + i_v15339) - i_v277 - 0.01
    i_v15353 = 4.0 * i_v277 * 0.01
    i_v15364 = jnp.sqrt(jnp.maximum(i_v15351 * i_v15351 + jnp.where(i_v15353 > 0.0, i_v15353, -i_v15353), 1e-300))
    i_v15399, i_v15549 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v15318, _t, _f),
        (
            i_v277 + 0.5 * (i_v15351 + i_v15364),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v15326, i_v15339))) * (0.5 * (1.0 + jnp.divide(i_v15351, i_v15364))),
        ),
        (i_v277, 0.0),
    )
    i_v15412 = i_v277 * NJH
    i_v15415 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v15399 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15399)), 1e-300, i_v15399))
        + jnp.divide(
            i_v7258 * (i_v15399 - i_v277), jnp.where((i_v15412 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15412)), 1e-300, i_v15412)
        )
    )
    i_v15475 = -230.25850929940458 - i_v15415
    i_v15514 = i_v15415 - 230.25850929940458
    i_v20640 = jnp.where(i_v118 == 0.0, i_v145 == 0.0, False)
    i_v13375 = 0.5 * (i_v12709 * i_v596)
    i_v13402 = -230.25850929940458 - i_v13375
    i_v13423 = i_v13375 - 230.25850929940458
    i_v16419 = jnp.where(
        i_v13211,
        jnp.where(
            jnp.where(i_v13375 < 0.0, -i_v13375, i_v13375) < 230.25850929940458,
            jnp.exp(jnp.clip(i_v13375, -709.0, 709.0)),
            jnp.where(
                i_v13375 < -230.25850929940458,
                jnp.divide(
                    1e-100,
                    jnp.where(
                        (1.0 + i_v13402 * (1.0 + 0.5 * (i_v13402 * (1.0 + i_v13402 * 0.3333333333333333))) == 0.0)
                        | jnp.logical_not(
                            jnp.isfinite(1.0 + i_v13402 * (1.0 + 0.5 * (i_v13402 * (1.0 + i_v13402 * 0.3333333333333333))))
                        ),
                        1e-300,
                        1.0 + i_v13402 * (1.0 + 0.5 * (i_v13402 * (1.0 + i_v13402 * 0.3333333333333333))),
                    ),
                ),
                1e100 * (1.0 + i_v13423 * (1.0 + 0.5 * (i_v13423 * (1.0 + i_v13423 * 0.3333333333333333)))),
            ),
        ),
        jnp.sqrt(jnp.maximum((1.0 + i_v15142 * i_v596) * i_v15213, 1e-300)),
    )
    i_v22351 = jnp.where(i_v12766, i_v16419, 0.0)
    i_v21227 = i_v76 == 0.5
    i_v16530 = jnp.divide(1.0, jnp.where((i_v16419 == 0.0) | jnp.logical_not(jnp.isfinite(i_v16419)), 1e-300, i_v16419))
    i_v17114 = jnp.where(
        i_v12709 > 0.0,
        2.0
        * (
            i_v594
            * jnp.log(jnp.maximum(2.0 + i_v16530 + jnp.sqrt(jnp.maximum((i_v16530 + 1.0) * (i_v16530 + 3.0), 1e-300)), 1e-300))
        ),
        -i_v12709
        + 2.0
        * (
            i_v594
            * jnp.log(
                jnp.maximum(2.0 * i_v16419 + 1.0 + jnp.sqrt(jnp.maximum((1.0 + i_v16419) * (1.0 + 3.0 * i_v16419), 1e-300)), 1e-300)
            )
        ),
    )
    i_v17115 = i_v8881 - i_v17114
    i_v20930 = jnp.where(
        i_v12766,
        0.5
        * (
            i_v12709
            + i_v17115
            - jnp.sqrt(jnp.maximum((i_v12709 - i_v17115) * (i_v12709 - i_v17115) + 4.0 * i_v594 * i_v594, 1e-300))
        ),
        0.0,
    )
    i_v20933 = i_v2324 - i_v20930
    i_v21980 = i_v3628 * jnp.where(
        i_v21227, jnp.sqrt(jnp.maximum(i_v20933 * i_v3907, 1e-300)), jnp.power(jnp.maximum(i_v20933 * i_v3907, 0.0), i_v76)
    )
    i_v22361 = i_v1108 * ((i_v22351 - 1.0) * i_v21980)
    i_v20942 = jnp.where(i_v12766, i_v17114, 0.0)
    i_v20947 = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.divide(i_v20942, jnp.where((i_v20933 == 0.0) | jnp.logical_not(jnp.isfinite(i_v20933)), 1e-300, i_v20933)),
            1e-300,
        )
    )
    i_v20948 = 1.0 - i_v20947
    i_v21246 = i_v20948 + jnp.where(
        i_v21227,
        0.0,
        (jnp.divide(i_v20948 * i_v20948 * jnp.log(jnp.maximum(i_v20948, 1e-300)), i_v20947) + i_v20948) * (1.0 - 2.0 * i_v76),
    )
    i_v22767 = i_v4127 * jnp.divide(
        jnp.where(i_v20640, 0.0, i_v21980) * i_v2516,
        jnp.where(
            (jnp.where(i_v20640, 0.0, i_v20933) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v20640, 0.0, i_v20933))),
            1e-300,
            jnp.where(i_v20640, 0.0, i_v20933),
        ),
    )
    i_v23124 = jnp.divide(
        0.666666666666667 * i_v3979, jnp.where((i_v22767 == 0.0) | jnp.logical_not(jnp.isfinite(i_v22767)), 1e-300, i_v22767)
    )
    i_v23125 = i_v23124 * i_v23124
    i_v23126 = i_v23125 * i_v23125
    i_v23131 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v23126, jnp.where((i_v23126 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v23126 + 1.0)), 1e-300, i_v23126 + 1.0)
            ),
            1e-300,
        )
    )
    i_v23135 = jnp.sqrt(jnp.maximum(jnp.where(i_v23131 < 0.0, -i_v23131, i_v23131), 1e-300))
    i_v23558 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v22767, i_v23135), 1e-300))
    i_v23588 = (2.0 * (i_v23124 * i_v23135) - i_v23131 - 1.0) * i_v23558
    i_v23591 = i_v23588 > 0.0
    i_v24014 = jnp.where(
        i_v23591,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v23588 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v23588)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v23588,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v23588 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v23588)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v23588,
            ),
        ),
    )
    i_v24392 = i_v24014 * i_v24014
    i_v23139 = i_v23131 * i_v23135
    i_v23585 = i_v3979 * i_v23124 * i_v23135 - i_v3979 * i_v23131 + 0.5 * (i_v22767 * i_v23139)
    i_v23982 = -(i_v23588 * i_v23588) + i_v23585
    i_v23995 = -230.25850929940458 - i_v23982
    i_v24775 = (0.29214664 * i_v24014 + 0.26992878119627894 * i_v24392 + 0.43792457880372104 * (i_v24392 * i_v24014)) * jnp.where(
        i_v23982 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v23982, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v23995 * (1.0 + 0.5 * (i_v23995 * (1.0 + i_v23995 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(jnp.isfinite(1.0 + i_v23995 * (1.0 + 0.5 * (i_v23995 * (1.0 + i_v23995 * 0.3333333333333333))))),
                1e-300,
                1.0 + i_v23995 * (1.0 + 0.5 * (i_v23995 * (1.0 + i_v23995 * 0.3333333333333333))),
            ),
        ),
    )
    i_v24791 = -230.25850929940458 - i_v23585
    i_v23544 = jnp.where(i_v20640, 0.0, i_v21246)
    i_v23515 = -i_v76 * i_v2837
    i_v23547 = jnp.where(
        i_v23515 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v22767 * i_v23139 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v22767 * i_v23139)),
                1e-300,
                1.0 + i_v22767 * i_v23139,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v22767 * i_v23139, 0.0), i_v23515),
    )
    i_v25724 = jnp.where(
        i_v12766,
        0.5
        * (i_v12709 + i_v9052 - jnp.sqrt(jnp.maximum((i_v12709 - i_v9052) * (i_v12709 - i_v9052) + 4.0 * i_v590 * i_v590, 1e-300))),
        0.0,
    )
    i_v26202 = i_v2837 * jnp.divide(
        (i_v61 - i_v25724) * i_v3898,
        jnp.where(
            (
                jnp.where(
                    i_v76 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v61 - i_v25724) * i_v3907, 1e-300)),
                    jnp.power(jnp.maximum((i_v61 - i_v25724) * i_v3907, 0.0), i_v76),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v76 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v61 - i_v25724) * i_v3907, 1e-300)),
                        jnp.power(jnp.maximum((i_v61 - i_v25724) * i_v3907, 0.0), i_v76),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v76 == 0.5,
                jnp.sqrt(jnp.maximum((i_v61 - i_v25724) * i_v3907, 1e-300)),
                jnp.power(jnp.maximum((i_v61 - i_v25724) * i_v3907, 0.0), i_v76),
            ),
        ),
    )
    i_v26579 = jnp.divide(-i_v5102, jnp.where((i_v26202 == 0.0) | jnp.logical_not(jnp.isfinite(i_v26202)), 1e-300, i_v26202))
    i_v26606 = -230.25850929940458 - i_v26579
    i_v26627 = i_v26579 - 230.25850929940458
    i_v27103 = jnp.where(i_v12766, 0.5 * (i_v12709 - jnp.sqrt(jnp.maximum(i_v12709 * i_v12709 + 4e-12, 1e-300))), 0.0)
    i_v28272 = i_v27103 * i_v6514
    i_v28273 = i_v28272 < 0.0
    i_v28311 = i_v27103 * i_v6514
    i_v14391 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v14402 = NJH - (NJDV * (i_v12709 - i_v14391) + i_v281) - 0.01
    i_v14404 = 4.0 * NJH * 0.01
    i_v14423 = (
        NJH
        - 0.5 * (i_v14402 + jnp.sqrt(jnp.maximum(i_v14402 * i_v14402 + jnp.where(i_v14404 > 0.0, i_v14404, -i_v14404), 1e-300)))
        - i_v281
        - 0.01
    )
    i_v14425 = 4.0 * i_v281 * 0.01
    i_v14461 = jnp.where(
        i_v281 < NJH,
        i_v281
        + 0.5 * (i_v14423 + jnp.sqrt(jnp.maximum(i_v14423 * i_v14423 + jnp.where(i_v14425 > 0.0, i_v14425, -i_v14425), 1e-300))),
        i_v281,
    )
    i_v14477 = i_v596 * (
        jnp.divide(i_v12709, jnp.where((i_v14461 == 0.0) | jnp.logical_not(jnp.isfinite(i_v14461)), 1e-300, i_v14461))
        + jnp.divide(
            i_v14391 * (i_v14461 - i_v281),
            jnp.where((i_v281 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v281 * NJH)), 1e-300, i_v281 * NJH),
        )
    )
    i_v14537 = -230.25850929940458 - i_v14477
    i_v14576 = i_v14477 - 230.25850929940458
    i_v15656 = i_v281 < NJH
    i_v15655 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v15664 = NJH - (NJDV * (i_v7952 - i_v15655) + i_v281) - 0.01
    i_v15666 = 4.0 * NJH * 0.01
    i_v15677 = jnp.sqrt(jnp.maximum(i_v15664 * i_v15664 + jnp.where(i_v15666 > 0.0, i_v15666, -i_v15666), 1e-300))
    i_v15688 = NJH - 0.5 * (i_v15664 + i_v15677) - i_v281 - 0.01
    i_v15690 = 4.0 * i_v281 * 0.01
    i_v15701 = jnp.sqrt(jnp.maximum(i_v15688 * i_v15688 + jnp.where(i_v15690 > 0.0, i_v15690, -i_v15690), 1e-300))
    i_v15734, i_v15884 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v15656, _t, _f),
        (
            i_v281 + 0.5 * (i_v15688 + i_v15701),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v15664, i_v15677))) * (0.5 * (1.0 + jnp.divide(i_v15688, i_v15701))),
        ),
        (i_v281, 0.0),
    )
    i_v15747 = i_v281 * NJH
    i_v15750 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v15734 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15734)), 1e-300, i_v15734))
        + jnp.divide(
            i_v15655 * (i_v15734 - i_v281), jnp.where((i_v15747 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15747)), 1e-300, i_v15747)
        )
    )
    i_v15810 = -230.25850929940458 - i_v15750
    i_v15849 = i_v15750 - 230.25850929940458
    i_v32842 = jnp.where(i_v124 == 0.0, i_v151 == 0.0, False)
    i_v33723 = i_v82 == 0.5
    i_v33282 = i_v2333 - i_v20930
    i_v34632 = i_v3763 * jnp.where(
        i_v33723, jnp.sqrt(jnp.maximum(i_v33282 * i_v3910, 1e-300)), jnp.power(jnp.maximum(i_v33282 * i_v3910, 0.0), i_v82)
    )
    i_v35152 = i_v1117 * ((i_v22351 - 1.0) * i_v34632)
    i_v33365 = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.divide(i_v20942, jnp.where((i_v33282 == 0.0) | jnp.logical_not(jnp.isfinite(i_v33282)), 1e-300, i_v33282)),
            1e-300,
        )
    )
    i_v33366 = 1.0 - i_v33365
    i_v33742 = i_v33366 + jnp.where(
        i_v33723,
        0.0,
        (jnp.divide(i_v33366 * i_v33366 * jnp.log(jnp.maximum(i_v33366, 1e-300)), i_v33365) + i_v33366) * (1.0 - 2.0 * i_v82),
    )
    i_v35636 = i_v4262 * jnp.divide(
        jnp.where(i_v32842, 0.0, i_v34632) * i_v2678,
        jnp.where(
            (jnp.where(i_v32842, 0.0, i_v33282) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v32842, 0.0, i_v33282))),
            1e-300,
            jnp.where(i_v32842, 0.0, i_v33282),
        ),
    )
    i_v36070 = jnp.divide(
        0.666666666666667 * i_v3984, jnp.where((i_v35636 == 0.0) | jnp.logical_not(jnp.isfinite(i_v35636)), 1e-300, i_v35636)
    )
    i_v36071 = i_v36070 * i_v36070
    i_v36072 = i_v36071 * i_v36071
    i_v36077 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v36072, jnp.where((i_v36072 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v36072 + 1.0)), 1e-300, i_v36072 + 1.0)
            ),
            1e-300,
        )
    )
    i_v36081 = jnp.sqrt(jnp.maximum(jnp.where(i_v36077 < 0.0, -i_v36077, i_v36077), 1e-300))
    i_v36581 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v35636, i_v36081), 1e-300))
    i_v36611 = (2.0 * (i_v36070 * i_v36081) - i_v36077 - 1.0) * i_v36581
    i_v36614 = i_v36611 > 0.0
    i_v36744 = jnp.where(
        i_v36614,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v36611 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v36611)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v36611,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v36611 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v36611)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v36611,
            ),
        ),
    )
    i_v36825 = i_v36744 * i_v36744
    i_v36085 = i_v36077 * i_v36081
    i_v36608 = i_v3984 * i_v36070 * i_v36081 - i_v3984 * i_v36077 + 0.5 * (i_v35636 * i_v36085)
    i_v36712 = -(i_v36611 * i_v36611) + i_v36608
    i_v36725 = -230.25850929940458 - i_v36712
    i_v36911 = (0.29214664 * i_v36744 + 0.26992878119627894 * i_v36825 + 0.43792457880372104 * (i_v36825 * i_v36744)) * jnp.where(
        i_v36712 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v36712, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v36725 * (1.0 + 0.5 * (i_v36725 * (1.0 + i_v36725 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(jnp.isfinite(1.0 + i_v36725 * (1.0 + 0.5 * (i_v36725 * (1.0 + i_v36725 * 0.3333333333333333))))),
                1e-300,
                1.0 + i_v36725 * (1.0 + 0.5 * (i_v36725 * (1.0 + i_v36725 * 0.3333333333333333))),
            ),
        ),
    )
    i_v36927 = -230.25850929940458 - i_v36608
    i_v36568 = jnp.where(i_v32842, 0.0, i_v33742)
    i_v36539 = -i_v82 * i_v2840
    i_v36571 = jnp.where(
        i_v36539 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v35636 * i_v36085 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v35636 * i_v36085)),
                1e-300,
                1.0 + i_v35636 * i_v36085,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v35636 * i_v36085, 0.0), i_v36539),
    )
    i_v38605 = i_v2840 * jnp.divide(
        (i_v65 - i_v25724) * i_v3901,
        jnp.where(
            (
                jnp.where(
                    i_v82 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v65 - i_v25724) * i_v3910, 1e-300)),
                    jnp.power(jnp.maximum((i_v65 - i_v25724) * i_v3910, 0.0), i_v82),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v82 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v65 - i_v25724) * i_v3910, 1e-300)),
                        jnp.power(jnp.maximum((i_v65 - i_v25724) * i_v3910, 0.0), i_v82),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v82 == 0.5,
                jnp.sqrt(jnp.maximum((i_v65 - i_v25724) * i_v3910, 1e-300)),
                jnp.power(jnp.maximum((i_v65 - i_v25724) * i_v3910, 0.0), i_v82),
            ),
        ),
    )
    i_v39057 = jnp.divide(-i_v5110, jnp.where((i_v38605 == 0.0) | jnp.logical_not(jnp.isfinite(i_v38605)), 1e-300, i_v38605))
    i_v39084 = -230.25850929940458 - i_v39057
    i_v39105 = i_v39057 - 230.25850929940458
    i_v40696 = i_v27103 * i_v6546
    i_v40697 = i_v40696 < 0.0
    i_v40735 = i_v27103 * i_v6546
    i_v14929 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v14940 = NJH - (NJDV * (i_v12709 - i_v14929) + i_v285) - 0.01
    i_v14942 = 4.0 * NJH * 0.01
    i_v14961 = (
        NJH
        - 0.5 * (i_v14940 + jnp.sqrt(jnp.maximum(i_v14940 * i_v14940 + jnp.where(i_v14942 > 0.0, i_v14942, -i_v14942), 1e-300)))
        - i_v285
        - 0.01
    )
    i_v14963 = 4.0 * i_v285 * 0.01
    i_v14999 = jnp.where(
        i_v285 < NJH,
        i_v285
        + 0.5 * (i_v14961 + jnp.sqrt(jnp.maximum(i_v14961 * i_v14961 + jnp.where(i_v14963 > 0.0, i_v14963, -i_v14963), 1e-300))),
        i_v285,
    )
    i_v15015 = i_v596 * (
        jnp.divide(i_v12709, jnp.where((i_v14999 == 0.0) | jnp.logical_not(jnp.isfinite(i_v14999)), 1e-300, i_v14999))
        + jnp.divide(
            i_v14929 * (i_v14999 - i_v285),
            jnp.where((i_v285 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v285 * NJH)), 1e-300, i_v285 * NJH),
        )
    )
    i_v15075 = -230.25850929940458 - i_v15015
    i_v15114 = i_v15015 - 230.25850929940458
    i_v16026 = i_v285 < NJH
    i_v16025 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v16034 = NJH - (NJDV * (i_v7952 - i_v16025) + i_v285) - 0.01
    i_v16036 = 4.0 * NJH * 0.01
    i_v16047 = jnp.sqrt(jnp.maximum(i_v16034 * i_v16034 + jnp.where(i_v16036 > 0.0, i_v16036, -i_v16036), 1e-300))
    i_v16058 = NJH - 0.5 * (i_v16034 + i_v16047) - i_v285 - 0.01
    i_v16060 = 4.0 * i_v285 * 0.01
    i_v16071 = jnp.sqrt(jnp.maximum(i_v16058 * i_v16058 + jnp.where(i_v16060 > 0.0, i_v16060, -i_v16060), 1e-300))
    i_v16104, i_v16254 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v16026, _t, _f),
        (
            i_v285 + 0.5 * (i_v16058 + i_v16071),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v16034, i_v16047))) * (0.5 * (1.0 + jnp.divide(i_v16058, i_v16071))),
        ),
        (i_v285, 0.0),
    )
    i_v16117 = i_v285 * NJH
    i_v16120 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v16104 == 0.0) | jnp.logical_not(jnp.isfinite(i_v16104)), 1e-300, i_v16104))
        + jnp.divide(
            i_v16025 * (i_v16104 - i_v285), jnp.where((i_v16117 == 0.0) | jnp.logical_not(jnp.isfinite(i_v16117)), 1e-300, i_v16117)
        )
    )
    i_v16180 = -230.25850929940458 - i_v16120
    i_v16219 = i_v16120 - 230.25850929940458
    i_v46115 = jnp.where(i_v130 == 0.0, i_v157 == 0.0, False)
    i_v47164 = i_v88 == 0.5
    i_v46639 = i_v2342 - i_v20930
    i_v48229 = i_v3895 * jnp.where(
        i_v47164, jnp.sqrt(jnp.maximum(i_v46639 * i_v3913, 1e-300)), jnp.power(jnp.maximum(i_v46639 * i_v3913, 0.0), i_v88)
    )
    i_v48827 = i_v1126 * ((i_v22351 - 1.0) * i_v48229)
    i_v46722 = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.divide(i_v20942, jnp.where((i_v46639 == 0.0) | jnp.logical_not(jnp.isfinite(i_v46639)), 1e-300, i_v46639)),
            1e-300,
        )
    )
    i_v46723 = 1.0 - i_v46722
    i_v47183 = i_v46723 + jnp.where(
        i_v47164,
        0.0,
        (jnp.divide(i_v46723 * i_v46723 * jnp.log(jnp.maximum(i_v46723, 1e-300)), i_v46722) + i_v46723) * (1.0 - 2.0 * i_v88),
    )
    i_v49389 = i_v4394 * jnp.divide(
        jnp.where(i_v46115, 0.0, i_v48229) * i_v2834,
        jnp.where(
            (jnp.where(i_v46115, 0.0, i_v46639) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v46115, 0.0, i_v46639))),
            1e-300,
            jnp.where(i_v46115, 0.0, i_v46639),
        ),
    )
    i_v49901 = jnp.divide(
        0.666666666666667 * i_v3986, jnp.where((i_v49389 == 0.0) | jnp.logical_not(jnp.isfinite(i_v49389)), 1e-300, i_v49389)
    )
    i_v49902 = i_v49901 * i_v49901
    i_v49903 = i_v49902 * i_v49902
    i_v49908 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v49903, jnp.where((i_v49903 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v49903 + 1.0)), 1e-300, i_v49903 + 1.0)
            ),
            1e-300,
        )
    )
    i_v49912 = jnp.sqrt(jnp.maximum(jnp.where(i_v49908 < 0.0, -i_v49908, i_v49908), 1e-300))
    i_v50490 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v49389, i_v49912), 1e-300))
    i_v50520 = (2.0 * (i_v49901 * i_v49912) - i_v49908 - 1.0) * i_v50490
    i_v50523 = i_v50520 > 0.0
    i_v50653 = jnp.where(
        i_v50523,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v50520 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v50520)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v50520,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v50520 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v50520)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v50520,
            ),
        ),
    )
    i_v50734 = i_v50653 * i_v50653
    i_v49916 = i_v49908 * i_v49912
    i_v50517 = i_v3986 * i_v49901 * i_v49912 - i_v3986 * i_v49908 + 0.5 * (i_v49389 * i_v49916)
    i_v50621 = -(i_v50520 * i_v50520) + i_v50517
    i_v50634 = -230.25850929940458 - i_v50621
    i_v50820 = (0.29214664 * i_v50653 + 0.26992878119627894 * i_v50734 + 0.43792457880372104 * (i_v50734 * i_v50653)) * jnp.where(
        i_v50621 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v50621, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v50634 * (1.0 + 0.5 * (i_v50634 * (1.0 + i_v50634 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(jnp.isfinite(1.0 + i_v50634 * (1.0 + 0.5 * (i_v50634 * (1.0 + i_v50634 * 0.3333333333333333))))),
                1e-300,
                1.0 + i_v50634 * (1.0 + 0.5 * (i_v50634 * (1.0 + i_v50634 * 0.3333333333333333))),
            ),
        ),
    )
    i_v50836 = -230.25850929940458 - i_v50517
    i_v50477 = jnp.where(i_v46115, 0.0, i_v47183)
    i_v50448 = -i_v88 * i_v2843
    i_v50480 = jnp.where(
        i_v50448 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v49389 * i_v49916 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v49389 * i_v49916)),
                1e-300,
                1.0 + i_v49389 * i_v49916,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v49389 * i_v49916, 0.0), i_v50448),
    )
    i_v52751 = i_v2843 * jnp.divide(
        (i_v69 - i_v25724) * i_v3904,
        jnp.where(
            (
                jnp.where(
                    i_v88 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v69 - i_v25724) * i_v3913, 1e-300)),
                    jnp.power(jnp.maximum((i_v69 - i_v25724) * i_v3913, 0.0), i_v88),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v88 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v69 - i_v25724) * i_v3913, 1e-300)),
                        jnp.power(jnp.maximum((i_v69 - i_v25724) * i_v3913, 0.0), i_v88),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v88 == 0.5,
                jnp.sqrt(jnp.maximum((i_v69 - i_v25724) * i_v3913, 1e-300)),
                jnp.power(jnp.maximum((i_v69 - i_v25724) * i_v3913, 0.0), i_v88),
            ),
        ),
    )
    i_v53278 = jnp.divide(-i_v5121, jnp.where((i_v52751 == 0.0) | jnp.logical_not(jnp.isfinite(i_v52751)), 1e-300, i_v52751))
    i_v53305 = -230.25850929940458 - i_v53278
    i_v53326 = i_v53278 - 230.25850929940458
    i_v55148 = i_v27103 * i_v6575
    i_v55149 = i_v55148 < 0.0
    i_v55187 = i_v27103 * i_v6575
    i_v57182 = (
        i_v7324
        * jnp.where(
            i_v8640,
            0.0,
            (
                i_v1474
                * jnp.where(
                    i_v12766,
                    jnp.where(
                        i_v13211,
                        jnp.where(
                            jnp.where(i_v13978 < 0.0, -i_v13978, i_v13978) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v13978, -709.0, 709.0)),
                            jnp.where(
                                i_v13978 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v14038 * (1.0 + 0.5 * (i_v14038 * (1.0 + i_v14038 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v14038 * (1.0 + 0.5 * (i_v14038 * (1.0 + i_v14038 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v14038 * (1.0 + 0.5 * (i_v14038 * (1.0 + i_v14038 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v14077 * (1.0 + 0.5 * (i_v14077 * (1.0 + i_v14077 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v15142
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v15399 - i_v7952 * i_v15549,
                                        jnp.where(
                                            (i_v15399 * i_v15399 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15399 * i_v15399)),
                                            1e-300,
                                            i_v15399 * i_v15399,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v7258 * i_v15549,
                                        jnp.where((i_v15412 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15412)), 1e-300, i_v15412),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v15415 < 0.0, -i_v15415, i_v15415) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v15415, -709.0, 709.0)),
                            jnp.where(
                                i_v15415 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v15475 * (1.0 + 0.5 * (i_v15475 * (1.0 + i_v15475 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v15475 * (1.0 + 0.5 * (i_v15475 * (1.0 + i_v15475 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v15475 * (1.0 + 0.5 * (i_v15475 * (1.0 + i_v15475 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v15514 * (1.0 + 0.5 * (i_v15514 * (1.0 + i_v15514 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v20640, 0.0, i_v118 * (i_v22361 * i_v21246))
                + jnp.where(
                    i_v145 == 0.0,
                    0.0,
                    i_v145
                    * (
                        jnp.where(i_v20640, 0.0, i_v22361)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3979
                                * jnp.where(
                                    i_v23591,
                                    i_v24775,
                                    2.0
                                    * jnp.where(
                                        i_v23585 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v23585, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v24791 * (1.0 + 0.5 * (i_v24791 * (1.0 + i_v24791 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v24791
                                                        * (1.0 + 0.5 * (i_v24791 * (1.0 + i_v24791 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0 + i_v24791 * (1.0 + 0.5 * (i_v24791 * (1.0 + i_v24791 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v24775,
                                ),
                                i_v23558,
                            )
                        )
                        * jnp.divide(
                            i_v23544 * i_v23547,
                            jnp.where(
                                (i_v23544 + i_v23547 == 0.0) | jnp.logical_not(jnp.isfinite(i_v23544 + i_v23547)),
                                1e-300,
                                i_v23544 + i_v23547,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v176 == 0.0,
                    0.0,
                    i_v176
                    * (
                        i_v12709
                        * i_v26202
                        * i_v26202
                        * jnp.where(
                            jnp.where(i_v26579 < 0.0, -i_v26579, i_v26579) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v26579, -709.0, 709.0)),
                            jnp.where(
                                i_v26579 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v26606 * (1.0 + 0.5 * (i_v26606 * (1.0 + i_v26606 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v26606 * (1.0 + 0.5 * (i_v26606 * (1.0 + i_v26606 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v26606 * (1.0 + 0.5 * (i_v26606 * (1.0 + i_v26606 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v26627 * (1.0 + 0.5 * (i_v26627 * (1.0 + i_v26627 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v26813 > 1000.0,
                1.0,
                jnp.where(
                    i_v27103 > -i_v5226 * i_v26813,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v217 == 4.0,
                                    jnp.where(i_v28273, -i_v28272, i_v28272)
                                    * jnp.where(i_v28273, -i_v28272, i_v28272)
                                    * jnp.where(i_v28273, -i_v28272, i_v28272)
                                    * jnp.where(i_v28273, -i_v28272, i_v28272),
                                    jnp.power(jnp.maximum(jnp.where(i_v28311 < 0.0, -i_v28311, i_v28311), 0.0), i_v217),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v217 == 4.0,
                                        jnp.where(i_v28273, -i_v28272, i_v28272)
                                        * jnp.where(i_v28273, -i_v28272, i_v28272)
                                        * jnp.where(i_v28273, -i_v28272, i_v28272)
                                        * jnp.where(i_v28273, -i_v28272, i_v28272),
                                        jnp.power(jnp.maximum(jnp.where(i_v28311 < 0.0, -i_v28311, i_v28311), 0.0), i_v217),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v217 == 4.0,
                                jnp.where(i_v28273, -i_v28272, i_v28272)
                                * jnp.where(i_v28273, -i_v28272, i_v28272)
                                * jnp.where(i_v28273, -i_v28272, i_v28272)
                                * jnp.where(i_v28273, -i_v28272, i_v28272),
                                jnp.power(jnp.maximum(jnp.where(i_v28311 < 0.0, -i_v28311, i_v28311), 0.0), i_v217),
                            ),
                        ),
                    ),
                    i_v5340 + (i_v27103 + i_v5226 * i_v26813) * i_v6517,
                ),
            ),
        )
        + i_v7338
        * jnp.where(
            i_v8686,
            0.0,
            (
                i_v1624
                * jnp.where(
                    i_v12766,
                    jnp.where(
                        i_v13211,
                        jnp.where(
                            jnp.where(i_v14477 < 0.0, -i_v14477, i_v14477) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v14477, -709.0, 709.0)),
                            jnp.where(
                                i_v14477 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v14537 * (1.0 + 0.5 * (i_v14537 * (1.0 + i_v14537 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v14537 * (1.0 + 0.5 * (i_v14537 * (1.0 + i_v14537 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v14537 * (1.0 + 0.5 * (i_v14537 * (1.0 + i_v14537 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v14576 * (1.0 + 0.5 * (i_v14576 * (1.0 + i_v14576 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v15142
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v15734 - i_v7952 * i_v15884,
                                        jnp.where(
                                            (i_v15734 * i_v15734 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15734 * i_v15734)),
                                            1e-300,
                                            i_v15734 * i_v15734,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v15655 * i_v15884,
                                        jnp.where((i_v15747 == 0.0) | jnp.logical_not(jnp.isfinite(i_v15747)), 1e-300, i_v15747),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v15750 < 0.0, -i_v15750, i_v15750) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v15750, -709.0, 709.0)),
                            jnp.where(
                                i_v15750 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v15810 * (1.0 + 0.5 * (i_v15810 * (1.0 + i_v15810 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v15810 * (1.0 + 0.5 * (i_v15810 * (1.0 + i_v15810 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v15810 * (1.0 + 0.5 * (i_v15810 * (1.0 + i_v15810 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v15849 * (1.0 + 0.5 * (i_v15849 * (1.0 + i_v15849 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v32842, 0.0, i_v124 * (i_v35152 * i_v33742))
                + jnp.where(
                    i_v151 == 0.0,
                    0.0,
                    i_v151
                    * (
                        jnp.where(i_v32842, 0.0, i_v35152)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3984
                                * jnp.where(
                                    i_v36614,
                                    i_v36911,
                                    2.0
                                    * jnp.where(
                                        i_v36608 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v36608, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v36927 * (1.0 + 0.5 * (i_v36927 * (1.0 + i_v36927 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v36927
                                                        * (1.0 + 0.5 * (i_v36927 * (1.0 + i_v36927 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0 + i_v36927 * (1.0 + 0.5 * (i_v36927 * (1.0 + i_v36927 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v36911,
                                ),
                                i_v36581,
                            )
                        )
                        * jnp.divide(
                            i_v36568 * i_v36571,
                            jnp.where(
                                (i_v36568 + i_v36571 == 0.0) | jnp.logical_not(jnp.isfinite(i_v36568 + i_v36571)),
                                1e-300,
                                i_v36568 + i_v36571,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v182 == 0.0,
                    0.0,
                    i_v182
                    * (
                        i_v12709
                        * i_v38605
                        * i_v38605
                        * jnp.where(
                            jnp.where(i_v39057 < 0.0, -i_v39057, i_v39057) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v39057, -709.0, 709.0)),
                            jnp.where(
                                i_v39057 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v39084 * (1.0 + 0.5 * (i_v39084 * (1.0 + i_v39084 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v39084 * (1.0 + 0.5 * (i_v39084 * (1.0 + i_v39084 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v39084 * (1.0 + 0.5 * (i_v39084 * (1.0 + i_v39084 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v39105 * (1.0 + 0.5 * (i_v39105 * (1.0 + i_v39105 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v39291 > 1000.0,
                1.0,
                jnp.where(
                    i_v27103 > -i_v5226 * i_v39291,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v221 == 4.0,
                                    jnp.where(i_v40697, -i_v40696, i_v40696)
                                    * jnp.where(i_v40697, -i_v40696, i_v40696)
                                    * jnp.where(i_v40697, -i_v40696, i_v40696)
                                    * jnp.where(i_v40697, -i_v40696, i_v40696),
                                    jnp.power(jnp.maximum(jnp.where(i_v40735 < 0.0, -i_v40735, i_v40735), 0.0), i_v221),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v221 == 4.0,
                                        jnp.where(i_v40697, -i_v40696, i_v40696)
                                        * jnp.where(i_v40697, -i_v40696, i_v40696)
                                        * jnp.where(i_v40697, -i_v40696, i_v40696)
                                        * jnp.where(i_v40697, -i_v40696, i_v40696),
                                        jnp.power(jnp.maximum(jnp.where(i_v40735 < 0.0, -i_v40735, i_v40735), 0.0), i_v221),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v221 == 4.0,
                                jnp.where(i_v40697, -i_v40696, i_v40696)
                                * jnp.where(i_v40697, -i_v40696, i_v40696)
                                * jnp.where(i_v40697, -i_v40696, i_v40696)
                                * jnp.where(i_v40697, -i_v40696, i_v40696),
                                jnp.power(jnp.maximum(jnp.where(i_v40735 < 0.0, -i_v40735, i_v40735), 0.0), i_v221),
                            ),
                        ),
                    ),
                    i_v5451 + (i_v27103 + i_v5226 * i_v39291) * i_v6548,
                ),
            ),
        )
        + i_v7352
        * jnp.where(
            i_v8764,
            0.0,
            (
                i_v1771
                * jnp.where(
                    i_v12766,
                    jnp.where(
                        i_v13211,
                        jnp.where(
                            jnp.where(i_v15015 < 0.0, -i_v15015, i_v15015) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v15015, -709.0, 709.0)),
                            jnp.where(
                                i_v15015 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v15075 * (1.0 + 0.5 * (i_v15075 * (1.0 + i_v15075 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v15075 * (1.0 + 0.5 * (i_v15075 * (1.0 + i_v15075 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v15075 * (1.0 + 0.5 * (i_v15075 * (1.0 + i_v15075 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v15114 * (1.0 + 0.5 * (i_v15114 * (1.0 + i_v15114 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v15142
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v16104 - i_v7952 * i_v16254,
                                        jnp.where(
                                            (i_v16104 * i_v16104 == 0.0) | jnp.logical_not(jnp.isfinite(i_v16104 * i_v16104)),
                                            1e-300,
                                            i_v16104 * i_v16104,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v16025 * i_v16254,
                                        jnp.where((i_v16117 == 0.0) | jnp.logical_not(jnp.isfinite(i_v16117)), 1e-300, i_v16117),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v16120 < 0.0, -i_v16120, i_v16120) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v16120, -709.0, 709.0)),
                            jnp.where(
                                i_v16120 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v16180 * (1.0 + 0.5 * (i_v16180 * (1.0 + i_v16180 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v16180 * (1.0 + 0.5 * (i_v16180 * (1.0 + i_v16180 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v16180 * (1.0 + 0.5 * (i_v16180 * (1.0 + i_v16180 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v16219 * (1.0 + 0.5 * (i_v16219 * (1.0 + i_v16219 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v46115, 0.0, i_v130 * (i_v48827 * i_v47183))
                + jnp.where(
                    i_v157 == 0.0,
                    0.0,
                    i_v157
                    * (
                        jnp.where(i_v46115, 0.0, i_v48827)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3986
                                * jnp.where(
                                    i_v50523,
                                    i_v50820,
                                    2.0
                                    * jnp.where(
                                        i_v50517 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v50517, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v50836 * (1.0 + 0.5 * (i_v50836 * (1.0 + i_v50836 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v50836
                                                        * (1.0 + 0.5 * (i_v50836 * (1.0 + i_v50836 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0 + i_v50836 * (1.0 + 0.5 * (i_v50836 * (1.0 + i_v50836 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v50820,
                                ),
                                i_v50490,
                            )
                        )
                        * jnp.divide(
                            i_v50477 * i_v50480,
                            jnp.where(
                                (i_v50477 + i_v50480 == 0.0) | jnp.logical_not(jnp.isfinite(i_v50477 + i_v50480)),
                                1e-300,
                                i_v50477 + i_v50480,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v188 == 0.0,
                    0.0,
                    i_v188
                    * (
                        i_v12709
                        * i_v52751
                        * i_v52751
                        * jnp.where(
                            jnp.where(i_v53278 < 0.0, -i_v53278, i_v53278) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v53278, -709.0, 709.0)),
                            jnp.where(
                                i_v53278 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v53305 * (1.0 + 0.5 * (i_v53305 * (1.0 + i_v53305 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v53305 * (1.0 + 0.5 * (i_v53305 * (1.0 + i_v53305 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v53305 * (1.0 + 0.5 * (i_v53305 * (1.0 + i_v53305 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v53326 * (1.0 + 0.5 * (i_v53326 * (1.0 + i_v53326 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v53512 > 1000.0,
                1.0,
                jnp.where(
                    i_v27103 > -i_v5226 * i_v53512,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v225 == 4.0,
                                    jnp.where(i_v55149, -i_v55148, i_v55148)
                                    * jnp.where(i_v55149, -i_v55148, i_v55148)
                                    * jnp.where(i_v55149, -i_v55148, i_v55148)
                                    * jnp.where(i_v55149, -i_v55148, i_v55148),
                                    jnp.power(jnp.maximum(jnp.where(i_v55187 < 0.0, -i_v55187, i_v55187), 0.0), i_v225),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v225 == 4.0,
                                        jnp.where(i_v55149, -i_v55148, i_v55148)
                                        * jnp.where(i_v55149, -i_v55148, i_v55148)
                                        * jnp.where(i_v55149, -i_v55148, i_v55148)
                                        * jnp.where(i_v55149, -i_v55148, i_v55148),
                                        jnp.power(jnp.maximum(jnp.where(i_v55187 < 0.0, -i_v55187, i_v55187), 0.0), i_v225),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v225 == 4.0,
                                jnp.where(i_v55149, -i_v55148, i_v55148)
                                * jnp.where(i_v55149, -i_v55148, i_v55148)
                                * jnp.where(i_v55149, -i_v55148, i_v55148)
                                * jnp.where(i_v55149, -i_v55148, i_v55148),
                                jnp.power(jnp.maximum(jnp.where(i_v55187 < 0.0, -i_v55187, i_v55187), 0.0), i_v225),
                            ),
                        ),
                    ),
                    i_v5559 + (i_v27103 + i_v5226 * i_v53512) * i_v6576,
                ),
            ),
        )
    )
    i_v57197 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v12712 = -0.65 * i_v369
    i_v58628 = i_v12712 < i_v7952
    i_v59951 = NJH - (NJDV * (i_v12712 - i_v7258) + i_v277) - 0.01
    i_v59953 = 4.0 * NJH * 0.01
    i_v59972 = (
        NJH
        - 0.5 * (i_v59951 + jnp.sqrt(jnp.maximum(i_v59951 * i_v59951 + jnp.where(i_v59953 > 0.0, i_v59953, -i_v59953), 1e-300)))
        - i_v277
        - 0.01
    )
    i_v59974 = 4.0 * i_v277 * 0.01
    i_v60010 = jnp.where(
        i_v277 < NJH,
        i_v277
        + 0.5 * (i_v59972 + jnp.sqrt(jnp.maximum(i_v59972 * i_v59972 + jnp.where(i_v59974 > 0.0, i_v59974, -i_v59974), 1e-300))),
        i_v277,
    )
    i_v60026 = i_v596 * (
        jnp.divide(i_v12712, jnp.where((i_v60010 == 0.0) | jnp.logical_not(jnp.isfinite(i_v60010)), 1e-300, i_v60010))
        + jnp.divide(
            i_v7258 * (i_v60010 - i_v277),
            jnp.where((i_v277 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v277 * NJH)), 1e-300, i_v277 * NJH),
        )
    )
    i_v60086 = -230.25850929940458 - i_v60026
    i_v60125 = i_v60026 - 230.25850929940458
    i_v61907 = i_v12712 - i_v7952
    i_v62288 = i_v277 < NJH
    i_v62296 = NJH - (NJDV * (i_v7952 - i_v7258) + i_v277) - 0.01
    i_v62298 = 4.0 * NJH * 0.01
    i_v62309 = jnp.sqrt(jnp.maximum(i_v62296 * i_v62296 + jnp.where(i_v62298 > 0.0, i_v62298, -i_v62298), 1e-300))
    i_v62320 = NJH - 0.5 * (i_v62296 + i_v62309) - i_v277 - 0.01
    i_v62322 = 4.0 * i_v277 * 0.01
    i_v62333 = jnp.sqrt(jnp.maximum(i_v62320 * i_v62320 + jnp.where(i_v62322 > 0.0, i_v62322, -i_v62322), 1e-300))
    i_v62366, i_v62516 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v62288, _t, _f),
        (
            i_v277 + 0.5 * (i_v62320 + i_v62333),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v62296, i_v62309))) * (0.5 * (1.0 + jnp.divide(i_v62320, i_v62333))),
        ),
        (i_v277, 0.0),
    )
    i_v62379 = i_v277 * NJH
    i_v62382 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v62366 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62366)), 1e-300, i_v62366))
        + jnp.divide(
            i_v7258 * (i_v62366 - i_v277), jnp.where((i_v62379 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62379)), 1e-300, i_v62379)
        )
    )
    i_v62442 = -230.25850929940458 - i_v62382
    i_v62481 = i_v62382 - 230.25850929940458
    i_v67852 = jnp.where(i_v118 == 0.0, i_v145 == 0.0, False)
    i_v58881 = 0.5 * (i_v12712 * i_v596)
    i_v58908 = -230.25850929940458 - i_v58881
    i_v58929 = i_v58881 - 230.25850929940458
    i_v63385 = jnp.where(
        i_v58628,
        jnp.where(
            jnp.where(i_v58881 < 0.0, -i_v58881, i_v58881) < 230.25850929940458,
            jnp.exp(jnp.clip(i_v58881, -709.0, 709.0)),
            jnp.where(
                i_v58881 < -230.25850929940458,
                jnp.divide(
                    1e-100,
                    jnp.where(
                        (1.0 + i_v58908 * (1.0 + 0.5 * (i_v58908 * (1.0 + i_v58908 * 0.3333333333333333))) == 0.0)
                        | jnp.logical_not(
                            jnp.isfinite(1.0 + i_v58908 * (1.0 + 0.5 * (i_v58908 * (1.0 + i_v58908 * 0.3333333333333333))))
                        ),
                        1e-300,
                        1.0 + i_v58908 * (1.0 + 0.5 * (i_v58908 * (1.0 + i_v58908 * 0.3333333333333333))),
                    ),
                ),
                1e100 * (1.0 + i_v58929 * (1.0 + 0.5 * (i_v58929 * (1.0 + i_v58929 * 0.3333333333333333)))),
            ),
        ),
        jnp.sqrt(jnp.maximum((1.0 + i_v61907 * i_v596) * i_v15213, 1e-300)),
    )
    i_v69700 = jnp.where(i_v57197, i_v63385, 0.0)
    i_v68584 = i_v76 == 0.5
    i_v63496 = jnp.divide(1.0, jnp.where((i_v63385 == 0.0) | jnp.logical_not(jnp.isfinite(i_v63385)), 1e-300, i_v63385))
    i_v64263 = jnp.where(
        i_v12712 > 0.0,
        2.0
        * (
            i_v594
            * jnp.log(jnp.maximum(2.0 + i_v63496 + jnp.sqrt(jnp.maximum((i_v63496 + 1.0) * (i_v63496 + 3.0), 1e-300)), 1e-300))
        ),
        -i_v12712
        + 2.0
        * (
            i_v594
            * jnp.log(
                jnp.maximum(2.0 * i_v63385 + 1.0 + jnp.sqrt(jnp.maximum((1.0 + i_v63385) * (1.0 + 3.0 * i_v63385), 1e-300)), 1e-300)
            )
        ),
    )
    i_v64264 = i_v8881 - i_v64263
    i_v68230 = jnp.where(
        i_v57197,
        0.5
        * (
            i_v12712
            + i_v64264
            - jnp.sqrt(jnp.maximum((i_v12712 - i_v64264) * (i_v12712 - i_v64264) + 4.0 * i_v594 * i_v594, 1e-300))
        ),
        0.0,
    )
    i_v68233 = i_v2324 - i_v68230
    i_v69323 = i_v3628 * jnp.where(
        i_v68584, jnp.sqrt(jnp.maximum(i_v68233 * i_v3907, 1e-300)), jnp.power(jnp.maximum(i_v68233 * i_v3907, 0.0), i_v76)
    )
    i_v69710 = i_v1108 * ((i_v69700 - 1.0) * i_v69323)
    i_v68242 = jnp.where(i_v57197, i_v64263, 0.0)
    i_v68247 = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.divide(i_v68242, jnp.where((i_v68233 == 0.0) | jnp.logical_not(jnp.isfinite(i_v68233)), 1e-300, i_v68233)),
            1e-300,
        )
    )
    i_v68248 = 1.0 - i_v68247
    i_v68603 = i_v68248 + jnp.where(
        i_v68584,
        0.0,
        (jnp.divide(i_v68248 * i_v68248 * jnp.log(jnp.maximum(i_v68248, 1e-300)), i_v68247) + i_v68248) * (1.0 - 2.0 * i_v76),
    )
    i_v70128 = i_v4127 * jnp.divide(
        jnp.where(i_v67852, 0.0, i_v69323) * i_v2516,
        jnp.where(
            (jnp.where(i_v67852, 0.0, i_v68233) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v67852, 0.0, i_v68233))),
            1e-300,
            jnp.where(i_v67852, 0.0, i_v68233),
        ),
    )
    i_v70478 = jnp.divide(
        0.666666666666667 * i_v3979, jnp.where((i_v70128 == 0.0) | jnp.logical_not(jnp.isfinite(i_v70128)), 1e-300, i_v70128)
    )
    i_v70479 = i_v70478 * i_v70478
    i_v70480 = i_v70479 * i_v70479
    i_v70485 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v70480, jnp.where((i_v70480 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v70480 + 1.0)), 1e-300, i_v70480 + 1.0)
            ),
            1e-300,
        )
    )
    i_v70489 = jnp.sqrt(jnp.maximum(jnp.where(i_v70485 < 0.0, -i_v70485, i_v70485), 1e-300))
    i_v70890 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v70128, i_v70489), 1e-300))
    i_v70920 = (2.0 * (i_v70478 * i_v70489) - i_v70485 - 1.0) * i_v70890
    i_v70923 = i_v70920 > 0.0
    i_v71185 = jnp.where(
        i_v70923,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v70920 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v70920)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v70920,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v70920 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v70920)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v70920,
            ),
        ),
    )
    i_v71398 = i_v71185 * i_v71185
    i_v70493 = i_v70485 * i_v70489
    i_v70917 = i_v3979 * i_v70478 * i_v70489 - i_v3979 * i_v70485 + 0.5 * (i_v70128 * i_v70493)
    i_v71153 = -(i_v70920 * i_v70920) + i_v70917
    i_v71166 = -230.25850929940458 - i_v71153
    i_v71616 = (0.29214664 * i_v71185 + 0.26992878119627894 * i_v71398 + 0.43792457880372104 * (i_v71398 * i_v71185)) * jnp.where(
        i_v71153 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v71153, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v71166 * (1.0 + 0.5 * (i_v71166 * (1.0 + i_v71166 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(jnp.isfinite(1.0 + i_v71166 * (1.0 + 0.5 * (i_v71166 * (1.0 + i_v71166 * 0.3333333333333333))))),
                1e-300,
                1.0 + i_v71166 * (1.0 + 0.5 * (i_v71166 * (1.0 + i_v71166 * 0.3333333333333333))),
            ),
        ),
    )
    i_v71632 = -230.25850929940458 - i_v70917
    i_v70877 = jnp.where(i_v67852, 0.0, i_v68603)
    i_v70848 = -i_v76 * i_v2837
    i_v70880 = jnp.where(
        i_v70848 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v70128 * i_v70493 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v70128 * i_v70493)),
                1e-300,
                1.0 + i_v70128 * i_v70493,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v70128 * i_v70493, 0.0), i_v70848),
    )
    i_v72485 = jnp.where(
        i_v57197,
        0.5
        * (i_v12712 + i_v9052 - jnp.sqrt(jnp.maximum((i_v12712 - i_v9052) * (i_v12712 - i_v9052) + 4.0 * i_v590 * i_v590, 1e-300))),
        0.0,
    )
    i_v72942 = i_v2837 * jnp.divide(
        (i_v61 - i_v72485) * i_v3898,
        jnp.where(
            (
                jnp.where(
                    i_v76 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v61 - i_v72485) * i_v3907, 1e-300)),
                    jnp.power(jnp.maximum((i_v61 - i_v72485) * i_v3907, 0.0), i_v76),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v76 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v61 - i_v72485) * i_v3907, 1e-300)),
                        jnp.power(jnp.maximum((i_v61 - i_v72485) * i_v3907, 0.0), i_v76),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v76 == 0.5,
                jnp.sqrt(jnp.maximum((i_v61 - i_v72485) * i_v3907, 1e-300)),
                jnp.power(jnp.maximum((i_v61 - i_v72485) * i_v3907, 0.0), i_v76),
            ),
        ),
    )
    i_v73303 = jnp.divide(-i_v5102, jnp.where((i_v72942 == 0.0) | jnp.logical_not(jnp.isfinite(i_v72942)), 1e-300, i_v72942))
    i_v73330 = -230.25850929940458 - i_v73303
    i_v73351 = i_v73303 - 230.25850929940458
    i_v73812 = jnp.where(i_v57197, 0.5 * (i_v12712 - jnp.sqrt(jnp.maximum(i_v12712 * i_v12712 + 4e-12, 1e-300))), 0.0)
    i_v74789 = i_v73812 * i_v6514
    i_v74790 = i_v74789 < 0.0
    i_v74828 = i_v73812 * i_v6514
    i_v60835 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v60846 = NJH - (NJDV * (i_v12712 - i_v60835) + i_v281) - 0.01
    i_v60848 = 4.0 * NJH * 0.01
    i_v60867 = (
        NJH
        - 0.5 * (i_v60846 + jnp.sqrt(jnp.maximum(i_v60846 * i_v60846 + jnp.where(i_v60848 > 0.0, i_v60848, -i_v60848), 1e-300)))
        - i_v281
        - 0.01
    )
    i_v60869 = 4.0 * i_v281 * 0.01
    i_v60905 = jnp.where(
        i_v281 < NJH,
        i_v281
        + 0.5 * (i_v60867 + jnp.sqrt(jnp.maximum(i_v60867 * i_v60867 + jnp.where(i_v60869 > 0.0, i_v60869, -i_v60869), 1e-300))),
        i_v281,
    )
    i_v60921 = i_v596 * (
        jnp.divide(i_v12712, jnp.where((i_v60905 == 0.0) | jnp.logical_not(jnp.isfinite(i_v60905)), 1e-300, i_v60905))
        + jnp.divide(
            i_v60835 * (i_v60905 - i_v281),
            jnp.where((i_v281 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v281 * NJH)), 1e-300, i_v281 * NJH),
        )
    )
    i_v60981 = -230.25850929940458 - i_v60921
    i_v61020 = i_v60921 - 230.25850929940458
    i_v62622 = i_v281 < NJH
    i_v62621 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v62630 = NJH - (NJDV * (i_v7952 - i_v62621) + i_v281) - 0.01
    i_v62632 = 4.0 * NJH * 0.01
    i_v62643 = jnp.sqrt(jnp.maximum(i_v62630 * i_v62630 + jnp.where(i_v62632 > 0.0, i_v62632, -i_v62632), 1e-300))
    i_v62654 = NJH - 0.5 * (i_v62630 + i_v62643) - i_v281 - 0.01
    i_v62656 = 4.0 * i_v281 * 0.01
    i_v62667 = jnp.sqrt(jnp.maximum(i_v62654 * i_v62654 + jnp.where(i_v62656 > 0.0, i_v62656, -i_v62656), 1e-300))
    i_v62700, i_v62850 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v62622, _t, _f),
        (
            i_v281 + 0.5 * (i_v62654 + i_v62667),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v62630, i_v62643))) * (0.5 * (1.0 + jnp.divide(i_v62654, i_v62667))),
        ),
        (i_v281, 0.0),
    )
    i_v62713 = i_v281 * NJH
    i_v62716 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v62700 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62700)), 1e-300, i_v62700))
        + jnp.divide(
            i_v62621 * (i_v62700 - i_v281), jnp.where((i_v62713 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62713)), 1e-300, i_v62713)
        )
    )
    i_v62776 = -230.25850929940458 - i_v62716
    i_v62815 = i_v62716 - 230.25850929940458
    i_v78726 = jnp.where(i_v124 == 0.0, i_v151 == 0.0, False)
    i_v79596 = i_v82 == 0.5
    i_v79176 = i_v2333 - i_v68230
    i_v80335 = i_v3763 * jnp.where(
        i_v79596, jnp.sqrt(jnp.maximum(i_v79176 * i_v3910, 1e-300)), jnp.power(jnp.maximum(i_v79176 * i_v3910, 0.0), i_v82)
    )
    i_v80783 = i_v1117 * ((i_v69700 - 1.0) * i_v80335)
    i_v79259 = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.divide(i_v68242, jnp.where((i_v79176 == 0.0) | jnp.logical_not(jnp.isfinite(i_v79176)), 1e-300, i_v79176)),
            1e-300,
        )
    )
    i_v79260 = 1.0 - i_v79259
    i_v79615 = i_v79260 + jnp.where(
        i_v79596,
        0.0,
        (jnp.divide(i_v79260 * i_v79260 * jnp.log(jnp.maximum(i_v79260, 1e-300)), i_v79259) + i_v79260) * (1.0 - 2.0 * i_v82),
    )
    i_v81201 = i_v4262 * jnp.divide(
        jnp.where(i_v78726, 0.0, i_v80335) * i_v2678,
        jnp.where(
            (jnp.where(i_v78726, 0.0, i_v79176) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v78726, 0.0, i_v79176))),
            1e-300,
            jnp.where(i_v78726, 0.0, i_v79176),
        ),
    )
    i_v81551 = jnp.divide(
        0.666666666666667 * i_v3984, jnp.where((i_v81201 == 0.0) | jnp.logical_not(jnp.isfinite(i_v81201)), 1e-300, i_v81201)
    )
    i_v81552 = i_v81551 * i_v81551
    i_v81553 = i_v81552 * i_v81552
    i_v81558 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v81553, jnp.where((i_v81553 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v81553 + 1.0)), 1e-300, i_v81553 + 1.0)
            ),
            1e-300,
        )
    )
    i_v81562 = jnp.sqrt(jnp.maximum(jnp.where(i_v81558 < 0.0, -i_v81558, i_v81558), 1e-300))
    i_v81963 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v81201, i_v81562), 1e-300))
    i_v81993 = (2.0 * (i_v81551 * i_v81562) - i_v81558 - 1.0) * i_v81963
    i_v81996 = i_v81993 > 0.0
    i_v82126 = jnp.where(
        i_v81996,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v81993 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v81993)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v81993,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v81993 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v81993)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v81993,
            ),
        ),
    )
    i_v82207 = i_v82126 * i_v82126
    i_v81566 = i_v81558 * i_v81562
    i_v81990 = i_v3984 * i_v81551 * i_v81562 - i_v3984 * i_v81558 + 0.5 * (i_v81201 * i_v81566)
    i_v82094 = -(i_v81993 * i_v81993) + i_v81990
    i_v82107 = -230.25850929940458 - i_v82094
    i_v82293 = (0.29214664 * i_v82126 + 0.26992878119627894 * i_v82207 + 0.43792457880372104 * (i_v82207 * i_v82126)) * jnp.where(
        i_v82094 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v82094, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v82107 * (1.0 + 0.5 * (i_v82107 * (1.0 + i_v82107 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(jnp.isfinite(1.0 + i_v82107 * (1.0 + 0.5 * (i_v82107 * (1.0 + i_v82107 * 0.3333333333333333))))),
                1e-300,
                1.0 + i_v82107 * (1.0 + 0.5 * (i_v82107 * (1.0 + i_v82107 * 0.3333333333333333))),
            ),
        ),
    )
    i_v82309 = -230.25850929940458 - i_v81990
    i_v81950 = jnp.where(i_v78726, 0.0, i_v79615)
    i_v81921 = -i_v82 * i_v2840
    i_v81953 = jnp.where(
        i_v81921 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v81201 * i_v81566 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v81201 * i_v81566)),
                1e-300,
                1.0 + i_v81201 * i_v81566,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v81201 * i_v81566, 0.0), i_v81921),
    )
    i_v83655 = i_v2840 * jnp.divide(
        (i_v65 - i_v72485) * i_v3901,
        jnp.where(
            (
                jnp.where(
                    i_v82 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v65 - i_v72485) * i_v3910, 1e-300)),
                    jnp.power(jnp.maximum((i_v65 - i_v72485) * i_v3910, 0.0), i_v82),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v82 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v65 - i_v72485) * i_v3910, 1e-300)),
                        jnp.power(jnp.maximum((i_v65 - i_v72485) * i_v3910, 0.0), i_v82),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v82 == 0.5,
                jnp.sqrt(jnp.maximum((i_v65 - i_v72485) * i_v3910, 1e-300)),
                jnp.power(jnp.maximum((i_v65 - i_v72485) * i_v3910, 0.0), i_v82),
            ),
        ),
    )
    i_v84016 = jnp.divide(-i_v5110, jnp.where((i_v83655 == 0.0) | jnp.logical_not(jnp.isfinite(i_v83655)), 1e-300, i_v83655))
    i_v84043 = -230.25850929940458 - i_v84016
    i_v84064 = i_v84016 - 230.25850929940458
    i_v85381 = i_v73812 * i_v6546
    i_v85382 = i_v85381 < 0.0
    i_v85420 = i_v73812 * i_v6546
    i_v61694 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v61705 = NJH - (NJDV * (i_v12712 - i_v61694) + i_v285) - 0.01
    i_v61707 = 4.0 * NJH * 0.01
    i_v61726 = (
        NJH
        - 0.5 * (i_v61705 + jnp.sqrt(jnp.maximum(i_v61705 * i_v61705 + jnp.where(i_v61707 > 0.0, i_v61707, -i_v61707), 1e-300)))
        - i_v285
        - 0.01
    )
    i_v61728 = 4.0 * i_v285 * 0.01
    i_v61764 = jnp.where(
        i_v285 < NJH,
        i_v285
        + 0.5 * (i_v61726 + jnp.sqrt(jnp.maximum(i_v61726 * i_v61726 + jnp.where(i_v61728 > 0.0, i_v61728, -i_v61728), 1e-300))),
        i_v285,
    )
    i_v61780 = i_v596 * (
        jnp.divide(i_v12712, jnp.where((i_v61764 == 0.0) | jnp.logical_not(jnp.isfinite(i_v61764)), 1e-300, i_v61764))
        + jnp.divide(
            i_v61694 * (i_v61764 - i_v285),
            jnp.where((i_v285 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v285 * NJH)), 1e-300, i_v285 * NJH),
        )
    )
    i_v61840 = -230.25850929940458 - i_v61780
    i_v61879 = i_v61780 - 230.25850929940458
    i_v62992 = i_v285 < NJH
    i_v62991 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v63000 = NJH - (NJDV * (i_v7952 - i_v62991) + i_v285) - 0.01
    i_v63002 = 4.0 * NJH * 0.01
    i_v63013 = jnp.sqrt(jnp.maximum(i_v63000 * i_v63000 + jnp.where(i_v63002 > 0.0, i_v63002, -i_v63002), 1e-300))
    i_v63024 = NJH - 0.5 * (i_v63000 + i_v63013) - i_v285 - 0.01
    i_v63026 = 4.0 * i_v285 * 0.01
    i_v63037 = jnp.sqrt(jnp.maximum(i_v63024 * i_v63024 + jnp.where(i_v63026 > 0.0, i_v63026, -i_v63026), 1e-300))
    i_v63070, i_v63220 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v62992, _t, _f),
        (
            i_v285 + 0.5 * (i_v63024 + i_v63037),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v63000, i_v63013))) * (0.5 * (1.0 + jnp.divide(i_v63024, i_v63037))),
        ),
        (i_v285, 0.0),
    )
    i_v63083 = i_v285 * NJH
    i_v63086 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v63070 == 0.0) | jnp.logical_not(jnp.isfinite(i_v63070)), 1e-300, i_v63070))
        + jnp.divide(
            i_v62991 * (i_v63070 - i_v285), jnp.where((i_v63083 == 0.0) | jnp.logical_not(jnp.isfinite(i_v63083)), 1e-300, i_v63083)
        )
    )
    i_v63146 = -230.25850929940458 - i_v63086
    i_v63185 = i_v63086 - 230.25850929940458
    i_v89471 = jnp.where(i_v130 == 0.0, i_v157 == 0.0, False)
    i_v90341 = i_v88 == 0.5
    i_v89921 = i_v2342 - i_v68230
    i_v91080 = i_v3895 * jnp.where(
        i_v90341, jnp.sqrt(jnp.maximum(i_v89921 * i_v3913, 1e-300)), jnp.power(jnp.maximum(i_v89921 * i_v3913, 0.0), i_v88)
    )
    i_v91528 = i_v1126 * ((i_v69700 - 1.0) * i_v91080)
    i_v90004 = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.divide(i_v68242, jnp.where((i_v89921 == 0.0) | jnp.logical_not(jnp.isfinite(i_v89921)), 1e-300, i_v89921)),
            1e-300,
        )
    )
    i_v90005 = 1.0 - i_v90004
    i_v90360 = i_v90005 + jnp.where(
        i_v90341,
        0.0,
        (jnp.divide(i_v90005 * i_v90005 * jnp.log(jnp.maximum(i_v90005, 1e-300)), i_v90004) + i_v90005) * (1.0 - 2.0 * i_v88),
    )
    i_v91946 = i_v4394 * jnp.divide(
        jnp.where(i_v89471, 0.0, i_v91080) * i_v2834,
        jnp.where(
            (jnp.where(i_v89471, 0.0, i_v89921) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v89471, 0.0, i_v89921))),
            1e-300,
            jnp.where(i_v89471, 0.0, i_v89921),
        ),
    )
    i_v92296 = jnp.divide(
        0.666666666666667 * i_v3986, jnp.where((i_v91946 == 0.0) | jnp.logical_not(jnp.isfinite(i_v91946)), 1e-300, i_v91946)
    )
    i_v92297 = i_v92296 * i_v92296
    i_v92298 = i_v92297 * i_v92297
    i_v92303 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v92298, jnp.where((i_v92298 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v92298 + 1.0)), 1e-300, i_v92298 + 1.0)
            ),
            1e-300,
        )
    )
    i_v92307 = jnp.sqrt(jnp.maximum(jnp.where(i_v92303 < 0.0, -i_v92303, i_v92303), 1e-300))
    i_v92708 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v91946, i_v92307), 1e-300))
    i_v92738 = (2.0 * (i_v92296 * i_v92307) - i_v92303 - 1.0) * i_v92708
    i_v92741 = i_v92738 > 0.0
    i_v92871 = jnp.where(
        i_v92741,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v92738 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v92738)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v92738,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v92738 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v92738)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v92738,
            ),
        ),
    )
    i_v92952 = i_v92871 * i_v92871
    i_v92311 = i_v92303 * i_v92307
    i_v92735 = i_v3986 * i_v92296 * i_v92307 - i_v3986 * i_v92303 + 0.5 * (i_v91946 * i_v92311)
    i_v92839 = -(i_v92738 * i_v92738) + i_v92735
    i_v92852 = -230.25850929940458 - i_v92839
    i_v93038 = (0.29214664 * i_v92871 + 0.26992878119627894 * i_v92952 + 0.43792457880372104 * (i_v92952 * i_v92871)) * jnp.where(
        i_v92839 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v92839, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v92852 * (1.0 + 0.5 * (i_v92852 * (1.0 + i_v92852 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(jnp.isfinite(1.0 + i_v92852 * (1.0 + 0.5 * (i_v92852 * (1.0 + i_v92852 * 0.3333333333333333))))),
                1e-300,
                1.0 + i_v92852 * (1.0 + 0.5 * (i_v92852 * (1.0 + i_v92852 * 0.3333333333333333))),
            ),
        ),
    )
    i_v93054 = -230.25850929940458 - i_v92735
    i_v92695 = jnp.where(i_v89471, 0.0, i_v90360)
    i_v92666 = -i_v88 * i_v2843
    i_v92698 = jnp.where(
        i_v92666 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v91946 * i_v92311 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v91946 * i_v92311)),
                1e-300,
                1.0 + i_v91946 * i_v92311,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v91946 * i_v92311, 0.0), i_v92666),
    )
    i_v94400 = i_v2843 * jnp.divide(
        (i_v69 - i_v72485) * i_v3904,
        jnp.where(
            (
                jnp.where(
                    i_v88 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v69 - i_v72485) * i_v3913, 1e-300)),
                    jnp.power(jnp.maximum((i_v69 - i_v72485) * i_v3913, 0.0), i_v88),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v88 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v69 - i_v72485) * i_v3913, 1e-300)),
                        jnp.power(jnp.maximum((i_v69 - i_v72485) * i_v3913, 0.0), i_v88),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v88 == 0.5,
                jnp.sqrt(jnp.maximum((i_v69 - i_v72485) * i_v3913, 1e-300)),
                jnp.power(jnp.maximum((i_v69 - i_v72485) * i_v3913, 0.0), i_v88),
            ),
        ),
    )
    i_v94761 = jnp.divide(-i_v5121, jnp.where((i_v94400 == 0.0) | jnp.logical_not(jnp.isfinite(i_v94400)), 1e-300, i_v94400))
    i_v94788 = -230.25850929940458 - i_v94761
    i_v94809 = i_v94761 - 230.25850929940458
    i_v96126 = i_v73812 * i_v6575
    i_v96127 = i_v96126 < 0.0
    i_v96165 = i_v73812 * i_v6575
    i_v97834 = (
        i_v7324
        * jnp.where(
            i_v8640,
            0.0,
            (
                i_v1474
                * jnp.where(
                    i_v57197,
                    jnp.where(
                        i_v58628,
                        jnp.where(
                            jnp.where(i_v60026 < 0.0, -i_v60026, i_v60026) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v60026, -709.0, 709.0)),
                            jnp.where(
                                i_v60026 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v60086 * (1.0 + 0.5 * (i_v60086 * (1.0 + i_v60086 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v60086 * (1.0 + 0.5 * (i_v60086 * (1.0 + i_v60086 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v60086 * (1.0 + 0.5 * (i_v60086 * (1.0 + i_v60086 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v60125 * (1.0 + 0.5 * (i_v60125 * (1.0 + i_v60125 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v61907
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v62366 - i_v7952 * i_v62516,
                                        jnp.where(
                                            (i_v62366 * i_v62366 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62366 * i_v62366)),
                                            1e-300,
                                            i_v62366 * i_v62366,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v7258 * i_v62516,
                                        jnp.where((i_v62379 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62379)), 1e-300, i_v62379),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v62382 < 0.0, -i_v62382, i_v62382) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v62382, -709.0, 709.0)),
                            jnp.where(
                                i_v62382 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v62442 * (1.0 + 0.5 * (i_v62442 * (1.0 + i_v62442 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v62442 * (1.0 + 0.5 * (i_v62442 * (1.0 + i_v62442 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v62442 * (1.0 + 0.5 * (i_v62442 * (1.0 + i_v62442 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v62481 * (1.0 + 0.5 * (i_v62481 * (1.0 + i_v62481 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v67852, 0.0, i_v118 * (i_v69710 * i_v68603))
                + jnp.where(
                    i_v145 == 0.0,
                    0.0,
                    i_v145
                    * (
                        jnp.where(i_v67852, 0.0, i_v69710)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3979
                                * jnp.where(
                                    i_v70923,
                                    i_v71616,
                                    2.0
                                    * jnp.where(
                                        i_v70917 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v70917, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v71632 * (1.0 + 0.5 * (i_v71632 * (1.0 + i_v71632 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v71632
                                                        * (1.0 + 0.5 * (i_v71632 * (1.0 + i_v71632 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0 + i_v71632 * (1.0 + 0.5 * (i_v71632 * (1.0 + i_v71632 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v71616,
                                ),
                                i_v70890,
                            )
                        )
                        * jnp.divide(
                            i_v70877 * i_v70880,
                            jnp.where(
                                (i_v70877 + i_v70880 == 0.0) | jnp.logical_not(jnp.isfinite(i_v70877 + i_v70880)),
                                1e-300,
                                i_v70877 + i_v70880,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v176 == 0.0,
                    0.0,
                    i_v176
                    * (
                        i_v12712
                        * i_v72942
                        * i_v72942
                        * jnp.where(
                            jnp.where(i_v73303 < 0.0, -i_v73303, i_v73303) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v73303, -709.0, 709.0)),
                            jnp.where(
                                i_v73303 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v73330 * (1.0 + 0.5 * (i_v73330 * (1.0 + i_v73330 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v73330 * (1.0 + 0.5 * (i_v73330 * (1.0 + i_v73330 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v73330 * (1.0 + 0.5 * (i_v73330 * (1.0 + i_v73330 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v73351 * (1.0 + 0.5 * (i_v73351 * (1.0 + i_v73351 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v26813 > 1000.0,
                1.0,
                jnp.where(
                    i_v73812 > -i_v5226 * i_v26813,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v217 == 4.0,
                                    jnp.where(i_v74790, -i_v74789, i_v74789)
                                    * jnp.where(i_v74790, -i_v74789, i_v74789)
                                    * jnp.where(i_v74790, -i_v74789, i_v74789)
                                    * jnp.where(i_v74790, -i_v74789, i_v74789),
                                    jnp.power(jnp.maximum(jnp.where(i_v74828 < 0.0, -i_v74828, i_v74828), 0.0), i_v217),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v217 == 4.0,
                                        jnp.where(i_v74790, -i_v74789, i_v74789)
                                        * jnp.where(i_v74790, -i_v74789, i_v74789)
                                        * jnp.where(i_v74790, -i_v74789, i_v74789)
                                        * jnp.where(i_v74790, -i_v74789, i_v74789),
                                        jnp.power(jnp.maximum(jnp.where(i_v74828 < 0.0, -i_v74828, i_v74828), 0.0), i_v217),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v217 == 4.0,
                                jnp.where(i_v74790, -i_v74789, i_v74789)
                                * jnp.where(i_v74790, -i_v74789, i_v74789)
                                * jnp.where(i_v74790, -i_v74789, i_v74789)
                                * jnp.where(i_v74790, -i_v74789, i_v74789),
                                jnp.power(jnp.maximum(jnp.where(i_v74828 < 0.0, -i_v74828, i_v74828), 0.0), i_v217),
                            ),
                        ),
                    ),
                    i_v5340 + (i_v73812 + i_v5226 * i_v26813) * i_v6517,
                ),
            ),
        )
        + i_v7338
        * jnp.where(
            i_v8686,
            0.0,
            (
                i_v1624
                * jnp.where(
                    i_v57197,
                    jnp.where(
                        i_v58628,
                        jnp.where(
                            jnp.where(i_v60921 < 0.0, -i_v60921, i_v60921) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v60921, -709.0, 709.0)),
                            jnp.where(
                                i_v60921 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v60981 * (1.0 + 0.5 * (i_v60981 * (1.0 + i_v60981 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v60981 * (1.0 + 0.5 * (i_v60981 * (1.0 + i_v60981 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v60981 * (1.0 + 0.5 * (i_v60981 * (1.0 + i_v60981 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v61020 * (1.0 + 0.5 * (i_v61020 * (1.0 + i_v61020 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v61907
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v62700 - i_v7952 * i_v62850,
                                        jnp.where(
                                            (i_v62700 * i_v62700 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62700 * i_v62700)),
                                            1e-300,
                                            i_v62700 * i_v62700,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v62621 * i_v62850,
                                        jnp.where((i_v62713 == 0.0) | jnp.logical_not(jnp.isfinite(i_v62713)), 1e-300, i_v62713),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v62716 < 0.0, -i_v62716, i_v62716) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v62716, -709.0, 709.0)),
                            jnp.where(
                                i_v62716 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v62776 * (1.0 + 0.5 * (i_v62776 * (1.0 + i_v62776 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v62776 * (1.0 + 0.5 * (i_v62776 * (1.0 + i_v62776 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v62776 * (1.0 + 0.5 * (i_v62776 * (1.0 + i_v62776 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v62815 * (1.0 + 0.5 * (i_v62815 * (1.0 + i_v62815 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v78726, 0.0, i_v124 * (i_v80783 * i_v79615))
                + jnp.where(
                    i_v151 == 0.0,
                    0.0,
                    i_v151
                    * (
                        jnp.where(i_v78726, 0.0, i_v80783)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3984
                                * jnp.where(
                                    i_v81996,
                                    i_v82293,
                                    2.0
                                    * jnp.where(
                                        i_v81990 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v81990, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v82309 * (1.0 + 0.5 * (i_v82309 * (1.0 + i_v82309 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v82309
                                                        * (1.0 + 0.5 * (i_v82309 * (1.0 + i_v82309 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0 + i_v82309 * (1.0 + 0.5 * (i_v82309 * (1.0 + i_v82309 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v82293,
                                ),
                                i_v81963,
                            )
                        )
                        * jnp.divide(
                            i_v81950 * i_v81953,
                            jnp.where(
                                (i_v81950 + i_v81953 == 0.0) | jnp.logical_not(jnp.isfinite(i_v81950 + i_v81953)),
                                1e-300,
                                i_v81950 + i_v81953,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v182 == 0.0,
                    0.0,
                    i_v182
                    * (
                        i_v12712
                        * i_v83655
                        * i_v83655
                        * jnp.where(
                            jnp.where(i_v84016 < 0.0, -i_v84016, i_v84016) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v84016, -709.0, 709.0)),
                            jnp.where(
                                i_v84016 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v84043 * (1.0 + 0.5 * (i_v84043 * (1.0 + i_v84043 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v84043 * (1.0 + 0.5 * (i_v84043 * (1.0 + i_v84043 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v84043 * (1.0 + 0.5 * (i_v84043 * (1.0 + i_v84043 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v84064 * (1.0 + 0.5 * (i_v84064 * (1.0 + i_v84064 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v39291 > 1000.0,
                1.0,
                jnp.where(
                    i_v73812 > -i_v5226 * i_v39291,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v221 == 4.0,
                                    jnp.where(i_v85382, -i_v85381, i_v85381)
                                    * jnp.where(i_v85382, -i_v85381, i_v85381)
                                    * jnp.where(i_v85382, -i_v85381, i_v85381)
                                    * jnp.where(i_v85382, -i_v85381, i_v85381),
                                    jnp.power(jnp.maximum(jnp.where(i_v85420 < 0.0, -i_v85420, i_v85420), 0.0), i_v221),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v221 == 4.0,
                                        jnp.where(i_v85382, -i_v85381, i_v85381)
                                        * jnp.where(i_v85382, -i_v85381, i_v85381)
                                        * jnp.where(i_v85382, -i_v85381, i_v85381)
                                        * jnp.where(i_v85382, -i_v85381, i_v85381),
                                        jnp.power(jnp.maximum(jnp.where(i_v85420 < 0.0, -i_v85420, i_v85420), 0.0), i_v221),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v221 == 4.0,
                                jnp.where(i_v85382, -i_v85381, i_v85381)
                                * jnp.where(i_v85382, -i_v85381, i_v85381)
                                * jnp.where(i_v85382, -i_v85381, i_v85381)
                                * jnp.where(i_v85382, -i_v85381, i_v85381),
                                jnp.power(jnp.maximum(jnp.where(i_v85420 < 0.0, -i_v85420, i_v85420), 0.0), i_v221),
                            ),
                        ),
                    ),
                    i_v5451 + (i_v73812 + i_v5226 * i_v39291) * i_v6548,
                ),
            ),
        )
        + i_v7352
        * jnp.where(
            i_v8764,
            0.0,
            (
                i_v1771
                * jnp.where(
                    i_v57197,
                    jnp.where(
                        i_v58628,
                        jnp.where(
                            jnp.where(i_v61780 < 0.0, -i_v61780, i_v61780) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v61780, -709.0, 709.0)),
                            jnp.where(
                                i_v61780 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v61840 * (1.0 + 0.5 * (i_v61840 * (1.0 + i_v61840 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v61840 * (1.0 + 0.5 * (i_v61840 * (1.0 + i_v61840 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v61840 * (1.0 + 0.5 * (i_v61840 * (1.0 + i_v61840 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v61879 * (1.0 + 0.5 * (i_v61879 * (1.0 + i_v61879 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v61907
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v63070 - i_v7952 * i_v63220,
                                        jnp.where(
                                            (i_v63070 * i_v63070 == 0.0) | jnp.logical_not(jnp.isfinite(i_v63070 * i_v63070)),
                                            1e-300,
                                            i_v63070 * i_v63070,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v62991 * i_v63220,
                                        jnp.where((i_v63083 == 0.0) | jnp.logical_not(jnp.isfinite(i_v63083)), 1e-300, i_v63083),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v63086 < 0.0, -i_v63086, i_v63086) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v63086, -709.0, 709.0)),
                            jnp.where(
                                i_v63086 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v63146 * (1.0 + 0.5 * (i_v63146 * (1.0 + i_v63146 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v63146 * (1.0 + 0.5 * (i_v63146 * (1.0 + i_v63146 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v63146 * (1.0 + 0.5 * (i_v63146 * (1.0 + i_v63146 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v63185 * (1.0 + 0.5 * (i_v63185 * (1.0 + i_v63185 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v89471, 0.0, i_v130 * (i_v91528 * i_v90360))
                + jnp.where(
                    i_v157 == 0.0,
                    0.0,
                    i_v157
                    * (
                        jnp.where(i_v89471, 0.0, i_v91528)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3986
                                * jnp.where(
                                    i_v92741,
                                    i_v93038,
                                    2.0
                                    * jnp.where(
                                        i_v92735 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v92735, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v93054 * (1.0 + 0.5 * (i_v93054 * (1.0 + i_v93054 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v93054
                                                        * (1.0 + 0.5 * (i_v93054 * (1.0 + i_v93054 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0 + i_v93054 * (1.0 + 0.5 * (i_v93054 * (1.0 + i_v93054 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v93038,
                                ),
                                i_v92708,
                            )
                        )
                        * jnp.divide(
                            i_v92695 * i_v92698,
                            jnp.where(
                                (i_v92695 + i_v92698 == 0.0) | jnp.logical_not(jnp.isfinite(i_v92695 + i_v92698)),
                                1e-300,
                                i_v92695 + i_v92698,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v188 == 0.0,
                    0.0,
                    i_v188
                    * (
                        i_v12712
                        * i_v94400
                        * i_v94400
                        * jnp.where(
                            jnp.where(i_v94761 < 0.0, -i_v94761, i_v94761) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v94761, -709.0, 709.0)),
                            jnp.where(
                                i_v94761 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (1.0 + i_v94788 * (1.0 + 0.5 * (i_v94788 * (1.0 + i_v94788 * 0.3333333333333333))) == 0.0)
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v94788 * (1.0 + 0.5 * (i_v94788 * (1.0 + i_v94788 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v94788 * (1.0 + 0.5 * (i_v94788 * (1.0 + i_v94788 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v94809 * (1.0 + 0.5 * (i_v94809 * (1.0 + i_v94809 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v53512 > 1000.0,
                1.0,
                jnp.where(
                    i_v73812 > -i_v5226 * i_v53512,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v225 == 4.0,
                                    jnp.where(i_v96127, -i_v96126, i_v96126)
                                    * jnp.where(i_v96127, -i_v96126, i_v96126)
                                    * jnp.where(i_v96127, -i_v96126, i_v96126)
                                    * jnp.where(i_v96127, -i_v96126, i_v96126),
                                    jnp.power(jnp.maximum(jnp.where(i_v96165 < 0.0, -i_v96165, i_v96165), 0.0), i_v225),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v225 == 4.0,
                                        jnp.where(i_v96127, -i_v96126, i_v96126)
                                        * jnp.where(i_v96127, -i_v96126, i_v96126)
                                        * jnp.where(i_v96127, -i_v96126, i_v96126)
                                        * jnp.where(i_v96127, -i_v96126, i_v96126),
                                        jnp.power(jnp.maximum(jnp.where(i_v96165 < 0.0, -i_v96165, i_v96165), 0.0), i_v225),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v225 == 4.0,
                                jnp.where(i_v96127, -i_v96126, i_v96126)
                                * jnp.where(i_v96127, -i_v96126, i_v96126)
                                * jnp.where(i_v96127, -i_v96126, i_v96126)
                                * jnp.where(i_v96127, -i_v96126, i_v96126),
                                jnp.power(jnp.maximum(jnp.where(i_v96165 < 0.0, -i_v96165, i_v96165), 0.0), i_v225),
                            ),
                        ),
                    ),
                    i_v5559 + (i_v73812 + i_v5226 * i_v53512) * i_v6576,
                ),
            ),
        )
    )
    i_v97849 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v12715 = -0.8 * i_v369
    i_v99646 = i_v12715 < i_v7952
    i_v100969 = NJH - (NJDV * (i_v12715 - i_v7258) + i_v277) - 0.01
    i_v100971 = 4.0 * NJH * 0.01
    i_v100990 = (
        NJH
        - 0.5
        * (i_v100969 + jnp.sqrt(jnp.maximum(i_v100969 * i_v100969 + jnp.where(i_v100971 > 0.0, i_v100971, -i_v100971), 1e-300)))
        - i_v277
        - 0.01
    )
    i_v100992 = 4.0 * i_v277 * 0.01
    i_v101028 = jnp.where(
        i_v277 < NJH,
        i_v277
        + 0.5
        * (i_v100990 + jnp.sqrt(jnp.maximum(i_v100990 * i_v100990 + jnp.where(i_v100992 > 0.0, i_v100992, -i_v100992), 1e-300))),
        i_v277,
    )
    i_v101044 = i_v596 * (
        jnp.divide(i_v12715, jnp.where((i_v101028 == 0.0) | jnp.logical_not(jnp.isfinite(i_v101028)), 1e-300, i_v101028))
        + jnp.divide(
            i_v7258 * (i_v101028 - i_v277),
            jnp.where((i_v277 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v277 * NJH)), 1e-300, i_v277 * NJH),
        )
    )
    i_v101104 = -230.25850929940458 - i_v101044
    i_v101143 = i_v101044 - 230.25850929940458
    i_v102925 = i_v12715 - i_v7952
    i_v103306 = i_v277 < NJH
    i_v103314 = NJH - (NJDV * (i_v7952 - i_v7258) + i_v277) - 0.01
    i_v103316 = 4.0 * NJH * 0.01
    i_v103327 = jnp.sqrt(jnp.maximum(i_v103314 * i_v103314 + jnp.where(i_v103316 > 0.0, i_v103316, -i_v103316), 1e-300))
    i_v103338 = NJH - 0.5 * (i_v103314 + i_v103327) - i_v277 - 0.01
    i_v103340 = 4.0 * i_v277 * 0.01
    i_v103351 = jnp.sqrt(jnp.maximum(i_v103338 * i_v103338 + jnp.where(i_v103340 > 0.0, i_v103340, -i_v103340), 1e-300))
    i_v103384, i_v103534 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v103306, _t, _f),
        (
            i_v277 + 0.5 * (i_v103338 + i_v103351),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v103314, i_v103327))) * (0.5 * (1.0 + jnp.divide(i_v103338, i_v103351))),
        ),
        (i_v277, 0.0),
    )
    i_v103397 = i_v277 * NJH
    i_v103400 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v103384 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103384)), 1e-300, i_v103384))
        + jnp.divide(
            i_v7258 * (i_v103384 - i_v277),
            jnp.where((i_v103397 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103397)), 1e-300, i_v103397),
        )
    )
    i_v103460 = -230.25850929940458 - i_v103400
    i_v103499 = i_v103400 - 230.25850929940458
    i_v108870 = jnp.where(i_v118 == 0.0, i_v145 == 0.0, False)
    i_v99899 = 0.5 * (i_v12715 * i_v596)
    i_v99926 = -230.25850929940458 - i_v99899
    i_v99947 = i_v99899 - 230.25850929940458
    i_v104403 = jnp.where(
        i_v99646,
        jnp.where(
            jnp.where(i_v99899 < 0.0, -i_v99899, i_v99899) < 230.25850929940458,
            jnp.exp(jnp.clip(i_v99899, -709.0, 709.0)),
            jnp.where(
                i_v99899 < -230.25850929940458,
                jnp.divide(
                    1e-100,
                    jnp.where(
                        (1.0 + i_v99926 * (1.0 + 0.5 * (i_v99926 * (1.0 + i_v99926 * 0.3333333333333333))) == 0.0)
                        | jnp.logical_not(
                            jnp.isfinite(1.0 + i_v99926 * (1.0 + 0.5 * (i_v99926 * (1.0 + i_v99926 * 0.3333333333333333))))
                        ),
                        1e-300,
                        1.0 + i_v99926 * (1.0 + 0.5 * (i_v99926 * (1.0 + i_v99926 * 0.3333333333333333))),
                    ),
                ),
                1e100 * (1.0 + i_v99947 * (1.0 + 0.5 * (i_v99947 * (1.0 + i_v99947 * 0.3333333333333333)))),
            ),
        ),
        jnp.sqrt(jnp.maximum((1.0 + i_v102925 * i_v596) * i_v15213, 1e-300)),
    )
    i_v110718 = jnp.where(i_v97849, i_v104403, 0.0)
    i_v109602 = i_v76 == 0.5
    i_v104514 = jnp.divide(1.0, jnp.where((i_v104403 == 0.0) | jnp.logical_not(jnp.isfinite(i_v104403)), 1e-300, i_v104403))
    i_v105281 = jnp.where(
        i_v12715 > 0.0,
        2.0
        * (
            i_v594
            * jnp.log(jnp.maximum(2.0 + i_v104514 + jnp.sqrt(jnp.maximum((i_v104514 + 1.0) * (i_v104514 + 3.0), 1e-300)), 1e-300))
        ),
        -i_v12715
        + 2.0
        * (
            i_v594
            * jnp.log(
                jnp.maximum(
                    2.0 * i_v104403 + 1.0 + jnp.sqrt(jnp.maximum((1.0 + i_v104403) * (1.0 + 3.0 * i_v104403), 1e-300)), 1e-300
                )
            )
        ),
    )
    i_v105282 = i_v8881 - i_v105281
    i_v109248 = jnp.where(
        i_v97849,
        0.5
        * (
            i_v12715
            + i_v105282
            - jnp.sqrt(jnp.maximum((i_v12715 - i_v105282) * (i_v12715 - i_v105282) + 4.0 * i_v594 * i_v594, 1e-300))
        ),
        0.0,
    )
    i_v109251 = i_v2324 - i_v109248
    i_v110341 = i_v3628 * jnp.where(
        i_v109602, jnp.sqrt(jnp.maximum(i_v109251 * i_v3907, 1e-300)), jnp.power(jnp.maximum(i_v109251 * i_v3907, 0.0), i_v76)
    )
    i_v110728 = i_v1108 * ((i_v110718 - 1.0) * i_v110341)
    i_v109260 = jnp.where(i_v97849, i_v105281, 0.0)
    i_v109265 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v109260, jnp.where((i_v109251 == 0.0) | jnp.logical_not(jnp.isfinite(i_v109251)), 1e-300, i_v109251)),
            1e-300,
        )
    )
    i_v109266 = 1.0 - i_v109265
    i_v109621 = i_v109266 + jnp.where(
        i_v109602,
        0.0,
        (jnp.divide(i_v109266 * i_v109266 * jnp.log(jnp.maximum(i_v109266, 1e-300)), i_v109265) + i_v109266) * (1.0 - 2.0 * i_v76),
    )
    i_v111146 = i_v4127 * jnp.divide(
        jnp.where(i_v108870, 0.0, i_v110341) * i_v2516,
        jnp.where(
            (jnp.where(i_v108870, 0.0, i_v109251) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v108870, 0.0, i_v109251))),
            1e-300,
            jnp.where(i_v108870, 0.0, i_v109251),
        ),
    )
    i_v111496 = jnp.divide(
        0.666666666666667 * i_v3979, jnp.where((i_v111146 == 0.0) | jnp.logical_not(jnp.isfinite(i_v111146)), 1e-300, i_v111146)
    )
    i_v111497 = i_v111496 * i_v111496
    i_v111498 = i_v111497 * i_v111497
    i_v111503 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v111498,
                jnp.where((i_v111498 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v111498 + 1.0)), 1e-300, i_v111498 + 1.0),
            ),
            1e-300,
        )
    )
    i_v111507 = jnp.sqrt(jnp.maximum(jnp.where(i_v111503 < 0.0, -i_v111503, i_v111503), 1e-300))
    i_v111908 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v111146, i_v111507), 1e-300))
    i_v111938 = (2.0 * (i_v111496 * i_v111507) - i_v111503 - 1.0) * i_v111908
    i_v111941 = i_v111938 > 0.0
    i_v112203 = jnp.where(
        i_v111941,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v111938 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v111938)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v111938,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v111938 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v111938)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v111938,
            ),
        ),
    )
    i_v112416 = i_v112203 * i_v112203
    i_v111511 = i_v111503 * i_v111507
    i_v111935 = i_v3979 * i_v111496 * i_v111507 - i_v3979 * i_v111503 + 0.5 * (i_v111146 * i_v111511)
    i_v112171 = -(i_v111938 * i_v111938) + i_v111935
    i_v112184 = -230.25850929940458 - i_v112171
    i_v112634 = (
        0.29214664 * i_v112203 + 0.26992878119627894 * i_v112416 + 0.43792457880372104 * (i_v112416 * i_v112203)
    ) * jnp.where(
        i_v112171 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v112171, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v112184 * (1.0 + 0.5 * (i_v112184 * (1.0 + i_v112184 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v112184 * (1.0 + 0.5 * (i_v112184 * (1.0 + i_v112184 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v112184 * (1.0 + 0.5 * (i_v112184 * (1.0 + i_v112184 * 0.3333333333333333))),
            ),
        ),
    )
    i_v112650 = -230.25850929940458 - i_v111935
    i_v111895 = jnp.where(i_v108870, 0.0, i_v109621)
    i_v111866 = -i_v76 * i_v2837
    i_v111898 = jnp.where(
        i_v111866 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v111146 * i_v111511 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v111146 * i_v111511)),
                1e-300,
                1.0 + i_v111146 * i_v111511,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v111146 * i_v111511, 0.0), i_v111866),
    )
    i_v113503 = jnp.where(
        i_v97849,
        0.5
        * (i_v12715 + i_v9052 - jnp.sqrt(jnp.maximum((i_v12715 - i_v9052) * (i_v12715 - i_v9052) + 4.0 * i_v590 * i_v590, 1e-300))),
        0.0,
    )
    i_v113960 = i_v2837 * jnp.divide(
        (i_v61 - i_v113503) * i_v3898,
        jnp.where(
            (
                jnp.where(
                    i_v76 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v61 - i_v113503) * i_v3907, 1e-300)),
                    jnp.power(jnp.maximum((i_v61 - i_v113503) * i_v3907, 0.0), i_v76),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v76 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v61 - i_v113503) * i_v3907, 1e-300)),
                        jnp.power(jnp.maximum((i_v61 - i_v113503) * i_v3907, 0.0), i_v76),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v76 == 0.5,
                jnp.sqrt(jnp.maximum((i_v61 - i_v113503) * i_v3907, 1e-300)),
                jnp.power(jnp.maximum((i_v61 - i_v113503) * i_v3907, 0.0), i_v76),
            ),
        ),
    )
    i_v114321 = jnp.divide(-i_v5102, jnp.where((i_v113960 == 0.0) | jnp.logical_not(jnp.isfinite(i_v113960)), 1e-300, i_v113960))
    i_v114348 = -230.25850929940458 - i_v114321
    i_v114369 = i_v114321 - 230.25850929940458
    i_v114830 = jnp.where(i_v97849, 0.5 * (i_v12715 - jnp.sqrt(jnp.maximum(i_v12715 * i_v12715 + 4e-12, 1e-300))), 0.0)
    i_v115807 = i_v114830 * i_v6514
    i_v115808 = i_v115807 < 0.0
    i_v115846 = i_v114830 * i_v6514
    i_v101853 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v101864 = NJH - (NJDV * (i_v12715 - i_v101853) + i_v281) - 0.01
    i_v101866 = 4.0 * NJH * 0.01
    i_v101885 = (
        NJH
        - 0.5
        * (i_v101864 + jnp.sqrt(jnp.maximum(i_v101864 * i_v101864 + jnp.where(i_v101866 > 0.0, i_v101866, -i_v101866), 1e-300)))
        - i_v281
        - 0.01
    )
    i_v101887 = 4.0 * i_v281 * 0.01
    i_v101923 = jnp.where(
        i_v281 < NJH,
        i_v281
        + 0.5
        * (i_v101885 + jnp.sqrt(jnp.maximum(i_v101885 * i_v101885 + jnp.where(i_v101887 > 0.0, i_v101887, -i_v101887), 1e-300))),
        i_v281,
    )
    i_v101939 = i_v596 * (
        jnp.divide(i_v12715, jnp.where((i_v101923 == 0.0) | jnp.logical_not(jnp.isfinite(i_v101923)), 1e-300, i_v101923))
        + jnp.divide(
            i_v101853 * (i_v101923 - i_v281),
            jnp.where((i_v281 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v281 * NJH)), 1e-300, i_v281 * NJH),
        )
    )
    i_v101999 = -230.25850929940458 - i_v101939
    i_v102038 = i_v101939 - 230.25850929940458
    i_v103640 = i_v281 < NJH
    i_v103639 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v103648 = NJH - (NJDV * (i_v7952 - i_v103639) + i_v281) - 0.01
    i_v103650 = 4.0 * NJH * 0.01
    i_v103661 = jnp.sqrt(jnp.maximum(i_v103648 * i_v103648 + jnp.where(i_v103650 > 0.0, i_v103650, -i_v103650), 1e-300))
    i_v103672 = NJH - 0.5 * (i_v103648 + i_v103661) - i_v281 - 0.01
    i_v103674 = 4.0 * i_v281 * 0.01
    i_v103685 = jnp.sqrt(jnp.maximum(i_v103672 * i_v103672 + jnp.where(i_v103674 > 0.0, i_v103674, -i_v103674), 1e-300))
    i_v103718, i_v103868 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v103640, _t, _f),
        (
            i_v281 + 0.5 * (i_v103672 + i_v103685),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v103648, i_v103661))) * (0.5 * (1.0 + jnp.divide(i_v103672, i_v103685))),
        ),
        (i_v281, 0.0),
    )
    i_v103731 = i_v281 * NJH
    i_v103734 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v103718 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103718)), 1e-300, i_v103718))
        + jnp.divide(
            i_v103639 * (i_v103718 - i_v281),
            jnp.where((i_v103731 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103731)), 1e-300, i_v103731),
        )
    )
    i_v103794 = -230.25850929940458 - i_v103734
    i_v103833 = i_v103734 - 230.25850929940458
    i_v119744 = jnp.where(i_v124 == 0.0, i_v151 == 0.0, False)
    i_v120614 = i_v82 == 0.5
    i_v120194 = i_v2333 - i_v109248
    i_v121353 = i_v3763 * jnp.where(
        i_v120614, jnp.sqrt(jnp.maximum(i_v120194 * i_v3910, 1e-300)), jnp.power(jnp.maximum(i_v120194 * i_v3910, 0.0), i_v82)
    )
    i_v121801 = i_v1117 * ((i_v110718 - 1.0) * i_v121353)
    i_v120277 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v109260, jnp.where((i_v120194 == 0.0) | jnp.logical_not(jnp.isfinite(i_v120194)), 1e-300, i_v120194)),
            1e-300,
        )
    )
    i_v120278 = 1.0 - i_v120277
    i_v120633 = i_v120278 + jnp.where(
        i_v120614,
        0.0,
        (jnp.divide(i_v120278 * i_v120278 * jnp.log(jnp.maximum(i_v120278, 1e-300)), i_v120277) + i_v120278) * (1.0 - 2.0 * i_v82),
    )
    i_v122219 = i_v4262 * jnp.divide(
        jnp.where(i_v119744, 0.0, i_v121353) * i_v2678,
        jnp.where(
            (jnp.where(i_v119744, 0.0, i_v120194) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v119744, 0.0, i_v120194))),
            1e-300,
            jnp.where(i_v119744, 0.0, i_v120194),
        ),
    )
    i_v122569 = jnp.divide(
        0.666666666666667 * i_v3984, jnp.where((i_v122219 == 0.0) | jnp.logical_not(jnp.isfinite(i_v122219)), 1e-300, i_v122219)
    )
    i_v122570 = i_v122569 * i_v122569
    i_v122571 = i_v122570 * i_v122570
    i_v122576 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v122571,
                jnp.where((i_v122571 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v122571 + 1.0)), 1e-300, i_v122571 + 1.0),
            ),
            1e-300,
        )
    )
    i_v122580 = jnp.sqrt(jnp.maximum(jnp.where(i_v122576 < 0.0, -i_v122576, i_v122576), 1e-300))
    i_v122981 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v122219, i_v122580), 1e-300))
    i_v123011 = (2.0 * (i_v122569 * i_v122580) - i_v122576 - 1.0) * i_v122981
    i_v123014 = i_v123011 > 0.0
    i_v123144 = jnp.where(
        i_v123014,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v123011 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v123011)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v123011,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v123011 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v123011)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v123011,
            ),
        ),
    )
    i_v123225 = i_v123144 * i_v123144
    i_v122584 = i_v122576 * i_v122580
    i_v123008 = i_v3984 * i_v122569 * i_v122580 - i_v3984 * i_v122576 + 0.5 * (i_v122219 * i_v122584)
    i_v123112 = -(i_v123011 * i_v123011) + i_v123008
    i_v123125 = -230.25850929940458 - i_v123112
    i_v123311 = (
        0.29214664 * i_v123144 + 0.26992878119627894 * i_v123225 + 0.43792457880372104 * (i_v123225 * i_v123144)
    ) * jnp.where(
        i_v123112 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v123112, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v123125 * (1.0 + 0.5 * (i_v123125 * (1.0 + i_v123125 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v123125 * (1.0 + 0.5 * (i_v123125 * (1.0 + i_v123125 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v123125 * (1.0 + 0.5 * (i_v123125 * (1.0 + i_v123125 * 0.3333333333333333))),
            ),
        ),
    )
    i_v123327 = -230.25850929940458 - i_v123008
    i_v122968 = jnp.where(i_v119744, 0.0, i_v120633)
    i_v122939 = -i_v82 * i_v2840
    i_v122971 = jnp.where(
        i_v122939 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v122219 * i_v122584 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v122219 * i_v122584)),
                1e-300,
                1.0 + i_v122219 * i_v122584,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v122219 * i_v122584, 0.0), i_v122939),
    )
    i_v124673 = i_v2840 * jnp.divide(
        (i_v65 - i_v113503) * i_v3901,
        jnp.where(
            (
                jnp.where(
                    i_v82 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v65 - i_v113503) * i_v3910, 1e-300)),
                    jnp.power(jnp.maximum((i_v65 - i_v113503) * i_v3910, 0.0), i_v82),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v82 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v65 - i_v113503) * i_v3910, 1e-300)),
                        jnp.power(jnp.maximum((i_v65 - i_v113503) * i_v3910, 0.0), i_v82),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v82 == 0.5,
                jnp.sqrt(jnp.maximum((i_v65 - i_v113503) * i_v3910, 1e-300)),
                jnp.power(jnp.maximum((i_v65 - i_v113503) * i_v3910, 0.0), i_v82),
            ),
        ),
    )
    i_v125034 = jnp.divide(-i_v5110, jnp.where((i_v124673 == 0.0) | jnp.logical_not(jnp.isfinite(i_v124673)), 1e-300, i_v124673))
    i_v125061 = -230.25850929940458 - i_v125034
    i_v125082 = i_v125034 - 230.25850929940458
    i_v126399 = i_v114830 * i_v6546
    i_v126400 = i_v126399 < 0.0
    i_v126438 = i_v114830 * i_v6546
    i_v102712 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v102723 = NJH - (NJDV * (i_v12715 - i_v102712) + i_v285) - 0.01
    i_v102725 = 4.0 * NJH * 0.01
    i_v102744 = (
        NJH
        - 0.5
        * (i_v102723 + jnp.sqrt(jnp.maximum(i_v102723 * i_v102723 + jnp.where(i_v102725 > 0.0, i_v102725, -i_v102725), 1e-300)))
        - i_v285
        - 0.01
    )
    i_v102746 = 4.0 * i_v285 * 0.01
    i_v102782 = jnp.where(
        i_v285 < NJH,
        i_v285
        + 0.5
        * (i_v102744 + jnp.sqrt(jnp.maximum(i_v102744 * i_v102744 + jnp.where(i_v102746 > 0.0, i_v102746, -i_v102746), 1e-300))),
        i_v285,
    )
    i_v102798 = i_v596 * (
        jnp.divide(i_v12715, jnp.where((i_v102782 == 0.0) | jnp.logical_not(jnp.isfinite(i_v102782)), 1e-300, i_v102782))
        + jnp.divide(
            i_v102712 * (i_v102782 - i_v285),
            jnp.where((i_v285 * NJH == 0.0) | jnp.logical_not(jnp.isfinite(i_v285 * NJH)), 1e-300, i_v285 * NJH),
        )
    )
    i_v102858 = -230.25850929940458 - i_v102798
    i_v102897 = i_v102798 - 230.25850929940458
    i_v104010 = i_v285 < NJH
    i_v104009 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v104018 = NJH - (NJDV * (i_v7952 - i_v104009) + i_v285) - 0.01
    i_v104020 = 4.0 * NJH * 0.01
    i_v104031 = jnp.sqrt(jnp.maximum(i_v104018 * i_v104018 + jnp.where(i_v104020 > 0.0, i_v104020, -i_v104020), 1e-300))
    i_v104042 = NJH - 0.5 * (i_v104018 + i_v104031) - i_v285 - 0.01
    i_v104044 = 4.0 * i_v285 * 0.01
    i_v104055 = jnp.sqrt(jnp.maximum(i_v104042 * i_v104042 + jnp.where(i_v104044 > 0.0, i_v104044, -i_v104044), 1e-300))
    i_v104088, i_v104238 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v104010, _t, _f),
        (
            i_v285 + 0.5 * (i_v104042 + i_v104055),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v104018, i_v104031))) * (0.5 * (1.0 + jnp.divide(i_v104042, i_v104055))),
        ),
        (i_v285, 0.0),
    )
    i_v104101 = i_v285 * NJH
    i_v104104 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v104088 == 0.0) | jnp.logical_not(jnp.isfinite(i_v104088)), 1e-300, i_v104088))
        + jnp.divide(
            i_v104009 * (i_v104088 - i_v285),
            jnp.where((i_v104101 == 0.0) | jnp.logical_not(jnp.isfinite(i_v104101)), 1e-300, i_v104101),
        )
    )
    i_v104164 = -230.25850929940458 - i_v104104
    i_v104203 = i_v104104 - 230.25850929940458
    i_v130489 = jnp.where(i_v130 == 0.0, i_v157 == 0.0, False)
    i_v131359 = i_v88 == 0.5
    i_v130939 = i_v2342 - i_v109248
    i_v132098 = i_v3895 * jnp.where(
        i_v131359, jnp.sqrt(jnp.maximum(i_v130939 * i_v3913, 1e-300)), jnp.power(jnp.maximum(i_v130939 * i_v3913, 0.0), i_v88)
    )
    i_v132546 = i_v1126 * ((i_v110718 - 1.0) * i_v132098)
    i_v131022 = jnp.sqrt(
        jnp.maximum(
            1.0
            - jnp.divide(i_v109260, jnp.where((i_v130939 == 0.0) | jnp.logical_not(jnp.isfinite(i_v130939)), 1e-300, i_v130939)),
            1e-300,
        )
    )
    i_v131023 = 1.0 - i_v131022
    i_v131378 = i_v131023 + jnp.where(
        i_v131359,
        0.0,
        (jnp.divide(i_v131023 * i_v131023 * jnp.log(jnp.maximum(i_v131023, 1e-300)), i_v131022) + i_v131023) * (1.0 - 2.0 * i_v88),
    )
    i_v132964 = i_v4394 * jnp.divide(
        jnp.where(i_v130489, 0.0, i_v132098) * i_v2834,
        jnp.where(
            (jnp.where(i_v130489, 0.0, i_v130939) == 0.0) | jnp.logical_not(jnp.isfinite(jnp.where(i_v130489, 0.0, i_v130939))),
            1e-300,
            jnp.where(i_v130489, 0.0, i_v130939),
        ),
    )
    i_v133314 = jnp.divide(
        0.666666666666667 * i_v3986, jnp.where((i_v132964 == 0.0) | jnp.logical_not(jnp.isfinite(i_v132964)), 1e-300, i_v132964)
    )
    i_v133315 = i_v133314 * i_v133314
    i_v133316 = i_v133315 * i_v133315
    i_v133321 = jnp.sqrt(
        jnp.maximum(
            jnp.divide(
                i_v133316,
                jnp.where((i_v133316 + 1.0 == 0.0) | jnp.logical_not(jnp.isfinite(i_v133316 + 1.0)), 1e-300, i_v133316 + 1.0),
            ),
            1e-300,
        )
    )
    i_v133325 = jnp.sqrt(jnp.maximum(jnp.where(i_v133321 < 0.0, -i_v133321, i_v133321), 1e-300))
    i_v133726 = jnp.sqrt(jnp.maximum(0.375 * jnp.divide(i_v132964, i_v133325), 1e-300))
    i_v133756 = (2.0 * (i_v133314 * i_v133325) - i_v133321 - 1.0) * i_v133726
    i_v133759 = i_v133756 > 0.0
    i_v133889 = jnp.where(
        i_v133759,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + 0.5178164370971076 * i_v133756 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + 0.5178164370971076 * i_v133756)),
                1e-300,
                1.0 + 0.5178164370971076 * i_v133756,
            ),
        ),
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 - 0.5178164370971076 * i_v133756 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 - 0.5178164370971076 * i_v133756)),
                1e-300,
                1.0 - 0.5178164370971076 * i_v133756,
            ),
        ),
    )
    i_v133970 = i_v133889 * i_v133889
    i_v133329 = i_v133321 * i_v133325
    i_v133753 = i_v3986 * i_v133314 * i_v133325 - i_v3986 * i_v133321 + 0.5 * (i_v132964 * i_v133329)
    i_v133857 = -(i_v133756 * i_v133756) + i_v133753
    i_v133870 = -230.25850929940458 - i_v133857
    i_v134056 = (
        0.29214664 * i_v133889 + 0.26992878119627894 * i_v133970 + 0.43792457880372104 * (i_v133970 * i_v133889)
    ) * jnp.where(
        i_v133857 > -230.25850929940458,
        jnp.exp(jnp.clip(i_v133857, -709.0, 709.0)),
        jnp.divide(
            1e-100,
            jnp.where(
                (1.0 + i_v133870 * (1.0 + 0.5 * (i_v133870 * (1.0 + i_v133870 * 0.3333333333333333))) == 0.0)
                | jnp.logical_not(
                    jnp.isfinite(1.0 + i_v133870 * (1.0 + 0.5 * (i_v133870 * (1.0 + i_v133870 * 0.3333333333333333))))
                ),
                1e-300,
                1.0 + i_v133870 * (1.0 + 0.5 * (i_v133870 * (1.0 + i_v133870 * 0.3333333333333333))),
            ),
        ),
    )
    i_v134072 = -230.25850929940458 - i_v133753
    i_v133713 = jnp.where(i_v130489, 0.0, i_v131378)
    i_v133684 = -i_v88 * i_v2843
    i_v133716 = jnp.where(
        i_v133684 == -1.0,
        jnp.divide(
            1.0,
            jnp.where(
                (1.0 + i_v132964 * i_v133329 == 0.0) | jnp.logical_not(jnp.isfinite(1.0 + i_v132964 * i_v133329)),
                1e-300,
                1.0 + i_v132964 * i_v133329,
            ),
        ),
        jnp.power(jnp.maximum(1.0 + i_v132964 * i_v133329, 0.0), i_v133684),
    )
    i_v135418 = i_v2843 * jnp.divide(
        (i_v69 - i_v113503) * i_v3904,
        jnp.where(
            (
                jnp.where(
                    i_v88 == 0.5,
                    jnp.sqrt(jnp.maximum((i_v69 - i_v113503) * i_v3913, 1e-300)),
                    jnp.power(jnp.maximum((i_v69 - i_v113503) * i_v3913, 0.0), i_v88),
                )
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.where(
                        i_v88 == 0.5,
                        jnp.sqrt(jnp.maximum((i_v69 - i_v113503) * i_v3913, 1e-300)),
                        jnp.power(jnp.maximum((i_v69 - i_v113503) * i_v3913, 0.0), i_v88),
                    )
                )
            ),
            1e-300,
            jnp.where(
                i_v88 == 0.5,
                jnp.sqrt(jnp.maximum((i_v69 - i_v113503) * i_v3913, 1e-300)),
                jnp.power(jnp.maximum((i_v69 - i_v113503) * i_v3913, 0.0), i_v88),
            ),
        ),
    )
    i_v135779 = jnp.divide(-i_v5121, jnp.where((i_v135418 == 0.0) | jnp.logical_not(jnp.isfinite(i_v135418)), 1e-300, i_v135418))
    i_v135806 = -230.25850929940458 - i_v135779
    i_v135827 = i_v135779 - 230.25850929940458
    i_v137144 = i_v114830 * i_v6575
    i_v137145 = i_v137144 < 0.0
    i_v137183 = i_v114830 * i_v6575
    i_v138852 = (
        i_v7324
        * jnp.where(
            i_v8640,
            0.0,
            (
                i_v1474
                * jnp.where(
                    i_v97849,
                    jnp.where(
                        i_v99646,
                        jnp.where(
                            jnp.where(i_v101044 < 0.0, -i_v101044, i_v101044) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v101044, -709.0, 709.0)),
                            jnp.where(
                                i_v101044 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v101104 * (1.0 + 0.5 * (i_v101104 * (1.0 + i_v101104 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v101104 * (1.0 + 0.5 * (i_v101104 * (1.0 + i_v101104 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v101104 * (1.0 + 0.5 * (i_v101104 * (1.0 + i_v101104 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v101143 * (1.0 + 0.5 * (i_v101143 * (1.0 + i_v101143 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v102925
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v103384 - i_v7952 * i_v103534,
                                        jnp.where(
                                            (i_v103384 * i_v103384 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103384 * i_v103384)),
                                            1e-300,
                                            i_v103384 * i_v103384,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v7258 * i_v103534,
                                        jnp.where((i_v103397 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103397)), 1e-300, i_v103397),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v103400 < 0.0, -i_v103400, i_v103400) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v103400, -709.0, 709.0)),
                            jnp.where(
                                i_v103400 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v103460 * (1.0 + 0.5 * (i_v103460 * (1.0 + i_v103460 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v103460 * (1.0 + 0.5 * (i_v103460 * (1.0 + i_v103460 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v103460 * (1.0 + 0.5 * (i_v103460 * (1.0 + i_v103460 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v103499 * (1.0 + 0.5 * (i_v103499 * (1.0 + i_v103499 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v108870, 0.0, i_v118 * (i_v110728 * i_v109621))
                + jnp.where(
                    i_v145 == 0.0,
                    0.0,
                    i_v145
                    * (
                        jnp.where(i_v108870, 0.0, i_v110728)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3979
                                * jnp.where(
                                    i_v111941,
                                    i_v112634,
                                    2.0
                                    * jnp.where(
                                        i_v111935 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v111935, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v112650 * (1.0 + 0.5 * (i_v112650 * (1.0 + i_v112650 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v112650
                                                        * (1.0 + 0.5 * (i_v112650 * (1.0 + i_v112650 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v112650 * (1.0 + 0.5 * (i_v112650 * (1.0 + i_v112650 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v112634,
                                ),
                                i_v111908,
                            )
                        )
                        * jnp.divide(
                            i_v111895 * i_v111898,
                            jnp.where(
                                (i_v111895 + i_v111898 == 0.0) | jnp.logical_not(jnp.isfinite(i_v111895 + i_v111898)),
                                1e-300,
                                i_v111895 + i_v111898,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v176 == 0.0,
                    0.0,
                    i_v176
                    * (
                        i_v12715
                        * i_v113960
                        * i_v113960
                        * jnp.where(
                            jnp.where(i_v114321 < 0.0, -i_v114321, i_v114321) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v114321, -709.0, 709.0)),
                            jnp.where(
                                i_v114321 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v114348 * (1.0 + 0.5 * (i_v114348 * (1.0 + i_v114348 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v114348 * (1.0 + 0.5 * (i_v114348 * (1.0 + i_v114348 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v114348 * (1.0 + 0.5 * (i_v114348 * (1.0 + i_v114348 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v114369 * (1.0 + 0.5 * (i_v114369 * (1.0 + i_v114369 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v26813 > 1000.0,
                1.0,
                jnp.where(
                    i_v114830 > -i_v5226 * i_v26813,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v217 == 4.0,
                                    jnp.where(i_v115808, -i_v115807, i_v115807)
                                    * jnp.where(i_v115808, -i_v115807, i_v115807)
                                    * jnp.where(i_v115808, -i_v115807, i_v115807)
                                    * jnp.where(i_v115808, -i_v115807, i_v115807),
                                    jnp.power(jnp.maximum(jnp.where(i_v115846 < 0.0, -i_v115846, i_v115846), 0.0), i_v217),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v217 == 4.0,
                                        jnp.where(i_v115808, -i_v115807, i_v115807)
                                        * jnp.where(i_v115808, -i_v115807, i_v115807)
                                        * jnp.where(i_v115808, -i_v115807, i_v115807)
                                        * jnp.where(i_v115808, -i_v115807, i_v115807),
                                        jnp.power(jnp.maximum(jnp.where(i_v115846 < 0.0, -i_v115846, i_v115846), 0.0), i_v217),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v217 == 4.0,
                                jnp.where(i_v115808, -i_v115807, i_v115807)
                                * jnp.where(i_v115808, -i_v115807, i_v115807)
                                * jnp.where(i_v115808, -i_v115807, i_v115807)
                                * jnp.where(i_v115808, -i_v115807, i_v115807),
                                jnp.power(jnp.maximum(jnp.where(i_v115846 < 0.0, -i_v115846, i_v115846), 0.0), i_v217),
                            ),
                        ),
                    ),
                    i_v5340 + (i_v114830 + i_v5226 * i_v26813) * i_v6517,
                ),
            ),
        )
        + i_v7338
        * jnp.where(
            i_v8686,
            0.0,
            (
                i_v1624
                * jnp.where(
                    i_v97849,
                    jnp.where(
                        i_v99646,
                        jnp.where(
                            jnp.where(i_v101939 < 0.0, -i_v101939, i_v101939) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v101939, -709.0, 709.0)),
                            jnp.where(
                                i_v101939 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v101999 * (1.0 + 0.5 * (i_v101999 * (1.0 + i_v101999 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v101999 * (1.0 + 0.5 * (i_v101999 * (1.0 + i_v101999 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v101999 * (1.0 + 0.5 * (i_v101999 * (1.0 + i_v101999 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v102038 * (1.0 + 0.5 * (i_v102038 * (1.0 + i_v102038 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v102925
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v103718 - i_v7952 * i_v103868,
                                        jnp.where(
                                            (i_v103718 * i_v103718 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103718 * i_v103718)),
                                            1e-300,
                                            i_v103718 * i_v103718,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v103639 * i_v103868,
                                        jnp.where((i_v103731 == 0.0) | jnp.logical_not(jnp.isfinite(i_v103731)), 1e-300, i_v103731),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v103734 < 0.0, -i_v103734, i_v103734) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v103734, -709.0, 709.0)),
                            jnp.where(
                                i_v103734 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v103794 * (1.0 + 0.5 * (i_v103794 * (1.0 + i_v103794 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v103794 * (1.0 + 0.5 * (i_v103794 * (1.0 + i_v103794 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v103794 * (1.0 + 0.5 * (i_v103794 * (1.0 + i_v103794 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v103833 * (1.0 + 0.5 * (i_v103833 * (1.0 + i_v103833 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v119744, 0.0, i_v124 * (i_v121801 * i_v120633))
                + jnp.where(
                    i_v151 == 0.0,
                    0.0,
                    i_v151
                    * (
                        jnp.where(i_v119744, 0.0, i_v121801)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3984
                                * jnp.where(
                                    i_v123014,
                                    i_v123311,
                                    2.0
                                    * jnp.where(
                                        i_v123008 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v123008, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v123327 * (1.0 + 0.5 * (i_v123327 * (1.0 + i_v123327 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v123327
                                                        * (1.0 + 0.5 * (i_v123327 * (1.0 + i_v123327 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v123327 * (1.0 + 0.5 * (i_v123327 * (1.0 + i_v123327 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v123311,
                                ),
                                i_v122981,
                            )
                        )
                        * jnp.divide(
                            i_v122968 * i_v122971,
                            jnp.where(
                                (i_v122968 + i_v122971 == 0.0) | jnp.logical_not(jnp.isfinite(i_v122968 + i_v122971)),
                                1e-300,
                                i_v122968 + i_v122971,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v182 == 0.0,
                    0.0,
                    i_v182
                    * (
                        i_v12715
                        * i_v124673
                        * i_v124673
                        * jnp.where(
                            jnp.where(i_v125034 < 0.0, -i_v125034, i_v125034) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v125034, -709.0, 709.0)),
                            jnp.where(
                                i_v125034 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v125061 * (1.0 + 0.5 * (i_v125061 * (1.0 + i_v125061 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v125061 * (1.0 + 0.5 * (i_v125061 * (1.0 + i_v125061 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v125061 * (1.0 + 0.5 * (i_v125061 * (1.0 + i_v125061 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v125082 * (1.0 + 0.5 * (i_v125082 * (1.0 + i_v125082 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v39291 > 1000.0,
                1.0,
                jnp.where(
                    i_v114830 > -i_v5226 * i_v39291,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v221 == 4.0,
                                    jnp.where(i_v126400, -i_v126399, i_v126399)
                                    * jnp.where(i_v126400, -i_v126399, i_v126399)
                                    * jnp.where(i_v126400, -i_v126399, i_v126399)
                                    * jnp.where(i_v126400, -i_v126399, i_v126399),
                                    jnp.power(jnp.maximum(jnp.where(i_v126438 < 0.0, -i_v126438, i_v126438), 0.0), i_v221),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v221 == 4.0,
                                        jnp.where(i_v126400, -i_v126399, i_v126399)
                                        * jnp.where(i_v126400, -i_v126399, i_v126399)
                                        * jnp.where(i_v126400, -i_v126399, i_v126399)
                                        * jnp.where(i_v126400, -i_v126399, i_v126399),
                                        jnp.power(jnp.maximum(jnp.where(i_v126438 < 0.0, -i_v126438, i_v126438), 0.0), i_v221),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v221 == 4.0,
                                jnp.where(i_v126400, -i_v126399, i_v126399)
                                * jnp.where(i_v126400, -i_v126399, i_v126399)
                                * jnp.where(i_v126400, -i_v126399, i_v126399)
                                * jnp.where(i_v126400, -i_v126399, i_v126399),
                                jnp.power(jnp.maximum(jnp.where(i_v126438 < 0.0, -i_v126438, i_v126438), 0.0), i_v221),
                            ),
                        ),
                    ),
                    i_v5451 + (i_v114830 + i_v5226 * i_v39291) * i_v6548,
                ),
            ),
        )
        + i_v7352
        * jnp.where(
            i_v8764,
            0.0,
            (
                i_v1771
                * jnp.where(
                    i_v97849,
                    jnp.where(
                        i_v99646,
                        jnp.where(
                            jnp.where(i_v102798 < 0.0, -i_v102798, i_v102798) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v102798, -709.0, 709.0)),
                            jnp.where(
                                i_v102798 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v102858 * (1.0 + 0.5 * (i_v102858 * (1.0 + i_v102858 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v102858 * (1.0 + 0.5 * (i_v102858 * (1.0 + i_v102858 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v102858 * (1.0 + 0.5 * (i_v102858 * (1.0 + i_v102858 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v102897 * (1.0 + 0.5 * (i_v102897 * (1.0 + i_v102897 * 0.3333333333333333)))),
                            ),
                        ),
                        (
                            1.0
                            + i_v102925
                            * (
                                i_v596
                                * (
                                    jnp.divide(
                                        i_v104088 - i_v7952 * i_v104238,
                                        jnp.where(
                                            (i_v104088 * i_v104088 == 0.0) | jnp.logical_not(jnp.isfinite(i_v104088 * i_v104088)),
                                            1e-300,
                                            i_v104088 * i_v104088,
                                        ),
                                    )
                                    + jnp.divide(
                                        i_v104009 * i_v104238,
                                        jnp.where((i_v104101 == 0.0) | jnp.logical_not(jnp.isfinite(i_v104101)), 1e-300, i_v104101),
                                    )
                                )
                            )
                        )
                        * jnp.where(
                            jnp.where(i_v104104 < 0.0, -i_v104104, i_v104104) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v104104, -709.0, 709.0)),
                            jnp.where(
                                i_v104104 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v104164 * (1.0 + 0.5 * (i_v104164 * (1.0 + i_v104164 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v104164 * (1.0 + 0.5 * (i_v104164 * (1.0 + i_v104164 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v104164 * (1.0 + 0.5 * (i_v104164 * (1.0 + i_v104164 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v104203 * (1.0 + 0.5 * (i_v104203 * (1.0 + i_v104203 * 0.3333333333333333)))),
                            ),
                        ),
                    )
                    - 1.0,
                    0.0,
                )
                + jnp.where(i_v130489, 0.0, i_v130 * (i_v132546 * i_v131378))
                + jnp.where(
                    i_v157 == 0.0,
                    0.0,
                    i_v157
                    * (
                        jnp.where(i_v130489, 0.0, i_v132546)
                        * (
                            0.886226925452758
                            * jnp.divide(
                                i_v3986
                                * jnp.where(
                                    i_v133759,
                                    i_v134056,
                                    2.0
                                    * jnp.where(
                                        i_v133753 > -230.25850929940458,
                                        jnp.exp(jnp.clip(i_v133753, -709.0, 709.0)),
                                        jnp.divide(
                                            1e-100,
                                            jnp.where(
                                                (
                                                    1.0
                                                    + i_v134072 * (1.0 + 0.5 * (i_v134072 * (1.0 + i_v134072 * 0.3333333333333333)))
                                                    == 0.0
                                                )
                                                | jnp.logical_not(
                                                    jnp.isfinite(
                                                        1.0
                                                        + i_v134072
                                                        * (1.0 + 0.5 * (i_v134072 * (1.0 + i_v134072 * 0.3333333333333333)))
                                                    )
                                                ),
                                                1e-300,
                                                1.0
                                                + i_v134072 * (1.0 + 0.5 * (i_v134072 * (1.0 + i_v134072 * 0.3333333333333333))),
                                            ),
                                        ),
                                    )
                                    - i_v134056,
                                ),
                                i_v133726,
                            )
                        )
                        * jnp.divide(
                            i_v133713 * i_v133716,
                            jnp.where(
                                (i_v133713 + i_v133716 == 0.0) | jnp.logical_not(jnp.isfinite(i_v133713 + i_v133716)),
                                1e-300,
                                i_v133713 + i_v133716,
                            ),
                        )
                    ),
                )
                + jnp.where(
                    i_v188 == 0.0,
                    0.0,
                    i_v188
                    * (
                        i_v12715
                        * i_v135418
                        * i_v135418
                        * jnp.where(
                            jnp.where(i_v135779 < 0.0, -i_v135779, i_v135779) < 230.25850929940458,
                            jnp.exp(jnp.clip(i_v135779, -709.0, 709.0)),
                            jnp.where(
                                i_v135779 < -230.25850929940458,
                                jnp.divide(
                                    1e-100,
                                    jnp.where(
                                        (
                                            1.0 + i_v135806 * (1.0 + 0.5 * (i_v135806 * (1.0 + i_v135806 * 0.3333333333333333)))
                                            == 0.0
                                        )
                                        | jnp.logical_not(
                                            jnp.isfinite(
                                                1.0 + i_v135806 * (1.0 + 0.5 * (i_v135806 * (1.0 + i_v135806 * 0.3333333333333333)))
                                            )
                                        ),
                                        1e-300,
                                        1.0 + i_v135806 * (1.0 + 0.5 * (i_v135806 * (1.0 + i_v135806 * 0.3333333333333333))),
                                    ),
                                ),
                                1e100 * (1.0 + i_v135827 * (1.0 + 0.5 * (i_v135827 * (1.0 + i_v135827 * 0.3333333333333333)))),
                            ),
                        )
                    ),
                )
            )
            * jnp.where(
                i_v53512 > 1000.0,
                1.0,
                jnp.where(
                    i_v114830 > -i_v5226 * i_v53512,
                    jnp.divide(
                        1.0,
                        jnp.where(
                            (
                                1.0
                                - jnp.where(
                                    i_v225 == 4.0,
                                    jnp.where(i_v137145, -i_v137144, i_v137144)
                                    * jnp.where(i_v137145, -i_v137144, i_v137144)
                                    * jnp.where(i_v137145, -i_v137144, i_v137144)
                                    * jnp.where(i_v137145, -i_v137144, i_v137144),
                                    jnp.power(jnp.maximum(jnp.where(i_v137183 < 0.0, -i_v137183, i_v137183), 0.0), i_v225),
                                )
                                == 0.0
                            )
                            | jnp.logical_not(
                                jnp.isfinite(
                                    1.0
                                    - jnp.where(
                                        i_v225 == 4.0,
                                        jnp.where(i_v137145, -i_v137144, i_v137144)
                                        * jnp.where(i_v137145, -i_v137144, i_v137144)
                                        * jnp.where(i_v137145, -i_v137144, i_v137144)
                                        * jnp.where(i_v137145, -i_v137144, i_v137144),
                                        jnp.power(jnp.maximum(jnp.where(i_v137183 < 0.0, -i_v137183, i_v137183), 0.0), i_v225),
                                    )
                                )
                            ),
                            1e-300,
                            1.0
                            - jnp.where(
                                i_v225 == 4.0,
                                jnp.where(i_v137145, -i_v137144, i_v137144)
                                * jnp.where(i_v137145, -i_v137144, i_v137144)
                                * jnp.where(i_v137145, -i_v137144, i_v137144)
                                * jnp.where(i_v137145, -i_v137144, i_v137144),
                                jnp.power(jnp.maximum(jnp.where(i_v137183 < 0.0, -i_v137183, i_v137183), 0.0), i_v225),
                            ),
                        ),
                    ),
                    i_v5559 + (i_v114830 + i_v5226 * i_v53512) * i_v6576,
                ),
            ),
        )
    )
    i_v238073 = jnp.where(jnp.where(i_v57182 < 0.0, i_v97834 < 0.0, False), i_v138852 < 0.0, False)
    i_v230312 = i_v12709 * i_v596
    i_v234220 = (
        i_v57182
        - i_v10873 * (jnp.exp(jnp.clip(i_v230312 * i_v224964, -709.0, 709.0)) - 1.0)
        - i_v230343 * (jnp.exp(jnp.clip(i_v230312 * i_v232280, -709.0, 709.0)) - 1.0)
    )
    i_v236495 = i_v12712 * i_v596
    i_v236506 = (
        i_v97834
        - i_v10873 * (jnp.exp(jnp.clip(i_v236495 * i_v224964, -709.0, 709.0)) - 1.0)
        - i_v230343 * (jnp.exp(jnp.clip(i_v236495 * i_v232280, -709.0, 709.0)) - 1.0)
    )
    i_v238049 = i_v12715 * i_v596
    i_v238060 = (
        i_v138852
        - i_v10873 * (jnp.exp(jnp.clip(i_v238049 * i_v224964, -709.0, 709.0)) - 1.0)
        - i_v230343 * (jnp.exp(jnp.clip(i_v238049 * i_v232280, -709.0, 709.0)) - 1.0)
    )
    i_v238162 = jnp.where(
        jnp.where(
            jnp.where(
                jnp.where(
                    jnp.where(
                        jnp.divide(
                            i_v234220, jnp.where((i_v57182 == 0.0) | jnp.logical_not(jnp.isfinite(i_v57182)), 1e-300, i_v57182)
                        )
                        > 0.001,
                        True,
                        jnp.divide(
                            i_v236506, jnp.where((i_v97834 == 0.0) | jnp.logical_not(jnp.isfinite(i_v97834)), 1e-300, i_v97834)
                        )
                        > 0.001,
                    ),
                    True,
                    jnp.divide(
                        i_v238060, jnp.where((i_v138852 == 0.0) | jnp.logical_not(jnp.isfinite(i_v138852)), 1e-300, i_v138852)
                    )
                    > 0.001,
                ),
                i_v234220 < 0.0,
                False,
            ),
            i_v236506 < 0.0,
            False,
        ),
        i_v238060 < 0.0,
        False,
    )
    i_v238178 = jnp.divide(i_v234220, jnp.where((i_v236506 == 0.0) | jnp.logical_not(jnp.isfinite(i_v236506)), 1e-300, i_v236506))
    i_v238255 = i_v12709 - i_v12712
    i_v238257 = i_v12712 - i_v12709
    i_v238275 = jnp.divide(
        -i_v594 * jnp.log(jnp.maximum(i_v238178, 1e-300)),
        jnp.where((i_v238255 == 0.0) | jnp.logical_not(jnp.isfinite(i_v238255)), 1e-300, i_v238255),
    ) + jnp.divide(
        i_v594
        * (i_v238178 - 1.0)
        * (
            jnp.power(
                jnp.maximum(i_v238178, 0.0),
                jnp.divide(i_v12712, jnp.where((i_v238257 == 0.0) | jnp.logical_not(jnp.isfinite(i_v238257)), 1e-300, i_v238257)),
            )
            - 1.0
        ),
        jnp.where(
            (
                jnp.power(
                    jnp.maximum(i_v238178, 0.0),
                    jnp.divide(
                        i_v12709, jnp.where((i_v238255 == 0.0) | jnp.logical_not(jnp.isfinite(i_v238255)), 1e-300, i_v238255)
                    ),
                )
                * i_v238257
                + i_v238178 * i_v12709
                - i_v12712
                == 0.0
            )
            | jnp.logical_not(
                jnp.isfinite(
                    jnp.power(
                        jnp.maximum(i_v238178, 0.0),
                        jnp.divide(
                            i_v12709, jnp.where((i_v238255 == 0.0) | jnp.logical_not(jnp.isfinite(i_v238255)), 1e-300, i_v238255)
                        ),
                    )
                    * i_v238257
                    + i_v238178 * i_v12709
                    - i_v12712
                )
            ),
            1e-300,
            jnp.power(
                jnp.maximum(i_v238178, 0.0),
                jnp.divide(i_v12709, jnp.where((i_v238255 == 0.0) | jnp.logical_not(jnp.isfinite(i_v238255)), 1e-300, i_v238255)),
            )
            * i_v238257
            + i_v238178 * i_v12709
            - i_v12712,
        ),
    )
    i_v238323 = i_v238049 * i_v238275
    i_v238327 = jnp.where(i_v238323 < 0.0, -i_v238323, i_v238323) < 1e-06
    i_v261764 = (
        jnp.where(
            i_v12433,
            jnp.where(i_v226864, jnp.where(i_v238073, jnp.where(i_v238162, jnp.where(i_v238327, 1.0, 0.0), 0.0), 0.0), 0.0),
            0.0,
        )
        > 0.0
    )
    i_v249229 = jnp.where(
        i_v226864,
        jnp.where(
            i_v238073,
            jnp.where(
                i_v238162,
                jnp.where(
                    i_v238327,
                    i_v238060
                    * (
                        jnp.divide(1.0, jnp.where((i_v12715 == 0.0) | jnp.logical_not(jnp.isfinite(i_v12715)), 1e-300, i_v12715))
                        + 0.5 * i_v596 * i_v238275
                    ),
                    jnp.divide(
                        -i_v238060,
                        jnp.where(
                            (jnp.exp(jnp.clip(-i_v12715 * i_v596 * i_v238275, -709.0, 709.0)) - 1.0 == 0.0)
                            | jnp.logical_not(jnp.isfinite(jnp.exp(jnp.clip(-i_v12715 * i_v596 * i_v238275, -709.0, 709.0)) - 1.0)),
                            1e-300,
                            jnp.exp(jnp.clip(-i_v12715 * i_v596 * i_v238275, -709.0, 709.0)) - 1.0,
                        ),
                    ),
                ),
                0.0,
            ),
            0.0,
        ),
        0.0,
    )
    i_v261780, i_v261813 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v12433, _t, _f),
        (
            i_v249229,
            jnp.where(
                i_v226864,
                jnp.where(
                    i_v238073,
                    jnp.where(
                        i_v238162,
                        jnp.where(
                            i_v238327,
                            jnp.divide(
                                -0.5 * i_v238060 * i_v238275 * i_v596,
                                jnp.where((i_v12715 == 0.0) | jnp.logical_not(jnp.isfinite(i_v12715)), 1e-300, i_v12715),
                            ),
                            i_v238275,
                        ),
                        1.0,
                    ),
                    1.0,
                ),
                1.0,
            ),
        ),
        (0.0, 1.0),
    )
    i_v263984 = -i_v261780
    i_v255203 = jnp.where(
        i_v247031,
        jnp.log(
            jnp.maximum(
                jnp.divide(
                    i_v249106,
                    jnp.where(
                        (jnp.where(i_v249229 < 0.0, -i_v249229, i_v249229) + 1e-21 == 0.0)
                        | jnp.logical_not(jnp.isfinite(jnp.where(i_v249229 < 0.0, -i_v249229, i_v249229) + 1e-21)),
                        1e-300,
                        jnp.where(i_v249229 < 0.0, -i_v249229, i_v249229) + 1e-21,
                    ),
                ),
                1e-300,
            )
        ),
        0.0,
    )
    i_v257195 = jnp.where(i_v255203 < 230.25850929940458, i_v255203, 230.25850929940458)
    i_v263853, i_v263875 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v12433, _t, _f), (i_v257195, jnp.exp(jnp.clip(i_v257195, -709.0, 709.0))), (0.0, 0.0)
    )
    i_v274582 = jnp.logical_not(jnp.where(jnp.where(i_v8640, i_v8686, False), i_v8764, False))
    i_v276390 = i_v277 < NJH
    i_v276402 = 4.0 * NJH * 0.01
    i_v276407 = jnp.where(i_v276402 > 0.0, i_v276402, -i_v276402)
    i_v276423 = 4.0 * i_v277 * 0.01
    i_v276428 = jnp.where(i_v276423 > 0.0, i_v276423, -i_v276423)
    i_v276472 = i_v277 * NJH
    i_v279217 = i_v277 < NJH
    i_v279225 = NJH - (NJDV * (i_v7952 - i_v7258) + i_v277) - 0.01
    i_v279227 = 4.0 * NJH * 0.01
    i_v279238 = jnp.sqrt(jnp.maximum(i_v279225 * i_v279225 + jnp.where(i_v279227 > 0.0, i_v279227, -i_v279227), 1e-300))
    i_v279249 = NJH - 0.5 * (i_v279225 + i_v279238) - i_v277 - 0.01
    i_v279251 = 4.0 * i_v277 * 0.01
    i_v279262 = jnp.sqrt(jnp.maximum(i_v279249 * i_v279249 + jnp.where(i_v279251 > 0.0, i_v279251, -i_v279251), 1e-300))
    i_v279295, i_v279445 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v279217, _t, _f),
        (
            i_v277 + 0.5 * (i_v279249 + i_v279262),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v279225, i_v279238))) * (0.5 * (1.0 + jnp.divide(i_v279249, i_v279262))),
        ),
        (i_v277, 0.0),
    )
    i_v279308 = i_v277 * NJH
    i_v279463 = i_v596 * (
        jnp.divide(
            i_v279295 - i_v7952 * i_v279445,
            jnp.where(
                (i_v279295 * i_v279295 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279295 * i_v279295)), 1e-300, i_v279295 * i_v279295
            ),
        )
        + jnp.divide(
            i_v7258 * i_v279445, jnp.where((i_v279308 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279308)), 1e-300, i_v279308)
        )
    )
    i_v279311 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v279295 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279295)), 1e-300, i_v279295))
        + jnp.divide(
            i_v7258 * (i_v279295 - i_v277),
            jnp.where((i_v279308 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279308)), 1e-300, i_v279308),
        )
    )
    i_v279371 = -230.25850929940458 - i_v279311
    i_v279410 = i_v279311 - 230.25850929940458
    i_v279486 = jnp.where(
        jnp.where(i_v279311 < 0.0, -i_v279311, i_v279311) < 230.25850929940458,
        jnp.exp(jnp.clip(i_v279311, -709.0, 709.0)),
        jnp.where(
            i_v279311 < -230.25850929940458,
            jnp.divide(
                1e-100,
                jnp.where(
                    (1.0 + i_v279371 * (1.0 + 0.5 * (i_v279371 * (1.0 + i_v279371 * 0.3333333333333333))) == 0.0)
                    | jnp.logical_not(
                        jnp.isfinite(1.0 + i_v279371 * (1.0 + 0.5 * (i_v279371 * (1.0 + i_v279371 * 0.3333333333333333))))
                    ),
                    1e-300,
                    1.0 + i_v279371 * (1.0 + 0.5 * (i_v279371 * (1.0 + i_v279371 * 0.3333333333333333))),
                ),
            ),
            1e100 * (1.0 + i_v279410 * (1.0 + 0.5 * (i_v279410 * (1.0 + i_v279410 * 0.3333333333333333)))),
        ),
    )
    i_v284273 = jnp.where(i_v118 == 0.0, i_v145 == 0.0, False)
    i_v285203 = i_v76 == 0.5
    i_v281114 = 4.0 * i_v594 * i_v594
    i_v285218 = 1.0 - 2.0 * i_v76
    i_v286646 = i_v145 == 0.0
    i_v287579 = 0.666666666666667 * i_v3979
    i_v288051 = -i_v76 * i_v2837
    i_v288054 = i_v288051 == -1.0
    i_v289688 = i_v176 == 0.0
    i_v282050 = 4.0 * i_v590 * i_v590
    i_v289708 = i_v76 == 0.5
    i_v291190 = -i_v5102
    i_v291782 = i_v26813 > 1000.0
    i_v292146 = -i_v5226 * i_v26813
    i_v292607 = i_v217 == 4.0
    i_v293580 = i_v5226 * i_v26813
    i_v277477 = i_v281 < NJH
    i_v277476 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v277489 = 4.0 * NJH * 0.01
    i_v277494 = jnp.where(i_v277489 > 0.0, i_v277489, -i_v277489)
    i_v277510 = 4.0 * i_v281 * 0.01
    i_v277515 = jnp.where(i_v277510 > 0.0, i_v277510, -i_v277510)
    i_v277559 = i_v281 * NJH
    i_v279551 = i_v281 < NJH
    i_v279550 = jnp.divide(i_v281, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7132,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7132 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7132)), 1e-300, i_v7132)),
                ),
            ),
            1e-300,
        )
    )
    i_v279559 = NJH - (NJDV * (i_v7952 - i_v279550) + i_v281) - 0.01
    i_v279561 = 4.0 * NJH * 0.01
    i_v279572 = jnp.sqrt(jnp.maximum(i_v279559 * i_v279559 + jnp.where(i_v279561 > 0.0, i_v279561, -i_v279561), 1e-300))
    i_v279583 = NJH - 0.5 * (i_v279559 + i_v279572) - i_v281 - 0.01
    i_v279585 = 4.0 * i_v281 * 0.01
    i_v279596 = jnp.sqrt(jnp.maximum(i_v279583 * i_v279583 + jnp.where(i_v279585 > 0.0, i_v279585, -i_v279585), 1e-300))
    i_v279629, i_v279779 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v279551, _t, _f),
        (
            i_v281 + 0.5 * (i_v279583 + i_v279596),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v279559, i_v279572))) * (0.5 * (1.0 + jnp.divide(i_v279583, i_v279596))),
        ),
        (i_v281, 0.0),
    )
    i_v279642 = i_v281 * NJH
    i_v279797 = i_v596 * (
        jnp.divide(
            i_v279629 - i_v7952 * i_v279779,
            jnp.where(
                (i_v279629 * i_v279629 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279629 * i_v279629)), 1e-300, i_v279629 * i_v279629
            ),
        )
        + jnp.divide(
            i_v279550 * i_v279779, jnp.where((i_v279642 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279642)), 1e-300, i_v279642)
        )
    )
    i_v279645 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v279629 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279629)), 1e-300, i_v279629))
        + jnp.divide(
            i_v279550 * (i_v279629 - i_v281),
            jnp.where((i_v279642 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279642)), 1e-300, i_v279642),
        )
    )
    i_v279705 = -230.25850929940458 - i_v279645
    i_v279744 = i_v279645 - 230.25850929940458
    i_v279820 = jnp.where(
        jnp.where(i_v279645 < 0.0, -i_v279645, i_v279645) < 230.25850929940458,
        jnp.exp(jnp.clip(i_v279645, -709.0, 709.0)),
        jnp.where(
            i_v279645 < -230.25850929940458,
            jnp.divide(
                1e-100,
                jnp.where(
                    (1.0 + i_v279705 * (1.0 + 0.5 * (i_v279705 * (1.0 + i_v279705 * 0.3333333333333333))) == 0.0)
                    | jnp.logical_not(
                        jnp.isfinite(1.0 + i_v279705 * (1.0 + 0.5 * (i_v279705 * (1.0 + i_v279705 * 0.3333333333333333))))
                    ),
                    1e-300,
                    1.0 + i_v279705 * (1.0 + 0.5 * (i_v279705 * (1.0 + i_v279705 * 0.3333333333333333))),
                ),
            ),
            1e100 * (1.0 + i_v279744 * (1.0 + 0.5 * (i_v279744 * (1.0 + i_v279744 * 0.3333333333333333)))),
        ),
    )
    i_v296887 = jnp.where(i_v124 == 0.0, i_v151 == 0.0, False)
    i_v297949 = i_v82 == 0.5
    i_v297963 = 1.0 - 2.0 * i_v82
    i_v299452 = i_v151 == 0.0
    i_v300383 = 0.666666666666667 * i_v3984
    i_v300850 = -i_v82 * i_v2840
    i_v300853 = i_v300850 == -1.0
    i_v301792 = i_v182 == 0.0
    i_v301811 = i_v82 == 0.5
    i_v303328 = -i_v5110
    i_v303920 = i_v39291 > 1000.0
    i_v304066 = -i_v5226 * i_v39291
    i_v304527 = i_v221 == 4.0
    i_v305500 = i_v5226 * i_v39291
    i_v278528 = i_v285 < NJH
    i_v278527 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v278540 = 4.0 * NJH * 0.01
    i_v278545 = jnp.where(i_v278540 > 0.0, i_v278540, -i_v278540)
    i_v278561 = 4.0 * i_v285 * 0.01
    i_v278566 = jnp.where(i_v278561 > 0.0, i_v278561, -i_v278561)
    i_v278610 = i_v285 * NJH
    i_v279921 = i_v285 < NJH
    i_v279920 = jnp.divide(i_v285, i_v596) * jnp.log(
        jnp.maximum(
            jnp.divide(
                i_v7135,
                jnp.where(
                    (
                        jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135))
                        == 0.0
                    )
                    | jnp.logical_not(
                        jnp.isfinite(
                            jnp.divide(
                                i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)
                            )
                        )
                    ),
                    1e-300,
                    jnp.divide(i_v7209, jnp.where((i_v7135 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7135)), 1e-300, i_v7135)),
                ),
            ),
            1e-300,
        )
    )
    i_v279929 = NJH - (NJDV * (i_v7952 - i_v279920) + i_v285) - 0.01
    i_v279931 = 4.0 * NJH * 0.01
    i_v279942 = jnp.sqrt(jnp.maximum(i_v279929 * i_v279929 + jnp.where(i_v279931 > 0.0, i_v279931, -i_v279931), 1e-300))
    i_v279953 = NJH - 0.5 * (i_v279929 + i_v279942) - i_v285 - 0.01
    i_v279955 = 4.0 * i_v285 * 0.01
    i_v279966 = jnp.sqrt(jnp.maximum(i_v279953 * i_v279953 + jnp.where(i_v279955 > 0.0, i_v279955, -i_v279955), 1e-300))
    i_v279999, i_v280149 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v279921, _t, _f),
        (
            i_v285 + 0.5 * (i_v279953 + i_v279966),
            NJDV * (0.5 * (1.0 + jnp.divide(i_v279929, i_v279942))) * (0.5 * (1.0 + jnp.divide(i_v279953, i_v279966))),
        ),
        (i_v285, 0.0),
    )
    i_v280012 = i_v285 * NJH
    i_v280167 = i_v596 * (
        jnp.divide(
            i_v279999 - i_v7952 * i_v280149,
            jnp.where(
                (i_v279999 * i_v279999 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279999 * i_v279999)), 1e-300, i_v279999 * i_v279999
            ),
        )
        + jnp.divide(
            i_v279920 * i_v280149, jnp.where((i_v280012 == 0.0) | jnp.logical_not(jnp.isfinite(i_v280012)), 1e-300, i_v280012)
        )
    )
    i_v280015 = i_v596 * (
        jnp.divide(i_v7952, jnp.where((i_v279999 == 0.0) | jnp.logical_not(jnp.isfinite(i_v279999)), 1e-300, i_v279999))
        + jnp.divide(
            i_v279920 * (i_v279999 - i_v285),
            jnp.where((i_v280012 == 0.0) | jnp.logical_not(jnp.isfinite(i_v280012)), 1e-300, i_v280012),
        )
    )
    i_v280075 = -230.25850929940458 - i_v280015
    i_v280114 = i_v280015 - 230.25850929940458
    i_v280190 = jnp.where(
        jnp.where(i_v280015 < 0.0, -i_v280015, i_v280015) < 230.25850929940458,
        jnp.exp(jnp.clip(i_v280015, -709.0, 709.0)),
        jnp.where(
            i_v280015 < -230.25850929940458,
            jnp.divide(
                1e-100,
                jnp.where(
                    (1.0 + i_v280075 * (1.0 + 0.5 * (i_v280075 * (1.0 + i_v280075 * 0.3333333333333333))) == 0.0)
                    | jnp.logical_not(
                        jnp.isfinite(1.0 + i_v280075 * (1.0 + 0.5 * (i_v280075 * (1.0 + i_v280075 * 0.3333333333333333))))
                    ),
                    1e-300,
                    1.0 + i_v280075 * (1.0 + 0.5 * (i_v280075 * (1.0 + i_v280075 * 0.3333333333333333))),
                ),
            ),
            1e100 * (1.0 + i_v280114 * (1.0 + 0.5 * (i_v280114 * (1.0 + i_v280114 * 0.3333333333333333)))),
        ),
    )
    i_v309272 = jnp.where(i_v130 == 0.0, i_v157 == 0.0, False)
    i_v310334 = i_v88 == 0.5
    i_v310348 = 1.0 - 2.0 * i_v88
    i_v311837 = i_v157 == 0.0
    i_v312768 = 0.666666666666667 * i_v3986
    i_v313235 = -i_v88 * i_v2843
    i_v313238 = i_v313235 == -1.0
    i_v314177 = i_v188 == 0.0
    i_v314196 = i_v88 == 0.5
    i_v315713 = -i_v5121
    i_v316305 = i_v53512 > 1000.0
    i_v316451 = -i_v5226 * i_v53512
    i_v316912 = i_v225 == 4.0
    i_v317885 = i_v5226 * i_v53512
    i_v325285 = CORECOVERY > 0.0
    i_v2344 = jnp.divide(1.0, jnp.where((i_v2324 == 0.0) | jnp.logical_not(jnp.isfinite(i_v2324)), 1e-300, i_v2324))
    i_v3033 = i_v47 * jnp.power(jnp.maximum(i_v61 * i_v2344, 0.0), i_v76)
    i_v242641 = i_v7324 * i_v3033
    i_v2346 = jnp.divide(1.0, jnp.where((i_v2333 == 0.0) | jnp.logical_not(jnp.isfinite(i_v2333)), 1e-300, i_v2333))
    i_v3220 = i_v52 * jnp.power(jnp.maximum(i_v65 * i_v2346, 0.0), i_v82)
    i_v244805 = i_v7338 * i_v3220
    i_v2348 = jnp.divide(1.0, jnp.where((i_v2342 == 0.0) | jnp.logical_not(jnp.isfinite(i_v2342)), 1e-300, i_v2342))
    i_v3404 = i_v56 * jnp.power(jnp.maximum(i_v69 * i_v2348, 0.0), i_v88)
    i_v246967 = i_v7352 * i_v3404
    i_v246969 = jnp.where(FJUNQ > 0.0, FJUNQ, 0.0) * (i_v242641 + i_v244805 + i_v246967)
    i_v267030 = jnp.where(i_v12433, jnp.where(i_v242641 <= i_v246969, 0.0, 1.0), 1.0) > 0.5
    i_v3407 = i_v3033 * i_v2324 * i_v2837
    i_v267366 = i_v2516 == 0.5
    i_v8893 = jnp.where(i_v8640, 0.9 * jnp.where(i_v82 < i_v88, i_v82, i_v88), i_v76)
    i_v8911 = jnp.where(i_v8686, 0.9 * jnp.where(i_v76 < i_v88, i_v76, i_v88), i_v82)
    i_v8926 = jnp.where(i_v8893 > i_v8911, i_v8893, i_v8911)
    i_v8933 = jnp.where(i_v8764, 0.9 * jnp.where(i_v76 < i_v82, i_v76, i_v82), i_v88)
    i_v8965 = i_v8881 * (
        1.0
        - jnp.power(
            jnp.maximum(2.0, 0.0),
            jnp.divide(
                -1.0,
                jnp.where(
                    (jnp.where(i_v8926 > i_v8933, i_v8926, i_v8933) == 0.0)
                    | jnp.logical_not(jnp.isfinite(jnp.where(i_v8926 > i_v8933, i_v8926, i_v8933))),
                    1e-300,
                    jnp.where(i_v8926 > i_v8933, i_v8926, i_v8933),
                ),
            ),
        )
    )
    i_v8883 = i_v8881 * 0.1
    i_v264987 = i_v8883 * jnp.divide(i_v8883, jnp.where((i_v8965 == 0.0) | jnp.logical_not(jnp.isfinite(i_v8965)), 1e-300, i_v8965))
    i_v264500 = 4.0 * i_v8883 * i_v8883
    i_v3416 = 2.0 * i_v3033
    i_v282351 = i_v2516 == 0.5
    i_v274610 = i_v8883 * jnp.divide(i_v8883, jnp.where((i_v8965 == 0.0) | jnp.logical_not(jnp.isfinite(i_v8965)), 1e-300, i_v8965))
    i_v274592 = 4.0 * i_v8883 * i_v8883
    i_v270479 = jnp.where(i_v12433, jnp.where(i_v244805 <= i_v246969, 0.0, 1.0), 1.0) > 0.5
    i_v3410 = i_v3220 * i_v2333 * i_v2840
    i_v270743 = i_v2678 == 0.5
    i_v3419 = 2.0 * i_v3220
    i_v294679 = i_v2678 == 0.5
    i_v273650 = jnp.where(i_v12433, jnp.where(i_v246967 <= i_v246969, 0.0, 1.0), 1.0) > 0.5
    i_v3413 = i_v3404 * i_v2342 * i_v2843
    i_v273842 = i_v2834 == 0.5
    i_v3422 = 2.0 * i_v3404
    i_v306752 = i_v2834 == 0.5
    i_v349807 = -(i_v7129 * i_v7324 * 1.6021918e-19) * WI
    i_v7214 = jnp.power(jnp.maximum(i_v584, 0.0), -1.5)
    i_v7235 = jnp.divide(0.14500000000000002 * i_v7214, i_v596)
    i_v7238 = jnp.divide(0.05 * i_v7214, i_v596)
    i_v7253 = jnp.sqrt(
        jnp.maximum(
            TAU
            * jnp.power(jnp.maximum(i_v584, 0.0), TAUT)
            * jnp.divide(
                2.0 * i_v7235 * i_v7238,
                jnp.where((i_v7235 + i_v7238 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7235 + i_v7238)), 1e-300, i_v7235 + i_v7238),
            ),
            1e-300,
        )
    )
    i_v341839 = NQS > 0.0
    i_v341834 = 1.6021918e-19 * i_v7324
    i_v334145 = INJ2 == 0.0
    i_v330476 = jnp.where(i_v12433, i_v191302, 0.0)
    i_v334179 = -INJ2
    i_v339217 = jnp.exp(
        jnp.clip(
            INJT
            * jnp.log(
                jnp.maximum(
                    jnp.divide(i_v570, jnp.where((i_v579 == 0.0) | jnp.logical_not(jnp.isfinite(i_v579)), 1e-300, i_v579)), 1e-300
                )
            ),
            -709.0,
            709.0,
        )
    )
    i_v352421 = jnp.exp(jnp.clip(jnp.divide(-WI, i_v7253), -709.0, 709.0))
    i_v349775 = DEPNQS > 0.0
    i_v349751 = 1.6021918e-19 * i_v7129
    i_v349759 = 4.0 * WI * 1e-07
    i_v349764 = jnp.where(i_v349759 > 0.0, i_v349759, -i_v349759)
    i_v7266 = i_v7255 * (i_v7257 + jnp.divide(WI, i_v7253))
    i_v333674 = i_v7266 - i_v7258
    i_v327819 = i_v277 < NJH
    i_v330407 = 4.0 * NJH * 0.01
    i_v330412 = jnp.where(i_v330407 > 0.0, i_v330407, -i_v330407)
    i_v330429 = 4.0 * i_v277 * 0.01
    i_v330434 = jnp.where(i_v330429 > 0.0, i_v330429, -i_v330429)
    i_v333983 = i_v277 * NJH
    i_v341903 = -INJ2
    i_v341941 = jnp.exp(
        jnp.clip(
            INJT
            * jnp.log(
                jnp.maximum(
                    jnp.divide(i_v570, jnp.where((i_v579 == 0.0) | jnp.logical_not(jnp.isfinite(i_v579)), 1e-300, i_v579)), 1e-300
                )
            ),
            -709.0,
            709.0,
        )
    )
    i_v397993 = i_v374514 * jnp.where(i_v325285, 0.0, jnp.where(TT > 0.0, TT, 0.0))
    i_v6809 = jnp.power(jnp.maximum(i_v584, 0.0), jnp.where(STRS > 0.0, STRS, 0.0))
    i_v6810 = jnp.where(RSBOT > 0.0, RSBOT, 0.0) * i_v6809
    i_v7023 = jnp.where(RSSTI > 0.0, RSSTI, 0.0) * i_v6809
    i_v6915 = jnp.where(RSGAT > 0.0, RSGAT, 0.0) * i_v6809
    i_v7125 = jnp.where(RSCOM > 0.0, RSCOM, 0.0) * i_v6809
    i_v11139 = jnp.where(
        i_v7324 * i_v6810 > 0.0,
        jnp.divide(i_v7324, jnp.where((i_v6810 == 0.0) | jnp.logical_not(jnp.isfinite(i_v6810)), 1e-300, i_v6810)),
        10000000000000.0,
    )
    i_v11266 = jnp.where(
        i_v7338 * i_v7023 > 0.0,
        jnp.divide(i_v7338, jnp.where((i_v7023 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7023)), 1e-300, i_v7023)) + i_v11139,
        i_v11139,
    )
    i_v11269 = jnp.where(
        i_v7352 * i_v6915 > 0.0,
        jnp.divide(i_v7352, jnp.where((i_v6915 == 0.0) | jnp.logical_not(jnp.isfinite(i_v6915)), 1e-300, i_v6915)) + i_v11266,
        i_v11266,
    )
    i_v11386 = jnp.divide(1.0, jnp.where((i_v11269 == 0.0) | jnp.logical_not(jnp.isfinite(i_v11269)), 1e-300, i_v11269)) + i_v7125
    i_v11389 = i_v11386 > 1.3999999999999998e-13
    i_v374520 = (
        jnp.where(
            jnp.where(
                jnp.where(
                    jnp.where(
                        jnp.where(
                            i_v11404,
                            jnp.divide(
                                i_v6810, jnp.where((i_v7324 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7324)), 1e-300, i_v7324)
                            )
                            < 1e-13,
                            False,
                        ),
                        True,
                        jnp.where(
                            i_v7338 > 0.0,
                            jnp.divide(
                                i_v7023, jnp.where((i_v7338 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7338)), 1e-300, i_v7338)
                            )
                            < 1e-13,
                            False,
                        ),
                    ),
                    True,
                    jnp.where(
                        i_v7352 > 0.0,
                        jnp.divide(i_v6915, jnp.where((i_v7352 == 0.0) | jnp.logical_not(jnp.isfinite(i_v7352)), 1e-300, i_v7352))
                        < 1e-13,
                        False,
                    ),
                ),
                i_v7125 < 1e-13,
                False,
            ),
            0,
            jnp.where(i_v11389, 1, 0),
        )
        == 1
    )
    i_v366767 = jnp.where(
        i_v11389,
        jnp.divide(1.0, jnp.where((i_v11386 == 0.0) | jnp.logical_not(jnp.isfinite(i_v11386)), 1e-300, i_v11386)),
        i_v11269,
    )
    i_v374608, i_v390134 = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v325285, _t, _f), (NQS > 0.0, DEPNQS > 0.0), (False, False)
    )
    return jnp.array(
        [
            i_v7358,
            i_v374514,
            i_v277,
            i_v285,
            i_v7324,
            i_v7352,
            i_v281,
            i_v7338,
            i_v12433,
            i_v594,
            i_v596,
            i_v1474,
            i_v1624,
            i_v1771,
            i_v259383,
            i_v257200,
            i_v224964,
            i_v8640,
            i_v8686,
            i_v8764,
            i_v259262,
            i_v259277,
            i_v7952,
            i_v7210,
            i_v7258,
            i_v118,
            i_v145,
            i_v1108,
            i_v15213,
            i_v3628,
            i_v76,
            i_v61,
            i_v2324,
            i_v65,
            i_v1117,
            i_v2333,
            i_v69,
            i_v1126,
            i_v2342,
            i_v8881,
            i_v3907,
            i_v3979,
            i_v4127,
            i_v2516,
            i_v2837,
            i_v176,
            i_v9052,
            i_v3898,
            i_v217,
            i_v6514,
            i_v5340,
            i_v6517,
            i_v124,
            i_v151,
            i_v3763,
            i_v82,
            i_v3910,
            i_v3984,
            i_v4262,
            i_v2678,
            i_v2840,
            i_v182,
            i_v3901,
            i_v221,
            i_v6546,
            i_v5451,
            i_v6548,
            i_v130,
            i_v157,
            i_v3895,
            i_v88,
            i_v3913,
            i_v3986,
            i_v4394,
            i_v2834,
            i_v2843,
            i_v188,
            i_v3904,
            i_v225,
            i_v6575,
            i_v5559,
            i_v6576,
            i_v259720,
            i_v259506,
            i_v259594,
            i_v259612,
            i_v261764,
            i_v261780,
            i_v261813,
            i_v263984,
            i_v263853,
            i_v263875,
            i_v274582,
            i_v276390,
            i_v276407,
            i_v276428,
            i_v276472,
            i_v279463,
            i_v279486,
            i_v284273,
            i_v285203,
            i_v281114,
            i_v285218,
            i_v286646,
            i_v287579,
            i_v288051,
            i_v288054,
            i_v289688,
            i_v282050,
            i_v289708,
            i_v291190,
            i_v291782,
            i_v292146,
            i_v292607,
            i_v293580,
            i_v277477,
            i_v277476,
            i_v277494,
            i_v277515,
            i_v277559,
            i_v279797,
            i_v279820,
            i_v296887,
            i_v297949,
            i_v297963,
            i_v299452,
            i_v300383,
            i_v300850,
            i_v300853,
            i_v301792,
            i_v301811,
            i_v303328,
            i_v303920,
            i_v304066,
            i_v304527,
            i_v305500,
            i_v278528,
            i_v278527,
            i_v278545,
            i_v278566,
            i_v278610,
            i_v280167,
            i_v280190,
            i_v309272,
            i_v310334,
            i_v310348,
            i_v311837,
            i_v312768,
            i_v313235,
            i_v313238,
            i_v314177,
            i_v314196,
            i_v315713,
            i_v316305,
            i_v316451,
            i_v316912,
            i_v317885,
            i_v325285,
            i_v2344,
            i_v2346,
            i_v2348,
            i_v267030,
            i_v3407,
            i_v267366,
            i_v8965,
            i_v264987,
            i_v264500,
            i_v3416,
            i_v282351,
            i_v274610,
            i_v274592,
            i_v270479,
            i_v3410,
            i_v270743,
            i_v3419,
            i_v294679,
            i_v273650,
            i_v3413,
            i_v273842,
            i_v3422,
            i_v306752,
            i_v349807,
            i_v7253,
            i_v341839,
            i_v341834,
            i_v334145,
            i_v330476,
            i_v334179,
            i_v339217,
            i_v352421,
            i_v349775,
            i_v349751,
            i_v349764,
            i_v7266,
            i_v333674,
            i_v327819,
            i_v330412,
            i_v330434,
            i_v333983,
            i_v341903,
            i_v341941,
            i_v397993,
            i_v374520,
            i_v366767,
            i_v374608,
            i_v390134,
        ]
    )


@component(ports=("A", "K"), states=("v_AIK", "v_charge_A", "v_charge_K", "v_depl_A"))
def DIODE_CMC(
    signals: Signals,
    s: States,
    init,
    TYPE: float = 1.0,
    DTA: float = 0.0,
    IMAX: float = 1000.0,
    TRJ: float = 21.0,
    CJORBOT: float = 0.001,
    CJORSTI: float = 1e-09,
    CJORGAT: float = 1e-09,
    VBIRBOT: float = 1.0,
    VBIRSTI: float = 1.0,
    VBIRGAT: float = 1.0,
    PBOT: float = 0.5,
    PSTI: float = 0.5,
    PGAT: float = 0.5,
    PHIGBOT: float = 1.16,
    PHIGSTI: float = 1.16,
    PHIGGAT: float = 1.16,
    IDSATRBOT: float = 1e-12,
    IDSATRSTI: float = 1e-18,
    IDSATRGAT: float = 1e-18,
    CSRHBOT: float = 100.0,
    CSRHSTI: float = 0.0001,
    CSRHGAT: float = 0.0001,
    XJUNSTI: float = 1e-07,
    XJUNGAT: float = 1e-07,
    CTATBOT: float = 100.0,
    CTATSTI: float = 0.0001,
    CTATGAT: float = 0.0001,
    MEFFTATBOT: float = 0.25,
    MEFFTATSTI: float = 0.25,
    MEFFTATGAT: float = 0.25,
    CBBTBOT: float = 1e-12,
    CBBTSTI: float = 1e-18,
    CBBTGAT: float = 1e-18,
    FBBTRBOT: float = 1000000000.0,
    FBBTRSTI: float = 1000000000.0,
    FBBTRGAT: float = 1000000000.0,
    STFBBTBOT: float = -0.001,
    STFBBTSTI: float = -0.001,
    STFBBTGAT: float = -0.001,
    VBRBOT: float = 10.0,
    VBRSTI: float = 10.0,
    VBRGAT: float = 10.0,
    PBRBOT: float = 4.0,
    PBRSTI: float = 4.0,
    PBRGAT: float = 4.0,
    FREV: float = 1000.0,
    RSBOT: float = 0.0,
    RSSTI: float = 0.0,
    RSGAT: float = 0.0,
    RSCOM: float = 0.0,
    STRS: float = 0.0,
    KF: float = 0.0,
    AF: float = 1.0,
    TT: float = 0.0,
    STVBRBOT1: float = 0.0,
    STVBRBOT2: float = 0.0,
    STVBRSTI1: float = 0.0,
    STVBRSTI2: float = 0.0,
    STVBRGAT1: float = 0.0,
    STVBRGAT2: float = 0.0,
    NFABOT: float = 1.0,
    NFASTI: float = 1.0,
    NFAGAT: float = 1.0,
    ABMIN: float = 0.0,
    ABMAX: float = 1.0,
    LSMIN: float = 0.0,
    LSMAX: float = 1.0,
    LGMIN: float = 0.0,
    LGMAX: float = 1.0,
    TEMPMIN: float = -55.0,
    TEMPMAX: float = 155.0,
    VFMAX: float = 0.0,
    VRMAX: float = 0.0,
    XTI: float = 3.0,
    SCALE: float = 1.0,
    SHRINK: float = 0.0,
    SWJUNEXP: float = 0.0,
    VJUNREF: float = 2.5,
    FJUNQ: float = 0.03,
    CORECOVERY: float = 0.0,
    NJH: float = 1.0,
    NJDV: float = 0.1,
    NDIBOT: float = 1e16,
    NDIGAT: float = 1e16,
    NDISTI: float = 1e16,
    INJ1: float = 1.0,
    INJ2: float = 10.0,
    NQS: float = 5e-09,
    TAU: float = 2e-07,
    WI: float = 5e-06,
    DEPNQS: float = 0.0,
    TNOM: float = 21.0,
    TAUT: float = 0.0,
    INJT: float = 0.0,
    AB: float = 1e-12,
    LS: float = 1e-06,
    LG: float = 0.0,
    MULT: float = 1.0,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """Dynamic junction diode with recovery and high-injection effects.

    Native current and integrated terminal charge support DC, AC, transient and HB without a VA runtime.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 2.0, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `diode_cmc.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/diode_cmc/vacode/diode_cmc.va).
    Attribution: NXP Semiconductors, Hiroshima University and Silicon Integration Initiative (CMC redistribution
    terms).
    The retained source and includes are under `tests/data/va_models/diode_cmc/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/diode_cmc/vacode/diode_cmc.va),
    with comparison cases in `tests/test_diode_cmc.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        diode_cmc.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.diode_cmc import DIODE_CMC

        jax.config.update("jax_enable_x64", True)
        model = DIODE_CMC()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(DIODE_CMC.__name__, "ports:", model.ports)
        ```

    """
    i_v7358 = init[0]
    i_v374514 = init[1]
    i_v277 = init[2]
    i_v285 = init[3]
    i_v7324 = init[4]
    i_v7352 = init[5]
    i_v281 = init[6]
    i_v7338 = init[7]
    i_v12433 = init[8]
    i_v594 = init[9]
    i_v596 = init[10]
    i_v1474 = init[11]
    i_v1624 = init[12]
    i_v1771 = init[13]
    i_v259383 = init[14]
    i_v257200 = init[15]
    i_v224964 = init[16]
    i_v8640 = init[17]
    i_v8686 = init[18]
    i_v8764 = init[19]
    i_v259262 = init[20]
    i_v259277 = init[21]
    i_v7952 = init[22]
    i_v7210 = init[23]
    i_v7258 = init[24]
    i_v118 = init[25]
    i_v145 = init[26]
    i_v1108 = init[27]
    i_v15213 = init[28]
    i_v3628 = init[29]
    i_v76 = init[30]
    i_v61 = init[31]
    i_v2324 = init[32]
    i_v65 = init[33]
    i_v1117 = init[34]
    i_v2333 = init[35]
    i_v69 = init[36]
    i_v1126 = init[37]
    i_v2342 = init[38]
    i_v8881 = init[39]
    i_v3907 = init[40]
    i_v3979 = init[41]
    i_v4127 = init[42]
    i_v2516 = init[43]
    i_v2837 = init[44]
    i_v176 = init[45]
    i_v9052 = init[46]
    i_v3898 = init[47]
    i_v217 = init[48]
    i_v6514 = init[49]
    i_v5340 = init[50]
    i_v6517 = init[51]
    i_v124 = init[52]
    i_v151 = init[53]
    i_v3763 = init[54]
    i_v82 = init[55]
    i_v3910 = init[56]
    i_v3984 = init[57]
    i_v4262 = init[58]
    i_v2678 = init[59]
    i_v2840 = init[60]
    i_v182 = init[61]
    i_v3901 = init[62]
    i_v221 = init[63]
    i_v6546 = init[64]
    i_v5451 = init[65]
    i_v6548 = init[66]
    i_v130 = init[67]
    i_v157 = init[68]
    i_v3895 = init[69]
    i_v88 = init[70]
    i_v3913 = init[71]
    i_v3986 = init[72]
    i_v4394 = init[73]
    i_v2834 = init[74]
    i_v2843 = init[75]
    i_v188 = init[76]
    i_v3904 = init[77]
    i_v225 = init[78]
    i_v6575 = init[79]
    i_v5559 = init[80]
    i_v6576 = init[81]
    i_v259720 = init[82]
    i_v259506 = init[83]
    i_v259594 = init[84]
    i_v259612 = init[85]
    i_v261764 = init[86]
    i_v261780 = init[87]
    i_v261813 = init[88]
    i_v263984 = init[89]
    i_v263853 = init[90]
    i_v263875 = init[91]
    i_v274582 = init[92]
    i_v276390 = init[93]
    i_v276407 = init[94]
    i_v276428 = init[95]
    i_v276472 = init[96]
    i_v279463 = init[97]
    i_v279486 = init[98]
    i_v284273 = init[99]
    i_v285203 = init[100]
    i_v281114 = init[101]
    i_v285218 = init[102]
    i_v286646 = init[103]
    i_v287579 = init[104]
    i_v288051 = init[105]
    i_v288054 = init[106]
    i_v289688 = init[107]
    i_v282050 = init[108]
    i_v289708 = init[109]
    i_v291190 = init[110]
    i_v291782 = init[111]
    i_v292146 = init[112]
    i_v292607 = init[113]
    i_v293580 = init[114]
    i_v277477 = init[115]
    i_v277476 = init[116]
    i_v277494 = init[117]
    i_v277515 = init[118]
    i_v277559 = init[119]
    i_v279797 = init[120]
    i_v279820 = init[121]
    i_v296887 = init[122]
    i_v297949 = init[123]
    i_v297963 = init[124]
    i_v299452 = init[125]
    i_v300383 = init[126]
    i_v300850 = init[127]
    i_v300853 = init[128]
    i_v301792 = init[129]
    i_v301811 = init[130]
    i_v303328 = init[131]
    i_v303920 = init[132]
    i_v304066 = init[133]
    i_v304527 = init[134]
    i_v305500 = init[135]
    i_v278528 = init[136]
    i_v278527 = init[137]
    i_v278545 = init[138]
    i_v278566 = init[139]
    i_v278610 = init[140]
    i_v280167 = init[141]
    i_v280190 = init[142]
    i_v309272 = init[143]
    i_v310334 = init[144]
    i_v310348 = init[145]
    i_v311837 = init[146]
    i_v312768 = init[147]
    i_v313235 = init[148]
    i_v313238 = init[149]
    i_v314177 = init[150]
    i_v314196 = init[151]
    i_v315713 = init[152]
    i_v316305 = init[153]
    i_v316451 = init[154]
    i_v316912 = init[155]
    i_v317885 = init[156]
    i_v325285 = init[157]
    i_v2344 = init[158]
    i_v2346 = init[159]
    i_v2348 = init[160]
    i_v267030 = init[161]
    i_v3407 = init[162]
    i_v267366 = init[163]
    i_v8965 = init[164]
    i_v264987 = init[165]
    i_v264500 = init[166]
    i_v3416 = init[167]
    i_v282351 = init[168]
    i_v274610 = init[169]
    i_v274592 = init[170]
    i_v270479 = init[171]
    i_v3410 = init[172]
    i_v270743 = init[173]
    i_v3419 = init[174]
    i_v294679 = init[175]
    i_v273650 = init[176]
    i_v3413 = init[177]
    i_v273842 = init[178]
    i_v3422 = init[179]
    i_v306752 = init[180]
    i_v349807 = init[181]
    i_v7253 = init[182]
    i_v341839 = init[183]
    i_v341834 = init[184]
    i_v334145 = init[185]
    i_v330476 = init[186]
    i_v334179 = init[187]
    i_v339217 = init[188]
    i_v352421 = init[189]
    i_v349775 = init[190]
    i_v349751 = init[191]
    i_v349764 = init[192]
    i_v7266 = init[193]
    i_v333674 = init[194]
    i_v327819 = init[195]
    i_v330412 = init[196]
    i_v330434 = init[197]
    i_v333983 = init[198]
    i_v341903 = init[199]
    i_v341941 = init[200]
    i_v397993 = init[201]
    i_v374520 = init[202]
    i_v366767 = init[203]
    i_v374608 = init[204]
    i_v390134 = init[205]
    v257202 = -(signals.A - s.v_AIK)
    v259131 = jnp.where(i_v257200, v257202, signals.A - s.v_AIK)
    v259175 = v259131 * i_v596
    v259247 = v259175 * i_v224964
    v259250 = v259247 < -230.25850929940458
    v259253 = -230.25850929940458 - v259247
    v259255 = v259253 + 1.0
    v259256 = jnp.divide(1e-100, jnp.where((v259255 == 0.0) | jnp.logical_not(jnp.isfinite(v259255)), 1e-300, v259255))
    v259271 = v259247 > i_v259262
    v259369 = v259247 - i_v259262
    v259371 = v259369 + 1.0
    v259372 = i_v259277 * v259371
    v259374 = jnp.exp(jnp.clip(v259247, -709.0, 709.0))
    v259375 = jnp.where(v259271, v259372, v259374)
    v259376 = jnp.where(v259250, v259256, v259375)
    v259485 = v259376 - 1.0
    v259486 = i_v259383 * v259485
    v259578 = v259175 * i_v259506
    v259580 = v259578 < -230.25850929940458
    v259583 = -230.25850929940458 - v259578
    v259585 = v259583 + 1.0
    v259586 = jnp.divide(1e-100, jnp.where((v259585 == 0.0) | jnp.logical_not(jnp.isfinite(v259585)), 1e-300, v259585))
    v259604 = v259578 > i_v259594
    v259705 = v259578 - i_v259594
    v259707 = v259705 + 1.0
    v259708 = i_v259612 * v259707
    v259710 = jnp.exp(jnp.clip(v259578, -709.0, 709.0))
    v259711 = jnp.where(v259604, v259708, v259710)
    v259712 = jnp.where(v259580, v259586, v259711)
    v259745 = v259712 - 1.0
    v259746 = i_v259720 * v259745
    v264011 = v259486 + v259746
    v263820 = v259131 * i_v261813
    v263821 = i_v261780 + v263820
    v263822 = v259131 * v263821
    v263824 = -v259131
    v263832 = v263824 * i_v596
    v263834 = v263832 * i_v261813
    v263836 = v263834 < -230.25850929940458
    v263839 = -230.25850929940458 - v263834
    v263841 = v263839 + 1.0
    v263842 = jnp.divide(1e-100, jnp.where((v263841 == 0.0) | jnp.logical_not(jnp.isfinite(v263841)), 1e-300, v263841))
    v263864 = v263834 > i_v263853
    v263969 = v263834 - i_v263853
    v263971 = v263969 + 1.0
    v263972 = i_v263875 * v263971
    v263974 = jnp.exp(jnp.clip(v263834, -709.0, 709.0))
    v263975 = jnp.where(v263864, v263972, v263974)
    v263976 = jnp.where(v263836, v263842, v263975)
    v263985 = v263976 - 1.0
    v263986 = i_v263984 * v263985
    v264012 = jnp.where(i_v261764, v263822, v263986)
    v264013 = v264011 + v264012
    v275029 = v259131 < i_v7952
    v276395 = v259131 - i_v7258
    v276396 = NJDV * v276395
    v276398 = v276396 + i_v277
    v276399 = NJH - v276398
    v276400 = v276399 - 0.01
    v276411 = v276400 * v276400
    v276412 = v276411 + i_v276407
    v276413 = jnp.sqrt(jnp.maximum(v276412, 1e-300))
    v276414 = v276400 + v276413
    v276415 = 0.5 * v276414
    v276416 = NJH - v276415
    v276420 = v276416 - i_v277
    v276421 = v276420 - 0.01
    v276432 = v276421 * v276421
    v276433 = v276432 + i_v276428
    v276434 = jnp.sqrt(jnp.maximum(v276433, 1e-300))
    v276438 = v276421 + v276434
    v276439 = 0.5 * v276438
    v276440 = i_v277 + v276439
    v276459 = jnp.where(i_v276390, v276440, i_v277)
    v276460 = jnp.divide(v259131, jnp.where((v276459 == 0.0) | jnp.logical_not(jnp.isfinite(v276459)), 1e-300, v276459))
    v276470 = v276459 - i_v277
    v276471 = i_v7258 * v276470
    v276473 = jnp.divide(v276471, jnp.where((i_v276472 == 0.0) | jnp.logical_not(jnp.isfinite(i_v276472)), 1e-300, i_v276472))
    v276474 = v276460 + v276473
    v276475 = i_v596 * v276474
    v276476 = v276475 < 0.0
    v276477 = -v276475
    v276478 = jnp.where(v276476, v276477, v276475)
    v276479 = v276478 < 230.25850929940458
    v276507 = jnp.exp(jnp.clip(v276475, -709.0, 709.0))
    v276521 = v276475 < -230.25850929940458
    v276535 = -230.25850929940458 - v276475
    v276554 = v276535 * 0.3333333333333333
    v276555 = 1.0 + v276554
    v276556 = v276535 * v276555
    v276557 = 0.5 * v276556
    v276558 = 1.0 + v276557
    v276559 = v276535 * v276558
    v276560 = 1.0 + v276559
    v276561 = jnp.divide(1e-100, jnp.where((v276560 == 0.0) | jnp.logical_not(jnp.isfinite(v276560)), 1e-300, v276560))
    v276574 = v276475 - 230.25850929940458
    v276591 = v276574 * 0.3333333333333333
    v276592 = 1.0 + v276591
    v276593 = v276574 * v276592
    v276594 = 0.5 * v276593
    v276595 = 1.0 + v276594
    v276596 = v276574 * v276595
    v276597 = 1.0 + v276596
    v276598 = 1e100 * v276597
    v280207 = jnp.where(v276521, v276561, v276598)
    v280206 = jnp.where(v276479, v276507, v280207)
    v278740 = v259131 - i_v7952
    v279484 = v278740 * i_v279463
    v279485 = 1.0 + v279484
    v279488 = v279485 * i_v279486
    v280193 = jnp.where(v275029, v280206, v279488)
    v280268 = v280193 - 1.0
    v283378 = jnp.where(i_v274582, v280268, 0.0)
    v283383 = i_v1474 * v283378
    v275040 = v259131 * i_v596
    v275041 = 0.5 * v275040
    v275042 = v275041 < 0.0
    v275043 = -v275041
    v275044 = jnp.where(v275042, v275043, v275041)
    v275045 = v275044 < 230.25850929940458
    v275056 = jnp.exp(jnp.clip(v275041, -709.0, 709.0))
    v275063 = v275041 < -230.25850929940458
    v275069 = -230.25850929940458 - v275041
    v275078 = v275069 * 0.3333333333333333
    v275079 = 1.0 + v275078
    v275080 = v275069 * v275079
    v275081 = 0.5 * v275080
    v275082 = 1.0 + v275081
    v275083 = v275069 * v275082
    v275084 = 1.0 + v275083
    v275085 = jnp.divide(1e-100, jnp.where((v275084 == 0.0) | jnp.logical_not(jnp.isfinite(v275084)), 1e-300, v275084))
    v275090 = v275041 - 230.25850929940458
    v275097 = v275090 * 0.3333333333333333
    v275098 = 1.0 + v275097
    v275099 = v275090 * v275098
    v275100 = 0.5 * v275099
    v275101 = 1.0 + v275100
    v275102 = v275090 * v275101
    v275103 = 1.0 + v275102
    v275104 = 1e100 * v275103
    v280334 = jnp.where(v275063, v275085, v275104)
    v280333 = jnp.where(v275045, v275056, v280334)
    v278742 = v278740 * i_v596
    v278743 = 1.0 + v278742
    v279206 = v278743 * i_v15213
    v279207 = jnp.sqrt(jnp.maximum(v279206, 1e-300))
    v280314 = jnp.where(v275029, v280333, v279207)
    v286608 = jnp.where(i_v274582, v280314, 0.0)
    v286616 = v286608 - 1.0
    v280431 = v259131 > 0.0
    v280425 = jnp.divide(1.0, jnp.where((v280314 == 0.0) | jnp.logical_not(jnp.isfinite(v280314)), 1e-300, v280314))
    v280609 = 2.0 + v280425
    v280610 = v280425 + 1.0
    v280611 = v280425 + 3.0
    v280612 = v280610 * v280611
    v280613 = jnp.sqrt(jnp.maximum(v280612, 1e-300))
    v280614 = v280609 + v280613
    v280615 = jnp.log(jnp.maximum(v280614, 1e-300))
    v280616 = i_v594 * v280615
    v280617 = 2.0 * v280616
    v280620 = -v259131
    v280624 = 2.0 * v280314
    v280626 = v280624 + 1.0
    v280628 = 1.0 + v280314
    v280631 = 3.0 * v280314
    v280632 = 1.0 + v280631
    v280633 = v280628 * v280632
    v280634 = jnp.sqrt(jnp.maximum(v280633, 1e-300))
    v280635 = v280626 + v280634
    v280636 = jnp.log(jnp.maximum(v280635, 1e-300))
    v280637 = i_v594 * v280636
    v280638 = 2.0 * v280637
    v280639 = v280620 + v280638
    v281102 = jnp.where(v280431, v280617, v280639)
    v281103 = i_v8881 - v281102
    v281107 = v259131 + v281103
    v281108 = v259131 - v281103
    v281110 = v281108 * v281108
    v281115 = v281110 + i_v281114
    v281116 = jnp.sqrt(jnp.maximum(v281115, 1e-300))
    v281117 = v281107 - v281116
    v281118 = 0.5 * v281117
    v284752 = jnp.where(i_v274582, v281118, 0.0)
    v284755 = i_v2324 - v284752
    v285665 = v284755 * i_v3907
    v285666 = jnp.sqrt(jnp.maximum(v285665, 1e-300))
    v285669 = v284755 * i_v3907
    v285671 = jnp.power(jnp.maximum(v285669, 0.0), i_v76)
    v286134 = jnp.where(i_v285203, v285666, v285671)
    v286135 = i_v3628 * v286134
    v286617 = v286616 * v286135
    v286618 = i_v1108 * v286617
    v284764 = jnp.where(i_v274582, v281102, 0.0)
    v284767 = jnp.divide(v284764, jnp.where((v284755 == 0.0) | jnp.logical_not(jnp.isfinite(v284755)), 1e-300, v284755))
    v284768 = 1.0 - v284767
    v284769 = jnp.sqrt(jnp.maximum(v284768, 1e-300))
    v284770 = 1.0 - v284769
    v285207 = v284770 * v284770
    v285208 = jnp.log(jnp.maximum(v284770, 1e-300))
    v285209 = v285207 * v285208
    v285212 = jnp.divide(v285209, v284769)
    v285213 = v285212 + v284770
    v285219 = v285213 * i_v285218
    v285222 = jnp.where(i_v285203, 0.0, v285219)
    v285223 = v284770 + v285222
    v286632 = v286618 * v285223
    v286633 = i_v118 * v286632
    v294124 = jnp.where(i_v284273, 0.0, v286633)
    v294179 = v283383 + v294124
    v289214, v287112 = jax.tree_util.tree_map(lambda _t, _f: jnp.where(i_v284273, _t, _f), (0.0, 0.0), (v286618, v286135))
    v287128 = v287112 * i_v2516
    v287130 = jnp.where(i_v284273, 0.0, v284755)
    v287132 = jnp.divide(v287128, jnp.where((v287130 == 0.0) | jnp.logical_not(jnp.isfinite(v287130)), 1e-300, v287130))
    v287133 = i_v4127 * v287132
    v287580 = jnp.divide(i_v287579, jnp.where((v287133 == 0.0) | jnp.logical_not(jnp.isfinite(v287133)), 1e-300, v287133))
    v287582 = v287580 * v287580
    v287584 = v287582 * v287582
    v287587 = v287584 + 1.0
    v287588 = jnp.divide(v287584, jnp.where((v287587 == 0.0) | jnp.logical_not(jnp.isfinite(v287587)), 1e-300, v287587))
    v287589 = jnp.sqrt(jnp.maximum(v287588, 1e-300))
    v287591 = v287589 < 0.0
    v287592 = -v287589
    v287593 = jnp.where(v287591, v287592, v287589)
    v287594 = jnp.sqrt(jnp.maximum(v287593, 1e-300))
    v288104 = v287580 * v287594
    v288105 = 2.0 * v288104
    v288109 = v288105 - v287589
    v288127 = v288109 - 1.0
    v288093 = jnp.divide(v287133, v287594)
    v288094 = 0.375 * v288093
    v288095 = jnp.sqrt(jnp.maximum(v288094, 1e-300))
    v288128 = v288127 * v288095
    v288133 = v288128 > 0.0
    v288442 = 0.5178164370971076 * v288128
    v288443 = 1.0 + v288442
    v288444 = jnp.divide(1.0, jnp.where((v288443 == 0.0) | jnp.logical_not(jnp.isfinite(v288443)), 1e-300, v288443))
    v288450 = 0.5178164370971076 * v288128
    v288451 = 1.0 - v288450
    v288452 = jnp.divide(1.0, jnp.where((v288451 == 0.0) | jnp.logical_not(jnp.isfinite(v288451)), 1e-300, v288451))
    v288492 = jnp.where(v288133, v288444, v288452)
    v288494 = 0.29214664 * v288492
    v288801 = v288492 * v288492
    v288802 = 0.26992878119627894 * v288801
    v288803 = v288494 + v288802
    v289111 = v288801 * v288492
    v289112 = 0.43792457880372104 * v289111
    v289113 = v288803 + v289112
    v288130 = v288128 * v288128
    v288456 = -v288130
    v288117 = i_v3979 * v287580
    v288118 = v288117 * v287594
    v288119 = i_v3979 * v287589
    v288120 = v288118 - v288119
    v287599 = v287589 * v287594
    v288122 = v287133 * v287599
    v288123 = 0.5 * v288122
    v288124 = v288120 + v288123
    v288460 = v288456 + v288124
    v288462 = v288460 > -230.25850929940458
    v288467 = jnp.exp(jnp.clip(v288460, -709.0, 709.0))
    v288473 = -230.25850929940458 - v288460
    v288482 = v288473 * 0.3333333333333333
    v288483 = 1.0 + v288482
    v288484 = v288473 * v288483
    v288485 = 0.5 * v288484
    v288486 = 1.0 + v288485
    v288487 = v288473 * v288486
    v288488 = 1.0 + v288487
    v288489 = jnp.divide(1e-100, jnp.where((v288488 == 0.0) | jnp.logical_not(jnp.isfinite(v288488)), 1e-300, v288488))
    v289114 = jnp.where(v288462, v288467, v288489)
    v289115 = v289113 * v289114
    v289128 = v288124 > -230.25850929940458
    v289130 = jnp.exp(jnp.clip(v288124, -709.0, 709.0))
    v289133 = -230.25850929940458 - v288124
    v289138 = v289133 * 0.3333333333333333
    v289139 = 1.0 + v289138
    v289140 = v289133 * v289139
    v289141 = 0.5 * v289140
    v289142 = 1.0 + v289141
    v289143 = v289133 * v289142
    v289144 = 1.0 + v289143
    v289145 = jnp.divide(1e-100, jnp.where((v289144 == 0.0) | jnp.logical_not(jnp.isfinite(v289144)), 1e-300, v289144))
    v289147 = jnp.where(v289128, v289130, v289145)
    v289148 = 2.0 * v289147
    v289153 = v289148 - v289115
    v289167 = jnp.where(v288133, v289115, v289153)
    v289168 = i_v3979 * v289167
    v289181 = jnp.divide(v289168, v288095)
    v289182 = 0.886226925452758 * v289181
    v289223 = v289214 * v289182
    v288081 = jnp.where(i_v284273, 0.0, v285223)
    v288062 = v287133 * v287599
    v288063 = 1.0 + v288062
    v288064 = jnp.divide(1.0, jnp.where((v288063 == 0.0) | jnp.logical_not(jnp.isfinite(v288063)), 1e-300, v288063))
    v288069 = v287133 * v287599
    v288070 = 1.0 + v288069
    v288075 = jnp.power(jnp.maximum(v288070, 0.0), i_v288051)
    v288084 = jnp.where(i_v288054, v288064, v288075)
    v288085 = v288081 * v288084
    v288086 = v288081 + v288084
    v288087 = jnp.divide(v288085, jnp.where((v288086 == 0.0) | jnp.logical_not(jnp.isfinite(v288086)), 1e-300, v288086))
    v289236 = v289223 * v288087
    v289237 = i_v145 * v289236
    v294184 = jnp.where(i_v286646, 0.0, v289237)
    v294220 = v294179 + v294184
    v281582 = v259131 + i_v9052
    v281583 = v259131 - i_v9052
    v281585 = v281583 * v281583
    v282051 = v281585 + i_v282050
    v282052 = jnp.sqrt(jnp.maximum(v282051, 1e-300))
    v282053 = v281582 - v282052
    v282054 = 0.5 * v282053
    v290180 = jnp.where(i_v274582, v282054, 0.0)
    v290267 = i_v61 - v290180
    v290730 = v290267 * i_v3898
    v290209 = i_v61 - v290180
    v290236 = v290209 * i_v3907
    v290237 = jnp.sqrt(jnp.maximum(v290236, 1e-300))
    v290240 = i_v61 - v290180
    v290242 = v290240 * i_v3907
    v290244 = jnp.power(jnp.maximum(v290242, 0.0), i_v76)
    v290731 = jnp.where(i_v289708, v290237, v290244)
    v290732 = jnp.divide(v290730, jnp.where((v290731 == 0.0) | jnp.logical_not(jnp.isfinite(v290731)), 1e-300, v290731))
    v290733 = i_v2837 * v290732
    v291315 = v259131 * v290733
    v291316 = v291315 * v290733
    v291191 = jnp.divide(i_v291190, jnp.where((v290733 == 0.0) | jnp.logical_not(jnp.isfinite(v290733)), 1e-300, v290733))
    v291192 = v291191 < 0.0
    v291193 = -v291191
    v291194 = jnp.where(v291192, v291193, v291191)
    v291195 = v291194 < 230.25850929940458
    v291206 = jnp.exp(jnp.clip(v291191, -709.0, 709.0))
    v291212 = v291191 < -230.25850929940458
    v291218 = -230.25850929940458 - v291191
    v291227 = v291218 * 0.3333333333333333
    v291228 = 1.0 + v291227
    v291229 = v291218 * v291228
    v291230 = 0.5 * v291229
    v291231 = 1.0 + v291230
    v291232 = v291218 * v291231
    v291233 = 1.0 + v291232
    v291234 = jnp.divide(1e-100, jnp.where((v291233 == 0.0) | jnp.logical_not(jnp.isfinite(v291233)), 1e-300, v291233))
    v291239 = v291191 - 230.25850929940458
    v291246 = v291239 * 0.3333333333333333
    v291247 = 1.0 + v291246
    v291248 = v291239 * v291247
    v291249 = 0.5 * v291248
    v291250 = 1.0 + v291249
    v291251 = v291239 * v291250
    v291252 = 1.0 + v291251
    v291253 = 1e100 * v291252
    v291318 = jnp.where(v291212, v291234, v291253)
    v291317 = jnp.where(v291195, v291206, v291318)
    v291319 = v291316 * v291317
    v291320 = i_v176 * v291319
    v294223 = jnp.where(i_v289688, 0.0, v291320)
    v294246 = v294220 + v294223
    v282062 = v259131 * v259131
    v282066 = v282062 + 4e-12
    v282067 = jnp.sqrt(jnp.maximum(v282066, 1e-300))
    v282068 = v259131 - v282067
    v282069 = 0.5 * v282068
    v291797 = jnp.where(i_v274582, v282069, 0.0)
    v292147 = v291797 > i_v292146
    v293062 = v291797 * i_v6514
    v293063 = v293062 < 0.0
    v293064 = -v293062
    v293065 = jnp.where(v293063, v293064, v293062)
    v293074 = -v293062
    v293075 = jnp.where(v293063, v293074, v293062)
    v293076 = v293065 * v293075
    v293085 = -v293062
    v293086 = jnp.where(v293063, v293085, v293062)
    v293087 = v293076 * v293086
    v293096 = -v293062
    v293097 = jnp.where(v293063, v293096, v293062)
    v293098 = v293087 * v293097
    v293101 = v291797 * i_v6514
    v293102 = v293101 < 0.0
    v293103 = -v293101
    v293104 = jnp.where(v293102, v293103, v293101)
    v293109 = jnp.power(jnp.maximum(v293104, 0.0), i_v217)
    v293112 = jnp.where(i_v292607, v293098, v293109)
    v293113 = 1.0 - v293112
    v293114 = jnp.divide(1.0, jnp.where((v293113 == 0.0) | jnp.logical_not(jnp.isfinite(v293113)), 1e-300, v293113))
    v293581 = v291797 + i_v293580
    v294044 = v293581 * i_v6517
    v294045 = i_v5340 + v294044
    v294248 = jnp.where(v292147, v293114, v294045)
    v294247 = jnp.where(i_v291782, 1.0, v294248)
    v294249 = v294246 * v294247
    v318796 = jnp.where(i_v8640, 0.0, v294249)
    v318949 = i_v7324 * v318796
    v277482 = v259131 - i_v277476
    v277483 = NJDV * v277482
    v277485 = v277483 + i_v281
    v277486 = NJH - v277485
    v277487 = v277486 - 0.01
    v277498 = v277487 * v277487
    v277499 = v277498 + i_v277494
    v277500 = jnp.sqrt(jnp.maximum(v277499, 1e-300))
    v277501 = v277487 + v277500
    v277502 = 0.5 * v277501
    v277503 = NJH - v277502
    v277507 = v277503 - i_v281
    v277508 = v277507 - 0.01
    v277519 = v277508 * v277508
    v277520 = v277519 + i_v277515
    v277521 = jnp.sqrt(jnp.maximum(v277520, 1e-300))
    v277525 = v277508 + v277521
    v277526 = 0.5 * v277525
    v277527 = i_v281 + v277526
    v277546 = jnp.where(i_v277477, v277527, i_v281)
    v277547 = jnp.divide(v259131, jnp.where((v277546 == 0.0) | jnp.logical_not(jnp.isfinite(v277546)), 1e-300, v277546))
    v277557 = v277546 - i_v281
    v277558 = i_v277476 * v277557
    v277560 = jnp.divide(v277558, jnp.where((i_v277559 == 0.0) | jnp.logical_not(jnp.isfinite(i_v277559)), 1e-300, i_v277559))
    v277561 = v277547 + v277560
    v277562 = i_v596 * v277561
    v277563 = v277562 < 0.0
    v277564 = -v277562
    v277565 = jnp.where(v277563, v277564, v277562)
    v277566 = v277565 < 230.25850929940458
    v277594 = jnp.exp(jnp.clip(v277562, -709.0, 709.0))
    v277608 = v277562 < -230.25850929940458
    v277622 = -230.25850929940458 - v277562
    v277641 = v277622 * 0.3333333333333333
    v277642 = 1.0 + v277641
    v277643 = v277622 * v277642
    v277644 = 0.5 * v277643
    v277645 = 1.0 + v277644
    v277646 = v277622 * v277645
    v277647 = 1.0 + v277646
    v277648 = jnp.divide(1e-100, jnp.where((v277647 == 0.0) | jnp.logical_not(jnp.isfinite(v277647)), 1e-300, v277647))
    v277661 = v277562 - 230.25850929940458
    v277678 = v277661 * 0.3333333333333333
    v277679 = 1.0 + v277678
    v277680 = v277661 * v277679
    v277681 = 0.5 * v277680
    v277682 = 1.0 + v277681
    v277683 = v277661 * v277682
    v277684 = 1.0 + v277683
    v277685 = 1e100 * v277684
    v280277 = jnp.where(v277608, v277648, v277685)
    v280276 = jnp.where(v277566, v277594, v280277)
    v279818 = v278740 * i_v279797
    v279819 = 1.0 + v279818
    v279822 = v279819 * i_v279820
    v280269 = jnp.where(v275029, v280276, v279822)
    v280308 = v280269 - 1.0
    v295917 = jnp.where(i_v274582, v280308, 0.0)
    v295998 = i_v1624 * v295917
    v299422 = v286608 - 1.0
    v297433 = i_v2333 - v284752
    v298410 = v297433 * i_v3910
    v298411 = jnp.sqrt(jnp.maximum(v298410, 1e-300))
    v298414 = v297433 * i_v3910
    v298416 = jnp.power(jnp.maximum(v298414, 0.0), i_v82)
    v298879 = jnp.where(i_v297949, v298411, v298416)
    v298880 = i_v3763 * v298879
    v299423 = v299422 * v298880
    v299424 = i_v1117 * v299423
    v297514 = jnp.divide(v284764, jnp.where((v297433 == 0.0) | jnp.logical_not(jnp.isfinite(v297433)), 1e-300, v297433))
    v297515 = 1.0 - v297514
    v297516 = jnp.sqrt(jnp.maximum(v297515, 1e-300))
    v297517 = 1.0 - v297516
    v297952 = v297517 * v297517
    v297953 = jnp.log(jnp.maximum(v297517, 1e-300))
    v297954 = v297952 * v297953
    v297957 = jnp.divide(v297954, v297516)
    v297958 = v297957 + v297517
    v297964 = v297958 * i_v297963
    v297967 = jnp.where(i_v297949, 0.0, v297964)
    v297968 = v297517 + v297967
    v299438 = v299424 * v297968
    v299439 = i_v124 * v299438
    v306044 = jnp.where(i_v296887, 0.0, v299439)
    v306099 = v295998 + v306044
    v301318, v299917 = jax.tree_util.tree_map(lambda _t, _f: jnp.where(i_v296887, _t, _f), (0.0, 0.0), (v299424, v298880))
    v299933 = v299917 * i_v2678
    v299935 = jnp.where(i_v296887, 0.0, v297433)
    v299937 = jnp.divide(v299933, jnp.where((v299935 == 0.0) | jnp.logical_not(jnp.isfinite(v299935)), 1e-300, v299935))
    v299938 = i_v4262 * v299937
    v300384 = jnp.divide(i_v300383, jnp.where((v299938 == 0.0) | jnp.logical_not(jnp.isfinite(v299938)), 1e-300, v299938))
    v300385 = v300384 * v300384
    v300386 = v300385 * v300385
    v300389 = v300386 + 1.0
    v300390 = jnp.divide(v300386, jnp.where((v300389 == 0.0) | jnp.logical_not(jnp.isfinite(v300389)), 1e-300, v300389))
    v300391 = jnp.sqrt(jnp.maximum(v300390, 1e-300))
    v300392 = v300391 < 0.0
    v300393 = -v300391
    v300394 = jnp.where(v300392, v300393, v300391)
    v300395 = jnp.sqrt(jnp.maximum(v300394, 1e-300))
    v300900 = v300384 * v300395
    v300901 = 2.0 * v300900
    v300905 = v300901 - v300391
    v300921 = v300905 - 1.0
    v300890 = jnp.divide(v299938, v300395)
    v300891 = 0.375 * v300890
    v300892 = jnp.sqrt(jnp.maximum(v300891, 1e-300))
    v300922 = v300921 * v300892
    v300925 = v300922 > 0.0
    v301006 = 0.5178164370971076 * v300922
    v301007 = 1.0 + v301006
    v301008 = jnp.divide(1.0, jnp.where((v301007 == 0.0) | jnp.logical_not(jnp.isfinite(v301007)), 1e-300, v301007))
    v301013 = 0.5178164370971076 * v300922
    v301014 = 1.0 - v301013
    v301015 = jnp.divide(1.0, jnp.where((v301014 == 0.0) | jnp.logical_not(jnp.isfinite(v301014)), 1e-300, v301014))
    v301055 = jnp.where(v300925, v301008, v301015)
    v301057 = 0.29214664 * v301055
    v301136 = v301055 * v301055
    v301137 = 0.26992878119627894 * v301136
    v301138 = v301057 + v301137
    v301218 = v301136 * v301055
    v301219 = 0.43792457880372104 * v301218
    v301220 = v301138 + v301219
    v300923 = v300922 * v300922
    v301019 = -v300923
    v300912 = i_v3984 * v300384
    v300913 = v300912 * v300395
    v300914 = i_v3984 * v300391
    v300915 = v300913 - v300914
    v300399 = v300391 * v300395
    v300917 = v299938 * v300399
    v300918 = 0.5 * v300917
    v300919 = v300915 + v300918
    v301023 = v301019 + v300919
    v301025 = v301023 > -230.25850929940458
    v301030 = jnp.exp(jnp.clip(v301023, -709.0, 709.0))
    v301036 = -230.25850929940458 - v301023
    v301045 = v301036 * 0.3333333333333333
    v301046 = 1.0 + v301045
    v301047 = v301036 * v301046
    v301048 = 0.5 * v301047
    v301049 = 1.0 + v301048
    v301050 = v301036 * v301049
    v301051 = 1.0 + v301050
    v301052 = jnp.divide(1e-100, jnp.where((v301051 == 0.0) | jnp.logical_not(jnp.isfinite(v301051)), 1e-300, v301051))
    v301221 = jnp.where(v301025, v301030, v301052)
    v301222 = v301220 * v301221
    v301233 = v300919 > -230.25850929940458
    v301235 = jnp.exp(jnp.clip(v300919, -709.0, 709.0))
    v301238 = -230.25850929940458 - v300919
    v301243 = v301238 * 0.3333333333333333
    v301244 = 1.0 + v301243
    v301245 = v301238 * v301244
    v301246 = 0.5 * v301245
    v301247 = 1.0 + v301246
    v301248 = v301238 * v301247
    v301249 = 1.0 + v301248
    v301250 = jnp.divide(1e-100, jnp.where((v301249 == 0.0) | jnp.logical_not(jnp.isfinite(v301249)), 1e-300, v301249))
    v301252 = jnp.where(v301233, v301235, v301250)
    v301253 = 2.0 * v301252
    v301258 = v301253 - v301222
    v301272 = jnp.where(v300925, v301222, v301258)
    v301273 = i_v3984 * v301272
    v301286 = jnp.divide(v301273, v300892)
    v301287 = 0.886226925452758 * v301286
    v301327 = v301318 * v301287
    v300879 = jnp.where(i_v296887, 0.0, v297968)
    v300861 = v299938 * v300399
    v300862 = 1.0 + v300861
    v300863 = jnp.divide(1.0, jnp.where((v300862 == 0.0) | jnp.logical_not(jnp.isfinite(v300862)), 1e-300, v300862))
    v300867 = v299938 * v300399
    v300868 = 1.0 + v300867
    v300873 = jnp.power(jnp.maximum(v300868, 0.0), i_v300850)
    v300882 = jnp.where(i_v300853, v300863, v300873)
    v300883 = v300879 * v300882
    v300884 = v300879 + v300882
    v300885 = jnp.divide(v300883, jnp.where((v300884 == 0.0) | jnp.logical_not(jnp.isfinite(v300884)), 1e-300, v300884))
    v301340 = v301327 * v300885
    v301341 = i_v151 * v301340
    v306104 = jnp.where(i_v299452, 0.0, v301341)
    v306140 = v306099 + v306104
    v302406 = i_v65 - v290180
    v302869 = v302406 * i_v3901
    v302348 = i_v65 - v290180
    v302375 = v302348 * i_v3910
    v302376 = jnp.sqrt(jnp.maximum(v302375, 1e-300))
    v302379 = i_v65 - v290180
    v302381 = v302379 * i_v3910
    v302383 = jnp.power(jnp.maximum(v302381, 0.0), i_v82)
    v302870 = jnp.where(i_v301811, v302376, v302383)
    v302871 = jnp.divide(v302869, jnp.where((v302870 == 0.0) | jnp.logical_not(jnp.isfinite(v302870)), 1e-300, v302870))
    v302872 = i_v2840 * v302871
    v303453 = v259131 * v302872
    v303454 = v303453 * v302872
    v303329 = jnp.divide(i_v303328, jnp.where((v302872 == 0.0) | jnp.logical_not(jnp.isfinite(v302872)), 1e-300, v302872))
    v303330 = v303329 < 0.0
    v303331 = -v303329
    v303332 = jnp.where(v303330, v303331, v303329)
    v303333 = v303332 < 230.25850929940458
    v303344 = jnp.exp(jnp.clip(v303329, -709.0, 709.0))
    v303350 = v303329 < -230.25850929940458
    v303356 = -230.25850929940458 - v303329
    v303365 = v303356 * 0.3333333333333333
    v303366 = 1.0 + v303365
    v303367 = v303356 * v303366
    v303368 = 0.5 * v303367
    v303369 = 1.0 + v303368
    v303370 = v303356 * v303369
    v303371 = 1.0 + v303370
    v303372 = jnp.divide(1e-100, jnp.where((v303371 == 0.0) | jnp.logical_not(jnp.isfinite(v303371)), 1e-300, v303371))
    v303377 = v303329 - 230.25850929940458
    v303384 = v303377 * 0.3333333333333333
    v303385 = 1.0 + v303384
    v303386 = v303377 * v303385
    v303387 = 0.5 * v303386
    v303388 = 1.0 + v303387
    v303389 = v303377 * v303388
    v303390 = 1.0 + v303389
    v303391 = 1e100 * v303390
    v303456 = jnp.where(v303350, v303372, v303391)
    v303455 = jnp.where(v303333, v303344, v303456)
    v303457 = v303454 * v303455
    v303458 = i_v182 * v303457
    v306143 = jnp.where(i_v301792, 0.0, v303458)
    v306166 = v306140 + v306143
    v304067 = v291797 > i_v304066
    v304982 = v291797 * i_v6546
    v304983 = v304982 < 0.0
    v304984 = -v304982
    v304985 = jnp.where(v304983, v304984, v304982)
    v304994 = -v304982
    v304995 = jnp.where(v304983, v304994, v304982)
    v304996 = v304985 * v304995
    v305005 = -v304982
    v305006 = jnp.where(v304983, v305005, v304982)
    v305007 = v304996 * v305006
    v305016 = -v304982
    v305017 = jnp.where(v304983, v305016, v304982)
    v305018 = v305007 * v305017
    v305021 = v291797 * i_v6546
    v305022 = v305021 < 0.0
    v305023 = -v305021
    v305024 = jnp.where(v305022, v305023, v305021)
    v305029 = jnp.power(jnp.maximum(v305024, 0.0), i_v221)
    v305032 = jnp.where(i_v304527, v305018, v305029)
    v305033 = 1.0 - v305032
    v305034 = jnp.divide(1.0, jnp.where((v305033 == 0.0) | jnp.logical_not(jnp.isfinite(v305033)), 1e-300, v305033))
    v305501 = v291797 + i_v305500
    v305964 = v305501 * i_v6548
    v305965 = i_v5451 + v305964
    v306168 = jnp.where(v304067, v305034, v305965)
    v306167 = jnp.where(i_v303920, 1.0, v306168)
    v306169 = v306166 * v306167
    v319108 = jnp.where(i_v8686, 0.0, v306169)
    v319185 = i_v7338 * v319108
    v319186 = v318949 + v319185
    v278533 = v259131 - i_v278527
    v278534 = NJDV * v278533
    v278536 = v278534 + i_v285
    v278537 = NJH - v278536
    v278538 = v278537 - 0.01
    v278549 = v278538 * v278538
    v278550 = v278549 + i_v278545
    v278551 = jnp.sqrt(jnp.maximum(v278550, 1e-300))
    v278552 = v278538 + v278551
    v278553 = 0.5 * v278552
    v278554 = NJH - v278553
    v278558 = v278554 - i_v285
    v278559 = v278558 - 0.01
    v278570 = v278559 * v278559
    v278571 = v278570 + i_v278566
    v278572 = jnp.sqrt(jnp.maximum(v278571, 1e-300))
    v278576 = v278559 + v278572
    v278577 = 0.5 * v278576
    v278578 = i_v285 + v278577
    v278597 = jnp.where(i_v278528, v278578, i_v285)
    v278598 = jnp.divide(v259131, jnp.where((v278597 == 0.0) | jnp.logical_not(jnp.isfinite(v278597)), 1e-300, v278597))
    v278608 = v278597 - i_v285
    v278609 = i_v278527 * v278608
    v278611 = jnp.divide(v278609, jnp.where((i_v278610 == 0.0) | jnp.logical_not(jnp.isfinite(i_v278610)), 1e-300, i_v278610))
    v278612 = v278598 + v278611
    v278613 = i_v596 * v278612
    v278614 = v278613 < 0.0
    v278615 = -v278613
    v278616 = jnp.where(v278614, v278615, v278613)
    v278617 = v278616 < 230.25850929940458
    v278645 = jnp.exp(jnp.clip(v278613, -709.0, 709.0))
    v278659 = v278613 < -230.25850929940458
    v278673 = -230.25850929940458 - v278613
    v278692 = v278673 * 0.3333333333333333
    v278693 = 1.0 + v278692
    v278694 = v278673 * v278693
    v278695 = 0.5 * v278694
    v278696 = 1.0 + v278695
    v278697 = v278673 * v278696
    v278698 = 1.0 + v278697
    v278699 = jnp.divide(1e-100, jnp.where((v278698 == 0.0) | jnp.logical_not(jnp.isfinite(v278698)), 1e-300, v278698))
    v278712 = v278613 - 230.25850929940458
    v278729 = v278712 * 0.3333333333333333
    v278730 = 1.0 + v278729
    v278731 = v278712 * v278730
    v278732 = 0.5 * v278731
    v278733 = 1.0 + v278732
    v278734 = v278712 * v278733
    v278735 = 1.0 + v278734
    v278736 = 1e100 * v278735
    v280311 = jnp.where(v278659, v278699, v278736)
    v280310 = jnp.where(v278617, v278645, v280311)
    v280188 = v278740 * i_v280167
    v280189 = 1.0 + v280188
    v280192 = v280189 * i_v280190
    v280309 = jnp.where(v275029, v280310, v280192)
    v280312 = v280309 - 1.0
    v308226 = jnp.where(i_v274582, v280312, 0.0)
    v308383 = i_v1771 * v308226
    v311807 = v286608 - 1.0
    v309818 = i_v2342 - v284752
    v310795 = v309818 * i_v3913
    v310796 = jnp.sqrt(jnp.maximum(v310795, 1e-300))
    v310799 = v309818 * i_v3913
    v310801 = jnp.power(jnp.maximum(v310799, 0.0), i_v88)
    v311264 = jnp.where(i_v310334, v310796, v310801)
    v311265 = i_v3895 * v311264
    v311808 = v311807 * v311265
    v311809 = i_v1126 * v311808
    v309899 = jnp.divide(v284764, jnp.where((v309818 == 0.0) | jnp.logical_not(jnp.isfinite(v309818)), 1e-300, v309818))
    v309900 = 1.0 - v309899
    v309901 = jnp.sqrt(jnp.maximum(v309900, 1e-300))
    v309902 = 1.0 - v309901
    v310337 = v309902 * v309902
    v310338 = jnp.log(jnp.maximum(v309902, 1e-300))
    v310339 = v310337 * v310338
    v310342 = jnp.divide(v310339, v309901)
    v310343 = v310342 + v309902
    v310349 = v310343 * i_v310348
    v310352 = jnp.where(i_v310334, 0.0, v310349)
    v310353 = v309902 + v310352
    v311823 = v311809 * v310353
    v311824 = i_v130 * v311823
    v318429 = jnp.where(i_v309272, 0.0, v311824)
    v318484 = v308383 + v318429
    v313703, v312302 = jax.tree_util.tree_map(lambda _t, _f: jnp.where(i_v309272, _t, _f), (0.0, 0.0), (v311809, v311265))
    v312318 = v312302 * i_v2834
    v312320 = jnp.where(i_v309272, 0.0, v309818)
    v312322 = jnp.divide(v312318, jnp.where((v312320 == 0.0) | jnp.logical_not(jnp.isfinite(v312320)), 1e-300, v312320))
    v312323 = i_v4394 * v312322
    v312769 = jnp.divide(i_v312768, jnp.where((v312323 == 0.0) | jnp.logical_not(jnp.isfinite(v312323)), 1e-300, v312323))
    v312770 = v312769 * v312769
    v312771 = v312770 * v312770
    v312774 = v312771 + 1.0
    v312775 = jnp.divide(v312771, jnp.where((v312774 == 0.0) | jnp.logical_not(jnp.isfinite(v312774)), 1e-300, v312774))
    v312776 = jnp.sqrt(jnp.maximum(v312775, 1e-300))
    v312777 = v312776 < 0.0
    v312778 = -v312776
    v312779 = jnp.where(v312777, v312778, v312776)
    v312780 = jnp.sqrt(jnp.maximum(v312779, 1e-300))
    v313285 = v312769 * v312780
    v313286 = 2.0 * v313285
    v313290 = v313286 - v312776
    v313306 = v313290 - 1.0
    v313275 = jnp.divide(v312323, v312780)
    v313276 = 0.375 * v313275
    v313277 = jnp.sqrt(jnp.maximum(v313276, 1e-300))
    v313307 = v313306 * v313277
    v313310 = v313307 > 0.0
    v313391 = 0.5178164370971076 * v313307
    v313392 = 1.0 + v313391
    v313393 = jnp.divide(1.0, jnp.where((v313392 == 0.0) | jnp.logical_not(jnp.isfinite(v313392)), 1e-300, v313392))
    v313398 = 0.5178164370971076 * v313307
    v313399 = 1.0 - v313398
    v313400 = jnp.divide(1.0, jnp.where((v313399 == 0.0) | jnp.logical_not(jnp.isfinite(v313399)), 1e-300, v313399))
    v313440 = jnp.where(v313310, v313393, v313400)
    v313442 = 0.29214664 * v313440
    v313521 = v313440 * v313440
    v313522 = 0.26992878119627894 * v313521
    v313523 = v313442 + v313522
    v313603 = v313521 * v313440
    v313604 = 0.43792457880372104 * v313603
    v313605 = v313523 + v313604
    v313308 = v313307 * v313307
    v313404 = -v313308
    v313297 = i_v3986 * v312769
    v313298 = v313297 * v312780
    v313299 = i_v3986 * v312776
    v313300 = v313298 - v313299
    v312784 = v312776 * v312780
    v313302 = v312323 * v312784
    v313303 = 0.5 * v313302
    v313304 = v313300 + v313303
    v313408 = v313404 + v313304
    v313410 = v313408 > -230.25850929940458
    v313415 = jnp.exp(jnp.clip(v313408, -709.0, 709.0))
    v313421 = -230.25850929940458 - v313408
    v313430 = v313421 * 0.3333333333333333
    v313431 = 1.0 + v313430
    v313432 = v313421 * v313431
    v313433 = 0.5 * v313432
    v313434 = 1.0 + v313433
    v313435 = v313421 * v313434
    v313436 = 1.0 + v313435
    v313437 = jnp.divide(1e-100, jnp.where((v313436 == 0.0) | jnp.logical_not(jnp.isfinite(v313436)), 1e-300, v313436))
    v313606 = jnp.where(v313410, v313415, v313437)
    v313607 = v313605 * v313606
    v313618 = v313304 > -230.25850929940458
    v313620 = jnp.exp(jnp.clip(v313304, -709.0, 709.0))
    v313623 = -230.25850929940458 - v313304
    v313628 = v313623 * 0.3333333333333333
    v313629 = 1.0 + v313628
    v313630 = v313623 * v313629
    v313631 = 0.5 * v313630
    v313632 = 1.0 + v313631
    v313633 = v313623 * v313632
    v313634 = 1.0 + v313633
    v313635 = jnp.divide(1e-100, jnp.where((v313634 == 0.0) | jnp.logical_not(jnp.isfinite(v313634)), 1e-300, v313634))
    v313637 = jnp.where(v313618, v313620, v313635)
    v313638 = 2.0 * v313637
    v313643 = v313638 - v313607
    v313657 = jnp.where(v313310, v313607, v313643)
    v313658 = i_v3986 * v313657
    v313671 = jnp.divide(v313658, v313277)
    v313672 = 0.886226925452758 * v313671
    v313712 = v313703 * v313672
    v313264 = jnp.where(i_v309272, 0.0, v310353)
    v313246 = v312323 * v312784
    v313247 = 1.0 + v313246
    v313248 = jnp.divide(1.0, jnp.where((v313247 == 0.0) | jnp.logical_not(jnp.isfinite(v313247)), 1e-300, v313247))
    v313252 = v312323 * v312784
    v313253 = 1.0 + v313252
    v313258 = jnp.power(jnp.maximum(v313253, 0.0), i_v313235)
    v313267 = jnp.where(i_v313238, v313248, v313258)
    v313268 = v313264 * v313267
    v313269 = v313264 + v313267
    v313270 = jnp.divide(v313268, jnp.where((v313269 == 0.0) | jnp.logical_not(jnp.isfinite(v313269)), 1e-300, v313269))
    v313725 = v313712 * v313270
    v313726 = i_v157 * v313725
    v318489 = jnp.where(i_v311837, 0.0, v313726)
    v318525 = v318484 + v318489
    v314791 = i_v69 - v290180
    v315254 = v314791 * i_v3904
    v314733 = i_v69 - v290180
    v314760 = v314733 * i_v3913
    v314761 = jnp.sqrt(jnp.maximum(v314760, 1e-300))
    v314764 = i_v69 - v290180
    v314766 = v314764 * i_v3913
    v314768 = jnp.power(jnp.maximum(v314766, 0.0), i_v88)
    v315255 = jnp.where(i_v314196, v314761, v314768)
    v315256 = jnp.divide(v315254, jnp.where((v315255 == 0.0) | jnp.logical_not(jnp.isfinite(v315255)), 1e-300, v315255))
    v315257 = i_v2843 * v315256
    v315838 = v259131 * v315257
    v315839 = v315838 * v315257
    v315714 = jnp.divide(i_v315713, jnp.where((v315257 == 0.0) | jnp.logical_not(jnp.isfinite(v315257)), 1e-300, v315257))
    v315715 = v315714 < 0.0
    v315716 = -v315714
    v315717 = jnp.where(v315715, v315716, v315714)
    v315718 = v315717 < 230.25850929940458
    v315729 = jnp.exp(jnp.clip(v315714, -709.0, 709.0))
    v315735 = v315714 < -230.25850929940458
    v315741 = -230.25850929940458 - v315714
    v315750 = v315741 * 0.3333333333333333
    v315751 = 1.0 + v315750
    v315752 = v315741 * v315751
    v315753 = 0.5 * v315752
    v315754 = 1.0 + v315753
    v315755 = v315741 * v315754
    v315756 = 1.0 + v315755
    v315757 = jnp.divide(1e-100, jnp.where((v315756 == 0.0) | jnp.logical_not(jnp.isfinite(v315756)), 1e-300, v315756))
    v315762 = v315714 - 230.25850929940458
    v315769 = v315762 * 0.3333333333333333
    v315770 = 1.0 + v315769
    v315771 = v315762 * v315770
    v315772 = 0.5 * v315771
    v315773 = 1.0 + v315772
    v315774 = v315762 * v315773
    v315775 = 1.0 + v315774
    v315776 = 1e100 * v315775
    v315841 = jnp.where(v315735, v315757, v315776)
    v315840 = jnp.where(v315718, v315729, v315841)
    v315842 = v315839 * v315840
    v315843 = i_v188 * v315842
    v318528 = jnp.where(i_v314177, 0.0, v315843)
    v318551 = v318525 + v318528
    v316452 = v291797 > i_v316451
    v317367 = v291797 * i_v6575
    v317368 = v317367 < 0.0
    v317369 = -v317367
    v317370 = jnp.where(v317368, v317369, v317367)
    v317379 = -v317367
    v317380 = jnp.where(v317368, v317379, v317367)
    v317381 = v317370 * v317380
    v317390 = -v317367
    v317391 = jnp.where(v317368, v317390, v317367)
    v317392 = v317381 * v317391
    v317401 = -v317367
    v317402 = jnp.where(v317368, v317401, v317367)
    v317403 = v317392 * v317402
    v317406 = v291797 * i_v6575
    v317407 = v317406 < 0.0
    v317408 = -v317406
    v317409 = jnp.where(v317407, v317408, v317406)
    v317414 = jnp.power(jnp.maximum(v317409, 0.0), i_v225)
    v317417 = jnp.where(i_v316912, v317403, v317414)
    v317418 = 1.0 - v317417
    v317419 = jnp.divide(1.0, jnp.where((v317418 == 0.0) | jnp.logical_not(jnp.isfinite(v317418)), 1e-300, v317418))
    v317886 = v291797 + i_v317885
    v318349 = v317886 * i_v6576
    v318350 = i_v5559 + v318349
    v318553 = jnp.where(v316452, v317419, v318350)
    v318552 = jnp.where(i_v316305, 1.0, v318553)
    v318554 = v318551 * v318552
    v319265 = jnp.where(i_v8764, 0.0, v318554)
    v319266 = i_v7352 * v319265
    v319267 = v319186 + v319266
    v355148 = jnp.where(i_v12433, v264013, v319267)
    v374518 = i_v374514 * v355148
    v374519 = 0.0 + v374518
    v397969 = v374519 + 0.0
    v398033 = v397969 + 0.0
    v410429 = _mfactor * v398033
    v264998 = v259131 * i_v8965
    v264988 = v259131 + i_v264987
    v264990 = i_v8965 + v264988
    v264992 = i_v8965 - v264988
    v264994 = v264992 * v264992
    v264995 = v264994 + i_v264500
    v264996 = jnp.sqrt(jnp.maximum(v264995, 1e-300))
    v264999 = v264990 + v264996
    v265000 = jnp.divide(v264998, jnp.where((v264999 == 0.0) | jnp.logical_not(jnp.isfinite(v264999)), 1e-300, v264999))
    v265001 = 2.0 * v265000
    v267720 = v265001 * i_v2344
    v267721 = 1.0 - v267720
    v267722 = jnp.sqrt(jnp.maximum(v267721, 1e-300))
    v267726 = v265001 * i_v2344
    v267727 = 1.0 - v267726
    v267729 = jnp.power(jnp.maximum(v267727, 0.0), i_v2516)
    v268082 = jnp.where(i_v267366, v267722, v267729)
    v268083 = 1.0 - v268082
    v268084 = i_v3407 * v268083
    v268441 = v259131 - v265001
    v268442 = i_v3416 * v268441
    v268443 = v268084 + v268442
    v319556 = jnp.where(i_v267030, v268443, 0.0)
    v274617 = v259131 * i_v8965
    v274611 = v259131 + i_v274610
    v274612 = i_v8965 + v274611
    v274613 = i_v8965 - v274611
    v274614 = v274613 * v274613
    v274615 = v274614 + i_v274592
    v274616 = jnp.sqrt(jnp.maximum(v274615, 1e-300))
    v274618 = v274612 + v274616
    v274619 = jnp.divide(v274617, jnp.where((v274618 == 0.0) | jnp.logical_not(jnp.isfinite(v274618)), 1e-300, v274618))
    v274620 = 2.0 * v274619
    v282355 = jnp.where(i_v274582, v274620, 0.0)
    v282614 = v282355 * i_v2344
    v282615 = 1.0 - v282614
    v282616 = jnp.sqrt(jnp.maximum(v282615, 1e-300))
    v282621 = v282355 * i_v2344
    v282622 = 1.0 - v282621
    v282624 = jnp.power(jnp.maximum(v282622, 0.0), i_v2516)
    v282763 = jnp.where(i_v282351, v282616, v282624)
    v282764 = 1.0 - v282763
    v282765 = i_v3407 * v282764
    v282910 = v259131 - v282355
    v282911 = i_v3416 * v282910
    v282912 = v282765 + v282911
    v319569 = jnp.where(i_v8640, 0.0, v282912)
    v319551 = jnp.where(i_v12433, v319556, v319569)
    v319794 = i_v7324 * v319551
    v271027 = v265001 * i_v2346
    v271028 = 1.0 - v271027
    v271029 = jnp.sqrt(jnp.maximum(v271028, 1e-300))
    v271033 = v265001 * i_v2346
    v271034 = 1.0 - v271033
    v271036 = jnp.power(jnp.maximum(v271034, 0.0), i_v2678)
    v271317 = jnp.where(i_v270743, v271029, v271036)
    v271318 = 1.0 - v271317
    v271319 = i_v3410 * v271318
    v271606 = v259131 - v265001
    v271607 = i_v3419 * v271606
    v271608 = v271319 + v271607
    v319839 = jnp.where(i_v270479, v271608, 0.0)
    v294970 = v282355 * i_v2346
    v294971 = 1.0 - v294970
    v294972 = jnp.sqrt(jnp.maximum(v294971, 1e-300))
    v294976 = v282355 * i_v2346
    v294977 = 1.0 - v294976
    v294979 = jnp.power(jnp.maximum(v294977, 0.0), i_v2678)
    v295196 = jnp.where(i_v294679, v294972, v294979)
    v295197 = 1.0 - v295196
    v295198 = i_v3410 * v295197
    v295447 = v259131 - v282355
    v295448 = i_v3419 * v295447
    v295449 = v295198 + v295448
    v319846 = jnp.where(i_v8686, 0.0, v295449)
    v319836 = jnp.where(i_v12433, v319839, v319846)
    v319995 = i_v7338 * v319836
    v319996 = v319794 + v319995
    v274054 = v265001 * i_v2348
    v274055 = 1.0 - v274054
    v274056 = jnp.sqrt(jnp.maximum(v274055, 1e-300))
    v274060 = v265001 * i_v2348
    v274061 = 1.0 - v274060
    v274063 = jnp.power(jnp.maximum(v274061, 0.0), i_v2834)
    v274272 = jnp.where(i_v273842, v274056, v274063)
    v274273 = 1.0 - v274272
    v274274 = i_v3413 * v274273
    v274489 = v259131 - v265001
    v274490 = i_v3422 * v274489
    v274491 = v274274 + v274490
    v320039 = jnp.where(i_v273650, v274491, 0.0)
    v307121 = v282355 * i_v2348
    v307122 = 1.0 - v307121
    v307123 = jnp.sqrt(jnp.maximum(v307122, 1e-300))
    v307127 = v282355 * i_v2348
    v307128 = 1.0 - v307127
    v307130 = jnp.power(jnp.maximum(v307128, 0.0), i_v2834)
    v307425 = jnp.where(i_v306752, v307123, v307130)
    v307426 = 1.0 - v307425
    v307427 = i_v3413 * v307426
    v307754 = v259131 - v282355
    v307755 = i_v3422 * v307754
    v307756 = v307427 + v307755
    v320040 = jnp.where(i_v8764, 0.0, v307756)
    v320038 = jnp.where(i_v12433, v320039, v320040)
    v320113 = i_v7352 * v320038
    v320114 = v319996 + v320113
    v334156 = v259131 < i_v7258
    v334157 = jnp.where(i_v334145, True, v334156)
    v330460 = jnp.where(i_v12433, i_v330476, v283378)
    v334175 = v330460 * INJ1
    v334178 = v330460 * INJ1
    v334186 = v259131 - i_v7258
    v334187 = i_v334179 * v334186
    v334189 = v334187 * v334186
    v339218 = v334189 * i_v339217
    v339219 = jnp.exp(jnp.clip(v339218, -709.0, 709.0))
    v339220 = v334178 * v339219
    v339221 = jnp.where(v334157, v334175, v339220)
    v339223 = v339221 > 1e20
    v339225 = jnp.where(v339223, 1e20, v339221)
    v341789 = i_v7210 * v339225
    v341835 = v341789 - i_v7210
    v341836 = i_v341834 * v341835
    v352410 = jnp.where(i_v341839, s.v_charge_A, v341836)
    v352418 = i_v7253 * v352410
    v344576 = 0.6 - v259131
    v344578 = v344576 * v344576
    v347159 = v344578 + 4e-06
    v347160 = jnp.sqrt(jnp.maximum(v347159, 1e-300))
    v347161 = v344576 + v347160
    v347162 = 0.5 * v347161
    v347163 = v347162 < 0.0
    v347167 = jnp.where(v347163, 0.0, v347162)
    v347169 = 2.069886e-10 * v347167
    v349752 = jnp.divide(v347169, jnp.where((i_v349751 == 0.0) | jnp.logical_not(jnp.isfinite(i_v349751)), 1e-300, i_v349751))
    v349753 = jnp.sqrt(jnp.maximum(v349752, 1e-300))
    v349755 = WI - v349753
    v349757 = v349755 - 1e-07
    v349768 = v349757 * v349757
    v349769 = v349768 + i_v349764
    v349770 = jnp.sqrt(jnp.maximum(v349769, 1e-300))
    v349771 = v349757 + v349770
    v349772 = 0.5 * v349771
    v349773 = WI - v349772
    v352422 = jnp.where(i_v349775, s.v_depl_A, v349773)
    v352423 = -v352422
    v352424 = jnp.divide(v352423, i_v7253)
    v352425 = jnp.exp(jnp.clip(v352424, -709.0, 709.0))
    v352426 = i_v352421 - v352425
    v352427 = v352418 * v352426
    v352447 = i_v349807 + v352427
    v341878 = v259131 < i_v7266
    v341879 = jnp.where(i_v334145, True, v341878)
    v333675 = v259131 - i_v333674
    v333677 = v333675 > 0.0
    v330399 = v259131 - i_v7266
    v330400 = NJDV * v330399
    v330402 = v330400 + i_v277
    v330404 = NJH - v330402
    v330405 = v330404 - 0.01
    v330416 = v330405 * v330405
    v330417 = v330416 + i_v330412
    v330418 = jnp.sqrt(jnp.maximum(v330417, 1e-300))
    v330419 = v330405 + v330418
    v330420 = 0.5 * v330419
    v330421 = NJH - v330420
    v330426 = v330421 - i_v277
    v330427 = v330426 - 0.01
    v330438 = v330427 * v330427
    v330439 = v330438 + i_v330434
    v330440 = jnp.sqrt(jnp.maximum(v330439, 1e-300))
    v330444 = v330427 + v330440
    v330445 = 0.5 * v330444
    v330446 = i_v277 + v330445
    v333972 = jnp.where(i_v327819, v330446, i_v277)
    v333973 = jnp.divide(v259131, jnp.where((v333972 == 0.0) | jnp.logical_not(jnp.isfinite(v333972)), 1e-300, v333972))
    v333977 = jnp.divide(i_v333674, jnp.where((v333972 == 0.0) | jnp.logical_not(jnp.isfinite(v333972)), 1e-300, v333972))
    v333978 = v333973 - v333977
    v333981 = v333972 - i_v277
    v333982 = i_v7266 * v333981
    v333984 = jnp.divide(v333982, jnp.where((i_v333983 == 0.0) | jnp.logical_not(jnp.isfinite(i_v333983)), 1e-300, i_v333983))
    v333985 = v333978 + v333984
    v333986 = i_v596 * v333985
    v333987 = v333986 < 0.0
    v333988 = -v333986
    v333989 = jnp.where(v333987, v333988, v333986)
    v333990 = v333989 < 230.25850929940458
    v334025 = jnp.exp(jnp.clip(v333986, -709.0, 709.0))
    v334044 = v333986 < -230.25850929940458
    v334062 = -230.25850929940458 - v333986
    v334087 = v334062 * 0.3333333333333333
    v334088 = 1.0 + v334087
    v334089 = v334062 * v334088
    v334090 = 0.5 * v334089
    v334091 = 1.0 + v334090
    v334092 = v334062 * v334091
    v334093 = 1.0 + v334092
    v334094 = jnp.divide(1e-100, jnp.where((v334093 == 0.0) | jnp.logical_not(jnp.isfinite(v334093)), 1e-300, v334093))
    v334111 = v333986 - 230.25850929940458
    v334134 = v334111 * 0.3333333333333333
    v334135 = 1.0 + v334134
    v334136 = v334111 * v334135
    v334137 = 0.5 * v334136
    v334138 = 1.0 + v334137
    v334139 = v334111 * v334138
    v334140 = 1.0 + v334139
    v334141 = 1e100 * v334140
    v341893 = jnp.where(v334044, v334094, v334141)
    v341892 = jnp.where(v333990, v334025, v341893)
    v341891 = jnp.where(v333677, v341892, 1.0)
    v341899 = v341891 * INJ1
    v341902 = v341891 * INJ1
    v341910 = v259131 - i_v7266
    v341911 = i_v341903 * v341910
    v341913 = v341911 * v341910
    v341942 = v341913 * i_v341941
    v341943 = jnp.exp(jnp.clip(v341942, -709.0, 709.0))
    v341944 = v341902 * v341943
    v341945 = jnp.where(v341879, v341899, v341944)
    v341946 = v341945 > 1e20
    v341948 = jnp.where(v341946, 1e20, v341945)
    v341961 = i_v7210 * v341948
    v341976 = v341961 - i_v7210
    v341977 = i_v341834 * v341976
    v352435 = jnp.where(i_v341839, s.v_charge_K, v341977)
    v352439 = i_v7253 * v352435
    v352440 = WI - v352422
    v352441 = -v352440
    v352442 = jnp.divide(v352441, i_v7253)
    v352443 = jnp.exp(jnp.clip(v352442, -709.0, 709.0))
    v352444 = v352443 - 1.0
    v352445 = v352439 * v352444
    v352448 = v352447 + v352445
    v352449 = -v352448
    v352518 = v320114 + v352449
    v397943 = jnp.where(i_v325285, v352518, v320114)
    v397952 = i_v374514 * v397943
    v264015 = v259746 + v264012
    v294250 = v294124 + v294184
    v294251 = v294250 + v294223
    v294252 = v294251 * v294247
    v319272 = jnp.where(i_v8640, 0.0, v294252)
    v319425 = i_v7324 * v319272
    v306170 = v306044 + v306104
    v306171 = v306170 + v306143
    v306172 = v306171 * v306167
    v319428 = jnp.where(i_v8686, 0.0, v306172)
    v319505 = i_v7338 * v319428
    v319506 = v319425 + v319505
    v318555 = v318429 + v318489
    v318556 = v318555 + v318528
    v318557 = v318556 * v318552
    v319507 = jnp.where(i_v8764, 0.0, v318557)
    v319508 = i_v7352 * v319507
    v319509 = v319506 + v319508
    v355180 = jnp.where(i_v12433, v264015, v319509)
    v357897 = v355148 - v355180
    v398031 = i_v397993 * v357897
    v408495 = v397952 + v398031
    v410430 = _mfactor * v408495
    v374598 = i_v7358 * (s.v_AIK - signals.K)
    v374600 = v374598 * i_v366767
    v374602 = 0.0 + v374600
    v399630 = jnp.where(i_v374520, v374602, 0.0)
    v408506 = -v399630
    v410432 = _mfactor * v408506
    v408501 = -v398033
    v408505 = v408501 + v399630
    v410434 = _mfactor * v408505
    v408502 = -v408495
    v410436 = _mfactor * v408502
    v341844 = NQS * s.v_charge_A
    v341841 = v341836 * NQS
    v341845 = v341844 - v341841
    v341846 = jnp.divide(v341845, jnp.where((NQS == 0.0) | jnp.logical_not(jnp.isfinite(NQS)), 1e-300, NQS))
    v377028 = jnp.where(i_v341839, v341846, 0.0)
    v374618 = jnp.where(i_v325285, v377028, 0.0)
    v382369 = v374618 + 0.0
    v402506 = jnp.where(i_v374608, v382369, 0.0)
    v410437 = _mfactor * v402506
    v382324 = jnp.where(i_v341839, v341844, 0.0)
    v379914 = jnp.where(i_v325285, v382324, 0.0)
    v408397 = jnp.where(i_v374608, v379914, 0.0)
    v410438 = _mfactor * v408397
    v341984 = NQS * s.v_charge_K
    v341981 = v341977 * NQS
    v341985 = v341984 - v341981
    v341986 = jnp.divide(v341985, jnp.where((NQS == 0.0) | jnp.logical_not(jnp.isfinite(NQS)), 1e-300, NQS))
    v384781 = jnp.where(i_v341839, v341986, 0.0)
    v382379 = jnp.where(i_v325285, v384781, 0.0)
    v390130 = v382379 + 0.0
    v405380 = jnp.where(i_v374608, v390130, 0.0)
    v410439 = _mfactor * v405380
    v390077 = jnp.where(i_v341839, v341984, 0.0)
    v387675 = jnp.where(i_v325285, v390077, 0.0)
    v408398 = jnp.where(i_v374608, v387675, 0.0)
    v410440 = _mfactor * v408398
    v349780 = DEPNQS * s.v_depl_A
    v349777 = v349773 * DEPNQS
    v349781 = v349780 - v349777
    v349782 = jnp.divide(v349781, jnp.where((DEPNQS == 0.0) | jnp.logical_not(jnp.isfinite(DEPNQS)), 1e-300, DEPNQS))
    v392544 = jnp.where(i_v349775, v349782, 0.0)
    v390148 = jnp.where(i_v325285, v392544, 0.0)
    v395074 = v390148 + 0.0
    v408244 = jnp.where(i_v390134, v395074, 0.0)
    v410441 = _mfactor * v408244
    v395013 = jnp.where(i_v349775, v349780, 0.0)
    v392617 = jnp.where(i_v325285, v395013, 0.0)
    v408399 = jnp.where(i_v390134, v392617, 0.0)
    v410442 = _mfactor * v408399
    return (
        {
            "A": v410429,
            "K": jnp.where(i_v374520, v410432, -v410429),
            "v_AIK": jnp.where(i_v374520, v410434, s.v_AIK - signals.K),
            "v_charge_A": jnp.where(i_v374608, v410437, s.v_charge_A),
            "v_charge_K": jnp.where(i_v374608, v410439, s.v_charge_K),
            "v_depl_A": jnp.where(i_v390134, v410441, s.v_depl_A),
        },
        {
            "A": v410430,
            "v_AIK": jnp.where(i_v374520, v410436, 0.0),
            "v_charge_A": v410438,
            "v_charge_K": v410440,
            "v_depl_A": v410442,
            "K": jnp.where(i_v374520, 0.0, -v410430),
        },
    )


@DIODE_CMC.setup
def _DIODE_CMC_register_setup(*_a, **_kw):
    return _DIODE_CMC_setup(*_a, **_kw)
