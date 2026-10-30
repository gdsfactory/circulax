"""Native JAX JUNCAP200 200.6.2 junction current and charge equations.

Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,
code/psp103/vacode/juncap200.va and its includes. Original equations copyright
NXP Semiconductors, CEA-Leti, Delft University of Technology and Arizona State
University; see retained source notices in tests/data/va_models/juncap.

Regenerate using tests/data/va_models/juncap/regenerate.py. Both full and
express branches remain selectable; LEVEL is fixed at 200. Noise and operating
point metadata are excluded. No OSDI or bosdi runtime is required.
"""

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, CPY001, E501, FBT003, PLR0915, PLR0917
# Generated reference equations retain their original parameter signatures.

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _JUNCAP200_setup(
    TYPE: int = 1,
    DTA: float = 0.0,
    AB: float = 1.0e-12,
    LS: float = 1.0e-6,
    LG: float = 1.0e-6,
    MULT: float = 1.0,
    TRISE: float = 0.0,
    IMAX: float = 1000.0,
    TRJ: float = 21.0,
    FREV: float = 1.0e3,
    IFACTOR: float = 1.0,
    CFACTOR: float = 1.0,
    CJORBOT: float = 1.0e-3,
    CJORSTI: float = 1.0e-9,
    CJORGAT: float = 1.0e-9,
    VBIRBOT: float = 1.0,
    VBIRSTI: float = 1.0,
    VBIRGAT: float = 1.0,
    PBOT: float = 0.5,
    PSTI: float = 0.5,
    PGAT: float = 0.5,
    PHIGBOT: float = 1.16,
    PHIGSTI: float = 1.16,
    PHIGGAT: float = 1.16,
    IDSATRBOT: float = 1.0e-12,
    IDSATRSTI: float = 1.0e-18,
    IDSATRGAT: float = 1.0e-18,
    CSRHBOT: float = 1.0e2,
    CSRHSTI: float = 1.0e-4,
    CSRHGAT: float = 1.0e-4,
    XJUNSTI: float = 1.0e-7,
    XJUNGAT: float = 1.0e-7,
    CTATBOT: float = 1.0e2,
    CTATSTI: float = 1.0e-4,
    CTATGAT: float = 1.0e-4,
    MEFFTATBOT: float = 0.25,
    MEFFTATSTI: float = 0.25,
    MEFFTATGAT: float = 0.25,
    CBBTBOT: float = 1.0e-12,
    CBBTSTI: float = 1.0e-18,
    CBBTGAT: float = 1.0e-18,
    FBBTRBOT: float = 1.0e9,
    FBBTRSTI: float = 1.0e9,
    FBBTRGAT: float = 1.0e9,
    STFBBTBOT: float = -1.0e-3,
    STFBBTSTI: float = -1.0e-3,
    STFBBTGAT: float = -1.0e-3,
    VBRBOT: float = 10.0,
    VBRSTI: float = 10.0,
    VBRGAT: float = 10.0,
    PBRBOT: float = 4.0,
    PBRSTI: float = 4.0,
    PBRGAT: float = 4.0,
    SWJUNEXP: float = 0.0,
    VJUNREF: float = 2.5,
    FJUNQ: float = 0.03,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    i_v144237 = TYPE * (jnp.where((MULT > 0.0), MULT, 0.0))
    i_v1435 = (jnp.where((SWJUNEXP > 0.5), 1.0, 0.0)) == 1.0
    i_v32 = (_temperature + DTA) + TRISE
    i_v37 = jnp.where((i_v32 > 23.149999999999977), i_v32, 23.149999999999977)
    i_v26 = 273.15 + TRJ
    i_v42 = jnp.divide(i_v37, jnp.where((i_v26 == 0.0) | ~jnp.isfinite(i_v26), 1e-300, i_v26))
    i_v86 = jnp.power(jnp.maximum(i_v42, 0.0), 1.5)
    i_v61 = 1108.0 + i_v26
    i_v62 = jnp.divide((-((0.000702 * i_v26) * i_v26)), jnp.where((i_v61 == 0.0) | ~jnp.isfinite(i_v61), 1e-300, i_v61))
    i_v48 = 8.61726105451295e-05 * i_v26
    i_v50 = jnp.divide(1.0, jnp.where((i_v48 == 0.0) | ~jnp.isfinite(i_v48), 1e-300, i_v48))
    i_v76 = 1108.0 + i_v37
    i_v77 = jnp.divide((-((0.000702 * i_v37) * i_v37)), jnp.where((i_v76 == 0.0) | ~jnp.isfinite(i_v76), 1e-300, i_v76))
    i_v79 = PHIGBOT + i_v77
    i_v52 = 8.61726105451295e-05 * i_v37
    i_v54 = jnp.divide(1.0, jnp.where((i_v52 == 0.0) | ~jnp.isfinite(i_v52), 1e-300, i_v52))
    i_v92 = i_v86 * (jnp.exp(jnp.clip((0.5 * (((PHIGBOT + i_v62) * i_v50) - (i_v79 * i_v54))), -709.0, 709.0)))
    i_v112 = (IDSATRBOT * i_v92) * i_v92
    i_v465 = jnp.where((AB > 0.0), AB, 0.0)
    i_v519 = i_v112 * i_v465
    i_v81 = PHIGSTI + i_v77
    i_v100 = i_v86 * (jnp.exp(jnp.clip((0.5 * (((PHIGSTI + i_v62) * i_v50) - (i_v81 * i_v54))), -709.0, 709.0)))
    i_v116 = (IDSATRSTI * i_v100) * i_v100
    i_v469 = jnp.where((LS > 0.0), LS, 0.0)
    i_v598 = i_v116 * i_v469
    i_v83 = PHIGGAT + i_v77
    i_v108 = i_v86 * (jnp.exp(jnp.clip((0.5 * (((PHIGGAT + i_v62) * i_v50) - (i_v83 * i_v54))), -709.0, 709.0)))
    i_v120 = (IDSATRGAT * i_v108) * i_v108
    i_v473 = jnp.where((LG > 0.0), LG, 0.0)
    i_v656 = i_v120 * i_v473
    i_v93336 = (i_v519 + i_v598) + i_v656
    i_v112095 = jnp.where(i_v1435, i_v93336, 0.0)
    i_v977 = i_v465 == 0.0
    i_v1012 = i_v469 == 0.0
    i_v1060 = i_v473 == 0.0
    i_v106001 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v106002 = 0.5 * IMAX
    i_v107259 = jnp.where(
        i_v106001,
        (
            jnp.log(
                jnp.maximum(
                    (
                        jnp.divide(
                            i_v106002,
                            jnp.where(((i_v93336 + 1e-21) == 0.0) | ~jnp.isfinite((i_v93336 + 1e-21)), 1e-300, (i_v93336 + 1e-21)),
                        )
                    ),
                    1e-300,
                )
            )
        ),
        0.0,
    )
    i_v108409 = jnp.where((i_v107259 < 230.25850929940458), i_v107259, 230.25850929940458)
    (
        i_v112062,
        i_v112074,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1435, _t, _f),
        (
            i_v108409,
            (jnp.exp(jnp.clip(i_v108409, -709.0, 709.0))),
        ),
        (
            0.0,
            0.0,
        ),
    )
    i_v95080 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v54707 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v673 = jnp.where(
        (i_v519 > 0.0),
        (
            i_v52
            * (
                jnp.log(
                    jnp.maximum(
                        ((jnp.divide(IMAX, jnp.where((i_v519 == 0.0) | ~jnp.isfinite(i_v519), 1e-300, i_v519))) + 1.0), 1e-300
                    )
                )
            )
        ),
        100000000.0,
    )
    i_v678 = jnp.where(
        (i_v598 > 0.0),
        (
            i_v52
            * (
                jnp.log(
                    jnp.maximum(
                        ((jnp.divide(IMAX, jnp.where((i_v598 == 0.0) | ~jnp.isfinite(i_v598), 1e-300, i_v598))) + 1.0), 1e-300
                    )
                )
            )
        ),
        100000000.0,
    )
    i_v681 = jnp.where((i_v673 < i_v678), i_v673, i_v678)
    i_v684 = jnp.where(
        (i_v656 > 0.0),
        (
            i_v52
            * (
                jnp.log(
                    jnp.maximum(
                        ((jnp.divide(IMAX, jnp.where((i_v656 == 0.0) | ~jnp.isfinite(i_v656), 1e-300, i_v656))) + 1.0), 1e-300
                    )
                )
            )
        ),
        100000000.0,
    )
    i_v687 = jnp.where((i_v681 < i_v684), i_v681, i_v684)
    i_v56006 = i_v687 > 0.1
    i_v56214 = 0.5 * (0.1 * i_v54)
    i_v56215 = i_v56214 < 0.0
    i_v56240 = -230.25850929940458 - i_v56214
    i_v56261 = i_v56214 - 230.25850929940458
    i_v56276 = jnp.where(
        ((jnp.where(i_v56215, (-i_v56214), i_v56214)) < 230.25850929940458),
        (jnp.exp(jnp.clip(i_v56214, -709.0, 709.0))),
        (
            jnp.where(
                i_v56215,
                (
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            ((1.0 + (i_v56240 * (1.0 + (0.5 * (i_v56240 * (1.0 + (i_v56240 * 0.3333333333333333))))))) == 0.0)
                            | ~jnp.isfinite(
                                1.0 + (i_v56240 * (1.0 + (0.5 * (i_v56240 * (1.0 + (i_v56240 * 0.3333333333333333))))))
                            ),
                            1e-300,
                            (1.0 + (i_v56240 * (1.0 + (0.5 * (i_v56240 * (1.0 + (i_v56240 * 0.3333333333333333))))))),
                        ),
                    )
                ),
                (1e100 * (1.0 + (i_v56261 * (1.0 + (0.5 * (i_v56261 * (1.0 + (i_v56261 * 0.3333333333333333)))))))),
            )
        ),
    )
    i_v725 = i_v687 * i_v54
    i_v726 = i_v725 < 0.0
    i_v750 = -230.25850929940458 - i_v725
    i_v770 = i_v725 - 230.25850929940458
    i_v1852 = jnp.where(
        ((jnp.where(i_v726, (-i_v725), i_v725)) < 230.25850929940458),
        (jnp.exp(jnp.clip(i_v725, -709.0, 709.0))),
        (
            jnp.where(
                i_v726,
                (
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            ((1.0 + (i_v750 * (1.0 + (0.5 * (i_v750 * (1.0 + (i_v750 * 0.3333333333333333))))))) == 0.0)
                            | ~jnp.isfinite(1.0 + (i_v750 * (1.0 + (0.5 * (i_v750 * (1.0 + (i_v750 * 0.3333333333333333))))))),
                            1e-300,
                            (1.0 + (i_v750 * (1.0 + (0.5 * (i_v750 * (1.0 + (i_v750 * 0.3333333333333333))))))),
                        ),
                    )
                ),
                (1e100 * (1.0 + (i_v770 * (1.0 + (0.5 * (i_v770 * (1.0 + (i_v770 * 0.3333333333333333)))))))),
            )
        ),
    )
    i_v56498 = (1.0 + ((0.1 - i_v687) * i_v54)) * i_v1852
    i_v35892 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v1495 = -0.8 * VJUNREF
    i_v36978 = i_v1495 < i_v687
    i_v37186 = 0.5 * (i_v1495 * i_v54)
    i_v37187 = i_v37186 < 0.0
    i_v37212 = -230.25850929940458 - i_v37186
    i_v37233 = i_v37186 - 230.25850929940458
    i_v37248 = jnp.where(
        ((jnp.where(i_v37187, (-i_v37186), i_v37186)) < 230.25850929940458),
        (jnp.exp(jnp.clip(i_v37186, -709.0, 709.0))),
        (
            jnp.where(
                i_v37187,
                (
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            ((1.0 + (i_v37212 * (1.0 + (0.5 * (i_v37212 * (1.0 + (i_v37212 * 0.3333333333333333))))))) == 0.0)
                            | ~jnp.isfinite(
                                1.0 + (i_v37212 * (1.0 + (0.5 * (i_v37212 * (1.0 + (i_v37212 * 0.3333333333333333))))))
                            ),
                            1e-300,
                            (1.0 + (i_v37212 * (1.0 + (0.5 * (i_v37212 * (1.0 + (i_v37212 * 0.3333333333333333))))))),
                        ),
                    )
                ),
                (1e100 * (1.0 + (i_v37233 * (1.0 + (0.5 * (i_v37233 * (1.0 + (i_v37233 * 0.3333333333333333)))))))),
            )
        ),
    )
    i_v37470 = (1.0 + ((i_v1495 - i_v687) * i_v54)) * i_v1852
    i_v17290 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v1492 = -0.65 * VJUNREF
    i_v18163 = i_v1492 < i_v687
    i_v18371 = 0.5 * (i_v1492 * i_v54)
    i_v18372 = i_v18371 < 0.0
    i_v18397 = -230.25850929940458 - i_v18371
    i_v18418 = i_v18371 - 230.25850929940458
    i_v18433 = jnp.where(
        ((jnp.where(i_v18372, (-i_v18371), i_v18371)) < 230.25850929940458),
        (jnp.exp(jnp.clip(i_v18371, -709.0, 709.0))),
        (
            jnp.where(
                i_v18372,
                (
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            ((1.0 + (i_v18397 * (1.0 + (0.5 * (i_v18397 * (1.0 + (i_v18397 * 0.3333333333333333))))))) == 0.0)
                            | ~jnp.isfinite(
                                1.0 + (i_v18397 * (1.0 + (0.5 * (i_v18397 * (1.0 + (i_v18397 * 0.3333333333333333))))))
                            ),
                            1e-300,
                            (1.0 + (i_v18397 * (1.0 + (0.5 * (i_v18397 * (1.0 + (i_v18397 * 0.3333333333333333))))))),
                        ),
                    )
                ),
                (1e100 * (1.0 + (i_v18418 * (1.0 + (0.5 * (i_v18418 * (1.0 + (i_v18418 * 0.3333333333333333)))))))),
            )
        ),
    )
    i_v18655 = (1.0 + ((i_v1492 - i_v687) * i_v54)) * i_v1852
    i_v1603 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v1489 = -0.4 * VJUNREF
    i_v1706 = i_v1489 < i_v687
    i_v1756 = 0.5 * (i_v1489 * i_v54)
    i_v1757 = i_v1756 < 0.0
    i_v1782 = -230.25850929940458 - i_v1756
    i_v1803 = i_v1756 - 230.25850929940458
    i_v1818 = jnp.where(
        ((jnp.where(i_v1757, (-i_v1756), i_v1756)) < 230.25850929940458),
        (jnp.exp(jnp.clip(i_v1756, -709.0, 709.0))),
        (
            jnp.where(
                i_v1757,
                (
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            ((1.0 + (i_v1782 * (1.0 + (0.5 * (i_v1782 * (1.0 + (i_v1782 * 0.3333333333333333))))))) == 0.0)
                            | ~jnp.isfinite(1.0 + (i_v1782 * (1.0 + (0.5 * (i_v1782 * (1.0 + (i_v1782 * 0.3333333333333333))))))),
                            1e-300,
                            (1.0 + (i_v1782 * (1.0 + (0.5 * (i_v1782 * (1.0 + (i_v1782 * 0.3333333333333333))))))),
                        ),
                    )
                ),
                (1e100 * (1.0 + (i_v1803 * (1.0 + (0.5 * (i_v1803 * (1.0 + (i_v1803 * 0.3333333333333333)))))))),
            )
        ),
    )
    i_v1874 = (1.0 + ((i_v1489 - i_v687) * i_v54)) * i_v1852
    i_v2824 = jnp.where(i_v1603, ((jnp.where(i_v1706, (i_v1818 * i_v1818), i_v1874)) - 1.0), 0.0)
    i_v20760 = jnp.where(i_v17290, ((jnp.where(i_v18163, (i_v18433 * i_v18433), i_v18655)) - 1.0), i_v2824)
    i_v39575 = jnp.where(i_v35892, ((jnp.where(i_v36978, (i_v37248 * i_v37248), i_v37470)) - 1.0), i_v20760)
    i_v58603 = jnp.where(i_v54707, ((jnp.where(i_v56006, (i_v56276 * i_v56276), i_v56498)) - 1.0), i_v39575)
    i_v58677 = jnp.where((CSRHBOT == 0.0), (CTATBOT == 0.0), False)
    i_v56502 = jnp.where(i_v56006, i_v56276, (jnp.sqrt(jnp.maximum(i_v56498, 1e-300))))
    i_v37474 = jnp.where(i_v36978, i_v37248, (jnp.sqrt(jnp.maximum(i_v37470, 1e-300))))
    i_v18659 = jnp.where(i_v18163, i_v18433, (jnp.sqrt(jnp.maximum(i_v18655, 1e-300))))
    i_v1878 = jnp.where(i_v1706, i_v1818, (jnp.sqrt(jnp.maximum(i_v1874, 1e-300))))
    i_v3358 = jnp.where(i_v1603, i_v1878, 0.0)
    i_v21792 = jnp.where(i_v17290, i_v18659, i_v3358)
    i_v40607 = jnp.where(i_v35892, i_v37474, i_v21792)
    i_v59635 = jnp.where(i_v54707, i_v56502, i_v40607)
    i_v228 = jnp.divide(1.0447941624768001e-10, jnp.where((CJORBOT == 0.0) | ~jnp.isfinite(CJORBOT), 1e-300, CJORBOT))
    i_v58983 = PBOT == 0.5
    i_v124 = 2.0 * i_v52
    i_v127 = (VBIRBOT * i_v42) - (i_v124 * (jnp.log(jnp.maximum(i_v92, 1e-300))))
    i_v151 = i_v127 + (
        i_v52 * (jnp.log(jnp.maximum((1.0 + (jnp.exp(jnp.clip(((0.05 - i_v127) * i_v54), -709.0, 709.0)))), 1e-300)))
    )
    i_v134 = (VBIRSTI * i_v42) - (i_v124 * (jnp.log(jnp.maximum(i_v100, 1e-300))))
    i_v160 = i_v134 + (
        i_v52 * (jnp.log(jnp.maximum((1.0 + (jnp.exp(jnp.clip(((0.05 - i_v134) * i_v54), -709.0, 709.0)))), 1e-300)))
    )
    i_v141 = (VBIRGAT * i_v42) - (i_v124 * (jnp.log(jnp.maximum(i_v108, 1e-300))))
    i_v169 = i_v141 + (
        i_v52 * (jnp.log(jnp.maximum((1.0 + (jnp.exp(jnp.clip(((0.05 - i_v141) * i_v54), -709.0, 709.0)))), 1e-300)))
    )
    i_v1088 = jnp.where(i_v977, (i_v160 + i_v169), i_v151)
    i_v1103 = jnp.where(i_v1012, (i_v151 + i_v169), i_v160)
    i_v1119 = jnp.where((i_v1088 < i_v1103), i_v1088, i_v1103)
    i_v1122 = jnp.where(i_v1060, (i_v151 + i_v160), i_v169)
    i_v1141 = jnp.where((i_v1119 < i_v1122), i_v1119, i_v1122)
    i_v56503 = jnp.divide(1.0, jnp.where((i_v56502 == 0.0) | ~jnp.isfinite(i_v56502), 1e-300, i_v56502))
    i_v56728 = 2.0 * (
        i_v52
        * (
            jnp.log(
                jnp.maximum(((2.0 + i_v56503) + (jnp.sqrt(jnp.maximum(((i_v56503 + 1.0) * (i_v56503 + 3.0)), 1e-300)))), 1e-300)
            )
        )
    )
    i_v56959 = i_v1141 - i_v56728
    i_v37475 = jnp.divide(1.0, jnp.where((i_v37474 == 0.0) | ~jnp.isfinite(i_v37474), 1e-300, i_v37474))
    i_v37930 = jnp.where(
        (i_v1495 > 0.0),
        (
            2.0
            * (
                i_v52
                * (
                    jnp.log(
                        jnp.maximum(
                            ((2.0 + i_v37475) + (jnp.sqrt(jnp.maximum(((i_v37475 + 1.0) * (i_v37475 + 3.0)), 1e-300)))), 1e-300
                        )
                    )
                )
            )
        ),
        (
            (-i_v1495)
            + (
                2.0
                * (
                    i_v52
                    * (
                        jnp.log(
                            jnp.maximum(
                                (
                                    ((2.0 * i_v37474) + 1.0)
                                    + (jnp.sqrt(jnp.maximum(((1.0 + i_v37474) * (1.0 + (3.0 * i_v37474))), 1e-300)))
                                ),
                                1e-300,
                            )
                        )
                    )
                )
            )
        ),
    )
    i_v37931 = i_v1141 - i_v37930
    i_v18660 = jnp.divide(1.0, jnp.where((i_v18659 == 0.0) | ~jnp.isfinite(i_v18659), 1e-300, i_v18659))
    i_v19115 = jnp.where(
        (i_v1492 > 0.0),
        (
            2.0
            * (
                i_v52
                * (
                    jnp.log(
                        jnp.maximum(
                            ((2.0 + i_v18660) + (jnp.sqrt(jnp.maximum(((i_v18660 + 1.0) * (i_v18660 + 3.0)), 1e-300)))), 1e-300
                        )
                    )
                )
            )
        ),
        (
            (-i_v1492)
            + (
                2.0
                * (
                    i_v52
                    * (
                        jnp.log(
                            jnp.maximum(
                                (
                                    ((2.0 * i_v18659) + 1.0)
                                    + (jnp.sqrt(jnp.maximum(((1.0 + i_v18659) * (1.0 + (3.0 * i_v18659))), 1e-300)))
                                ),
                                1e-300,
                            )
                        )
                    )
                )
            )
        ),
    )
    i_v19116 = i_v1141 - i_v19115
    i_v1879 = jnp.divide(1.0, jnp.where((i_v1878 == 0.0) | ~jnp.isfinite(i_v1878), 1e-300, i_v1878))
    i_v2013 = jnp.where(
        (i_v1489 > 0.0),
        (
            2.0
            * (
                i_v52
                * (
                    jnp.log(
                        jnp.maximum(
                            ((2.0 + i_v1879) + (jnp.sqrt(jnp.maximum(((i_v1879 + 1.0) * (i_v1879 + 3.0)), 1e-300)))), 1e-300
                        )
                    )
                )
            )
        ),
        (
            (-i_v1489)
            + (
                2.0
                * (
                    i_v52
                    * (
                        jnp.log(
                            jnp.maximum(
                                (
                                    ((2.0 * i_v1878) + 1.0)
                                    + (jnp.sqrt(jnp.maximum(((1.0 + i_v1878) * (1.0 + (3.0 * i_v1878))), 1e-300)))
                                ),
                                1e-300,
                            )
                        )
                    )
                )
            )
        ),
    )
    i_v2014 = i_v1141 - i_v2013
    i_v2904 = jnp.where(
        i_v1603,
        (
            0.5
            * (
                (i_v1489 + i_v2014)
                - (jnp.sqrt(jnp.maximum((((i_v1489 - i_v2014) * (i_v1489 - i_v2014)) + ((4.0 * i_v52) * i_v52)), 1e-300)))
            )
        ),
        0.0,
    )
    i_v21054 = jnp.where(
        i_v17290,
        (
            0.5
            * (
                (i_v1492 + i_v19116)
                - (jnp.sqrt(jnp.maximum((((i_v1492 - i_v19116) * (i_v1492 - i_v19116)) + ((4.0 * i_v52) * i_v52)), 1e-300)))
            )
        ),
        i_v2904,
    )
    i_v39869 = jnp.where(
        i_v35892,
        (
            0.5
            * (
                (i_v1495 + i_v37931)
                - (jnp.sqrt(jnp.maximum((((i_v1495 - i_v37931) * (i_v1495 - i_v37931)) + ((4.0 * i_v52) * i_v52)), 1e-300)))
            )
        ),
        i_v21054,
    )
    i_v58897 = jnp.where(
        i_v54707,
        (
            0.5
            * (
                (0.1 + i_v56959)
                - (jnp.sqrt(jnp.maximum((((0.1 - i_v56959) * (0.1 - i_v56959)) + ((4.0 * i_v52) * i_v52)), 1e-300)))
            )
        ),
        i_v39869,
    )
    i_v58962 = i_v151 - i_v58897
    i_v244 = jnp.divide(1.0, jnp.where((VBIRBOT == 0.0) | ~jnp.isfinite(VBIRBOT), 1e-300, VBIRBOT))
    i_v59411 = i_v228 * (
        jnp.where(
            i_v58983, (jnp.sqrt(jnp.maximum((i_v58962 * i_v244), 1e-300))), (jnp.power(jnp.maximum((i_v58962 * i_v244), 0.0), PBOT))
        )
    )
    i_v59700 = i_v92 * ((i_v59635 - 1.0) * i_v59411)
    i_v58969 = jnp.where(i_v54707, i_v56728, 0.0)
    i_v58981 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v58969, jnp.where((i_v58962 == 0.0) | ~jnp.isfinite(i_v58962), 1e-300, i_v58962)))), 1e-300
        )
    )
    i_v58982 = 1.0 - i_v58981
    i_v58997 = i_v58982 + (
        jnp.where(
            i_v58983,
            0.0,
            (
                ((jnp.divide(((i_v58982 * i_v58982) * (jnp.log(jnp.maximum(i_v58982, 1e-300)))), i_v58981)) + i_v58982)
                * (1.0 - (2.0 * PBOT))
            ),
        )
    )
    i_v2841 = jnp.where((CSRHBOT == 0.0), (CTATBOT == 0.0), False)
    i_v2935 = PBOT == 0.5
    i_v2914 = i_v151 - i_v2904
    i_v3218 = i_v228 * (
        jnp.where(
            i_v2935, (jnp.sqrt(jnp.maximum((i_v2914 * i_v244), 1e-300))), (jnp.power(jnp.maximum((i_v2914 * i_v244), 0.0), PBOT))
        )
    )
    i_v3374 = i_v92 * ((i_v3358 - 1.0) * i_v3218)
    i_v4497 = jnp.where(i_v2841, 0.0, i_v3374)
    i_v9092 = jnp.where(i_v977, 0.0, i_v4497)
    i_v7024 = jnp.where((CSRHSTI == 0.0), (CTATSTI == 0.0), False)
    i_v232 = jnp.divide((XJUNSTI * 1.0447941624768001e-10), jnp.where((CJORSTI == 0.0) | ~jnp.isfinite(CJORSTI), 1e-300, CJORSTI))
    i_v7275 = PSTI == 0.5
    i_v7207 = i_v160 - i_v2904
    i_v246 = jnp.divide(1.0, jnp.where((VBIRSTI == 0.0) | ~jnp.isfinite(VBIRSTI), 1e-300, VBIRSTI))
    i_v7684 = i_v232 * (
        jnp.where(
            i_v7275, (jnp.sqrt(jnp.maximum((i_v7207 * i_v246), 1e-300))), (jnp.power(jnp.maximum((i_v7207 * i_v246), 0.0), PSTI))
        )
    )
    i_v7942 = i_v100 * ((i_v3358 - 1.0) * i_v7684)
    i_v9085 = jnp.where(i_v7024, i_v9092, i_v7942)
    i_v14749 = jnp.where(i_v1012, i_v9092, i_v9085)
    i_v12234 = jnp.where((CSRHGAT == 0.0), (CTATGAT == 0.0), False)
    i_v236 = jnp.divide((XJUNGAT * 1.0447941624768001e-10), jnp.where((CJORGAT == 0.0) | ~jnp.isfinite(CJORGAT), 1e-300, CJORGAT))
    i_v12554 = PGAT == 0.5
    i_v12486 = i_v169 - i_v2904
    i_v248 = jnp.divide(1.0, jnp.where((VBIRGAT == 0.0) | ~jnp.isfinite(VBIRGAT), 1e-300, VBIRGAT))
    i_v13089 = i_v236 * (
        jnp.where(
            i_v12554, (jnp.sqrt(jnp.maximum((i_v12486 * i_v248), 1e-300))), (jnp.power(jnp.maximum((i_v12486 * i_v248), 0.0), PGAT))
        )
    )
    i_v13410 = i_v108 * ((i_v3358 - 1.0) * i_v13089)
    i_v14742 = jnp.where(i_v12234, i_v14749, i_v13410)
    i_v23193 = jnp.where(i_v1060, i_v14749, i_v14742)
    i_v20834 = jnp.where((CSRHBOT == 0.0), (CTATBOT == 0.0), False)
    i_v21140 = PBOT == 0.5
    i_v21119 = i_v151 - i_v21054
    i_v21568 = i_v228 * (
        jnp.where(
            i_v21140, (jnp.sqrt(jnp.maximum((i_v21119 * i_v244), 1e-300))), (jnp.power(jnp.maximum((i_v21119 * i_v244), 0.0), PBOT))
        )
    )
    i_v21857 = i_v92 * ((i_v21792 - 1.0) * i_v21568)
    i_v23180 = jnp.where(i_v20834, i_v23193, i_v21857)
    i_v28412 = jnp.where(i_v977, i_v23193, i_v23180)
    i_v26174 = jnp.where((CSRHSTI == 0.0), (CTATSTI == 0.0), False)
    i_v26519 = PSTI == 0.5
    i_v26451 = i_v160 - i_v21054
    i_v26947 = i_v232 * (
        jnp.where(
            i_v26519, (jnp.sqrt(jnp.maximum((i_v26451 * i_v246), 1e-300))), (jnp.power(jnp.maximum((i_v26451 * i_v246), 0.0), PSTI))
        )
    )
    i_v27226 = i_v100 * ((i_v21792 - 1.0) * i_v26947)
    i_v28405 = jnp.where(i_v26174, i_v28412, i_v27226)
    i_v33658 = jnp.where(i_v1012, i_v28412, i_v28405)
    i_v31420 = jnp.where((CSRHGAT == 0.0), (CTATGAT == 0.0), False)
    i_v31765 = PGAT == 0.5
    i_v31697 = i_v169 - i_v21054
    i_v32193 = i_v236 * (
        jnp.where(
            i_v31765, (jnp.sqrt(jnp.maximum((i_v31697 * i_v248), 1e-300))), (jnp.power(jnp.maximum((i_v31697 * i_v248), 0.0), PGAT))
        )
    )
    i_v32472 = i_v108 * ((i_v21792 - 1.0) * i_v32193)
    i_v33651 = jnp.where(i_v31420, i_v33658, i_v32472)
    i_v42008 = jnp.where(i_v1060, i_v33658, i_v33651)
    i_v39649 = jnp.where((CSRHBOT == 0.0), (CTATBOT == 0.0), False)
    i_v39955 = PBOT == 0.5
    i_v39934 = i_v151 - i_v39869
    i_v40383 = i_v228 * (
        jnp.where(
            i_v39955, (jnp.sqrt(jnp.maximum((i_v39934 * i_v244), 1e-300))), (jnp.power(jnp.maximum((i_v39934 * i_v244), 0.0), PBOT))
        )
    )
    i_v40672 = i_v92 * ((i_v40607 - 1.0) * i_v40383)
    i_v41995 = jnp.where(i_v39649, i_v42008, i_v40672)
    i_v47227 = jnp.where(i_v977, i_v42008, i_v41995)
    i_v44989 = jnp.where((CSRHSTI == 0.0), (CTATSTI == 0.0), False)
    i_v45334 = PSTI == 0.5
    i_v45266 = i_v160 - i_v39869
    i_v45762 = i_v232 * (
        jnp.where(
            i_v45334, (jnp.sqrt(jnp.maximum((i_v45266 * i_v246), 1e-300))), (jnp.power(jnp.maximum((i_v45266 * i_v246), 0.0), PSTI))
        )
    )
    i_v46041 = i_v100 * ((i_v40607 - 1.0) * i_v45762)
    i_v47220 = jnp.where(i_v44989, i_v47227, i_v46041)
    i_v52473 = jnp.where(i_v1012, i_v47227, i_v47220)
    i_v50235 = jnp.where((CSRHGAT == 0.0), (CTATGAT == 0.0), False)
    i_v50580 = PGAT == 0.5
    i_v50512 = i_v169 - i_v39869
    i_v51008 = i_v236 * (
        jnp.where(
            i_v50580, (jnp.sqrt(jnp.maximum((i_v50512 * i_v248), 1e-300))), (jnp.power(jnp.maximum((i_v50512 * i_v248), 0.0), PGAT))
        )
    )
    i_v51287 = i_v108 * ((i_v40607 - 1.0) * i_v51008)
    i_v52466 = jnp.where(i_v50235, i_v52473, i_v51287)
    i_v61036 = jnp.where(i_v1060, i_v52473, i_v52466)
    i_v61023 = jnp.where(i_v58677, i_v61036, i_v59700)
    i_v267 = 0.5 * i_v79
    i_v269 = jnp.where((i_v267 > i_v52), i_v267, i_v52)
    i_v309 = i_v269 * i_v54
    i_v330 = jnp.divide(
        (jnp.sqrt(jnp.maximum(((((32.0 * MEFFTATBOT) * 9.1093826e-31) * 1.6021918e-19) * ((i_v269 * i_v269) * i_v269)), 1e-300))),
        3.1637150399999996e-34,
    )
    i_v3505 = jnp.where(i_v2841, 0.0, i_v3218)
    i_v8143 = jnp.where(i_v977, 0.0, i_v3505)
    i_v8136 = jnp.where(i_v7024, i_v8143, i_v7684)
    i_v13674 = jnp.where(i_v1012, i_v8143, i_v8136)
    i_v13667 = jnp.where(i_v12234, i_v13674, i_v13089)
    i_v22091 = jnp.where(i_v1060, i_v13674, i_v13667)
    i_v22078 = jnp.where(i_v20834, i_v22091, i_v21568)
    i_v27454 = jnp.where(i_v977, i_v22091, i_v22078)
    i_v27447 = jnp.where(i_v26174, i_v27454, i_v26947)
    i_v32700 = jnp.where(i_v1012, i_v27454, i_v27447)
    i_v32693 = jnp.where(i_v31420, i_v32700, i_v32193)
    i_v40906 = jnp.where(i_v1060, i_v32700, i_v32693)
    i_v40893 = jnp.where(i_v39649, i_v40906, i_v40383)
    i_v46269 = jnp.where(i_v977, i_v40906, i_v40893)
    i_v46262 = jnp.where(i_v44989, i_v46269, i_v45762)
    i_v51515 = jnp.where(i_v1012, i_v46269, i_v46262)
    i_v51508 = jnp.where(i_v50235, i_v51515, i_v51008)
    i_v59934 = jnp.where(i_v1060, i_v51515, i_v51508)
    i_v59921 = jnp.where(i_v58677, i_v59934, i_v59411)
    i_v178 = 1.0 - PBOT
    i_v3555 = jnp.where(i_v2841, 0.0, i_v2914)
    i_v8215 = jnp.where(i_v977, 0.0, i_v3555)
    i_v8208 = jnp.where(i_v7024, i_v8215, i_v7207)
    i_v13746 = jnp.where(i_v1012, i_v8215, i_v8208)
    i_v13739 = jnp.where(i_v12234, i_v13746, i_v12486)
    i_v22187 = jnp.where(i_v1060, i_v13746, i_v13739)
    i_v22174 = jnp.where(i_v20834, i_v22187, i_v21119)
    i_v27526 = jnp.where(i_v977, i_v22187, i_v22174)
    i_v27519 = jnp.where(i_v26174, i_v27526, i_v26451)
    i_v32772 = jnp.where(i_v1012, i_v27526, i_v27519)
    i_v32765 = jnp.where(i_v31420, i_v32772, i_v31697)
    i_v41002 = jnp.where(i_v1060, i_v32772, i_v32765)
    i_v40989 = jnp.where(i_v39649, i_v41002, i_v39934)
    i_v46341 = jnp.where(i_v977, i_v41002, i_v40989)
    i_v46334 = jnp.where(i_v44989, i_v46341, i_v45266)
    i_v51587 = jnp.where(i_v1012, i_v46341, i_v46334)
    i_v51580 = jnp.where(i_v50235, i_v51587, i_v50512)
    i_v60030 = jnp.where(i_v1060, i_v51587, i_v51580)
    i_v60017 = jnp.where(i_v58677, i_v60030, i_v58962)
    i_v60098 = i_v330 * (jnp.divide((i_v59921 * i_v178), jnp.where((i_v60017 == 0.0) | ~jnp.isfinite(i_v60017), 1e-300, i_v60017)))
    i_v60298 = jnp.divide((0.666666666666667 * i_v309), jnp.where((i_v60098 == 0.0) | ~jnp.isfinite(i_v60098), 1e-300, i_v60098))
    i_v60299 = i_v60298 * i_v60298
    i_v60300 = i_v60299 * i_v60299
    i_v60304 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v60300, jnp.where(((i_v60300 + 1.0) == 0.0) | ~jnp.isfinite((i_v60300 + 1.0)), 1e-300, (i_v60300 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v60305 = jnp.sqrt(jnp.maximum(i_v60304, 1e-300))
    i_v60606 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v60098, i_v60305))), 1e-300))
    i_v60628 = (((2.0 * (i_v60298 * i_v60305)) - i_v60304) - 1.0) * i_v60606
    i_v60630 = i_v60628 > 0.0
    i_v60765 = jnp.where(
        i_v60630,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v60628)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v60628)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v60628)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v60628)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v60628)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v60628)),
                ),
            )
        ),
    )
    i_v60855 = i_v60765 * i_v60765
    i_v60306 = i_v60304 * i_v60305
    i_v60626 = (((i_v309 * i_v60298) * i_v60305) - (i_v309 * i_v60304)) + (0.5 * (i_v60098 * i_v60306))
    i_v60733 = (-(i_v60628 * i_v60628)) + i_v60626
    i_v60746 = -230.25850929940458 - i_v60733
    i_v60950 = (((0.29214664 * i_v60765) + (0.26992878119627894 * i_v60855)) + (0.43792457880372104 * (i_v60855 * i_v60765))) * (
        jnp.where(
            (i_v60733 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v60733, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v60746 * (1.0 + (0.5 * (i_v60746 * (1.0 + (i_v60746 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v60746 * (1.0 + (0.5 * (i_v60746 * (1.0 + (i_v60746 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v60746 * (1.0 + (0.5 * (i_v60746 * (1.0 + (i_v60746 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v60965 = -230.25850929940458 - i_v60626
    i_v2921 = jnp.where(i_v1603, i_v2013, 0.0)
    i_v2933 = jnp.sqrt(
        jnp.maximum((1.0 - (jnp.divide(i_v2921, jnp.where((i_v2914 == 0.0) | ~jnp.isfinite(i_v2914), 1e-300, i_v2914)))), 1e-300)
    )
    i_v2934 = 1.0 - i_v2933
    i_v2949 = i_v2934 + (
        jnp.where(
            i_v2935,
            0.0,
            (
                ((jnp.divide(((i_v2934 * i_v2934) * (jnp.log(jnp.maximum(i_v2934, 1e-300)))), i_v2933)) + i_v2934)
                * (1.0 - (2.0 * PBOT))
            ),
        )
    )
    i_v3877 = jnp.where(i_v2841, 0.0, i_v2949)
    i_v8684 = jnp.where(i_v977, 0.0, i_v3877)
    i_v7273 = jnp.sqrt(
        jnp.maximum((1.0 - (jnp.divide(i_v2921, jnp.where((i_v7207 == 0.0) | ~jnp.isfinite(i_v7207), 1e-300, i_v7207)))), 1e-300)
    )
    i_v7274 = 1.0 - i_v7273
    i_v7289 = i_v7274 + (
        jnp.where(
            i_v7275,
            0.0,
            (
                ((jnp.divide(((i_v7274 * i_v7274) * (jnp.log(jnp.maximum(i_v7274, 1e-300)))), i_v7273)) + i_v7274)
                * (1.0 - (2.0 * PSTI))
            ),
        )
    )
    i_v8677 = jnp.where(i_v7024, i_v8684, i_v7289)
    i_v14341 = jnp.where(i_v1012, i_v8684, i_v8677)
    i_v12552 = jnp.sqrt(
        jnp.maximum((1.0 - (jnp.divide(i_v2921, jnp.where((i_v12486 == 0.0) | ~jnp.isfinite(i_v12486), 1e-300, i_v12486)))), 1e-300)
    )
    i_v12553 = 1.0 - i_v12552
    i_v12568 = i_v12553 + (
        jnp.where(
            i_v12554,
            0.0,
            (
                ((jnp.divide(((i_v12553 * i_v12553) * (jnp.log(jnp.maximum(i_v12553, 1e-300)))), i_v12552)) + i_v12553)
                * (1.0 - (2.0 * PGAT))
            ),
        )
    )
    i_v14334 = jnp.where(i_v12234, i_v14341, i_v12568)
    i_v22689 = jnp.where(i_v1060, i_v14341, i_v14334)
    i_v21126 = jnp.where(i_v17290, i_v19115, 0.0)
    i_v21138 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v21126, jnp.where((i_v21119 == 0.0) | ~jnp.isfinite(i_v21119), 1e-300, i_v21119)))), 1e-300
        )
    )
    i_v21139 = 1.0 - i_v21138
    i_v21154 = i_v21139 + (
        jnp.where(
            i_v21140,
            0.0,
            (
                ((jnp.divide(((i_v21139 * i_v21139) * (jnp.log(jnp.maximum(i_v21139, 1e-300)))), i_v21138)) + i_v21139)
                * (1.0 - (2.0 * PBOT))
            ),
        )
    )
    i_v22676 = jnp.where(i_v20834, i_v22689, i_v21154)
    i_v28004 = jnp.where(i_v977, i_v22689, i_v22676)
    i_v26517 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v21126, jnp.where((i_v26451 == 0.0) | ~jnp.isfinite(i_v26451), 1e-300, i_v26451)))), 1e-300
        )
    )
    i_v26518 = 1.0 - i_v26517
    i_v26533 = i_v26518 + (
        jnp.where(
            i_v26519,
            0.0,
            (
                ((jnp.divide(((i_v26518 * i_v26518) * (jnp.log(jnp.maximum(i_v26518, 1e-300)))), i_v26517)) + i_v26518)
                * (1.0 - (2.0 * PSTI))
            ),
        )
    )
    i_v27997 = jnp.where(i_v26174, i_v28004, i_v26533)
    i_v33250 = jnp.where(i_v1012, i_v28004, i_v27997)
    i_v31763 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v21126, jnp.where((i_v31697 == 0.0) | ~jnp.isfinite(i_v31697), 1e-300, i_v31697)))), 1e-300
        )
    )
    i_v31764 = 1.0 - i_v31763
    i_v31779 = i_v31764 + (
        jnp.where(
            i_v31765,
            0.0,
            (
                ((jnp.divide(((i_v31764 * i_v31764) * (jnp.log(jnp.maximum(i_v31764, 1e-300)))), i_v31763)) + i_v31764)
                * (1.0 - (2.0 * PGAT))
            ),
        )
    )
    i_v33243 = jnp.where(i_v31420, i_v33250, i_v31779)
    i_v41504 = jnp.where(i_v1060, i_v33250, i_v33243)
    i_v39941 = jnp.where(i_v35892, i_v37930, 0.0)
    i_v39953 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v39941, jnp.where((i_v39934 == 0.0) | ~jnp.isfinite(i_v39934), 1e-300, i_v39934)))), 1e-300
        )
    )
    i_v39954 = 1.0 - i_v39953
    i_v39969 = i_v39954 + (
        jnp.where(
            i_v39955,
            0.0,
            (
                ((jnp.divide(((i_v39954 * i_v39954) * (jnp.log(jnp.maximum(i_v39954, 1e-300)))), i_v39953)) + i_v39954)
                * (1.0 - (2.0 * PBOT))
            ),
        )
    )
    i_v41491 = jnp.where(i_v39649, i_v41504, i_v39969)
    i_v46819 = jnp.where(i_v977, i_v41504, i_v41491)
    i_v45332 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v39941, jnp.where((i_v45266 == 0.0) | ~jnp.isfinite(i_v45266), 1e-300, i_v45266)))), 1e-300
        )
    )
    i_v45333 = 1.0 - i_v45332
    i_v45348 = i_v45333 + (
        jnp.where(
            i_v45334,
            0.0,
            (
                ((jnp.divide(((i_v45333 * i_v45333) * (jnp.log(jnp.maximum(i_v45333, 1e-300)))), i_v45332)) + i_v45333)
                * (1.0 - (2.0 * PSTI))
            ),
        )
    )
    i_v46812 = jnp.where(i_v44989, i_v46819, i_v45348)
    i_v52065 = jnp.where(i_v1012, i_v46819, i_v46812)
    i_v50578 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v39941, jnp.where((i_v50512 == 0.0) | ~jnp.isfinite(i_v50512), 1e-300, i_v50512)))), 1e-300
        )
    )
    i_v50579 = 1.0 - i_v50578
    i_v50594 = i_v50579 + (
        jnp.where(
            i_v50580,
            0.0,
            (
                ((jnp.divide(((i_v50579 * i_v50579) * (jnp.log(jnp.maximum(i_v50579, 1e-300)))), i_v50578)) + i_v50579)
                * (1.0 - (2.0 * PGAT))
            ),
        )
    )
    i_v52058 = jnp.where(i_v50235, i_v52065, i_v50594)
    i_v60532 = jnp.where(i_v1060, i_v52065, i_v52058)
    i_v60519 = jnp.where(i_v58677, i_v60532, i_v58997)
    i_v186 = jnp.divide(1.0, jnp.where((i_v178 == 0.0) | ~jnp.isfinite(i_v178), 1e-300, i_v178))
    i_v60500 = (-PBOT) * i_v186
    i_v60596 = jnp.where(
        (i_v60500 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v60098 * i_v60306)) == 0.0) | ~jnp.isfinite(1.0 + (i_v60098 * i_v60306)),
                    1e-300,
                    (1.0 + (i_v60098 * i_v60306)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v60098 * i_v60306)), 0.0), i_v60500)),
    )
    i_v1238 = jnp.where(i_v977, (VBIRSTI + VBIRGAT), VBIRBOT)
    i_v1262 = jnp.where(i_v1012, (VBIRBOT + VBIRGAT), VBIRSTI)
    i_v1279 = jnp.where((i_v1238 < i_v1262), i_v1238, i_v1262)
    i_v1290 = jnp.where(i_v1060, (VBIRBOT + VBIRSTI), VBIRGAT)
    i_v1311 = (jnp.where((i_v1279 < i_v1290), i_v1279, i_v1290)) - 0.05
    i_v61122 = jnp.where(
        i_v54707,
        (
            0.5
            * ((0.1 + i_v1311) - (jnp.sqrt(jnp.maximum((((0.1 - i_v1311) * (0.1 - i_v1311)) + ((4.0 * i_v48) * i_v48)), 1e-300))))
        ),
        0.0,
    )
    i_v238 = jnp.divide(1.0, i_v228)
    i_v61424 = i_v186 * (
        jnp.divide(
            ((VBIRBOT - i_v61122) * i_v238),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PBOT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v61122) * i_v244), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRBOT - i_v61122) * i_v244), 0.0), PBOT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v61122) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v61122) * i_v244), 0.0), PBOT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v61122) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v61122) * i_v244), 0.0), PBOT)),
                    )
                ),
            ),
        )
    )
    i_v374 = i_v37 - i_v26
    i_v377 = FBBTRBOT * (1.0 + (STFBBTBOT * i_v374))
    i_v395 = jnp.where((i_v377 > 0.0), i_v377, 0.0)
    i_v61632 = jnp.divide((-i_v395), jnp.where((i_v61424 == 0.0) | ~jnp.isfinite(i_v61424), 1e-300, i_v61424))
    i_v61633 = i_v61632 < 0.0
    i_v61658 = -230.25850929940458 - i_v61632
    i_v61679 = i_v61632 - 230.25850929940458
    i_v5078 = jnp.where(i_v1603, (0.5 * (i_v1489 - (jnp.sqrt(jnp.maximum(((i_v1489 * i_v1489) + 4e-12), 1e-300))))), 0.0)
    i_v23915 = jnp.where(i_v17290, (0.5 * (i_v1492 - (jnp.sqrt(jnp.maximum(((i_v1492 * i_v1492) + 4e-12), 1e-300))))), i_v5078)
    i_v42730 = jnp.where(i_v35892, (0.5 * (i_v1495 - (jnp.sqrt(jnp.maximum(((i_v1495 * i_v1495) + 4e-12), 1e-300))))), i_v23915)
    i_v61758 = jnp.where(i_v54707, -1.000000082740371e-11, i_v42730)
    i_v413 = 1.0 - (jnp.divide(1.0, jnp.where((FREV == 0.0) | ~jnp.isfinite(FREV), 1e-300, FREV)))
    i_v431 = jnp.divide(1.0, jnp.where((VBRBOT == 0.0) | ~jnp.isfinite(VBRBOT), 1e-300, VBRBOT))
    i_v62117 = i_v61758 * i_v431
    i_v62126 = i_v61758 * i_v431
    i_v417 = 1.0 - (jnp.power(jnp.maximum(i_v413, 0.0), PBRBOT))
    i_v418 = jnp.divide(1.0, jnp.where((i_v417 == 0.0) | ~jnp.isfinite(i_v417), 1e-300, i_v417))
    i_v445 = ((-((i_v418 * i_v418) * (jnp.power(jnp.maximum(i_v413, 0.0), (PBRBOT - 1.0))))) * PBRBOT) * i_v431
    i_v64017 = jnp.where((CSRHSTI == 0.0), (CTATSTI == 0.0), False)
    i_v64362 = PSTI == 0.5
    i_v64294 = i_v160 - i_v58897
    i_v64790 = i_v232 * (
        jnp.where(
            i_v64362, (jnp.sqrt(jnp.maximum((i_v64294 * i_v246), 1e-300))), (jnp.power(jnp.maximum((i_v64294 * i_v246), 0.0), PSTI))
        )
    )
    i_v65069 = i_v100 * ((i_v59635 - 1.0) * i_v64790)
    i_v64360 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v58969, jnp.where((i_v64294 == 0.0) | ~jnp.isfinite(i_v64294), 1e-300, i_v64294)))), 1e-300
        )
    )
    i_v64361 = 1.0 - i_v64360
    i_v64376 = i_v64361 + (
        jnp.where(
            i_v64362,
            0.0,
            (
                ((jnp.divide(((i_v64361 * i_v64361) * (jnp.log(jnp.maximum(i_v64361, 1e-300)))), i_v64360)) + i_v64361)
                * (1.0 - (2.0 * PSTI))
            ),
        )
    )
    i_v66255 = jnp.where(i_v977, i_v61036, i_v61023)
    i_v66248 = jnp.where(i_v64017, i_v66255, i_v65069)
    i_v274 = 0.5 * i_v81
    i_v279 = jnp.where((i_v274 > i_v52), i_v274, i_v52)
    i_v314 = i_v279 * i_v54
    i_v341 = jnp.divide(
        (jnp.sqrt(jnp.maximum(((((32.0 * MEFFTATSTI) * 9.1093826e-31) * 1.6021918e-19) * ((i_v279 * i_v279) * i_v279)), 1e-300))),
        3.1637150399999996e-34,
    )
    i_v65297 = jnp.where(i_v977, i_v59934, i_v59921)
    i_v65290 = jnp.where(i_v64017, i_v65297, i_v64790)
    i_v181 = 1.0 - PSTI
    i_v65369 = jnp.where(i_v977, i_v60030, i_v60017)
    i_v65362 = jnp.where(i_v64017, i_v65369, i_v64294)
    i_v65419 = i_v341 * (jnp.divide((i_v65290 * i_v181), jnp.where((i_v65362 == 0.0) | ~jnp.isfinite(i_v65362), 1e-300, i_v65362)))
    i_v65619 = jnp.divide((0.666666666666667 * i_v314), jnp.where((i_v65419 == 0.0) | ~jnp.isfinite(i_v65419), 1e-300, i_v65419))
    i_v65620 = i_v65619 * i_v65619
    i_v65621 = i_v65620 * i_v65620
    i_v65625 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v65621, jnp.where(((i_v65621 + 1.0) == 0.0) | ~jnp.isfinite((i_v65621 + 1.0)), 1e-300, (i_v65621 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v65626 = jnp.sqrt(jnp.maximum(i_v65625, 1e-300))
    i_v65903 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v65419, i_v65626))), 1e-300))
    i_v65925 = (((2.0 * (i_v65619 * i_v65626)) - i_v65625) - 1.0) * i_v65903
    i_v65927 = i_v65925 > 0.0
    i_v66038 = jnp.where(
        i_v65927,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v65925)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v65925)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v65925)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v65925)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v65925)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v65925)),
                ),
            )
        ),
    )
    i_v66104 = i_v66038 * i_v66038
    i_v65627 = i_v65625 * i_v65626
    i_v65923 = (((i_v314 * i_v65619) * i_v65626) - (i_v314 * i_v65625)) + (0.5 * (i_v65419 * i_v65627))
    i_v66006 = (-(i_v65925 * i_v65925)) + i_v65923
    i_v66019 = -230.25850929940458 - i_v66006
    i_v66175 = (((0.29214664 * i_v66038) + (0.26992878119627894 * i_v66104)) + (0.43792457880372104 * (i_v66104 * i_v66038))) * (
        jnp.where(
            (i_v66006 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v66006, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v66019 * (1.0 + (0.5 * (i_v66019 * (1.0 + (i_v66019 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v66019 * (1.0 + (0.5 * (i_v66019 * (1.0 + (i_v66019 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v66019 * (1.0 + (0.5 * (i_v66019 * (1.0 + (i_v66019 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v66190 = -230.25850929940458 - i_v65923
    i_v65847 = jnp.where(i_v977, i_v60532, i_v60519)
    i_v65840 = jnp.where(i_v64017, i_v65847, i_v64376)
    i_v188 = jnp.divide(1.0, jnp.where((i_v181 == 0.0) | ~jnp.isfinite(i_v181), 1e-300, i_v181))
    i_v65821 = (-PSTI) * i_v188
    i_v65893 = jnp.where(
        (i_v65821 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v65419 * i_v65627)) == 0.0) | ~jnp.isfinite(1.0 + (i_v65419 * i_v65627)),
                    1e-300,
                    (1.0 + (i_v65419 * i_v65627)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v65419 * i_v65627)), 0.0), i_v65821)),
    )
    i_v240 = jnp.divide(1.0, jnp.where((i_v232 == 0.0) | ~jnp.isfinite(i_v232), 1e-300, i_v232))
    i_v66642 = i_v188 * (
        jnp.divide(
            ((VBIRSTI - i_v61122) * i_v240),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PSTI == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v61122) * i_v246), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRSTI - i_v61122) * i_v246), 0.0), PSTI)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v61122) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v61122) * i_v246), 0.0), PSTI)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v61122) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v61122) * i_v246), 0.0), PSTI)),
                    )
                ),
            ),
        )
    )
    i_v384 = FBBTRSTI * (1.0 + (STFBBTSTI * i_v374))
    i_v401 = jnp.where((i_v384 > 0.0), i_v384, 0.0)
    i_v66850 = jnp.divide((-i_v401), jnp.where((i_v66642 == 0.0) | ~jnp.isfinite(i_v66642), 1e-300, i_v66642))
    i_v66851 = i_v66850 < 0.0
    i_v66876 = -230.25850929940458 - i_v66850
    i_v66897 = i_v66850 - 230.25850929940458
    i_v434 = jnp.divide(1.0, jnp.where((VBRSTI == 0.0) | ~jnp.isfinite(VBRSTI), 1e-300, VBRSTI))
    i_v67303 = i_v61758 * i_v434
    i_v67312 = i_v61758 * i_v434
    i_v422 = 1.0 - (jnp.power(jnp.maximum(i_v413, 0.0), PBRSTI))
    i_v423 = jnp.divide(1.0, jnp.where((i_v422 == 0.0) | ~jnp.isfinite(i_v422), 1e-300, i_v422))
    i_v453 = ((-((i_v423 * i_v423) * (jnp.power(jnp.maximum(i_v413, 0.0), (PBRSTI - 1.0))))) * PBRSTI) * i_v434
    i_v69263 = jnp.where((CSRHGAT == 0.0), (CTATGAT == 0.0), False)
    i_v69608 = PGAT == 0.5
    i_v69540 = i_v169 - i_v58897
    i_v70036 = i_v236 * (
        jnp.where(
            i_v69608, (jnp.sqrt(jnp.maximum((i_v69540 * i_v248), 1e-300))), (jnp.power(jnp.maximum((i_v69540 * i_v248), 0.0), PGAT))
        )
    )
    i_v70315 = i_v108 * ((i_v59635 - 1.0) * i_v70036)
    i_v69606 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v58969, jnp.where((i_v69540 == 0.0) | ~jnp.isfinite(i_v69540), 1e-300, i_v69540)))), 1e-300
        )
    )
    i_v69607 = 1.0 - i_v69606
    i_v69622 = i_v69607 + (
        jnp.where(
            i_v69608,
            0.0,
            (
                ((jnp.divide(((i_v69607 * i_v69607) * (jnp.log(jnp.maximum(i_v69607, 1e-300)))), i_v69606)) + i_v69607)
                * (1.0 - (2.0 * PGAT))
            ),
        )
    )
    i_v71501 = jnp.where(i_v1012, i_v66255, i_v66248)
    i_v71494 = jnp.where(i_v69263, i_v71501, i_v70315)
    i_v287 = 0.5 * i_v83
    i_v292 = jnp.where((i_v287 > i_v52), i_v287, i_v52)
    i_v316 = i_v292 * i_v54
    i_v352 = jnp.divide(
        (jnp.sqrt(jnp.maximum(((((32.0 * MEFFTATGAT) * 9.1093826e-31) * 1.6021918e-19) * ((i_v292 * i_v292) * i_v292)), 1e-300))),
        3.1637150399999996e-34,
    )
    i_v70543 = jnp.where(i_v1012, i_v65297, i_v65290)
    i_v70536 = jnp.where(i_v69263, i_v70543, i_v70036)
    i_v184 = 1.0 - PGAT
    i_v70615 = jnp.where(i_v1012, i_v65369, i_v65362)
    i_v70608 = jnp.where(i_v69263, i_v70615, i_v69540)
    i_v70665 = i_v352 * (jnp.divide((i_v70536 * i_v184), jnp.where((i_v70608 == 0.0) | ~jnp.isfinite(i_v70608), 1e-300, i_v70608)))
    i_v70865 = jnp.divide((0.666666666666667 * i_v316), jnp.where((i_v70665 == 0.0) | ~jnp.isfinite(i_v70665), 1e-300, i_v70665))
    i_v70866 = i_v70865 * i_v70865
    i_v70867 = i_v70866 * i_v70866
    i_v70871 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v70867, jnp.where(((i_v70867 + 1.0) == 0.0) | ~jnp.isfinite((i_v70867 + 1.0)), 1e-300, (i_v70867 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v70872 = jnp.sqrt(jnp.maximum(i_v70871, 1e-300))
    i_v71149 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v70665, i_v70872))), 1e-300))
    i_v71171 = (((2.0 * (i_v70865 * i_v70872)) - i_v70871) - 1.0) * i_v71149
    i_v71173 = i_v71171 > 0.0
    i_v71284 = jnp.where(
        i_v71173,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v71171)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v71171)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v71171)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v71171)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v71171)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v71171)),
                ),
            )
        ),
    )
    i_v71350 = i_v71284 * i_v71284
    i_v70873 = i_v70871 * i_v70872
    i_v71169 = (((i_v316 * i_v70865) * i_v70872) - (i_v316 * i_v70871)) + (0.5 * (i_v70665 * i_v70873))
    i_v71252 = (-(i_v71171 * i_v71171)) + i_v71169
    i_v71265 = -230.25850929940458 - i_v71252
    i_v71421 = (((0.29214664 * i_v71284) + (0.26992878119627894 * i_v71350)) + (0.43792457880372104 * (i_v71350 * i_v71284))) * (
        jnp.where(
            (i_v71252 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v71252, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v71265 * (1.0 + (0.5 * (i_v71265 * (1.0 + (i_v71265 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v71265 * (1.0 + (0.5 * (i_v71265 * (1.0 + (i_v71265 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v71265 * (1.0 + (0.5 * (i_v71265 * (1.0 + (i_v71265 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v71436 = -230.25850929940458 - i_v71169
    i_v71093 = jnp.where(i_v1012, i_v65847, i_v65840)
    i_v71086 = jnp.where(i_v69263, i_v71093, i_v69622)
    i_v190 = jnp.divide(1.0, jnp.where((i_v184 == 0.0) | ~jnp.isfinite(i_v184), 1e-300, i_v184))
    i_v71067 = (-PGAT) * i_v190
    i_v71139 = jnp.where(
        (i_v71067 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v70665 * i_v70873)) == 0.0) | ~jnp.isfinite(1.0 + (i_v70665 * i_v70873)),
                    1e-300,
                    (1.0 + (i_v70665 * i_v70873)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v70665 * i_v70873)), 0.0), i_v71067)),
    )
    i_v242 = jnp.divide(1.0, jnp.where((i_v236 == 0.0) | ~jnp.isfinite(i_v236), 1e-300, i_v236))
    i_v71888 = i_v190 * (
        jnp.divide(
            ((VBIRGAT - i_v61122) * i_v242),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PGAT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v61122) * i_v248), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRGAT - i_v61122) * i_v248), 0.0), PGAT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v61122) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v61122) * i_v248), 0.0), PGAT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v61122) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v61122) * i_v248), 0.0), PGAT)),
                    )
                ),
            ),
        )
    )
    i_v391 = FBBTRGAT * (1.0 + (STFBBTGAT * i_v374))
    i_v410 = jnp.where((i_v391 > 0.0), i_v391, 0.0)
    i_v72096 = jnp.divide((-i_v410), jnp.where((i_v71888 == 0.0) | ~jnp.isfinite(i_v71888), 1e-300, i_v71888))
    i_v72097 = i_v72096 < 0.0
    i_v72122 = -230.25850929940458 - i_v72096
    i_v72143 = i_v72096 - 230.25850929940458
    i_v437 = jnp.divide(1.0, jnp.where((VBRGAT == 0.0) | ~jnp.isfinite(VBRGAT), 1e-300, VBRGAT))
    i_v72549 = i_v61758 * i_v437
    i_v72558 = i_v61758 * i_v437
    i_v427 = 1.0 - (jnp.power(jnp.maximum(i_v413, 0.0), PBRGAT))
    i_v428 = jnp.divide(1.0, jnp.where((i_v427 == 0.0) | ~jnp.isfinite(i_v427), 1e-300, i_v427))
    i_v461 = ((-((i_v428 * i_v428) * (jnp.power(jnp.maximum(i_v413, 0.0), (PBRGAT - 1.0))))) * PBRGAT) * i_v437
    i_v73723 = (
        (
            i_v465
            * (
                jnp.where(
                    i_v977,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v112 * i_v58603) + (jnp.where(i_v58677, 0.0, (CSRHBOT * (i_v59700 * i_v58997)))))
                                    + (
                                        jnp.where(
                                            (CTATBOT == 0.0),
                                            0.0,
                                            (
                                                CTATBOT
                                                * (
                                                    (
                                                        i_v61023
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v309
                                                                        * (
                                                                            jnp.where(
                                                                                i_v60630,
                                                                                i_v60950,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v60626 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v60626, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v60965
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v60965
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v60965
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v60965
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v60965
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v60965
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v60965
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v60965
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v60965
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v60950
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v60606,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v60519 * i_v60596),
                                                            jnp.where(
                                                                ((i_v60519 + i_v60596) == 0.0)
                                                                | ~jnp.isfinite((i_v60519 + i_v60596)),
                                                                1e-300,
                                                                (i_v60519 + i_v60596),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTBOT == 0.0),
                                        0.0,
                                        (
                                            CBBTBOT
                                            * (
                                                ((0.1 * i_v61424) * i_v61424)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v61633, (-i_v61632), i_v61632)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v61632, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v61633,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v61658
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v61658
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v61658
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v61658
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v61658
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v61658
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v61658
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v61658
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v61658
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v61679
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v61679
                                                                                        * (1.0 + (i_v61679 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRBOT > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v61758 > ((-i_v413) * VBRBOT)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v62117 * i_v62117) * i_v62117) * i_v62117),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v62126 < 0.0), (-i_v62126), i_v62126
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v62117 * i_v62117) * i_v62117) * i_v62117),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v62126 < 0.0), (-i_v62126), i_v62126
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRBOT == 4.0),
                                                                (((i_v62117 * i_v62117) * i_v62117) * i_v62117),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v62126 < 0.0), (-i_v62126), i_v62126)),
                                                                            0.0,
                                                                        ),
                                                                        PBRBOT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v418 + ((i_v61758 + (i_v413 * VBRBOT)) * i_v445)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
        + (
            i_v469
            * (
                jnp.where(
                    i_v1012,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v116 * i_v58603) + (jnp.where(i_v64017, 0.0, (CSRHSTI * (i_v65069 * i_v64376)))))
                                    + (
                                        jnp.where(
                                            (CTATSTI == 0.0),
                                            0.0,
                                            (
                                                CTATSTI
                                                * (
                                                    (
                                                        i_v66248
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v314
                                                                        * (
                                                                            jnp.where(
                                                                                i_v65927,
                                                                                i_v66175,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v65923 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v65923, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v66190
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v66190
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v66190
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v66190
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v66190
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v66190
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v66190
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v66190
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v66190
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v66175
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v65903,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v65840 * i_v65893),
                                                            jnp.where(
                                                                ((i_v65840 + i_v65893) == 0.0)
                                                                | ~jnp.isfinite((i_v65840 + i_v65893)),
                                                                1e-300,
                                                                (i_v65840 + i_v65893),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTSTI == 0.0),
                                        0.0,
                                        (
                                            CBBTSTI
                                            * (
                                                ((0.1 * i_v66642) * i_v66642)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v66851, (-i_v66850), i_v66850)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v66850, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v66851,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v66876
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v66876
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v66876
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v66876
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v66876
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v66876
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v66876
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v66876
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v66876
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v66897
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v66897
                                                                                        * (1.0 + (i_v66897 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRSTI > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v61758 > ((-i_v413) * VBRSTI)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRSTI == 4.0),
                                                                    (((i_v67303 * i_v67303) * i_v67303) * i_v67303),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v67312 < 0.0), (-i_v67312), i_v67312
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRSTI,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v67303 * i_v67303) * i_v67303) * i_v67303),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v67312 < 0.0), (-i_v67312), i_v67312)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v67303 * i_v67303) * i_v67303) * i_v67303),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v67312 < 0.0), (-i_v67312), i_v67312)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v423 + ((i_v61758 + (i_v413 * VBRSTI)) * i_v453)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
    ) + (
        i_v473
        * (
            jnp.where(
                i_v1060,
                0.0,
                (
                    (
                        IFACTOR
                        * (
                            (
                                ((i_v120 * i_v58603) + (jnp.where(i_v69263, 0.0, (CSRHGAT * (i_v70315 * i_v69622)))))
                                + (
                                    jnp.where(
                                        (CTATGAT == 0.0),
                                        0.0,
                                        (
                                            CTATGAT
                                            * (
                                                (
                                                    i_v71494
                                                    * (
                                                        0.886226925452758
                                                        * (
                                                            jnp.divide(
                                                                (
                                                                    i_v316
                                                                    * (
                                                                        jnp.where(
                                                                            i_v71173,
                                                                            i_v71421,
                                                                            (
                                                                                (
                                                                                    2.0
                                                                                    * (
                                                                                        jnp.where(
                                                                                            (i_v71169 > -230.25850929940458),
                                                                                            (
                                                                                                jnp.exp(
                                                                                                    jnp.clip(
                                                                                                        i_v71169, -709.0, 709.0
                                                                                                    )
                                                                                                )
                                                                                            ),
                                                                                            (
                                                                                                jnp.divide(
                                                                                                    1e-100,
                                                                                                    jnp.where(
                                                                                                        (
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v71436
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v71436
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v71436
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                            == 0.0
                                                                                                        )
                                                                                                        | ~jnp.isfinite(
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v71436
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v71436
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v71436
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                        1e-300,
                                                                                                        (
                                                                                                            1.0
                                                                                                            + (
                                                                                                                i_v71436
                                                                                                                * (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        0.5
                                                                                                                        * (
                                                                                                                            i_v71436
                                                                                                                            * (
                                                                                                                                1.0
                                                                                                                                + (
                                                                                                                                    i_v71436
                                                                                                                                    * 0.3333333333333333
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                    ),
                                                                                                )
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                                - i_v71421
                                                                            ),
                                                                        )
                                                                    )
                                                                ),
                                                                i_v71149,
                                                            )
                                                        )
                                                    )
                                                )
                                                * (
                                                    jnp.divide(
                                                        (i_v71086 * i_v71139),
                                                        jnp.where(
                                                            ((i_v71086 + i_v71139) == 0.0) | ~jnp.isfinite((i_v71086 + i_v71139)),
                                                            1e-300,
                                                            (i_v71086 + i_v71139),
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            + (
                                jnp.where(
                                    (CBBTGAT == 0.0),
                                    0.0,
                                    (
                                        CBBTGAT
                                        * (
                                            ((0.1 * i_v71888) * i_v71888)
                                            * (
                                                jnp.where(
                                                    ((jnp.where(i_v72097, (-i_v72096), i_v72096)) < 230.25850929940458),
                                                    (jnp.exp(jnp.clip(i_v72096, -709.0, 709.0))),
                                                    (
                                                        jnp.where(
                                                            i_v72097,
                                                            (
                                                                jnp.divide(
                                                                    1e-100,
                                                                    jnp.where(
                                                                        (
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v72122
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v72122
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v72122
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                            == 0.0
                                                                        )
                                                                        | ~jnp.isfinite(
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v72122
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v72122
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v72122
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                        1e-300,
                                                                        (
                                                                            1.0
                                                                            + (
                                                                                i_v72122
                                                                                * (
                                                                                    1.0
                                                                                    + (
                                                                                        0.5
                                                                                        * (
                                                                                            i_v72122
                                                                                            * (
                                                                                                1.0
                                                                                                + (i_v72122 * 0.3333333333333333)
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                    ),
                                                                )
                                                            ),
                                                            (
                                                                1e100
                                                                * (
                                                                    1.0
                                                                    + (
                                                                        i_v72143
                                                                        * (
                                                                            1.0
                                                                            + (
                                                                                0.5
                                                                                * (
                                                                                    i_v72143
                                                                                    * (1.0 + (i_v72143 * 0.3333333333333333))
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            ),
                                                        )
                                                    ),
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                    * (
                        jnp.where(
                            (VBRGAT > 1000.0),
                            1.0,
                            (
                                jnp.where(
                                    (i_v61758 > ((-i_v413) * VBRGAT)),
                                    (
                                        jnp.divide(
                                            1.0,
                                            jnp.where(
                                                (
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRGAT == 4.0),
                                                                (((i_v72549 * i_v72549) * i_v72549) * i_v72549),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v72558 < 0.0), (-i_v72558), i_v72558)),
                                                                            0.0,
                                                                        ),
                                                                        PBRGAT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    )
                                                    == 0.0
                                                )
                                                | ~jnp.isfinite(
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v72549 * i_v72549) * i_v72549) * i_v72549),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v72558 < 0.0), (-i_v72558), i_v72558)),
                                                                        0.0,
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                                1e-300,
                                                (
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v72549 * i_v72549) * i_v72549) * i_v72549),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v72558 < 0.0), (-i_v72558), i_v72558)), 0.0
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                            ),
                                        )
                                    ),
                                    (i_v428 + ((i_v61758 + (i_v413 * VBRGAT)) * i_v461)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
    )
    i_v73735 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v75247 = i_v687 > 0.2
    i_v75455 = 0.5 * (0.2 * i_v54)
    i_v75456 = i_v75455 < 0.0
    i_v75481 = -230.25850929940458 - i_v75455
    i_v75502 = i_v75455 - 230.25850929940458
    i_v75517 = jnp.where(
        ((jnp.where(i_v75456, (-i_v75455), i_v75455)) < 230.25850929940458),
        (jnp.exp(jnp.clip(i_v75455, -709.0, 709.0))),
        (
            jnp.where(
                i_v75456,
                (
                    jnp.divide(
                        1e-100,
                        jnp.where(
                            ((1.0 + (i_v75481 * (1.0 + (0.5 * (i_v75481 * (1.0 + (i_v75481 * 0.3333333333333333))))))) == 0.0)
                            | ~jnp.isfinite(
                                1.0 + (i_v75481 * (1.0 + (0.5 * (i_v75481 * (1.0 + (i_v75481 * 0.3333333333333333))))))
                            ),
                            1e-300,
                            (1.0 + (i_v75481 * (1.0 + (0.5 * (i_v75481 * (1.0 + (i_v75481 * 0.3333333333333333))))))),
                        ),
                    )
                ),
                (1e100 * (1.0 + (i_v75502 * (1.0 + (0.5 * (i_v75502 * (1.0 + (i_v75502 * 0.3333333333333333)))))))),
            )
        ),
    )
    i_v75739 = (1.0 + ((0.2 - i_v687) * i_v54)) * i_v1852
    i_v77844 = jnp.where(i_v73735, ((jnp.where(i_v75247, (i_v75517 * i_v75517), i_v75739)) - 1.0), i_v58603)
    i_v77918 = jnp.where((CSRHBOT == 0.0), (CTATBOT == 0.0), False)
    i_v75743 = jnp.where(i_v75247, i_v75517, (jnp.sqrt(jnp.maximum(i_v75739, 1e-300))))
    i_v78876 = jnp.where(i_v73735, i_v75743, i_v59635)
    i_v78224 = PBOT == 0.5
    i_v75744 = jnp.divide(1.0, jnp.where((i_v75743 == 0.0) | ~jnp.isfinite(i_v75743), 1e-300, i_v75743))
    i_v75969 = 2.0 * (
        i_v52
        * (
            jnp.log(
                jnp.maximum(((2.0 + i_v75744) + (jnp.sqrt(jnp.maximum(((i_v75744 + 1.0) * (i_v75744 + 3.0)), 1e-300)))), 1e-300)
            )
        )
    )
    i_v76200 = i_v1141 - i_v75969
    i_v78138 = jnp.where(
        i_v73735,
        (
            0.5
            * (
                (0.2 + i_v76200)
                - (jnp.sqrt(jnp.maximum((((0.2 - i_v76200) * (0.2 - i_v76200)) + ((4.0 * i_v52) * i_v52)), 1e-300)))
            )
        ),
        i_v58897,
    )
    i_v78203 = i_v151 - i_v78138
    i_v78652 = i_v228 * (
        jnp.where(
            i_v78224, (jnp.sqrt(jnp.maximum((i_v78203 * i_v244), 1e-300))), (jnp.power(jnp.maximum((i_v78203 * i_v244), 0.0), PBOT))
        )
    )
    i_v78941 = i_v92 * ((i_v78876 - 1.0) * i_v78652)
    i_v78210 = jnp.where(i_v73735, i_v75969, 0.0)
    i_v78222 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v78210, jnp.where((i_v78203 == 0.0) | ~jnp.isfinite(i_v78203), 1e-300, i_v78203)))), 1e-300
        )
    )
    i_v78223 = 1.0 - i_v78222
    i_v78238 = i_v78223 + (
        jnp.where(
            i_v78224,
            0.0,
            (
                ((jnp.divide(((i_v78223 * i_v78223) * (jnp.log(jnp.maximum(i_v78223, 1e-300)))), i_v78222)) + i_v78223)
                * (1.0 - (2.0 * PBOT))
            ),
        )
    )
    i_v80277 = jnp.where(i_v1060, i_v71501, i_v71494)
    i_v80264 = jnp.where(i_v77918, i_v80277, i_v78941)
    i_v79175 = jnp.where(i_v1060, i_v70543, i_v70536)
    i_v79162 = jnp.where(i_v77918, i_v79175, i_v78652)
    i_v79271 = jnp.where(i_v1060, i_v70615, i_v70608)
    i_v79258 = jnp.where(i_v77918, i_v79271, i_v78203)
    i_v79339 = i_v330 * (jnp.divide((i_v79162 * i_v178), jnp.where((i_v79258 == 0.0) | ~jnp.isfinite(i_v79258), 1e-300, i_v79258)))
    i_v79539 = jnp.divide((0.666666666666667 * i_v309), jnp.where((i_v79339 == 0.0) | ~jnp.isfinite(i_v79339), 1e-300, i_v79339))
    i_v79540 = i_v79539 * i_v79539
    i_v79541 = i_v79540 * i_v79540
    i_v79545 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v79541, jnp.where(((i_v79541 + 1.0) == 0.0) | ~jnp.isfinite((i_v79541 + 1.0)), 1e-300, (i_v79541 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v79546 = jnp.sqrt(jnp.maximum(i_v79545, 1e-300))
    i_v79847 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v79339, i_v79546))), 1e-300))
    i_v79869 = (((2.0 * (i_v79539 * i_v79546)) - i_v79545) - 1.0) * i_v79847
    i_v79871 = i_v79869 > 0.0
    i_v80006 = jnp.where(
        i_v79871,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v79869)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v79869)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v79869)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v79869)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v79869)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v79869)),
                ),
            )
        ),
    )
    i_v80096 = i_v80006 * i_v80006
    i_v79547 = i_v79545 * i_v79546
    i_v79867 = (((i_v309 * i_v79539) * i_v79546) - (i_v309 * i_v79545)) + (0.5 * (i_v79339 * i_v79547))
    i_v79974 = (-(i_v79869 * i_v79869)) + i_v79867
    i_v79987 = -230.25850929940458 - i_v79974
    i_v80191 = (((0.29214664 * i_v80006) + (0.26992878119627894 * i_v80096)) + (0.43792457880372104 * (i_v80096 * i_v80006))) * (
        jnp.where(
            (i_v79974 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v79974, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v79987 * (1.0 + (0.5 * (i_v79987 * (1.0 + (i_v79987 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v79987 * (1.0 + (0.5 * (i_v79987 * (1.0 + (i_v79987 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v79987 * (1.0 + (0.5 * (i_v79987 * (1.0 + (i_v79987 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v80206 = -230.25850929940458 - i_v79867
    i_v79773 = jnp.where(i_v1060, i_v71093, i_v71086)
    i_v79760 = jnp.where(i_v77918, i_v79773, i_v78238)
    i_v79741 = (-PBOT) * i_v186
    i_v79837 = jnp.where(
        (i_v79741 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v79339 * i_v79547)) == 0.0) | ~jnp.isfinite(1.0 + (i_v79339 * i_v79547)),
                    1e-300,
                    (1.0 + (i_v79339 * i_v79547)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v79339 * i_v79547)), 0.0), i_v79741)),
    )
    i_v80363 = jnp.where(
        i_v73735,
        (
            0.5
            * ((0.2 + i_v1311) - (jnp.sqrt(jnp.maximum((((0.2 - i_v1311) * (0.2 - i_v1311)) + ((4.0 * i_v48) * i_v48)), 1e-300))))
        ),
        0.0,
    )
    i_v80665 = i_v186 * (
        jnp.divide(
            ((VBIRBOT - i_v80363) * i_v238),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PBOT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v80363) * i_v244), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRBOT - i_v80363) * i_v244), 0.0), PBOT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v80363) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v80363) * i_v244), 0.0), PBOT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v80363) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v80363) * i_v244), 0.0), PBOT)),
                    )
                ),
            ),
        )
    )
    i_v80873 = jnp.divide((-i_v395), jnp.where((i_v80665 == 0.0) | ~jnp.isfinite(i_v80665), 1e-300, i_v80665))
    i_v80874 = i_v80873 < 0.0
    i_v80899 = -230.25850929940458 - i_v80873
    i_v80920 = i_v80873 - 230.25850929940458
    i_v80999 = jnp.where(i_v73735, -5.000000413701855e-12, i_v61758)
    i_v81358 = i_v80999 * i_v431
    i_v81367 = i_v80999 * i_v431
    i_v83258 = jnp.where((CSRHSTI == 0.0), (CTATSTI == 0.0), False)
    i_v83603 = PSTI == 0.5
    i_v83535 = i_v160 - i_v78138
    i_v84031 = i_v232 * (
        jnp.where(
            i_v83603, (jnp.sqrt(jnp.maximum((i_v83535 * i_v246), 1e-300))), (jnp.power(jnp.maximum((i_v83535 * i_v246), 0.0), PSTI))
        )
    )
    i_v84310 = i_v100 * ((i_v78876 - 1.0) * i_v84031)
    i_v83601 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v78210, jnp.where((i_v83535 == 0.0) | ~jnp.isfinite(i_v83535), 1e-300, i_v83535)))), 1e-300
        )
    )
    i_v83602 = 1.0 - i_v83601
    i_v83617 = i_v83602 + (
        jnp.where(
            i_v83603,
            0.0,
            (
                ((jnp.divide(((i_v83602 * i_v83602) * (jnp.log(jnp.maximum(i_v83602, 1e-300)))), i_v83601)) + i_v83602)
                * (1.0 - (2.0 * PSTI))
            ),
        )
    )
    i_v85496 = jnp.where(i_v977, i_v80277, i_v80264)
    i_v85489 = jnp.where(i_v83258, i_v85496, i_v84310)
    i_v84538 = jnp.where(i_v977, i_v79175, i_v79162)
    i_v84531 = jnp.where(i_v83258, i_v84538, i_v84031)
    i_v84610 = jnp.where(i_v977, i_v79271, i_v79258)
    i_v84603 = jnp.where(i_v83258, i_v84610, i_v83535)
    i_v84660 = i_v341 * (jnp.divide((i_v84531 * i_v181), jnp.where((i_v84603 == 0.0) | ~jnp.isfinite(i_v84603), 1e-300, i_v84603)))
    i_v84860 = jnp.divide((0.666666666666667 * i_v314), jnp.where((i_v84660 == 0.0) | ~jnp.isfinite(i_v84660), 1e-300, i_v84660))
    i_v84861 = i_v84860 * i_v84860
    i_v84862 = i_v84861 * i_v84861
    i_v84866 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v84862, jnp.where(((i_v84862 + 1.0) == 0.0) | ~jnp.isfinite((i_v84862 + 1.0)), 1e-300, (i_v84862 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v84867 = jnp.sqrt(jnp.maximum(i_v84866, 1e-300))
    i_v85144 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v84660, i_v84867))), 1e-300))
    i_v85166 = (((2.0 * (i_v84860 * i_v84867)) - i_v84866) - 1.0) * i_v85144
    i_v85168 = i_v85166 > 0.0
    i_v85279 = jnp.where(
        i_v85168,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v85166)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v85166)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v85166)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v85166)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v85166)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v85166)),
                ),
            )
        ),
    )
    i_v85345 = i_v85279 * i_v85279
    i_v84868 = i_v84866 * i_v84867
    i_v85164 = (((i_v314 * i_v84860) * i_v84867) - (i_v314 * i_v84866)) + (0.5 * (i_v84660 * i_v84868))
    i_v85247 = (-(i_v85166 * i_v85166)) + i_v85164
    i_v85260 = -230.25850929940458 - i_v85247
    i_v85416 = (((0.29214664 * i_v85279) + (0.26992878119627894 * i_v85345)) + (0.43792457880372104 * (i_v85345 * i_v85279))) * (
        jnp.where(
            (i_v85247 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v85247, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v85260 * (1.0 + (0.5 * (i_v85260 * (1.0 + (i_v85260 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v85260 * (1.0 + (0.5 * (i_v85260 * (1.0 + (i_v85260 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v85260 * (1.0 + (0.5 * (i_v85260 * (1.0 + (i_v85260 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v85431 = -230.25850929940458 - i_v85164
    i_v85088 = jnp.where(i_v977, i_v79773, i_v79760)
    i_v85081 = jnp.where(i_v83258, i_v85088, i_v83617)
    i_v85062 = (-PSTI) * i_v188
    i_v85134 = jnp.where(
        (i_v85062 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v84660 * i_v84868)) == 0.0) | ~jnp.isfinite(1.0 + (i_v84660 * i_v84868)),
                    1e-300,
                    (1.0 + (i_v84660 * i_v84868)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v84660 * i_v84868)), 0.0), i_v85062)),
    )
    i_v85883 = i_v188 * (
        jnp.divide(
            ((VBIRSTI - i_v80363) * i_v240),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PSTI == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v80363) * i_v246), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRSTI - i_v80363) * i_v246), 0.0), PSTI)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v80363) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v80363) * i_v246), 0.0), PSTI)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v80363) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v80363) * i_v246), 0.0), PSTI)),
                    )
                ),
            ),
        )
    )
    i_v86091 = jnp.divide((-i_v401), jnp.where((i_v85883 == 0.0) | ~jnp.isfinite(i_v85883), 1e-300, i_v85883))
    i_v86092 = i_v86091 < 0.0
    i_v86117 = -230.25850929940458 - i_v86091
    i_v86138 = i_v86091 - 230.25850929940458
    i_v86544 = i_v80999 * i_v434
    i_v86553 = i_v80999 * i_v434
    i_v88504 = jnp.where((CSRHGAT == 0.0), (CTATGAT == 0.0), False)
    i_v88849 = PGAT == 0.5
    i_v88781 = i_v169 - i_v78138
    i_v89277 = i_v236 * (
        jnp.where(
            i_v88849, (jnp.sqrt(jnp.maximum((i_v88781 * i_v248), 1e-300))), (jnp.power(jnp.maximum((i_v88781 * i_v248), 0.0), PGAT))
        )
    )
    i_v89556 = i_v108 * ((i_v78876 - 1.0) * i_v89277)
    i_v88847 = jnp.sqrt(
        jnp.maximum(
            (1.0 - (jnp.divide(i_v78210, jnp.where((i_v88781 == 0.0) | ~jnp.isfinite(i_v88781), 1e-300, i_v88781)))), 1e-300
        )
    )
    i_v88848 = 1.0 - i_v88847
    i_v88863 = i_v88848 + (
        jnp.where(
            i_v88849,
            0.0,
            (
                ((jnp.divide(((i_v88848 * i_v88848) * (jnp.log(jnp.maximum(i_v88848, 1e-300)))), i_v88847)) + i_v88848)
                * (1.0 - (2.0 * PGAT))
            ),
        )
    )
    i_v89906 = i_v352 * (
        jnp.divide(
            ((jnp.where(i_v88504, (jnp.where(i_v1012, i_v84538, i_v84531)), i_v89277)) * i_v184),
            jnp.where(
                ((jnp.where(i_v88504, (jnp.where(i_v1012, i_v84610, i_v84603)), i_v88781)) == 0.0)
                | ~jnp.isfinite(jnp.where(i_v88504, (jnp.where(i_v1012, i_v84610, i_v84603)), i_v88781)),
                1e-300,
                (jnp.where(i_v88504, (jnp.where(i_v1012, i_v84610, i_v84603)), i_v88781)),
            ),
        )
    )
    i_v90106 = jnp.divide((0.666666666666667 * i_v316), jnp.where((i_v89906 == 0.0) | ~jnp.isfinite(i_v89906), 1e-300, i_v89906))
    i_v90107 = i_v90106 * i_v90106
    i_v90108 = i_v90107 * i_v90107
    i_v90112 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v90108, jnp.where(((i_v90108 + 1.0) == 0.0) | ~jnp.isfinite((i_v90108 + 1.0)), 1e-300, (i_v90108 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v90113 = jnp.sqrt(jnp.maximum(i_v90112, 1e-300))
    i_v90390 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v89906, i_v90113))), 1e-300))
    i_v90412 = (((2.0 * (i_v90106 * i_v90113)) - i_v90112) - 1.0) * i_v90390
    i_v90414 = i_v90412 > 0.0
    i_v90525 = jnp.where(
        i_v90414,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v90412)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v90412)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v90412)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v90412)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v90412)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v90412)),
                ),
            )
        ),
    )
    i_v90591 = i_v90525 * i_v90525
    i_v90114 = i_v90112 * i_v90113
    i_v90410 = (((i_v316 * i_v90106) * i_v90113) - (i_v316 * i_v90112)) + (0.5 * (i_v89906 * i_v90114))
    i_v90493 = (-(i_v90412 * i_v90412)) + i_v90410
    i_v90506 = -230.25850929940458 - i_v90493
    i_v90662 = (((0.29214664 * i_v90525) + (0.26992878119627894 * i_v90591)) + (0.43792457880372104 * (i_v90591 * i_v90525))) * (
        jnp.where(
            (i_v90493 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v90493, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v90506 * (1.0 + (0.5 * (i_v90506 * (1.0 + (i_v90506 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v90506 * (1.0 + (0.5 * (i_v90506 * (1.0 + (i_v90506 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v90506 * (1.0 + (0.5 * (i_v90506 * (1.0 + (i_v90506 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v90677 = -230.25850929940458 - i_v90410
    i_v90327 = jnp.where(i_v88504, (jnp.where(i_v1012, i_v85088, i_v85081)), i_v88863)
    i_v90308 = (-PGAT) * i_v190
    i_v90380 = jnp.where(
        (i_v90308 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v89906 * i_v90114)) == 0.0) | ~jnp.isfinite(1.0 + (i_v89906 * i_v90114)),
                    1e-300,
                    (1.0 + (i_v89906 * i_v90114)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v89906 * i_v90114)), 0.0), i_v90308)),
    )
    i_v91129 = i_v190 * (
        jnp.divide(
            ((VBIRGAT - i_v80363) * i_v242),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PGAT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v80363) * i_v248), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRGAT - i_v80363) * i_v248), 0.0), PGAT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v80363) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v80363) * i_v248), 0.0), PGAT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v80363) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v80363) * i_v248), 0.0), PGAT)),
                    )
                ),
            ),
        )
    )
    i_v91337 = jnp.divide((-i_v410), jnp.where((i_v91129 == 0.0) | ~jnp.isfinite(i_v91129), 1e-300, i_v91129))
    i_v91338 = i_v91337 < 0.0
    i_v91363 = -230.25850929940458 - i_v91337
    i_v91384 = i_v91337 - 230.25850929940458
    i_v91790 = i_v80999 * i_v437
    i_v91799 = i_v80999 * i_v437
    i_v92964 = (
        (
            i_v465
            * (
                jnp.where(
                    i_v977,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v112 * i_v77844) + (jnp.where(i_v77918, 0.0, (CSRHBOT * (i_v78941 * i_v78238)))))
                                    + (
                                        jnp.where(
                                            (CTATBOT == 0.0),
                                            0.0,
                                            (
                                                CTATBOT
                                                * (
                                                    (
                                                        i_v80264
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v309
                                                                        * (
                                                                            jnp.where(
                                                                                i_v79871,
                                                                                i_v80191,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v79867 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v79867, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v80206
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v80206
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v80206
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v80206
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v80206
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v80206
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v80206
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v80206
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v80206
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v80191
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v79847,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v79760 * i_v79837),
                                                            jnp.where(
                                                                ((i_v79760 + i_v79837) == 0.0)
                                                                | ~jnp.isfinite((i_v79760 + i_v79837)),
                                                                1e-300,
                                                                (i_v79760 + i_v79837),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTBOT == 0.0),
                                        0.0,
                                        (
                                            CBBTBOT
                                            * (
                                                ((0.2 * i_v80665) * i_v80665)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v80874, (-i_v80873), i_v80873)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v80873, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v80874,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v80899
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v80899
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v80899
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v80899
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v80899
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v80899
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v80899
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v80899
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v80899
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v80920
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v80920
                                                                                        * (1.0 + (i_v80920 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRBOT > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v80999 > ((-i_v413) * VBRBOT)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v81358 * i_v81358) * i_v81358) * i_v81358),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v81367 < 0.0), (-i_v81367), i_v81367
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v81358 * i_v81358) * i_v81358) * i_v81358),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v81367 < 0.0), (-i_v81367), i_v81367
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRBOT == 4.0),
                                                                (((i_v81358 * i_v81358) * i_v81358) * i_v81358),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v81367 < 0.0), (-i_v81367), i_v81367)),
                                                                            0.0,
                                                                        ),
                                                                        PBRBOT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v418 + ((i_v80999 + (i_v413 * VBRBOT)) * i_v445)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
        + (
            i_v469
            * (
                jnp.where(
                    i_v1012,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v116 * i_v77844) + (jnp.where(i_v83258, 0.0, (CSRHSTI * (i_v84310 * i_v83617)))))
                                    + (
                                        jnp.where(
                                            (CTATSTI == 0.0),
                                            0.0,
                                            (
                                                CTATSTI
                                                * (
                                                    (
                                                        i_v85489
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v314
                                                                        * (
                                                                            jnp.where(
                                                                                i_v85168,
                                                                                i_v85416,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v85164 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v85164, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v85431
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v85431
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v85431
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v85431
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v85431
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v85431
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v85431
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v85431
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v85431
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v85416
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v85144,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v85081 * i_v85134),
                                                            jnp.where(
                                                                ((i_v85081 + i_v85134) == 0.0)
                                                                | ~jnp.isfinite((i_v85081 + i_v85134)),
                                                                1e-300,
                                                                (i_v85081 + i_v85134),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTSTI == 0.0),
                                        0.0,
                                        (
                                            CBBTSTI
                                            * (
                                                ((0.2 * i_v85883) * i_v85883)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v86092, (-i_v86091), i_v86091)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v86091, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v86092,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v86117
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v86117
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v86117
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v86117
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v86117
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v86117
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v86117
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v86117
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v86117
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v86138
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v86138
                                                                                        * (1.0 + (i_v86138 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRSTI > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v80999 > ((-i_v413) * VBRSTI)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRSTI == 4.0),
                                                                    (((i_v86544 * i_v86544) * i_v86544) * i_v86544),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v86553 < 0.0), (-i_v86553), i_v86553
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRSTI,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v86544 * i_v86544) * i_v86544) * i_v86544),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v86553 < 0.0), (-i_v86553), i_v86553)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v86544 * i_v86544) * i_v86544) * i_v86544),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v86553 < 0.0), (-i_v86553), i_v86553)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v423 + ((i_v80999 + (i_v413 * VBRSTI)) * i_v453)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
    ) + (
        i_v473
        * (
            jnp.where(
                i_v1060,
                0.0,
                (
                    (
                        IFACTOR
                        * (
                            (
                                ((i_v120 * i_v77844) + (jnp.where(i_v88504, 0.0, (CSRHGAT * (i_v89556 * i_v88863)))))
                                + (
                                    jnp.where(
                                        (CTATGAT == 0.0),
                                        0.0,
                                        (
                                            CTATGAT
                                            * (
                                                (
                                                    (jnp.where(i_v88504, (jnp.where(i_v1012, i_v85496, i_v85489)), i_v89556))
                                                    * (
                                                        0.886226925452758
                                                        * (
                                                            jnp.divide(
                                                                (
                                                                    i_v316
                                                                    * (
                                                                        jnp.where(
                                                                            i_v90414,
                                                                            i_v90662,
                                                                            (
                                                                                (
                                                                                    2.0
                                                                                    * (
                                                                                        jnp.where(
                                                                                            (i_v90410 > -230.25850929940458),
                                                                                            (
                                                                                                jnp.exp(
                                                                                                    jnp.clip(
                                                                                                        i_v90410, -709.0, 709.0
                                                                                                    )
                                                                                                )
                                                                                            ),
                                                                                            (
                                                                                                jnp.divide(
                                                                                                    1e-100,
                                                                                                    jnp.where(
                                                                                                        (
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v90677
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v90677
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v90677
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                            == 0.0
                                                                                                        )
                                                                                                        | ~jnp.isfinite(
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v90677
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v90677
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v90677
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                        1e-300,
                                                                                                        (
                                                                                                            1.0
                                                                                                            + (
                                                                                                                i_v90677
                                                                                                                * (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        0.5
                                                                                                                        * (
                                                                                                                            i_v90677
                                                                                                                            * (
                                                                                                                                1.0
                                                                                                                                + (
                                                                                                                                    i_v90677
                                                                                                                                    * 0.3333333333333333
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                    ),
                                                                                                )
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                                - i_v90662
                                                                            ),
                                                                        )
                                                                    )
                                                                ),
                                                                i_v90390,
                                                            )
                                                        )
                                                    )
                                                )
                                                * (
                                                    jnp.divide(
                                                        (i_v90327 * i_v90380),
                                                        jnp.where(
                                                            ((i_v90327 + i_v90380) == 0.0) | ~jnp.isfinite((i_v90327 + i_v90380)),
                                                            1e-300,
                                                            (i_v90327 + i_v90380),
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            + (
                                jnp.where(
                                    (CBBTGAT == 0.0),
                                    0.0,
                                    (
                                        CBBTGAT
                                        * (
                                            ((0.2 * i_v91129) * i_v91129)
                                            * (
                                                jnp.where(
                                                    ((jnp.where(i_v91338, (-i_v91337), i_v91337)) < 230.25850929940458),
                                                    (jnp.exp(jnp.clip(i_v91337, -709.0, 709.0))),
                                                    (
                                                        jnp.where(
                                                            i_v91338,
                                                            (
                                                                jnp.divide(
                                                                    1e-100,
                                                                    jnp.where(
                                                                        (
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v91363
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v91363
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v91363
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                            == 0.0
                                                                        )
                                                                        | ~jnp.isfinite(
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v91363
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v91363
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v91363
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                        1e-300,
                                                                        (
                                                                            1.0
                                                                            + (
                                                                                i_v91363
                                                                                * (
                                                                                    1.0
                                                                                    + (
                                                                                        0.5
                                                                                        * (
                                                                                            i_v91363
                                                                                            * (
                                                                                                1.0
                                                                                                + (i_v91363 * 0.3333333333333333)
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                    ),
                                                                )
                                                            ),
                                                            (
                                                                1e100
                                                                * (
                                                                    1.0
                                                                    + (
                                                                        i_v91384
                                                                        * (
                                                                            1.0
                                                                            + (
                                                                                0.5
                                                                                * (
                                                                                    i_v91384
                                                                                    * (1.0 + (i_v91384 * 0.3333333333333333))
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            ),
                                                        )
                                                    ),
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                    * (
                        jnp.where(
                            (VBRGAT > 1000.0),
                            1.0,
                            (
                                jnp.where(
                                    (i_v80999 > ((-i_v413) * VBRGAT)),
                                    (
                                        jnp.divide(
                                            1.0,
                                            jnp.where(
                                                (
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRGAT == 4.0),
                                                                (((i_v91790 * i_v91790) * i_v91790) * i_v91790),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v91799 < 0.0), (-i_v91799), i_v91799)),
                                                                            0.0,
                                                                        ),
                                                                        PBRGAT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    )
                                                    == 0.0
                                                )
                                                | ~jnp.isfinite(
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v91790 * i_v91790) * i_v91790) * i_v91790),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v91799 < 0.0), (-i_v91799), i_v91799)),
                                                                        0.0,
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                                1e-300,
                                                (
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v91790 * i_v91790) * i_v91790) * i_v91790),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v91799 < 0.0), (-i_v91799), i_v91799)), 0.0
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                            ),
                                        )
                                    ),
                                    (i_v428 + ((i_v80999 + (i_v413 * VBRGAT)) * i_v461)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
    )
    i_v95098 = jnp.where((i_v73723 > 0.0), (i_v92964 > 0.0), False)
    i_v93976 = 0.1 * i_v54
    i_v95047 = i_v73723 - (i_v93336 * ((jnp.exp(jnp.clip(i_v93976, -709.0, 709.0))) - 1.0))
    i_v95069 = i_v92964 - (i_v93336 * ((jnp.exp(jnp.clip((0.2 * i_v54), -709.0, 709.0))) - 1.0))
    i_v95160 = jnp.where(
        (
            jnp.where(
                (
                    jnp.where(
                        (
                            jnp.where(
                                (
                                    (jnp.divide(i_v95047, jnp.where((i_v73723 == 0.0) | ~jnp.isfinite(i_v73723), 1e-300, i_v73723)))
                                    > 0.001
                                ),
                                True,
                                (
                                    (jnp.divide(i_v95069, jnp.where((i_v92964 == 0.0) | ~jnp.isfinite(i_v92964), 1e-300, i_v92964)))
                                    > 0.001
                                ),
                            )
                        ),
                        (i_v95047 > 0.0),
                        False,
                    )
                ),
                (i_v95069 > 0.0),
                False,
            )
        ),
        (i_v95069 > i_v95047),
        False,
    )
    i_v95434 = jnp.divide(
        (
            i_v52
            * (
                jnp.log(
                    jnp.maximum(
                        (jnp.divide(i_v95047, jnp.where((i_v95069 == 0.0) | ~jnp.isfinite(i_v95069), 1e-300, i_v95069))), 1e-300
                    )
                )
            )
        ),
        -0.1,
    )
    i_v97306 = jnp.where(
        i_v95098,
        (
            jnp.where(
                i_v95160,
                (
                    jnp.divide(
                        i_v95047,
                        jnp.where(
                            (((jnp.exp(jnp.clip((i_v93976 * i_v95434), -709.0, 709.0))) - 1.0) == 0.0)
                            | ~jnp.isfinite((jnp.exp(jnp.clip((i_v93976 * i_v95434), -709.0, 709.0))) - 1.0),
                            1e-300,
                            ((jnp.exp(jnp.clip((i_v93976 * i_v95434), -709.0, 709.0))) - 1.0),
                        ),
                    )
                ),
                0.0,
            )
        ),
        0.0,
    )
    i_v106070 = jnp.where(i_v95080, i_v97306, 0.0)
    i_v112256 = jnp.where(i_v1435, i_v106070, 0.0)
    i_v98400 = jnp.where(i_v95098, (jnp.where(i_v95160, i_v95434, 1.0)), 1.0)
    i_v112134 = jnp.where(i_v1435, (jnp.where(i_v95080, i_v98400, 1.0)), 1.0)
    i_v108413 = jnp.where(
        i_v106001,
        (
            jnp.log(
                jnp.maximum(
                    (
                        jnp.divide(
                            i_v106002,
                            jnp.where(
                                ((i_v106070 + 1e-21) == 0.0) | ~jnp.isfinite((i_v106070 + 1e-21)), 1e-300, (i_v106070 + 1e-21)
                            ),
                        )
                    ),
                    1e-300,
                )
            )
        ),
        0.0,
    )
    i_v109564 = jnp.where((i_v108413 < 230.25850929940458), i_v108413, 230.25850929940458)
    (
        i_v112218,
        i_v112233,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1435, _t, _f),
        (
            i_v109564,
            (jnp.exp(jnp.clip(i_v109564, -709.0, 709.0))),
        ),
        (
            0.0,
            0.0,
        ),
    )
    i_v3590 = i_v330 * (jnp.divide((i_v3505 * i_v178), jnp.where((i_v3555 == 0.0) | ~jnp.isfinite(i_v3555), 1e-300, i_v3555)))
    i_v3716 = jnp.divide((0.666666666666667 * i_v309), jnp.where((i_v3590 == 0.0) | ~jnp.isfinite(i_v3590), 1e-300, i_v3590))
    i_v3717 = i_v3716 * i_v3716
    i_v3718 = i_v3717 * i_v3717
    i_v3722 = jnp.sqrt(
        jnp.maximum(
            (jnp.divide(i_v3718, jnp.where(((i_v3718 + 1.0) == 0.0) | ~jnp.isfinite((i_v3718 + 1.0)), 1e-300, (i_v3718 + 1.0)))),
            1e-300,
        )
    )
    i_v3723 = jnp.sqrt(jnp.maximum(i_v3722, 1e-300))
    i_v3922 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v3590, i_v3723))), 1e-300))
    i_v3944 = (((2.0 * (i_v3716 * i_v3723)) - i_v3722) - 1.0) * i_v3922
    i_v3946 = i_v3944 > 0.0
    i_v4131 = jnp.where(
        i_v3946,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v3944)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v3944)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v3944)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v3944)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v3944)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v3944)),
                ),
            )
        ),
    )
    i_v4275 = i_v4131 * i_v4131
    i_v3724 = i_v3722 * i_v3723
    i_v3942 = (((i_v309 * i_v3716) * i_v3723) - (i_v309 * i_v3722)) + (0.5 * (i_v3590 * i_v3724))
    i_v4099 = (-(i_v3944 * i_v3944)) + i_v3942
    i_v4112 = -230.25850929940458 - i_v4099
    i_v4424 = (((0.29214664 * i_v4131) + (0.26992878119627894 * i_v4275)) + (0.43792457880372104 * (i_v4275 * i_v4131))) * (
        jnp.where(
            (i_v4099 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v4099, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v4112 * (1.0 + (0.5 * (i_v4112 * (1.0 + (i_v4112 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v4112 * (1.0 + (0.5 * (i_v4112 * (1.0 + (i_v4112 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v4112 * (1.0 + (0.5 * (i_v4112 * (1.0 + (i_v4112 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v4439 = -230.25850929940458 - i_v3942
    i_v3858 = (-PBOT) * i_v186
    i_v3911 = jnp.where(
        (i_v3858 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v3590 * i_v3724)) == 0.0) | ~jnp.isfinite(1.0 + (i_v3590 * i_v3724)),
                    1e-300,
                    (1.0 + (i_v3590 * i_v3724)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v3590 * i_v3724)), 0.0), i_v3858)),
    )
    i_v4566 = jnp.where(
        i_v1603,
        (
            0.5
            * (
                (i_v1489 + i_v1311)
                - (jnp.sqrt(jnp.maximum((((i_v1489 - i_v1311) * (i_v1489 - i_v1311)) + ((4.0 * i_v48) * i_v48)), 1e-300)))
            )
        ),
        0.0,
    )
    i_v4808 = i_v186 * (
        jnp.divide(
            ((VBIRBOT - i_v4566) * i_v238),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PBOT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v4566) * i_v244), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRBOT - i_v4566) * i_v244), 0.0), PBOT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v4566) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v4566) * i_v244), 0.0), PBOT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v4566) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v4566) * i_v244), 0.0), PBOT)),
                    )
                ),
            ),
        )
    )
    i_v4951 = jnp.divide((-i_v395), jnp.where((i_v4808 == 0.0) | ~jnp.isfinite(i_v4808), 1e-300, i_v4808))
    i_v4952 = i_v4951 < 0.0
    i_v4977 = -230.25850929940458 - i_v4951
    i_v4998 = i_v4951 - 230.25850929940458
    i_v5426 = i_v5078 * i_v431
    i_v5435 = i_v5078 * i_v431
    i_v8265 = i_v341 * (jnp.divide((i_v8136 * i_v181), jnp.where((i_v8208 == 0.0) | ~jnp.isfinite(i_v8208), 1e-300, i_v8208)))
    i_v8453 = jnp.divide((0.666666666666667 * i_v314), jnp.where((i_v8265 == 0.0) | ~jnp.isfinite(i_v8265), 1e-300, i_v8265))
    i_v8454 = i_v8453 * i_v8453
    i_v8455 = i_v8454 * i_v8454
    i_v8459 = jnp.sqrt(
        jnp.maximum(
            (jnp.divide(i_v8455, jnp.where(((i_v8455 + 1.0) == 0.0) | ~jnp.isfinite((i_v8455 + 1.0)), 1e-300, (i_v8455 + 1.0)))),
            1e-300,
        )
    )
    i_v8460 = jnp.sqrt(jnp.maximum(i_v8459, 1e-300))
    i_v8740 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v8265, i_v8460))), 1e-300))
    i_v8762 = (((2.0 * (i_v8453 * i_v8460)) - i_v8459) - 1.0) * i_v8740
    i_v8764 = i_v8762 > 0.0
    i_v8875 = jnp.where(
        i_v8764,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v8762)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v8762)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v8762)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v8762)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v8762)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v8762)),
                ),
            )
        ),
    )
    i_v8941 = i_v8875 * i_v8875
    i_v8461 = i_v8459 * i_v8460
    i_v8760 = (((i_v314 * i_v8453) * i_v8460) - (i_v314 * i_v8459)) + (0.5 * (i_v8265 * i_v8461))
    i_v8843 = (-(i_v8762 * i_v8762)) + i_v8760
    i_v8856 = -230.25850929940458 - i_v8843
    i_v9012 = (((0.29214664 * i_v8875) + (0.26992878119627894 * i_v8941)) + (0.43792457880372104 * (i_v8941 * i_v8875))) * (
        jnp.where(
            (i_v8843 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v8843, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v8856 * (1.0 + (0.5 * (i_v8856 * (1.0 + (i_v8856 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v8856 * (1.0 + (0.5 * (i_v8856 * (1.0 + (i_v8856 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v8856 * (1.0 + (0.5 * (i_v8856 * (1.0 + (i_v8856 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v9027 = -230.25850929940458 - i_v8760
    i_v8658 = (-PSTI) * i_v188
    i_v8730 = jnp.where(
        (i_v8658 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v8265 * i_v8461)) == 0.0) | ~jnp.isfinite(1.0 + (i_v8265 * i_v8461)),
                    1e-300,
                    (1.0 + (i_v8265 * i_v8461)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v8265 * i_v8461)), 0.0), i_v8658)),
    )
    i_v9483 = i_v188 * (
        jnp.divide(
            ((VBIRSTI - i_v4566) * i_v240),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PSTI == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v4566) * i_v246), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRSTI - i_v4566) * i_v246), 0.0), PSTI)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v4566) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v4566) * i_v246), 0.0), PSTI)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v4566) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v4566) * i_v246), 0.0), PSTI)),
                    )
                ),
            ),
        )
    )
    i_v9686 = jnp.divide((-i_v401), jnp.where((i_v9483 == 0.0) | ~jnp.isfinite(i_v9483), 1e-300, i_v9483))
    i_v9687 = i_v9686 < 0.0
    i_v9712 = -230.25850929940458 - i_v9686
    i_v9733 = i_v9686 - 230.25850929940458
    i_v10139 = i_v5078 * i_v434
    i_v10148 = i_v5078 * i_v434
    i_v13796 = i_v352 * (jnp.divide((i_v13667 * i_v184), jnp.where((i_v13739 == 0.0) | ~jnp.isfinite(i_v13739), 1e-300, i_v13739)))
    i_v14047 = jnp.divide((0.666666666666667 * i_v316), jnp.where((i_v13796 == 0.0) | ~jnp.isfinite(i_v13796), 1e-300, i_v13796))
    i_v14048 = i_v14047 * i_v14047
    i_v14049 = i_v14048 * i_v14048
    i_v14053 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v14049, jnp.where(((i_v14049 + 1.0) == 0.0) | ~jnp.isfinite((i_v14049 + 1.0)), 1e-300, (i_v14049 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v14054 = jnp.sqrt(jnp.maximum(i_v14053, 1e-300))
    i_v14397 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v13796, i_v14054))), 1e-300))
    i_v14419 = (((2.0 * (i_v14047 * i_v14054)) - i_v14053) - 1.0) * i_v14397
    i_v14421 = i_v14419 > 0.0
    i_v14532 = jnp.where(
        i_v14421,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v14419)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v14419)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v14419)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v14419)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v14419)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v14419)),
                ),
            )
        ),
    )
    i_v14598 = i_v14532 * i_v14532
    i_v14055 = i_v14053 * i_v14054
    i_v14417 = (((i_v316 * i_v14047) * i_v14054) - (i_v316 * i_v14053)) + (0.5 * (i_v13796 * i_v14055))
    i_v14500 = (-(i_v14419 * i_v14419)) + i_v14417
    i_v14513 = -230.25850929940458 - i_v14500
    i_v14669 = (((0.29214664 * i_v14532) + (0.26992878119627894 * i_v14598)) + (0.43792457880372104 * (i_v14598 * i_v14532))) * (
        jnp.where(
            (i_v14500 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v14500, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v14513 * (1.0 + (0.5 * (i_v14513 * (1.0 + (i_v14513 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v14513 * (1.0 + (0.5 * (i_v14513 * (1.0 + (i_v14513 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v14513 * (1.0 + (0.5 * (i_v14513 * (1.0 + (i_v14513 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v14684 = -230.25850929940458 - i_v14417
    i_v14315 = (-PGAT) * i_v190
    i_v14387 = jnp.where(
        (i_v14315 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v13796 * i_v14055)) == 0.0) | ~jnp.isfinite(1.0 + (i_v13796 * i_v14055)),
                    1e-300,
                    (1.0 + (i_v13796 * i_v14055)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v13796 * i_v14055)), 0.0), i_v14315)),
    )
    i_v15203 = i_v190 * (
        jnp.divide(
            ((VBIRGAT - i_v4566) * i_v242),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PGAT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v4566) * i_v248), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRGAT - i_v4566) * i_v248), 0.0), PGAT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v4566) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v4566) * i_v248), 0.0), PGAT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v4566) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v4566) * i_v248), 0.0), PGAT)),
                    )
                ),
            ),
        )
    )
    i_v15466 = jnp.divide((-i_v410), jnp.where((i_v15203 == 0.0) | ~jnp.isfinite(i_v15203), 1e-300, i_v15203))
    i_v15467 = i_v15466 < 0.0
    i_v15492 = -230.25850929940458 - i_v15466
    i_v15513 = i_v15466 - 230.25850929940458
    i_v15982 = i_v5078 * i_v437
    i_v15991 = i_v5078 * i_v437
    i_v17278 = (
        (
            i_v465
            * (
                jnp.where(
                    i_v977,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v112 * i_v2824) + (jnp.where(i_v2841, 0.0, (CSRHBOT * (i_v3374 * i_v2949)))))
                                    + (
                                        jnp.where(
                                            (CTATBOT == 0.0),
                                            0.0,
                                            (
                                                CTATBOT
                                                * (
                                                    (
                                                        i_v4497
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v309
                                                                        * (
                                                                            jnp.where(
                                                                                i_v3946,
                                                                                i_v4424,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v3942 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v3942, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v4439
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v4439
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v4439
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v4439
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v4439
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v4439
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v4439
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v4439
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v4439
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v4424
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v3922,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v3877 * i_v3911),
                                                            jnp.where(
                                                                ((i_v3877 + i_v3911) == 0.0) | ~jnp.isfinite((i_v3877 + i_v3911)),
                                                                1e-300,
                                                                (i_v3877 + i_v3911),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTBOT == 0.0),
                                        0.0,
                                        (
                                            CBBTBOT
                                            * (
                                                ((i_v1489 * i_v4808) * i_v4808)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v4952, (-i_v4951), i_v4951)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v4951, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v4952,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v4977
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v4977
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v4977
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v4977
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v4977
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v4977
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v4977
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v4977
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (i_v4977 * 0.3333333333333333)
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v4998
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v4998
                                                                                        * (1.0 + (i_v4998 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRBOT > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v5078 > ((-i_v413) * VBRBOT)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v5426 * i_v5426) * i_v5426) * i_v5426),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (jnp.where((i_v5435 < 0.0), (-i_v5435), i_v5435)),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v5426 * i_v5426) * i_v5426) * i_v5426),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (jnp.where((i_v5435 < 0.0), (-i_v5435), i_v5435)),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRBOT == 4.0),
                                                                (((i_v5426 * i_v5426) * i_v5426) * i_v5426),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v5435 < 0.0), (-i_v5435), i_v5435)), 0.0
                                                                        ),
                                                                        PBRBOT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v418 + ((i_v5078 + (i_v413 * VBRBOT)) * i_v445)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
        + (
            i_v469
            * (
                jnp.where(
                    i_v1012,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v116 * i_v2824) + (jnp.where(i_v7024, 0.0, (CSRHSTI * (i_v7942 * i_v7289)))))
                                    + (
                                        jnp.where(
                                            (CTATSTI == 0.0),
                                            0.0,
                                            (
                                                CTATSTI
                                                * (
                                                    (
                                                        i_v9085
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v314
                                                                        * (
                                                                            jnp.where(
                                                                                i_v8764,
                                                                                i_v9012,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v8760 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v8760, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v9027
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v9027
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v9027
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v9027
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v9027
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v9027
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v9027
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v9027
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v9027
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v9012
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v8740,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v8677 * i_v8730),
                                                            jnp.where(
                                                                ((i_v8677 + i_v8730) == 0.0) | ~jnp.isfinite((i_v8677 + i_v8730)),
                                                                1e-300,
                                                                (i_v8677 + i_v8730),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTSTI == 0.0),
                                        0.0,
                                        (
                                            CBBTSTI
                                            * (
                                                ((i_v1489 * i_v9483) * i_v9483)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v9687, (-i_v9686), i_v9686)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v9686, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v9687,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v9712
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v9712
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v9712
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v9712
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v9712
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v9712
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v9712
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v9712
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (i_v9712 * 0.3333333333333333)
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v9733
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v9733
                                                                                        * (1.0 + (i_v9733 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRSTI > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v5078 > ((-i_v413) * VBRSTI)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRSTI == 4.0),
                                                                    (((i_v10139 * i_v10139) * i_v10139) * i_v10139),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v10148 < 0.0), (-i_v10148), i_v10148
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRSTI,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v10139 * i_v10139) * i_v10139) * i_v10139),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v10148 < 0.0), (-i_v10148), i_v10148)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v10139 * i_v10139) * i_v10139) * i_v10139),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v10148 < 0.0), (-i_v10148), i_v10148)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v423 + ((i_v5078 + (i_v413 * VBRSTI)) * i_v453)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
    ) + (
        i_v473
        * (
            jnp.where(
                i_v1060,
                0.0,
                (
                    (
                        IFACTOR
                        * (
                            (
                                ((i_v120 * i_v2824) + (jnp.where(i_v12234, 0.0, (CSRHGAT * (i_v13410 * i_v12568)))))
                                + (
                                    jnp.where(
                                        (CTATGAT == 0.0),
                                        0.0,
                                        (
                                            CTATGAT
                                            * (
                                                (
                                                    i_v14742
                                                    * (
                                                        0.886226925452758
                                                        * (
                                                            jnp.divide(
                                                                (
                                                                    i_v316
                                                                    * (
                                                                        jnp.where(
                                                                            i_v14421,
                                                                            i_v14669,
                                                                            (
                                                                                (
                                                                                    2.0
                                                                                    * (
                                                                                        jnp.where(
                                                                                            (i_v14417 > -230.25850929940458),
                                                                                            (
                                                                                                jnp.exp(
                                                                                                    jnp.clip(
                                                                                                        i_v14417, -709.0, 709.0
                                                                                                    )
                                                                                                )
                                                                                            ),
                                                                                            (
                                                                                                jnp.divide(
                                                                                                    1e-100,
                                                                                                    jnp.where(
                                                                                                        (
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v14684
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v14684
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v14684
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                            == 0.0
                                                                                                        )
                                                                                                        | ~jnp.isfinite(
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v14684
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v14684
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v14684
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                        1e-300,
                                                                                                        (
                                                                                                            1.0
                                                                                                            + (
                                                                                                                i_v14684
                                                                                                                * (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        0.5
                                                                                                                        * (
                                                                                                                            i_v14684
                                                                                                                            * (
                                                                                                                                1.0
                                                                                                                                + (
                                                                                                                                    i_v14684
                                                                                                                                    * 0.3333333333333333
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                    ),
                                                                                                )
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                                - i_v14669
                                                                            ),
                                                                        )
                                                                    )
                                                                ),
                                                                i_v14397,
                                                            )
                                                        )
                                                    )
                                                )
                                                * (
                                                    jnp.divide(
                                                        (i_v14334 * i_v14387),
                                                        jnp.where(
                                                            ((i_v14334 + i_v14387) == 0.0) | ~jnp.isfinite((i_v14334 + i_v14387)),
                                                            1e-300,
                                                            (i_v14334 + i_v14387),
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            + (
                                jnp.where(
                                    (CBBTGAT == 0.0),
                                    0.0,
                                    (
                                        CBBTGAT
                                        * (
                                            ((i_v1489 * i_v15203) * i_v15203)
                                            * (
                                                jnp.where(
                                                    ((jnp.where(i_v15467, (-i_v15466), i_v15466)) < 230.25850929940458),
                                                    (jnp.exp(jnp.clip(i_v15466, -709.0, 709.0))),
                                                    (
                                                        jnp.where(
                                                            i_v15467,
                                                            (
                                                                jnp.divide(
                                                                    1e-100,
                                                                    jnp.where(
                                                                        (
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v15492
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v15492
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v15492
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                            == 0.0
                                                                        )
                                                                        | ~jnp.isfinite(
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v15492
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v15492
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v15492
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                        1e-300,
                                                                        (
                                                                            1.0
                                                                            + (
                                                                                i_v15492
                                                                                * (
                                                                                    1.0
                                                                                    + (
                                                                                        0.5
                                                                                        * (
                                                                                            i_v15492
                                                                                            * (
                                                                                                1.0
                                                                                                + (i_v15492 * 0.3333333333333333)
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                    ),
                                                                )
                                                            ),
                                                            (
                                                                1e100
                                                                * (
                                                                    1.0
                                                                    + (
                                                                        i_v15513
                                                                        * (
                                                                            1.0
                                                                            + (
                                                                                0.5
                                                                                * (
                                                                                    i_v15513
                                                                                    * (1.0 + (i_v15513 * 0.3333333333333333))
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            ),
                                                        )
                                                    ),
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                    * (
                        jnp.where(
                            (VBRGAT > 1000.0),
                            1.0,
                            (
                                jnp.where(
                                    (i_v5078 > ((-i_v413) * VBRGAT)),
                                    (
                                        jnp.divide(
                                            1.0,
                                            jnp.where(
                                                (
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRGAT == 4.0),
                                                                (((i_v15982 * i_v15982) * i_v15982) * i_v15982),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v15991 < 0.0), (-i_v15991), i_v15991)),
                                                                            0.0,
                                                                        ),
                                                                        PBRGAT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    )
                                                    == 0.0
                                                )
                                                | ~jnp.isfinite(
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v15982 * i_v15982) * i_v15982) * i_v15982),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v15991 < 0.0), (-i_v15991), i_v15991)),
                                                                        0.0,
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                                1e-300,
                                                (
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v15982 * i_v15982) * i_v15982) * i_v15982),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v15991 < 0.0), (-i_v15991), i_v15991)), 0.0
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                            ),
                                        )
                                    ),
                                    (i_v428 + ((i_v5078 + (i_v413 * VBRGAT)) * i_v461)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
    )
    i_v22255 = i_v330 * (jnp.divide((i_v22078 * i_v178), jnp.where((i_v22174 == 0.0) | ~jnp.isfinite(i_v22174), 1e-300, i_v22174)))
    i_v22455 = jnp.divide((0.666666666666667 * i_v309), jnp.where((i_v22255 == 0.0) | ~jnp.isfinite(i_v22255), 1e-300, i_v22255))
    i_v22456 = i_v22455 * i_v22455
    i_v22457 = i_v22456 * i_v22456
    i_v22461 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v22457, jnp.where(((i_v22457 + 1.0) == 0.0) | ~jnp.isfinite((i_v22457 + 1.0)), 1e-300, (i_v22457 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v22462 = jnp.sqrt(jnp.maximum(i_v22461, 1e-300))
    i_v22763 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v22255, i_v22462))), 1e-300))
    i_v22785 = (((2.0 * (i_v22455 * i_v22462)) - i_v22461) - 1.0) * i_v22763
    i_v22787 = i_v22785 > 0.0
    i_v22922 = jnp.where(
        i_v22787,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v22785)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v22785)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v22785)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v22785)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v22785)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v22785)),
                ),
            )
        ),
    )
    i_v23012 = i_v22922 * i_v22922
    i_v22463 = i_v22461 * i_v22462
    i_v22783 = (((i_v309 * i_v22455) * i_v22462) - (i_v309 * i_v22461)) + (0.5 * (i_v22255 * i_v22463))
    i_v22890 = (-(i_v22785 * i_v22785)) + i_v22783
    i_v22903 = -230.25850929940458 - i_v22890
    i_v23107 = (((0.29214664 * i_v22922) + (0.26992878119627894 * i_v23012)) + (0.43792457880372104 * (i_v23012 * i_v22922))) * (
        jnp.where(
            (i_v22890 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v22890, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v22903 * (1.0 + (0.5 * (i_v22903 * (1.0 + (i_v22903 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v22903 * (1.0 + (0.5 * (i_v22903 * (1.0 + (i_v22903 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v22903 * (1.0 + (0.5 * (i_v22903 * (1.0 + (i_v22903 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v23122 = -230.25850929940458 - i_v22783
    i_v22657 = (-PBOT) * i_v186
    i_v22753 = jnp.where(
        (i_v22657 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v22255 * i_v22463)) == 0.0) | ~jnp.isfinite(1.0 + (i_v22255 * i_v22463)),
                    1e-300,
                    (1.0 + (i_v22255 * i_v22463)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v22255 * i_v22463)), 0.0), i_v22657)),
    )
    i_v23279 = jnp.where(
        i_v17290,
        (
            0.5
            * (
                (i_v1492 + i_v1311)
                - (jnp.sqrt(jnp.maximum((((i_v1492 - i_v1311) * (i_v1492 - i_v1311)) + ((4.0 * i_v48) * i_v48)), 1e-300)))
            )
        ),
        0.0,
    )
    i_v23581 = i_v186 * (
        jnp.divide(
            ((VBIRBOT - i_v23279) * i_v238),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PBOT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v23279) * i_v244), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRBOT - i_v23279) * i_v244), 0.0), PBOT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v23279) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v23279) * i_v244), 0.0), PBOT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v23279) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v23279) * i_v244), 0.0), PBOT)),
                    )
                ),
            ),
        )
    )
    i_v23789 = jnp.divide((-i_v395), jnp.where((i_v23581 == 0.0) | ~jnp.isfinite(i_v23581), 1e-300, i_v23581))
    i_v23790 = i_v23789 < 0.0
    i_v23815 = -230.25850929940458 - i_v23789
    i_v23836 = i_v23789 - 230.25850929940458
    i_v24274 = i_v23915 * i_v431
    i_v24283 = i_v23915 * i_v431
    i_v27576 = i_v341 * (jnp.divide((i_v27447 * i_v181), jnp.where((i_v27519 == 0.0) | ~jnp.isfinite(i_v27519), 1e-300, i_v27519)))
    i_v27776 = jnp.divide((0.666666666666667 * i_v314), jnp.where((i_v27576 == 0.0) | ~jnp.isfinite(i_v27576), 1e-300, i_v27576))
    i_v27777 = i_v27776 * i_v27776
    i_v27778 = i_v27777 * i_v27777
    i_v27782 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v27778, jnp.where(((i_v27778 + 1.0) == 0.0) | ~jnp.isfinite((i_v27778 + 1.0)), 1e-300, (i_v27778 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v27783 = jnp.sqrt(jnp.maximum(i_v27782, 1e-300))
    i_v28060 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v27576, i_v27783))), 1e-300))
    i_v28082 = (((2.0 * (i_v27776 * i_v27783)) - i_v27782) - 1.0) * i_v28060
    i_v28084 = i_v28082 > 0.0
    i_v28195 = jnp.where(
        i_v28084,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v28082)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v28082)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v28082)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v28082)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v28082)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v28082)),
                ),
            )
        ),
    )
    i_v28261 = i_v28195 * i_v28195
    i_v27784 = i_v27782 * i_v27783
    i_v28080 = (((i_v314 * i_v27776) * i_v27783) - (i_v314 * i_v27782)) + (0.5 * (i_v27576 * i_v27784))
    i_v28163 = (-(i_v28082 * i_v28082)) + i_v28080
    i_v28176 = -230.25850929940458 - i_v28163
    i_v28332 = (((0.29214664 * i_v28195) + (0.26992878119627894 * i_v28261)) + (0.43792457880372104 * (i_v28261 * i_v28195))) * (
        jnp.where(
            (i_v28163 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v28163, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v28176 * (1.0 + (0.5 * (i_v28176 * (1.0 + (i_v28176 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v28176 * (1.0 + (0.5 * (i_v28176 * (1.0 + (i_v28176 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v28176 * (1.0 + (0.5 * (i_v28176 * (1.0 + (i_v28176 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v28347 = -230.25850929940458 - i_v28080
    i_v27978 = (-PSTI) * i_v188
    i_v28050 = jnp.where(
        (i_v27978 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v27576 * i_v27784)) == 0.0) | ~jnp.isfinite(1.0 + (i_v27576 * i_v27784)),
                    1e-300,
                    (1.0 + (i_v27576 * i_v27784)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v27576 * i_v27784)), 0.0), i_v27978)),
    )
    i_v28799 = i_v188 * (
        jnp.divide(
            ((VBIRSTI - i_v23279) * i_v240),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PSTI == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v23279) * i_v246), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRSTI - i_v23279) * i_v246), 0.0), PSTI)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v23279) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v23279) * i_v246), 0.0), PSTI)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v23279) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v23279) * i_v246), 0.0), PSTI)),
                    )
                ),
            ),
        )
    )
    i_v29007 = jnp.divide((-i_v401), jnp.where((i_v28799 == 0.0) | ~jnp.isfinite(i_v28799), 1e-300, i_v28799))
    i_v29008 = i_v29007 < 0.0
    i_v29033 = -230.25850929940458 - i_v29007
    i_v29054 = i_v29007 - 230.25850929940458
    i_v29460 = i_v23915 * i_v434
    i_v29469 = i_v23915 * i_v434
    i_v32822 = i_v352 * (jnp.divide((i_v32693 * i_v184), jnp.where((i_v32765 == 0.0) | ~jnp.isfinite(i_v32765), 1e-300, i_v32765)))
    i_v33022 = jnp.divide((0.666666666666667 * i_v316), jnp.where((i_v32822 == 0.0) | ~jnp.isfinite(i_v32822), 1e-300, i_v32822))
    i_v33023 = i_v33022 * i_v33022
    i_v33024 = i_v33023 * i_v33023
    i_v33028 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v33024, jnp.where(((i_v33024 + 1.0) == 0.0) | ~jnp.isfinite((i_v33024 + 1.0)), 1e-300, (i_v33024 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v33029 = jnp.sqrt(jnp.maximum(i_v33028, 1e-300))
    i_v33306 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v32822, i_v33029))), 1e-300))
    i_v33328 = (((2.0 * (i_v33022 * i_v33029)) - i_v33028) - 1.0) * i_v33306
    i_v33330 = i_v33328 > 0.0
    i_v33441 = jnp.where(
        i_v33330,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v33328)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v33328)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v33328)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v33328)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v33328)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v33328)),
                ),
            )
        ),
    )
    i_v33507 = i_v33441 * i_v33441
    i_v33030 = i_v33028 * i_v33029
    i_v33326 = (((i_v316 * i_v33022) * i_v33029) - (i_v316 * i_v33028)) + (0.5 * (i_v32822 * i_v33030))
    i_v33409 = (-(i_v33328 * i_v33328)) + i_v33326
    i_v33422 = -230.25850929940458 - i_v33409
    i_v33578 = (((0.29214664 * i_v33441) + (0.26992878119627894 * i_v33507)) + (0.43792457880372104 * (i_v33507 * i_v33441))) * (
        jnp.where(
            (i_v33409 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v33409, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v33422 * (1.0 + (0.5 * (i_v33422 * (1.0 + (i_v33422 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v33422 * (1.0 + (0.5 * (i_v33422 * (1.0 + (i_v33422 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v33422 * (1.0 + (0.5 * (i_v33422 * (1.0 + (i_v33422 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v33593 = -230.25850929940458 - i_v33326
    i_v33224 = (-PGAT) * i_v190
    i_v33296 = jnp.where(
        (i_v33224 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v32822 * i_v33030)) == 0.0) | ~jnp.isfinite(1.0 + (i_v32822 * i_v33030)),
                    1e-300,
                    (1.0 + (i_v32822 * i_v33030)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v32822 * i_v33030)), 0.0), i_v33224)),
    )
    i_v34045 = i_v190 * (
        jnp.divide(
            ((VBIRGAT - i_v23279) * i_v242),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PGAT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v23279) * i_v248), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRGAT - i_v23279) * i_v248), 0.0), PGAT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v23279) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v23279) * i_v248), 0.0), PGAT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v23279) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v23279) * i_v248), 0.0), PGAT)),
                    )
                ),
            ),
        )
    )
    i_v34253 = jnp.divide((-i_v410), jnp.where((i_v34045 == 0.0) | ~jnp.isfinite(i_v34045), 1e-300, i_v34045))
    i_v34254 = i_v34253 < 0.0
    i_v34279 = -230.25850929940458 - i_v34253
    i_v34300 = i_v34253 - 230.25850929940458
    i_v34706 = i_v23915 * i_v437
    i_v34715 = i_v23915 * i_v437
    i_v35880 = (
        (
            i_v465
            * (
                jnp.where(
                    i_v977,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v112 * i_v20760) + (jnp.where(i_v20834, 0.0, (CSRHBOT * (i_v21857 * i_v21154)))))
                                    + (
                                        jnp.where(
                                            (CTATBOT == 0.0),
                                            0.0,
                                            (
                                                CTATBOT
                                                * (
                                                    (
                                                        i_v23180
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v309
                                                                        * (
                                                                            jnp.where(
                                                                                i_v22787,
                                                                                i_v23107,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v22783 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v22783, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v23122
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v23122
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v23122
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v23122
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v23122
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v23122
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v23122
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v23122
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v23122
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v23107
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v22763,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v22676 * i_v22753),
                                                            jnp.where(
                                                                ((i_v22676 + i_v22753) == 0.0)
                                                                | ~jnp.isfinite((i_v22676 + i_v22753)),
                                                                1e-300,
                                                                (i_v22676 + i_v22753),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTBOT == 0.0),
                                        0.0,
                                        (
                                            CBBTBOT
                                            * (
                                                ((i_v1492 * i_v23581) * i_v23581)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v23790, (-i_v23789), i_v23789)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v23789, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v23790,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v23815
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v23815
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v23815
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v23815
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v23815
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v23815
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v23815
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v23815
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v23815
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v23836
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v23836
                                                                                        * (1.0 + (i_v23836 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRBOT > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v23915 > ((-i_v413) * VBRBOT)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v24274 * i_v24274) * i_v24274) * i_v24274),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v24283 < 0.0), (-i_v24283), i_v24283
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v24274 * i_v24274) * i_v24274) * i_v24274),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v24283 < 0.0), (-i_v24283), i_v24283
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRBOT == 4.0),
                                                                (((i_v24274 * i_v24274) * i_v24274) * i_v24274),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v24283 < 0.0), (-i_v24283), i_v24283)),
                                                                            0.0,
                                                                        ),
                                                                        PBRBOT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v418 + ((i_v23915 + (i_v413 * VBRBOT)) * i_v445)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
        + (
            i_v469
            * (
                jnp.where(
                    i_v1012,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v116 * i_v20760) + (jnp.where(i_v26174, 0.0, (CSRHSTI * (i_v27226 * i_v26533)))))
                                    + (
                                        jnp.where(
                                            (CTATSTI == 0.0),
                                            0.0,
                                            (
                                                CTATSTI
                                                * (
                                                    (
                                                        i_v28405
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v314
                                                                        * (
                                                                            jnp.where(
                                                                                i_v28084,
                                                                                i_v28332,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v28080 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v28080, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v28347
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v28347
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v28347
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v28347
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v28347
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v28347
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v28347
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v28347
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v28347
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v28332
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v28060,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v27997 * i_v28050),
                                                            jnp.where(
                                                                ((i_v27997 + i_v28050) == 0.0)
                                                                | ~jnp.isfinite((i_v27997 + i_v28050)),
                                                                1e-300,
                                                                (i_v27997 + i_v28050),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTSTI == 0.0),
                                        0.0,
                                        (
                                            CBBTSTI
                                            * (
                                                ((i_v1492 * i_v28799) * i_v28799)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v29008, (-i_v29007), i_v29007)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v29007, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v29008,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v29033
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v29033
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v29033
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v29033
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v29033
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v29033
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v29033
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v29033
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v29033
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v29054
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v29054
                                                                                        * (1.0 + (i_v29054 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRSTI > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v23915 > ((-i_v413) * VBRSTI)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRSTI == 4.0),
                                                                    (((i_v29460 * i_v29460) * i_v29460) * i_v29460),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v29469 < 0.0), (-i_v29469), i_v29469
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRSTI,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v29460 * i_v29460) * i_v29460) * i_v29460),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v29469 < 0.0), (-i_v29469), i_v29469)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v29460 * i_v29460) * i_v29460) * i_v29460),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v29469 < 0.0), (-i_v29469), i_v29469)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v423 + ((i_v23915 + (i_v413 * VBRSTI)) * i_v453)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
    ) + (
        i_v473
        * (
            jnp.where(
                i_v1060,
                0.0,
                (
                    (
                        IFACTOR
                        * (
                            (
                                ((i_v120 * i_v20760) + (jnp.where(i_v31420, 0.0, (CSRHGAT * (i_v32472 * i_v31779)))))
                                + (
                                    jnp.where(
                                        (CTATGAT == 0.0),
                                        0.0,
                                        (
                                            CTATGAT
                                            * (
                                                (
                                                    i_v33651
                                                    * (
                                                        0.886226925452758
                                                        * (
                                                            jnp.divide(
                                                                (
                                                                    i_v316
                                                                    * (
                                                                        jnp.where(
                                                                            i_v33330,
                                                                            i_v33578,
                                                                            (
                                                                                (
                                                                                    2.0
                                                                                    * (
                                                                                        jnp.where(
                                                                                            (i_v33326 > -230.25850929940458),
                                                                                            (
                                                                                                jnp.exp(
                                                                                                    jnp.clip(
                                                                                                        i_v33326, -709.0, 709.0
                                                                                                    )
                                                                                                )
                                                                                            ),
                                                                                            (
                                                                                                jnp.divide(
                                                                                                    1e-100,
                                                                                                    jnp.where(
                                                                                                        (
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v33593
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v33593
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v33593
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                            == 0.0
                                                                                                        )
                                                                                                        | ~jnp.isfinite(
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v33593
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v33593
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v33593
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                        1e-300,
                                                                                                        (
                                                                                                            1.0
                                                                                                            + (
                                                                                                                i_v33593
                                                                                                                * (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        0.5
                                                                                                                        * (
                                                                                                                            i_v33593
                                                                                                                            * (
                                                                                                                                1.0
                                                                                                                                + (
                                                                                                                                    i_v33593
                                                                                                                                    * 0.3333333333333333
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                    ),
                                                                                                )
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                                - i_v33578
                                                                            ),
                                                                        )
                                                                    )
                                                                ),
                                                                i_v33306,
                                                            )
                                                        )
                                                    )
                                                )
                                                * (
                                                    jnp.divide(
                                                        (i_v33243 * i_v33296),
                                                        jnp.where(
                                                            ((i_v33243 + i_v33296) == 0.0) | ~jnp.isfinite((i_v33243 + i_v33296)),
                                                            1e-300,
                                                            (i_v33243 + i_v33296),
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            + (
                                jnp.where(
                                    (CBBTGAT == 0.0),
                                    0.0,
                                    (
                                        CBBTGAT
                                        * (
                                            ((i_v1492 * i_v34045) * i_v34045)
                                            * (
                                                jnp.where(
                                                    ((jnp.where(i_v34254, (-i_v34253), i_v34253)) < 230.25850929940458),
                                                    (jnp.exp(jnp.clip(i_v34253, -709.0, 709.0))),
                                                    (
                                                        jnp.where(
                                                            i_v34254,
                                                            (
                                                                jnp.divide(
                                                                    1e-100,
                                                                    jnp.where(
                                                                        (
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v34279
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v34279
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v34279
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                            == 0.0
                                                                        )
                                                                        | ~jnp.isfinite(
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v34279
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v34279
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v34279
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                        1e-300,
                                                                        (
                                                                            1.0
                                                                            + (
                                                                                i_v34279
                                                                                * (
                                                                                    1.0
                                                                                    + (
                                                                                        0.5
                                                                                        * (
                                                                                            i_v34279
                                                                                            * (
                                                                                                1.0
                                                                                                + (i_v34279 * 0.3333333333333333)
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                    ),
                                                                )
                                                            ),
                                                            (
                                                                1e100
                                                                * (
                                                                    1.0
                                                                    + (
                                                                        i_v34300
                                                                        * (
                                                                            1.0
                                                                            + (
                                                                                0.5
                                                                                * (
                                                                                    i_v34300
                                                                                    * (1.0 + (i_v34300 * 0.3333333333333333))
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            ),
                                                        )
                                                    ),
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                    * (
                        jnp.where(
                            (VBRGAT > 1000.0),
                            1.0,
                            (
                                jnp.where(
                                    (i_v23915 > ((-i_v413) * VBRGAT)),
                                    (
                                        jnp.divide(
                                            1.0,
                                            jnp.where(
                                                (
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRGAT == 4.0),
                                                                (((i_v34706 * i_v34706) * i_v34706) * i_v34706),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v34715 < 0.0), (-i_v34715), i_v34715)),
                                                                            0.0,
                                                                        ),
                                                                        PBRGAT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    )
                                                    == 0.0
                                                )
                                                | ~jnp.isfinite(
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v34706 * i_v34706) * i_v34706) * i_v34706),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v34715 < 0.0), (-i_v34715), i_v34715)),
                                                                        0.0,
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                                1e-300,
                                                (
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v34706 * i_v34706) * i_v34706) * i_v34706),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v34715 < 0.0), (-i_v34715), i_v34715)), 0.0
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                            ),
                                        )
                                    ),
                                    (i_v428 + ((i_v23915 + (i_v413 * VBRGAT)) * i_v461)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
    )
    i_v41070 = i_v330 * (jnp.divide((i_v40893 * i_v178), jnp.where((i_v40989 == 0.0) | ~jnp.isfinite(i_v40989), 1e-300, i_v40989)))
    i_v41270 = jnp.divide((0.666666666666667 * i_v309), jnp.where((i_v41070 == 0.0) | ~jnp.isfinite(i_v41070), 1e-300, i_v41070))
    i_v41271 = i_v41270 * i_v41270
    i_v41272 = i_v41271 * i_v41271
    i_v41276 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v41272, jnp.where(((i_v41272 + 1.0) == 0.0) | ~jnp.isfinite((i_v41272 + 1.0)), 1e-300, (i_v41272 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v41277 = jnp.sqrt(jnp.maximum(i_v41276, 1e-300))
    i_v41578 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v41070, i_v41277))), 1e-300))
    i_v41600 = (((2.0 * (i_v41270 * i_v41277)) - i_v41276) - 1.0) * i_v41578
    i_v41602 = i_v41600 > 0.0
    i_v41737 = jnp.where(
        i_v41602,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v41600)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v41600)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v41600)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v41600)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v41600)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v41600)),
                ),
            )
        ),
    )
    i_v41827 = i_v41737 * i_v41737
    i_v41278 = i_v41276 * i_v41277
    i_v41598 = (((i_v309 * i_v41270) * i_v41277) - (i_v309 * i_v41276)) + (0.5 * (i_v41070 * i_v41278))
    i_v41705 = (-(i_v41600 * i_v41600)) + i_v41598
    i_v41718 = -230.25850929940458 - i_v41705
    i_v41922 = (((0.29214664 * i_v41737) + (0.26992878119627894 * i_v41827)) + (0.43792457880372104 * (i_v41827 * i_v41737))) * (
        jnp.where(
            (i_v41705 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v41705, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v41718 * (1.0 + (0.5 * (i_v41718 * (1.0 + (i_v41718 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v41718 * (1.0 + (0.5 * (i_v41718 * (1.0 + (i_v41718 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v41718 * (1.0 + (0.5 * (i_v41718 * (1.0 + (i_v41718 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v41937 = -230.25850929940458 - i_v41598
    i_v41472 = (-PBOT) * i_v186
    i_v41568 = jnp.where(
        (i_v41472 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v41070 * i_v41278)) == 0.0) | ~jnp.isfinite(1.0 + (i_v41070 * i_v41278)),
                    1e-300,
                    (1.0 + (i_v41070 * i_v41278)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v41070 * i_v41278)), 0.0), i_v41472)),
    )
    i_v42094 = jnp.where(
        i_v35892,
        (
            0.5
            * (
                (i_v1495 + i_v1311)
                - (jnp.sqrt(jnp.maximum((((i_v1495 - i_v1311) * (i_v1495 - i_v1311)) + ((4.0 * i_v48) * i_v48)), 1e-300)))
            )
        ),
        0.0,
    )
    i_v42396 = i_v186 * (
        jnp.divide(
            ((VBIRBOT - i_v42094) * i_v238),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PBOT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v42094) * i_v244), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRBOT - i_v42094) * i_v244), 0.0), PBOT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v42094) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v42094) * i_v244), 0.0), PBOT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PBOT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRBOT - i_v42094) * i_v244), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRBOT - i_v42094) * i_v244), 0.0), PBOT)),
                    )
                ),
            ),
        )
    )
    i_v42604 = jnp.divide((-i_v395), jnp.where((i_v42396 == 0.0) | ~jnp.isfinite(i_v42396), 1e-300, i_v42396))
    i_v42605 = i_v42604 < 0.0
    i_v42630 = -230.25850929940458 - i_v42604
    i_v42651 = i_v42604 - 230.25850929940458
    i_v43089 = i_v42730 * i_v431
    i_v43098 = i_v42730 * i_v431
    i_v46391 = i_v341 * (jnp.divide((i_v46262 * i_v181), jnp.where((i_v46334 == 0.0) | ~jnp.isfinite(i_v46334), 1e-300, i_v46334)))
    i_v46591 = jnp.divide((0.666666666666667 * i_v314), jnp.where((i_v46391 == 0.0) | ~jnp.isfinite(i_v46391), 1e-300, i_v46391))
    i_v46592 = i_v46591 * i_v46591
    i_v46593 = i_v46592 * i_v46592
    i_v46597 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v46593, jnp.where(((i_v46593 + 1.0) == 0.0) | ~jnp.isfinite((i_v46593 + 1.0)), 1e-300, (i_v46593 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v46598 = jnp.sqrt(jnp.maximum(i_v46597, 1e-300))
    i_v46875 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v46391, i_v46598))), 1e-300))
    i_v46897 = (((2.0 * (i_v46591 * i_v46598)) - i_v46597) - 1.0) * i_v46875
    i_v46899 = i_v46897 > 0.0
    i_v47010 = jnp.where(
        i_v46899,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v46897)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v46897)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v46897)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v46897)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v46897)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v46897)),
                ),
            )
        ),
    )
    i_v47076 = i_v47010 * i_v47010
    i_v46599 = i_v46597 * i_v46598
    i_v46895 = (((i_v314 * i_v46591) * i_v46598) - (i_v314 * i_v46597)) + (0.5 * (i_v46391 * i_v46599))
    i_v46978 = (-(i_v46897 * i_v46897)) + i_v46895
    i_v46991 = -230.25850929940458 - i_v46978
    i_v47147 = (((0.29214664 * i_v47010) + (0.26992878119627894 * i_v47076)) + (0.43792457880372104 * (i_v47076 * i_v47010))) * (
        jnp.where(
            (i_v46978 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v46978, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v46991 * (1.0 + (0.5 * (i_v46991 * (1.0 + (i_v46991 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v46991 * (1.0 + (0.5 * (i_v46991 * (1.0 + (i_v46991 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v46991 * (1.0 + (0.5 * (i_v46991 * (1.0 + (i_v46991 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v47162 = -230.25850929940458 - i_v46895
    i_v46793 = (-PSTI) * i_v188
    i_v46865 = jnp.where(
        (i_v46793 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v46391 * i_v46599)) == 0.0) | ~jnp.isfinite(1.0 + (i_v46391 * i_v46599)),
                    1e-300,
                    (1.0 + (i_v46391 * i_v46599)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v46391 * i_v46599)), 0.0), i_v46793)),
    )
    i_v47614 = i_v188 * (
        jnp.divide(
            ((VBIRSTI - i_v42094) * i_v240),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PSTI == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v42094) * i_v246), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRSTI - i_v42094) * i_v246), 0.0), PSTI)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v42094) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v42094) * i_v246), 0.0), PSTI)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PSTI == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRSTI - i_v42094) * i_v246), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRSTI - i_v42094) * i_v246), 0.0), PSTI)),
                    )
                ),
            ),
        )
    )
    i_v47822 = jnp.divide((-i_v401), jnp.where((i_v47614 == 0.0) | ~jnp.isfinite(i_v47614), 1e-300, i_v47614))
    i_v47823 = i_v47822 < 0.0
    i_v47848 = -230.25850929940458 - i_v47822
    i_v47869 = i_v47822 - 230.25850929940458
    i_v48275 = i_v42730 * i_v434
    i_v48284 = i_v42730 * i_v434
    i_v51637 = i_v352 * (jnp.divide((i_v51508 * i_v184), jnp.where((i_v51580 == 0.0) | ~jnp.isfinite(i_v51580), 1e-300, i_v51580)))
    i_v51837 = jnp.divide((0.666666666666667 * i_v316), jnp.where((i_v51637 == 0.0) | ~jnp.isfinite(i_v51637), 1e-300, i_v51637))
    i_v51838 = i_v51837 * i_v51837
    i_v51839 = i_v51838 * i_v51838
    i_v51843 = jnp.sqrt(
        jnp.maximum(
            (
                jnp.divide(
                    i_v51839, jnp.where(((i_v51839 + 1.0) == 0.0) | ~jnp.isfinite((i_v51839 + 1.0)), 1e-300, (i_v51839 + 1.0))
                )
            ),
            1e-300,
        )
    )
    i_v51844 = jnp.sqrt(jnp.maximum(i_v51843, 1e-300))
    i_v52121 = jnp.sqrt(jnp.maximum((0.375 * (jnp.divide(i_v51637, i_v51844))), 1e-300))
    i_v52143 = (((2.0 * (i_v51837 * i_v51844)) - i_v51843) - 1.0) * i_v52121
    i_v52145 = i_v52143 > 0.0
    i_v52256 = jnp.where(
        i_v52145,
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (0.5178164370971076 * i_v52143)) == 0.0) | ~jnp.isfinite(1.0 + (0.5178164370971076 * i_v52143)),
                    1e-300,
                    (1.0 + (0.5178164370971076 * i_v52143)),
                ),
            )
        ),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 - (0.5178164370971076 * i_v52143)) == 0.0) | ~jnp.isfinite(1.0 - (0.5178164370971076 * i_v52143)),
                    1e-300,
                    (1.0 - (0.5178164370971076 * i_v52143)),
                ),
            )
        ),
    )
    i_v52322 = i_v52256 * i_v52256
    i_v51845 = i_v51843 * i_v51844
    i_v52141 = (((i_v316 * i_v51837) * i_v51844) - (i_v316 * i_v51843)) + (0.5 * (i_v51637 * i_v51845))
    i_v52224 = (-(i_v52143 * i_v52143)) + i_v52141
    i_v52237 = -230.25850929940458 - i_v52224
    i_v52393 = (((0.29214664 * i_v52256) + (0.26992878119627894 * i_v52322)) + (0.43792457880372104 * (i_v52322 * i_v52256))) * (
        jnp.where(
            (i_v52224 > -230.25850929940458),
            (jnp.exp(jnp.clip(i_v52224, -709.0, 709.0))),
            (
                jnp.divide(
                    1e-100,
                    jnp.where(
                        ((1.0 + (i_v52237 * (1.0 + (0.5 * (i_v52237 * (1.0 + (i_v52237 * 0.3333333333333333))))))) == 0.0)
                        | ~jnp.isfinite(1.0 + (i_v52237 * (1.0 + (0.5 * (i_v52237 * (1.0 + (i_v52237 * 0.3333333333333333))))))),
                        1e-300,
                        (1.0 + (i_v52237 * (1.0 + (0.5 * (i_v52237 * (1.0 + (i_v52237 * 0.3333333333333333))))))),
                    ),
                )
            ),
        )
    )
    i_v52408 = -230.25850929940458 - i_v52141
    i_v52039 = (-PGAT) * i_v190
    i_v52111 = jnp.where(
        (i_v52039 == -1.0),
        (
            jnp.divide(
                1.0,
                jnp.where(
                    ((1.0 + (i_v51637 * i_v51845)) == 0.0) | ~jnp.isfinite(1.0 + (i_v51637 * i_v51845)),
                    1e-300,
                    (1.0 + (i_v51637 * i_v51845)),
                ),
            )
        ),
        (jnp.power(jnp.maximum((1.0 + (i_v51637 * i_v51845)), 0.0), i_v52039)),
    )
    i_v52860 = i_v190 * (
        jnp.divide(
            ((VBIRGAT - i_v42094) * i_v242),
            jnp.where(
                (
                    (
                        jnp.where(
                            (PGAT == 0.5),
                            (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v42094) * i_v248), 1e-300))),
                            (jnp.power(jnp.maximum(((VBIRGAT - i_v42094) * i_v248), 0.0), PGAT)),
                        )
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v42094) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v42094) * i_v248), 0.0), PGAT)),
                    )
                ),
                1e-300,
                (
                    jnp.where(
                        (PGAT == 0.5),
                        (jnp.sqrt(jnp.maximum(((VBIRGAT - i_v42094) * i_v248), 1e-300))),
                        (jnp.power(jnp.maximum(((VBIRGAT - i_v42094) * i_v248), 0.0), PGAT)),
                    )
                ),
            ),
        )
    )
    i_v53068 = jnp.divide((-i_v410), jnp.where((i_v52860 == 0.0) | ~jnp.isfinite(i_v52860), 1e-300, i_v52860))
    i_v53069 = i_v53068 < 0.0
    i_v53094 = -230.25850929940458 - i_v53068
    i_v53115 = i_v53068 - 230.25850929940458
    i_v53521 = i_v42730 * i_v437
    i_v53530 = i_v42730 * i_v437
    i_v54695 = (
        (
            i_v465
            * (
                jnp.where(
                    i_v977,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v112 * i_v39575) + (jnp.where(i_v39649, 0.0, (CSRHBOT * (i_v40672 * i_v39969)))))
                                    + (
                                        jnp.where(
                                            (CTATBOT == 0.0),
                                            0.0,
                                            (
                                                CTATBOT
                                                * (
                                                    (
                                                        i_v41995
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v309
                                                                        * (
                                                                            jnp.where(
                                                                                i_v41602,
                                                                                i_v41922,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v41598 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v41598, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v41937
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v41937
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v41937
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v41937
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v41937
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v41937
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v41937
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v41937
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v41937
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v41922
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v41578,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v41491 * i_v41568),
                                                            jnp.where(
                                                                ((i_v41491 + i_v41568) == 0.0)
                                                                | ~jnp.isfinite((i_v41491 + i_v41568)),
                                                                1e-300,
                                                                (i_v41491 + i_v41568),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTBOT == 0.0),
                                        0.0,
                                        (
                                            CBBTBOT
                                            * (
                                                ((i_v1495 * i_v42396) * i_v42396)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v42605, (-i_v42604), i_v42604)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v42604, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v42605,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v42630
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v42630
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v42630
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v42630
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v42630
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v42630
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v42630
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v42630
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v42630
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v42651
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v42651
                                                                                        * (1.0 + (i_v42651 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRBOT > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v42730 > ((-i_v413) * VBRBOT)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v43089 * i_v43089) * i_v43089) * i_v43089),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v43098 < 0.0), (-i_v43098), i_v43098
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRBOT == 4.0),
                                                                    (((i_v43089 * i_v43089) * i_v43089) * i_v43089),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v43098 < 0.0), (-i_v43098), i_v43098
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRBOT,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRBOT == 4.0),
                                                                (((i_v43089 * i_v43089) * i_v43089) * i_v43089),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v43098 < 0.0), (-i_v43098), i_v43098)),
                                                                            0.0,
                                                                        ),
                                                                        PBRBOT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v418 + ((i_v42730 + (i_v413 * VBRBOT)) * i_v445)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
        + (
            i_v469
            * (
                jnp.where(
                    i_v1012,
                    0.0,
                    (
                        (
                            IFACTOR
                            * (
                                (
                                    ((i_v116 * i_v39575) + (jnp.where(i_v44989, 0.0, (CSRHSTI * (i_v46041 * i_v45348)))))
                                    + (
                                        jnp.where(
                                            (CTATSTI == 0.0),
                                            0.0,
                                            (
                                                CTATSTI
                                                * (
                                                    (
                                                        i_v47220
                                                        * (
                                                            0.886226925452758
                                                            * (
                                                                jnp.divide(
                                                                    (
                                                                        i_v314
                                                                        * (
                                                                            jnp.where(
                                                                                i_v46899,
                                                                                i_v47147,
                                                                                (
                                                                                    (
                                                                                        2.0
                                                                                        * (
                                                                                            jnp.where(
                                                                                                (i_v46895 > -230.25850929940458),
                                                                                                (
                                                                                                    jnp.exp(
                                                                                                        jnp.clip(
                                                                                                            i_v46895, -709.0, 709.0
                                                                                                        )
                                                                                                    )
                                                                                                ),
                                                                                                (
                                                                                                    jnp.divide(
                                                                                                        1e-100,
                                                                                                        jnp.where(
                                                                                                            (
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v47162
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v47162
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v47162
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                                == 0.0
                                                                                                            )
                                                                                                            | ~jnp.isfinite(
                                                                                                                (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        i_v47162
                                                                                                                        * (
                                                                                                                            1.0
                                                                                                                            + (
                                                                                                                                0.5
                                                                                                                                * (
                                                                                                                                    i_v47162
                                                                                                                                    * (
                                                                                                                                        1.0
                                                                                                                                        + (
                                                                                                                                            i_v47162
                                                                                                                                            * 0.3333333333333333
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                            1e-300,
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v47162
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v47162
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v47162
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            ),
                                                                                                        ),
                                                                                                    )
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    - i_v47147
                                                                                ),
                                                                            )
                                                                        )
                                                                    ),
                                                                    i_v46875,
                                                                )
                                                            )
                                                        )
                                                    )
                                                    * (
                                                        jnp.divide(
                                                            (i_v46812 * i_v46865),
                                                            jnp.where(
                                                                ((i_v46812 + i_v46865) == 0.0)
                                                                | ~jnp.isfinite((i_v46812 + i_v46865)),
                                                                1e-300,
                                                                (i_v46812 + i_v46865),
                                                            ),
                                                        )
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                )
                                + (
                                    jnp.where(
                                        (CBBTSTI == 0.0),
                                        0.0,
                                        (
                                            CBBTSTI
                                            * (
                                                ((i_v1495 * i_v47614) * i_v47614)
                                                * (
                                                    jnp.where(
                                                        ((jnp.where(i_v47823, (-i_v47822), i_v47822)) < 230.25850929940458),
                                                        (jnp.exp(jnp.clip(i_v47822, -709.0, 709.0))),
                                                        (
                                                            jnp.where(
                                                                i_v47823,
                                                                (
                                                                    jnp.divide(
                                                                        1e-100,
                                                                        jnp.where(
                                                                            (
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v47848
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v47848
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v47848
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                                == 0.0
                                                                            )
                                                                            | ~jnp.isfinite(
                                                                                (
                                                                                    1.0
                                                                                    + (
                                                                                        i_v47848
                                                                                        * (
                                                                                            1.0
                                                                                            + (
                                                                                                0.5
                                                                                                * (
                                                                                                    i_v47848
                                                                                                    * (
                                                                                                        1.0
                                                                                                        + (
                                                                                                            i_v47848
                                                                                                            * 0.3333333333333333
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                            1e-300,
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v47848
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v47848
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v47848
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        ),
                                                                    )
                                                                ),
                                                                (
                                                                    1e100
                                                                    * (
                                                                        1.0
                                                                        + (
                                                                            i_v47869
                                                                            * (
                                                                                1.0
                                                                                + (
                                                                                    0.5
                                                                                    * (
                                                                                        i_v47869
                                                                                        * (1.0 + (i_v47869 * 0.3333333333333333))
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                        )
                        * (
                            jnp.where(
                                (VBRSTI > 1000.0),
                                1.0,
                                (
                                    jnp.where(
                                        (i_v42730 > ((-i_v413) * VBRSTI)),
                                        (
                                            jnp.divide(
                                                1.0,
                                                jnp.where(
                                                    (
                                                        (
                                                            1.0
                                                            - (
                                                                jnp.where(
                                                                    (PBRSTI == 4.0),
                                                                    (((i_v48275 * i_v48275) * i_v48275) * i_v48275),
                                                                    (
                                                                        jnp.power(
                                                                            jnp.maximum(
                                                                                (
                                                                                    jnp.where(
                                                                                        (i_v48284 < 0.0), (-i_v48284), i_v48284
                                                                                    )
                                                                                ),
                                                                                0.0,
                                                                            ),
                                                                            PBRSTI,
                                                                        )
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        == 0.0
                                                    )
                                                    | ~jnp.isfinite(
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v48275 * i_v48275) * i_v48275) * i_v48275),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v48284 < 0.0), (-i_v48284), i_v48284)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                    1e-300,
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRSTI == 4.0),
                                                                (((i_v48275 * i_v48275) * i_v48275) * i_v48275),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v48284 < 0.0), (-i_v48284), i_v48284)),
                                                                            0.0,
                                                                        ),
                                                                        PBRSTI,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                ),
                                            )
                                        ),
                                        (i_v423 + ((i_v42730 + (i_v413 * VBRSTI)) * i_v453)),
                                    )
                                ),
                            )
                        )
                    ),
                )
            )
        )
    ) + (
        i_v473
        * (
            jnp.where(
                i_v1060,
                0.0,
                (
                    (
                        IFACTOR
                        * (
                            (
                                ((i_v120 * i_v39575) + (jnp.where(i_v50235, 0.0, (CSRHGAT * (i_v51287 * i_v50594)))))
                                + (
                                    jnp.where(
                                        (CTATGAT == 0.0),
                                        0.0,
                                        (
                                            CTATGAT
                                            * (
                                                (
                                                    i_v52466
                                                    * (
                                                        0.886226925452758
                                                        * (
                                                            jnp.divide(
                                                                (
                                                                    i_v316
                                                                    * (
                                                                        jnp.where(
                                                                            i_v52145,
                                                                            i_v52393,
                                                                            (
                                                                                (
                                                                                    2.0
                                                                                    * (
                                                                                        jnp.where(
                                                                                            (i_v52141 > -230.25850929940458),
                                                                                            (
                                                                                                jnp.exp(
                                                                                                    jnp.clip(
                                                                                                        i_v52141, -709.0, 709.0
                                                                                                    )
                                                                                                )
                                                                                            ),
                                                                                            (
                                                                                                jnp.divide(
                                                                                                    1e-100,
                                                                                                    jnp.where(
                                                                                                        (
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v52408
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v52408
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v52408
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                            == 0.0
                                                                                                        )
                                                                                                        | ~jnp.isfinite(
                                                                                                            (
                                                                                                                1.0
                                                                                                                + (
                                                                                                                    i_v52408
                                                                                                                    * (
                                                                                                                        1.0
                                                                                                                        + (
                                                                                                                            0.5
                                                                                                                            * (
                                                                                                                                i_v52408
                                                                                                                                * (
                                                                                                                                    1.0
                                                                                                                                    + (
                                                                                                                                        i_v52408
                                                                                                                                        * 0.3333333333333333
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                        1e-300,
                                                                                                        (
                                                                                                            1.0
                                                                                                            + (
                                                                                                                i_v52408
                                                                                                                * (
                                                                                                                    1.0
                                                                                                                    + (
                                                                                                                        0.5
                                                                                                                        * (
                                                                                                                            i_v52408
                                                                                                                            * (
                                                                                                                                1.0
                                                                                                                                + (
                                                                                                                                    i_v52408
                                                                                                                                    * 0.3333333333333333
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        ),
                                                                                                    ),
                                                                                                )
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                                - i_v52393
                                                                            ),
                                                                        )
                                                                    )
                                                                ),
                                                                i_v52121,
                                                            )
                                                        )
                                                    )
                                                )
                                                * (
                                                    jnp.divide(
                                                        (i_v52058 * i_v52111),
                                                        jnp.where(
                                                            ((i_v52058 + i_v52111) == 0.0) | ~jnp.isfinite((i_v52058 + i_v52111)),
                                                            1e-300,
                                                            (i_v52058 + i_v52111),
                                                        ),
                                                    )
                                                )
                                            )
                                        ),
                                    )
                                )
                            )
                            + (
                                jnp.where(
                                    (CBBTGAT == 0.0),
                                    0.0,
                                    (
                                        CBBTGAT
                                        * (
                                            ((i_v1495 * i_v52860) * i_v52860)
                                            * (
                                                jnp.where(
                                                    ((jnp.where(i_v53069, (-i_v53068), i_v53068)) < 230.25850929940458),
                                                    (jnp.exp(jnp.clip(i_v53068, -709.0, 709.0))),
                                                    (
                                                        jnp.where(
                                                            i_v53069,
                                                            (
                                                                jnp.divide(
                                                                    1e-100,
                                                                    jnp.where(
                                                                        (
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v53094
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v53094
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v53094
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                            == 0.0
                                                                        )
                                                                        | ~jnp.isfinite(
                                                                            (
                                                                                1.0
                                                                                + (
                                                                                    i_v53094
                                                                                    * (
                                                                                        1.0
                                                                                        + (
                                                                                            0.5
                                                                                            * (
                                                                                                i_v53094
                                                                                                * (
                                                                                                    1.0
                                                                                                    + (
                                                                                                        i_v53094
                                                                                                        * 0.3333333333333333
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                        1e-300,
                                                                        (
                                                                            1.0
                                                                            + (
                                                                                i_v53094
                                                                                * (
                                                                                    1.0
                                                                                    + (
                                                                                        0.5
                                                                                        * (
                                                                                            i_v53094
                                                                                            * (
                                                                                                1.0
                                                                                                + (i_v53094 * 0.3333333333333333)
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        ),
                                                                    ),
                                                                )
                                                            ),
                                                            (
                                                                1e100
                                                                * (
                                                                    1.0
                                                                    + (
                                                                        i_v53115
                                                                        * (
                                                                            1.0
                                                                            + (
                                                                                0.5
                                                                                * (
                                                                                    i_v53115
                                                                                    * (1.0 + (i_v53115 * 0.3333333333333333))
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            ),
                                                        )
                                                    ),
                                                )
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                    * (
                        jnp.where(
                            (VBRGAT > 1000.0),
                            1.0,
                            (
                                jnp.where(
                                    (i_v42730 > ((-i_v413) * VBRGAT)),
                                    (
                                        jnp.divide(
                                            1.0,
                                            jnp.where(
                                                (
                                                    (
                                                        1.0
                                                        - (
                                                            jnp.where(
                                                                (PBRGAT == 4.0),
                                                                (((i_v53521 * i_v53521) * i_v53521) * i_v53521),
                                                                (
                                                                    jnp.power(
                                                                        jnp.maximum(
                                                                            (jnp.where((i_v53530 < 0.0), (-i_v53530), i_v53530)),
                                                                            0.0,
                                                                        ),
                                                                        PBRGAT,
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    )
                                                    == 0.0
                                                )
                                                | ~jnp.isfinite(
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v53521 * i_v53521) * i_v53521) * i_v53521),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v53530 < 0.0), (-i_v53530), i_v53530)),
                                                                        0.0,
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                                1e-300,
                                                (
                                                    1.0
                                                    - (
                                                        jnp.where(
                                                            (PBRGAT == 4.0),
                                                            (((i_v53521 * i_v53521) * i_v53521) * i_v53521),
                                                            (
                                                                jnp.power(
                                                                    jnp.maximum(
                                                                        (jnp.where((i_v53530 < 0.0), (-i_v53530), i_v53530)), 0.0
                                                                    ),
                                                                    PBRGAT,
                                                                )
                                                            ),
                                                        )
                                                    )
                                                ),
                                            ),
                                        )
                                    ),
                                    (i_v428 + ((i_v42730 + (i_v413 * VBRGAT)) * i_v461)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
    )
    i_v101805 = jnp.where((jnp.where((i_v17278 < 0.0), (i_v35880 < 0.0), False)), (i_v54695 < 0.0), False)
    i_v97272 = i_v1489 * i_v54
    i_v99497 = (i_v17278 - (i_v93336 * ((jnp.exp(jnp.clip(i_v97272, -709.0, 709.0))) - 1.0))) - (
        i_v97306 * ((jnp.exp(jnp.clip((i_v97272 * i_v98400), -709.0, 709.0))) - 1.0)
    )
    i_v100848 = i_v1492 * i_v54
    i_v100859 = (i_v35880 - (i_v93336 * ((jnp.exp(jnp.clip(i_v100848, -709.0, 709.0))) - 1.0))) - (
        i_v97306 * ((jnp.exp(jnp.clip((i_v100848 * i_v98400), -709.0, 709.0))) - 1.0)
    )
    i_v101784 = i_v1495 * i_v54
    i_v101795 = (i_v54695 - (i_v93336 * ((jnp.exp(jnp.clip(i_v101784, -709.0, 709.0))) - 1.0))) - (
        i_v97306 * ((jnp.exp(jnp.clip((i_v101784 * i_v98400), -709.0, 709.0))) - 1.0)
    )
    i_v101891 = jnp.where(
        (
            jnp.where(
                (
                    jnp.where(
                        (
                            jnp.where(
                                (
                                    jnp.where(
                                        (
                                            (
                                                jnp.divide(
                                                    i_v99497,
                                                    jnp.where((i_v17278 == 0.0) | ~jnp.isfinite(i_v17278), 1e-300, i_v17278),
                                                )
                                            )
                                            > 0.001
                                        ),
                                        True,
                                        (
                                            (
                                                jnp.divide(
                                                    i_v100859,
                                                    jnp.where((i_v35880 == 0.0) | ~jnp.isfinite(i_v35880), 1e-300, i_v35880),
                                                )
                                            )
                                            > 0.001
                                        ),
                                    )
                                ),
                                True,
                                (
                                    (
                                        jnp.divide(
                                            i_v101795, jnp.where((i_v54695 == 0.0) | ~jnp.isfinite(i_v54695), 1e-300, i_v54695)
                                        )
                                    )
                                    > 0.001
                                ),
                            )
                        ),
                        (i_v99497 < 0.0),
                        False,
                    )
                ),
                (i_v100859 < 0.0),
                False,
            )
        ),
        (i_v101795 < 0.0),
        False,
    )
    i_v101907 = jnp.divide(i_v99497, jnp.where((i_v100859 == 0.0) | ~jnp.isfinite(i_v100859), 1e-300, i_v100859))
    i_v101984 = i_v1489 - i_v1492
    i_v101986 = i_v1492 - i_v1489
    i_v102002 = (
        jnp.divide(
            ((-i_v52) * (jnp.log(jnp.maximum(i_v101907, 1e-300)))),
            jnp.where((i_v101984 == 0.0) | ~jnp.isfinite(i_v101984), 1e-300, i_v101984),
        )
    ) + (
        jnp.divide(
            (
                (i_v52 * (i_v101907 - 1.0))
                * (
                    (
                        jnp.power(
                            jnp.maximum(i_v101907, 0.0),
                            (jnp.divide(i_v1492, jnp.where((i_v101986 == 0.0) | ~jnp.isfinite(i_v101986), 1e-300, i_v101986))),
                        )
                    )
                    - 1.0
                )
            ),
            jnp.where(
                (
                    (
                        (
                            (
                                (
                                    jnp.power(
                                        jnp.maximum(i_v101907, 0.0),
                                        (
                                            jnp.divide(
                                                i_v1489, jnp.where((i_v101984 == 0.0) | ~jnp.isfinite(i_v101984), 1e-300, i_v101984)
                                            )
                                        ),
                                    )
                                )
                                * i_v101986
                            )
                            + (i_v101907 * i_v1489)
                        )
                        - i_v1492
                    )
                    == 0.0
                )
                | ~jnp.isfinite(
                    (
                        (
                            (
                                jnp.power(
                                    jnp.maximum(i_v101907, 0.0),
                                    (
                                        jnp.divide(
                                            i_v1489, jnp.where((i_v101984 == 0.0) | ~jnp.isfinite(i_v101984), 1e-300, i_v101984)
                                        )
                                    ),
                                )
                            )
                            * i_v101986
                        )
                        + (i_v101907 * i_v1489)
                    )
                    - i_v1492
                ),
                1e-300,
                (
                    (
                        (
                            (
                                jnp.power(
                                    jnp.maximum(i_v101907, 0.0),
                                    (
                                        jnp.divide(
                                            i_v1489, jnp.where((i_v101984 == 0.0) | ~jnp.isfinite(i_v101984), 1e-300, i_v101984)
                                        )
                                    ),
                                )
                            )
                            * i_v101986
                        )
                        + (i_v101907 * i_v1489)
                    )
                    - i_v1492
                ),
            ),
        )
    )
    i_v102050 = i_v101784 * i_v102002
    i_v102054 = (jnp.where((i_v102050 < 0.0), (-i_v102050), i_v102050)) < 1e-06
    i_v113452 = (
        jnp.where(
            i_v1435,
            (jnp.where(i_v95080, (jnp.where(i_v101805, (jnp.where(i_v101891, (jnp.where(i_v102054, 1.0, 0.0)), 0.0)), 0.0)), 0.0)),
            0.0,
        )
    ) > 0.0
    i_v106125 = jnp.where(
        i_v95080,
        (
            jnp.where(
                i_v101805,
                (
                    jnp.where(
                        i_v101891,
                        (
                            jnp.where(
                                i_v102054,
                                (
                                    i_v101795
                                    * (
                                        (jnp.divide(1.0, jnp.where((i_v1495 == 0.0) | ~jnp.isfinite(i_v1495), 1e-300, i_v1495)))
                                        + ((0.5 * i_v54) * i_v102002)
                                    )
                                ),
                                (
                                    jnp.divide(
                                        (-i_v101795),
                                        jnp.where(
                                            (((jnp.exp(jnp.clip((((-i_v1495) * i_v54) * i_v102002), -709.0, 709.0))) - 1.0) == 0.0)
                                            | ~jnp.isfinite(
                                                (jnp.exp(jnp.clip((((-i_v1495) * i_v54) * i_v102002), -709.0, 709.0))) - 1.0
                                            ),
                                            1e-300,
                                            ((jnp.exp(jnp.clip((((-i_v1495) * i_v54) * i_v102002), -709.0, 709.0))) - 1.0),
                                        ),
                                    )
                                ),
                            )
                        ),
                        0.0,
                    )
                ),
                0.0,
            )
        ),
        0.0,
    )
    (
        i_v113466,
        i_v113496,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1435, _t, _f),
        (
            i_v106125,
            (
                jnp.where(
                    i_v95080,
                    (
                        jnp.where(
                            i_v101805,
                            (
                                jnp.where(
                                    i_v101891,
                                    (
                                        jnp.where(
                                            i_v102054,
                                            (
                                                jnp.divide(
                                                    (((-0.5 * i_v101795) * i_v102002) * i_v54),
                                                    jnp.where((i_v1495 == 0.0) | ~jnp.isfinite(i_v1495), 1e-300, i_v1495),
                                                )
                                            ),
                                            i_v102002,
                                        )
                                    ),
                                    1.0,
                                )
                            ),
                            1.0,
                        )
                    ),
                    1.0,
                )
            ),
        ),
        (
            0.0,
            1.0,
        ),
    )
    i_v114734 = -i_v113466
    i_v109570 = jnp.where(
        i_v106001,
        (
            jnp.log(
                jnp.maximum(
                    (
                        jnp.divide(
                            i_v106002,
                            jnp.where(
                                (((jnp.where((i_v106125 < 0.0), (-i_v106125), i_v106125)) + 1e-21) == 0.0)
                                | ~jnp.isfinite(((jnp.where((i_v106125 < 0.0), (-i_v106125), i_v106125)) + 1e-21)),
                                1e-300,
                                ((jnp.where((i_v106125 < 0.0), (-i_v106125), i_v106125)) + 1e-21),
                            ),
                        )
                    ),
                    1e-300,
                )
            )
        ),
        0.0,
    )
    i_v110719 = jnp.where((i_v109570 < 230.25850929940458), i_v109570, 230.25850929940458)
    (
        i_v114689,
        i_v114708,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v1435, _t, _f),
        (
            i_v110719,
            (jnp.exp(jnp.clip(i_v110719, -709.0, 709.0))),
        ),
        (
            0.0,
            0.0,
        ),
    )
    i_v122091 = ~(jnp.where((jnp.where(i_v977, i_v1012, False)), i_v1060, False))
    i_v124222 = jnp.where((CSRHBOT == 0.0), (CTATBOT == 0.0), False)
    i_v124570 = PBOT == 0.5
    i_v123250 = (4.0 * i_v52) * i_v52
    i_v124579 = 1.0 - (2.0 * PBOT)
    i_v125533 = CTATBOT == 0.0
    i_v126226 = 0.666666666666667 * i_v309
    i_v126525 = (-PBOT) * i_v186
    i_v126527 = i_v126525 == -1.0
    i_v127350 = CBBTBOT == 0.0
    i_v123878 = (4.0 * i_v48) * i_v48
    i_v127351 = PBOT == 0.5
    i_v128064 = -i_v395
    i_v128178 = VBRBOT > 1000.0
    i_v128421 = (-i_v413) * VBRBOT
    i_v128423 = PBRBOT == 4.0
    i_v129062 = i_v413 * VBRBOT
    i_v130328 = jnp.where((CSRHSTI == 0.0), (CTATSTI == 0.0), False)
    i_v130769 = PSTI == 0.5
    i_v130778 = 1.0 - (2.0 * PSTI)
    i_v131770 = CTATSTI == 0.0
    i_v132505 = 0.666666666666667 * i_v314
    i_v132804 = (-PSTI) * i_v188
    i_v132806 = i_v132804 == -1.0
    i_v133293 = CBBTSTI == 0.0
    i_v133294 = PSTI == 0.5
    i_v134024 = -i_v401
    i_v134138 = VBRSTI > 1000.0
    i_v134261 = (-i_v413) * VBRSTI
    i_v134263 = PBRSTI == 4.0
    i_v134902 = i_v413 * VBRSTI
    i_v136543 = jnp.where((CSRHGAT == 0.0), (CTATGAT == 0.0), False)
    i_v136984 = PGAT == 0.5
    i_v136993 = 1.0 - (2.0 * PGAT)
    i_v137985 = CTATGAT == 0.0
    i_v138720 = 0.666666666666667 * i_v316
    i_v139019 = (-PGAT) * i_v190
    i_v139021 = i_v139019 == -1.0
    i_v139508 = CBBTGAT == 0.0
    i_v139509 = PGAT == 0.5
    i_v140239 = -i_v410
    i_v140353 = VBRGAT > 1000.0
    i_v140476 = (-i_v413) * VBRGAT
    i_v140478 = PBRGAT == 4.0
    i_v141117 = i_v413 * VBRGAT
    i_v171 = jnp.divide(1.0, jnp.where((i_v151 == 0.0) | ~jnp.isfinite(i_v151), 1e-300, i_v151))
    i_v195 = CJORBOT * (jnp.power(jnp.maximum((VBIRBOT * i_v171), 0.0), PBOT))
    i_v103378 = i_v465 * i_v195
    i_v173 = jnp.divide(1.0, jnp.where((i_v160 == 0.0) | ~jnp.isfinite(i_v160), 1e-300, i_v160))
    i_v200 = CJORSTI * (jnp.power(jnp.maximum((VBIRSTI * i_v173), 0.0), PSTI))
    i_v104660 = i_v469 * i_v200
    i_v175 = jnp.divide(1.0, jnp.where((i_v169 == 0.0) | ~jnp.isfinite(i_v169), 1e-300, i_v169))
    i_v205 = CJORGAT * (jnp.power(jnp.maximum((VBIRGAT * i_v175), 0.0), PGAT))
    i_v105940 = i_v473 * i_v205
    i_v105942 = FJUNQ * ((i_v103378 + i_v104660) + i_v105940)
    i_v116615 = (jnp.where(i_v1435, (jnp.where((i_v103378 <= i_v105942), 0.0, 1.0)), 1.0)) > 0.5
    i_v208 = (i_v195 * i_v151) * i_v186
    i_v116906 = i_v178 == 0.5
    i_v1154 = jnp.where(i_v977, (0.9 * (jnp.where((PSTI < PGAT), PSTI, PGAT))), PBOT)
    i_v1172 = jnp.where(i_v1012, (0.9 * (jnp.where((PBOT < PGAT), PBOT, PGAT))), PSTI)
    i_v1187 = jnp.where((i_v1154 > i_v1172), i_v1154, i_v1172)
    i_v1194 = jnp.where(i_v1060, (0.9 * (jnp.where((PBOT < PSTI), PBOT, PSTI))), PGAT)
    i_v1224 = i_v1141 * (
        1.0
        - (
            jnp.power(
                jnp.maximum(2.0, 0.0),
                (
                    jnp.divide(
                        -1.0,
                        jnp.where(
                            ((jnp.where((i_v1187 > i_v1194), i_v1187, i_v1194)) == 0.0)
                            | ~jnp.isfinite(jnp.where((i_v1187 > i_v1194), i_v1187, i_v1194)),
                            1e-300,
                            (jnp.where((i_v1187 > i_v1194), i_v1187, i_v1194)),
                        ),
                    )
                ),
            )
        )
    )
    i_v1144 = i_v1141 * 0.1
    i_v115422 = i_v1144 * (jnp.divide(i_v1144, jnp.where((i_v1224 == 0.0) | ~jnp.isfinite(i_v1224), 1e-300, i_v1224)))
    i_v115090 = (4.0 * i_v1144) * i_v1144
    i_v216 = 2.0 * i_v195
    i_v123944 = i_v178 == 0.5
    i_v122119 = i_v1144 * (jnp.divide(i_v1144, jnp.where((i_v1224 == 0.0) | ~jnp.isfinite(i_v1224), 1e-300, i_v1224)))
    i_v122101 = (4.0 * i_v1144) * i_v1144
    i_v119034 = (jnp.where(i_v1435, (jnp.where((i_v104660 <= i_v105942), 0.0, 1.0)), 1.0)) > 0.5
    i_v211 = (i_v200 * i_v160) * i_v188
    i_v119268 = i_v181 == 0.5
    i_v218 = 2.0 * i_v200
    i_v129702 = i_v181 == 0.5
    i_v121235 = (jnp.where(i_v1435, (jnp.where((i_v105940 <= i_v105942), 0.0, 1.0)), 1.0)) > 0.5
    i_v214 = (i_v205 * i_v169) * i_v190
    i_v121412 = i_v184 == 0.5
    i_v220 = 2.0 * i_v205
    i_v135665 = i_v184 == 0.5
    return jnp.array(
        [
            i_v144237,
            i_v1435,
            i_v52,
            i_v54,
            i_v92,
            i_v112,
            i_v465,
            i_v100,
            i_v116,
            i_v469,
            i_v108,
            i_v120,
            i_v473,
            i_v112095,
            i_v977,
            i_v1012,
            i_v1060,
            i_v112062,
            i_v112074,
            i_v687,
            i_v1852,
            i_v228,
            i_v151,
            i_v160,
            i_v169,
            i_v1141,
            i_v244,
            i_v232,
            i_v246,
            i_v236,
            i_v248,
            i_v309,
            i_v330,
            i_v178,
            i_v186,
            i_v1311,
            i_v238,
            i_v431,
            i_v418,
            i_v445,
            i_v314,
            i_v341,
            i_v181,
            i_v188,
            i_v240,
            i_v434,
            i_v423,
            i_v453,
            i_v316,
            i_v352,
            i_v184,
            i_v190,
            i_v242,
            i_v437,
            i_v428,
            i_v461,
            i_v112256,
            i_v112134,
            i_v112218,
            i_v112233,
            i_v113452,
            i_v113466,
            i_v113496,
            i_v114734,
            i_v114689,
            i_v114708,
            i_v122091,
            i_v124222,
            i_v124570,
            i_v123250,
            i_v124579,
            i_v125533,
            i_v126226,
            i_v126525,
            i_v126527,
            i_v127350,
            i_v123878,
            i_v127351,
            i_v128064,
            i_v128178,
            i_v128421,
            i_v128423,
            i_v129062,
            i_v130328,
            i_v130769,
            i_v130778,
            i_v131770,
            i_v132505,
            i_v132804,
            i_v132806,
            i_v133293,
            i_v133294,
            i_v134024,
            i_v134138,
            i_v134261,
            i_v134263,
            i_v134902,
            i_v136543,
            i_v136984,
            i_v136993,
            i_v137985,
            i_v138720,
            i_v139019,
            i_v139021,
            i_v139508,
            i_v139509,
            i_v140239,
            i_v140353,
            i_v140476,
            i_v140478,
            i_v141117,
            i_v171,
            i_v173,
            i_v175,
            i_v116615,
            i_v208,
            i_v116906,
            i_v1224,
            i_v115422,
            i_v115090,
            i_v216,
            i_v123944,
            i_v122119,
            i_v122101,
            i_v119034,
            i_v211,
            i_v119268,
            i_v218,
            i_v129702,
            i_v121235,
            i_v214,
            i_v121412,
            i_v220,
            i_v135665,
        ]
    )


@component(ports=("A", "K"))
def JUNCAP200(
    signals: Signals,
    s: States,
    init,
    TYPE: int = 1,
    DTA: float = 0.0,
    AB: float = 1.0e-12,
    LS: float = 1.0e-6,
    LG: float = 1.0e-6,
    MULT: float = 1.0,
    TRISE: float = 0.0,
    IMAX: float = 1000.0,
    TRJ: float = 21.0,
    FREV: float = 1.0e3,
    IFACTOR: float = 1.0,
    CFACTOR: float = 1.0,
    CJORBOT: float = 1.0e-3,
    CJORSTI: float = 1.0e-9,
    CJORGAT: float = 1.0e-9,
    VBIRBOT: float = 1.0,
    VBIRSTI: float = 1.0,
    VBIRGAT: float = 1.0,
    PBOT: float = 0.5,
    PSTI: float = 0.5,
    PGAT: float = 0.5,
    PHIGBOT: float = 1.16,
    PHIGSTI: float = 1.16,
    PHIGGAT: float = 1.16,
    IDSATRBOT: float = 1.0e-12,
    IDSATRSTI: float = 1.0e-18,
    IDSATRGAT: float = 1.0e-18,
    CSRHBOT: float = 1.0e2,
    CSRHSTI: float = 1.0e-4,
    CSRHGAT: float = 1.0e-4,
    XJUNSTI: float = 1.0e-7,
    XJUNGAT: float = 1.0e-7,
    CTATBOT: float = 1.0e2,
    CTATSTI: float = 1.0e-4,
    CTATGAT: float = 1.0e-4,
    MEFFTATBOT: float = 0.25,
    MEFFTATSTI: float = 0.25,
    MEFFTATGAT: float = 0.25,
    CBBTBOT: float = 1.0e-12,
    CBBTSTI: float = 1.0e-18,
    CBBTGAT: float = 1.0e-18,
    FBBTRBOT: float = 1.0e9,
    FBBTRSTI: float = 1.0e9,
    FBBTRGAT: float = 1.0e9,
    STFBBTBOT: float = -1.0e-3,
    STFBBTSTI: float = -1.0e-3,
    STFBBTGAT: float = -1.0e-3,
    VBRBOT: float = 10.0,
    VBRSTI: float = 10.0,
    VBRGAT: float = 10.0,
    PBRBOT: float = 4.0,
    PBRSTI: float = 4.0,
    PBRGAT: float = 4.0,
    SWJUNEXP: float = 0.0,
    VJUNREF: float = 2.5,
    FJUNQ: float = 0.03,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """Geometry-dependent junction current and integrated charge.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 200.6.2, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `juncap200.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/psp103/vacode/juncap200.va).
    Attribution: NXP Semiconductors, CEA-Leti, Delft University of Technology and Arizona State University; original
    notices are retained.
    The retained source and includes are under `tests/data/va_models/juncap/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/psp103/vacode/juncap200.va),
    with comparison cases in `tests/test_juncap.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        juncap.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.juncap import JUNCAP200

        jax.config.update("jax_enable_x64", True)
        model = JUNCAP200()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(JUNCAP200.__name__, "ports:", model.ports)
        ```

    """
    i_v144237 = init[0]
    i_v1435 = init[1]
    i_v52 = init[2]
    i_v54 = init[3]
    i_v92 = init[4]
    i_v112 = init[5]
    i_v465 = init[6]
    i_v100 = init[7]
    i_v116 = init[8]
    i_v469 = init[9]
    i_v108 = init[10]
    i_v120 = init[11]
    i_v473 = init[12]
    i_v112095 = init[13]
    i_v977 = init[14]
    i_v1012 = init[15]
    i_v1060 = init[16]
    i_v112062 = init[17]
    i_v112074 = init[18]
    i_v687 = init[19]
    i_v1852 = init[20]
    i_v228 = init[21]
    i_v151 = init[22]
    i_v160 = init[23]
    i_v169 = init[24]
    i_v1141 = init[25]
    i_v244 = init[26]
    i_v232 = init[27]
    i_v246 = init[28]
    i_v236 = init[29]
    i_v248 = init[30]
    i_v309 = init[31]
    i_v330 = init[32]
    i_v178 = init[33]
    i_v186 = init[34]
    i_v1311 = init[35]
    i_v238 = init[36]
    i_v431 = init[37]
    i_v418 = init[38]
    i_v445 = init[39]
    i_v314 = init[40]
    i_v341 = init[41]
    i_v181 = init[42]
    i_v188 = init[43]
    i_v240 = init[44]
    i_v434 = init[45]
    i_v423 = init[46]
    i_v453 = init[47]
    i_v316 = init[48]
    i_v352 = init[49]
    i_v184 = init[50]
    i_v190 = init[51]
    i_v242 = init[52]
    i_v437 = init[53]
    i_v428 = init[54]
    i_v461 = init[55]
    i_v112256 = init[56]
    i_v112134 = init[57]
    i_v112218 = init[58]
    i_v112233 = init[59]
    i_v113452 = init[60]
    i_v113466 = init[61]
    i_v113496 = init[62]
    i_v114734 = init[63]
    i_v114689 = init[64]
    i_v114708 = init[65]
    i_v122091 = init[66]
    i_v124222 = init[67]
    i_v124570 = init[68]
    i_v123250 = init[69]
    i_v124579 = init[70]
    i_v125533 = init[71]
    i_v126226 = init[72]
    i_v126525 = init[73]
    i_v126527 = init[74]
    i_v127350 = init[75]
    i_v123878 = init[76]
    i_v127351 = init[77]
    i_v128064 = init[78]
    i_v128178 = init[79]
    i_v128421 = init[80]
    i_v128423 = init[81]
    i_v129062 = init[82]
    i_v130328 = init[83]
    i_v130769 = init[84]
    i_v130778 = init[85]
    i_v131770 = init[86]
    i_v132505 = init[87]
    i_v132804 = init[88]
    i_v132806 = init[89]
    i_v133293 = init[90]
    i_v133294 = init[91]
    i_v134024 = init[92]
    i_v134138 = init[93]
    i_v134261 = init[94]
    i_v134263 = init[95]
    i_v134902 = init[96]
    i_v136543 = init[97]
    i_v136984 = init[98]
    i_v136993 = init[99]
    i_v137985 = init[100]
    i_v138720 = init[101]
    i_v139019 = init[102]
    i_v139021 = init[103]
    i_v139508 = init[104]
    i_v139509 = init[105]
    i_v140239 = init[106]
    i_v140353 = init[107]
    i_v140476 = init[108]
    i_v140478 = init[109]
    i_v141117 = init[110]
    i_v171 = init[111]
    i_v173 = init[112]
    i_v175 = init[113]
    i_v116615 = init[114]
    i_v208 = init[115]
    i_v116906 = init[116]
    i_v1224 = init[117]
    i_v115422 = init[118]
    i_v115090 = init[119]
    i_v216 = init[120]
    i_v123944 = init[121]
    i_v122119 = init[122]
    i_v122101 = init[123]
    i_v119034 = init[124]
    i_v211 = init[125]
    i_v119268 = init[126]
    i_v218 = init[127]
    i_v129702 = init[128]
    i_v121235 = init[129]
    i_v214 = init[130]
    i_v121412 = init[131]
    i_v220 = init[132]
    i_v135665 = init[133]
    v110776 = TYPE * (signals.A - signals.K)
    v111981 = v110776 * i_v54
    v112053 = v111981 < -230.25850929940458
    v112056 = -230.25850929940458 - v111981
    v112057 = v112056 + 1.0
    v112058 = jnp.divide(1e-100, jnp.where((v112057 == 0.0) | ~jnp.isfinite(v112057), 1e-300, v112057))
    v112070 = v111981 > i_v112062
    v112084 = v111981 - i_v112062
    v112085 = v112084 + 1.0
    v112086 = i_v112074 * v112085
    v112088 = jnp.exp(jnp.clip(v111981, -709.0, 709.0))
    v112089 = jnp.where(v112070, v112086, v112088)
    v112090 = jnp.where(v112053, v112058, v112089)
    v112115 = v112090 - 1.0
    v112116 = i_v112095 * v112115
    v112205 = v111981 * i_v112134
    v112207 = v112205 < -230.25850929940458
    v112210 = -230.25850929940458 - v112205
    v112211 = v112210 + 1.0
    v112212 = jnp.divide(1e-100, jnp.where((v112211 == 0.0) | ~jnp.isfinite(v112211), 1e-300, v112211))
    v112227 = v112205 > i_v112218
    v112244 = v112205 - i_v112218
    v112245 = v112244 + 1.0
    v112246 = i_v112233 * v112245
    v112248 = jnp.exp(jnp.clip(v112205, -709.0, 709.0))
    v112249 = jnp.where(v112227, v112246, v112248)
    v112250 = jnp.where(v112207, v112212, v112249)
    v112280 = v112250 - 1.0
    v112281 = i_v112256 * v112280
    v114761 = v112116 + v112281
    v114659 = v110776 * i_v113496
    v114660 = i_v113466 + v114659
    v114661 = v110776 * v114660
    v114663 = -v110776
    v114671 = v114663 * i_v54
    v114673 = v114671 * i_v113496
    v114675 = v114673 < -230.25850929940458
    v114678 = -230.25850929940458 - v114673
    v114679 = v114678 + 1.0
    v114680 = jnp.divide(1e-100, jnp.where((v114679 == 0.0) | ~jnp.isfinite(v114679), 1e-300, v114679))
    v114699 = v114673 > i_v114689
    v114720 = v114673 - i_v114689
    v114721 = v114720 + 1.0
    v114722 = i_v114708 * v114721
    v114724 = jnp.exp(jnp.clip(v114673, -709.0, 709.0))
    v114725 = jnp.where(v114699, v114722, v114724)
    v114726 = jnp.where(v114675, v114680, v114725)
    v114735 = v114726 - 1.0
    v114736 = i_v114734 * v114735
    v114762 = jnp.where(i_v113452, v114661, v114736)
    v114763 = v114761 + v114762
    v122438 = v110776 < i_v687
    v122449 = v110776 * i_v54
    v122450 = 0.5 * v122449
    v122451 = v122450 < 0.0
    v122452 = -v122450
    v122453 = jnp.where(v122451, v122452, v122450)
    v122454 = v122453 < 230.25850929940458
    v122465 = jnp.exp(jnp.clip(v122450, -709.0, 709.0))
    v122476 = -230.25850929940458 - v122450
    v122485 = v122476 * 0.3333333333333333
    v122486 = 1.0 + v122485
    v122487 = v122476 * v122486
    v122488 = 0.5 * v122487
    v122489 = 1.0 + v122488
    v122490 = v122476 * v122489
    v122491 = 1.0 + v122490
    v122492 = jnp.divide(1e-100, jnp.where((v122491 == 0.0) | ~jnp.isfinite(v122491), 1e-300, v122491))
    v122497 = v122450 - 230.25850929940458
    v122504 = v122497 * 0.3333333333333333
    v122505 = 1.0 + v122504
    v122506 = v122497 * v122505
    v122507 = 0.5 * v122506
    v122508 = 1.0 + v122507
    v122509 = v122497 * v122508
    v122510 = 1.0 + v122509
    v122511 = 1e100 * v122510
    v122513 = jnp.where(v122451, v122492, v122511)
    v122512 = jnp.where(v122454, v122465, v122513)
    v122514 = v122512 * v122512
    v122517 = v110776 - i_v687
    v122519 = v122517 * i_v54
    v122520 = 1.0 + v122519
    v122830 = v122520 * i_v1852
    v122832 = jnp.where(v122438, v122514, v122830)
    v122833 = v122832 - 1.0
    v124206 = jnp.where(i_v122091, v122833, 0.0)
    v124219 = i_v112 * v124206
    v122831 = jnp.sqrt(jnp.maximum(v122830, 1e-300))
    v122834 = jnp.where(v122438, v122512, v122831)
    v125510 = jnp.where(i_v122091, v122834, 0.0)
    v125525 = v125510 - 1.0
    v122839 = v110776 > 0.0
    v122835 = jnp.divide(1.0, jnp.where((v122834 == 0.0) | ~jnp.isfinite(v122834), 1e-300, v122834))
    v122906 = 2.0 + v122835
    v122907 = v122835 + 1.0
    v122908 = v122835 + 3.0
    v122909 = v122907 * v122908
    v122910 = jnp.sqrt(jnp.maximum(v122909, 1e-300))
    v122911 = v122906 + v122910
    v122912 = jnp.log(jnp.maximum(v122911, 1e-300))
    v122913 = i_v52 * v122912
    v122914 = 2.0 * v122913
    v122916 = -v110776
    v122919 = 2.0 * v122834
    v122920 = v122919 + 1.0
    v122921 = 1.0 + v122834
    v122922 = 3.0 * v122834
    v122923 = 1.0 + v122922
    v122924 = v122921 * v122923
    v122925 = jnp.sqrt(jnp.maximum(v122924, 1e-300))
    v122926 = v122920 + v122925
    v122927 = jnp.log(jnp.maximum(v122926, 1e-300))
    v122928 = i_v52 * v122927
    v122929 = 2.0 * v122928
    v122930 = v122916 + v122929
    v123240 = jnp.where(v122839, v122914, v122930)
    v123241 = i_v1141 - v123240
    v123244 = v110776 + v123241
    v123245 = v110776 - v123241
    v123247 = v123245 * v123245
    v123251 = v123247 + i_v123250
    v123252 = jnp.sqrt(jnp.maximum(v123251, 1e-300))
    v123253 = v123244 - v123252
    v123254 = 0.5 * v123253
    v124538 = jnp.where(i_v122091, v123254, 0.0)
    v124549 = i_v151 - v124538
    v124874 = v124549 * i_v244
    v124875 = jnp.sqrt(jnp.maximum(v124874, 1e-300))
    v124878 = v124549 * i_v244
    v124879 = jnp.power(jnp.maximum(v124878, 0.0), PBOT)
    v125189 = jnp.where(i_v124570, v124875, v124879)
    v125190 = i_v228 * v125189
    v125526 = v125525 * v125190
    v125527 = i_v92 * v125526
    v124556 = jnp.where(i_v122091, v123240, 0.0)
    v124566 = jnp.divide(v124556, jnp.where((v124549 == 0.0) | ~jnp.isfinite(v124549), 1e-300, v124549))
    v124567 = 1.0 - v124566
    v124568 = jnp.sqrt(jnp.maximum(v124567, 1e-300))
    v124569 = 1.0 - v124568
    v124572 = v124569 * v124569
    v124573 = jnp.log(jnp.maximum(v124569, 1e-300))
    v124574 = v124572 * v124573
    v124576 = jnp.divide(v124574, v124568)
    v124577 = v124576 + v124569
    v124580 = v124577 * i_v124579
    v124583 = jnp.where(i_v124570, 0.0, v124580)
    v124584 = v124569 + v124583
    v125531 = v125527 * v124584
    v125532 = CSRHBOT * v125531
    v129438 = jnp.where(i_v124222, 0.0, v125532)
    v129478 = v124219 + v129438
    (
        v127294,
        v125844,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v124222, _t, _f),
        (
            0.0,
            0.0,
        ),
        (
            v125527,
            v125190,
        ),
    )
    v125893 = v125844 * i_v178
    v125895 = jnp.where(i_v124222, 0.0, v124549)
    v125930 = jnp.divide(v125893, jnp.where((v125895 == 0.0) | ~jnp.isfinite(v125895), 1e-300, v125895))
    v125931 = i_v330 * v125930
    v126227 = jnp.divide(i_v126226, jnp.where((v125931 == 0.0) | ~jnp.isfinite(v125931), 1e-300, v125931))
    v126228 = v126227 * v126227
    v126229 = v126228 * v126228
    v126231 = v126229 + 1.0
    v126232 = jnp.divide(v126229, jnp.where((v126231 == 0.0) | ~jnp.isfinite(v126231), 1e-300, v126231))
    v126233 = jnp.sqrt(jnp.maximum(v126232, 1e-300))
    v126234 = jnp.sqrt(jnp.maximum(v126233, 1e-300))
    v126593 = v126227 * v126234
    v126594 = 2.0 * v126593
    v126598 = v126594 - v126233
    v126610 = v126598 - 1.0
    v126587 = jnp.divide(v125931, v126234)
    v126588 = 0.375 * v126587
    v126589 = jnp.sqrt(jnp.maximum(v126588, 1e-300))
    v126611 = v126610 * v126589
    v126613 = v126611 > 0.0
    v126797 = 0.5178164370971076 * v126611
    v126798 = 1.0 + v126797
    v126799 = jnp.divide(1.0, jnp.where((v126798 == 0.0) | ~jnp.isfinite(v126798), 1e-300, v126798))
    v126802 = 0.5178164370971076 * v126611
    v126803 = 1.0 - v126802
    v126804 = jnp.divide(1.0, jnp.where((v126803 == 0.0) | ~jnp.isfinite(v126803), 1e-300, v126803))
    v126844 = jnp.where(v126613, v126799, v126804)
    v126846 = 0.29214664 * v126844
    v127030 = v126844 * v126844
    v127031 = 0.26992878119627894 * v127030
    v127032 = v126846 + v127031
    v127217 = v127030 * v126844
    v127218 = 0.43792457880372104 * v127217
    v127219 = v127032 + v127218
    v126612 = v126611 * v126611
    v126808 = -v126612
    v126602 = i_v309 * v126227
    v126603 = v126602 * v126234
    v126604 = i_v309 * v126233
    v126605 = v126603 - v126604
    v126235 = v126233 * v126234
    v126607 = v125931 * v126235
    v126608 = 0.5 * v126607
    v126609 = v126605 + v126608
    v126812 = v126808 + v126609
    v126814 = v126812 > -230.25850929940458
    v126819 = jnp.exp(jnp.clip(v126812, -709.0, 709.0))
    v126825 = -230.25850929940458 - v126812
    v126834 = v126825 * 0.3333333333333333
    v126835 = 1.0 + v126834
    v126836 = v126825 * v126835
    v126837 = 0.5 * v126836
    v126838 = 1.0 + v126837
    v126839 = v126825 * v126838
    v126840 = 1.0 + v126839
    v126841 = jnp.divide(1e-100, jnp.where((v126840 == 0.0) | ~jnp.isfinite(v126840), 1e-300, v126840))
    v127220 = jnp.where(v126814, v126819, v126841)
    v127221 = v127219 * v127220
    v127231 = v126609 > -230.25850929940458
    v127233 = jnp.exp(jnp.clip(v126609, -709.0, 709.0))
    v127236 = -230.25850929940458 - v126609
    v127241 = v127236 * 0.3333333333333333
    v127242 = 1.0 + v127241
    v127243 = v127236 * v127242
    v127244 = 0.5 * v127243
    v127245 = 1.0 + v127244
    v127246 = v127236 * v127245
    v127247 = 1.0 + v127246
    v127248 = jnp.divide(1e-100, jnp.where((v127247 == 0.0) | ~jnp.isfinite(v127247), 1e-300, v127247))
    v127249 = jnp.where(v127231, v127233, v127248)
    v127250 = 2.0 * v127249
    v127255 = v127250 - v127221
    v127269 = jnp.where(v126613, v127221, v127255)
    v127270 = i_v309 * v127269
    v127283 = jnp.divide(v127270, v126589)
    v127284 = 0.886226925452758 * v127283
    v127335 = v127294 * v127284
    v126544 = jnp.where(i_v124222, 0.0, v124584)
    v126530 = v125931 * v126235
    v126531 = 1.0 + v126530
    v126532 = jnp.divide(1.0, jnp.where((v126531 == 0.0) | ~jnp.isfinite(v126531), 1e-300, v126531))
    v126535 = v125931 * v126235
    v126536 = 1.0 + v126535
    v126540 = jnp.power(jnp.maximum(v126536, 0.0), i_v126525)
    v126579 = jnp.where(i_v126527, v126532, v126540)
    v126580 = v126544 * v126579
    v126581 = v126544 + v126579
    v126582 = jnp.divide(v126580, jnp.where((v126581 == 0.0) | ~jnp.isfinite(v126581), 1e-300, v126581))
    v127348 = v127335 * v126582
    v127349 = CTATBOT * v127348
    v129483 = jnp.where(i_v125533, 0.0, v127349)
    v129507 = v129478 + v129483
    v123564 = v110776 + i_v1311
    v123565 = v110776 - i_v1311
    v123567 = v123565 * v123565
    v123879 = v123567 + i_v123878
    v123880 = jnp.sqrt(jnp.maximum(v123879, 1e-300))
    v123881 = v123564 - v123880
    v123882 = 0.5 * v123881
    v127363 = jnp.where(i_v122091, v123882, 0.0)
    v127448 = VBIRBOT - v127363
    v127758 = v127448 * i_v238
    v127396 = VBIRBOT - v127363
    v127420 = v127396 * i_v244
    v127421 = jnp.sqrt(jnp.maximum(v127420, 1e-300))
    v127423 = VBIRBOT - v127363
    v127425 = v127423 * i_v244
    v127426 = jnp.power(jnp.maximum(v127425, 0.0), PBOT)
    v127759 = jnp.where(i_v127351, v127421, v127426)
    v127760 = jnp.divide(v127758, jnp.where((v127759 == 0.0) | ~jnp.isfinite(v127759), 1e-300, v127759))
    v127761 = i_v186 * v127760
    v128172 = v110776 * v127761
    v128173 = v128172 * v127761
    v128065 = jnp.divide(i_v128064, jnp.where((v127761 == 0.0) | ~jnp.isfinite(v127761), 1e-300, v127761))
    v128066 = v128065 < 0.0
    v128067 = -v128065
    v128068 = jnp.where(v128066, v128067, v128065)
    v128069 = v128068 < 230.25850929940458
    v128080 = jnp.exp(jnp.clip(v128065, -709.0, 709.0))
    v128091 = -230.25850929940458 - v128065
    v128100 = v128091 * 0.3333333333333333
    v128101 = 1.0 + v128100
    v128102 = v128091 * v128101
    v128103 = 0.5 * v128102
    v128104 = 1.0 + v128103
    v128105 = v128091 * v128104
    v128106 = 1.0 + v128105
    v128107 = jnp.divide(1e-100, jnp.where((v128106 == 0.0) | ~jnp.isfinite(v128106), 1e-300, v128106))
    v128112 = v128065 - 230.25850929940458
    v128119 = v128112 * 0.3333333333333333
    v128120 = 1.0 + v128119
    v128121 = v128112 * v128120
    v128122 = 0.5 * v128121
    v128123 = 1.0 + v128122
    v128124 = v128112 * v128123
    v128125 = 1.0 + v128124
    v128126 = 1e100 * v128125
    v128175 = jnp.where(v128066, v128107, v128126)
    v128174 = jnp.where(v128069, v128080, v128175)
    v128176 = v128173 * v128174
    v128177 = CBBTBOT * v128176
    v129510 = jnp.where(i_v127350, 0.0, v128177)
    v129521 = v129507 + v129510
    v129522 = IFACTOR * v129521
    v123886 = v110776 * v110776
    v123889 = v123886 + 4e-12
    v123890 = jnp.sqrt(jnp.maximum(v123889, 1e-300))
    v123891 = v110776 - v123890
    v123892 = 0.5 * v123891
    v128191 = jnp.where(i_v122091, v123892, 0.0)
    v128422 = v128191 > i_v128421
    v128734 = v128191 * i_v431
    v128736 = v128734 * v128734
    v128738 = v128736 * v128734
    v128740 = v128738 * v128734
    v128743 = v128191 * i_v431
    v128744 = v128743 < 0.0
    v128745 = -v128743
    v128746 = jnp.where(v128744, v128745, v128743)
    v128747 = jnp.power(jnp.maximum(v128746, 0.0), PBRBOT)
    v128748 = jnp.where(i_v128423, v128740, v128747)
    v128749 = 1.0 - v128748
    v128750 = jnp.divide(1.0, jnp.where((v128749 == 0.0) | ~jnp.isfinite(v128749), 1e-300, v128749))
    v129063 = v128191 + i_v129062
    v129373 = v129063 * i_v445
    v129374 = i_v418 + v129373
    v129524 = jnp.where(v128422, v128750, v129374)
    v129523 = jnp.where(i_v128178, 1.0, v129524)
    v129525 = v129522 * v129523
    v141774 = jnp.where(i_v977, 0.0, v129525)
    v141897 = i_v465 * v141774
    v130325 = i_v116 * v124206
    v131762 = v125510 - 1.0
    v130701 = i_v160 - v124538
    v131073 = v130701 * i_v246
    v131074 = jnp.sqrt(jnp.maximum(v131073, 1e-300))
    v131077 = v130701 * i_v246
    v131078 = jnp.power(jnp.maximum(v131077, 0.0), PSTI)
    v131388 = jnp.where(i_v130769, v131074, v131078)
    v131389 = i_v232 * v131388
    v131763 = v131762 * v131389
    v131764 = i_v100 * v131763
    v130765 = jnp.divide(v124556, jnp.where((v130701 == 0.0) | ~jnp.isfinite(v130701), 1e-300, v130701))
    v130766 = 1.0 - v130765
    v130767 = jnp.sqrt(jnp.maximum(v130766, 1e-300))
    v130768 = 1.0 - v130767
    v130771 = v130768 * v130768
    v130772 = jnp.log(jnp.maximum(v130768, 1e-300))
    v130773 = v130771 * v130772
    v130775 = jnp.divide(v130773, v130767)
    v130776 = v130775 + v130768
    v130779 = v130776 * i_v130778
    v130782 = jnp.where(i_v130769, 0.0, v130779)
    v130783 = v130768 + v130782
    v131768 = v131764 * v130783
    v131769 = CSRHSTI * v131768
    v135278 = jnp.where(i_v130328, 0.0, v131769)
    v135318 = v130325 + v135278
    v133238 = jnp.where(i_v977, 0.0, v127294)
    v133231 = jnp.where(i_v130328, v133238, v131764)
    v132088 = jnp.where(i_v977, 0.0, v125844)
    v132081 = jnp.where(i_v130328, v132088, v131389)
    v132151 = v132081 * i_v181
    v132160 = jnp.where(i_v977, 0.0, v125895)
    v132153 = jnp.where(i_v130328, v132160, v130701)
    v132209 = jnp.divide(v132151, jnp.where((v132153 == 0.0) | ~jnp.isfinite(v132153), 1e-300, v132153))
    v132210 = i_v341 * v132209
    v132506 = jnp.divide(i_v132505, jnp.where((v132210 == 0.0) | ~jnp.isfinite(v132210), 1e-300, v132210))
    v132507 = v132506 * v132506
    v132508 = v132507 * v132507
    v132510 = v132508 + 1.0
    v132511 = jnp.divide(v132508, jnp.where((v132510 == 0.0) | ~jnp.isfinite(v132510), 1e-300, v132510))
    v132512 = jnp.sqrt(jnp.maximum(v132511, 1e-300))
    v132513 = jnp.sqrt(jnp.maximum(v132512, 1e-300))
    v132890 = v132506 * v132513
    v132891 = 2.0 * v132890
    v132895 = v132891 - v132512
    v132907 = v132895 - 1.0
    v132884 = jnp.divide(v132210, v132513)
    v132885 = 0.375 * v132884
    v132886 = jnp.sqrt(jnp.maximum(v132885, 1e-300))
    v132908 = v132907 * v132886
    v132910 = v132908 > 0.0
    v132974 = 0.5178164370971076 * v132908
    v132975 = 1.0 + v132974
    v132976 = jnp.divide(1.0, jnp.where((v132975 == 0.0) | ~jnp.isfinite(v132975), 1e-300, v132975))
    v132979 = 0.5178164370971076 * v132908
    v132980 = 1.0 - v132979
    v132981 = jnp.divide(1.0, jnp.where((v132980 == 0.0) | ~jnp.isfinite(v132980), 1e-300, v132980))
    v133021 = jnp.where(v132910, v132976, v132981)
    v133023 = 0.29214664 * v133021
    v133087 = v133021 * v133021
    v133088 = 0.26992878119627894 * v133087
    v133089 = v133023 + v133088
    v133154 = v133087 * v133021
    v133155 = 0.43792457880372104 * v133154
    v133156 = v133089 + v133155
    v132909 = v132908 * v132908
    v132985 = -v132909
    v132899 = i_v314 * v132506
    v132900 = v132899 * v132513
    v132901 = i_v314 * v132512
    v132902 = v132900 - v132901
    v132514 = v132512 * v132513
    v132904 = v132210 * v132514
    v132905 = 0.5 * v132904
    v132906 = v132902 + v132905
    v132989 = v132985 + v132906
    v132991 = v132989 > -230.25850929940458
    v132996 = jnp.exp(jnp.clip(v132989, -709.0, 709.0))
    v133002 = -230.25850929940458 - v132989
    v133011 = v133002 * 0.3333333333333333
    v133012 = 1.0 + v133011
    v133013 = v133002 * v133012
    v133014 = 0.5 * v133013
    v133015 = 1.0 + v133014
    v133016 = v133002 * v133015
    v133017 = 1.0 + v133016
    v133018 = jnp.divide(1e-100, jnp.where((v133017 == 0.0) | ~jnp.isfinite(v133017), 1e-300, v133017))
    v133157 = jnp.where(v132991, v132996, v133018)
    v133158 = v133156 * v133157
    v133168 = v132906 > -230.25850929940458
    v133170 = jnp.exp(jnp.clip(v132906, -709.0, 709.0))
    v133173 = -230.25850929940458 - v132906
    v133178 = v133173 * 0.3333333333333333
    v133179 = 1.0 + v133178
    v133180 = v133173 * v133179
    v133181 = 0.5 * v133180
    v133182 = 1.0 + v133181
    v133183 = v133173 * v133182
    v133184 = 1.0 + v133183
    v133185 = jnp.divide(1e-100, jnp.where((v133184 == 0.0) | ~jnp.isfinite(v133184), 1e-300, v133184))
    v133186 = jnp.where(v133168, v133170, v133185)
    v133187 = 2.0 * v133186
    v133192 = v133187 - v133158
    v133206 = jnp.where(v132910, v133158, v133192)
    v133207 = i_v314 * v133206
    v133220 = jnp.divide(v133207, v132886)
    v133221 = 0.886226925452758 * v133220
    v133278 = v133231 * v133221
    v132830 = jnp.where(i_v977, 0.0, v126544)
    v132823 = jnp.where(i_v130328, v132830, v130783)
    v132809 = v132210 * v132514
    v132810 = 1.0 + v132809
    v132811 = jnp.divide(1.0, jnp.where((v132810 == 0.0) | ~jnp.isfinite(v132810), 1e-300, v132810))
    v132814 = v132210 * v132514
    v132815 = 1.0 + v132814
    v132819 = jnp.power(jnp.maximum(v132815, 0.0), i_v132804)
    v132876 = jnp.where(i_v132806, v132811, v132819)
    v132877 = v132823 * v132876
    v132878 = v132823 + v132876
    v132879 = jnp.divide(v132877, jnp.where((v132878 == 0.0) | ~jnp.isfinite(v132878), 1e-300, v132878))
    v133291 = v133278 * v132879
    v133292 = CTATSTI * v133291
    v135323 = jnp.where(i_v131770, 0.0, v133292)
    v135347 = v135318 + v135323
    v133408 = VBIRSTI - v127363
    v133718 = v133408 * i_v240
    v133356 = VBIRSTI - v127363
    v133380 = v133356 * i_v246
    v133381 = jnp.sqrt(jnp.maximum(v133380, 1e-300))
    v133383 = VBIRSTI - v127363
    v133385 = v133383 * i_v246
    v133386 = jnp.power(jnp.maximum(v133385, 0.0), PSTI)
    v133719 = jnp.where(i_v133294, v133381, v133386)
    v133720 = jnp.divide(v133718, jnp.where((v133719 == 0.0) | ~jnp.isfinite(v133719), 1e-300, v133719))
    v133721 = i_v188 * v133720
    v134132 = v110776 * v133721
    v134133 = v134132 * v133721
    v134025 = jnp.divide(i_v134024, jnp.where((v133721 == 0.0) | ~jnp.isfinite(v133721), 1e-300, v133721))
    v134026 = v134025 < 0.0
    v134027 = -v134025
    v134028 = jnp.where(v134026, v134027, v134025)
    v134029 = v134028 < 230.25850929940458
    v134040 = jnp.exp(jnp.clip(v134025, -709.0, 709.0))
    v134051 = -230.25850929940458 - v134025
    v134060 = v134051 * 0.3333333333333333
    v134061 = 1.0 + v134060
    v134062 = v134051 * v134061
    v134063 = 0.5 * v134062
    v134064 = 1.0 + v134063
    v134065 = v134051 * v134064
    v134066 = 1.0 + v134065
    v134067 = jnp.divide(1e-100, jnp.where((v134066 == 0.0) | ~jnp.isfinite(v134066), 1e-300, v134066))
    v134072 = v134025 - 230.25850929940458
    v134079 = v134072 * 0.3333333333333333
    v134080 = 1.0 + v134079
    v134081 = v134072 * v134080
    v134082 = 0.5 * v134081
    v134083 = 1.0 + v134082
    v134084 = v134072 * v134083
    v134085 = 1.0 + v134084
    v134086 = 1e100 * v134085
    v134135 = jnp.where(v134026, v134067, v134086)
    v134134 = jnp.where(v134029, v134040, v134135)
    v134136 = v134133 * v134134
    v134137 = CBBTSTI * v134136
    v135350 = jnp.where(i_v133293, 0.0, v134137)
    v135361 = v135347 + v135350
    v135362 = IFACTOR * v135361
    v134262 = v128191 > i_v134261
    v134574 = v128191 * i_v434
    v134576 = v134574 * v134574
    v134578 = v134576 * v134574
    v134580 = v134578 * v134574
    v134583 = v128191 * i_v434
    v134584 = v134583 < 0.0
    v134585 = -v134583
    v134586 = jnp.where(v134584, v134585, v134583)
    v134587 = jnp.power(jnp.maximum(v134586, 0.0), PBRSTI)
    v134588 = jnp.where(i_v134263, v134580, v134587)
    v134589 = 1.0 - v134588
    v134590 = jnp.divide(1.0, jnp.where((v134589 == 0.0) | ~jnp.isfinite(v134589), 1e-300, v134589))
    v134903 = v128191 + i_v134902
    v135213 = v134903 * i_v453
    v135214 = i_v423 + v135213
    v135364 = jnp.where(v134262, v134590, v135214)
    v135363 = jnp.where(i_v134138, 1.0, v135364)
    v135365 = v135362 * v135363
    v142026 = jnp.where(i_v1012, 0.0, v135365)
    v142088 = i_v469 * v142026
    v142089 = v141897 + v142088
    v136540 = i_v120 * v124206
    v137977 = v125510 - 1.0
    v136916 = i_v169 - v124538
    v137288 = v136916 * i_v248
    v137289 = jnp.sqrt(jnp.maximum(v137288, 1e-300))
    v137292 = v136916 * i_v248
    v137293 = jnp.power(jnp.maximum(v137292, 0.0), PGAT)
    v137603 = jnp.where(i_v136984, v137289, v137293)
    v137604 = i_v236 * v137603
    v137978 = v137977 * v137604
    v137979 = i_v108 * v137978
    v136980 = jnp.divide(v124556, jnp.where((v136916 == 0.0) | ~jnp.isfinite(v136916), 1e-300, v136916))
    v136981 = 1.0 - v136980
    v136982 = jnp.sqrt(jnp.maximum(v136981, 1e-300))
    v136983 = 1.0 - v136982
    v136986 = v136983 * v136983
    v136987 = jnp.log(jnp.maximum(v136983, 1e-300))
    v136988 = v136986 * v136987
    v136990 = jnp.divide(v136988, v136982)
    v136991 = v136990 + v136983
    v136994 = v136991 * i_v136993
    v136997 = jnp.where(i_v136984, 0.0, v136994)
    v136998 = v136983 + v136997
    v137983 = v137979 * v136998
    v137984 = CSRHGAT * v137983
    v141493 = jnp.where(i_v136543, 0.0, v137984)
    v141533 = v136540 + v141493
    v139453 = jnp.where(i_v1012, v133238, v133231)
    v139446 = jnp.where(i_v136543, v139453, v137979)
    v138303 = jnp.where(i_v1012, v132088, v132081)
    v138296 = jnp.where(i_v136543, v138303, v137604)
    v138366 = v138296 * i_v184
    v138375 = jnp.where(i_v1012, v132160, v132153)
    v138368 = jnp.where(i_v136543, v138375, v136916)
    v138424 = jnp.divide(v138366, jnp.where((v138368 == 0.0) | ~jnp.isfinite(v138368), 1e-300, v138368))
    v138425 = i_v352 * v138424
    v138721 = jnp.divide(i_v138720, jnp.where((v138425 == 0.0) | ~jnp.isfinite(v138425), 1e-300, v138425))
    v138722 = v138721 * v138721
    v138723 = v138722 * v138722
    v138725 = v138723 + 1.0
    v138726 = jnp.divide(v138723, jnp.where((v138725 == 0.0) | ~jnp.isfinite(v138725), 1e-300, v138725))
    v138727 = jnp.sqrt(jnp.maximum(v138726, 1e-300))
    v138728 = jnp.sqrt(jnp.maximum(v138727, 1e-300))
    v139105 = v138721 * v138728
    v139106 = 2.0 * v139105
    v139110 = v139106 - v138727
    v139122 = v139110 - 1.0
    v139099 = jnp.divide(v138425, v138728)
    v139100 = 0.375 * v139099
    v139101 = jnp.sqrt(jnp.maximum(v139100, 1e-300))
    v139123 = v139122 * v139101
    v139125 = v139123 > 0.0
    v139189 = 0.5178164370971076 * v139123
    v139190 = 1.0 + v139189
    v139191 = jnp.divide(1.0, jnp.where((v139190 == 0.0) | ~jnp.isfinite(v139190), 1e-300, v139190))
    v139194 = 0.5178164370971076 * v139123
    v139195 = 1.0 - v139194
    v139196 = jnp.divide(1.0, jnp.where((v139195 == 0.0) | ~jnp.isfinite(v139195), 1e-300, v139195))
    v139236 = jnp.where(v139125, v139191, v139196)
    v139238 = 0.29214664 * v139236
    v139302 = v139236 * v139236
    v139303 = 0.26992878119627894 * v139302
    v139304 = v139238 + v139303
    v139369 = v139302 * v139236
    v139370 = 0.43792457880372104 * v139369
    v139371 = v139304 + v139370
    v139124 = v139123 * v139123
    v139200 = -v139124
    v139114 = i_v316 * v138721
    v139115 = v139114 * v138728
    v139116 = i_v316 * v138727
    v139117 = v139115 - v139116
    v138729 = v138727 * v138728
    v139119 = v138425 * v138729
    v139120 = 0.5 * v139119
    v139121 = v139117 + v139120
    v139204 = v139200 + v139121
    v139206 = v139204 > -230.25850929940458
    v139211 = jnp.exp(jnp.clip(v139204, -709.0, 709.0))
    v139217 = -230.25850929940458 - v139204
    v139226 = v139217 * 0.3333333333333333
    v139227 = 1.0 + v139226
    v139228 = v139217 * v139227
    v139229 = 0.5 * v139228
    v139230 = 1.0 + v139229
    v139231 = v139217 * v139230
    v139232 = 1.0 + v139231
    v139233 = jnp.divide(1e-100, jnp.where((v139232 == 0.0) | ~jnp.isfinite(v139232), 1e-300, v139232))
    v139372 = jnp.where(v139206, v139211, v139233)
    v139373 = v139371 * v139372
    v139383 = v139121 > -230.25850929940458
    v139385 = jnp.exp(jnp.clip(v139121, -709.0, 709.0))
    v139388 = -230.25850929940458 - v139121
    v139393 = v139388 * 0.3333333333333333
    v139394 = 1.0 + v139393
    v139395 = v139388 * v139394
    v139396 = 0.5 * v139395
    v139397 = 1.0 + v139396
    v139398 = v139388 * v139397
    v139399 = 1.0 + v139398
    v139400 = jnp.divide(1e-100, jnp.where((v139399 == 0.0) | ~jnp.isfinite(v139399), 1e-300, v139399))
    v139401 = jnp.where(v139383, v139385, v139400)
    v139402 = 2.0 * v139401
    v139407 = v139402 - v139373
    v139421 = jnp.where(v139125, v139373, v139407)
    v139422 = i_v316 * v139421
    v139435 = jnp.divide(v139422, v139101)
    v139436 = 0.886226925452758 * v139435
    v139493 = v139446 * v139436
    v139045 = jnp.where(i_v1012, v132830, v132823)
    v139038 = jnp.where(i_v136543, v139045, v136998)
    v139024 = v138425 * v138729
    v139025 = 1.0 + v139024
    v139026 = jnp.divide(1.0, jnp.where((v139025 == 0.0) | ~jnp.isfinite(v139025), 1e-300, v139025))
    v139029 = v138425 * v138729
    v139030 = 1.0 + v139029
    v139034 = jnp.power(jnp.maximum(v139030, 0.0), i_v139019)
    v139091 = jnp.where(i_v139021, v139026, v139034)
    v139092 = v139038 * v139091
    v139093 = v139038 + v139091
    v139094 = jnp.divide(v139092, jnp.where((v139093 == 0.0) | ~jnp.isfinite(v139093), 1e-300, v139093))
    v139506 = v139493 * v139094
    v139507 = CTATGAT * v139506
    v141538 = jnp.where(i_v137985, 0.0, v139507)
    v141562 = v141533 + v141538
    v139623 = VBIRGAT - v127363
    v139933 = v139623 * i_v242
    v139571 = VBIRGAT - v127363
    v139595 = v139571 * i_v248
    v139596 = jnp.sqrt(jnp.maximum(v139595, 1e-300))
    v139598 = VBIRGAT - v127363
    v139600 = v139598 * i_v248
    v139601 = jnp.power(jnp.maximum(v139600, 0.0), PGAT)
    v139934 = jnp.where(i_v139509, v139596, v139601)
    v139935 = jnp.divide(v139933, jnp.where((v139934 == 0.0) | ~jnp.isfinite(v139934), 1e-300, v139934))
    v139936 = i_v190 * v139935
    v140347 = v110776 * v139936
    v140348 = v140347 * v139936
    v140240 = jnp.divide(i_v140239, jnp.where((v139936 == 0.0) | ~jnp.isfinite(v139936), 1e-300, v139936))
    v140241 = v140240 < 0.0
    v140242 = -v140240
    v140243 = jnp.where(v140241, v140242, v140240)
    v140244 = v140243 < 230.25850929940458
    v140255 = jnp.exp(jnp.clip(v140240, -709.0, 709.0))
    v140266 = -230.25850929940458 - v140240
    v140275 = v140266 * 0.3333333333333333
    v140276 = 1.0 + v140275
    v140277 = v140266 * v140276
    v140278 = 0.5 * v140277
    v140279 = 1.0 + v140278
    v140280 = v140266 * v140279
    v140281 = 1.0 + v140280
    v140282 = jnp.divide(1e-100, jnp.where((v140281 == 0.0) | ~jnp.isfinite(v140281), 1e-300, v140281))
    v140287 = v140240 - 230.25850929940458
    v140294 = v140287 * 0.3333333333333333
    v140295 = 1.0 + v140294
    v140296 = v140287 * v140295
    v140297 = 0.5 * v140296
    v140298 = 1.0 + v140297
    v140299 = v140287 * v140298
    v140300 = 1.0 + v140299
    v140301 = 1e100 * v140300
    v140350 = jnp.where(v140241, v140282, v140301)
    v140349 = jnp.where(v140244, v140255, v140350)
    v140351 = v140348 * v140349
    v140352 = CBBTGAT * v140351
    v141565 = jnp.where(i_v139508, 0.0, v140352)
    v141576 = v141562 + v141565
    v141577 = IFACTOR * v141576
    v140477 = v128191 > i_v140476
    v140789 = v128191 * i_v437
    v140791 = v140789 * v140789
    v140793 = v140791 * v140789
    v140795 = v140793 * v140789
    v140798 = v128191 * i_v437
    v140799 = v140798 < 0.0
    v140800 = -v140798
    v140801 = jnp.where(v140799, v140800, v140798)
    v140802 = jnp.power(jnp.maximum(v140801, 0.0), PBRGAT)
    v140803 = jnp.where(i_v140478, v140795, v140802)
    v140804 = 1.0 - v140803
    v140805 = jnp.divide(1.0, jnp.where((v140804 == 0.0) | ~jnp.isfinite(v140804), 1e-300, v140804))
    v141118 = v128191 + i_v141117
    v141428 = v141118 * i_v461
    v141429 = i_v428 + v141428
    v141579 = jnp.where(v140477, v140805, v141429)
    v141578 = jnp.where(i_v140353, 1.0, v141579)
    v141580 = v141577 * v141578
    v142153 = jnp.where(i_v1060, 0.0, v141580)
    v142154 = i_v473 * v142153
    v142155 = v142089 + v142154
    v144238 = jnp.where(i_v1435, v114763, v142155)
    v144257 = i_v144237 * v144238
    v145774 = v144257 + 0.0
    v145791 = v145774 + 0.0
    v149652 = _mfactor * v145791
    v115429 = v110776 * i_v1224
    v115423 = v110776 + i_v115422
    v115424 = i_v1224 + v115423
    v115425 = i_v1224 - v115423
    v115426 = v115425 * v115425
    v115427 = v115426 + i_v115090
    v115428 = jnp.sqrt(jnp.maximum(v115427, 1e-300))
    v115430 = v115424 + v115428
    v115431 = jnp.divide(v115429, jnp.where((v115430 == 0.0) | ~jnp.isfinite(v115430), 1e-300, v115430))
    v115432 = 2.0 * v115431
    v117214 = v115432 * i_v171
    v117215 = 1.0 - v117214
    v117216 = jnp.sqrt(jnp.maximum(v117215, 1e-300))
    v117219 = v115432 * i_v171
    v117220 = 1.0 - v117219
    v117222 = jnp.power(jnp.maximum(v117220, 0.0), i_v178)
    v117529 = jnp.where(i_v116906, v117216, v117222)
    v117530 = 1.0 - v117529
    v117531 = i_v208 * v117530
    v117843 = v110776 - v115432
    v117844 = i_v216 * v117843
    v117845 = v117531 + v117844
    v142202 = jnp.where(i_v116615, v117845, 0.0)
    v122126 = v110776 * i_v1224
    v122120 = v110776 + i_v122119
    v122121 = i_v1224 + v122120
    v122122 = i_v1224 - v122120
    v122123 = v122122 * v122122
    v122124 = v122123 + i_v122101
    v122125 = jnp.sqrt(jnp.maximum(v122124, 1e-300))
    v122127 = v122121 + v122125
    v122128 = jnp.divide(v122126, jnp.where((v122127 == 0.0) | ~jnp.isfinite(v122127), 1e-300, v122127))
    v122129 = 2.0 * v122128
    v123947 = jnp.where(i_v122091, v122129, 0.0)
    v123998 = v123947 * i_v171
    v123999 = 1.0 - v123998
    v124000 = jnp.sqrt(jnp.maximum(v123999, 1e-300))
    v124003 = v123947 * i_v171
    v124004 = 1.0 - v124003
    v124006 = jnp.power(jnp.maximum(v124004, 0.0), i_v178)
    v124036 = jnp.where(i_v123944, v124000, v124006)
    v124037 = 1.0 - v124036
    v124038 = i_v208 * v124037
    v124075 = v110776 - v123947
    v124076 = i_v216 * v124075
    v124077 = v124038 + v124076
    v124078 = CFACTOR * v124077
    v142238 = jnp.where(i_v977, 0.0, v124078)
    v142197 = jnp.where(i_v1435, v142202, v142238)
    v142418 = i_v465 * v142197
    v119521 = v115432 * i_v173
    v119522 = 1.0 - v119521
    v119523 = jnp.sqrt(jnp.maximum(v119522, 1e-300))
    v119526 = v115432 * i_v173
    v119527 = 1.0 - v119526
    v119529 = jnp.power(jnp.maximum(v119527, 0.0), i_v181)
    v119779 = jnp.where(i_v119268, v119523, v119529)
    v119780 = 1.0 - v119779
    v119781 = i_v211 * v119780
    v120038 = v110776 - v115432
    v120039 = i_v218 * v120038
    v120040 = v119781 + v120039
    v142463 = jnp.where(i_v119034, v120040, 0.0)
    v129854 = v123947 * i_v173
    v129855 = 1.0 - v129854
    v129856 = jnp.sqrt(jnp.maximum(v129855, 1e-300))
    v129859 = v123947 * i_v173
    v129860 = 1.0 - v129859
    v129862 = jnp.power(jnp.maximum(v129860, 0.0), i_v181)
    v129955 = jnp.where(i_v129702, v129856, v129862)
    v129956 = 1.0 - v129955
    v129957 = i_v211 * v129956
    v130071 = v110776 - v123947
    v130072 = i_v218 * v130071
    v130073 = v129957 + v130072
    v130074 = CFACTOR * v130073
    v142499 = jnp.where(i_v1012, 0.0, v130074)
    v142460 = jnp.where(i_v1435, v142463, v142499)
    v142618 = i_v469 * v142460
    v142619 = v142418 + v142618
    v121608 = v115432 * i_v175
    v121609 = 1.0 - v121608
    v121610 = jnp.sqrt(jnp.maximum(v121609, 1e-300))
    v121613 = v115432 * i_v175
    v121614 = 1.0 - v121613
    v121616 = jnp.power(jnp.maximum(v121614, 0.0), i_v184)
    v121809 = jnp.where(i_v121412, v121610, v121616)
    v121810 = 1.0 - v121809
    v121811 = i_v214 * v121810
    v122011 = v110776 - v115432
    v122012 = i_v220 * v122011
    v122013 = v121811 + v122012
    v142662 = jnp.where(i_v121235, v122013, 0.0)
    v135880 = v123947 * i_v175
    v135881 = 1.0 - v135880
    v135882 = jnp.sqrt(jnp.maximum(v135881, 1e-300))
    v135885 = v123947 * i_v175
    v135886 = 1.0 - v135885
    v135888 = jnp.power(jnp.maximum(v135886, 0.0), i_v184)
    v136044 = jnp.where(i_v135665, v135882, v135888)
    v136045 = 1.0 - v136044
    v136046 = i_v214 * v136045
    v136223 = v110776 - v123947
    v136224 = i_v220 * v136223
    v136225 = v136046 + v136224
    v136226 = CFACTOR * v136225
    v142698 = jnp.where(i_v1060, 0.0, v136226)
    v142661 = jnp.where(i_v1435, v142662, v142698)
    v142756 = i_v473 * v142661
    v142757 = v142619 + v142756
    v145772 = i_v144237 * v142757
    v149653 = _mfactor * v145772
    v148287 = -v145791
    v149655 = _mfactor * v148287
    v148288 = -v145772
    v149657 = _mfactor * v148288
    return {"A": v149652, "K": v149655}, {"A": v149653, "K": v149657}


@JUNCAP200.setup
def _JUNCAP200_register_setup(*_a, **_kw):
    return _JUNCAP200_setup(*_a, **_kw)
