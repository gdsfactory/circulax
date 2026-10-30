# ruff: noqa: A002, ANN001, ANN002, ANN003, ANN202, FBT003, PLR0915, PLR0917
# Copyright (c) 2023 Silicon Integration Initiative, Inc.
# Licensed under the Educational Community License, Version 2.0.
# https://opensource.org/licenses/ECL-2.0
# Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See tests/data/va_models/r3_cmc/LICENSE.txt and NOTICE.txt.
# Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a.
"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with tests/data/va_models/r3_cmc/generate.py (bosdi + OpenVAF required only for regeneration).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _R3_CMC_setup(
    w: float = 1.0e-06,
    l: float = 1.0e-06,
    wd: float = 0.0,
    a1: float = 0.0,
    p1: float = 0.0,
    c1: int = 0,
    a2: float = 0.0,
    p2: float = 0.0,
    c2: int = 0,
    trise: float = 0.0,
    nsmm_rsh: float = 0.0,
    nsmm_w: float = 0.0,
    nsmm_l: float = 0.0,
    sw_noise: int = 1,
    sw_et: int = 1,
    sw_lin: int = 0,
    sw_mman: int = 0,
    version: int = 1,
    subversion: int = 1,
    revision: int = 2,
    level: int = 1003,
    type: int = -1,
    scale: float = 1.0,
    shrink: float = 0.0,
    tmin: float = -100.0,
    tmax: float = 500.0,
    rthresh: float = 1e-3,
    imax: float = 1.0,
    tnom: float = 27.0,
    lmin: float = 0.0,
    lmax: float = 9.9e09,
    wmin: float = 0.0,
    wmax: float = 9.9e09,
    jmax: float = 100.0,
    vmax: float = 9.9e09,
    tminclip: float = -100.0,
    tmaxclip: float = 500.0,
    rsh: float = 100.0,
    xw: float = 0.0,
    nwxw: float = 0.0,
    wexw: float = 0.0,
    fdrw: float = 1.0,
    fdxwinf: float = 0.0,
    xl: float = 0.0,
    xlw: float = 0.0,
    dxlsat: float = 0.0,
    nst: float = 1.0,
    ats: float = 0.0,
    atsl: float = 0.0,
    dfinf: float = 0.01,
    dfw: float = 0.0,
    dfl: float = 0.0,
    dfwl: float = 0.0,
    sw_dfgeo: int = 1,
    dp: float = 2.0,
    dpw: float = 0.0,
    dpwe: float = 0.5,
    dpl: float = 0.0,
    dple: float = 2.0,
    dpwl: float = 0.0,
    ecrit: float = 4.0,
    ecorn: float = 0.4,
    sw_vsatt: int = 0,
    sw_accpo: int = 0,
    grpo: float = 1.0e-12,
    du: float = 0.02,
    rc: float = 0.0,
    rcw: float = 0.0,
    fc: float = 0.9,
    isa: float = 0.0,
    na: float = 1.0,
    ca: float = 0.0,
    cja: float = 0.0,
    pa: float = 0.75,
    ma: float = 0.33,
    aja: float = -0.5,
    isp: float = 0.0,
    np: float = 1.0,
    cp: float = 0.0,
    cjp: float = 0.0,
    pp: float = 0.75,
    mp: float = 0.33,
    ajp: float = -0.5,
    vbv: float = 0.0,
    ibv: float = 1.0e-06,
    nbv: float = 1.0,
    kfn: float = 0.0,
    afn: float = 2.0,
    bfn: float = 1.0,
    sw_fngeo: int = 0,
    ea: float = 1.12,
    xis: float = 3.0,
    xvsat: float = 0.0,
    tc1: float = 0.0,
    tc2: float = 0.0,
    tc1l: float = 0.0,
    tc2l: float = 0.0,
    tc1w: float = 0.0,
    tc2w: float = 0.0,
    tc1wl: float = 0.0,
    tc2wl: float = 0.0,
    tc1rc: float = 0.0,
    tc2rc: float = 0.0,
    tc1dp: float = 0.0,
    tc2dp: float = 0.0,
    tc1vbv: float = 0.0,
    tc2vbv: float = 0.0,
    tc1nbv: float = 0.0,
    tc1kfn: float = 0.0,
    tegth: float = 0.0,
    gth0: float = 1.0e06,
    gthp: float = 0.0,
    gtha: float = 0.0,
    gthc: float = 0.0,
    cth0: float = 0.0,
    cthp: float = 0.0,
    ctha: float = 0.0,
    cthc: float = 0.0,
    nsig_rsh: float = 0.0,
    nsig_w: float = 0.0,
    nsig_l: float = 0.0,
    sig_rsh: float = 0.0,
    sig_w: float = 0.0,
    sig_l: float = 0.0,
    smm_rsh: float = 0.0,
    smm_w: float = 0.0,
    smm_l: float = 0.0,
    sw_mmgeo: int = 0,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    i_v16281 = sw_noise != 0
    i_v38 = ((1.0 - (0.01 * shrink)) * scale) * 1000000.0
    i_v40 = i_v38 * i_v38
    i_v198 = a1 * i_v40
    i_v2222 = isa > 0.0
    i_v2224 = -ea
    i_v49 = _temperature + trise
    i_v65 = tminclip + 1.0
    i_v1937 = tmaxclip - 1.0
    i_v44 = 273.15 + tnom
    i_v201 = p1 * i_v38
    i_v2256 = isp > 0.0
    i_v2257 = -ea
    i_v1911 = -type
    i_v2505 = vbv > 0.0
    i_v6156 = -ibv
    i_v2545 = jnp.divide(imax, jnp.where((ibv == 0.0) | ~jnp.isfinite(ibv), 1e-300, ibv))
    i_v204 = a2 * i_v40
    i_v207 = p2 * i_v38
    i_v7131 = -ibv
    i_v8439 = ((cja * i_v198) + (cjp * i_v201)) > 0.0
    i_v2378 = cja > 0.0
    i_v2388 = 0.5 * pa
    i_v2393 = -0.5 * pa
    i_v9392 = aja <= 0.0
    i_v2088 = sw_accpo != 0
    i_v752 = sw_lin != 0
    i_v554 = sw_dfgeo != 0
    i_v279 = sw_mman != 0
    i_v129 = l * i_v38
    i_v225 = c1 > 0
    i_v228 = c2 > 0
    i_v230 = jnp.asarray(i_v225, dtype=jnp.int32) + jnp.asarray(i_v228, dtype=jnp.int32)
    i_v242 = 0.5 * i_v230
    i_v126 = w * i_v38
    i_v268 = i_v129 + (i_v242 * (xl + (jnp.divide(xlw, jnp.where((i_v126 == 0.0) | ~jnp.isfinite(i_v126), 1e-300, i_v126)))))
    i_v271 = sw_mmgeo != 0
    i_v221 = i_v129 * i_v126
    i_v265 = 1.0 - (jnp.divide((wexw * (wd * i_v38)), jnp.where((i_v221 == 0.0) | ~jnp.isfinite(i_v221), 1e-300, i_v221)))
    i_v266 = jnp.divide(
        (
            ((i_v126 + xw) + (jnp.divide(nwxw, jnp.where((i_v126 == 0.0) | ~jnp.isfinite(i_v126), 1e-300, i_v126))))
            + (
                fdxwinf
                * (
                    1.0
                    - (
                        jnp.exp(
                            jnp.clip(
                                (jnp.divide((-i_v126), jnp.where((fdrw == 0.0) | ~jnp.isfinite(fdrw), 1e-300, fdrw))), -709.0, 709.0
                            )
                        )
                    )
                )
            )
        ),
        jnp.where((i_v265 == 0.0) | ~jnp.isfinite(i_v265), 1e-300, i_v265),
    )
    i_v338 = jnp.where(i_v271, i_v266, i_v126)
    i_v530 = jnp.where(
        i_v279,
        ((i_v268 + (nsig_l * sig_l)) + (jnp.divide((nsmm_l * smm_l), (jnp.sqrt(jnp.maximum((_mfactor * i_v338), 1e-300)))))),
        (
            jnp.where(
                (jnp.where((nsig_l != 0.0), (jnp.where((smm_l > 0.0), True, (sig_l > 0.0))), False)),
                (
                    i_v268
                    + (
                        nsig_l
                        * (
                            jnp.sqrt(
                                jnp.maximum(
                                    (
                                        (sig_l * sig_l)
                                        + (
                                            (jnp.divide(smm_l, (jnp.sqrt(jnp.maximum((_mfactor * i_v338), 1e-300)))))
                                            * (jnp.divide(smm_l, (jnp.sqrt(jnp.maximum((_mfactor * i_v338), 1e-300)))))
                                        )
                                    ),
                                    1e-300,
                                )
                            )
                        )
                    )
                ),
                i_v268,
            )
        ),
    )
    i_v658 = jnp.where(i_v554, i_v530, i_v129)
    i_v322 = jnp.where(i_v271, i_v268, i_v129)
    i_v323 = _mfactor * i_v322
    i_v505 = jnp.where(
        i_v279,
        ((i_v266 + (nsig_w * sig_w)) + (jnp.divide((nsmm_w * smm_w), (jnp.sqrt(jnp.maximum(i_v323, 1e-300)))))),
        (
            jnp.where(
                (jnp.where((nsig_w != 0.0), (jnp.where((smm_w > 0.0), True, (sig_w > 0.0))), False)),
                (
                    i_v266
                    + (
                        nsig_w
                        * (
                            jnp.sqrt(
                                jnp.maximum(
                                    (
                                        (sig_w * sig_w)
                                        + (
                                            (jnp.divide(smm_w, (jnp.sqrt(jnp.maximum((_mfactor * i_v322), 1e-300)))))
                                            * (jnp.divide(smm_w, (jnp.sqrt(jnp.maximum((_mfactor * i_v322), 1e-300)))))
                                        )
                                    ),
                                    1e-300,
                                )
                            )
                        )
                    )
                ),
                i_v266,
            )
        ),
    )
    i_v653 = jnp.where(i_v554, i_v505, i_v126)
    i_v772 = jnp.where(
        i_v752,
        0.0,
        (
            dfinf
            + (
                jnp.divide(
                    (((dfw * i_v658) + (dfl * i_v653)) + dfwl),
                    jnp.where(((i_v658 * i_v653) == 0.0) | ~jnp.isfinite(i_v658 * i_v653), 1e-300, (i_v658 * i_v653)),
                )
            )
        ),
    )
    i_v655 = jnp.power(jnp.maximum(i_v653, 0.0), dpwe)
    i_v656 = jnp.divide(1.0, jnp.where((i_v655 == 0.0) | ~jnp.isfinite(i_v655), 1e-300, i_v655))
    i_v660 = jnp.power(jnp.maximum(i_v658, 0.0), dple)
    i_v661 = jnp.divide(1.0, jnp.where((i_v660 == 0.0) | ~jnp.isfinite(i_v660), 1e-300, i_v660))
    i_v50 = i_v49 - 273.15
    i_v66 = i_v50 < i_v65
    i_v84 = (
        jnp.where(
            i_v66,
            (tminclip + (jnp.exp(jnp.clip(((i_v50 - tminclip) - 1.0), -709.0, 709.0)))),
            (
                jnp.where(
                    (i_v50 > (tmaxclip - 1.0)), (tmaxclip - (jnp.exp(jnp.clip(((tmaxclip - i_v50) - 1.0), -709.0, 709.0)))), i_v50
                )
            ),
        )
    ) + 273.15
    i_v108 = i_v84 - i_v44
    i_v740 = (((dp * (1.0 + (dpw * i_v656))) * (1.0 + (dpl * i_v661))) * (1.0 + ((dpwl * i_v656) * i_v661))) * (
        1.0 + (i_v108 * (tc1dp + (i_v108 * tc2dp)))
    )
    i_v745 = jnp.where((i_v740 > 0.1), i_v740, 0.1)
    i_v746 = jnp.sqrt(jnp.maximum(i_v745, 1e-300))
    i_v748 = i_v745 + 10000.0
    i_v749 = jnp.divide(i_v746, jnp.where((i_v748 == 0.0) | ~jnp.isfinite(i_v748), 1e-300, i_v748))
    i_v777 = i_v772 < i_v749
    i_v790 = jnp.where(i_v777, (i_v749 * i_v749), (i_v772 * i_v772))
    i_v801 = i_v745 * 0.5
    i_v802 = (jnp.divide(0.5, jnp.where((i_v790 == 0.0) | ~jnp.isfinite(i_v790), 1e-300, i_v790))) - i_v801
    i_v9400 = 1.0 - fc
    i_v9402 = jnp.power(jnp.maximum(i_v9400, 0.0), (-ma))
    i_v9408 = 1.0 - (i_v9402 * i_v9400)
    i_v9410 = 1.0 - ma
    i_v9427 = 1.0 - ma
    i_v9414 = 0.5 * ma
    i_v9440 = (4.0 * aja) * aja
    i_v9465 = 1.0 - ma
    i_v9470 = 1.0 - fc
    i_v9472 = jnp.power(jnp.maximum(i_v9470, 0.0), (-ma))
    i_v9476 = 0.5 * ma
    i_v2428 = cjp > 0.0
    i_v2438 = 0.5 * pp
    i_v2443 = -0.5 * pp
    i_v9698 = ajp <= 0.0
    i_v9710 = 1.0 - fc
    i_v9712 = jnp.power(jnp.maximum(i_v9710, 0.0), (-mp))
    i_v9718 = 1.0 - (i_v9712 * i_v9710)
    i_v9720 = 1.0 - mp
    i_v9737 = 1.0 - mp
    i_v9724 = 0.5 * mp
    i_v9750 = (4.0 * ajp) * ajp
    i_v9775 = 1.0 - mp
    i_v9780 = 1.0 - fc
    i_v9782 = jnp.power(jnp.maximum(i_v9780, 0.0), (-mp))
    i_v9786 = 0.5 * mp
    i_v1712 = (ca * i_v198) + (cp * i_v201)
    i_v10074 = ((cja * i_v204) + (cjp * i_v207)) > 0.0
    i_v10421 = aja <= 0.0
    i_v10429 = 1.0 - fc
    i_v10431 = jnp.power(jnp.maximum(i_v10429, 0.0), (-ma))
    i_v10437 = 1.0 - (i_v10431 * i_v10429)
    i_v10439 = 1.0 - ma
    i_v10456 = 1.0 - ma
    i_v10443 = 0.5 * ma
    i_v10469 = (4.0 * aja) * aja
    i_v10494 = 1.0 - ma
    i_v10499 = 1.0 - fc
    i_v10501 = jnp.power(jnp.maximum(i_v10499, 0.0), (-ma))
    i_v10505 = 0.5 * ma
    i_v10548 = ajp <= 0.0
    i_v10560 = 1.0 - fc
    i_v10562 = jnp.power(jnp.maximum(i_v10560, 0.0), (-mp))
    i_v10568 = 1.0 - (i_v10562 * i_v10560)
    i_v10570 = 1.0 - mp
    i_v10587 = 1.0 - mp
    i_v10574 = 0.5 * mp
    i_v10600 = (4.0 * ajp) * ajp
    i_v10625 = 1.0 - mp
    i_v10630 = 1.0 - fc
    i_v10632 = jnp.power(jnp.maximum(i_v10630, 0.0), (-mp))
    i_v10636 = 0.5 * mp
    i_v1896 = (ca * i_v204) + (cp * i_v207)
    i_v233 = (2.0 * i_v129) + (i_v230 * i_v126)
    i_v1335 = c1 + c2
    i_v7452 = jnp.where(
        i_v752,
        0.0,
        (
            (((gth0 + (gthp * i_v233)) + (gtha * i_v221)) + (gthc * i_v1335))
            * (
                jnp.power(
                    jnp.maximum((jnp.divide(i_v84, jnp.where((i_v44 == 0.0) | ~jnp.isfinite(i_v44), 1e-300, i_v44))), 0.0), tegth
                )
            )
        ),
    )
    i_v7553 = jnp.where((jnp.where((i_v7452 > 0.0), (sw_et != 0), False)), (jnp.logical_not(i_v752)), False)
    i_v7563 = tegth == 0.0
    i_v7761 = tegth + 1.0
    i_v7765 = (jnp.where((i_v7761 < 0.0), (-i_v7761), i_v7761)) > 0.1
    i_v7759 = (
        jnp.where(
            i_v66,
            (tminclip + (jnp.exp(jnp.clip(((i_v50 - tminclip) - 1.0), -709.0, 709.0)))),
            (
                jnp.where(
                    (i_v50 > (tmaxclip - 1.0)), (tmaxclip - (jnp.exp(jnp.clip(((tmaxclip - i_v50) - 1.0), -709.0, 709.0)))), i_v50
                )
            ),
        )
    ) + 273.15
    i_v7781 = i_v7452 * i_v7759
    i_v7804 = 0.5 * tegth
    i_v1056 = jnp.where(i_v777, (jnp.where((i_v772 > 0.0), i_v772, 0.0)), i_v772)
    i_v1082 = 1.0 - (i_v1056 * i_v746)
    i_v1083 = (
        (
            rsh
            * (
                jnp.where(
                    i_v279,
                    (
                        jnp.exp(
                            jnp.clip(
                                (
                                    0.01
                                    * (
                                        (nsig_rsh * sig_rsh)
                                        + (jnp.divide((nsmm_rsh * smm_rsh), (jnp.sqrt(jnp.maximum((i_v323 * i_v338), 1e-300)))))
                                    )
                                ),
                                -709.0,
                                709.0,
                            )
                        )
                    ),
                    (
                        jnp.where(
                            (jnp.where((nsig_rsh != 0.0), (jnp.where((smm_rsh > 0.0), True, (sig_rsh > 0.0))), False)),
                            (
                                jnp.exp(
                                    jnp.clip(
                                        (
                                            (0.01 * nsig_rsh)
                                            * (
                                                jnp.sqrt(
                                                    jnp.maximum(
                                                        (
                                                            (sig_rsh * sig_rsh)
                                                            + (
                                                                (
                                                                    jnp.divide(
                                                                        smm_rsh,
                                                                        (
                                                                            jnp.sqrt(
                                                                                jnp.maximum(((_mfactor * i_v322) * i_v338), 1e-300)
                                                                            )
                                                                        ),
                                                                    )
                                                                )
                                                                * (
                                                                    jnp.divide(
                                                                        smm_rsh,
                                                                        (
                                                                            jnp.sqrt(
                                                                                jnp.maximum(((_mfactor * i_v322) * i_v338), 1e-300)
                                                                            )
                                                                        ),
                                                                    )
                                                                )
                                                            )
                                                        ),
                                                        1e-300,
                                                    )
                                                )
                                            )
                                        ),
                                        -709.0,
                                        709.0,
                                    )
                                )
                            ),
                            1.0,
                        )
                    ),
                )
            )
        )
        * (jnp.divide(i_v530, jnp.where((i_v505 == 0.0) | ~jnp.isfinite(i_v505), 1e-300, i_v505)))
    ) * i_v1082
    i_v2176 = i_v1083 * i_v1082
    i_v1507 = (tc1 + (jnp.divide(tc1w, jnp.where((i_v505 == 0.0) | ~jnp.isfinite(i_v505), 1e-300, i_v505)))) + (
        jnp.divide(
            (i_v242 * (tc1l + (jnp.divide(tc1wl, jnp.where((i_v505 == 0.0) | ~jnp.isfinite(i_v505), 1e-300, i_v505))))),
            jnp.where((i_v530 == 0.0) | ~jnp.isfinite(i_v530), 1e-300, i_v530),
        )
    )
    i_v1526 = (tc2 + (jnp.divide(tc2w, jnp.where((i_v505 == 0.0) | ~jnp.isfinite(i_v505), 1e-300, i_v505)))) + (
        jnp.divide(
            (i_v242 * (tc2l + (jnp.divide(tc2wl, jnp.where((i_v505 == 0.0) | ~jnp.isfinite(i_v505), 1e-300, i_v505))))),
            jnp.where((i_v530 == 0.0) | ~jnp.isfinite(i_v530), 1e-300, i_v530),
        )
    )
    i_v805 = sw_accpo > 1
    i_v3928 = jnp.where(i_v805, (i_v1056 > 1e-09), False)
    i_v2554 = jnp.where((ecrit > 0.0), (jnp.logical_not(i_v752)), False)
    i_v2556 = sw_vsatt != 0
    i_v2951 = jnp.where(
        i_v805,
        (i_v802 - (jnp.divide((2.0 * grpo), jnp.where((i_v790 == 0.0) | ~jnp.isfinite(i_v790), 1e-300, i_v790)))),
        (
            jnp.where(
                (sw_accpo > 0),
                (
                    i_v802
                    - (
                        jnp.sqrt(
                            jnp.maximum(
                                (jnp.divide((2.0 * grpo), jnp.where((i_v790 == 0.0) | ~jnp.isfinite(i_v790), 1e-300, i_v790))),
                                1e-300,
                            )
                        )
                    )
                ),
                i_v802,
            )
        ),
    )
    i_v89 = jnp.divide((1.3806505e-23 * i_v84), 1.60217653e-19)
    i_v952 = sw_accpo > 0
    i_v3030 = jnp.where(i_v805, (nst * i_v89), (jnp.where(i_v952, ((2.0 * nst) * i_v89), (nst * i_v89))))
    i_v3178 = -0.4 * i_v745
    i_v3365 = 1.5 * i_v790
    i_v549 = i_v530 + dxlsat
    i_v3330 = 3.0 * i_v790
    i_v3664 = jnp.where(
        i_v805, ((jnp.divide(0.1666666666666667, jnp.where((i_v790 == 0.0) | ~jnp.isfinite(i_v790), 1e-300, i_v790))) - i_v801), 0.0
    )
    i_v3830 = 3.0 * i_v790
    i_v3839 = 4.5 * i_v790
    i_v2624 = (4.0 * du) * du
    i_v2630 = 2.0 * du
    i_v4605 = sw_accpo > 2
    i_v852 = 1.0 + (jnp.divide(atsl, jnp.where((i_v530 == 0.0) | ~jnp.isfinite(i_v530), 1e-300, i_v530)))
    i_v853 = jnp.divide(ats, jnp.where((i_v852 == 0.0) | ~jnp.isfinite(i_v852), 1e-300, i_v852))
    i_v4485 = jnp.where(
        i_v805,
        (
            jnp.where(
                (sw_accpo > 2),
                (
                    (0.55 * i_v89)
                    * (
                        1.0
                        + (
                            jnp.exp(
                                jnp.clip(
                                    (jnp.divide((-i_v853), jnp.where((i_v89 == 0.0) | ~jnp.isfinite(i_v89), 1e-300, i_v89))),
                                    -709.0,
                                    709.0,
                                )
                            )
                        )
                    )
                ),
                (1.1 * i_v89),
            )
        ),
        (jnp.where(i_v952, ((4.0 * i_v853) * i_v853), ((4.0 * i_v853) * i_v853))),
    )
    i_v11481 = jnp.where(i_v752, 0.0, (((cth0 + (cthp * i_v233)) + (ctha * i_v221)) + (cthc * i_v1335)))
    i_v1089 = rc > 0.0
    i_v13670 = jnp.where(
        (jnp.where(i_v1089, i_v225, False)),
        (
            jnp.divide(
                (rc + (jnp.divide(rcw, jnp.where((i_v126 == 0.0) | ~jnp.isfinite(i_v126), 1e-300, i_v126)))),
                jnp.where((c1 == 0.0) | ~jnp.isfinite(c1), 1e-300, c1),
            )
        ),
        0.0,
    )
    i_v14160 = (jnp.divide(i_v13670, jnp.where((_mfactor == 0.0) | ~jnp.isfinite(_mfactor), 1e-300, _mfactor))) <= rthresh
    i_v21337 = jnp.where(i_v14160, True, False)
    i_v15279 = jnp.where(
        (jnp.where(i_v1089, i_v228, False)),
        (
            jnp.divide(
                (rc + (jnp.divide(rcw, jnp.where((i_v126 == 0.0) | ~jnp.isfinite(i_v126), 1e-300, i_v126)))),
                jnp.where((c2 == 0.0) | ~jnp.isfinite(c2), 1e-300, c2),
            )
        ),
        0.0,
    )
    i_v15452 = (jnp.divide(i_v15279, jnp.where((_mfactor == 0.0) | ~jnp.isfinite(_mfactor), 1e-300, _mfactor))) <= rthresh
    i_v21485 = jnp.where(i_v15452, True, False)
    return jnp.array(
        [
            i_v16281,
            i_v198,
            i_v2222,
            i_v2224,
            i_v49,
            i_v65,
            i_v1937,
            i_v44,
            i_v201,
            i_v2256,
            i_v2257,
            i_v1911,
            i_v2505,
            i_v6156,
            i_v2545,
            i_v204,
            i_v207,
            i_v7131,
            i_v8439,
            i_v2378,
            i_v2388,
            i_v2393,
            i_v9392,
            i_v2088,
            i_v745,
            i_v790,
            i_v802,
            i_v9400,
            i_v9402,
            i_v9408,
            i_v9410,
            i_v9427,
            i_v9414,
            i_v9440,
            i_v9465,
            i_v9470,
            i_v9472,
            i_v9476,
            i_v2428,
            i_v2438,
            i_v2443,
            i_v9698,
            i_v9710,
            i_v9712,
            i_v9718,
            i_v9720,
            i_v9737,
            i_v9724,
            i_v9750,
            i_v9775,
            i_v9780,
            i_v9782,
            i_v9786,
            i_v1712,
            i_v10074,
            i_v10421,
            i_v10429,
            i_v10431,
            i_v10437,
            i_v10439,
            i_v10456,
            i_v10443,
            i_v10469,
            i_v10494,
            i_v10499,
            i_v10501,
            i_v10505,
            i_v10548,
            i_v10560,
            i_v10562,
            i_v10568,
            i_v10570,
            i_v10587,
            i_v10574,
            i_v10600,
            i_v10625,
            i_v10630,
            i_v10632,
            i_v10636,
            i_v1896,
            i_v7452,
            i_v7553,
            i_v7563,
            i_v7761,
            i_v7765,
            i_v7759,
            i_v7781,
            i_v7804,
            i_v1056,
            i_v1083,
            i_v2176,
            i_v1507,
            i_v1526,
            i_v3928,
            i_v2554,
            i_v2556,
            i_v2951,
            i_v89,
            i_v3030,
            i_v3178,
            i_v3365,
            i_v549,
            i_v3330,
            i_v3664,
            i_v3830,
            i_v3839,
            i_v2624,
            i_v2630,
            i_v4605,
            i_v4485,
            i_v11481,
            i_v13670,
            i_v14160,
            i_v21337,
            i_v15279,
            i_v15452,
            i_v21485,
        ]
    )


@component(ports=("n1", "nc", "n2", "dt"), states=("v_i1", "v_i2", "i_b_re1", "i_b_re2"))
def R3_CMC(
    signals: Signals,
    s: States,
    init,
    w: float = 1.0e-06,
    l: float = 1.0e-06,
    wd: float = 0.0,
    a1: float = 0.0,
    p1: float = 0.0,
    c1: int = 0,
    a2: float = 0.0,
    p2: float = 0.0,
    c2: int = 0,
    trise: float = 0.0,
    nsmm_rsh: float = 0.0,
    nsmm_w: float = 0.0,
    nsmm_l: float = 0.0,
    sw_noise: int = 1,
    sw_et: int = 1,
    sw_lin: int = 0,
    sw_mman: int = 0,
    version: int = 1,
    subversion: int = 1,
    revision: int = 2,
    level: int = 1003,
    type: int = -1,
    scale: float = 1.0,
    shrink: float = 0.0,
    tmin: float = -100.0,
    tmax: float = 500.0,
    rthresh: float = 1e-3,
    imax: float = 1.0,
    tnom: float = 27.0,
    lmin: float = 0.0,
    lmax: float = 9.9e09,
    wmin: float = 0.0,
    wmax: float = 9.9e09,
    jmax: float = 100.0,
    vmax: float = 9.9e09,
    tminclip: float = -100.0,
    tmaxclip: float = 500.0,
    rsh: float = 100.0,
    xw: float = 0.0,
    nwxw: float = 0.0,
    wexw: float = 0.0,
    fdrw: float = 1.0,
    fdxwinf: float = 0.0,
    xl: float = 0.0,
    xlw: float = 0.0,
    dxlsat: float = 0.0,
    nst: float = 1.0,
    ats: float = 0.0,
    atsl: float = 0.0,
    dfinf: float = 0.01,
    dfw: float = 0.0,
    dfl: float = 0.0,
    dfwl: float = 0.0,
    sw_dfgeo: int = 1,
    dp: float = 2.0,
    dpw: float = 0.0,
    dpwe: float = 0.5,
    dpl: float = 0.0,
    dple: float = 2.0,
    dpwl: float = 0.0,
    ecrit: float = 4.0,
    ecorn: float = 0.4,
    sw_vsatt: int = 0,
    sw_accpo: int = 0,
    grpo: float = 1.0e-12,
    du: float = 0.02,
    rc: float = 0.0,
    rcw: float = 0.0,
    fc: float = 0.9,
    isa: float = 0.0,
    na: float = 1.0,
    ca: float = 0.0,
    cja: float = 0.0,
    pa: float = 0.75,
    ma: float = 0.33,
    aja: float = -0.5,
    isp: float = 0.0,
    np: float = 1.0,
    cp: float = 0.0,
    cjp: float = 0.0,
    pp: float = 0.75,
    mp: float = 0.33,
    ajp: float = -0.5,
    vbv: float = 0.0,
    ibv: float = 1.0e-06,
    nbv: float = 1.0,
    kfn: float = 0.0,
    afn: float = 2.0,
    bfn: float = 1.0,
    sw_fngeo: int = 0,
    ea: float = 1.12,
    xis: float = 3.0,
    xvsat: float = 0.0,
    tc1: float = 0.0,
    tc2: float = 0.0,
    tc1l: float = 0.0,
    tc2l: float = 0.0,
    tc1w: float = 0.0,
    tc2w: float = 0.0,
    tc1wl: float = 0.0,
    tc2wl: float = 0.0,
    tc1rc: float = 0.0,
    tc2rc: float = 0.0,
    tc1dp: float = 0.0,
    tc2dp: float = 0.0,
    tc1vbv: float = 0.0,
    tc2vbv: float = 0.0,
    tc1nbv: float = 0.0,
    tc1kfn: float = 0.0,
    tegth: float = 0.0,
    gth0: float = 1.0e06,
    gthp: float = 0.0,
    gtha: float = 0.0,
    gthc: float = 0.0,
    cth0: float = 0.0,
    cthp: float = 0.0,
    ctha: float = 0.0,
    cthc: float = 0.0,
    nsig_rsh: float = 0.0,
    nsig_w: float = 0.0,
    nsig_l: float = 0.0,
    sig_rsh: float = 0.0,
    sig_w: float = 0.0,
    sig_l: float = 0.0,
    smm_rsh: float = 0.0,
    smm_w: float = 0.0,
    smm_l: float = 0.0,
    sw_mmgeo: int = 0,
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """Body-controlled three-terminal resistor with charge and thermal behavior.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 1.1.2, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `r3_cmc.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r3_cmc/vacode/r3_cmc.va).
    Attribution: Silicon Integration Initiative (Educational Community License 2.0).
    The retained source and includes are under `tests/data/va_models/r3_cmc/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r3_cmc/vacode/r3_cmc.va),
    with comparison cases in `tests/test_r3.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.
    For the documented full-pinch-off cases, reference derivatives use finite
    differences of the OSDI residual because its analytic Jacobian disagrees.

    Device group:
        Electronic / Compact devices

    Device reference:
        r3.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.r3 import R3_CMC

        jax.config.update("jax_enable_x64", True)
        model = R3_CMC()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(R3_CMC.__name__, "ports:", model.ports)
        ```

    """
    i_v16281 = init[0]
    i_v198 = init[1]
    i_v2222 = init[2]
    i_v2224 = init[3]
    i_v49 = init[4]
    i_v65 = init[5]
    i_v1937 = init[6]
    i_v44 = init[7]
    i_v201 = init[8]
    i_v2256 = init[9]
    i_v2257 = init[10]
    i_v1911 = init[11]
    i_v2505 = init[12]
    i_v6156 = init[13]
    i_v2545 = init[14]
    i_v204 = init[15]
    i_v207 = init[16]
    i_v7131 = init[17]
    i_v8439 = init[18]
    i_v2378 = init[19]
    i_v2388 = init[20]
    i_v2393 = init[21]
    i_v9392 = init[22]
    i_v2088 = init[23]
    i_v745 = init[24]
    i_v790 = init[25]
    i_v802 = init[26]
    i_v9400 = init[27]
    i_v9402 = init[28]
    i_v9408 = init[29]
    i_v9410 = init[30]
    i_v9427 = init[31]
    i_v9414 = init[32]
    i_v9440 = init[33]
    i_v9465 = init[34]
    i_v9470 = init[35]
    i_v9472 = init[36]
    i_v9476 = init[37]
    i_v2428 = init[38]
    i_v2438 = init[39]
    i_v2443 = init[40]
    i_v9698 = init[41]
    i_v9710 = init[42]
    i_v9712 = init[43]
    i_v9718 = init[44]
    i_v9720 = init[45]
    i_v9737 = init[46]
    i_v9724 = init[47]
    i_v9750 = init[48]
    i_v9775 = init[49]
    i_v9780 = init[50]
    i_v9782 = init[51]
    i_v9786 = init[52]
    i_v1712 = init[53]
    i_v10074 = init[54]
    i_v10421 = init[55]
    i_v10429 = init[56]
    i_v10431 = init[57]
    i_v10437 = init[58]
    i_v10439 = init[59]
    i_v10456 = init[60]
    i_v10443 = init[61]
    i_v10469 = init[62]
    i_v10494 = init[63]
    i_v10499 = init[64]
    i_v10501 = init[65]
    i_v10505 = init[66]
    i_v10548 = init[67]
    i_v10560 = init[68]
    i_v10562 = init[69]
    i_v10568 = init[70]
    i_v10570 = init[71]
    i_v10587 = init[72]
    i_v10574 = init[73]
    i_v10600 = init[74]
    i_v10625 = init[75]
    i_v10630 = init[76]
    i_v10632 = init[77]
    i_v10636 = init[78]
    i_v1896 = init[79]
    i_v7452 = init[80]
    i_v7553 = init[81]
    i_v7563 = init[82]
    i_v7761 = init[83]
    i_v7765 = init[84]
    i_v7759 = init[85]
    i_v7781 = init[86]
    i_v7804 = init[87]
    i_v1056 = init[88]
    i_v1083 = init[89]
    i_v2176 = init[90]
    i_v1507 = init[91]
    i_v1526 = init[92]
    i_v3928 = init[93]
    i_v2554 = init[94]
    i_v2556 = init[95]
    i_v2951 = init[96]
    i_v89 = init[97]
    i_v3030 = init[98]
    i_v3178 = init[99]
    i_v3365 = init[100]
    i_v549 = init[101]
    i_v3330 = init[102]
    i_v3664 = init[103]
    i_v3830 = init[104]
    i_v3839 = init[105]
    i_v2624 = init[106]
    i_v2630 = init[107]
    i_v4605 = init[108]
    i_v4485 = init[109]
    i_v11481 = init[110]
    i_v13670 = init[111]
    i_v14160 = init[112]
    i_v21337 = init[113]
    i_v15279 = init[114]
    i_v15452 = init[115]
    i_v21485 = init[116]
    v29219 = _mfactor * s.i_b_re1
    v1927 = i_v49 + signals.dt
    v1928 = v1927 - 273.15
    v1930 = v1928 < i_v65
    v1932 = v1928 - tminclip
    v1933 = v1932 - 1.0
    v1934 = jnp.exp(jnp.clip(v1933, -709.0, 709.0))
    v1935 = tminclip + v1934
    v1938 = v1928 > i_v1937
    v1940 = tmaxclip - v1928
    v1941 = v1940 - 1.0
    v1942 = jnp.exp(jnp.clip(v1941, -709.0, 709.0))
    v1943 = tmaxclip - v1942
    v1946 = jnp.where(v1938, v1943, v1928)
    v1945 = jnp.where(v1930, v1935, v1946)
    v1947 = v1945 + 273.15
    v2059 = jnp.divide(v1947, jnp.where((i_v44 == 0.0) | ~jnp.isfinite(i_v44), 1e-300, i_v44))
    v2226 = 1.0 - v2059
    v2227 = i_v2224 * v2226
    v1948 = 1.3806505e-23 * v1947
    v1949 = jnp.divide(v1948, 1.60217653e-19)
    v2238 = jnp.divide(v2227, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2240 = jnp.log(jnp.maximum(v2059, 1e-300))
    v2241 = xis * v2240
    v2242 = v2238 + v2241
    v2244 = jnp.divide(v2242, jnp.where((na == 0.0) | ~jnp.isfinite(na), 1e-300, na))
    v2245 = jnp.exp(jnp.clip(v2244, -709.0, 709.0))
    v2246 = isa * v2245
    v2304 = jnp.where(i_v2222, v2246, 0.0)
    v2306 = i_v198 * v2304
    v2261 = 1.0 - v2059
    v2262 = i_v2257 * v2261
    v2266 = jnp.divide(v2262, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2267 = jnp.log(jnp.maximum(v2059, 1e-300))
    v2268 = xis * v2267
    v2269 = v2266 + v2268
    v2271 = jnp.divide(v2269, jnp.where((np == 0.0) | ~jnp.isfinite(np), 1e-300, np))
    v2272 = jnp.exp(jnp.clip(v2271, -709.0, 709.0))
    v2273 = isp * v2272
    v2328 = jnp.where(i_v2256, v2273, 0.0)
    v2329 = i_v201 * v2328
    v2330 = v2306 + v2329
    v5036 = v2330 > 0.0
    v5417 = v2306 > 0.0
    v1919 = i_v1911 * (signals.nc - s.v_i1)
    v2248 = na * v1949
    v2250 = jnp.divide(imax, jnp.where((v2246 == 0.0) | ~jnp.isfinite(v2246), 1e-300, v2246))
    v2251 = 1.0 + v2250
    v2252 = jnp.log(jnp.maximum(v2251, 1e-300))
    v2253 = v2248 * v2252
    v5620 = jnp.where(i_v2222, v2253, 0.0)
    v5669 = v1919 < v5620
    v5500 = na * v1949
    v5501 = jnp.divide(1.0, jnp.where((v5500 == 0.0) | ~jnp.isfinite(v5500), 1e-300, v5500))
    v5672 = v1919 * v5501
    v5673 = jnp.exp(jnp.clip(v5672, -709.0, 709.0))
    v5677 = v5620 * v5501
    v5678 = jnp.exp(jnp.clip(v5677, -709.0, 709.0))
    v5680 = v1919 - v5620
    v5681 = v5680 * v5501
    v5682 = 1.0 + v5681
    v5683 = v5678 * v5682
    v5688 = jnp.where(v5669, v5673, v5683)
    v5689 = v5688 - 1.0
    v5690 = v2306 * v5689
    v5837 = jnp.where(v5417, v5690, 0.0)
    v5698 = v2329 > 0.0
    v2275 = np * v1949
    v2276 = jnp.divide(imax, jnp.where((v2273 == 0.0) | ~jnp.isfinite(v2273), 1e-300, v2273))
    v2277 = 1.0 + v2276
    v2278 = jnp.log(jnp.maximum(v2277, 1e-300))
    v2279 = v2275 * v2278
    v5761 = jnp.where(i_v2256, v2279, 0.0)
    v5813 = v1919 < v5761
    v5705 = np * v1949
    v5706 = jnp.divide(1.0, jnp.where((v5705 == 0.0) | ~jnp.isfinite(v5705), 1e-300, v5705))
    v5816 = v1919 * v5706
    v5817 = jnp.exp(jnp.clip(v5816, -709.0, 709.0))
    v5820 = v5761 * v5706
    v5821 = jnp.exp(jnp.clip(v5820, -709.0, 709.0))
    v5823 = v1919 - v5761
    v5824 = v5823 * v5706
    v5825 = 1.0 + v5824
    v5826 = v5821 * v5825
    v5831 = jnp.where(v5813, v5817, v5826)
    v5832 = v5831 - 1.0
    v5833 = v2329 * v5832
    v5842 = jnp.where(v5698, v5833, 0.0)
    v5843 = v5837 + v5842
    v2060 = v1947 - i_v44
    v2512 = v2060 * tc2vbv
    v2513 = tc1vbv + v2512
    v2514 = v2060 * v2513
    v2515 = 1.0 + v2514
    v2516 = vbv * v2515
    v2518 = v2516 > 0.0
    v2520 = jnp.where(v2518, v2516, 0.0)
    v5888 = jnp.where(i_v2505, v2520, vbv)
    v5937 = v5888 > 0.0
    v5939 = -v5888
    v5944 = v5939 - v1919
    v2526 = tc1nbv * v2060
    v2527 = 1.0 + v2526
    v2528 = nbv * v2527
    v2539 = v2528 * v1949
    v2540 = -v2520
    v2542 = jnp.divide(v2540, jnp.where((v2539 == 0.0) | ~jnp.isfinite(v2539), 1e-300, v2539))
    v2543 = jnp.exp(jnp.clip(v2542, -709.0, 709.0))
    v2546 = v2543 + i_v2545
    v2547 = jnp.log(jnp.maximum(v2546, 1e-300))
    v2548 = v2539 * v2547
    v6092 = jnp.where(i_v2505, v2548, 1.0)
    v6141 = v5944 < v6092
    v5990 = jnp.where(i_v2505, v2528, nbv)
    v6045 = v5990 * v1949
    v6046 = jnp.divide(1.0, jnp.where((v6045 == 0.0) | ~jnp.isfinite(v6045), 1e-300, v6045))
    v6144 = v5944 * v6046
    v6145 = jnp.exp(jnp.clip(v6144, -709.0, 709.0))
    v6149 = v6092 * v6046
    v6150 = jnp.exp(jnp.clip(v6149, -709.0, 709.0))
    v6152 = v5944 - v6092
    v6153 = v6152 * v6046
    v6154 = 1.0 + v6153
    v6155 = v6150 * v6154
    v6157 = jnp.where(v6141, v6145, v6155)
    v6163 = v5939 * v6046
    v6164 = jnp.exp(jnp.clip(v6163, -709.0, 709.0))
    v6165 = v6157 - v6164
    v6166 = i_v6156 * v6165
    v6174 = jnp.where(v5937, v6166, 0.0)
    v6175 = v5843 + v6174
    v6432 = v6175 + 0.0
    v7331 = jnp.where(v5036, v6432, 0.0)
    v7865 = i_v1911 * v7331
    v16251 = v7865 + 0.0
    v19309 = v16251 + 0.0
    v21076 = jnp.where(v5036, v19309, v16251)
    v21072 = jnp.where(i_v16281, v21076, v16251)
    v2353 = i_v204 * v2304
    v2375 = i_v207 * v2328
    v2376 = v2353 + v2375
    v6548 = v2376 > 0.0
    v6825 = v2353 > 0.0
    v1924 = i_v1911 * (signals.nc - s.v_i2)
    v6946 = v1924 < v5620
    v6835 = na * v1949
    v6836 = jnp.divide(1.0, jnp.where((v6835 == 0.0) | ~jnp.isfinite(v6835), 1e-300, v6835))
    v6949 = v1924 * v6836
    v6950 = jnp.exp(jnp.clip(v6949, -709.0, 709.0))
    v6954 = v5620 * v6836
    v6955 = jnp.exp(jnp.clip(v6954, -709.0, 709.0))
    v6957 = v1924 - v5620
    v6958 = v6957 * v6836
    v6959 = 1.0 + v6958
    v6960 = v6955 * v6959
    v6965 = jnp.where(v6946, v6950, v6960)
    v6966 = v6965 - 1.0
    v6967 = v2353 * v6966
    v7032 = jnp.where(v6825, v6967, 0.0)
    v6975 = v2375 > 0.0
    v7008 = v1924 < v5761
    v6982 = np * v1949
    v6983 = jnp.divide(1.0, jnp.where((v6982 == 0.0) | ~jnp.isfinite(v6982), 1e-300, v6982))
    v7011 = v1924 * v6983
    v7012 = jnp.exp(jnp.clip(v7011, -709.0, 709.0))
    v7015 = v5761 * v6983
    v7016 = jnp.exp(jnp.clip(v7015, -709.0, 709.0))
    v7018 = v1924 - v5761
    v7019 = v7018 * v6983
    v7020 = 1.0 + v7019
    v7021 = v7016 * v7020
    v7026 = jnp.where(v7008, v7012, v7021)
    v7027 = v7026 - 1.0
    v7028 = v2375 * v7027
    v7037 = jnp.where(v6975, v7028, 0.0)
    v7038 = v7032 + v7037
    v7057 = v5888 > 0.0
    v7059 = -v5888
    v7064 = v7059 - v1924
    v7116 = v7064 < v6092
    v7093 = v5990 * v1949
    v7094 = jnp.divide(1.0, jnp.where((v7093 == 0.0) | ~jnp.isfinite(v7093), 1e-300, v7093))
    v7119 = v7064 * v7094
    v7120 = jnp.exp(jnp.clip(v7119, -709.0, 709.0))
    v7124 = v6092 * v7094
    v7125 = jnp.exp(jnp.clip(v7124, -709.0, 709.0))
    v7127 = v7064 - v6092
    v7128 = v7127 * v7094
    v7129 = 1.0 + v7128
    v7130 = v7125 * v7129
    v7132 = jnp.where(v7116, v7120, v7130)
    v7138 = v7059 * v7094
    v7139 = jnp.exp(jnp.clip(v7138, -709.0, 709.0))
    v7140 = v7132 - v7139
    v7141 = i_v7131 * v7140
    v7149 = jnp.where(v7057, v7141, 0.0)
    v7150 = v7038 + v7149
    v7178 = v7150 + 0.0
    v7376 = jnp.where(v6548, v7178, 0.0)
    v7892 = i_v1911 * v7376
    v16265 = v7892 + 0.0
    v19783 = v16265 + 0.0
    v21185 = jnp.where(v6548, v19783, v16265)
    v21183 = jnp.where(i_v16281, v21185, v16265)
    v26428 = v21072 + v21183
    v29221 = _mfactor * v26428
    v2385 = jnp.divide(v1949, jnp.where((v2059 == 0.0) | ~jnp.isfinite(v2059), 1e-300, v2059))
    v2386 = 2.0 * v2385
    v2389 = i_v2388 * v2059
    v2390 = jnp.divide(v2389, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2391 = jnp.exp(jnp.clip(v2390, -709.0, 709.0))
    v2394 = i_v2393 * v2059
    v2395 = jnp.divide(v2394, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2396 = jnp.exp(jnp.clip(v2395, -709.0, 709.0))
    v2397 = v2391 - v2396
    v2398 = jnp.log(jnp.maximum(v2397, 1e-300))
    v2399 = v2386 * v2398
    v2401 = v2399 * v2059
    v2402 = 3.0 * v1949
    v2403 = jnp.log(jnp.maximum(v2059, 1e-300))
    v2404 = v2402 * v2403
    v2405 = v2401 - v2404
    v2406 = v2059 - 1.0
    v2407 = ea * v2406
    v2408 = v2405 - v2407
    v2410 = 2.0 * v1949
    v2411 = -v2408
    v2412 = jnp.divide(v2411, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2413 = jnp.exp(jnp.clip(v2412, -709.0, 709.0))
    v2414 = 4.0 * v2413
    v2415 = 1.0 + v2414
    v2416 = jnp.sqrt(jnp.maximum(v2415, 1e-300))
    v2417 = 1.0 + v2416
    v2418 = 0.5 * v2417
    v2419 = jnp.log(jnp.maximum(v2418, 1e-300))
    v2420 = v2410 * v2419
    v2421 = v2408 + v2420
    v2423 = jnp.divide(pa, jnp.where((v2421 == 0.0) | ~jnp.isfinite(v2421), 1e-300, v2421))
    v2425 = jnp.power(jnp.maximum(v2423, 0.0), ma)
    v2426 = cja * v2425
    v8789 = jnp.where(i_v2378, v2426, 0.0)
    v8903 = i_v198 * v8789
    v9194 = v8903 > 0.0
    v8602 = v1919 - i_v802
    v8603 = v1919 + i_v802
    v8605 = v8603 * v8603
    v8607 = v8605 + 0.04
    v8608 = jnp.sqrt(jnp.maximum(v8607, 1e-300))
    v8609 = v8602 + v8608
    v8610 = 0.5 * v8609
    v9395 = jnp.where(i_v2088, v8610, v1919)
    v9273 = jnp.where(i_v2378, v2421, pa)
    v9387 = -v9273
    v9389 = v9387 * fc
    v9397 = v9395 + v9389
    v9399 = v9397 > 0.0
    v9409 = v9273 * i_v9408
    v9411 = jnp.divide(v9409, jnp.where((i_v9410 == 0.0) | ~jnp.isfinite(i_v9410), 1e-300, i_v9410))
    v9425 = jnp.divide(v9395, jnp.where((v9273 == 0.0) | ~jnp.isfinite(v9273), 1e-300, v9273))
    v9426 = 1.0 - v9425
    v9428 = jnp.power(jnp.maximum(v9426, 0.0), i_v9427)
    v9429 = 1.0 - v9428
    v9430 = v9273 * v9429
    v9432 = jnp.divide(v9430, jnp.where((i_v9427 == 0.0) | ~jnp.isfinite(i_v9427), 1e-300, i_v9427))
    v9433 = jnp.where(v9399, v9411, v9432)
    v9415 = i_v9414 * v9397
    v9417 = v9273 * i_v9400
    v9418 = jnp.divide(v9415, jnp.where((v9417 == 0.0) | ~jnp.isfinite(v9417), 1e-300, v9417))
    v9419 = 1.0 + v9418
    v9420 = v9397 * v9419
    v9421 = v9420 * i_v9402
    v9434 = jnp.where(v9399, v9421, 0.0)
    v9435 = v9433 + v9434
    v9449 = v9395 + v9389
    v9451 = v9449 * v9449
    v9454 = v9451 + i_v9440
    v9455 = jnp.sqrt(jnp.maximum(v9454, 1e-300))
    v9457 = v9449 - v9455
    v9458 = 0.5 * v9457
    v9459 = v9458 - v9389
    v9463 = jnp.divide(v9459, jnp.where((v9273 == 0.0) | ~jnp.isfinite(v9273), 1e-300, v9273))
    v9464 = 1.0 - v9463
    v9466 = jnp.power(jnp.maximum(v9464, 0.0), i_v9465)
    v9467 = v9387 * v9466
    v9469 = jnp.divide(v9467, jnp.where((i_v9465 == 0.0) | ~jnp.isfinite(i_v9465), 1e-300, i_v9465))
    v9473 = v9395 - v9459
    v9438 = v9389 * v9389
    v9441 = v9438 + i_v9440
    v9442 = jnp.sqrt(jnp.maximum(v9441, 1e-300))
    v9445 = v9389 + v9442
    v9446 = -0.5 * v9445
    v9474 = v9473 + v9446
    v9475 = i_v9472 * v9474
    v9479 = i_v9476 * v9474
    v9481 = v9273 * i_v9470
    v9482 = jnp.divide(v9479, jnp.where((v9481 == 0.0) | ~jnp.isfinite(v9481), 1e-300, v9481))
    v9483 = 1.0 + v9482
    v9484 = v9475 * v9483
    v9485 = v9469 + v9484
    v9817 = jnp.where(i_v9392, v9435, v9485)
    v9816 = jnp.where(v9194, v9817, 0.0)
    v9825 = v8903 * v9816
    v2435 = jnp.divide(v1949, jnp.where((v2059 == 0.0) | ~jnp.isfinite(v2059), 1e-300, v2059))
    v2436 = 2.0 * v2435
    v2439 = i_v2438 * v2059
    v2440 = jnp.divide(v2439, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2441 = jnp.exp(jnp.clip(v2440, -709.0, 709.0))
    v2444 = i_v2443 * v2059
    v2445 = jnp.divide(v2444, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2446 = jnp.exp(jnp.clip(v2445, -709.0, 709.0))
    v2447 = v2441 - v2446
    v2448 = jnp.log(jnp.maximum(v2447, 1e-300))
    v2449 = v2436 * v2448
    v2451 = v2449 * v2059
    v2452 = 3.0 * v1949
    v2453 = jnp.log(jnp.maximum(v2059, 1e-300))
    v2454 = v2452 * v2453
    v2455 = v2451 - v2454
    v2456 = v2059 - 1.0
    v2457 = ea * v2456
    v2458 = v2455 - v2457
    v2460 = 2.0 * v1949
    v2461 = -v2458
    v2462 = jnp.divide(v2461, jnp.where((v1949 == 0.0) | ~jnp.isfinite(v1949), 1e-300, v1949))
    v2463 = jnp.exp(jnp.clip(v2462, -709.0, 709.0))
    v2464 = 4.0 * v2463
    v2465 = 1.0 + v2464
    v2466 = jnp.sqrt(jnp.maximum(v2465, 1e-300))
    v2467 = 1.0 + v2466
    v2468 = 0.5 * v2467
    v2469 = jnp.log(jnp.maximum(v2468, 1e-300))
    v2470 = v2460 * v2469
    v2471 = v2458 + v2470
    v2473 = jnp.divide(pp, jnp.where((v2471 == 0.0) | ~jnp.isfinite(v2471), 1e-300, v2471))
    v2475 = jnp.power(jnp.maximum(v2473, 0.0), mp)
    v2476 = cjp * v2475
    v9079 = jnp.where(i_v2428, v2476, 0.0)
    v9192 = i_v201 * v9079
    v9495 = v9192 > 0.0
    v9574 = jnp.where(i_v2428, v2471, pp)
    v9694 = -v9574
    v9695 = v9694 * fc
    v9707 = v9395 + v9695
    v9709 = v9707 > 0.0
    v9719 = v9574 * i_v9718
    v9721 = jnp.divide(v9719, jnp.where((i_v9720 == 0.0) | ~jnp.isfinite(i_v9720), 1e-300, i_v9720))
    v9735 = jnp.divide(v9395, jnp.where((v9574 == 0.0) | ~jnp.isfinite(v9574), 1e-300, v9574))
    v9736 = 1.0 - v9735
    v9738 = jnp.power(jnp.maximum(v9736, 0.0), i_v9737)
    v9739 = 1.0 - v9738
    v9740 = v9574 * v9739
    v9742 = jnp.divide(v9740, jnp.where((i_v9737 == 0.0) | ~jnp.isfinite(i_v9737), 1e-300, i_v9737))
    v9743 = jnp.where(v9709, v9721, v9742)
    v9725 = i_v9724 * v9707
    v9727 = v9574 * i_v9710
    v9728 = jnp.divide(v9725, jnp.where((v9727 == 0.0) | ~jnp.isfinite(v9727), 1e-300, v9727))
    v9729 = 1.0 + v9728
    v9730 = v9707 * v9729
    v9731 = v9730 * i_v9712
    v9744 = jnp.where(v9709, v9731, 0.0)
    v9745 = v9743 + v9744
    v9759 = v9395 + v9695
    v9761 = v9759 * v9759
    v9764 = v9761 + i_v9750
    v9765 = jnp.sqrt(jnp.maximum(v9764, 1e-300))
    v9767 = v9759 - v9765
    v9768 = 0.5 * v9767
    v9769 = v9768 - v9695
    v9773 = jnp.divide(v9769, jnp.where((v9574 == 0.0) | ~jnp.isfinite(v9574), 1e-300, v9574))
    v9774 = 1.0 - v9773
    v9776 = jnp.power(jnp.maximum(v9774, 0.0), i_v9775)
    v9777 = v9694 * v9776
    v9779 = jnp.divide(v9777, jnp.where((i_v9775 == 0.0) | ~jnp.isfinite(i_v9775), 1e-300, i_v9775))
    v9783 = v9395 - v9769
    v9748 = v9695 * v9695
    v9751 = v9748 + i_v9750
    v9752 = jnp.sqrt(jnp.maximum(v9751, 1e-300))
    v9755 = v9695 + v9752
    v9756 = -0.5 * v9755
    v9784 = v9783 + v9756
    v9785 = i_v9782 * v9784
    v9789 = i_v9786 * v9784
    v9791 = v9574 * i_v9780
    v9792 = jnp.divide(v9789, jnp.where((v9791 == 0.0) | ~jnp.isfinite(v9791), 1e-300, v9791))
    v9793 = 1.0 + v9792
    v9794 = v9785 * v9793
    v9795 = v9779 + v9794
    v9836 = jnp.where(i_v9698, v9745, v9795)
    v9835 = jnp.where(v9495, v9836, 0.0)
    v9837 = v9192 * v9835
    v9838 = v9825 + v9837
    v10692 = jnp.where(i_v8439, v9838, 0.0)
    v11018 = i_v1712 * v1919
    v11019 = v10692 + v11018
    v11304 = i_v1911 * v11019
    v10267 = i_v204 * v8789
    v10397 = v10267 > 0.0
    v10132 = v1924 - i_v802
    v10133 = v1924 + i_v802
    v10135 = v10133 * v10133
    v10136 = v10135 + 0.04
    v10137 = jnp.sqrt(jnp.maximum(v10136, 1e-300))
    v10138 = v10132 + v10137
    v10139 = 0.5 * v10138
    v10424 = jnp.where(i_v2088, v10139, v1924)
    v10418 = -v9273
    v10419 = v10418 * fc
    v10426 = v10424 + v10419
    v10428 = v10426 > 0.0
    v10438 = v9273 * i_v10437
    v10440 = jnp.divide(v10438, jnp.where((i_v10439 == 0.0) | ~jnp.isfinite(i_v10439), 1e-300, i_v10439))
    v10454 = jnp.divide(v10424, jnp.where((v9273 == 0.0) | ~jnp.isfinite(v9273), 1e-300, v9273))
    v10455 = 1.0 - v10454
    v10457 = jnp.power(jnp.maximum(v10455, 0.0), i_v10456)
    v10458 = 1.0 - v10457
    v10459 = v9273 * v10458
    v10461 = jnp.divide(v10459, jnp.where((i_v10456 == 0.0) | ~jnp.isfinite(i_v10456), 1e-300, i_v10456))
    v10462 = jnp.where(v10428, v10440, v10461)
    v10444 = i_v10443 * v10426
    v10446 = v9273 * i_v10429
    v10447 = jnp.divide(v10444, jnp.where((v10446 == 0.0) | ~jnp.isfinite(v10446), 1e-300, v10446))
    v10448 = 1.0 + v10447
    v10449 = v10426 * v10448
    v10450 = v10449 * i_v10431
    v10463 = jnp.where(v10428, v10450, 0.0)
    v10464 = v10462 + v10463
    v10478 = v10424 + v10419
    v10480 = v10478 * v10478
    v10483 = v10480 + i_v10469
    v10484 = jnp.sqrt(jnp.maximum(v10483, 1e-300))
    v10486 = v10478 - v10484
    v10487 = 0.5 * v10486
    v10488 = v10487 - v10419
    v10492 = jnp.divide(v10488, jnp.where((v9273 == 0.0) | ~jnp.isfinite(v9273), 1e-300, v9273))
    v10493 = 1.0 - v10492
    v10495 = jnp.power(jnp.maximum(v10493, 0.0), i_v10494)
    v10496 = v10418 * v10495
    v10498 = jnp.divide(v10496, jnp.where((i_v10494 == 0.0) | ~jnp.isfinite(i_v10494), 1e-300, i_v10494))
    v10502 = v10424 - v10488
    v10467 = v10419 * v10419
    v10470 = v10467 + i_v10469
    v10471 = jnp.sqrt(jnp.maximum(v10470, 1e-300))
    v10474 = v10419 + v10471
    v10475 = -0.5 * v10474
    v10503 = v10502 + v10475
    v10504 = i_v10501 * v10503
    v10508 = i_v10505 * v10503
    v10510 = v9273 * i_v10499
    v10511 = jnp.divide(v10508, jnp.where((v10510 == 0.0) | ~jnp.isfinite(v10510), 1e-300, v10510))
    v10512 = 1.0 + v10511
    v10513 = v10504 * v10512
    v10514 = v10498 + v10513
    v10667 = jnp.where(i_v10421, v10464, v10514)
    v10666 = jnp.where(v10397, v10667, 0.0)
    v10675 = v10267 * v10666
    v10395 = i_v207 * v9079
    v10524 = v10395 > 0.0
    v10545 = -v9574
    v10546 = v10545 * fc
    v10557 = v10424 + v10546
    v10559 = v10557 > 0.0
    v10569 = v9574 * i_v10568
    v10571 = jnp.divide(v10569, jnp.where((i_v10570 == 0.0) | ~jnp.isfinite(i_v10570), 1e-300, i_v10570))
    v10585 = jnp.divide(v10424, jnp.where((v9574 == 0.0) | ~jnp.isfinite(v9574), 1e-300, v9574))
    v10586 = 1.0 - v10585
    v10588 = jnp.power(jnp.maximum(v10586, 0.0), i_v10587)
    v10589 = 1.0 - v10588
    v10590 = v9574 * v10589
    v10592 = jnp.divide(v10590, jnp.where((i_v10587 == 0.0) | ~jnp.isfinite(i_v10587), 1e-300, i_v10587))
    v10593 = jnp.where(v10559, v10571, v10592)
    v10575 = i_v10574 * v10557
    v10577 = v9574 * i_v10560
    v10578 = jnp.divide(v10575, jnp.where((v10577 == 0.0) | ~jnp.isfinite(v10577), 1e-300, v10577))
    v10579 = 1.0 + v10578
    v10580 = v10557 * v10579
    v10581 = v10580 * i_v10562
    v10594 = jnp.where(v10559, v10581, 0.0)
    v10595 = v10593 + v10594
    v10609 = v10424 + v10546
    v10611 = v10609 * v10609
    v10614 = v10611 + i_v10600
    v10615 = jnp.sqrt(jnp.maximum(v10614, 1e-300))
    v10617 = v10609 - v10615
    v10618 = 0.5 * v10617
    v10619 = v10618 - v10546
    v10623 = jnp.divide(v10619, jnp.where((v9574 == 0.0) | ~jnp.isfinite(v9574), 1e-300, v9574))
    v10624 = 1.0 - v10623
    v10626 = jnp.power(jnp.maximum(v10624, 0.0), i_v10625)
    v10627 = v10545 * v10626
    v10629 = jnp.divide(v10627, jnp.where((i_v10625 == 0.0) | ~jnp.isfinite(i_v10625), 1e-300, i_v10625))
    v10633 = v10424 - v10619
    v10598 = v10546 * v10546
    v10601 = v10598 + i_v10600
    v10602 = jnp.sqrt(jnp.maximum(v10601, 1e-300))
    v10605 = v10546 + v10602
    v10606 = -0.5 * v10605
    v10634 = v10633 + v10606
    v10635 = i_v10632 * v10634
    v10639 = i_v10636 * v10634
    v10641 = v9574 * i_v10630
    v10642 = jnp.divide(v10639, jnp.where((v10641 == 0.0) | ~jnp.isfinite(v10641), 1e-300, v10641))
    v10643 = 1.0 + v10642
    v10644 = v10635 * v10643
    v10645 = v10629 + v10644
    v10686 = jnp.where(i_v10548, v10595, v10645)
    v10685 = jnp.where(v10524, v10686, 0.0)
    v10687 = v10395 * v10685
    v10688 = v10675 + v10687
    v11020 = jnp.where(i_v10074, v10688, 0.0)
    v11300 = i_v1896 * v1924
    v11301 = v11020 + v11300
    v11307 = i_v1911 * v11301
    v26429 = v11304 + v11307
    v29223 = _mfactor * v26429
    v29224 = _mfactor * s.i_b_re2
    v7736 = i_v7452 * signals.dt
    v7793 = jnp.divide(signals.dt, jnp.where((i_v7759 == 0.0) | ~jnp.isfinite(i_v7759), 1e-300, i_v7759))
    v7794 = 1.0 + v7793
    v7796 = jnp.power(jnp.maximum(v7794, 0.0), i_v7761)
    v7797 = v7796 - 1.0
    v7798 = i_v7781 * v7797
    v7800 = jnp.divide(v7798, jnp.where((i_v7761 == 0.0) | ~jnp.isfinite(i_v7761), 1e-300, i_v7761))
    v7803 = i_v7452 * signals.dt
    v7805 = i_v7804 * signals.dt
    v7807 = jnp.divide(v7805, jnp.where((i_v7759 == 0.0) | ~jnp.isfinite(i_v7759), 1e-300, i_v7759))
    v7808 = 1.0 + v7807
    v7809 = v7803 * v7808
    v13038 = jnp.where(i_v7765, v7800, v7809)
    v13037 = jnp.where(i_v7563, v7736, v13038)
    v7811 = 1000000.0 * signals.dt
    v13036 = jnp.where(i_v7553, v13037, v7811)
    v1914 = i_v1911 * (s.v_i2 - s.v_i1)
    v2798 = v1914 < 0.0
    v4900 = jnp.where(v2798, -1.0, 1.0)
    v2073 = v2060 * i_v1526
    v2074 = i_v1507 + v2073
    v2075 = v2060 * v2074
    v2076 = 1.0 + v2075
    v2079 = v2076 < 0.11
    v2081 = v2076 - 0.01
    v2082 = 10.0 * v2081
    v2083 = v2082 - 1.0
    v2084 = jnp.exp(jnp.clip(v2083, -709.0, 709.0))
    v2085 = 0.1 * v2084
    v2086 = 0.01 + v2085
    v2178 = jnp.where(v2079, v2086, v2076)
    v2179 = i_v2176 * v2178
    v2180 = jnp.divide(1.0, jnp.where((v2179 == 0.0) | ~jnp.isfinite(v2179), 1e-300, v2179))
    v2184 = i_v1083 * v2178
    v2185 = jnp.divide(1.0, jnp.where((v2184 == 0.0) | ~jnp.isfinite(v2184), 1e-300, v2184))
    v4807 = jnp.where(i_v2088, v2180, v2185)
    v2219 = jnp.power(jnp.maximum(v2059, 0.0), xvsat)
    v2617 = ecrit * v2219
    v2618 = v2617 * v2178
    v2625 = jnp.where(i_v2556, v2618, ecrit)
    v2646 = jnp.divide(1.0, jnp.where((v2625 == 0.0) | ~jnp.isfinite(v2625), 1e-300, v2625))
    v3210 = jnp.where(i_v2554, v2646, 0.0)
    v3223 = v3210 > 0.0
    v2850 = -v1924
    v2904 = -v1919
    v2906 = jnp.where(v2798, v2850, v2904)
    v2987 = v2906 > i_v2951
    v3065 = i_v2951 - v2906
    v3066 = jnp.divide(v3065, jnp.where((i_v3030 == 0.0) | ~jnp.isfinite(i_v3030), 1e-300, i_v3030))
    v3067 = jnp.exp(jnp.clip(v3066, -709.0, 709.0))
    v3068 = 1.0 + v3067
    v3069 = jnp.log(jnp.maximum(v3068, 1e-300))
    v3070 = i_v3030 * v3069
    v3071 = i_v2951 - v3070
    v3076 = v2906 - i_v2951
    v3077 = jnp.divide(v3076, jnp.where((i_v3030 == 0.0) | ~jnp.isfinite(i_v3030), 1e-300, i_v3030))
    v3078 = jnp.exp(jnp.clip(v3077, -709.0, 709.0))
    v3079 = 1.0 + v3078
    v3080 = jnp.log(jnp.maximum(v3079, 1e-300))
    v3081 = i_v3030 * v3080
    v3082 = v2906 - v3081
    v3085 = jnp.where(v2987, v3071, v3082)
    v2853 = -v1914
    v3136 = jnp.where(v2798, v2853, v1914)
    v3140 = i_v2951 - v3085
    v3141 = v3136 < v3140
    v3146 = jnp.where(v3141, v3136, v3140)
    v3147 = i_v745 + v3146
    v3148 = -0.4 * v3147
    v3149 = v3085 < v3148
    v3170 = jnp.where(v3141, v3136, v3140)
    v3171 = i_v745 + v3170
    v3172 = -0.4 * v3171
    v3193 = jnp.where(v3149, v3172, v3085)
    v3179 = v3085 < i_v3178
    v3194 = jnp.where(v3179, i_v3178, v3085)
    v3192 = jnp.where(i_v2088, v3193, v3194)
    v3195 = 2.0 * v3192
    v3196 = i_v745 + v3195
    v2584 = ecorn * v2219
    v2615 = v2584 * v2178
    v2620 = jnp.where(i_v2556, v2615, ecorn)
    v2644 = v2625 - v2620
    v2745 = jnp.where(i_v2554, v2644, 1000.0)
    v2746 = i_v549 * v2745
    v2749 = v2746 > 100000.0
    v3347 = jnp.where(v2749, 100000.0, v2746)
    v3366 = jnp.divide(i_v3365, jnp.where((v3347 == 0.0) | ~jnp.isfinite(v3347), 1e-300, v3347))
    v3368 = 4.0 * v3347
    v3369 = v3368 * v3347
    v3370 = jnp.divide(v3369, jnp.where((i_v790 == 0.0) | ~jnp.isfinite(i_v790), 1e-300, i_v790))
    v3378 = v3366 * v3370
    v3380 = v3378 * v3378
    v3536 = 0.75 * v3380
    v3516 = 0.25 * v3380
    v3360 = jnp.divide(v3196, jnp.where((v3347 == 0.0) | ~jnp.isfinite(v3347), 1e-300, v3347))
    v3361 = 2.25 + v3360
    v3362 = i_v790 * v3361
    v3376 = v3362 * v3370
    v3532 = v3516 - v3376
    v3388 = 4.0 * v3376
    v3325 = i_v790 * v3196
    v3326 = v3325 * v3196
    v3327 = v3326 - v3196
    v3372 = v3327 * v3370
    v3389 = v3388 * v3372
    v3331 = i_v3330 * v3196
    v3332 = -1.0 + v3331
    v3374 = v3332 * v3370
    v3390 = v3374 * v3374
    v3391 = v3389 - v3390
    v3392 = v3372 * v3380
    v3393 = v3391 - v3392
    v3382 = -v3376
    v3384 = v3378 * v3374
    v3385 = 4.0 * v3372
    v3386 = v3384 - v3385
    v3395 = v3382 * v3382
    v3397 = v3395 * 0.3333333333333333
    v3398 = v3386 - v3397
    v3400 = 2.0 * v3398
    v3401 = v3386 + v3400
    v3402 = v3382 * v3401
    v3403 = jnp.divide(v3402, 9.0)
    v3404 = v3393 - v3403
    v3418 = v3404 < 0.0
    v3421 = -0.5 * v3404
    v3412 = 0.25 * v3404
    v3413 = v3412 * v3404
    v3406 = v3398 * v3398
    v3407 = v3406 * v3398
    v3409 = jnp.divide(v3407, 27.0)
    v3414 = v3413 + v3409
    v3416 = jnp.sqrt(jnp.maximum(v3414, 1e-300))
    v3423 = v3421 + v3416
    v3435 = -v3409
    v3431 = -0.5 * v3404
    v3433 = v3431 - v3416
    v3436 = jnp.divide(v3435, jnp.where((v3433 == 0.0) | ~jnp.isfinite(v3433), 1e-300, v3433))
    v3437 = jnp.where(v3418, v3423, v3436)
    v3439 = v3437 > 1e-06
    v3441 = jnp.power(jnp.maximum(v3437, 0.0), 0.3333333333333333)
    v3445 = v3437 < -1e-06
    v3447 = -v3437
    v3448 = jnp.power(jnp.maximum(v3447, 0.0), 0.3333333333333333)
    v3449 = -v3448
    v3451 = 10000.0 * v3437
    v3475 = jnp.where(v3445, v3449, v3451)
    v3474 = jnp.where(v3439, v3441, v3475)
    v3426 = -v3409
    v3427 = jnp.divide(v3426, jnp.where((v3423 == 0.0) | ~jnp.isfinite(v3423), 1e-300, v3423))
    v3454 = jnp.where(v3418, v3427, v3433)
    v3459 = v3454 > 1e-06
    v3461 = jnp.power(jnp.maximum(v3454, 0.0), 0.3333333333333333)
    v3465 = v3454 < -1e-06
    v3467 = -v3454
    v3468 = jnp.power(jnp.maximum(v3467, 0.0), 0.3333333333333333)
    v3469 = -v3468
    v3471 = 10000.0 * v3454
    v3481 = jnp.where(v3465, v3469, v3471)
    v3480 = jnp.where(v3459, v3461, v3481)
    v3482 = v3474 + v3480
    v3498 = v3382 * 0.3333333333333333
    v3499 = v3482 - v3498
    v3533 = v3532 + v3499
    v3538 = v3536 - v3533
    v3539 = 2.0 * v3376
    v3540 = v3538 - v3539
    v3557 = v3378 * v3376
    v3573 = 2.0 * v3374
    v3574 = v3557 - v3573
    v3576 = v3516 * v3378
    v3577 = v3574 - v3576
    v3534 = jnp.sqrt(jnp.maximum(v3533, 1e-300))
    v3578 = jnp.divide(v3577, v3534)
    v3580 = v3540 + v3578
    v3582 = v3580 > 0.0
    v3588 = -0.25 * v3378
    v3584 = jnp.sqrt(jnp.maximum(v3580, 1e-300))
    v3590 = v3584 + v3534
    v3591 = 0.5 * v3590
    v3592 = v3588 + v3591
    v3605 = -0.25 * v3378
    v3596 = v3540 - v3578
    v3598 = v3596 * v3596
    v3600 = v3598 + 0.0001
    v3601 = jnp.sqrt(jnp.maximum(v3600, 1e-300))
    v3602 = jnp.sqrt(jnp.maximum(v3601, 1e-300))
    v3607 = v3602 - v3534
    v3608 = 0.5 * v3607
    v3609 = v3605 + v3608
    v3933 = jnp.where(v3582, v3592, v3609)
    v3710 = v3192 > i_v3664
    v3814 = i_v802 - v3192
    v3815 = i_v790 * v3814
    v3817 = 2.0 * v3815
    v3818 = 1.0 - v3817
    v3819 = 2.0 * v3818
    v3821 = v3819 * v3814
    v3822 = 3.0 * v3815
    v3823 = 1.0 - v3822
    v3824 = 1.5 * v3815
    v3825 = 1.0 - v3824
    v3826 = jnp.sqrt(jnp.maximum(v3825, 1e-300))
    v3827 = v3823 + v3826
    v3828 = jnp.divide(v3821, jnp.where((v3827 == 0.0) | ~jnp.isfinite(v3827), 1e-300, v3827))
    v3833 = i_v3830 * v3196
    v3834 = 1.0 - v3833
    v3835 = 1.0 + v3833
    v3836 = jnp.sqrt(jnp.maximum(v3835, 1e-300))
    v3837 = v3834 + v3836
    v3840 = jnp.divide(v3837, jnp.where((i_v3839 == 0.0) | ~jnp.isfinite(i_v3839), 1e-300, i_v3839))
    v3934 = jnp.where(v3710, v3828, v3840)
    v3932 = jnp.where(v3223, v3933, v3934)
    v4084 = v3196 + v3932
    v4085 = jnp.sqrt(jnp.maximum(v4084, 1e-300))
    v4086 = i_v1056 * v4085
    v4363 = 2.0 * v4086
    v4364 = 1.0 - v4086
    v4365 = v4363 * v4364
    v4054 = v3932 + i_v89
    v4171 = jnp.divide(v4054, jnp.where((i_v549 == 0.0) | ~jnp.isfinite(i_v549), 1e-300, i_v549))
    v2621 = v2620 * v2620
    v2626 = i_v2624 * v2625
    v2627 = v2626 * v2625
    v2628 = v2621 + v2627
    v2629 = jnp.sqrt(jnp.maximum(v2628, 1e-300))
    v2631 = i_v2630 * v2625
    v2632 = v2629 - v2631
    v4198 = jnp.where(i_v2554, v2632, 0.0)
    v4226 = v4171 - v4198
    v4227 = 0.5 * v4226
    v4229 = v4227 * v3210
    v4236 = v4229 * v4229
    v2634 = du * v2632
    v2635 = jnp.divide(v2634, jnp.where((v2625 == 0.0) | ~jnp.isfinite(v2625), 1e-300, v2625))
    v4263 = jnp.where(i_v2554, v2635, 0.0)
    v4291 = v4236 + v4263
    v4292 = jnp.sqrt(jnp.maximum(v4291, 1e-300))
    v4355 = jnp.divide(v4229, v4292)
    v4232 = v4171 + v4198
    v4233 = 0.5 * v4232
    v4234 = v4233 * v3210
    v4294 = v4234 * v4234
    v4295 = v4294 + v4263
    v4296 = jnp.sqrt(jnp.maximum(v4295, 1e-300))
    v4356 = jnp.divide(v4234, v4296)
    v4357 = v4355 + v4356
    v4358 = 0.5 * v4357
    v4359 = v4358 * v3210
    v4360 = jnp.divide(v4359, jnp.where((i_v549 == 0.0) | ~jnp.isfinite(i_v549), 1e-300, i_v549))
    v4366 = v4360 * v4054
    v4298 = v4292 + v4296
    v2637 = v2632 * v2632
    v2638 = v2625 * v2625
    v2639 = jnp.divide(v2637, jnp.where((v2638 == 0.0) | ~jnp.isfinite(v2638), 1e-300, v2638))
    v2640 = 4.0 * v2635
    v2641 = v2639 + v2640
    v2642 = jnp.sqrt(jnp.maximum(v2641, 1e-300))
    v4325 = jnp.where(i_v2554, v2642, 0.0)
    v4353 = v4298 - v4325
    v4367 = 1.0 + v4353
    v4368 = jnp.divide(v4366, jnp.where((v4367 == 0.0) | ~jnp.isfinite(v4367), 1e-300, v4367))
    v4369 = 1.0 - v4368
    v4370 = v4365 * v4369
    v4371 = jnp.divide(v4370, jnp.where((v4054 == 0.0) | ~jnp.isfinite(v4054), 1e-300, v4054))
    v4372 = jnp.sqrt(jnp.maximum(v4371, 1e-300))
    v4375 = 2.0 * v4086
    v4376 = 1.0 - v4086
    v4377 = v4375 * v4376
    v4379 = jnp.divide(v4377, jnp.where((v4054 == 0.0) | ~jnp.isfinite(v4054), 1e-300, v4054))
    v4380 = jnp.sqrt(jnp.maximum(v4379, 1e-300))
    v4416 = jnp.where(v3223, v4372, v4380)
    v4415 = i_v790 * v4084
    v4417 = v4416 * v4416
    v4418 = jnp.divide(v4415, jnp.where((v4417 == 0.0) | ~jnp.isfinite(v4417), 1e-300, v4417))
    v4420 = v4418 - v4054
    v4590 = 2.0 * v3136
    v4591 = v4590 * v4054
    v4592 = v3136 - v4054
    v4594 = v4592 * v4592
    v4542 = ats * v3932
    v4543 = ats + v4054
    v4544 = jnp.divide(v4542, jnp.where((v4543 == 0.0) | ~jnp.isfinite(v4543), 1e-300, v4543))
    v4545 = i_v4485 + v4544
    v4547 = 4.0 * v4545
    v4548 = v4547 * v4545
    v4595 = v4594 + v4548
    v4596 = jnp.sqrt(jnp.maximum(v4595, 1e-300))
    v4597 = v3136 + v4054
    v4599 = v4597 * v4597
    v4600 = v4599 + v4548
    v4601 = jnp.sqrt(jnp.maximum(v4600, 1e-300))
    v4602 = v4596 + v4601
    v4603 = jnp.divide(v4591, jnp.where((v4602 == 0.0) | ~jnp.isfinite(v4602), 1e-300, v4602))
    v4608 = ats * v4603
    v4611 = jnp.divide(v4608, jnp.where((v4543 == 0.0) | ~jnp.isfinite(v4543), 1e-300, v4543))
    v4612 = i_v4485 + v4611
    v4613 = 4.0 * v4612
    v4614 = v4613 * v4612
    v4621 = v4594 + v4614
    v4622 = jnp.sqrt(jnp.maximum(v4621, 1e-300))
    v4626 = v4599 + v4614
    v4627 = jnp.sqrt(jnp.maximum(v4626, 1e-300))
    v4628 = v4622 + v4627
    v4629 = jnp.divide(v4591, jnp.where((v4628 == 0.0) | ~jnp.isfinite(v4628), 1e-300, v4628))
    v4636 = jnp.where(i_v4605, v4629, v4603)
    v4638 = v4420 + v4636
    v4639 = jnp.sqrt(jnp.maximum(v4638, 1e-300))
    v4640 = v4416 * v4639
    v4641 = 1.0 - v4640
    v4692 = 2.0 * v3136
    v4694 = v4692 * v3932
    v4695 = v3136 - v3932
    v4697 = v4695 * v4695
    v4699 = v4697 + i_v4485
    v4700 = jnp.sqrt(jnp.maximum(v4699, 1e-300))
    v4701 = v3136 + v3932
    v4703 = v4701 * v4701
    v4704 = v4703 + i_v4485
    v4705 = jnp.sqrt(jnp.maximum(v4704, 1e-300))
    v4706 = v4700 + v4705
    v4707 = jnp.divide(v4694, jnp.where((v4706 == 0.0) | ~jnp.isfinite(v4706), 1e-300, v4706))
    v4746 = v3196 + v4707
    v4747 = jnp.sqrt(jnp.maximum(v4746, 1e-300))
    v4748 = i_v1056 * v4747
    v4749 = 1.0 - v4748
    v4750 = jnp.where(i_v3928, v4641, v4749)
    v4754 = v4750 < grpo
    v4858 = jnp.where(v4754, grpo, v4750)
    v4860 = v4807 * v4858
    v4656 = jnp.divide(v4636, jnp.where((i_v549 == 0.0) | ~jnp.isfinite(i_v549), 1e-300, i_v549))
    v4663 = v4656 - v4198
    v4664 = 0.5 * v4663
    v4666 = v4664 * v3210
    v4671 = v4666 * v4666
    v4678 = v4671 + v4263
    v4679 = jnp.sqrt(jnp.maximum(v4678, 1e-300))
    v4668 = v4656 + v4198
    v4669 = 0.5 * v4668
    v4670 = v4669 * v3210
    v4680 = v4670 * v4670
    v4681 = v4680 + v4263
    v4682 = jnp.sqrt(jnp.maximum(v4681, 1e-300))
    v4683 = v4679 + v4682
    v4690 = v4683 - v4325
    v4864 = jnp.where(v3223, v4690, 0.0)
    v4713 = jnp.divide(v4707, jnp.where((i_v549 == 0.0) | ~jnp.isfinite(i_v549), 1e-300, i_v549))
    v4716 = v4713 - v4198
    v4717 = 0.5 * v4716
    v4719 = v4717 * v3210
    v4724 = v4719 * v4719
    v4727 = v4724 + v4263
    v4728 = jnp.sqrt(jnp.maximum(v4727, 1e-300))
    v4721 = v4713 + v4198
    v4722 = 0.5 * v4721
    v4723 = v4722 * v3210
    v4729 = v4723 * v4723
    v4730 = v4729 + v4263
    v4731 = jnp.sqrt(jnp.maximum(v4730, 1e-300))
    v4732 = v4728 + v4731
    v4735 = v4732 - v4325
    v4865 = jnp.where(v3223, v4735, 0.0)
    v4863 = jnp.where(i_v3928, v4864, v4865)
    v4867 = 1.0 + v4863
    v4868 = jnp.divide(v4860, jnp.where((v4867 == 0.0) | ~jnp.isfinite(v4867), 1e-300, v4867))
    v4934 = v4900 * v4868
    v4937 = jnp.where(i_v3928, v4636, v4707)
    v4941 = v4934 * v4937
    v7328 = v4941 * v1914
    v7374 = v7331 * v1919
    v7375 = v7328 + v7374
    v7379 = v7376 * v1924
    v7380 = v7375 + v7379
    v7383 = s.i_b_re1 * (signals.n1 - s.v_i1)
    v7384 = v7380 + v7383
    v7387 = s.i_b_re2 * (signals.n2 - s.v_i2)
    v7388 = v7384 + v7387
    v7561 = -v7388
    v13501 = jnp.where(i_v7553, v7561, 0.0)
    v13571 = v13036 + v13501
    v16279 = v13571 + 0.0
    v29225 = _mfactor * v16279
    v11648 = signals.dt * i_v11481
    v29226 = _mfactor * v11648
    v7838 = i_v1911 * v4941
    v17918 = v7838 + 0.0
    v17923 = v17918 + 0.0
    v20948 = jnp.where(i_v16281, v17923, v7838)
    v26419 = -v20948
    v26424 = v26419 - v21072
    v26439 = v26424 - s.i_b_re1
    v29228 = _mfactor * v26439
    v26425 = -v11304
    v29230 = _mfactor * v26425
    v26430 = v20948 - v21183
    v26445 = v26430 - s.i_b_re2
    v29232 = _mfactor * v26445
    v26431 = -v11307
    v29234 = _mfactor * v26431
    v14162 = s.i_b_re1 * i_v13670
    v2194 = v2060 * tc2rc
    v2195 = tc1rc + v2194
    v2196 = v2060 * v2195
    v2197 = 1.0 + v2196
    v2200 = v2197 < 0.11
    v2202 = v2197 - 0.01
    v2203 = 10.0 * v2202
    v2204 = v2203 - 1.0
    v2205 = jnp.exp(jnp.clip(v2204, -709.0, 709.0))
    v2206 = 0.1 * v2205
    v2207 = 0.01 + v2206
    v14248 = jnp.where(v2200, v2207, v2197)
    v14408 = v14162 * v14248
    v21437 = jnp.where(i_v14160, v14408, 0.0)
    v14796 = i_v13670 * v14248
    v14797 = jnp.divide(signals.n1 - s.v_i1, jnp.where((v14796 == 0.0) | ~jnp.isfinite(v14796), 1e-300, v14796))
    v21387 = jnp.where(i_v14160, 0.0, v14797)
    (
        v26435,
        v26434,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v21337, _t, _f),
        (
            v21437,
            signals.n1 - s.v_i1,
        ),
        (
            v21387,
            s.i_b_re1,
        ),
    )
    v26437 = v26435 - v26434
    v15454 = s.i_b_re2 * i_v15279
    v15457 = v15454 * v14248
    v21579 = jnp.where(i_v15452, v15457, 0.0)
    v15848 = i_v15279 * v14248
    v15849 = jnp.divide(signals.n2 - s.v_i2, jnp.where((v15848 == 0.0) | ~jnp.isfinite(v15848), 1e-300, v15848))
    v21532 = jnp.where(i_v15452, 0.0, v15849)
    (
        v26441,
        v26440,
    ) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v21485, _t, _f),
        (
            v21579,
            signals.n2 - s.v_i2,
        ),
        (
            v21532,
            s.i_b_re2,
        ),
    )
    v26443 = v26441 - v26440
    return {
        "n1": v29219,
        "nc": v29221,
        "n2": v29224,
        "dt": v29225,
        "v_i1": v29228,
        "v_i2": v29232,
        "i_b_re1": v26437,
        "i_b_re2": v26443,
    }, {"nc": v29223, "dt": v29226, "v_i1": v29230, "v_i2": v29234}


@R3_CMC.setup
def _R3_CMC_register_setup(*_a, **_kw):
    return _R3_CMC_setup(*_a, **_kw)
