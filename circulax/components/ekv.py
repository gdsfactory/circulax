"""Native JAX EKV 2.6 rev.15 current and charge equations.

Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,
code/ekv/vacode/ekv26.va. Regenerate using tests/data/va_models/ekv26/regenerate.py.
No OSDI or bosdi runtime is required. Original notice follows verbatim.

EKV MOS model version 2.6 rev.15 with documentation at: http://ekv.epfl.ch
Matthias Bucher, Christophe Lallement, Christian Enz, Fabien Theodoloz, Francois Krummenacher
Electronics Laboratories, Swiss Federal Institute of Technology Lausanne, Switzerland
This Verilog-A was developed by Wladek Grabinski with modifications
by Tiburon Design Automation (www.tiburon-da.com).
This software has been provided pursuant to a License Agreement containing restrictions on its use.
It may not be copied or distributed in any form or medium, disclosed to third parties,
reverse engineered or used in any manner not provided for in said License Agreement
except with the prior written authorization.
Licensed under the Educational Community License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

You may obtain a copy of the License at http://opensource.org/licenses/ECL-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.

$RCSfile: ekv.va,v $ $Revision: 1.9 $    $Date: 2003/12/17 01:20:10 $
$RCSfile: ekv.va,v $ $Revision: 2.6.15 $ $Date: 2020/05/29 11:50:10 $
"""

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, CPY001, E712, FBT003, PLR0915, PLR0917

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, States, component


def _EKV26_setup(
    TYPE: int = 1,
    Noise: int = 1,
    TEMP: float = float("nan"),
    TNOM: float = float("nan"),
    L: float = float("nan"),
    W: float = float("nan"),
    M: int = 1,
    NS: int = 1,
    DTEMP: float = float("nan"),
    AS: float = float("nan"),
    AD: float = float("nan"),
    PS: float = float("nan"),
    PD: float = float("nan"),
    NRS: float = float("nan"),
    NRD: float = float("nan"),
    COX: float = float("nan"),
    XJ: float = float("nan"),
    VTO: float = float("nan"),
    TCV: float = float("nan"),
    GAMMA: float = float("nan"),
    PHI: float = float("nan"),
    KP: float = float("nan"),
    BEX: float = float("nan"),
    THETA: float = float("nan"),
    E0: float = float("nan"),
    UCRIT: float = float("nan"),
    UCEX: float = float("nan"),
    LAMBDA: float = float("nan"),
    DL: float = float("nan"),
    DW: float = float("nan"),
    WETA: float = float("nan"),
    LETA: float = float("nan"),
    Q0: float = float("nan"),
    LK: float = float("nan"),
    IBA: float = float("nan"),
    IBB: float = float("nan"),
    IBBT: float = float("nan"),
    IBN: float = float("nan"),
    RSH: float = float("nan"),
    HDIF: float = float("nan"),
    AVTO: float = float("nan"),
    AKP: float = float("nan"),
    AGAMMA: float = float("nan"),
    AF: float = float("nan"),
    KF: float = float("nan"),
    Cgso: float = float("nan"),
    Cgdo: float = float("nan"),
    Cgbo: float = float("nan"),
    n: float = float("nan"),
    js: float = float("nan"),
    jsw: float = float("nan"),
    jswg: float = float("nan"),
    mj: float = float("nan"),
    mjsw: float = float("nan"),
    mjswg: float = float("nan"),
    pb: float = float("nan"),
    pbsw: float = float("nan"),
    pbswg: float = float("nan"),
    cj: float = float("nan"),
    cjsw: float = float("nan"),
    cjswg: float = float("nan"),
    xjbv: float = float("nan"),
    bv: float = float("nan"),
    njts: float = float("nan"),
    njtssw: float = float("nan"),
    njtsswg: float = float("nan"),
    vts: float = float("nan"),
    vtssw: float = float("nan"),
    vtsswg: float = float("nan"),
    xti: float = float("nan"),
    tcj: float = float("nan"),
    tcjsw: float = float("nan"),
    tcjswg: float = float("nan"),
    tpb: float = float("nan"),
    tpbsw: float = float("nan"),
    tpbswg: float = float("nan"),
    tnjts: float = float("nan"),
    tnjtssw: float = float("nan"),
    tnjtsswg: float = float("nan"),
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> jnp.ndarray:
    """Evaluate native EKV reference equations with parameter-presence flags."""
    given_AD = jnp.logical_not(jnp.isnan(AD))
    AD = jnp.where(given_AD, AD, 0.0)
    given_AF = jnp.logical_not(jnp.isnan(AF))
    AF = jnp.where(given_AF, AF, 1.0)
    given_AGAMMA = jnp.logical_not(jnp.isnan(AGAMMA))
    AGAMMA = jnp.where(given_AGAMMA, AGAMMA, 1e-6)
    given_AKP = jnp.logical_not(jnp.isnan(AKP))
    AKP = jnp.where(given_AKP, AKP, 1e-6)
    given_AS = jnp.logical_not(jnp.isnan(AS))
    AS = jnp.where(given_AS, AS, 0.0)
    given_AVTO = jnp.logical_not(jnp.isnan(AVTO))
    AVTO = jnp.where(given_AVTO, AVTO, 1e-6)
    given_BEX = jnp.logical_not(jnp.isnan(BEX))
    BEX = jnp.where(given_BEX, BEX, -1.5)
    given_COX = jnp.logical_not(jnp.isnan(COX))
    COX = jnp.where(given_COX, COX, 2.0e-3)
    given_Cgbo = jnp.logical_not(jnp.isnan(Cgbo))
    Cgbo = jnp.where(given_Cgbo, Cgbo, 4.0e-10)
    given_Cgdo = jnp.logical_not(jnp.isnan(Cgdo))
    Cgdo = jnp.where(given_Cgdo, Cgdo, 1.5e-10)
    given_Cgso = jnp.logical_not(jnp.isnan(Cgso))
    Cgso = jnp.where(given_Cgso, Cgso, 1.5e-10)
    given_DL = jnp.logical_not(jnp.isnan(DL))
    DL = jnp.where(given_DL, DL, -0.01e-6)
    given_DTEMP = jnp.logical_not(jnp.isnan(DTEMP))
    DTEMP = jnp.where(given_DTEMP, DTEMP, 0.0)
    given_DW = jnp.logical_not(jnp.isnan(DW))
    DW = jnp.where(given_DW, DW, -0.01e-6)
    given_E0 = jnp.logical_not(jnp.isnan(E0))
    E0 = jnp.where(given_E0, E0, 1.0e8)
    given_GAMMA = jnp.logical_not(jnp.isnan(GAMMA))
    GAMMA = jnp.where(given_GAMMA, GAMMA, 0.7)
    given_HDIF = jnp.logical_not(jnp.isnan(HDIF))
    HDIF = jnp.where(given_HDIF, HDIF, 0.0)
    given_IBA = jnp.logical_not(jnp.isnan(IBA))
    IBA = jnp.where(given_IBA, IBA, 5.0e8)
    given_IBB = jnp.logical_not(jnp.isnan(IBB))
    IBB = jnp.where(given_IBB, IBB, 4.0e8)
    given_IBBT = jnp.logical_not(jnp.isnan(IBBT))
    IBBT = jnp.where(given_IBBT, IBBT, 9.0e-4)
    given_IBN = jnp.logical_not(jnp.isnan(IBN))
    IBN = jnp.where(given_IBN, IBN, 1.0)
    given_KF = jnp.logical_not(jnp.isnan(KF))
    KF = jnp.where(given_KF, KF, 0.0)
    given_KP = jnp.logical_not(jnp.isnan(KP))
    KP = jnp.where(given_KP, KP, 150e-6)
    given_L = jnp.logical_not(jnp.isnan(L))
    L = jnp.where(given_L, L, 1.0e-5)
    given_LAMBDA = jnp.logical_not(jnp.isnan(LAMBDA))
    LAMBDA = jnp.where(given_LAMBDA, LAMBDA, 0.8)
    given_LETA = jnp.logical_not(jnp.isnan(LETA))
    LETA = jnp.where(given_LETA, LETA, 0.3)
    given_LK = jnp.logical_not(jnp.isnan(LK))
    LK = jnp.where(given_LK, LK, 0.4e-6)
    given_M = jnp.logical_not(jnp.isnan(M))
    M = jnp.where(given_M, M, 1)
    given_NRD = jnp.logical_not(jnp.isnan(NRD))
    NRD = jnp.where(given_NRD, NRD, 1.0)
    given_NRS = jnp.logical_not(jnp.isnan(NRS))
    NRS = jnp.where(given_NRS, NRS, 1.0)
    given_NS = jnp.logical_not(jnp.isnan(NS))
    NS = jnp.where(given_NS, NS, 1)
    given_Noise = jnp.logical_not(jnp.isnan(Noise))
    Noise = jnp.where(given_Noise, Noise, 1)
    given_PD = jnp.logical_not(jnp.isnan(PD))
    PD = jnp.where(given_PD, PD, 0.0)
    given_PHI = jnp.logical_not(jnp.isnan(PHI))
    PHI = jnp.where(given_PHI, PHI, 0.5)
    given_PS = jnp.logical_not(jnp.isnan(PS))
    PS = jnp.where(given_PS, PS, 0.0)
    given_Q0 = jnp.logical_not(jnp.isnan(Q0))
    Q0 = jnp.where(given_Q0, Q0, 230e-6)
    given_RSH = jnp.logical_not(jnp.isnan(RSH))
    RSH = jnp.where(given_RSH, RSH, 0.0)
    given_TCV = jnp.logical_not(jnp.isnan(TCV))
    TCV = jnp.where(given_TCV, TCV, 1.0e-3)
    given_TEMP = jnp.logical_not(jnp.isnan(TEMP))
    TEMP = jnp.where(given_TEMP, TEMP, 27.0)
    given_THETA = jnp.logical_not(jnp.isnan(THETA))
    THETA = jnp.where(given_THETA, THETA, 0.0)
    given_TNOM = jnp.logical_not(jnp.isnan(TNOM))
    TNOM = jnp.where(given_TNOM, TNOM, 27.0)
    given_TYPE = jnp.logical_not(jnp.isnan(TYPE))
    TYPE = jnp.where(given_TYPE, TYPE, 1)
    given_UCEX = jnp.logical_not(jnp.isnan(UCEX))
    UCEX = jnp.where(given_UCEX, UCEX, 0.8)
    given_UCRIT = jnp.logical_not(jnp.isnan(UCRIT))
    UCRIT = jnp.where(given_UCRIT, UCRIT, 2.0e6)
    given_VTO = jnp.logical_not(jnp.isnan(VTO))
    VTO = jnp.where(given_VTO, VTO, 0.5)
    given_W = jnp.logical_not(jnp.isnan(W))
    W = jnp.where(given_W, W, 1.0e-5)
    given_WETA = jnp.logical_not(jnp.isnan(WETA))
    WETA = jnp.where(given_WETA, WETA, 0.2)
    given_XJ = jnp.logical_not(jnp.isnan(XJ))
    XJ = jnp.where(given_XJ, XJ, 300e-9)
    given_bv = jnp.logical_not(jnp.isnan(bv))
    bv = jnp.where(given_bv, bv, 10.0)
    given_cj = jnp.logical_not(jnp.isnan(cj))
    cj = jnp.where(given_cj, cj, 1.0e-09)
    given_cjsw = jnp.logical_not(jnp.isnan(cjsw))
    cjsw = jnp.where(given_cjsw, cjsw, 1.0e-12)
    given_cjswg = jnp.logical_not(jnp.isnan(cjswg))
    cjswg = jnp.where(given_cjswg, cjswg, 1.0e-12)
    given_js = jnp.logical_not(jnp.isnan(js))
    js = jnp.where(given_js, js, 1.0e-09)
    given_jsw = jnp.logical_not(jnp.isnan(jsw))
    jsw = jnp.where(given_jsw, jsw, 1.0e-12)
    given_jswg = jnp.logical_not(jnp.isnan(jswg))
    jswg = jnp.where(given_jswg, jswg, 1.0e-12)
    given_mj = jnp.logical_not(jnp.isnan(mj))
    mj = jnp.where(given_mj, mj, 0.900)
    given_mjsw = jnp.logical_not(jnp.isnan(mjsw))
    mjsw = jnp.where(given_mjsw, mjsw, 0.700)
    given_mjswg = jnp.logical_not(jnp.isnan(mjswg))
    mjswg = jnp.where(given_mjswg, mjswg, 0.700)
    given_n = jnp.logical_not(jnp.isnan(n))
    n = jnp.where(given_n, n, 1.0)
    given_njts = jnp.logical_not(jnp.isnan(njts))
    njts = jnp.where(given_njts, njts, 1.0)
    given_njtssw = jnp.logical_not(jnp.isnan(njtssw))
    njtssw = jnp.where(given_njtssw, njtssw, 1.0)
    given_njtsswg = jnp.logical_not(jnp.isnan(njtsswg))
    njtsswg = jnp.where(given_njtsswg, njtsswg, 1.0)
    given_pb = jnp.logical_not(jnp.isnan(pb))
    pb = jnp.where(given_pb, pb, 0.800)
    given_pbsw = jnp.logical_not(jnp.isnan(pbsw))
    pbsw = jnp.where(given_pbsw, pbsw, 0.600)
    given_pbswg = jnp.logical_not(jnp.isnan(pbswg))
    pbswg = jnp.where(given_pbswg, pbswg, 0.600)
    given_tcj = jnp.logical_not(jnp.isnan(tcj))
    tcj = jnp.where(given_tcj, tcj, 0.0)
    given_tcjsw = jnp.logical_not(jnp.isnan(tcjsw))
    tcjsw = jnp.where(given_tcjsw, tcjsw, 0.0)
    given_tcjswg = jnp.logical_not(jnp.isnan(tcjswg))
    tcjswg = jnp.where(given_tcjswg, tcjswg, 0.0)
    given_tnjts = jnp.logical_not(jnp.isnan(tnjts))
    tnjts = jnp.where(given_tnjts, tnjts, 0.0)
    given_tnjtssw = jnp.logical_not(jnp.isnan(tnjtssw))
    tnjtssw = jnp.where(given_tnjtssw, tnjtssw, 0.0)
    given_tnjtsswg = jnp.logical_not(jnp.isnan(tnjtsswg))
    tnjtsswg = jnp.where(given_tnjtsswg, tnjtsswg, 0.0)
    given_tpb = jnp.logical_not(jnp.isnan(tpb))
    tpb = jnp.where(given_tpb, tpb, 0.0)
    given_tpbsw = jnp.logical_not(jnp.isnan(tpbsw))
    tpbsw = jnp.where(given_tpbsw, tpbsw, 0.0)
    given_tpbswg = jnp.logical_not(jnp.isnan(tpbswg))
    tpbswg = jnp.where(given_tpbswg, tpbswg, 0.0)
    given_vts = jnp.logical_not(jnp.isnan(vts))
    vts = jnp.where(given_vts, vts, 0.0)
    given_vtssw = jnp.logical_not(jnp.isnan(vtssw))
    vtssw = jnp.where(given_vtssw, vtssw, 0.0)
    given_vtsswg = jnp.logical_not(jnp.isnan(vtsswg))
    vtsswg = jnp.where(given_vtsswg, vtsswg, 0.0)
    given_xjbv = jnp.logical_not(jnp.isnan(xjbv))
    xjbv = jnp.where(given_xjbv, xjbv, 0.0)
    given_xti = jnp.logical_not(jnp.isnan(xti))
    xti = jnp.where(given_xti, xti, 3.0)
    i_v4746 = Noise != 0
    i_v76 = jnp.where((given_TEMP == False), (_temperature + DTEMP), (TEMP + 273.15))
    i_v78 = 8.617332384960955e-05 * i_v76
    i_v89 = i_v78 * i_v78
    i_v91 = i_v89 + i_v89
    i_v184 = W + DW
    i_v180 = L + DL
    i_v195 = i_v184 * i_v180
    i_v197 = jnp.divide(1.0, (jnp.sqrt(jnp.maximum(i_v195, 1e-300))))
    i_v269 = jnp.where((AGAMMA != 1e-06), (GAMMA + ((AGAMMA - 1e-06) * i_v197)), GAMMA)
    i_v49 = TYPE > 0
    i_v105 = jnp.where((given_TNOM == False), 300.15, (TNOM + 273.15))
    i_v112 = i_v76 - i_v105
    i_v119 = VTO - (TCV * i_v112)
    i_v382 = jnp.where(
        i_v49,
        (jnp.where((AVTO != 1e-06), ((i_v197 * (AVTO - 1e-06)) + i_v119), i_v119)),
        (jnp.where((AVTO != 1e-06), ((i_v197 * (1e-06 - AVTO)) - i_v119), (-i_v119))),
    )
    i_v46 = jnp.divide((Q0 + Q0), jnp.where((COX == 0.0) | ~jnp.isfinite(COX), 1e-300, COX))
    i_v337 = 0.28 * ((jnp.divide(i_v180, jnp.where(((LK * NS) == 0.0) | ~jnp.isfinite((LK * NS)), 1e-300, (LK * NS)))) - 0.1)
    i_v345 = 1.0 + (0.5 * (i_v337 + (jnp.sqrt(jnp.maximum(((i_v337 * i_v337) + 0.001936), 1e-300)))))
    i_v346 = jnp.divide(1.0, jnp.where((i_v345 == 0.0) | ~jnp.isfinite(i_v345), 1e-300, i_v345))
    i_v390 = jnp.where((i_v46 == 0.0), 0.0, ((i_v46 * i_v346) * i_v346))
    i_v114 = jnp.divide(i_v76, jnp.where((i_v105 == 0.0) | ~jnp.isfinite(i_v105), 1e-300, i_v105))
    i_v140 = jnp.log(jnp.maximum(i_v114, 1e-300))
    i_v110 = 1.16 - (
        jnp.divide(
            ((0.000702 * i_v105) * i_v105),
            jnp.where(((i_v105 + 1108.0) == 0.0) | ~jnp.isfinite(i_v105 + 1108.0), 1e-300, (i_v105 + 1108.0)),
        )
    )
    i_v103 = 1.16 - (
        jnp.divide(
            ((0.000702 * i_v76) * i_v76),
            jnp.where(((i_v76 + 1108.0) == 0.0) | ~jnp.isfinite(i_v76 + 1108.0), 1e-300, (i_v76 + 1108.0)),
        )
    )
    i_v149 = ((((PHI * i_v114) - ((3.0 * i_v78) * i_v140)) - (i_v110 * i_v114)) + i_v103) - 0.2
    i_v157 = (0.5 * (i_v149 + (jnp.sqrt(jnp.maximum(((i_v149 * i_v149) + i_v89), 1e-300))))) + 0.2
    i_v286 = i_v269 * (jnp.sqrt(jnp.maximum(i_v157, 1e-300)))
    i_v94 = 16.0 * i_v89
    i_v445 = 2.0 * i_v94
    i_v21 = jnp.divide(1.035939974589e-10, jnp.where((COX == 0.0) | ~jnp.isfinite(COX), 1e-300, COX))
    i_v560 = jnp.divide(((i_v21 * LETA) * NS), jnp.where((i_v180 == 0.0) | ~jnp.isfinite(i_v180), 1e-300, i_v180))
    i_v521 = jnp.divide((((3.0 * i_v21) * WETA) * M), jnp.where((i_v184 == 0.0) | ~jnp.isfinite(i_v184), 1e-300, i_v184))
    i_v570 = (0.25 * i_v269) * i_v269
    i_v575 = 0.5 * i_v269
    i_v81 = 0.1 * i_v78
    i_v85 = i_v78 + i_v78
    i_v87 = i_v85 + i_v85
    i_v1371 = E0 == 0.0
    i_v124 = KP * (jnp.power(jnp.maximum(i_v114, 0.0), BEX))
    i_v258 = i_v184 * (jnp.where((AKP != 1e-06), (i_v124 * (1.0 + ((AKP - 1e-06) * i_v197))), i_v124))
    i_v25 = jnp.sqrt(jnp.maximum((i_v21 * XJ), 1e-300))
    i_v28 = i_v25 * LAMBDA
    i_v129 = UCRIT * (jnp.power(jnp.maximum(i_v114, 0.0), UCEX))
    i_v186 = i_v129 * i_v180
    i_v83 = jnp.divide(1.0, jnp.where((i_v78 == 0.0) | ~jnp.isfinite(i_v78), 1e-300, i_v78))
    i_v778 = jnp.divide(i_v78, jnp.where((i_v186 == 0.0) | ~jnp.isfinite(i_v186), 1e-300, i_v186))
    i_v171 = i_v25 * i_v129
    i_v160 = jnp.divide(1.0, jnp.where((i_v129 == 0.0) | ~jnp.isfinite(i_v129), 1e-300, i_v129))
    i_v1137 = 0.1 * i_v180
    i_v1140 = i_v1137 * i_v1137
    i_v41 = 1.035939974589e-10 * E0
    i_v42 = jnp.divide(COX, jnp.where((i_v41 == 0.0) | ~jnp.isfinite(i_v41), 1e-300, i_v41))
    i_v1657 = i_v258 * (1.0 + (i_v42 * i_v286))
    i_v1366 = -0.5 * i_v269
    i_v52 = jnp.where(i_v49, 0.5, 0.3333333333333)
    i_v193 = i_v78 * ((jnp.log(jnp.maximum(((0.5 * i_v186) * i_v83), 1e-300))) - 0.6)
    i_v3409 = -i_v269
    i_v2047 = -i_v560
    i_v2478 = (i_v87 + i_v87) * LAMBDA
    i_v3650 = given_HDIF == True
    i_v3656 = jnp.divide((RSH * HDIF), jnp.where((W == 0.0) | ~jnp.isfinite(W), 1e-300, W))
    (i_v3669, i_v3675) = jax.tree_util.tree_map(
        lambda _t, _f: jnp.where(i_v3650, _t, _f), (i_v3656, i_v3656), ((RSH * NRS), (RSH * NRD))
    )
    i_v38 = IBN + IBN
    i_v135 = IBB * (1.0 + (IBBT * i_v112))
    i_v176 = jnp.divide(IBA, jnp.where((i_v135 == 0.0) | ~jnp.isfinite(i_v135), 1e-300, i_v135))
    i_v3893 = i_v176 > 0.0
    i_v3968 = -(i_v25 * i_v135)
    i_v5339 = jnp.exp(
        jnp.clip(
            (
                jnp.divide(
                    (
                        (
                            (
                                jnp.divide(
                                    i_v110,
                                    jnp.where(
                                        ((8.617332384960955e-05 * i_v105) == 0.0) | ~jnp.isfinite((8.617332384960955e-05 * i_v105)),
                                        1e-300,
                                        (8.617332384960955e-05 * i_v105),
                                    ),
                                )
                            )
                            - (jnp.divide(i_v103, jnp.where((i_v78 == 0.0) | ~jnp.isfinite(i_v78), 1e-300, i_v78)))
                        )
                        + (xti * i_v140)
                    ),
                    jnp.where((n == 0.0) | ~jnp.isfinite(n), 1e-300, n),
                )
            ),
            -709.0,
            709.0,
        )
    )
    i_v5342 = js * i_v5339
    i_v5512 = jnp.where((jnp.where((given_AD == False), i_v3650, False)), ((2.0 * HDIF) * i_v184), AD)
    i_v5515 = i_v5342 * i_v5512
    i_v5345 = jsw * i_v5339
    i_v5516 = jnp.where((jnp.where((given_PD == False), i_v3650, False)), ((4.0 * HDIF) + i_v184), PD)
    i_v5517 = i_v5345 * i_v5516
    i_v5348 = jswg * i_v5339
    i_v5521 = i_v5348 * i_v184
    i_v5522 = (i_v5515 + i_v5517) + i_v5521
    i_v5506 = i_v78 * n
    i_v5573 = (-i_v184) * i_v5348
    i_v5486 = i_v114 - 1.0
    i_v5593 = i_v78 * (njtsswg * (1.0 + (i_v5486 * tnjtsswg)))
    i_v5669 = i_v78 * (njtssw * (1.0 + (i_v5486 * tnjtssw)))
    i_v5760 = i_v78 * (njts * (1.0 + (i_v5486 * tnjts)))
    i_v7552 = M * TYPE
    i_v4024 = i_v195 * COX
    i_v5471 = cj * (1.0 + (tcj * i_v112))
    i_v6887 = i_v5471 * i_v5512
    i_v6889 = -mj
    i_v5455 = pb - (tpb * i_v112)
    i_v7260 = i_v5471 * i_v5512
    i_v5477 = cjsw * (1.0 + (tcjsw * i_v112))
    i_v7051 = i_v5477 * i_v5516
    i_v7053 = -mjsw
    i_v5460 = pbsw - (tpbsw * i_v112)
    i_v7269 = i_v5477 * i_v5516
    i_v5483 = cjswg * (1.0 + (tcjswg * i_v112))
    i_v7193 = i_v5483 * i_v184
    i_v7195 = -mjswg
    i_v5465 = pbswg - (tpbswg * i_v112)
    i_v7277 = i_v5483 * i_v184
    i_v7457 = Cgdo * i_v184
    i_v7452 = Cgso * i_v184
    i_v7548 = Cgbo * i_v180
    i_v6055 = jnp.where((jnp.where((given_AS == False), i_v3650, False)), ((2.0 * HDIF) * i_v184), AS)
    i_v6071 = i_v5342 * i_v6055
    i_v6113 = jnp.where((jnp.where((given_PS == False), i_v3650, False)), ((4.0 * HDIF) + i_v184), PS)
    i_v6127 = i_v5345 * i_v6113
    i_v6172 = (i_v6071 + i_v6127) + i_v5521
    i_v7346 = i_v5471 * i_v6055
    i_v7347 = -mj
    i_v7407 = i_v5471 * i_v6055
    i_v7379 = i_v5477 * i_v6113
    i_v7380 = -mjsw
    i_v7416 = i_v5477 * i_v6113
    i_v7394 = i_v5483 * i_v184
    i_v7395 = -mjswg
    i_v7424 = i_v5483 * i_v184
    return jnp.array(
        [
            i_v4746,
            i_v78,
            i_v91,
            i_v180,
            i_v269,
            i_v382,
            i_v390,
            i_v114,
            i_v157,
            i_v286,
            i_v94,
            i_v445,
            i_v560,
            i_v521,
            i_v570,
            i_v575,
            i_v81,
            i_v87,
            i_v1371,
            i_v258,
            i_v28,
            i_v186,
            i_v83,
            i_v778,
            i_v171,
            i_v160,
            i_v1140,
            i_v42,
            i_v1657,
            i_v1366,
            i_v52,
            i_v193,
            i_v3409,
            i_v2047,
            i_v2478,
            i_v3669,
            i_v3675,
            i_v38,
            i_v176,
            i_v3893,
            i_v3968,
            i_v5515,
            i_v5517,
            i_v5522,
            i_v5506,
            1e-12,
            i_v5573,
            i_v5593,
            i_v5669,
            i_v5760,
            i_v7552,
            i_v4024,
            i_v6887,
            i_v6889,
            i_v5455,
            i_v7260,
            i_v7051,
            i_v7053,
            i_v5460,
            i_v7269,
            i_v7193,
            i_v7195,
            i_v5465,
            i_v7277,
            i_v7457,
            i_v7452,
            i_v7548,
            i_v6071,
            i_v6127,
            i_v6172,
            i_v7346,
            i_v7347,
            i_v7407,
            i_v7379,
            i_v7380,
            i_v7416,
            i_v7394,
            i_v7395,
            i_v7424,
        ]
    )


@component(ports=("d", "g", "s", "b"))
def EKV26(
    signals: Signals,
    s: States,
    init,
    TYPE: int = 1,
    Noise: int = 1,
    TEMP: float = float("nan"),
    TNOM: float = float("nan"),
    L: float = float("nan"),
    W: float = float("nan"),
    M: int = 1,
    NS: int = 1,
    DTEMP: float = float("nan"),
    AS: float = float("nan"),
    AD: float = float("nan"),
    PS: float = float("nan"),
    PD: float = float("nan"),
    NRS: float = float("nan"),
    NRD: float = float("nan"),
    COX: float = float("nan"),
    XJ: float = float("nan"),
    VTO: float = float("nan"),
    TCV: float = float("nan"),
    GAMMA: float = float("nan"),
    PHI: float = float("nan"),
    KP: float = float("nan"),
    BEX: float = float("nan"),
    THETA: float = float("nan"),
    E0: float = float("nan"),
    UCRIT: float = float("nan"),
    UCEX: float = float("nan"),
    LAMBDA: float = float("nan"),
    DL: float = float("nan"),
    DW: float = float("nan"),
    WETA: float = float("nan"),
    LETA: float = float("nan"),
    Q0: float = float("nan"),
    LK: float = float("nan"),
    IBA: float = float("nan"),
    IBB: float = float("nan"),
    IBBT: float = float("nan"),
    IBN: float = float("nan"),
    RSH: float = float("nan"),
    HDIF: float = float("nan"),
    AVTO: float = float("nan"),
    AKP: float = float("nan"),
    AGAMMA: float = float("nan"),
    AF: float = float("nan"),
    KF: float = float("nan"),
    Cgso: float = float("nan"),
    Cgdo: float = float("nan"),
    Cgbo: float = float("nan"),
    n: float = float("nan"),
    js: float = float("nan"),
    jsw: float = float("nan"),
    jswg: float = float("nan"),
    mj: float = float("nan"),
    mjsw: float = float("nan"),
    mjswg: float = float("nan"),
    pb: float = float("nan"),
    pbsw: float = float("nan"),
    pbswg: float = float("nan"),
    cj: float = float("nan"),
    cjsw: float = float("nan"),
    cjswg: float = float("nan"),
    xjbv: float = float("nan"),
    bv: float = float("nan"),
    njts: float = float("nan"),
    njtssw: float = float("nan"),
    njtsswg: float = float("nan"),
    vts: float = float("nan"),
    vtssw: float = float("nan"),
    vtsswg: float = float("nan"),
    xti: float = float("nan"),
    tcj: float = float("nan"),
    tcjsw: float = float("nan"),
    tcjswg: float = float("nan"),
    tpb: float = float("nan"),
    tpbsw: float = float("nan"),
    tpbswg: float = float("nan"),
    tnjts: float = float("nan"),
    tnjtssw: float = float("nan"),
    tnjtsswg: float = float("nan"),
    _temperature: float = 300.15,
    _mfactor: float = 1.0,
) -> PhysicsReturn:
    """Four-terminal NMOS/PMOS compact model with intrinsic and overlap charge.

    Reference parameters retain their units and polarity conventions; see the model reference for supported settings.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 2.6 rev.15, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `ekv26.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/ekv/vacode/ekv26.va).
    Attribution: EPFL model authors and Wladek Grabinski, with Tiburon Design Automation modifications; the original
    mixed license notice is retained.
    The retained source and includes are under `tests/data/va_models/ekv26/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/ekv/vacode/ekv26.va),
    with comparison cases in `tests/test_ekv.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        ekv.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.ekv import EKV26

        jax.config.update("jax_enable_x64", True)
        model = EKV26()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(EKV26.__name__, "ports:", model.ports)
        ```

    """
    given_AD = jnp.logical_not(jnp.isnan(AD))
    AD = jnp.where(given_AD, AD, 0.0)
    given_AF = jnp.logical_not(jnp.isnan(AF))
    AF = jnp.where(given_AF, AF, 1.0)
    given_AGAMMA = jnp.logical_not(jnp.isnan(AGAMMA))
    AGAMMA = jnp.where(given_AGAMMA, AGAMMA, 1e-6)
    given_AKP = jnp.logical_not(jnp.isnan(AKP))
    AKP = jnp.where(given_AKP, AKP, 1e-6)
    given_AS = jnp.logical_not(jnp.isnan(AS))
    AS = jnp.where(given_AS, AS, 0.0)
    given_AVTO = jnp.logical_not(jnp.isnan(AVTO))
    AVTO = jnp.where(given_AVTO, AVTO, 1e-6)
    given_BEX = jnp.logical_not(jnp.isnan(BEX))
    BEX = jnp.where(given_BEX, BEX, -1.5)
    given_COX = jnp.logical_not(jnp.isnan(COX))
    COX = jnp.where(given_COX, COX, 2.0e-3)
    given_Cgbo = jnp.logical_not(jnp.isnan(Cgbo))
    Cgbo = jnp.where(given_Cgbo, Cgbo, 4.0e-10)
    given_Cgdo = jnp.logical_not(jnp.isnan(Cgdo))
    Cgdo = jnp.where(given_Cgdo, Cgdo, 1.5e-10)
    given_Cgso = jnp.logical_not(jnp.isnan(Cgso))
    Cgso = jnp.where(given_Cgso, Cgso, 1.5e-10)
    given_DL = jnp.logical_not(jnp.isnan(DL))
    DL = jnp.where(given_DL, DL, -0.01e-6)
    given_DTEMP = jnp.logical_not(jnp.isnan(DTEMP))
    DTEMP = jnp.where(given_DTEMP, DTEMP, 0.0)
    given_DW = jnp.logical_not(jnp.isnan(DW))
    DW = jnp.where(given_DW, DW, -0.01e-6)
    given_E0 = jnp.logical_not(jnp.isnan(E0))
    E0 = jnp.where(given_E0, E0, 1.0e8)
    given_GAMMA = jnp.logical_not(jnp.isnan(GAMMA))
    GAMMA = jnp.where(given_GAMMA, GAMMA, 0.7)
    given_HDIF = jnp.logical_not(jnp.isnan(HDIF))
    HDIF = jnp.where(given_HDIF, HDIF, 0.0)
    given_IBA = jnp.logical_not(jnp.isnan(IBA))
    IBA = jnp.where(given_IBA, IBA, 5.0e8)
    given_IBB = jnp.logical_not(jnp.isnan(IBB))
    IBB = jnp.where(given_IBB, IBB, 4.0e8)
    given_IBBT = jnp.logical_not(jnp.isnan(IBBT))
    IBBT = jnp.where(given_IBBT, IBBT, 9.0e-4)
    given_IBN = jnp.logical_not(jnp.isnan(IBN))
    IBN = jnp.where(given_IBN, IBN, 1.0)
    given_KF = jnp.logical_not(jnp.isnan(KF))
    KF = jnp.where(given_KF, KF, 0.0)
    given_KP = jnp.logical_not(jnp.isnan(KP))
    KP = jnp.where(given_KP, KP, 150e-6)
    given_L = jnp.logical_not(jnp.isnan(L))
    L = jnp.where(given_L, L, 1.0e-5)
    given_LAMBDA = jnp.logical_not(jnp.isnan(LAMBDA))
    LAMBDA = jnp.where(given_LAMBDA, LAMBDA, 0.8)
    given_LETA = jnp.logical_not(jnp.isnan(LETA))
    LETA = jnp.where(given_LETA, LETA, 0.3)
    given_LK = jnp.logical_not(jnp.isnan(LK))
    LK = jnp.where(given_LK, LK, 0.4e-6)
    given_M = jnp.logical_not(jnp.isnan(M))
    M = jnp.where(given_M, M, 1)
    given_NRD = jnp.logical_not(jnp.isnan(NRD))
    NRD = jnp.where(given_NRD, NRD, 1.0)
    given_NRS = jnp.logical_not(jnp.isnan(NRS))
    NRS = jnp.where(given_NRS, NRS, 1.0)
    given_NS = jnp.logical_not(jnp.isnan(NS))
    NS = jnp.where(given_NS, NS, 1)
    given_Noise = jnp.logical_not(jnp.isnan(Noise))
    Noise = jnp.where(given_Noise, Noise, 1)
    given_PD = jnp.logical_not(jnp.isnan(PD))
    PD = jnp.where(given_PD, PD, 0.0)
    given_PHI = jnp.logical_not(jnp.isnan(PHI))
    PHI = jnp.where(given_PHI, PHI, 0.5)
    given_PS = jnp.logical_not(jnp.isnan(PS))
    PS = jnp.where(given_PS, PS, 0.0)
    given_Q0 = jnp.logical_not(jnp.isnan(Q0))
    Q0 = jnp.where(given_Q0, Q0, 230e-6)
    given_RSH = jnp.logical_not(jnp.isnan(RSH))
    RSH = jnp.where(given_RSH, RSH, 0.0)
    given_TCV = jnp.logical_not(jnp.isnan(TCV))
    TCV = jnp.where(given_TCV, TCV, 1.0e-3)
    given_TEMP = jnp.logical_not(jnp.isnan(TEMP))
    TEMP = jnp.where(given_TEMP, TEMP, 27.0)
    given_THETA = jnp.logical_not(jnp.isnan(THETA))
    THETA = jnp.where(given_THETA, THETA, 0.0)
    given_TNOM = jnp.logical_not(jnp.isnan(TNOM))
    TNOM = jnp.where(given_TNOM, TNOM, 27.0)
    given_TYPE = jnp.logical_not(jnp.isnan(TYPE))
    TYPE = jnp.where(given_TYPE, TYPE, 1)
    given_UCEX = jnp.logical_not(jnp.isnan(UCEX))
    UCEX = jnp.where(given_UCEX, UCEX, 0.8)
    given_UCRIT = jnp.logical_not(jnp.isnan(UCRIT))
    UCRIT = jnp.where(given_UCRIT, UCRIT, 2.0e6)
    given_VTO = jnp.logical_not(jnp.isnan(VTO))
    VTO = jnp.where(given_VTO, VTO, 0.5)
    given_W = jnp.logical_not(jnp.isnan(W))
    W = jnp.where(given_W, W, 1.0e-5)
    given_WETA = jnp.logical_not(jnp.isnan(WETA))
    WETA = jnp.where(given_WETA, WETA, 0.2)
    given_XJ = jnp.logical_not(jnp.isnan(XJ))
    XJ = jnp.where(given_XJ, XJ, 300e-9)
    given_bv = jnp.logical_not(jnp.isnan(bv))
    bv = jnp.where(given_bv, bv, 10.0)
    given_cj = jnp.logical_not(jnp.isnan(cj))
    cj = jnp.where(given_cj, cj, 1.0e-09)
    given_cjsw = jnp.logical_not(jnp.isnan(cjsw))
    cjsw = jnp.where(given_cjsw, cjsw, 1.0e-12)
    given_cjswg = jnp.logical_not(jnp.isnan(cjswg))
    cjswg = jnp.where(given_cjswg, cjswg, 1.0e-12)
    given_js = jnp.logical_not(jnp.isnan(js))
    js = jnp.where(given_js, js, 1.0e-09)
    given_jsw = jnp.logical_not(jnp.isnan(jsw))
    jsw = jnp.where(given_jsw, jsw, 1.0e-12)
    given_jswg = jnp.logical_not(jnp.isnan(jswg))
    jswg = jnp.where(given_jswg, jswg, 1.0e-12)
    given_mj = jnp.logical_not(jnp.isnan(mj))
    mj = jnp.where(given_mj, mj, 0.900)
    given_mjsw = jnp.logical_not(jnp.isnan(mjsw))
    mjsw = jnp.where(given_mjsw, mjsw, 0.700)
    given_mjswg = jnp.logical_not(jnp.isnan(mjswg))
    mjswg = jnp.where(given_mjswg, mjswg, 0.700)
    given_n = jnp.logical_not(jnp.isnan(n))
    n = jnp.where(given_n, n, 1.0)
    given_njts = jnp.logical_not(jnp.isnan(njts))
    njts = jnp.where(given_njts, njts, 1.0)
    given_njtssw = jnp.logical_not(jnp.isnan(njtssw))
    njtssw = jnp.where(given_njtssw, njtssw, 1.0)
    given_njtsswg = jnp.logical_not(jnp.isnan(njtsswg))
    njtsswg = jnp.where(given_njtsswg, njtsswg, 1.0)
    given_pb = jnp.logical_not(jnp.isnan(pb))
    pb = jnp.where(given_pb, pb, 0.800)
    given_pbsw = jnp.logical_not(jnp.isnan(pbsw))
    pbsw = jnp.where(given_pbsw, pbsw, 0.600)
    given_pbswg = jnp.logical_not(jnp.isnan(pbswg))
    pbswg = jnp.where(given_pbswg, pbswg, 0.600)
    given_tcj = jnp.logical_not(jnp.isnan(tcj))
    tcj = jnp.where(given_tcj, tcj, 0.0)
    given_tcjsw = jnp.logical_not(jnp.isnan(tcjsw))
    tcjsw = jnp.where(given_tcjsw, tcjsw, 0.0)
    given_tcjswg = jnp.logical_not(jnp.isnan(tcjswg))
    tcjswg = jnp.where(given_tcjswg, tcjswg, 0.0)
    given_tnjts = jnp.logical_not(jnp.isnan(tnjts))
    tnjts = jnp.where(given_tnjts, tnjts, 0.0)
    given_tnjtssw = jnp.logical_not(jnp.isnan(tnjtssw))
    tnjtssw = jnp.where(given_tnjtssw, tnjtssw, 0.0)
    given_tnjtsswg = jnp.logical_not(jnp.isnan(tnjtsswg))
    tnjtsswg = jnp.where(given_tnjtsswg, tnjtsswg, 0.0)
    given_tpb = jnp.logical_not(jnp.isnan(tpb))
    tpb = jnp.where(given_tpb, tpb, 0.0)
    given_tpbsw = jnp.logical_not(jnp.isnan(tpbsw))
    tpbsw = jnp.where(given_tpbsw, tpbsw, 0.0)
    given_tpbswg = jnp.logical_not(jnp.isnan(tpbswg))
    tpbswg = jnp.where(given_tpbswg, tpbswg, 0.0)
    given_vts = jnp.logical_not(jnp.isnan(vts))
    vts = jnp.where(given_vts, vts, 0.0)
    given_vtssw = jnp.logical_not(jnp.isnan(vtssw))
    vtssw = jnp.where(given_vtssw, vtssw, 0.0)
    given_vtsswg = jnp.logical_not(jnp.isnan(vtsswg))
    vtsswg = jnp.where(given_vtsswg, vtsswg, 0.0)
    given_xjbv = jnp.logical_not(jnp.isnan(xjbv))
    xjbv = jnp.where(given_xjbv, xjbv, 0.0)
    given_xti = jnp.logical_not(jnp.isnan(xti))
    xti = jnp.where(given_xti, xti, 3.0)
    """Auto-generated from Verilog-A."""
    i_v4746 = init[0]
    i_v78 = init[1]
    i_v91 = init[2]
    i_v180 = init[3]
    i_v269 = init[4]
    i_v382 = init[5]
    i_v390 = init[6]
    i_v114 = init[7]
    i_v157 = init[8]
    i_v286 = init[9]
    i_v94 = init[10]
    i_v445 = init[11]
    i_v560 = init[12]
    i_v521 = init[13]
    i_v570 = init[14]
    i_v575 = init[15]
    i_v81 = init[16]
    i_v87 = init[17]
    i_v1371 = init[18]
    i_v258 = init[19]
    i_v28 = init[20]
    i_v186 = init[21]
    i_v83 = init[22]
    i_v778 = init[23]
    i_v171 = init[24]
    i_v160 = init[25]
    i_v1140 = init[26]
    i_v42 = init[27]
    i_v1657 = init[28]
    i_v1366 = init[29]
    i_v52 = init[30]
    i_v193 = init[31]
    i_v3409 = init[32]
    i_v2047 = init[33]
    i_v2478 = init[34]
    i_v3669 = init[35]
    i_v3675 = init[36]
    i_v38 = init[37]
    i_v176 = init[38]
    i_v3893 = init[39]
    i_v3968 = init[40]
    i_v5515 = init[41]
    i_v5517 = init[42]
    i_v5522 = init[43]
    i_v5506 = init[44]
    i_v5867 = init[45]
    i_v5573 = init[46]
    i_v5593 = init[47]
    i_v5669 = init[48]
    i_v5760 = init[49]
    i_v7552 = init[50]
    i_v4024 = init[51]
    i_v6887 = init[52]
    i_v6889 = init[53]
    i_v5455 = init[54]
    i_v7260 = init[55]
    i_v7051 = init[56]
    i_v7053 = init[57]
    i_v5460 = init[58]
    i_v7269 = init[59]
    i_v7193 = init[60]
    i_v7195 = init[61]
    i_v5465 = init[62]
    i_v7277 = init[63]
    i_v7457 = init[64]
    i_v7452 = init[65]
    i_v7548 = init[66]
    i_v6071 = init[67]
    i_v6127 = init[68]
    i_v6172 = init[69]
    i_v7346 = init[70]
    i_v7347 = init[71]
    i_v7407 = init[72]
    i_v7379 = init[73]
    i_v7380 = init[74]
    i_v7416 = init[75]
    i_v7394 = init[76]
    i_v7395 = init[77]
    i_v7424 = init[78]
    v361 = TYPE * (signals.d - signals.b)
    v357 = TYPE * (signals.s - signals.b)
    v363 = v361 - v357
    v365 = v363 < 0.0
    v4343 = jnp.where(v365, -1, 1)
    v4377 = TYPE * v4343
    v353 = TYPE * (signals.g - signals.b)
    v387 = v353 - i_v382
    v392 = v387 - i_v390
    v414 = v392 + i_v157
    v421 = v414 + i_v286
    v423 = v421 * v421
    v446 = v423 + i_v445
    v447 = jnp.sqrt(jnp.maximum(v446, 1e-300))
    v449 = v421 + v447
    v450 = 0.5 * v449
    v574 = v450 - i_v157
    v452 = jnp.where(v365, v361, v357)
    v454 = i_v157 + v452
    v456 = v454 * v454
    v457 = v456 + i_v94
    v458 = jnp.sqrt(jnp.maximum(v457, 1e-300))
    v460 = v454 + v458
    v461 = 0.5 * v460
    v462 = jnp.sqrt(jnp.maximum(v461, 1e-300))
    v464 = jnp.where(v365, v357, v361)
    v466 = i_v157 + v464
    v468 = v466 * v466
    v469 = v468 + i_v94
    v470 = jnp.sqrt(jnp.maximum(v469, 1e-300))
    v472 = v466 + v470
    v473 = 0.5 * v472
    v474 = jnp.sqrt(jnp.maximum(v473, 1e-300))
    v605 = v462 + v474
    v606 = i_v560 * v605
    v607 = i_v269 - v606
    v571 = v450 + i_v570
    v572 = jnp.sqrt(jnp.maximum(v571, 1e-300))
    v576 = v572 - i_v575
    v577 = i_v269 * v576
    v578 = v574 - v577
    v580 = v578 + i_v157
    v602 = v580 + i_v81
    v603 = jnp.sqrt(jnp.maximum(v602, 1e-300))
    v608 = i_v521 * v603
    v609 = v607 + v608
    v611 = v609 * v609
    v612 = v611 + i_v81
    v613 = jnp.sqrt(jnp.maximum(v612, 1e-300))
    v615 = v609 + v613
    v616 = 0.5 * v615
    v618 = 0.25 * v616
    v619 = v618 * v616
    v620 = v450 + v619
    v621 = jnp.sqrt(jnp.maximum(v620, 1e-300))
    v624 = 0.5 * v616
    v625 = v621 - v624
    v626 = v616 * v625
    v627 = v574 - v626
    v1295 = v627 + i_v157
    v1731 = v1295 + i_v87
    v1732 = jnp.sqrt(jnp.maximum(v1731, 1e-300))
    v1740 = 2.0 * v1732
    v1741 = jnp.divide(i_v269, v1740)
    v1742 = 1.0 + v1741
    v1815 = i_v91 * v1742
    v805 = v464 - v452
    v806 = 0.5 * v805
    v629 = v627 - v452
    v651 = v629 * i_v83
    v654 = v651 > -0.35
    v665 = 1.0 + v651
    v657 = 1.3 + v651
    v659 = v651 + 1.6
    v660 = jnp.log(jnp.maximum(v659, 1e-300))
    v661 = v657 - v660
    v662 = jnp.divide(2.0, jnp.where((v661 == 0.0) | ~jnp.isfinite(v661), 1e-300, v661))
    v664 = 2.0 + v662
    v666 = jnp.log(jnp.maximum(v662, 1e-300))
    v667 = v665 + v666
    v668 = jnp.divide(v664, jnp.where((v667 == 0.0) | ~jnp.isfinite(v667), 1e-300, v667))
    v671 = jnp.log(jnp.maximum(v668, 1e-300))
    v672 = v665 + v671
    v673 = 2.0 + v668
    v674 = jnp.divide(v672, jnp.where((v673 == 0.0) | ~jnp.isfinite(v673), 1e-300, v673))
    v679 = v651 > -15.0
    v686 = 1.0 + v651
    v682 = -v651
    v683 = jnp.exp(jnp.clip(v682, -709.0, 709.0))
    v684 = 1.55 + v683
    v685 = 2.0 + v684
    v687 = jnp.log(jnp.maximum(v684, 1e-300))
    v688 = v686 + v687
    v689 = jnp.divide(v685, jnp.where((v688 == 0.0) | ~jnp.isfinite(v688), 1e-300, v688))
    v691 = jnp.log(jnp.maximum(v689, 1e-300))
    v692 = v686 + v691
    v693 = 2.0 + v689
    v694 = jnp.divide(v692, jnp.where((v693 == 0.0) | ~jnp.isfinite(v693), 1e-300, v693))
    v698 = v651 > -23.0
    v700 = -v651
    v701 = jnp.exp(jnp.clip(v700, -709.0, 709.0))
    v702 = 2.0 + v701
    v703 = jnp.divide(1.0, jnp.where((v702 == 0.0) | ~jnp.isfinite(v702), 1e-300, v702))
    v705 = jnp.exp(jnp.clip(v651, -709.0, 709.0))
    v707 = v705 + 1e-64
    v710 = jnp.where(v698, v703, v707)
    v709 = jnp.where(v679, v694, v710)
    v708 = jnp.where(v654, v674, v709)
    v711 = 1.0 + v708
    v712 = v708 * v711
    v715 = jnp.sqrt(jnp.maximum(v712, 1e-300))
    v780 = v715 * i_v778
    v781 = 0.25 + v780
    v782 = jnp.sqrt(jnp.maximum(v781, 1e-300))
    v784 = v782 - 0.5
    v785 = i_v186 * v784
    v833 = v785 * v785
    v826 = v785 * i_v83
    v827 = v715 - v826
    v828 = LAMBDA * v827
    v830 = v828 + 0.015625
    v831 = i_v94 * v830
    v834 = v833 + v831
    v835 = jnp.sqrt(jnp.maximum(v834, 1e-300))
    v837 = v806 - v785
    v839 = v837 * v837
    v840 = v839 + v831
    v841 = jnp.sqrt(jnp.maximum(v840, 1e-300))
    v843 = v835 - v841
    v1030 = v806 - v843
    v1070 = jnp.divide(v1030, jnp.where((i_v171 == 0.0) | ~jnp.isfinite(i_v171), 1e-300, i_v171))
    v1071 = 1.0 + v1070
    v1072 = jnp.log(jnp.maximum(v1071, 1e-300))
    v1073 = i_v28 * v1072
    v1093 = i_v180 - v1073
    v1094 = v806 + v843
    v1134 = v1094 * i_v160
    v1135 = v1093 + v1134
    v1139 = v1135 * v1135
    v1141 = v1139 + i_v1140
    v1142 = jnp.sqrt(jnp.maximum(v1141, 1e-300))
    v1144 = v1135 + v1142
    v1145 = 0.5 * v1144
    v1373 = v627 * v627
    v1423 = v1373 + i_v91
    v1424 = jnp.sqrt(jnp.maximum(v1423, 1e-300))
    v1426 = v627 + v1424
    v1427 = 0.5 * v1426
    v1430 = THETA * v1427
    v1431 = 1.0 + v1430
    v1480 = v1145 * v1431
    v1481 = jnp.divide(i_v258, jnp.where((v1480 == 0.0) | ~jnp.isfinite(v1480), 1e-300, v1480))
    v1296 = v1295 + 1e-06
    v1298 = jnp.sqrt(jnp.maximum(v1296, 1e-300))
    v1299 = 2.0 * v1298
    v1367 = i_v1366 * v1299
    v1330 = v1299 + i_v269
    v1331 = jnp.divide(i_v269, jnp.where((v1330 == 0.0) | ~jnp.isfinite(v1330), 1e-300, v1330))
    v1328 = jnp.divide(i_v269, v1299)
    v1333 = 1.0 + v1328
    v1334 = -v1333
    v1353 = v1334 * i_v78
    v1165 = v627 - v464
    v1175 = v1165 * i_v83
    v1177 = v1175 > -0.35
    v1185 = 1.0 + v1175
    v1179 = 1.3 + v1175
    v1180 = v1175 + 1.6
    v1181 = jnp.log(jnp.maximum(v1180, 1e-300))
    v1182 = v1179 - v1181
    v1183 = jnp.divide(2.0, jnp.where((v1182 == 0.0) | ~jnp.isfinite(v1182), 1e-300, v1182))
    v1184 = 2.0 + v1183
    v1186 = jnp.log(jnp.maximum(v1183, 1e-300))
    v1187 = v1185 + v1186
    v1188 = jnp.divide(v1184, jnp.where((v1187 == 0.0) | ~jnp.isfinite(v1187), 1e-300, v1187))
    v1190 = jnp.log(jnp.maximum(v1188, 1e-300))
    v1191 = v1185 + v1190
    v1192 = 2.0 + v1188
    v1193 = jnp.divide(v1191, jnp.where((v1192 == 0.0) | ~jnp.isfinite(v1192), 1e-300, v1192))
    v1196 = v1175 > -15.0
    v1202 = 1.0 + v1175
    v1198 = -v1175
    v1199 = jnp.exp(jnp.clip(v1198, -709.0, 709.0))
    v1200 = 1.55 + v1199
    v1201 = 2.0 + v1200
    v1203 = jnp.log(jnp.maximum(v1200, 1e-300))
    v1204 = v1202 + v1203
    v1205 = jnp.divide(v1201, jnp.where((v1204 == 0.0) | ~jnp.isfinite(v1204), 1e-300, v1204))
    v1207 = jnp.log(jnp.maximum(v1205, 1e-300))
    v1208 = v1202 + v1207
    v1209 = 2.0 + v1205
    v1210 = jnp.divide(v1208, jnp.where((v1209 == 0.0) | ~jnp.isfinite(v1209), 1e-300, v1209))
    v1213 = v1175 > -23.0
    v1215 = -v1175
    v1216 = jnp.exp(jnp.clip(v1215, -709.0, 709.0))
    v1217 = 2.0 + v1216
    v1218 = jnp.divide(1.0, jnp.where((v1217 == 0.0) | ~jnp.isfinite(v1217), 1e-300, v1217))
    v1220 = jnp.exp(jnp.clip(v1175, -709.0, 709.0))
    v1221 = v1220 + 1e-64
    v1224 = jnp.where(v1213, v1218, v1221)
    v1223 = jnp.where(v1196, v1210, v1224)
    v1222 = jnp.where(v1177, v1193, v1223)
    v1225 = 1.0 + v1222
    v1226 = v1222 * v1225
    v1249 = 0.25 + v1226
    v1253 = jnp.sqrt(jnp.maximum(v1249, 1e-300))
    v1247 = 0.25 + v712
    v1251 = jnp.sqrt(jnp.maximum(v1247, 1e-300))
    v1356 = v1253 * v1251
    v1357 = v1249 + v1356
    v1358 = v1357 + v1247
    v1359 = 1.33333332 * v1358
    v1255 = v1251 + v1253
    v1361 = jnp.divide(v1359, jnp.where((v1255 == 0.0) | ~jnp.isfinite(v1255), 1e-300, v1255))
    v1362 = v1361 - 1.0
    v1363 = v1353 * v1362
    v1368 = v1331 * v1363
    v1369 = v1367 - v1368
    v1540 = i_v52 * v1363
    v1541 = v1369 + v1540
    v1542 = v1541 > 0.0
    v1607 = i_v42 * v1541
    v1608 = 1.0 + v1607
    v1616 = i_v42 * v1541
    v1617 = 1.0 - v1616
    v1662 = jnp.where(v1542, v1608, v1617)
    v1663 = v1145 * v1662
    v1664 = jnp.divide(i_v1657, jnp.where((v1663 == 0.0) | ~jnp.isfinite(v1663), 1e-300, v1663))
    v1816 = jnp.where(i_v1371, v1481, v1664)
    v1817 = v1815 * v1816
    v906 = v627 - v806
    v907 = v906 - v452
    v846 = jnp.log(jnp.maximum(v712, 1e-300))
    v847 = 0.75 * v846
    v848 = v715 - v847
    v849 = v848 * i_v778
    v850 = 0.25 + v849
    v851 = jnp.sqrt(jnp.maximum(v850, 1e-300))
    v853 = v851 - 0.5
    v854 = i_v186 * v853
    v885 = v854 + i_v193
    v889 = v885 * v885
    v890 = v889 + v831
    v891 = jnp.sqrt(jnp.maximum(v890, 1e-300))
    v908 = v907 - v891
    v887 = v806 - v885
    v893 = v887 * v887
    v894 = v893 + v831
    v895 = jnp.sqrt(jnp.maximum(v894, 1e-300))
    v909 = v908 + v895
    v910 = v909 * i_v83
    v912 = v910 > -0.35
    v920 = 1.0 + v910
    v914 = 1.3 + v910
    v915 = v910 + 1.6
    v916 = jnp.log(jnp.maximum(v915, 1e-300))
    v917 = v914 - v916
    v918 = jnp.divide(2.0, jnp.where((v917 == 0.0) | ~jnp.isfinite(v917), 1e-300, v917))
    v919 = 2.0 + v918
    v921 = jnp.log(jnp.maximum(v918, 1e-300))
    v922 = v920 + v921
    v923 = jnp.divide(v919, jnp.where((v922 == 0.0) | ~jnp.isfinite(v922), 1e-300, v922))
    v925 = jnp.log(jnp.maximum(v923, 1e-300))
    v926 = v920 + v925
    v927 = 2.0 + v923
    v928 = jnp.divide(v926, jnp.where((v927 == 0.0) | ~jnp.isfinite(v927), 1e-300, v927))
    v931 = v910 > -15.0
    v937 = 1.0 + v910
    v933 = -v910
    v934 = jnp.exp(jnp.clip(v933, -709.0, 709.0))
    v935 = 1.55 + v934
    v936 = 2.0 + v935
    v938 = jnp.log(jnp.maximum(v935, 1e-300))
    v939 = v937 + v938
    v940 = jnp.divide(v936, jnp.where((v939 == 0.0) | ~jnp.isfinite(v939), 1e-300, v939))
    v942 = jnp.log(jnp.maximum(v940, 1e-300))
    v943 = v937 + v942
    v944 = 2.0 + v940
    v945 = jnp.divide(v943, jnp.where((v944 == 0.0) | ~jnp.isfinite(v944), 1e-300, v944))
    v948 = v910 > -23.0
    v950 = -v910
    v951 = jnp.exp(jnp.clip(v950, -709.0, 709.0))
    v952 = 2.0 + v951
    v953 = jnp.divide(1.0, jnp.where((v952 == 0.0) | ~jnp.isfinite(v952), 1e-300, v952))
    v955 = jnp.exp(jnp.clip(v910, -709.0, 709.0))
    v956 = v955 + 1e-64
    v959 = jnp.where(v948, v953, v956)
    v958 = jnp.where(v931, v945, v959)
    v957 = jnp.where(v912, v928, v958)
    v960 = 1.0 + v957
    v961 = v957 * v960
    v1808 = v712 - v961
    v1819 = v1817 * v1808
    v3474 = -v1817
    v3416 = 4.0 * v1742
    v3423 = v3416 * v1732
    v3435 = v3423 * v1731
    v3436 = jnp.divide(i_v3409, jnp.where((v3435 == 0.0) | ~jnp.isfinite(v3435), 1e-300, v3435))
    v2363 = jnp.divide(v1295, v621)
    v2365 = -v2363
    v1936 = v613 + v613
    v1937 = jnp.divide(v616, jnp.where((v1936 == 0.0) | ~jnp.isfinite(v1936), 1e-300, v1936))
    v2048 = i_v2047 * v1937
    v2162 = v2048 * v462
    v2199 = jnp.divide(v2162, v458)
    v2369 = v2365 * v2199
    v3443 = v3436 * v2369
    v2915 = jnp.divide(1.0, v1142)
    v2866 = i_v171 + v806
    v2885 = v2866 - v843
    v2886 = jnp.divide(i_v28, jnp.where((v2885 == 0.0) | ~jnp.isfinite(v2885), 1e-300, v2885))
    v2462 = 4.0 * v782
    v2466 = v2462 * v715
    v2467 = jnp.divide(i_v78, v2466)
    v2424 = v708 * i_v83
    v2427 = v2369 - 1.0
    v2428 = v2424 * v2427
    v2470 = v2467 * v2428
    v2604 = v785 * v2470
    v2479 = v715 + v715
    v2480 = jnp.divide(i_v78, jnp.where((v2479 == 0.0) | ~jnp.isfinite(v2479), 1e-300, v2479))
    v2485 = v2428 * v2480
    v2486 = v2485 - v2470
    v2487 = i_v2478 * v2486
    v2605 = v2604 + v2487
    v2520 = jnp.divide(1.0, v835)
    v2606 = v2605 * v2520
    v2608 = -0.5 - v2470
    v2609 = v837 * v2608
    v2610 = v2609 + v2487
    v2548 = jnp.divide(1.0, v841)
    v2611 = v2610 * v2548
    v2612 = v2606 - v2611
    v2891 = -0.5 - v2612
    v2892 = v2886 * v2891
    v2940 = -v2892
    v2942 = -0.5 + v2612
    v2943 = v2942 * i_v160
    v2944 = v2940 + v2943
    v2945 = v2915 * v2944
    v3298 = -v2945
    v3086 = jnp.where(i_v1371, v1427, 0.0)
    v3149 = THETA * v3086
    (v3153, v3219) = jax.tree_util.tree_map(lambda _t, _f: jnp.where(i_v1371, _t, _f), (v1431, v1424), (0.0, 0.0))
    v3282 = v3153 * v3219
    v3283 = jnp.divide(v3149, jnp.where((v3282 == 0.0) | ~jnp.isfinite(v3282), 1e-300, v3282))
    v3288 = v3283 * v2369
    v3299 = v3298 - v3288
    v3391 = -v2945
    v3314 = jnp.where(i_v1371, 0.0, v1662)
    v3374 = jnp.divide(i_v42, jnp.where((v3314 == 0.0) | ~jnp.isfinite(v3314), 1e-300, v3314))
    v3067 = -v1331
    v3054 = 2.0 * v1333
    v3055 = v3054 * v1296
    v3056 = jnp.divide(v1363, jnp.where((v3055 == 0.0) | ~jnp.isfinite(v3055), 1e-300, v3055))
    v3057 = v1333 - v3056
    v3073 = v3057 * v2369
    v3016 = -v1328
    v3020 = v3016 * v1363
    v3021 = 2.0 + v1328
    v3022 = v3021 + v1328
    v3032 = v3022 * v1296
    v3033 = jnp.divide(v3020, jnp.where((v3032 == 0.0) | ~jnp.isfinite(v3032), 1e-300, v3032))
    v3040 = v3033 * v2369
    v2981 = v1353 * 0.66666666
    v1257 = v1255 * v1255
    v2991 = jnp.divide(v2981, jnp.where((v1257 == 0.0) | ~jnp.isfinite(v1257), 1e-300, v1257))
    v3010 = 2.0 * v1253
    v3011 = v1251 + v3010
    v3012 = v2991 * v3011
    v3041 = v3012 * v2428
    v3042 = v3040 + v3041
    v3013 = 2.0 * v1251
    v3014 = v1253 + v3013
    v3015 = v2991 * v3014
    v2961 = v1222 * i_v83
    v2965 = v2961 * v2369
    v3043 = v3015 * v2965
    v3044 = v3042 + v3043
    v3074 = v3073 + v3044
    v3075 = v3067 * v3074
    v3394 = i_v52 * v3044
    v3395 = v3075 + v3394
    v3396 = v3374 * v3395
    v3397 = v3391 + v3396
    v3475 = jnp.where(i_v1371, v3299, v3397)
    v3476 = v3443 + v3475
    v3477 = v3476 * v1808
    v3481 = v3477 + v2428
    v2683 = v957 * i_v83
    v2806 = v2369 - 0.5
    v2624 = v715 - 1.5
    v2625 = i_v78 * v2624
    v2653 = 4.0 * v851
    v2657 = v2653 * v712
    v2658 = jnp.divide(v2625, jnp.where((v2657 == 0.0) | ~jnp.isfinite(v2657), 1e-300, v2657))
    v2661 = v2658 * v2428
    v2807 = v885 * v2661
    v2808 = v2807 + v2487
    v2711 = jnp.divide(1.0, v891)
    v2809 = v2808 * v2711
    v2810 = v2806 - v2809
    v2812 = -0.5 - v2661
    v2813 = v887 * v2812
    v2814 = v2813 + v2487
    v2739 = jnp.divide(1.0, v895)
    v2815 = v2814 * v2739
    v2816 = v2810 + v2815
    v2817 = v2683 * v2816
    v3485 = v3481 - v2817
    v3486 = v3474 * v3485
    v3670 = v3486 * i_v3669
    v3671 = 1.0 + v3670
    v2085 = v2048 * v474
    v2122 = jnp.divide(v2085, v470)
    v2366 = v2365 * v2122
    v3439 = v3436 * v2366
    v2425 = v2424 * v2366
    v2468 = v2467 * v2425
    v2595 = v785 * v2468
    v2481 = v2425 * v2480
    v2482 = v2481 - v2468
    v2483 = i_v2478 * v2482
    v2596 = v2595 + v2483
    v2597 = v2596 * v2520
    v2598 = 0.5 - v2468
    v2599 = v837 * v2598
    v2600 = v2599 + v2483
    v2601 = v2600 * v2548
    v2602 = v2597 - v2601
    v2887 = 0.5 - v2602
    v2888 = v2886 * v2887
    v2916 = -v2888
    v2917 = 0.5 + v2602
    v2936 = v2917 * i_v160
    v2937 = v2916 + v2936
    v2938 = v2915 * v2937
    v3294 = -v2938
    v3285 = v3283 * v2366
    v3295 = v3294 - v3285
    v3376 = -v2938
    v3068 = v3057 * v2366
    v3034 = v3033 * v2366
    v3035 = v3012 * v2425
    v3036 = v3034 + v3035
    v2962 = v2366 - 1.0
    v2963 = v2961 * v2962
    v3037 = v3015 * v2963
    v3038 = v3036 + v3037
    v3069 = v3068 + v3038
    v3070 = v3067 * v3069
    v3386 = i_v52 * v3038
    v3387 = v3070 + v3386
    v3388 = v3374 * v3387
    v3389 = v3376 + v3388
    v3455 = jnp.where(i_v1371, v3295, v3389)
    v3456 = v3439 + v3455
    v3463 = v3456 * v1808
    v3467 = v3463 + v2425
    v2740 = v2366 - 0.5
    v2659 = v2658 * v2425
    v2768 = v885 * v2659
    v2769 = v2768 + v2483
    v2770 = v2769 * v2711
    v2771 = v2740 - v2770
    v2799 = 0.5 - v2659
    v2800 = v887 * v2799
    v2801 = v2800 + v2483
    v2802 = v2801 * v2739
    v2803 = v2771 + v2802
    v2804 = v2683 * v2803
    v3471 = v3467 - v2804
    v3472 = v1817 * v3471
    v3676 = v3472 * i_v3675
    v3677 = v3671 + v3676
    v3678 = jnp.divide(1.0, jnp.where((v3677 == 0.0) | ~jnp.isfinite(v3677), 1e-300, v3677))
    v3688 = v1819 * v3678
    v4381 = v4377 * v3688
    v4821 = v4381 + 0.0
    v4824 = v4821 + 0.0
    v9660 = jnp.where(i_v4746, v4824, v4381)
    v4463 = v4343 == 1
    v3825 = i_v38 * v785
    v3826 = v805 - v3825
    v3828 = v3826 > 0.0
    v3894 = jnp.where(v3828, i_v3893, False)
    v3988 = i_v176 * v3826
    v3899 = jnp.divide(1.0, jnp.where((v3826 == 0.0) | ~jnp.isfinite(v3826), 1e-300, v3826))
    v3969 = i_v3968 * v3899
    v3973 = v3969 < -35.0
    v3975 = jnp.where(v3973, -35.0, v3969)
    v3977 = jnp.exp(jnp.clip(v3975, -709.0, 709.0))
    v3989 = v3988 * v3977
    v3998 = v3989 * v3688
    v4470 = jnp.where(v3894, v3998, 0.0)
    v4471 = TYPE * v4470
    v4645 = jnp.where(v4463, v4471, 0.0)
    v4647 = v4645 + 0.0
    v11072 = v9660 + v4647
    v5527 = -v361
    v5528 = v5527 * i_v114
    v5529 = jnp.divide(v5528, jnp.where((i_v5506 == 0.0) | ~jnp.isfinite(i_v5506), 1e-300, i_v5506))
    v5533 = v5529 < -40.0
    v5816 = jnp.where(v5533, -40.0, v5529)
    v5825 = v5816 > 69.07755278982137
    v5826 = v5816 - 69.07755278982137
    v5827 = 1e30 * v5826
    v5828 = 1e30 + v5827
    v5829 = jnp.exp(jnp.clip(v5816, -709.0, 709.0))
    v5830 = jnp.where(v5825, v5828, v5829)
    v5831 = 1.0 - v5830
    v5832 = i_v5522 * v5831
    v5540 = v5527 + bv
    v5544 = v5540 * i_v114
    v5548 = jnp.divide(v5544, jnp.where((i_v5506 == 0.0) | ~jnp.isfinite(i_v5506), 1e-300, i_v5506))
    v5552 = v5548 > 70.0
    v5556 = -v5548
    v5557 = jnp.exp(jnp.clip(v5556, -709.0, 709.0))
    v5558 = xjbv * v5557
    v5559 = 1.0 + v5558
    v5847 = jnp.where(v5552, 1.0, v5559)
    v5855 = v5832 * v5847
    v5868 = v361 * i_v5867
    v5869 = v5855 + v5868
    v5580 = v361 * i_v114
    v5594 = jnp.divide(v5580, jnp.where((i_v5593 == 0.0) | ~jnp.isfinite(i_v5593), 1e-300, i_v5593))
    v5596 = v5594 * vtsswg
    v5597 = vtsswg + v361
    v5599 = v5597 > 0.001
    v5600 = jnp.where(v5599, v5597, 0.001)
    v5601 = jnp.divide(v5596, jnp.where((v5600 == 0.0) | ~jnp.isfinite(v5600), 1e-300, v5600))
    v5604 = v5601 > 69.07755278982137
    v5605 = v5601 - 69.07755278982137
    v5606 = 1e30 * v5605
    v5607 = 1e30 + v5606
    v5608 = jnp.exp(jnp.clip(v5601, -709.0, 709.0))
    v5609 = jnp.where(v5604, v5607, v5608)
    v5610 = v5609 - 1.0
    v5611 = i_v5573 * v5610
    v5670 = jnp.divide(v5580, jnp.where((i_v5669 == 0.0) | ~jnp.isfinite(i_v5669), 1e-300, i_v5669))
    v5672 = v5670 * vtssw
    v5673 = vtssw + v361
    v5674 = v5673 > 0.001
    v5675 = jnp.where(v5674, v5673, 0.001)
    v5676 = jnp.divide(v5672, jnp.where((v5675 == 0.0) | ~jnp.isfinite(v5675), 1e-300, v5675))
    v5677 = v5676 > 69.07755278982137
    v5678 = v5676 - 69.07755278982137
    v5679 = 1e30 * v5678
    v5680 = 1e30 + v5679
    v5681 = jnp.exp(jnp.clip(v5676, -709.0, 709.0))
    v5682 = jnp.where(v5677, v5680, v5681)
    v5683 = v5682 - 1.0
    v5684 = i_v5517 * v5683
    v5685 = v5611 - v5684
    v5761 = jnp.divide(v5580, jnp.where((i_v5760 == 0.0) | ~jnp.isfinite(i_v5760), 1e-300, i_v5760))
    v5763 = v5761 * vts
    v5764 = vts + v361
    v5765 = v5764 > 0.001
    v5766 = jnp.where(v5765, v5764, 0.001)
    v5767 = jnp.divide(v5763, jnp.where((v5766 == 0.0) | ~jnp.isfinite(v5766), 1e-300, v5766))
    v5768 = v5767 > 69.07755278982137
    v5769 = v5767 - 69.07755278982137
    v5770 = 1e30 * v5769
    v5771 = 1e30 + v5770
    v5772 = jnp.exp(jnp.clip(v5767, -709.0, 709.0))
    v5773 = jnp.where(v5768, v5771, v5772)
    v5774 = v5773 - 1.0
    v5775 = i_v5515 * v5774
    v5776 = v5685 - v5775
    v5873 = v5869 + v5776
    v5875 = v5873 * TYPE
    v5877 = v5875 * M
    v7326 = v5877 + 0.0
    v11084 = v11072 + v7326
    v11098 = v11084 - 0.0
    v13677 = _mfactor * v11098
    v4132 = 0.5 * v627
    v4133 = i_v157 + v4132
    v4134 = jnp.sqrt(jnp.maximum(v4133, 1e-300))
    v4135 = v4134 + v4134
    v4152 = jnp.divide(v616, jnp.where((v4135 == 0.0) | ~jnp.isfinite(v4135), 1e-300, v4135))
    v4153 = 1.0 + v4152
    v4169 = v4153 * i_v78
    v4170 = v4169 * i_v4024
    v4172 = -v4170
    v4106 = v1253 * v1249
    v4174 = 3.0 * v4106
    v4176 = 6.0 * v1249
    v4177 = v4176 * v1251
    v4178 = v4174 + v4177
    v4179 = 4.0 * v1253
    v4180 = v4179 * v1247
    v4181 = v4178 + v4180
    v4065 = v1251 * v1247
    v4182 = 2.0 * v4065
    v4183 = v4181 + v4182
    v4184 = 0.266666666 * v4183
    v4200 = jnp.divide(v4184, jnp.where((v1257 == 0.0) | ~jnp.isfinite(v1257), 1e-300, v1257))
    v4201 = v4200 - 0.5
    v4202 = v4172 * v4201
    v4205 = 3.0 * v4065
    v4206 = 6.0 * v1247
    v4207 = v4206 * v1253
    v4208 = v4205 + v4207
    v4209 = 4.0 * v1251
    v4210 = v4209 * v1249
    v4211 = v4208 + v4210
    v4212 = 2.0 * v4106
    v4213 = v4211 + v4212
    v4214 = 0.266666666 * v4213
    v4215 = jnp.divide(v4214, jnp.where((v1257 == 0.0) | ~jnp.isfinite(v1257), 1e-300, v1257))
    v4216 = v4215 - 0.5
    v4217 = v4172 * v4216
    v4642 = jnp.where(v4463, v4202, v4217)
    v10530 = TYPE * v4642
    v6794 = v361 > 0.0
    v6946 = jnp.divide(v361, jnp.where((i_v5455 == 0.0) | ~jnp.isfinite(i_v5455), 1e-300, i_v5455))
    v6947 = 1.0 + v6946
    v6948 = jnp.log(jnp.maximum(v6947, 1e-300))
    v6949 = i_v6889 * v6948
    v6950 = jnp.exp(jnp.clip(v6949, -709.0, 709.0))
    v6951 = i_v6887 * v6950
    v7262 = mj * v361
    v7264 = jnp.divide(v7262, jnp.where((i_v5455 == 0.0) | ~jnp.isfinite(i_v5455), 1e-300, i_v5455))
    v7265 = 1.0 - v7264
    v7266 = i_v7260 * v7265
    v7283 = jnp.where(v6794, v6951, v7266)
    v7109 = jnp.divide(v361, jnp.where((i_v5460 == 0.0) | ~jnp.isfinite(i_v5460), 1e-300, i_v5460))
    v7110 = 1.0 + v7109
    v7111 = jnp.log(jnp.maximum(v7110, 1e-300))
    v7112 = i_v7053 * v7111
    v7113 = jnp.exp(jnp.clip(v7112, -709.0, 709.0))
    v7114 = i_v7051 * v7113
    v7270 = mjsw * v361
    v7272 = jnp.divide(v7270, jnp.where((i_v5460 == 0.0) | ~jnp.isfinite(i_v5460), 1e-300, i_v5460))
    v7273 = 1.0 - v7272
    v7274 = i_v7269 * v7273
    v7284 = jnp.where(v6794, v7114, v7274)
    v7285 = v7283 + v7284
    v7251 = jnp.divide(v361, jnp.where((i_v5465 == 0.0) | ~jnp.isfinite(i_v5465), 1e-300, i_v5465))
    v7252 = 1.0 + v7251
    v7253 = jnp.log(jnp.maximum(v7252, 1e-300))
    v7254 = i_v7195 * v7253
    v7255 = jnp.exp(jnp.clip(v7254, -709.0, 709.0))
    v7256 = i_v7193 * v7255
    v7278 = mjswg * v361
    v7280 = jnp.divide(v7278, jnp.where((i_v5465 == 0.0) | ~jnp.isfinite(i_v5465), 1e-300, i_v5465))
    v7281 = 1.0 - v7280
    v7282 = i_v7277 * v7281
    v7286 = jnp.where(v6794, v7256, v7282)
    v7287 = v7285 + v7286
    v7289 = v7287 * v361
    v10533 = v7289 * TYPE
    v10534 = v10533 * M
    v11085 = v10530 + v10534
    v7459 = i_v7457 * (signals.g - signals.d)
    v10538 = i_v7552 * v7459
    v11099 = v11085 - v10538
    v13679 = _mfactor * v11099
    v4219 = v4217 + v4202
    v4322 = -v4219
    v4222 = -0.5 * v616
    v4247 = v4222 * v1299
    v4263 = v4247 + v450
    v4315 = v4263 - v421
    v4316 = i_v4024 * v4315
    v4317 = v4219 * v616
    v4318 = v616 + v4135
    v4319 = jnp.divide(v4317, jnp.where((v4318 == 0.0) | ~jnp.isfinite(v4318), 1e-300, v4318))
    v4320 = v4316 - v4319
    v4323 = v4322 - v4320
    v10532 = TYPE * v4323
    v7454 = i_v7452 * (signals.g - signals.s)
    v10537 = i_v7552 * v7454
    v11093 = v10532 + v10537
    v11097 = v11093 + v10538
    v7549 = i_v7548 * (signals.g - signals.b)
    v10539 = i_v7552 * v7549
    v11101 = v11097 + v10539
    v13683 = _mfactor * v11101
    v11071 = -v9660
    v4558 = TYPE * v4470
    v4652 = jnp.where(v4463, 0.0, v4558)
    v4654 = v4652 + 0.0
    v11076 = v11071 + v4654
    v6177 = -v357
    v6187 = v6177 * i_v114
    v6212 = jnp.divide(v6187, jnp.where((i_v5506 == 0.0) | ~jnp.isfinite(i_v5506), 1e-300, i_v5506))
    v6215 = v6212 < -40.0
    v6542 = jnp.where(v6215, -40.0, v6212)
    v6551 = v6542 > 69.07755278982137
    v6552 = v6542 - 69.07755278982137
    v6553 = 1e30 * v6552
    v6554 = 1e30 + v6553
    v6555 = jnp.exp(jnp.clip(v6542, -709.0, 709.0))
    v6556 = jnp.where(v6551, v6554, v6555)
    v6557 = 1.0 - v6556
    v6558 = i_v6172 * v6557
    v6221 = v6177 + bv
    v6225 = v6221 * i_v114
    v6229 = jnp.divide(v6225, jnp.where((i_v5506 == 0.0) | ~jnp.isfinite(i_v5506), 1e-300, i_v5506))
    v6231 = v6229 > 70.0
    v6234 = -v6229
    v6235 = jnp.exp(jnp.clip(v6234, -709.0, 709.0))
    v6236 = xjbv * v6235
    v6237 = 1.0 + v6236
    v6573 = jnp.where(v6231, 1.0, v6237)
    v6581 = v6558 * v6573
    v6592 = v357 * i_v5867
    v6593 = v6581 + v6592
    v6258 = v357 * i_v114
    v6302 = jnp.divide(v6258, jnp.where((i_v5593 == 0.0) | ~jnp.isfinite(i_v5593), 1e-300, i_v5593))
    v6303 = v6302 * vtsswg
    v6304 = vtsswg + v357
    v6305 = v6304 > 0.001
    v6306 = jnp.where(v6305, v6304, 0.001)
    v6307 = jnp.divide(v6303, jnp.where((v6306 == 0.0) | ~jnp.isfinite(v6306), 1e-300, v6306))
    v6308 = v6307 > 69.07755278982137
    v6309 = v6307 - 69.07755278982137
    v6310 = 1e30 * v6309
    v6311 = 1e30 + v6310
    v6312 = jnp.exp(jnp.clip(v6307, -709.0, 709.0))
    v6313 = jnp.where(v6308, v6311, v6312)
    v6314 = v6313 - 1.0
    v6315 = i_v5573 * v6314
    v6389 = jnp.divide(v6258, jnp.where((i_v5669 == 0.0) | ~jnp.isfinite(i_v5669), 1e-300, i_v5669))
    v6390 = v6389 * vtssw
    v6391 = vtssw + v357
    v6392 = v6391 > 0.001
    v6393 = jnp.where(v6392, v6391, 0.001)
    v6394 = jnp.divide(v6390, jnp.where((v6393 == 0.0) | ~jnp.isfinite(v6393), 1e-300, v6393))
    v6395 = v6394 > 69.07755278982137
    v6396 = v6394 - 69.07755278982137
    v6397 = 1e30 * v6396
    v6398 = 1e30 + v6397
    v6399 = jnp.exp(jnp.clip(v6394, -709.0, 709.0))
    v6400 = jnp.where(v6395, v6398, v6399)
    v6401 = v6400 - 1.0
    v6402 = i_v6127 * v6401
    v6403 = v6315 - v6402
    v6488 = jnp.divide(v6258, jnp.where((i_v5760 == 0.0) | ~jnp.isfinite(i_v5760), 1e-300, i_v5760))
    v6489 = v6488 * vts
    v6490 = vts + v357
    v6491 = v6490 > 0.001
    v6492 = jnp.where(v6491, v6490, 0.001)
    v6493 = jnp.divide(v6489, jnp.where((v6492 == 0.0) | ~jnp.isfinite(v6492), 1e-300, v6492))
    v6494 = v6493 > 69.07755278982137
    v6495 = v6493 - 69.07755278982137
    v6496 = 1e30 * v6495
    v6497 = 1e30 + v6496
    v6498 = jnp.exp(jnp.clip(v6493, -709.0, 709.0))
    v6499 = jnp.where(v6494, v6497, v6498)
    v6500 = v6499 - 1.0
    v6501 = i_v6071 * v6500
    v6502 = v6403 - v6501
    v6597 = v6593 + v6502
    v6599 = v6597 * TYPE
    v6601 = v6599 * M
    v7449 = v6601 + 0.0
    v11088 = v11076 + v7449
    v11094 = v11088 - 0.0
    v13685 = _mfactor * v11094
    v4649 = jnp.where(v4463, v4217, v4202)
    v10531 = TYPE * v4649
    v7330 = v357 > 0.0
    v7351 = jnp.divide(v357, jnp.where((i_v5455 == 0.0) | ~jnp.isfinite(i_v5455), 1e-300, i_v5455))
    v7352 = 1.0 + v7351
    v7353 = jnp.log(jnp.maximum(v7352, 1e-300))
    v7354 = i_v7347 * v7353
    v7355 = jnp.exp(jnp.clip(v7354, -709.0, 709.0))
    v7356 = i_v7346 * v7355
    v7409 = mj * v357
    v7411 = jnp.divide(v7409, jnp.where((i_v5455 == 0.0) | ~jnp.isfinite(i_v5455), 1e-300, i_v5455))
    v7412 = 1.0 - v7411
    v7413 = i_v7407 * v7412
    v7430 = jnp.where(v7330, v7356, v7413)
    v7383 = jnp.divide(v357, jnp.where((i_v5460 == 0.0) | ~jnp.isfinite(i_v5460), 1e-300, i_v5460))
    v7384 = 1.0 + v7383
    v7385 = jnp.log(jnp.maximum(v7384, 1e-300))
    v7386 = i_v7380 * v7385
    v7387 = jnp.exp(jnp.clip(v7386, -709.0, 709.0))
    v7388 = i_v7379 * v7387
    v7417 = mjsw * v357
    v7419 = jnp.divide(v7417, jnp.where((i_v5460 == 0.0) | ~jnp.isfinite(i_v5460), 1e-300, i_v5460))
    v7420 = 1.0 - v7419
    v7421 = i_v7416 * v7420
    v7431 = jnp.where(v7330, v7388, v7421)
    v7432 = v7430 + v7431
    v7398 = jnp.divide(v357, jnp.where((i_v5465 == 0.0) | ~jnp.isfinite(i_v5465), 1e-300, i_v5465))
    v7399 = 1.0 + v7398
    v7400 = jnp.log(jnp.maximum(v7399, 1e-300))
    v7401 = i_v7395 * v7400
    v7402 = jnp.exp(jnp.clip(v7401, -709.0, 709.0))
    v7403 = i_v7394 * v7402
    v7425 = mjswg * v357
    v7427 = jnp.divide(v7425, jnp.where((i_v5465 == 0.0) | ~jnp.isfinite(i_v5465), 1e-300, i_v5465))
    v7428 = 1.0 - v7427
    v7429 = i_v7424 * v7428
    v7433 = jnp.where(v7330, v7403, v7429)
    v7434 = v7432 + v7433
    v7436 = v7434 * v357
    v10535 = v7436 * TYPE
    v10536 = v10535 * M
    v11089 = v10531 + v10536
    v11095 = v11089 - v10537
    v13687 = _mfactor * v11095
    v11074 = -v4647
    v11078 = v11074 - v4654
    v11082 = v11078 - 0.0
    v11086 = v11082 - v7326
    v11090 = v11086 - v7449
    v11102 = v11090 - 0.0
    v13689 = _mfactor * v11102
    v11075 = -v10530
    v11079 = v11075 - v10531
    v11083 = v11079 - v10532
    v11087 = v11083 - v10534
    v11091 = v11087 - v10536
    v11103 = v11091 - v10539
    v13691 = _mfactor * v11103
    return {"d": v13677, "s": v13685, "b": v13689}, {"d": v13679, "g": v13683, "s": v13687, "b": v13691}


@EKV26.setup
def _EKV26_register_setup(*_a, **_kw):
    return _EKV26_setup(*_a, **_kw)
