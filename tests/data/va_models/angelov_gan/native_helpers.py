"""Source-exact case formulas replacing multiway-phi lowering artifacts.

Derived from angelov_gan.va, Copyright (c) 2002-2015 Tiburon design Automation, Inc.
MIT license and full notice are preserved in the generated module and source.
This file is embedded by regenerate.py; it is not a runtime dependency.


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

# ruff: noqa: ANN001, ANN202

import jax.numpy as jnp


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
