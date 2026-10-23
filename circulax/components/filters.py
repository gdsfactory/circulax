"""Buffered continuous-time voltage filters for driver and behavioral models.

These unilateral blocks have infinite input impedance and ideal zero output
impedance. They are active behavioral components, not passive RF two-ports.
Frequencies are in Hz; ``f3db`` and ``Q`` must be positive.
"""

import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component


@component(ports=("inp", "out"), states=("i_out",), holomorphic=True)
def LowPass1(signals: Signals, f3db: float = 10e9) -> PhysicsReturn:
    """Unity-DC-gain first-order low-pass; ``f3db`` is the -3 dB frequency.

    ``H(s)=1/(1+s*tau)``, ``tau=1/(2*pi*f3db)``. Input current is zero;
    the output voltage is independent of the connected load.
    """
    tau = 1.0 / (2.0 * jnp.pi * f3db)
    return {
        "inp": 0.0 * signals.inp,
        "out": signals.i_out,
        "i_out": signals.out - signals.inp,
    }, {"i_out": tau * signals.out}


@component(ports=("inp", "out"), states=("i_out", "x"), holomorphic=True)
def Biquad(signals: Signals, f3db: float = 10e9, Q: float = 0.5773502692) -> PhysicsReturn:
    """Unity-DC-gain second-order low-pass with -3 dB frequency and quality factor.

    ``H(s)=w0**2/(s**2+w0*s/Q+w0**2)``. ``w0`` is computed so the magnitude
    is 1/sqrt(2) at ``f3db`` for every positive Q. For Q>1/sqrt(2), the
    response has peaking. This is an ideal buffered voltage block.
    """
    b = 1.0 / Q**2 - 2.0
    # Positive root of u**2 + b*u - 1 = 0, without cancellation at small Q.
    u = 2.0 / (b + jnp.sqrt(b**2 + 4.0))
    w0 = 2.0 * jnp.pi * f3db / jnp.sqrt(u)
    return {
        "inp": 0.0 * signals.inp,
        "out": signals.i_out,
        "i_out": w0 * (signals.out + signals.x / Q - signals.inp),
        "x": -w0 * signals.x,
    }, {"i_out": signals.x, "x": signals.out}


@component(ports=("inp", "out"), states=("i_out", "x", "z"), holomorphic=True)
def Bessel3(signals: Signals, f3db: float = 35e9) -> PhysicsReturn:
    """Unity-DC-gain third-order Bessel low-pass, normalized at -3 dB.

    The delay-normalized prototype is ``15/(p**3+6*p**2+15*p+15)``.
    Its -3 dB angular frequency is 1.7556723686812106; scaling by it makes
    ``f3db`` the voltage magnitude cutoff. The DC group delay is
    ``1.7556723686812106/(2*pi*f3db)``. This buffered block includes that
    filter delay; add physical flight time separately when needed.
    """
    r3db = 1.7556723686812106
    w = 2.0 * jnp.pi * f3db / r3db
    return {
        "inp": 0.0 * signals.inp,
        "out": signals.i_out,
        "i_out": w * (15 * signals.out + 15 * signals.x + 6 * signals.z - 15 * signals.inp),
        "x": -w * signals.x,
        "z": -w * signals.z,
    }, {"i_out": signals.z, "x": signals.out, "z": signals.x}
