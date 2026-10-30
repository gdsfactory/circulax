# Copyright (c) 2026 Circulax contributors.
# ruff: noqa: ANN001, ANN202, PLR0917 -- promoted notebook APIs retain their signatures.
"""Legacy example ring models and independent reciprocal single-resonance CMT.

Legacy RingModulator/EO/Setup retain the notebook field and detuning conventions.
Use AllPassRing/AddDropRing for power-normalized passive scattering interfaces.
"""

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from circulax.circuit import Circuit

from circulax.components.base_component import PhysicsReturn, Signals, component


@component(ports=("p1", "p2"), states=("a", "i_out"))
def RingModulator(
    signals: Signals,
    ng: float = 3.8,
    L: float = 3.14159265e-5,
    gamma: float = 0.976,
    alpha0: float = 0.969,
    alpha1: float = 0.0,
    voltage: float = 0.0,
    f_operating: float = 2.2904e14,
    f_resonance: float = 2.2901e14,
    v_to_wr: float = 0.0,
) -> PhysicsReturn:
    """Optical ring modulator via Temporal Coupled-Mode Theory (t-CMT).

    Tracks the complex ring energy amplitude ``a(t)`` as an internal state and
    enforces the output field relation ``E_o = E_i - j sqrt(2/tau_e) a`` via an
    algebraic constraint state ``i_out``.

    The ring energy ODE is expressed in the circulax DAE form F + dQ/dt = 0:

    - F["a"] = -(da/dt RHS) = j sqrt(2/tau_e) V_p1 - j Delta_omega a + a/tau
    - Q["a"] = a

    Gives: da/dt = -j sqrt(2/tau_e) E_i + j Delta_omega a - a/tau.

    Port ``p1`` contributes zero current (S11 = 0 approximation).
    Port ``p2`` is driven by ``i_out`` which enforces the output field.

    Args:
        signals: Complex field amplitudes at input (``p1``) and output (``p2``).
        s: Internal states: ``a`` (complex ring energy) and ``i_out`` (output
            constraint current).
        ng: Group refractive index. Defaults to ``3.8``.
        L: Ring circumference in metres. Defaults to ``2*pi*5e-6``.
        gamma: Through-coupling amplitude coefficient. Defaults to ``0.976``.
        alpha0: Voltage-independent round-trip amplitude. Defaults to ``0.969``.
        alpha1: Linear voltage coefficient of round-trip amplitude (1/V).
            Defaults to ``0.0``.
        voltage: Applied DC bias voltage. Defaults to ``0.0``.
        f_operating: Laser frequency in Hz. Defaults to ``2.2904e14``.
        f_resonance: Ring resonance frequency in Hz. Defaults to ``2.2901e14``.
        v_to_wr: Electro-optic coefficient mapping voltage to resonance shift
            in rad/s/V. Defaults to ``0.0``.

    Promoted illustrative models preserving older ring notebook behavior.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.resonators import RingModulator

        jax.config.update("jax_enable_x64", True)
        model = RingModulator()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(RingModulator.__name__, "ports:", model.ports)
        ```

    """
    c = 2.998e8  # speed of light, m/s

    # Photon lifetimes
    tau_e = 2.0 * ng * L / ((1.0 - gamma**2) * c)
    alpha_v = alpha0 + alpha1 * voltage
    tau_l = 2.0 * ng * L / ((1.0 - alpha_v**2) * c)
    tau = 1.0 / (1.0 / tau_e + 1.0 / tau_l)

    coupling = jnp.sqrt(2.0 / tau_e)

    # Laser-resonance detuning (+ EO shift)
    delta_omega = 2.0 * jnp.pi * (f_operating - f_resonance) + v_to_wr * voltage

    # Ring energy ODE RHS: da/dt = -j*coupling*E_i + j*delta_omega*a - a/tau
    rhs_a = -1j * coupling * signals.p1 + 1j * delta_omega * signals.a - signals.a / tau

    # Output field from the ring: E_o = E_i - j*coupling*a
    E_o_expected = signals.p1 - 1j * coupling * signals.a

    f = {
        "p1": 0.0 + 0.0j,  # ring is transparent at input (S11=0)
        "p2": signals.i_out,
        "i_out": signals.p2 - E_o_expected,  # enforce E_o = E_i - j*coupling*a
        "a": -rhs_a,  # ring energy ODE (negated RHS)
    }
    q = {"a": signals.a}
    return f, q


@component(ports=("in_", "thru"))
def RingModSetup(
    signals: Signals,
    init,
    kappa: float = 0.3,
    neff: float = 2.4,
    alpha: float = 1e-3,
    L: float = float(2 * jnp.pi * 5e-6),
    V_pi: float = 2.0,
    V: float = 0.0,
) -> PhysicsReturn:
    """Promoted illustrative models preserving older ring notebook behavior.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.resonators import RingModSetup

        jax.config.update("jax_enable_x64", True)
        model = RingModSetup()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(RingModSetup.__name__, "ports:", model.ports)
        ```

    """
    phi = init["phi"] * (1.0 + V / V_pi)
    ejphi = jnp.exp(1j * phi)
    H = (init["t"] - init["a"] * ejphi) / (1.0 - init["t"] * init["a"] * ejphi)
    absorbed = (1.0 - jnp.abs(H) ** 2) * signals.in_
    return {"in_": absorbed, "thru": signals.thru - H * signals.in_}, {}


@RingModSetup.setup
def _round_trip(
    kappa: float = 0.3,
    neff: float = 2.4,
    alpha: float = 1e-3,
    L: float = float(2 * jnp.pi * 5e-6),
) -> dict:
    """Round-trip model: derive CMT coefficients from physical device parameters."""
    return {
        "a": jnp.exp(-alpha * L / 2.0),  # round-trip amplitude (loss)
        "t": jnp.sqrt(1.0 - kappa**2),  # through-coupling amplitude
        "phi": 2.0 * jnp.pi * neff * L,  # round-trip phase at reference wavelength
    }


@component(ports=("p1", "p2", "v_e"), states=("a", "i_out"))
def RingModulatorEO(
    signals: Signals,
    ng: float = 3.8,
    L: float = 3.14159265e-5,
    gamma: float = 0.976,
    alpha0: float = 0.969,
    alpha1: float = 0.0,
    f_operating: float = 2.2904e14,
    f_resonance: float = 2.2901e14,
    v_to_wr: float = 0.0,
) -> PhysicsReturn:
    """Ring modulator with an external electrical voltage port ``v_e``.

    The voltage at port ``v_e`` modulates the ring resonance frequency via
    ``v_to_wr`` (rad/s/V).  The port is high-impedance: it draws no current
    (``F["v_e"] = 0``), so any driver circuit sees an open circuit.

    The physics is otherwise identical to :func:`RingModulator`, but ``voltage``
    is read from the external port rather than being a fixed parameter.

    Promoted illustrative models preserving older ring notebook behavior.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.resonators import RingModulatorEO

        jax.config.update("jax_enable_x64", True)
        model = RingModulatorEO()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(RingModulatorEO.__name__, "ports:", model.ports)
        ```

    """
    c = 2.998e8
    voltage = jnp.real(signals.v_e)  # electrical port — real part only

    tau_e = 2.0 * ng * L / ((1.0 - gamma**2) * c)
    alpha_v = alpha0 + alpha1 * voltage
    tau_l = 2.0 * ng * L / ((1.0 - alpha_v**2) * c)
    tau = 1.0 / (1.0 / tau_e + 1.0 / tau_l)
    coupling = jnp.sqrt(2.0 / tau_e)

    delta_omega = 2.0 * jnp.pi * (f_operating - f_resonance) + v_to_wr * voltage

    rhs_a = -1j * coupling * signals.p1 + 1j * delta_omega * signals.a - signals.a / tau
    E_o_expected = signals.p1 - 1j * coupling * signals.a

    f = {
        "p1": 0.0 + 0.0j,
        "p2": signals.i_out,
        "v_e": 0.0 + 0.0j,  # high-impedance electrical port
        "i_out": signals.p2 - E_o_expected,
        "a": -rhs_a,
    }
    q = {"a": signals.a}
    return f, q


def _ring_stamp(signals, ports, wave_names, gamma_i, gamma_1, gamma_2, detuning, z0):
    """Two uncoupled counterpropagating modes, sqrt(J), exact wave constraints."""
    invalid = (gamma_i < 0) | (gamma_1 < 0) | (gamma_2 < 0) | (gamma_i + gamma_1 + gamma_2 <= 0) | (z0 <= 0)
    invalid = invalid | ~jnp.isfinite(gamma_i + gamma_1 + gamma_2 + detuning + z0)
    gamma_i = eqx.error_if(
        jnp.asarray(gamma_i), invalid, "Ring requires finite nonnegative decay rates, positive total decay and z0, finite detuning"
    )
    a = jnp.array([getattr(signals, name) for name in wave_names])
    k1, k2 = jnp.sqrt(2 * gamma_1), jnp.sqrt(2 * gamma_2)
    if len(ports) == 2:
        forward, reverse = k1 * a[0], k1 * a[1]
        b = jnp.array([a[1] - k1 * signals.c_reverse, a[0] - k1 * signals.c_forward])
    else:
        forward, reverse = k1 * a[0] + k2 * a[2], k1 * a[1] + k2 * a[3]
        b = jnp.array(
            [
                a[1] - k1 * signals.c_reverse,
                a[0] - k1 * signals.c_forward,
                a[3] - k2 * signals.c_reverse,
                a[2] - k2 * signals.c_forward,
            ]
        )
    f = {p: (a[k] - b[k]) / jnp.sqrt(z0) for k, p in enumerate(ports)}
    f.update({s: getattr(signals, ports[k]) - jnp.sqrt(z0) * (a[k] + b[k]) for k, s in enumerate(wave_names)})
    decay = gamma_i + gamma_1 + gamma_2 - 1j * detuning
    f.update(c_forward=decay * signals.c_forward - forward, c_reverse=decay * signals.c_reverse - reverse)
    return f, {"c_forward": signals.c_forward, "c_reverse": signals.c_reverse}


@component(ports=("p1", "p2"), states=("a1", "a2", "c_forward", "c_reverse"), holomorphic=True)
def AllPassRing(
    signals: Signals, gamma_i: float = 1e10, gamma_e: float = 1e10, detuning: float = 0.0, z0: float = 1.0
) -> PhysicsReturn:
    """Reciprocal matched two-port single-resonance cavity.

    gamma_i/gamma_e are amplitude decay rates in 1/s (nonnegative, sum>0).
    detuning=2*pi*(f_res-f_ref) in rad/s, carrier convention exp(+i omega_ref t).
    c_forward/c_reverse have units sqrt(J). Energy decays at twice amplitude rate.
    Through H(s)=1-2*gamma_e/(s+gamma_i+gamma_e-i*detuning).
    Valid within one resonance's narrowband CMT region, not multiple FSRs.

    A single resonance coupled to one bus, with intrinsic loss and independent counterpropagating modes.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.resonators import AllPassRing

        jax.config.update("jax_enable_x64", True)
        model = AllPassRing()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(AllPassRing.__name__, "ports:", model.ports)
        ```

    """
    return _ring_stamp(signals, ("p1", "p2"), ("a1", "a2"), gamma_i, gamma_e, 0.0, detuning, z0)


@component(ports=("p1", "p2", "p3", "p4"), states=("a1", "a2", "a3", "a4", "c_forward", "c_reverse"), holomorphic=True)
def AddDropRing(
    signals: Signals, gamma_i: float = 0.0, gamma_1: float = 1e10, gamma_2: float = 1e10, detuning: float = 0.0, z0: float = 1.0
) -> PhysicsReturn:
    """Reciprocal four-port ring with independent forward/reverse cavity states.

    Bus pairs are p1/p2 and p3/p4; excitation p1 exits through p2 and drops p4.
    Couplings gamma_1/gamma_2 and intrinsic gamma_i are amplitude rates 1/s.
    detuning=omega_res-omega_ref. Optical waves sqrt(W), cavity states sqrt(J).
    Zero couplings are exact; total amplitude rate must remain positive.

    A single resonance coupled to two buses.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.resonators import AddDropRing

        jax.config.update("jax_enable_x64", True)
        model = AddDropRing()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(AddDropRing.__name__, "ports:", model.ports)
        ```

    """
    return _ring_stamp(signals, ("p1", "p2", "p3", "p4"), ("a1", "a2", "a3", "a4"), gamma_i, gamma_1, gamma_2, detuning, z0)


@component(ports=("p1", "p2", "v_e"), states=("a1", "a2", "c_forward", "c_reverse"))
def AllPassRingEO(
    signals: Signals, gamma_i: float = 1e10, gamma_e: float = 1e10, detuning: float = 0.0, tuning: float = 1e10, z0: float = 1.0
) -> PhysicsReturn:
    """Ring control core; tuning is resonance shift in rad/s/V at real v_e.

    Electrical input is high impedance for physical real voltage. A unit
    imaginary restoring conductance fixes the auxiliary complex lift and does
    not draw physical real current. Add drive R/C externally or use loaded_ring.

    Resonance detuning controlled by a real electrical voltage.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.resonators import AllPassRingEO

        jax.config.update("jax_enable_x64", True)
        model = AllPassRingEO()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(AllPassRingEO.__name__, "ports:", model.ports)
        ```

    """
    f, q = _ring_stamp(signals, ("p1", "p2"), ("a1", "a2"), gamma_i, gamma_e, 0.0, detuning + tuning * jnp.real(signals.v_e), z0)
    f["v_e"] = 1j * jnp.imag(signals.v_e)
    return f, q


def loaded_ring(*, resistance=50.0, capacitance=100e-15, **settings: float) -> "Circuit":
    """Compose AllPassRingEO with series R/shunt C; external drive port is drive.

    Ports: p1/p2 optical; drive electrical referenced to gnd; junction exposes
    internal physical loading voltage. SI ohms/farads, no duplicated cavity pole.

    Electrically tuned ring behind series R/shunt C.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        resonators.md

    Examples:
        ```python
        from circulax.components.resonators import loaded_ring

        device = loaded_ring()
        assert device.sys_size > 0
        print("External ports:", tuple(device.ports))
        ```

    """
    import numpy as np

    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, Resistor

    if not np.isfinite(resistance) or resistance <= 0 or not np.isfinite(capacitance) or capacitance <= 0:
        msg = "resistance and capacitance must be finite and positive"
        raise ValueError(msg)
    return compile_circuit(
        {
            "instances": {
                "R": {"component": "R", "settings": {"R": resistance}},
                "C": {"component": "C", "settings": {"C": capacitance}},
                "ring": {"component": "_allpass_ring_core", "settings": settings},
            },
            "connections": {"R,p2": "ring,v_e", "C,p1": "ring,v_e"},
            "ports": {"p1": "ring,p1", "p2": "ring,p2", "drive": "R,p1", "gnd": "C,p2", "junction": "ring,v_e"},
        },
        {"R": Resistor, "C": Capacitor, "_allpass_ring_core": AllPassRingEO},
        is_complex=True,
        backend="dense",
    )
