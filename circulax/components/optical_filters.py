# Copyright (c) 2026 Circulax contributors.
"""Causal matched reciprocal first-order optical envelope filter."""

import equinox as eqx
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component


@component(ports=("p1", "p2"), states=("a1", "a2", "c1", "c2"), holomorphic=True)
def OpticalFilter(
    signals: Signals, bandwidth_hz: float = 1e9, detuning: float = 0.0, transmission: float = 1.0, z0: float = 1.0
) -> PhysicsReturn:
    """Matched low-pass/Lorentzian H(s)=transmission*k/(s+k-i*detuning).

    k=2*pi*bandwidth_hz; bandwidth_hz>0 is intensity half-width at half maximum
    around envelope frequency detuning/(2*pi). transmission is real amplitude
    in [0,1]; z0>0. Cavity-like states c1/c2 store sqrt(J), with dc scaling
    b2=sqrt(k)*transmission*c1. Internal dissipation yields a contractive S.
    This is a generic single-pole envelope filter, not a broadband resonator.

    A matched, reciprocal, causal one-pole Lorentzian envelope filter.

    Device group:
        Photonic / Rings and causal filters

    Device reference:
        optical_filters.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import matplotlib.pyplot as plt
        import numpy as np

        from circulax import attach_testbench, compile_circuit, quadrature_response
        from circulax.components import OpticalCW, OpticalFilter
        from circulax.components.electronic import Resistor

        jax.config.update("jax_enable_x64", True)
        net = {
            "instances": {"FILTER": {"component": "filter", "settings": {"bandwidth_hz": 1e9}}},
            "ports": {"in": "FILTER,p1", "out": "FILTER,p2"},
        }
        net = attach_testbench(
            net,
            sources={"in": {"name": "LIGHT", "component": "light", "settings": {"power": 1e-3}}},
            loads={"out": {"name": "LOAD", "component": "resistor", "settings": {"R": 1.0}}},
        )
        circuit = compile_circuit(net, {"filter": OpticalFilter, "light": OpticalCW, "resistor": Resistor}, is_complex=True)
        frequencies = jnp.geomspace(1e7, 1e10, 40)
        response = quadrature_response(circuit, freqs=frequencies, parameter_drives={"LIGHT.power": 2 * np.sqrt(1e-3)})
        index = circuit.port_map["out"]
        transfer = np.asarray(response["in_phase"][:, index] + 1j * response["quadrature"][:, index])
        np.testing.assert_allclose(transfer, 1 / (1 + 1j * np.asarray(frequencies) / 1e9), atol=1e-9)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.semilogx(np.asarray(frequencies) / 1e9, 20 * np.log10(abs(transfer)))
        ax.set(xlabel="Envelope frequency (GHz)", ylabel="Transmission (dB)", title="Native optical filter")
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        ```

    """
    invalid = (bandwidth_hz <= 0) | (transmission < 0) | (transmission > 1) | (z0 <= 0)
    invalid = invalid | ~jnp.isfinite(bandwidth_hz + transmission + z0 + detuning)
    bandwidth_hz = eqx.error_if(
        jnp.asarray(bandwidth_hz),
        invalid,
        "Filter requires finite positive bandwidth/z0, transmission in [0,1] and finite detuning",
    )
    rate = 2 * jnp.pi * bandwidth_hz
    coupling = jnp.sqrt(rate)
    b1 = transmission * jnp.sqrt(rate) * signals.c2
    b2 = transmission * jnp.sqrt(rate) * signals.c1
    return {
        "p1": (signals.a1 - b1) / jnp.sqrt(z0),
        "p2": (signals.a2 - b2) / jnp.sqrt(z0),
        "a1": signals.p1 - jnp.sqrt(z0) * (signals.a1 + b1),
        "a2": signals.p2 - jnp.sqrt(z0) * (signals.a2 + b2),
        "c1": (rate - 1j * detuning) * signals.c1 - coupling * signals.a1,
        "c2": (rate - 1j * detuning) * signals.c2 - coupling * signals.a2,
    }, {"c1": signals.c1, "c2": signals.c2}
