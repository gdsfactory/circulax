"""Wavelength characterization uses declared optical boundaries and matched waves."""

import jax.numpy as jnp
import pytest
import sax

from circulax import Circuit, compile_circuit
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photonic import OpticalSource
from circulax.netlist import sax_to_kfnetlist


def _optical(wl: float = 1.55) -> sax.SDict:
    """Return a reciprocal two-port spectrum with zero reflection.

    @tags circulax simulation
    """
    transmission = 0.6 * jnp.exp(1j * wl)
    return {("o1", "o1"): 0.0, ("o2", "o2"): 0.0, ("o1", "o2"): transmission, ("o2", "o1"): transmission}


@component(ports=("o1", "o2", "bias", "gnd"))
def _biased_optical(signals: Signals) -> PhysicsReturn:
    """A linear optical two-port controlled by a retained electrical voltage.

    @tags circulax simulation
    """
    transmission = 0.2 + 0.05 * jnp.real(signals.bias - signals.gnd)
    diagonal = (1 + transmission**2) / (1 - transmission**2)
    cross = -2 * transmission / (1 - transmission**2)
    return {
        "o1": diagonal * signals.o1 + cross * signals.o2,
        "o2": diagonal * signals.o2 + cross * signals.o1,
        "bias": 0.0,
        "gnd": 0.0,
    }, {}


def _device(*, physical_load: bool = False, clamp: bool = False, controlled: bool = False, physical_drive: bool = False) -> Circuit:
    """Keep an electrical bias circuit and optional real optical load in the DUT.

    @tags circulax simulation
    """
    instances = {
        "optical": {"component": "optical"},
        "bias": {"component": "voltage", "settings": {"V": 3.0}},
        "bias_load": {"component": "load", "settings": {"R": 1000.0}},
        "GND": {"component": "ground"},
    }
    connections = {"bias,p1": "bias_load,p1", "GND,p1": ("bias,p2", "bias_load,p2")}
    if physical_load:
        instances["physical_load"] = {"component": "load", "settings": {"R": 1.0}}
        connections["optical,o2"] = "physical_load,p1"
        connections["GND,p1"] += ("physical_load,p2",)
    if clamp:
        instances["laser"] = {"component": "laser", "settings": {"power": 1e-6}}
        connections["optical,o1"] = "laser,p1"
        connections["GND,p1"] += ("laser,p2",)
    if controlled:
        connections["optical,bias"] = "bias,p1"
        connections["optical,gnd"] = "GND,p1"
    if physical_drive:
        instances["laser"] = {"component": "laser", "settings": {"power": 1e-6}}
        instances["driver_load"] = {"component": "load", "settings": {"R": 1.0}}
        connections["laser,p1"] = "driver_load,p1"
        connections["driver_load,p2"] = "optical,o1"
        connections["GND,p1"] += ("laser,p2",)
    return compile_circuit(
        {"instances": instances, "connections": connections, "ports": {"in": "optical,o1", "out": "optical,o2", "bias": "bias,p1"}},
        {
            "optical": _biased_optical if controlled else _optical,
            "voltage": VoltageSource,
            "load": Resistor,
            "laser": OpticalSource,
        },
        is_complex=True,
        g_leak=0,
    )


@pytest.mark.parametrize("amplitude", [1e-4, 3e-4])
def test_reciprocal_full_matrix_and_bias_are_preserved(amplitude: float) -> None:
    """Independent matched excitations recover every S-matrix column.

    @tags circulax simulation
    """
    circuit = _device()
    wavelengths = jnp.array([1.50, 1.55, 1.60])
    result = circuit.sp_wavelength(ports=["in", "out"], wavelengths=wavelengths, amplitude=amplitude, amplitude_unit="sqrt(W)")
    expected = 0.6 * jnp.exp(1j * wavelengths)
    assert result["s"].shape == (3, 2, 2)
    assert jnp.allclose(result["s"][:, 0, 1], expected, atol=1e-6)
    assert jnp.allclose(result["s"][:, 1, 0], expected, atol=1e-6)
    assert jnp.allclose(result["s"][:, [0, 1], [0, 1]], 0, atol=1e-6)
    index = result["port_map"]["bias,p1"]
    assert jnp.allclose(result["baseline_solution"][:, index], 3.0)
    assert jnp.allclose(result["solution"][:, :, index], 3.0)
    assert jnp.array_equal(result["wavelengths"], wavelengths)
    assert set(circuit.source_netlist["instances"]) == {"optical", "bias", "bias_load", "GND"}


def test_physical_load_changes_characterization_and_is_not_removed() -> None:
    """An actual DUT load remains in addition to temporary matched boundaries.

    @tags circulax simulation
    """
    kwargs = {"ports": ["in", "out"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    bare = _device().sp_wavelength(**kwargs)
    loaded_circuit = _device(physical_load=True)
    loaded = loaded_circuit.sp_wavelength(**kwargs)
    assert jnp.abs(loaded["s"][0, 1, 0]) < jnp.abs(bare["s"][0, 1, 0])
    assert "physical_load" in loaded_circuit.source_netlist["instances"]


def test_internal_terminal_is_not_a_characterization_port() -> None:
    """Only selected declared external ports define scattering boundaries.

    @tags circulax simulation
    """
    with pytest.raises(ValueError, match="declared external"):
        _device().sp_wavelength(ports=["optical,o1"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")


def test_ideal_source_clamp_is_rejected_without_removing_source() -> None:
    """A clamped boundary cannot be silently converted into an analysis source.

    @tags circulax simulation
    """
    circuit = _device(clamp=True)
    with pytest.raises(ValueError, match=r"ideal.*constraint"):
        circuit.sp_wavelength(ports=["in", "out"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")
    assert "laser" in circuit.source_netlist["instances"]


def test_electrical_bias_controls_optical_characterization() -> None:
    """Electrical source overrides stay fixed across all optical excitations.

    @tags circulax simulation
    """
    circuit = _device(controlled=True)
    kwargs = {"ports": ["in", "out"], "wavelengths": jnp.array([1.5, 1.6]), "amplitude_unit": "sqrt(W)"}
    default = circuit.sp_wavelength(**kwargs)
    biased = circuit.sp_wavelength(**kwargs, params={"bias.V": 4.0})
    assert jnp.allclose(default["s"][:, 1, 0], 0.35, atol=1e-6)
    assert jnp.allclose(biased["s"][:, 1, 0], 0.4, atol=1e-6)


def test_physical_optical_source_baseline_is_subtracted() -> None:
    """An internal physical optical source remains, without contaminating S.

    @tags circulax simulation
    """
    circuit = _device(physical_drive=True)
    kwargs = {"ports": ["in", "out"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    first = circuit.sp_wavelength(**kwargs)
    second = circuit.sp_wavelength(**kwargs, params={"laser.power": 4e-6})
    assert not jnp.allclose(first["baseline_solution"], second["baseline_solution"])
    assert jnp.allclose(first["s"], second["s"], atol=1e-6)
    assert "laser" in circuit.source_netlist["instances"]


def test_wavelength_characterization_accepts_canonical_kfnetlist() -> None:
    """Native kfnetlist inputs preserve their boundary through analysis fixtures.

    @tags circulax simulation
    """
    original = _device()
    native, overrides = sax_to_kfnetlist(original.source_netlist)
    assert overrides == {}
    circuit = compile_circuit(native, original.source_models, is_complex=True, g_leak=0)
    result = circuit.sp_wavelength(ports=["in", "out"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")
    assert jnp.allclose(result["s"][0, 1, 0], 0.6 * jnp.exp(1.55j), atol=1e-6)
    assert set(circuit.ports) == {"in", "out", "bias"}
