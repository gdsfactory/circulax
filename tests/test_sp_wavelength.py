"""Wavelength characterization uses declared optical boundaries and matched waves."""

import jax
import jax.numpy as jnp
import pytest
import sax

from circulax import Circuit, compile_circuit
from circulax.components.base_component import PhysicsReturn, Signals, component, source
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photonic import OpticalSource
from circulax.netlist import sax_to_kfnetlist
from circulax.wavelength import _validate_boundary


def _optical(wl: float = 1.55) -> sax.SDict:
    """Return a reciprocal two-port spectrum with zero reflection."""
    transmission = 0.6 * jnp.exp(1j * wl)
    return {("o1", "o1"): 0.0, ("o2", "o2"): 0.0, ("o1", "o2"): transmission, ("o2", "o1"): transmission}


@component(ports=("o1", "o2", "bias", "gnd"))
def _biased_optical(signals: Signals) -> PhysicsReturn:
    """A linear optical two-port controlled by a retained electrical voltage."""
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
    """Keep an electrical bias circuit and optional real optical load in the DUT."""
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
    """Independent matched excitations recover every S-matrix column."""
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
    """An actual DUT load remains in addition to temporary matched boundaries."""
    kwargs = {"ports": ["in", "out"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    bare = _device().sp_wavelength(**kwargs)
    loaded_circuit = _device(physical_load=True)
    loaded = loaded_circuit.sp_wavelength(**kwargs)
    assert jnp.abs(loaded["s"][0, 1, 0]) < jnp.abs(bare["s"][0, 1, 0])
    assert "physical_load" in loaded_circuit.source_netlist["instances"]


def test_internal_terminal_is_not_a_characterization_port() -> None:
    """Only selected declared external ports define scattering boundaries."""
    with pytest.raises(ValueError, match="declared external"):
        _device().sp_wavelength(ports=["optical,o1"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")


def test_ideal_source_clamp_is_rejected_without_removing_source() -> None:
    """A clamped boundary cannot be silently converted into an analysis source."""
    circuit = _device(clamp=True)
    with pytest.raises(ValueError, match=r"ideal.*constraint"):
        circuit.sp_wavelength(ports=["in", "out"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")
    assert "laser" in circuit.source_netlist["instances"]


def _custom_clamp_circuit(*, declared: bool) -> Circuit:
    """Clamp port ``a`` with a user-defined ideal source that may declare its constrained terminals."""

    @source(
        ports=("p1", "p2"),
        states=("i_src",),
        holomorphic=True,
        ideal_voltage_constraint_ports=("p1", "p2") if declared else (),
    )
    def field_source(signals: Signals, t: float, V: float = 1e-6) -> PhysicsReturn:  # noqa: ARG001 -- source signature carries time
        return {"p1": signals.i_src, "p2": -signals.i_src, "i_src": (signals.p1 - signals.p2) - V}, {}

    return compile_circuit(
        {
            "instances": {
                "X": {"component": "m", "settings": {"g": 1.0}},
                "E": {"component": "field_source"},
                "GND": {"component": "ground"},
            },
            "connections": {"E,p1": "X,p1", "GND,p1": "E,p2"},
            "ports": {"a": "X,p1", "b": "X,p2"},
        },
        {"m": _series_conductance, "field_source": field_source},
        is_complex=True,
        g_leak=1e-9,
    )


def test_custom_ideal_source_declares_its_clamp_through_the_decorator() -> None:
    """A user-defined source is rejected only because it declares ideal_voltage_constraint_ports."""
    declared = _custom_clamp_circuit(declared=True)
    with pytest.raises(ValueError, match=r"ideal.*constraint"):
        declared.sp_wavelength(ports=["a", "b"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")
    # Clamps are never inferred from the physics: an undeclared source is not detected.
    _validate_boundary(_custom_clamp_circuit(declared=False), ("a", "b"))


def test_electrical_bias_controls_optical_characterization() -> None:
    """Electrical source overrides stay fixed across all optical excitations."""
    circuit = _device(controlled=True)
    kwargs = {"ports": ["in", "out"], "wavelengths": jnp.array([1.5, 1.6]), "amplitude_unit": "sqrt(W)"}
    default = circuit.sp_wavelength(**kwargs)
    biased = circuit.sp_wavelength(**kwargs, params={"bias.V": 4.0})
    assert jnp.allclose(default["s"][:, 1, 0], 0.35, atol=1e-6)
    assert jnp.allclose(biased["s"][:, 1, 0], 0.4, atol=1e-6)


def test_physical_optical_source_baseline_is_subtracted() -> None:
    """An internal physical optical source remains, without contaminating S."""
    circuit = _device(physical_drive=True)
    kwargs = {"ports": ["in", "out"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    first = circuit.sp_wavelength(**kwargs)
    second = circuit.sp_wavelength(**kwargs, params={"laser.power": 4e-6})
    assert not jnp.allclose(first["baseline_solution"], second["baseline_solution"])
    assert jnp.allclose(first["s"], second["s"], atol=1e-6)
    assert "laser" in circuit.source_netlist["instances"]


def test_wavelength_characterization_accepts_canonical_kfnetlist() -> None:
    """Native kfnetlist inputs preserve their boundary through analysis fixtures."""
    original = _device()
    native, overrides = sax_to_kfnetlist(original.source_netlist)
    assert overrides == {}
    circuit = compile_circuit(native, original.source_models, is_complex=True, g_leak=0)
    result = circuit.sp_wavelength(ports=["in", "out"], wavelengths=jnp.array([1.55]), amplitude_unit="sqrt(W)")
    assert jnp.allclose(result["s"][0, 1, 0], 0.6 * jnp.exp(1.55j), atol=1e-6)
    assert set(circuit.ports) == {"in", "out", "bias"}


def _two_port(model: object, *, settings: dict | None = None, **kwargs: object) -> Circuit:
    """Compile a lone two-terminal-pair model whose terminals are both declared ports."""
    return compile_circuit(
        {
            "instances": {"X": {"component": "m", "settings": settings or {}}, "GND": {"component": "ground"}},
            "connections": {},
            "ports": {"a": "X,p1", "b": "X,p2"},
        },
        {"m": model},
        is_complex=True,
        g_leak=0,
        **kwargs,
    )


_WL = jnp.array([1.55])


@component(ports=("p1", "p2"), holomorphic=True)
def _series_conductance(signals: Signals, g: float = 1.0) -> PhysicsReturn:
    """Linear series conductance g between the two ports."""
    current = g * (signals.p1 - signals.p2)
    return {"p1": current, "p2": -current}, {}


@component(ports=("p1", "p2"), holomorphic=True)
def _cubic_conductance(signals: Signals, g: float = 1.0, k: float = 1e6) -> PhysicsReturn:
    """Weakly nonlinear conductance whose S depends on excitation amplitude."""
    voltage = signals.p1 - signals.p2
    current = g * voltage + k * voltage**3
    return {"p1": current, "p2": -current}, {}


@component(ports=("p1", "p2"), holomorphic=False)
def _conjugate_conductance(signals: Signals, alpha: float = 0.5) -> PhysicsReturn:
    """Response that is real-linear but not complex-linear in the field."""
    voltage = signals.p1 - signals.p2
    current = voltage + alpha * jnp.conj(voltage)
    return {"p1": current, "p2": -current}, {}


def test_series_element_has_nonzero_reflection_with_correct_sign() -> None:
    """S11 = z/(z+2) and S21 = 2/(z+2) for a series impedance z between matched ports (z0=1)."""
    result = _two_port(_series_conductance, settings={"g": 1 / 3}).sp_wavelength(
        ports=["a", "b"], wavelengths=_WL, amplitude_unit="sqrt(W)"
    )
    s = result["s"][0]
    assert jnp.allclose(s, jnp.array([[0.6, 0.4], [0.4, 0.6]]), atol=1e-9)


def test_nonreciprocal_two_port_orientation_is_output_from_input() -> None:
    """S[output, input] follows (I - Y)(I + Y)^-1 for an asymmetric admittance matrix.

    A transposed result would pass every reciprocal fixture above.
    """
    y = jnp.array([[1.0, 0.2], [0.7, 0.5]])

    @component(ports=("a", "b"), holomorphic=True)
    def nonreciprocal(signals: Signals) -> PhysicsReturn:
        return {"a": y[0, 0] * signals.a + y[0, 1] * signals.b, "b": y[1, 0] * signals.a + y[1, 1] * signals.b}, {}

    circuit = compile_circuit(
        {
            "instances": {"X": {"component": "m"}, "GND": {"component": "ground"}},
            "connections": {},
            "ports": {"a": "X,a", "b": "X,b"},
        },
        {"m": nonreciprocal},
        is_complex=True,
        g_leak=0,
    )
    s = circuit.sp_wavelength(ports=["a", "b"], wavelengths=_WL, amplitude_unit="sqrt(W)")["s"][0]
    identity = jnp.eye(2)
    expected = (identity - y) @ jnp.linalg.inv(identity + y)
    assert jnp.allclose(s, expected, atol=1e-9)
    assert not jnp.allclose(s, expected.T, atol=1e-3)


def test_normalized_s_is_amplitude_independent_in_linear_regime_only() -> None:
    """Two small amplitudes agree for a linear model; a nonlinear model separates them.

    The fixture proves the check can fail: S moves with amplitude once the model leaves its linear regime.
    """
    kwargs = {"ports": ["a", "b"], "wavelengths": _WL, "amplitude_unit": "sqrt(W)"}
    linear = _two_port(_series_conductance)
    small, smaller = (linear.sp_wavelength(**kwargs, amplitude=a)["s"] for a in (1e-4, 1e-5))
    assert jnp.allclose(small, smaller, atol=1e-9)
    nonlinear = _two_port(_cubic_conductance)
    large, small = (nonlinear.sp_wavelength(**kwargs, amplitude=a)["s"] for a in (2e-4, 1e-6))
    assert not jnp.allclose(large, small, atol=1e-3)


def test_unconverged_dc_solve_is_reported_not_returned_as_scattering() -> None:
    """An exhausted Newton budget raises instead of returning a finite but wrong S."""
    circuit = _two_port(_cubic_conductance, max_steps=1)
    with pytest.raises(ValueError, match="did not converge"):
        circuit.sp_wavelength(ports=["a", "b"], wavelengths=_WL, amplitude=1.0, amplitude_unit="sqrt(W)")


def test_converged_flags_are_returned_per_wavelength_and_column() -> None:
    """The result carries one flag per wavelength, baseline and excitation."""
    result = _device().sp_wavelength(ports=["in", "out"], wavelengths=jnp.array([1.5, 1.6]), amplitude_unit="sqrt(W)")
    assert result["converged"].shape == (2, 3)
    assert bool(jnp.all(result["converged"]))


def test_response_that_is_not_complex_linear_is_refused() -> None:
    """A conjugate-coupled model is real-linear only and must not be reported as a complex S."""
    circuit = _two_port(_conjugate_conductance)
    with pytest.raises(ValueError, match="not complex-linear"):
        circuit.sp_wavelength(ports=["a", "b"], wavelengths=_WL, amplitude_unit="sqrt(W)")


def test_ground_and_aliased_ports_are_rejected() -> None:
    """Selected ports must be distinct non-ground boundary nodes."""
    circuit = compile_circuit(
        {
            "instances": {"R": {"component": "r", "settings": {"R": 1.0}}, "GND": {"component": "ground"}},
            "connections": {"R,p2": "GND,p1"},
            "ports": {"a": "R,p1", "a_alias": "R,p1", "gnd": "GND,p1"},
        },
        {"r": Resistor},
        is_complex=True,
        g_leak=0,
    )
    kwargs = {"wavelengths": _WL, "amplitude_unit": "sqrt(W)"}
    with pytest.raises(ValueError, match="distinct non-ground"):
        circuit.sp_wavelength(ports=["a", "a_alias"], **kwargs)
    with pytest.raises(ValueError, match="distinct non-ground"):
        circuit.sp_wavelength(ports=["a", "gnd"], **kwargs)


def test_circuit_g_leak_is_forwarded_to_the_characterization_bench() -> None:
    """A floating electrical island that solves in the circuit also solves in the bench."""

    @component(ports=("o1", "o2", "e"), holomorphic=True)
    def through(signals: Signals) -> PhysicsReturn:
        return {"o1": signals.o1 - 0.5 * signals.o2, "o2": signals.o2 - 0.5 * signals.o1, "e": 0.0}, {}

    circuit = compile_circuit(
        {
            "instances": {
                "T": {"component": "t"},
                "F1": {"component": "r", "settings": {"R": 10.0}},
                "F2": {"component": "r", "settings": {"R": 10.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"F1,p1": ("T,e", "F2,p2"), "F1,p2": "F2,p1"},
            "ports": {"in": "T,o1", "out": "T,o2"},
        },
        {"t": through, "r": Resistor},
        is_complex=True,
        g_leak=1e-12,
    )
    result = circuit.sp_wavelength(ports=["in", "out"], wavelengths=_WL, amplitude_unit="sqrt(W)")
    assert jnp.allclose(result["s"][0, 1, 0], 4 / 15, atol=1e-6)


def test_sp_wavelength_is_jittable_and_differentiable() -> None:
    """Concrete checks are skipped under tracing so the analysis composes with jit and grad."""
    circuit = _two_port(_series_conductance, settings={"g": 1.0})

    def s11(g: jax.Array) -> jax.Array:
        return circuit.sp_wavelength(ports=["a", "b"], wavelengths=_WL, amplitude_unit="sqrt(W)", params={"X.g": g})["s"][
            0, 0, 0
        ].real

    z = 1 / 2.0
    assert jnp.allclose(jax.jit(s11)(2.0), z / (z + 2), atol=1e-9)
    # d/dg [ (1/g) / (1/g + 2) ] = d/dg [ 1 / (1 + 2g) ] = -2 / (1 + 2g)^2
    assert jnp.allclose(jax.grad(s11)(2.0), -2 / (1 + 2 * 2.0) ** 2, atol=1e-7)
