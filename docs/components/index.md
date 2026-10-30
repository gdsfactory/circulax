# Electronic sample components

For a practical introduction, start with [Devices: choosing and connecting components](devices.md).

These models use the ordinary Circulax current/charge interface,
`F(y) + dQ(y)/dt = 0`. They support time-domain modeling rather than requiring
frequency-domain admittance tables. Linearized AC uses the derivatives of the
same equations at the operating point.

Use float64 for the compact models:

```python
import jax

jax.config.update("jax_enable_x64", True)

from circulax.components import BJT_PNP_Dynamic, JUNCAP200, R2CMC

models = {
    "pnp": BJT_PNP_Dynamic,
    "junction": JUNCAP200,
    "resistor": R2CMC,
}
```

Register component classes in the `models_map` passed to `compile_circuit`.
Instance `settings` use the names documented for each model; physical compact
model parameters retain their reference names and units. Internal states are
solver unknowns rather than external netlist terminals.

## Model documentation

| Family | Documentation |
|---|---|
| PNP bipolar transistor, static and dynamic | [PNP models](pnp.md) |
| Generic PMOS with intrinsic and overlap charges | [PMOSDynamic](pmos_dynamic.md) |
| Two-terminal nonlinear resistor, including electrothermal variant | [R2_CMC](r2.md) |
| Three-terminal nonlinear resistor and electrothermal network | [R3_CMC](r3.md) |
| Junction diode and junction-varactor charge | [JUNCAP200](juncap.md) |
| Dynamic junction diode with recovery and high-injection behavior | [DIODE_CMC](diode_cmc.md) |
| MOS varactor with finite-resistance internal network | [MOSVAR](mosvar.md) |
| Four-terminal NMOS/PMOS compact model | [EKV 2.6](ekv.md) |
| RF transistor with parasitic RLC, dispersion and thermal dynamics | [Angelov](angelov.md) |
| GaN RF transistor with external thermal/dispersion ports | [Angelov GaN](angelov_gan.md) |
| Amplifier with gain, bandwidth, output resistance, slew and rail limits | [OpAmp](opamp.md) |
| Comparator with finite response and dynamic hysteresis | [Comparator](comparator.md) |

The model pages describe supported modes and physical parameter domains. In
particular, MOSVAR fixes `SWRES=1` to retain its finite-resistance topology.
The Angelov variants retain their finite-resistance networks and validate
strictly positive resistances; zero-valued inductance and delay settings remain
supported. Their noise-selection parameters are fixed to deterministic operation.
EKV26 preserves the reference's PMOS overlap-charge sign convention; use the
generic `PMOSDynamic` for conventional positive overlap capacitances. R3's
full-pinch-off verification uses finite differences where the reference's
analytic Jacobian disagrees with its residual. Hysteretic comparator circuits
need an initial branch or a prior unambiguous bias, as described on that page.

## Composition

Keep coupled nonlinear current and charge calculations in one core so they can
share intermediate values and maintain consistent terminal charge. Use
subcircuits for surrounding parasitic resistors, capacitors, inductors and
thermal networks. [R2's thermal subcircuit](r2.md#thermal-subcircuit-composition)
is an example using ordinary resistor and capacitor branches with temperature
rise as voltage and heat flow as current.

Circulax flattens hierarchy before solving. Subcircuits improve reuse and
readability, but do not provide separate XLA compilation boundaries or
automatically reduce JIT time.

## Verification and reproduction

The native compact models run without bosdi, OpenVAF, or compiled OSDI files.
Their reference tests optionally compile retained upstream Verilog-A sources
with `openvaf-r` and evaluate them through bosdi. Tests skip those comparisons
when the optional tools are absent; ordinary native checks still run.

Reference comparisons check current `F`, stored charge `Q`, conductance
`dF/dy`, and charge derivatives `dQ/dy`. Tests use selected bias, geometry,
temperature and model-option combinations. Agreement of constitutive
equations does not establish convergence or transient-trajectory agreement
for every possible circuit. Each model page records the checks actually run
and its supported options.

Sources are pinned to VA-Models commit
`6099b4e12e6f66ff530f59b8d01fb909094be85a`. Generated models retain regeneration
scripts under `tests/data/va_models/`. The reference sources' individual notices
are retained there and packaged in `circulax/components/NOTICE.txt` and
`ECL-2.0.txt`. Deterministic models do not implement the references' stochastic
noise-analysis contributions or simulator operating-point metadata.

Measure residual and Jacobian tracing, compilation, and warm execution in a
fresh process:

```sh
python -m benchmarks.sample_components circulax.components.bipolar BJT_PNP_Dynamic \
    --bias '[-2, -0.6, 0]' --settings '{"Tf": 1e-9}'
```

The benchmark keeps model parameters as JAX inputs. This avoids artificially
small timings from folding a single closed-over parameter set into constants.
The measurements include Python dispatch and synchronization, and do not
represent complete circuit-solve time.
