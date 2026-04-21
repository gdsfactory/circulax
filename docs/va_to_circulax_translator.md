# Verilog-A → circulax `@component` translator — design notes

**Status:** planning / scoping; no code yet.
**Context:** notes from an investigation of [ChipFlow vajax](https://github.com/ChipFlow/vajax),
a JAX-based SPICE simulator that compiles Verilog-A to JAX functions for
GPU-accelerated device evaluation.  Captures the competitive landscape,
the insight about AOT-compiled HLO artefacts, and the proposed
circulax-native translator design.

---

## 1. The vajax comparison

### Architecture

| | circulax + bosdi | vajax |
|---|---|---|
| Device eval | OpenVAF → `.osdi` (x86-64 shared lib) → bosdi FFI | OpenVAF IR → JAX Python functions → XLA |
| CPU/GPU | CPU only (x86-64 ABI) | **CPU + GPU** |
| Per-step overhead | ~377 µs/step (9-stage ring, post-Tier-3) | ~5–12 µs fixed + per-device JAX work |
| Model startup | ms (load `.osdi` ELF) | **Undocumented, likely seconds to minutes** (compile VA to JAX, then JIT-compile to XLA) |
| `jax.grad` / `jax.vmap` | Yes (via bosdi's `custom_jvp`) | Yes (native) |

### vajax's published numbers

| Benchmark | vajax CPU | vajax GPU | VACASK (inferred) |
|---|---|---|---|
| c6288 (~5 k nodes) | 88 ms/step | **19.81** | ~58 ms/step (from "2.9× faster than VACASK" claim) |
| mul64 (~266 k transistors) | 8 325 | 648 | not stated |

**Key observation:** vajax is only faster than VACASK *on GPU*.  On CPU
it's ~1.5× *slower* than native C++ SPICE on c6288.  So the value prop
of "compile Verilog-A to JAX" is GPU support, not CPU speed.

### Code structure review

Looked at the vajax repo briefly; the OpenVAF-to-JAX compiler is
**not separable** — it's tightly coupled to vajax's MNA simulator
expectations (returns explicit `(cur, cond, chg, cap)` tuples with
hardcoded shape assumptions).  Porting it into circulax would be a
significant refactor: we'd import more of their simulator's
assumptions than we want.

---

## 2. The `.jaxhlo` insight: AOT-compile devices, ship from fab

The startup-cost concern for any VA-to-JAX path is real — compiling
a PSP103-scale Verilog-A model to JAX and then JIT-compiling to XLA
is minutes, not milliseconds.  `.osdi`'s advantage is that it's
precompiled *once* (at foundry/OpenVAF time) and loaded in ms.

**Idea:** do the equivalent for JAX.  `jax.export` serialises a
traced JAX function to StableHLO bytes that can be saved to disk
and reloaded without re-tracing:

```python
# At fab / PDK build-time (once per foundry release):
dev_fn = openvaf_to_jax("psp103.va")              # compile VA → JAX
lowered = jax.jit(dev_fn).lower(v_shape, ...)
hlo = jax.export.export(lowered).serialize()
Path("psp103.jaxhlo").write_bytes(hlo)

# At simulator-time (ms to load):
hlo = jax.export.deserialize(Path("psp103.jaxhlo").read_bytes())
def device_eval(v, p, s):
    return hlo.call(v, p, s)   # inlines into simulator's JIT graph
```

This gives us a **`.jaxhlo` artefact: JAX's answer to `.osdi`** —
precompiled, loads fast, but carries GPU capability and automatic
differentiation.

### `.jaxhlo` vs `.osdi` tradeoffs

| | `.osdi` | `.jaxhlo` |
|---|---|---|
| Precompile & ship | ✅ | ✅ |
| CPU | ✅ | ✅ |
| **GPU** | ❌ | **✅** |
| Differentiable (auto-gradient) | needs bosdi's hand-plumbed `custom_jvp` | ✅ native |
| Cross-simulator | any OSDI-0.4 consumer (VACASK, ngspice, bosdi) | JAX/XLA only |
| Version stability | stable ABI | StableHLO (newer, slightly more brittle across JAX upgrades) |
| Shape polymorphism | built-in | needs `jax.export` dim vars |
| Platform portability | x86-64 CPU | different HLO per CPU/CUDA/TPU |

### Does `.jaxhlo` exist today?

No.  Closest pieces:

- **vajax's internal pipeline** does Verilog-A → JAX at *simulator
  import time*, not as a shipped artefact.
- **OpenVAF's IR** is public with an LLVM backend; no JAX backend yet.
- **`jax.export`** is production-ready.

So the missing piece is a standalone "OpenVAF-IR-to-JAX-component"
tool that emits the `.jaxhlo` artefact.

---

## 3. Circulax-style interface (simpler than vajax's MNA coupling)

vajax's device functions hardcode the MNA contract: `(v, p, s) →
(cur, cond, chg, cap, new_state)`.  Circulax's `@component` contract
is more abstract and MNA-agnostic — `(signals, s, t, **params) →
(f_dict, q_dict)` where `f_dict` is a flow (current) map per port and
`q_dict` is storage (charge) per port.  The compiler handles Jacobian
assembly, vmap batching, static-field separation, etc.

**Proposal:** the VA-to-circulax translator emits a
`@component`-decorated Python function per device.  Consumers use it
like any other circulax component:

```python
# psp103.py  (auto-generated from psp103.va via OpenVAF IR)
from circulax.components.base_component import component

@component(ports=("D", "G", "S", "B"), states=("di", "si"))
def psp103(signals, s, t,
           TYPE=1.0, TR=27.0, DTA=0.0, SWGEO=1.0, ...  # ~280 params
           ) -> tuple[dict, dict]:
    V_D, V_G, V_S, V_B = signals.D, signals.G, signals.S, signals.B
    V_di, V_si = s.di, s.si

    # Hundreds of lines of PSP103 equations in jax.numpy ops,
    # mechanically translated from OpenVAF IR.

    f = {"D": I_D, "G": I_G, "S": I_S, "B": I_B, "di": I_di, "si": I_si}
    q = {"D": Q_D, "G": Q_G, "S": 0.0, "B": 0.0, "di": Q_di, "si": Q_si}
    return f, q
```

User code becomes:

```python
from circulax_devices.psp103 import psp103
models = {"nmos": psp103, "pmos": psp103}
groups, sys_size, port_map = compile_netlist(netlist, models)
```

No `OsdiComponentGroup`, no `_assemble_osdi_group`, no
`osdi_eval_with_handle`, no handle-lifetime rules.  **All the
OSDI-specific machinery accumulated in `circulax/components/osdi.py`
and `circulax/solvers/assembly.py::_assemble_osdi_group` deletes.**

### Wins the translator buys

| Feature | Today (bosdi FFI) | With VA translator |
|---|---|---|
| GPU dispatch | ❌ (x86-64 ABI) | ✅ (pure JAX) |
| `jax.grad` | via hand-plumbed `custom_jvp` | ✅ automatic |
| `jax.vmap` over params | needs `with_params` + handle rebuild + JIT re-cache | ✅ params are kwargs, trivially vmappable |
| Residual-only fast path | explicit Tier-2 API | ✅ `assemble_residual_only_real` already does this |
| Schur reduction | special flag + branch in assembly | just call `jax.jacfwd` + Schur on the result |
| Per-circuit startup | ms (load `.osdi`) | slow (minutes, PSP103 scale) → needs `jax.export` caching |
| Codebase complexity | `OsdiComponentGroup`, 4 API tiers, handle lifetime rules | deletes all of that |

---

## 4. Firm constraint: foundries ship `.osdi`, not `.va`

Production foundries ship `.osdi` binaries, not Verilog-A source,
for IP reasons (source would expose proprietary model tweaks).  The
OSDI standard doesn't include IR in the binary, so you can't
round-trip `.osdi` → JAX without the source.

This means:

- Foundry-PDK workflows are stuck on bosdi + `.osdi` + CPU for the
  foreseeable future.  **GPU is not going to happen for these users
  without foundry-level cooperation on a new artefact format.**
- The VA-to-circulax translator is **additive, not a replacement**.
  It gives a second path for users who do have Verilog-A source
  (open-source models, academic research, custom model development).

### User populations

| User | Has | Path | GPU? |
|---|---|---|---|
| Foundry-PDK design (production) | `.osdi` | bosdi FFI | ❌ |
| Open-source / research | `.va` | VA translator | ✅ |
| Mixed (custom + PDK) | Both | Both paths coexist in one netlist | partial |

Circulax's `models_map` dict already handles mixing — a foundry
PSP103 loaded via `osdi_component(...)` alongside a translated
research model is just two entries in the dict.

---

## 5. Design decisions to settle before coding

1. **Residual + Jacobian: auto-diff vs explicit.**
   Circulax today uses `jax.jacfwd` on `@component` functions.
   For PSP103 (hundreds of equations) that's 6 forward-mode sweeps —
   slow to JIT-compile.  Options:
   - **A.** Rely on `jax.jacfwd`.  Simplest, works today, pays the
     compile-time cost up front.
   - **B.** Emit a parallel `psp103_jac` function from OpenVAF's
     derivative IR.  Faster, more plumbing.
   - **C.** Emit just the residual (`jax.jvp`-friendly).  Reverse-mode
     for grad, forward-mode for the circuit Jacobian.  Best of both
     if feasible.

2. **Where does the OpenVAF IR come from.**
   - OpenVAF (Rust) emits in-memory IR consumed by its LLVM and OSDI
     backends.  A new backend emitting Python text (or JAX ops at
     runtime) would live alongside those.
   - vajax consumes the IR directly at runtime, not as shipped text.
   - Our translator could emit Python files (ship from foundry,
     debuggable) or emit JAX at runtime (no artefact).  Leaning
     toward Python files because they're debuggable and composable
     with `jax.export` for the `.jaxhlo` story.

3. **Startup cost.**
   Solved by `jax.export`-caching: foundry (or maintainer) runs the
   VA translator once, runs circulax's `jax.export` once, ships
   `psp103.jaxhlo`.  User's first call is ms, not minutes.

---

## 6. Revised project scope

Given the `.osdi`-foundry constraint, this is **additive, not
replacement**, so scope is smaller than a full bosdi-replacement
effort.

### Proposal

- **Phase 1 (proof-of-concept, ~1–2 weeks):**
  Verilog-A resistor, capacitor, and diode translated to
  `@component` functions.  Verify they compose with existing
  circulax solvers end-to-end.  Demonstrate GPU dispatch on a small
  circuit that doesn't use OSDI.

- **Phase 2 (mid-complexity devices, ~2–3 weeks):**
  BSIM4 or similar.  Requires more of OpenVAF's IR surface — analog
  operators, charge equations, temperature scaling.

- **Phase 3 (PSP103 coverage, ~3–4 weeks):**
  The full foundry-complexity case.  Edge cases like `$limit`,
  `$simparam`, bin/stretch interactions, smoke-test against the
  VACASK reference.

- **Phase 4 (`.jaxhlo` tooling):**
  `jax.export` pipeline; `circulax-compile psp103.py > psp103.jaxhlo`
  CLI.  Loader that transparently swaps to the compiled artefact.

Total ~2 months for full PSP103 coverage; much shorter for
"demonstration of concept" on simple devices.

### Collaboration question

This is a tool that belongs one layer below circulax — a shared
`openvaf-to-jax-component` thing that vajax (or any other
JAX-based simulator) could also consume if they wanted to harmonise.
Two options:

- **A.** Pitch as collaboration with ChipFlow.  They'd benefit
  (solves their startup problem too) and we'd split the engineering.
  Less control, less risk.
- **B.** Ship as circulax-only.  More control, more work.

---

## 7. Impact on current positioning

The durable circulax-vs-VACASK wins for **foundry-PDK workloads**
(where `.osdi` is the only input available):

1. `jax.grad` — already works through bosdi's `custom_jvp`.
2. Unified DC/AC/HB/transient — one compile, all analyses.
3. JAX-ecosystem composition — Optax, Equinox, photonic components,
   circuit-in-the-loop ML.

We concede GPU and per-sim wall for these users.  That's real, stop
handwaving it.

The VA translator opens a separate door to a research / open-source
audience that also gets GPU.  Different user, different value prop,
doesn't change the foundry-PDK story.

---

## 8. Open questions (for future discussion)

1. Is the scope above ("additive, VA-only, don't replace bosdi")
   what we actually want, or do we want the translator to
   eventually be complete enough that someone with `.va` sources
   could fully replace their bosdi workflow?
2. Do we prototype standalone or try to align with ChipFlow on
   sharing a common VA-to-JAX tool?
3. If we do the `.jaxhlo` work, does it make sense to propose this
   as a new artefact standard (industry coordination) or keep it
   circulax-internal?
4. What's the cost/benefit of emitting explicit Jacobians from
   OpenVAF's derivative IR vs relying on `jax.jacfwd`?  We should
   probably measure on a mid-complexity device (BSIM3?) before
   committing.
