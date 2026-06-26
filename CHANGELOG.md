# Changelog

All notable changes to circulax are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **netlist**: Accept GDSFactory 'nets' list alongside 'connections'([`1c5ce34`](https://github.com/gdsfactory/circulax/commit/1c5ce348fa56a12d6402fc226e712a452873c8a8))
- **compiler**: GDSFactory interop — filter unknown settings, static non-numeric fields([`98af777`](https://github.com/gdsfactory/circulax/commit/98af7777bf1fbf671c84337933d1aacc50e76e68))
- **changelog**: Add git-cliff config, pixi tasks, and GitHub Release automation([`92fcc6c`](https://github.com/gdsfactory/circulax/commit/92fcc6c46fc832df8e34f9d4eea372c5bff19ac3))

### Changed

- Revert tol to 1e-6, reserve 'ground', clean up from review([`e5ddc10`](https://github.com/gdsfactory/circulax/commit/e5ddc10497879e980755969c00e4f39c92c555d7))

### Documentation

- **compiler**: Document silent settings-filter behaviour in compile_netlist([`94d7dad`](https://github.com/gdsfactory/circulax/commit/94d7daddbca6d1038d523029b719852ac1db4967))

### Fixed

- **solver**: Scale-invariant Newton damping + expose rtol/atol([`0c1b88a`](https://github.com/gdsfactory/circulax/commit/0c1b88a6cde6e86cc470dd918404dd108bf97b3e))
- **testbench**: Include testbench.py that was missing from e1235a7([`a8253c1`](https://github.com/gdsfactory/circulax/commit/a8253c1b3bdd6d0300a8f90093a45cdca12d840f))
## [0.1.5] - 2026-04-23

### Documentation

- Added animations to start of examples([`75828a4`](https://github.com/gdsfactory/circulax/commit/75828a4c9aab88d01fdfd16797ee1c6d1fbd8633))
## [0.1.4] - 2026-04-17

### Added

- Add SAX-like Circuit API with vmap broadcasting([`ffed239`](https://github.com/gdsfactory/circulax/commit/ffed239ee1b2102b7ea6d1047ac6a6058f06474f))
- Integrate bodi OSDI device model interface([`0bda450`](https://github.com/gdsfactory/circulax/commit/0bda45093046db6779c8c6c7433a86eae6932c92))
- **photonic**: Add TunableBeamSplitter component([`98ea51c`](https://github.com/gdsfactory/circulax/commit/98ea51caedb00a23faf6eee30d75433babdd7afe))
- Add three backpropagation advantage demo notebooks([`afdca13`](https://github.com/gdsfactory/circulax/commit/afdca1371f8f1a690e50f6dd0a73ca786be1c50a))
- Add osc_node multi-start to setup_harmonic_balance([`5fc15ee`](https://github.com/gdsfactory/circulax/commit/5fc15ee71fc85213564ebdce57eb2b6be3a70e29))
- **demos**: Rewrite photonic demux demo with MZI topology and length parameterisation([`aa2b119`](https://github.com/gdsfactory/circulax/commit/aa2b119f577a6958eb96fce08bb845f4190206b1))
- **demos**: Adds lattice filter demo([`6873441`](https://github.com/gdsfactory/circulax/commit/6873441e6ad55817c4ea65ee1e5fa6cee2ca3e11))
- **demos**: Added hemt_pa_optimization([`355b954`](https://github.com/gdsfactory/circulax/commit/355b9542a4a43ec9612fb0b2dc4941a8a1fb7dcc))

### Changed

- **examples**: Reorganise into getting_started / electrical / photonic / inverse_design([`00acb53`](https://github.com/gdsfactory/circulax/commit/00acb53330ddd7d6fc06dca7159c80d45d1dfc14))

### Documentation

- Migrate all notebooks and docs to compile_circuit API([`008ea4f`](https://github.com/gdsfactory/circulax/commit/008ea4f018158830baffeb379e5532620943e5d3))
- Updating docs([`80c7719`](https://github.com/gdsfactory/circulax/commit/80c7719417cb1f62eba9293ab2862238bd2df74c))
- Reorganise([`cffa3ea`](https://github.com/gdsfactory/circulax/commit/cffa3eaf9e5afad788f01d1a0c8c428d4bea4527))
- Rewrite README/index as short punchy quickstart with animated LCR demo([`0c85a24`](https://github.com/gdsfactory/circulax/commit/0c85a244541204780fb0a526de92bd7f22f92f44))
- Restore references/ directory nav entry to include all API reference pages([`9ae30ab`](https://github.com/gdsfactory/circulax/commit/9ae30ab3e46fa34328be577ce51a10895d7a8a03))
- Add brief component definition example to README([`65fb246`](https://github.com/gdsfactory/circulax/commit/65fb2464b9e52e357b52943c52832b7ff5ddb7bd))
- Add inverse design guide page, component examples, and backprop section in README([`a7a5657`](https://github.com/gdsfactory/circulax/commit/a7a5657cf30aaabc315865cc3307a68ed2d84f06))
- Reorganise examples, remove tracked SVGs, update notebooks([`7474a4d`](https://github.com/gdsfactory/circulax/commit/7474a4d257c2f618e34499b5d117a9cb27ed21ef))
- Update all docs and cleanup([`ecceafb`](https://github.com/gdsfactory/circulax/commit/ecceafb453a3fed0ea3849a675812a3e8b4cf4b9))
- Tighten prose, remove AI-isms and filler across all docs([`d98288a`](https://github.com/gdsfactory/circulax/commit/d98288a332fd3c698069a481a22219961b487f71))

### Fixed

- **demos**: Correct LC filter netlist port topology([`0687618`](https://github.com/gdsfactory/circulax/commit/0687618682cfcf1e67b045921ab0caa7d86284ae))
- **hb**: Add y_flat_init to run_hb; fix oscillator notebook([`b5050e0`](https://github.com/gdsfactory/circulax/commit/b5050e0fdb53ca236da48b8ee0c78ce2f2cb3011))
- **demos**: Use physically-consistent HB initial condition for VDP oscillator([`771f0ee`](https://github.com/gdsfactory/circulax/commit/771f0ee8fcce0035f3fd5865725e590b452e21b6))
- **demos**: Set A_init=2.0 for VDP HB initial condition([`f7fb303`](https://github.com/gdsfactory/circulax/commit/f7fb3030d168ed173b25d1c7682787b772ead3f8))
- **nb2**: Use fixed warm-start inside loss functions for correct gradients([`5329a6f`](https://github.com/gdsfactory/circulax/commit/5329a6f57351ef04ed48030f4c56ac75dd304741))
- **nb2**: Fix VdP formula so amplitude is tunable via mu([`efbb2f5`](https://github.com/gdsfactory/circulax/commit/efbb2f5e202f40aa1b11eefc893a9d723665380e))
## [0.1.3] - 2026-03-11

### Added

- Add frequency-domain component support in harmonic balance([`6d07248`](https://github.com/gdsfactory/circulax/commit/6d07248f2e7fdc8cd0235fd8fb8f37062d40d85f))
- Add DC homotopy convergence (GMIN + source stepping)([`70bd9c9`](https://github.com/gdsfactory/circulax/commit/70bd9c9bc1c49693862ed76500a36107e4ce56bb))
- Fix KLUSplitFactorSolver and activate FactorizedTransientSolver([`02b2a9d`](https://github.com/gdsfactory/circulax/commit/02b2a9de7481ca5b0f35b9fdedc42d86165a5917))
- Add KLUSplitLinear/Quadratic and RefactoringTransientSolver([`f5cf37e`](https://github.com/gdsfactory/circulax/commit/f5cf37e62c506e98685912ee953bda257b461481))
- Add BDF2 (2nd-order) transient solvers via companion method([`040a609`](https://github.com/gdsfactory/circulax/commit/040a60914ea11a442cf5e165386180b7bd11c88d))
- Add SDIRK3 (3rd-order A-stable) transient solvers([`1ae1664`](https://github.com/gdsfactory/circulax/commit/1ae1664e140c58977dccacf65b4a4b28ba7b91d9))
- Make BDF2VectorizedTransientSolver the default in setup_transient([`e939eda`](https://github.com/gdsfactory/circulax/commit/e939eda716793ee232e342741d1843b5b758f652))
- Add NGSpice accuracy benchmarks for electronic circuits([`aa95c55`](https://github.com/gdsfactory/circulax/commit/aa95c55a02f2811834d17a546f54effc1d910623))
- Add LC ladder scalability testbench([`5d51ef7`](https://github.com/gdsfactory/circulax/commit/5d51ef7860f5f6f7cded0eff60d9b26f7ef17e41))
- Add circuit_diffeqsolve stripped of unused diffrax features([`894ef07`](https://github.com/gdsfactory/circulax/commit/894ef07cf5d0bb11a3c3bd452d01d808dc261fe1))
- Add diode clipper HB vs NGSpice benchmark with frequency sweep([`70e2fae`](https://github.com/gdsfactory/circulax/commit/70e2fae11574189912b1dd1412fb02494b9882ea))

### Changed

- Rename benchmarks to *_testbench and add extensible runner([`3b09c5f`](https://github.com/gdsfactory/circulax/commit/3b09c5feeac96941ee46be09a9cbbbe70b1b6779))

### Documentation

- Document all four component decorators in writing_components.md([`b098b87`](https://github.com/gdsfactory/circulax/commit/b098b87cdb50a4dec8002ab419f1c6a9ff83e1c1))
- Switch to light mode default, fix logo, flatten examples, simplify pixi tasks([`b81d69b`](https://github.com/gdsfactory/circulax/commit/b81d69b3a2fb54d197e7144f97a56af6b266d64a))
- Pre-release update — fix stale API, add AC sweep, add HB frequency sweep guide([`bd1825f`](https://github.com/gdsfactory/circulax/commit/bd1825fb58bc1bd2a80622741f98c613a9996394))
- Add 'Choosing a Solver' guide([`4ffd471`](https://github.com/gdsfactory/circulax/commit/4ffd471215ee303b78b5559bcb394f372530e60a))
- Correct solver guide — all backends support jax.vmap and jax.grad([`515d4d9`](https://github.com/gdsfactory/circulax/commit/515d4d95956583437a0ac4b2aef93f6868566c9c))
- Highlight GPU support for sparse backend([`de8f9db`](https://github.com/gdsfactory/circulax/commit/de8f9dbf20f35333e5ddd08f34e4b9ff5e75ced8))
- Add transient simulation guide covering accuracy and PID step control([`66bde72`](https://github.com/gdsfactory/circulax/commit/66bde726fd59d89c954e62a452c99604a192afcd))
- Reframe solver guide — KLU is the default and gold standard([`195b4e8`](https://github.com/gdsfactory/circulax/commit/195b4e890274e7ab81f5c45f1fa2522966930629))
- Regenerate example pages and update lock file([`6267c39`](https://github.com/gdsfactory/circulax/commit/6267c394fbb739fc456bc79176904254d81d5e7f))

### Fixed

- Correct _junction_charge sign and linear extrapolation anchor([`37b24eb`](https://github.com/gdsfactory/circulax/commit/37b24eb890128dfbc941140e5a28d78c5c0ad464))
## [0.1.2] - 2026-03-04

### Added

- Add Harmonic Balance solver with jax.vmap frequency sweep demo([`f2f0295`](https://github.com/gdsfactory/circulax/commit/f2f02959b11938724c369da40f72e726a44ef7dd))
- Expose internal state indices in port_map; clean up ring modulator notebook([`ce4b9c8`](https://github.com/gdsfactory/circulax/commit/ce4b9c89753a6fdabf427249c1158270adcb14af))
- Add Part 4 NRZ eye diagram to ring modulator notebook([`617e91a`](https://github.com/gdsfactory/circulax/commit/617e91a020b031df79ced3893a469050d45f61bf))

### Changed

- Reorganize examples into electrical/photonic subfolders by analysis type([`b7f6d8f`](https://github.com/gdsfactory/circulax/commit/b7f6d8f764498dfe9b923c59242d859c50d554f2))
## [0.1.1] - 2026-03-01

### Changed

- Deduplicate solver factory methods and rename from_circuit([`ee48b98`](https://github.com/gdsfactory/circulax/commit/ee48b9893d9a90a9a5e698bb02200e168af1cab6))
## [0.1.0] - 2026-02-18

### Documentation

- Edits([`dda6c8d`](https://github.com/gdsfactory/circulax/commit/dda6c8d2bdc99bfcfc6a2669358ebbd44c6ef210))
## [0.1.0a1] - 2026-02-18

### Added

- Added KlursSplitSolver([`912fc14`](https://github.com/gdsfactory/circulax/commit/912fc14d8af3041193ad818fc9c71458fa6cda21))
- Added sax decorator([`3e50b8c`](https://github.com/gdsfactory/circulax/commit/3e50b8c8a6b5b32f4a501acc65d0a0d55d7d56ce))

### Changed

- Functional definition. test update([`94dace4`](https://github.com/gdsfactory/circulax/commit/94dace46ff4d9d9341077cd5ef621f062f454338))

### Documentation

- Modified writing components.md([`9e1b187`](https://github.com/gdsfactory/circulax/commit/9e1b18759089733825ef0c505a2f46e2856ef893))
- Updated readme([`8e58ceb`](https://github.com/gdsfactory/circulax/commit/8e58cebd004b15270d84a102a2ff0cc7ce5a8b2c))
- Removed link to favicon in mkdocs.yaml([`da8b803`](https://github.com/gdsfactory/circulax/commit/da8b803397e1a0b7fd51e50d6e33a6d7f20de22e))
- Cleaned index([`791e10e`](https://github.com/gdsfactory/circulax/commit/791e10e4ee5abd0ad6eb123c2e84097e243c5e87))
