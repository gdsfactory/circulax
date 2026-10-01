"""Compare original IHP native VACASK libraries against Circulax.

Run with python benchmarks/ihp_parity/run.py --help. Results record tolerances,
absolute error and vector scales; unsupported runtime features are explicit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from circulax.netlist_io import Library
from circulax.netlist_io.reference import run_vacask
from circulax.solvers.assembly import assemble_gc_real, assemble_residual_only_real


def compare(actual: Any, expected: Any, *, rtol: float = 1e-5, atol: float = 1e-9) -> dict[str, Any]:
    """Compare finite vectors with recorded relative and absolute tolerances."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual - expected)
    return {
        "passed": bool(np.all(np.isfinite(actual)) and np.all(error <= atol + rtol * np.abs(expected))),
        "max_absolute_error": float(error.max()),
        "reference_max": float(np.abs(expected).max()),
        "rtol": rtol,
        "atol": atol,
    }


def main() -> None:  # noqa: C901, PLR0912, PLR0915 -- explicit comparison matrix
    """Run the selected numerical comparisons and save their measured errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdk", type=Path, required=True)
    parser.add_argument("--vacask", type=Path, required=True)
    parser.add_argument("--module-path", type=Path, action="append", default=[])
    parser.add_argument("--shared-library-path", type=Path, action="append", default=[])
    parser.add_argument("--compiler", default="openvaf-r")
    parser.add_argument("--only", default="", help="Substring selecting comparisons")
    parser.add_argument("--output", type=Path, default=Path("ihp-parity.json"))
    args = parser.parse_args()
    root = args.pdk.resolve() / "ihp/models/vacask/models"
    results = {}

    def run(  # noqa: PLR0915 -- reference/analysis dispatch
        name: str,
        library: tuple[str, str],
        statements: str,
        analysis: str,
        raw: str = "op1.raw",
        *,
        measure: str = "out",
        mode: str = "op",
        temperature_c: float = 26.85,
    ) -> None:
        if args.only and args.only not in name:
            return
        header = f'include "{root / library[0]}" section={library[1]}\n'
        circuit_text = header + "model vs vsource\nmodel parity_r sp_resistor\n" + statements
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "circuit.lib"
            path.write_text(circuit_text)
            resolved = Library.from_file(path, temperature_c=temperature_c).resolve()
            circuit = resolved.compile(module_paths=tuple(args.module_path), compiler=args.compiler, state_policy="limiting_only")
            tolerance_options = "reltol=1e-6 vntol=1e-8 abstol=1e-12" if mode == "tran" else "reltol=1e-8 vntol=1e-10 abstol=1e-14"
            deck = (
                "IHP parity: "
                + name
                + "\nground 0\n"
                + circuit_text
                + f'\ncontrol\nabort always\noptions temp={temperature_c} rawfile="binary" {tolerance_options}\n'
                + analysis
                + "\nendc\n"
            )
            reference = run_vacask(
                deck,
                binary=args.vacask,
                module_paths=tuple(args.module_path),
                shared_library_paths=tuple(args.shared_library_path),
                compiler=args.compiler,
                resolved=resolved,
                raw_filename=raw,
            )

            def read(y: jax.Array) -> jax.Array:
                if ":flow(br)" not in measure:
                    return circuit.port(y, measure)
                source = measure.split(":", maxsplit=1)[0]
                key = f"device{next(i for i, leaf in enumerate(resolved.instances) if leaf.name == source)}"
                group = next(group for group in circuit.groups.values() if key in (group.index_map or {}))
                return y[..., group.var_indices[group.index_map[key], -1]]

            if mode == "op":
                actual = read(circuit.dc())
            elif mode == "dc":
                actual = read(circuit.dc(params={"device0.V": jnp.asarray(reference.vectors["vin"].real)}))
            elif mode == "ac":
                dc = circuit.dc()
                # VACASK retains its OP conductance matrix and adds only the
                # reactive Jacobian from a separate AC evaluation (important for idt).
                gv, _ = assemble_gc_real(dc, circuit.groups)
                circuit = circuit._for_analysis("ac")  # noqa: SLF001 -- inspect native AC registration
                _, cv = assemble_gc_real(dc, circuit.groups)
                rows = np.concatenate([np.asarray(group.jac_rows).reshape(-1) for _, group in sorted(circuit.groups.items())])
                cols = np.concatenate([np.asarray(group.jac_cols).reshape(-1) for _, group in sorted(circuit.groups.items())])
                g = np.zeros((circuit.sys_size, circuit.sys_size))
                c = np.zeros_like(g)
                np.add.at(g, (rows, cols), np.asarray(gv))
                np.add.at(c, (rows, cols), np.asarray(cv))

                # Drive the source constraint with a unit small-signal voltage.
                # dF/dV is computed from the same DC component, avoiding index assumptions.
                def residual(amplitude: jax.Array) -> jax.Array:
                    groups = circuit._with_param_values({"device0.V": amplitude})  # noqa: SLF001 -- benchmark source excitation
                    return assemble_residual_only_real(dc, groups, 0.0, 0.0)[0]

                rhs = -jax.jacfwd(residual)(jnp.asarray(0.0))
                omega = 2 * np.pi * reference.vectors["frequency"].real
                actual = []
                for frequency in omega:
                    matrix = np.asarray(g + 1j * frequency * c).copy()
                    matrix[0, :] = 0
                    matrix[0, 0] = 1
                    forcing = np.asarray(rhs).copy()
                    forcing[0] = 0
                    y = np.linalg.solve(matrix, forcing)
                    actual.append(read(jnp.asarray(y)))
                np.savez(
                    args.output.with_name(name + ".npz"),
                    frequency=reference.vectors["frequency"],
                    actual=actual,
                    reference=reference.vectors[measure],
                    conductance=g,
                    capacitance=c,
                    dc=dc,
                )
            else:
                dc = circuit.dc()
                times = reference.vectors["time"].real
                solution = circuit.transient(
                    t0=0.0, t1=float(times[-1]), dt0=1e-12, y0=dc, saveat=jnp.asarray(times), max_steps=20000, throw=True
                )
                actual = circuit.port(solution.ys, measure)
                np.savez(
                    args.output.with_suffix(".npz"),
                    time=times,
                    actual=actual,
                    reference=reference.vectors[measure],
                    source=circuit.port(solution.ys, "in"),
                    source_reference=reference.vectors["in"],
                )
            results[name] = compare(
                actual, reference.vectors[measure], rtol=3e-3 if mode == "tran" else 1e-5, atol=1e-4 if mode == "tran" else 1e-9
            )
            if mode == "tran":
                source_comparison = compare(circuit.port(solution.ys, "in"), reference.vectors["in"], rtol=1e-9, atol=1e-9)
                results[name]["input_max_absolute_error"] = source_comparison["max_absolute_error"]
                results[name]["passed"] &= source_comparison["passed"]
            args.output.write_text(json.dumps(results, indent=2) + "\n")

    for nx in [1, 4, 10]:
        for nqs in [0, 1]:
            statements = (
                "vb (base 0) vs dc=0.8 mag=1\nvc (collector 0) vs dc=1.2\n"
                f"q1 (collector base 0 0) npn13G2 nx={nx} sw_nqs={nqs} selft=0\n"
            )
            for mode, analysis, raw in [
                ("op", "analysis op1 op", "op1.raw"),
                ("ac", 'analysis ac1 ac from=1e6 to=1e9 mode="dec" points=3', "ac1.raw"),
            ]:
                name = f"hbt_nx{nx}_nqs{nqs}_{mode}"
                try:
                    run(name, ("cornerHBT.lib", "hbt_typ"), statements, analysis, raw, measure="vc:flow(br)", mode=mode)
                except RuntimeError as error:
                    results[name] = {"passed": False, "error": str(error), "selft": 0}
                    args.output.write_text(json.dumps(results, indent=2) + "\n")

    for corner in ["mos_tt", "mos_ss", "mos_ff", "mos_sf", "mos_fs"]:
        for nf in [1, 2, 3]:
            name = f"lv_nmos_dc_{corner}_nf{nf}"
            run(
                name,
                ("cornerMOSlv.lib", corner),
                f"vg (gate 0) vs dc=0.7\nvd (drain 0) vs dc=0.8\nx1 (drain gate 0 0) sg13_lv_nmos w=2u l=0.13u ng={nf} m=2\n",
                "analysis op1 op",
                measure="vd:flow(br)",
            )
    for corner in ["mos_tt", "mos_ss", "mos_ff", "mos_sf", "mos_fs"]:
        run(
            "rf_nmos_" + corner,
            ("cornerMOSlv.lib", corner),
            "vg (gate 0) vs dc=0.7\nvd (drain 0) vs dc=0.8\nx1 (drain gate 0 0) sg13_lv_nmos w=2u l=0.13u ng=2 rfmode=1\n",
            "analysis op1 op",
            measure="vd:flow(br)",
        )
        run(
            "hv_nmos_" + corner,
            ("cornerMOShv.lib", corner),
            "vg (gate 0) vs dc=2\nvd (drain 0) vs dc=2\nx1 (drain gate 0 0) sg13_hv_nmos w=2u l=0.45u ng=2\n",
            "analysis op1 op",
            measure="vd:flow(br)",
        )
    for corner in ["res_typ", "res_bcs", "res_wcs"]:
        for model in ["rsil", "rhigh", "rppd"]:
            run(
                model + "_" + corner,
                ("cornerRES.lib", corner),
                f"v1 (in 0) vs dc=0.2\nx1 (in 0 0) {model} w=1u l=5u\n",
                "analysis op1 op",
                measure="v1:flow(br)",
            )
    for temperature in [-40.0, 27.0, 125.0]:
        run(
            f"temperature_nmos_{temperature}",
            ("cornerMOSlv.lib", "mos_tt"),
            "vg (gate 0) vs dc=0.7\nvd (drain 0) vs dc=0.8\nx1 (drain gate 0 0) sg13_lv_nmos w=2u l=0.13u ng=2\n",
            "analysis op1 op",
            measure="vd:flow(br)",
            temperature_c=temperature,
        )
        run(
            f"temperature_resistor_{temperature}",
            ("cornerRES.lib", "res_typ"),
            "v1 (in 0) vs dc=0.2\nx1 (in 0 0) rppd w=1u l=5u\n",
            "analysis op1 op",
            measure="v1:flow(br)",
            temperature_c=temperature,
        )
    for rfmode in [0, 1]:
        run(
            f"nmos_rfmode{rfmode}_ac",
            ("cornerMOSlv.lib", "mos_tt"),
            "vg (gate 0) vs dc=0.6 mag=1\nvdd (vdd 0) vs dc=1.2\nr1 (vdd out) parity_r r=1k\n"
            f"x1 (out gate 0 0) sg13_lv_nmos w=1u l=0.13u ng=2 rfmode={rfmode}\n",
            'analysis ac1 ac from=1e6 to=1e9 mode="dec" points=3',
            raw="ac1.raw",
            mode="ac",
        )
    for logic in ["nand", "nor"]:
        for a, b in [(0, 0), (0, 1.2), (1.2, 0), (1.2, 1.2)]:
            statements = f"va (a 0) vs dc={a}\nvb (b 0) vs dc={b}\nvdd (vdd 0) vs dc=1.2\n"
            if logic == "nand":
                statements += (
                    "xn1 (out a mid 0) sg13_lv_nmos w=1u l=0.13u\n"
                    "xn2 (mid b 0 0) sg13_lv_nmos w=1u l=0.13u\n"
                    "xp1 (out a vdd vdd) sg13_lv_pmos w=2u l=0.13u\n"
                    "xp2 (out b vdd vdd) sg13_lv_pmos w=2u l=0.13u\n"
                )
            else:
                statements += (
                    "xn1 (out a 0 0) sg13_lv_nmos w=1u l=0.13u\n"
                    "xn2 (out b 0 0) sg13_lv_nmos w=1u l=0.13u\n"
                    "xp1 (out a mid vdd) sg13_lv_pmos w=2u l=0.13u\n"
                    "xp2 (mid b vdd vdd) sg13_lv_pmos w=2u l=0.13u\n"
                )
            run(f"{logic}_{a}_{b}", ("cornerMOSlv.lib", "mos_tt"), statements, "analysis op1 op")
    # Inverter exercises meaningful currents and an unconstrained output voltage.
    for gate in [0.0, 0.4, 0.6, 0.8, 1.2]:
        run(
            f"inverter_vg{gate}",
            ("cornerMOSlv.lib", "mos_tt"),
            f"vg (gate 0) vs dc={gate}\n"
            f"vdd (vdd 0) vs dc=1.2\n"
            f"xn (out gate 0 0) sg13_lv_nmos w=1u l=0.13u\n"
            f"xp (out gate vdd vdd) sg13_lv_pmos w=2u l=0.13u\n",
            "analysis op1 op",
        )
    run(
        "inverter_dc_sweep",
        ("cornerMOSlv.lib", "mos_tt"),
        "vg (gate 0) vs dc=0\n"
        "vdd (vdd 0) vs dc=1.2\n"
        "xn (out gate 0 0) sg13_lv_nmos w=1u l=0.13u\n"
        "xp (out gate vdd vdd) sg13_lv_pmos w=2u l=0.13u\n",
        'sweep vin instance="vg" parameter="dc" from=0 to=1.2 mode="lin" points=13\nanalysis op1 op',
        mode="dc",
    )
    run(
        "inverter_tran",
        ("cornerMOSlv.lib", "mos_tt"),
        f'include "{root / "cornerCAP.lib"}" section=cap_typ\n'
        'vg (in 0) vs type="pulse" val0=0 val1=1.2 delay=1n rise=0.1n fall=0.1n width=2n period=4n\n'
        "vdd (vdd 0) vs dc=1.2\n"
        "xn (out in 0 0) sg13_lv_nmos w=1u l=0.13u\n"
        "xp (out in vdd vdd) sg13_lv_pmos w=2u l=0.13u\n"
        "xc (out 0) cap_cmim w=10u l=10u\n",
        "analysis tran1 tran stop=5n step=1p maxstep=1p",
        raw="tran1.raw",
        mode="tran",
    )
    for model in ["cap_cmim", "cap_rfcmim"]:
        statements = (
            "v1 (in 0) vs dc=0 mag=1\nr1 (in out) parity_r r=1k\nx1 (out 0"
            + (" 0" if model == "cap_rfcmim" else "")
            + f") {model} w=10u l=10u\n"
        )
        run(
            model + "_ac",
            ("cornerCAP.lib", "cap_typ"),
            statements,
            'analysis ac1 ac from=1e6 to=1e9 mode="dec" points=3',
            raw="ac1.raw",
            mode="ac",
        )
    statements = (
        'v1 (in 0) vs type="pulse" val0=0 val1=1 delay=1n rise=0.1n fall=0.1n width=10n period=20n\n'
        "r1 (in out) parity_r r=1k\n"
        "x1 (out 0) cap_cmim w=10u l=10u\n"
    )
    run(
        "cmim_tran",
        ("cornerCAP.lib", "cap_typ"),
        statements,
        "analysis tran1 tran stop=5n step=1p maxstep=1p",
        raw="tran1.raw",
        mode="tran",
    )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not all(result["passed"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
