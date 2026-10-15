"""Compare IHP native/Circulax models against ngspice or converted VACASK libraries.

Run with python benchmarks/ihp_parity/run.py --help. Results record tolerances,
absolute error and vector scales; unsupported runtime features are explicit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.utils.vacask_reference import run_ngspice, run_vacask
from circulax.netlist_io import Library
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


def vacask_testbench(text: str, *, model_root: Path) -> str:
    """Render this harness's native SPICE templates as a VACASK root deck.

    PDK files use gdsfactory/IHP's preconverted VACASK model tree.
    This adapter handles only the fixed benchmark testbench vocabulary.
    """
    lines = []
    for line in text.splitlines():
        tokens = shlex.split(line)
        if not tokens:
            continue
        if tokens[0].lower() == ".lib":
            lines.append(f'include "{model_root / Path(tokens[1]).name}" section={tokens[2]}')
        elif tokens[0][0].lower() == "x":
            master = next(i for i, token in enumerate(tokens[1:], 1) if "=" in token) - 1
            lines.append(f"{tokens[0]} ({' '.join(tokens[1:master])}) {' '.join(tokens[master:])}")
        elif tokens[0][0].lower() == "r":
            lines.append(f"{tokens[0]} ({tokens[1]} {tokens[2]}) parity_r r={tokens[3]}")
        elif tokens[0][0].lower() == "v":
            settings = tokens[3:]
            if settings[0].lower().startswith("pulse("):
                values = " ".join(settings)[6:-1].split()
                fields = ("val0", "val1", "delay", "rise", "fall", "width", "period")
                properties = 'type="pulse" ' + " ".join(f"{k}={v}" for k, v in zip(fields, values, strict=True))
            else:
                properties = f"dc={settings[1]}"
                if len(settings) > 2:
                    properties += f" mag={settings[3]}"
            lines.append(f"{tokens[0]} ({tokens[1]} {tokens[2]}) vs {properties}")
        else:
            msg = f"unsupported benchmark template: {line}"
            raise ValueError(msg)
    return "\n".join(lines) + "\n"


def main() -> None:  # noqa: C901, PLR0912, PLR0915 -- explicit comparison matrix
    """Run the selected numerical comparisons and save their measured errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdk", type=Path, help="PDK checkout; defaults to the installed ihp-gdsfactory package")
    parser.add_argument("--simulator", choices=("ngspice", "vacask"), default="ngspice")
    parser.add_argument("--vacask", type=Path, default=Path("vacask"))
    parser.add_argument("--ngspice", type=Path, default=Path("ngspice"))
    parser.add_argument("--osdi-module", type=Path, action="append", default=[], help="Circulax ABI 0.4 OSDI module or VA source")
    parser.add_argument("--vacask-models", type=Path, help="Converted gdsfactory/IHP VACASK models directory")
    parser.add_argument("--vacask-compiler", default=None, help="Compiler used by the VACASK reference")
    parser.add_argument("--ngspice-osdi-module", type=Path, action="append", default=[], help="ngspice ABI OSDI binary")
    parser.add_argument("--module-path", type=Path, action="append", default=[])
    parser.add_argument("--shared-library-path", type=Path, action="append", default=[])
    parser.add_argument("--compiler", default="openvaf-r")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--only", default="", help="Substring selecting comparisons")
    parser.add_argument("--output", type=Path, default=Path("ihp-parity.json"))
    args = parser.parse_args()
    if args.pdk is None:
        package = importlib.util.find_spec("ihp")
        if package is None or package.origin is None:
            parser.error("supply --pdk or install the ihp-parity Pixi environment")
        args.pdk = Path(package.origin).parent.parent
    root = args.pdk.resolve()
    vacask_root = args.vacask_models.resolve() if args.vacask_models else None
    if (root / "ihp/models/ngspice/models").is_dir():
        vacask_root = vacask_root or root / "ihp/models/vacask/models"
        root /= "ihp/models/ngspice/models"
    if (root / "ihp-sg13g2").is_dir():
        root /= "ihp-sg13g2"
    if (root / "libs.tech/ngspice/models").is_dir():
        root /= "libs.tech/ngspice/models"
    if not (root / "cornerMOSlv.lib").is_file():
        parser.error("--pdk must name gdsfactory/IHP, IHP-Open-PDK, or a native ngspice models directory")
    if args.simulator == "vacask" and (vacask_root is None or not (vacask_root / "cornerMOSlv.lib").is_file()):
        parser.error("VACASK requires gdsfactory/IHP converted libraries; supply --pdk or --vacask-models")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    def run_case(  # noqa: C901, PLR0915 -- reference/analysis dispatch
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
        header = f'.lib "{root / library[0]}" {library[1]}\n'
        circuit_text = header + statements
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "circuit.lib"
            path.write_text(circuit_text)
            resolved = Library.from_file(path, temperature_c=temperature_c, statistical_mode="nominal").resolve()
            circuit = resolved.compile(
                module_paths=tuple(args.module_path),
                osdi_modules=tuple(args.osdi_module),
                compiler=args.compiler,
                cache_dir=args.cache_dir,
                state_policy="limiting_only",
                simparams={"gmin": 1e-12} if args.simulator == "vacask" else None,
            )
            tolerance_options = "reltol=1e-6 vntol=1e-8 abstol=1e-12" if mode == "tran" else "reltol=1e-8 vntol=1e-10 abstol=1e-14"
            if args.simulator == "vacask":
                deck = (
                    "IHP parity: "
                    + name
                    + "\nground 0\nmodel vs vsource\nmodel parity_r sp_resistor\n"
                    + vacask_testbench(circuit_text, model_root=vacask_root)
                    + f'\ncontrol\nabort always\noptions temp={temperature_c} rawfile="binary" {tolerance_options}\n'
                    + analysis
                    + "\nendc\n"
                )
                reference = run_vacask(
                    deck,
                    binary=args.vacask,
                    module_paths=tuple(args.module_path),
                    shared_library_paths=tuple(args.shared_library_path),
                    compiler=args.vacask_compiler,
                    raw_filename=raw,
                    va_sources=tuple(
                        vacask_root.parent.parent / "ngspice/va" / source
                        for source in ("psp103/psp103.va", "psp103/psp103_nqs.va", "r3_cmc/r3_cmc.va", "mosvar/mosvar.va")
                    ),
                    cache_dir=args.cache_dir,
                )
            else:
                commands = {
                    "op": "op",
                    "dc": "dc vg 0 1.2 0.1",
                    "ac": "ac dec 3 1meg 1g",
                    "tran": "tran 1p 5n 0 1p",
                }
                deck = (
                    "IHP parity: "
                    + name
                    + "\n"
                    + circuit_text
                    + f"\n.options temp={temperature_c} {tolerance_options}\n.control\n"
                    + "set filetype=binary\nsetseed 1\n"
                    + commands[mode]
                    + f"\nwrite {raw} all\nquit\n.endc\n.end\n"
                )
                reference = run_ngspice(deck, binary=args.ngspice, osdi_modules=tuple(args.ngspice_osdi_module), raw_filename=raw)

            def read(y: jax.Array) -> jax.Array:
                if ":flow(br)" not in measure:
                    return circuit.port(y, measure)
                source = measure.split(":", maxsplit=1)[0]
                key = f"device{next(i for i, leaf in enumerate(resolved.instances) if leaf.name == source)}"
                group = next(group for group in circuit.groups.values() if key in (group.index_map or {}))
                return y[..., group.var_indices[group.index_map[key], -1]]

            if mode == "op":
                actual = read(circuit.dc(rtol=1e-10, atol=1e-12))
            elif mode == "dc":
                actual = read(circuit.dc(rtol=1e-10, atol=1e-12, params={"device0.V": jnp.asarray(reference.vectors["vin"].real)}))
            elif mode == "ac":
                dc = circuit.dc(rtol=1e-10, atol=1e-12)
                # VACASK retains OP conductance; ngspice evaluates both G and
                # C in small-signal mode. NQS idt equations distinguish these.
                gv, _ = assemble_gc_real(dc, circuit.groups)
                circuit = circuit._for_analysis("ac")  # noqa: SLF001 -- inspect native AC registration
                ac_gv, cv = assemble_gc_real(dc, circuit.groups)
                if args.simulator == "ngspice":
                    gv = ac_gv
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
                dc = circuit.dc(rtol=1e-10, atol=1e-12)
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
            results[name]["simulator"] = args.simulator
            args.output.write_text(json.dumps(results, indent=2) + "\n")

    def run(name: str, *arguments: Any, **options: Any) -> None:
        try:
            run_case(name, *arguments, **options)
        except (RuntimeError, ValueError, NotImplementedError, FileNotFoundError) as error:
            results[name] = {"passed": False, "simulator": args.simulator, "error": str(error)}
            args.output.write_text(json.dumps(results, indent=2) + "\n")

    for corner in ["mos_tt", "mos_ss", "mos_ff", "mos_sf", "mos_fs"]:
        for nf in [1, 2, 3]:
            name = f"lv_nmos_dc_{corner}_nf{nf}"
            run(
                name,
                ("cornerMOSlv.lib", corner),
                f"vg gate 0 DC 0.7\nvd drain 0 DC 0.8\nx1 drain gate 0 0 sg13_lv_nmos w=2u l=0.13u ng={nf} m=2\n",
                "analysis op1 op",
                measure="vd:flow(br)",
            )
    for corner in ["mos_tt", "mos_ss", "mos_ff", "mos_sf", "mos_fs"]:
        run(
            "rf_nmos_" + corner,
            ("cornerMOSlv.lib", corner),
            "vg gate 0 DC 0.7\nvd drain 0 DC 0.8\nx1 drain gate 0 0 sg13_lv_nmos w=2u l=0.13u ng=2 rfmode=1\n",
            "analysis op1 op",
            measure="vd:flow(br)",
        )
        run(
            "hv_nmos_" + corner,
            ("cornerMOShv.lib", corner),
            "vg gate 0 DC 2\nvd drain 0 DC 2\nx1 drain gate 0 0 sg13_hv_nmos w=2u l=0.45u ng=2\n",
            "analysis op1 op",
            measure="vd:flow(br)",
        )
    for corner in ["res_typ", "res_bcs", "res_wcs"]:
        for model in ["rsil", "rhigh", "rppd"]:
            run(
                model + "_" + corner,
                ("cornerRES.lib", corner),
                f"v1 in 0 DC 0.2\nx1 in 0 0 {model} w=1u l=5u\n",
                "analysis op1 op",
                measure="v1:flow(br)",
            )
    for temperature in [-40.0, 27.0, 125.0]:
        run(
            f"temperature_nmos_{temperature}",
            ("cornerMOSlv.lib", "mos_tt"),
            "vg gate 0 DC 0.7\nvd drain 0 DC 0.8\nx1 drain gate 0 0 sg13_lv_nmos w=2u l=0.13u ng=2\n",
            "analysis op1 op",
            measure="vd:flow(br)",
            temperature_c=temperature,
        )
        run(
            f"temperature_resistor_{temperature}",
            ("cornerRES.lib", "res_typ"),
            "v1 in 0 DC 0.2\nx1 in 0 0 rppd w=1u l=5u\n",
            "analysis op1 op",
            measure="v1:flow(br)",
            temperature_c=temperature,
        )
    for rfmode in [0, 1]:
        run(
            f"nmos_rfmode{rfmode}_ac",
            ("cornerMOSlv.lib", "mos_tt"),
            "vg gate 0 DC 0.6 AC 1\nvdd vdd 0 DC 1.2\nr1 vdd out 1k\n"
            f"x1 out gate 0 0 sg13_lv_nmos w=1u l=0.13u ng=2 rfmode={rfmode}\n",
            'analysis ac1 ac from=1e6 to=1e9 mode="dec" points=3',
            raw="ac1.raw",
            mode="ac",
        )
    for logic in ["nand", "nor"]:
        for a, b in [(0, 0), (0, 1.2), (1.2, 0), (1.2, 1.2)]:
            statements = f"va a 0 DC {a}\nvb b 0 DC {b}\nvdd vdd 0 DC 1.2\n"
            if logic == "nand":
                statements += (
                    "xn1 out a mid 0 sg13_lv_nmos w=1u l=0.13u\n"
                    "xn2 mid b 0 0 sg13_lv_nmos w=1u l=0.13u\n"
                    "xp1 out a vdd vdd sg13_lv_pmos w=2u l=0.13u\n"
                    "xp2 out b vdd vdd sg13_lv_pmos w=2u l=0.13u\n"
                )
            else:
                statements += (
                    "xn1 out a 0 0 sg13_lv_nmos w=1u l=0.13u\n"
                    "xn2 out b 0 0 sg13_lv_nmos w=1u l=0.13u\n"
                    "xp1 out a mid vdd sg13_lv_pmos w=2u l=0.13u\n"
                    "xp2 mid b vdd vdd sg13_lv_pmos w=2u l=0.13u\n"
                )
            run(f"{logic}_{a}_{b}", ("cornerMOSlv.lib", "mos_tt"), statements, "analysis op1 op")
    # Inverter exercises meaningful currents and an unconstrained output voltage.
    for gate in [0.0, 0.4, 0.6, 0.8, 1.2]:
        run(
            f"inverter_vg{gate}",
            ("cornerMOSlv.lib", "mos_tt"),
            f"vg gate 0 DC {gate}\n"
            f"vdd vdd 0 DC 1.2\n"
            f"xn out gate 0 0 sg13_lv_nmos w=1u l=0.13u\n"
            f"xp out gate vdd vdd sg13_lv_pmos w=2u l=0.13u\n",
            "analysis op1 op",
        )
    run(
        "inverter_dc_sweep",
        ("cornerMOSlv.lib", "mos_tt"),
        "vg gate 0 DC 0\n"
        "vdd vdd 0 DC 1.2\n"
        "xn out gate 0 0 sg13_lv_nmos w=1u l=0.13u\n"
        "xp out gate vdd vdd sg13_lv_pmos w=2u l=0.13u\n",
        'sweep vin instance="vg" parameter="dc" from=0 to=1.2 mode="lin" points=13\nanalysis op1 op',
        mode="dc",
    )
    run(
        "inverter_tran",
        ("cornerMOSlv.lib", "mos_tt"),
        f'.lib "{root / "cornerCAP.lib"}" cap_typ\n'
        "vg in 0 PULSE(0 1.2 1n 0.1n 0.1n 2n 4n)\n"
        "vdd vdd 0 DC 1.2\n"
        "xn out in 0 0 sg13_lv_nmos w=1u l=0.13u\n"
        "xp out in vdd vdd sg13_lv_pmos w=2u l=0.13u\n"
        "xc out 0 cap_cmim w=10u l=10u\n",
        "analysis tran1 tran stop=5n step=1p maxstep=1p",
        raw="tran1.raw",
        mode="tran",
    )
    for model in ["cap_cmim", "cap_rfcmim"]:
        statements = (
            "v1 in 0 DC 0 AC 1\nr1 in out 1k\nx1 out 0" + (" 0" if model == "cap_rfcmim" else "") + f" {model} w=10u l=10u\n"
        )
        run(
            model + "_ac",
            ("cornerCAP.lib", "cap_typ"),
            statements,
            'analysis ac1 ac from=1e6 to=1e9 mode="dec" points=3',
            raw="ac1.raw",
            mode="ac",
        )
    statements = "v1 in 0 PULSE(0 1 1n 0.1n 0.1n 10n 20n)\nr1 in out 1k\nx1 out 0 cap_cmim w=10u l=10u\n"
    run(
        "cmim_tran",
        ("cornerCAP.lib", "cap_typ"),
        statements,
        "analysis tran1 tran stop=5n step=1p maxstep=1p",
        raw="tran1.raw",
        mode="tran",
    )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not results:
        parser.error("--only matched no comparisons")
    if not all(result["passed"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
