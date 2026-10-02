"""GF180 card-level native VACASK/bosdi DC and AC comparisons.

This deliberately bypasses SPICE wrappers/bin selection. NetlistParse reads the
original cards and scopes; both simulators receive identical numeric parameters.
No PDK model is substituted. Stateful-model Circulax integration remains separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netlist_parser
import numpy as np
from osdi_jax import osdi_eval
from osdi_loader import load_osdi_model

from benchmarks.ihp_parity.run import compare
from benchmarks.utils.vacask_reference import run_vacask
from circulax.netlist_io import Library
from circulax.netlist_io.expressions import Scope, evaluate
from circulax.netlist_io.osdi import module_metadata
from circulax.netlist_io.syntax import children, parameters
from circulax.solvers.assembly import assemble_gc_real, assemble_residual_only_real

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 -- JAX configuration API


def cards(source: Path, corner: str) -> tuple[Scope, dict[str, Any]]:
    """Read corner scope and original binned cards using NetlistParse CSTs."""
    root = netlist_parser.parse_spice(source.read_text())
    if netlist_parser.errors(root):
        msg = "NetlistParse rejected original GF180 library"
        raise ValueError(msg)
    sections = {children(n, "Identifier")[0].text.lower(): n for n in children(root, "LibStatement")}
    scope = Scope()
    scope.dialect = "spice"
    settings = netlist_parser.parse_spice(source.with_name("settings.inc").read_text())
    for node in children(settings, "ParamStatement"):
        scope.bindings.update(parameters(node))
    models = {}

    def walk(section: str) -> None:
        for node in children(sections[section]):
            if node.kind == "ParamStatement":
                scope.bindings.update(parameters(node))
            elif node.kind == "Model":
                models[children(node, "HierarchialNode")[0].text] = node
            elif node.kind == "LibInclude":
                walk(children(node, "Identifier")[0].text.lower())

    walk(corner)
    return scope, models


def operating_point(model: Any, params: jax.Array, terminals: list[float]) -> tuple[np.ndarray, Any]:
    """Solve every internal equation with externally fixed terminal biases."""
    voltage = np.zeros((1, model.num_nodes))
    voltage[0, :4] = terminals
    parent = list(range(model.num_nodes))
    for first, second in model.collapsible_pairs:
        if max(first, second) >= len(parent):
            continue
        low, high = sorted([parent[first], parent[second]])
        parent = [low if node == high else node for node in parent]
    for node in range(4, model.num_nodes):
        if parent[node] < 4:
            voltage[0, node] = voltage[0, parent[node]]
    states = jnp.zeros((1, model.num_states), dtype=jnp.float64)
    for _ in range(80):
        result = osdi_eval(model.id, jnp.asarray(voltage), params, states)
        residual = np.asarray(result[0])[0, 4:]
        matrix = np.asarray(result[1])[0].reshape(model.num_nodes, -1)[4:, 4:]
        if np.max(np.abs(residual), initial=0) < 1e-12:
            return voltage, result
        voltage[0, 4:] -= np.linalg.lstsq(matrix, residual, rcond=1e-15)[0]
    msg = "Internal/equality equations did not converge"
    raise ValueError(msg)


def main() -> None:  # noqa: C901, PLR0912, PLR0915 -- explicit reference comparison matrix
    """Compare the same cards and OSDI binaries through both runtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdk", type=Path, required=True)
    parser.add_argument("--vacask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("gf180-parity.json"))
    parser.add_argument("--only", default="")
    parser.add_argument("--model-osdi", type=Path, help="Optional alternate BSIM4 implementation")
    args = parser.parse_args()
    source = args.pdk / "gf180mcu/models/ngspice/sm141064.ngspice"
    modules = args.vacask_root / "build/lib/vacask/mod"
    binary = (args.model_osdi or modules / "spice/bsim4v8.osdi").resolve()
    module_name, aliases = module_metadata(binary)
    dc_model = load_osdi_model(str(binary), analysis="dc")
    ac_model = load_osdi_model(str(binary), analysis="ac")
    names = {name.lower(): i for i, name in enumerate(dc_model.param_names)}
    names.update({alias: names[canonical.lower()] for alias, canonical in aliases.items() if canonical.lower() in names})
    report = {
        "scope": "card-level DC/AC, explicit bin; no full SPICE wrapper or transient claim",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "osdi_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "raw_nodes": dc_model.num_nodes,
        "state_slots": dc_model.num_states,
        "results": {},
    }
    for corner in ["typical", "ff", "ss", "fs", "sf"]:
        scope, models = cards(source, corner)
        for family in ["nmos_3p3", "pmos_3p3", "nmos_6p0", "pmos_6p0", "nmos_6p0_nat"]:
            geometries = [(0, 0.3e-6, 0.3e-6), (8, 5e-6, 0.3e-6)] if "3p3" in family else [(0, 5e-6, 2e-6)]
            for bin_number, width, length in geometries:
                for parasitics in [0, 1]:
                    name = f"{family}_{corner}_bin{bin_number}_parasitics{parasitics}"
                    if args.only and args.only not in name:
                        continue
                    try:
                        values = {k: evaluate(v, scope) for k, v in parameters(models[f"{family}.{bin_number}"]).items()}
                        if values.pop("level") != 54:
                            msg = "Expected a level-54 GF180 MOS card"
                            raise ValueError(msg)  # noqa: TRY301 -- record per-card failure
                        card_version = values.pop("version")
                        if card_version not in (4.5, 4.6):
                            msg = f"Unexpected GF180 version selector: {card_version}"
                            raise ValueError(msg)  # noqa: TRY301 -- record per-card failure
                        if "rgeomod" in values and "instance_rgeomod" in names and "rgeomod" not in names:
                            values["instance_rgeomod"] = values.pop("rgeomod")
                        values.update(
                            type=-1 if family.startswith("p") else 1,
                            w=width,
                            l=length,
                            nf=1,
                            rgatemod=parasitics,
                            rbodymod=parasitics,
                            rdsmod=parasitics,
                        )
                        missing = set(values) - names.keys()
                        if missing:
                            msg = f"OSDI parameters missing: {sorted(missing)}"
                            raise ValueError(msg)  # noqa: TRY301 -- record per-card failure
                        p = np.full((1, dc_model.num_params), np.nan)
                        p[0, names["$mfactor"]] = 1
                        for key, value in values.items():
                            p[0, names[key]] = value
                        p = jnp.asarray(p)
                        sign = values["type"]
                        vd, vg = sign * 1.2, sign * 1.0
                        voltage, dc = operating_point(dc_model, p, [vd, vg, 0, 0])
                        states = jnp.zeros((1, ac_model.num_states), dtype=jnp.float64)
                        ac = osdi_eval(ac_model.id, jnp.asarray(voltage), p, states)
                        g = np.asarray(dc[1])[0].reshape(dc_model.num_nodes, -1)
                        c = np.asarray(ac[3])[0].reshape(dc_model.num_nodes, -1)
                        frequencies = np.logspace(3, 9, 7)
                        currents = []
                        for frequency in frequencies:
                            matrix = g + 2j * np.pi * frequency * c
                            v = np.zeros(dc_model.num_nodes, dtype=complex)
                            v[1] = 1
                            v[4:] = np.linalg.lstsq(matrix[4:, 4:], -matrix[4:, :4] @ v[:4], rcond=1e-15)[0]
                            currents.append(-(matrix @ v)[0])
                        card = " ".join(f"{k}={v:.16e}" for k, v in values.items())
                        deck = (
                            f'GF180 {name}\nground 0\nload "{binary}"\n'
                            f"model dut {module_name} {card}\nmodel vs vsource\n"
                            f"vd (d 0) vs dc={vd}\nvg (g 0) vs dc={vg} mag=1\n"
                            "m (d g 0 0) dut\ncontrol\nabort always\n"
                            'options temp=26.85 reltol=1e-8 vntol=1e-10 abstol=1e-14 rawfile="binary"\n'
                            'analysis op1 op\nanalysis ac1 ac from=1k to=1G points=1 mode="dec"\nendc\n'
                        )
                        kwargs = {
                            "binary": args.vacask_root / "build/simulator/vacask",
                            "module_paths": (modules,),
                            "shared_library_paths": (args.vacask_root / ".pixi/envs/default/lib",),
                        }
                        reference_dc = run_vacask(deck, raw_filename="op1.raw", **kwargs)
                        reference_ac = run_vacask(deck, raw_filename="ac1.raw", **kwargs)
                        np.testing.assert_allclose(reference_ac.vectors["frequency"].real, frequencies)
                        report["results"][name] = {
                            "pdk_version_selector": card_version,
                            "implementation": module_name,
                            "dc": compare(-np.asarray(dc[0])[0, 0], reference_dc.vectors["vd:flow(br)"], atol=1e-12),
                            "ac": compare(currents, reference_ac.vectors["vd:flow(br)"], atol=1e-12),
                        }
                        try:
                            with tempfile.TemporaryDirectory() as temporary:
                                path = Path(temporary) / "card.lib"
                                path.write_text(deck.split("\n", 1)[1].split("control\n", 1)[0])
                                resolved = Library.from_file(path, temperature_c=26.85).resolve()
                                circuit = resolved.compile(module_paths=(modules,), state_policy="limiting_only")
                                solution = circuit.dc()
                                group = next(g for g in circuit.groups.values() if "device0" in (g.index_map or {}))
                                actual = solution[..., group.var_indices[group.index_map["device0"], -1]]
                                report["results"][name]["circulax_dc"] = compare(
                                    actual, reference_dc.vectors["vd:flow(br)"], atol=1e-12
                                )
                                gv, _ = assemble_gc_real(solution, circuit.groups)
                                ac_circuit = circuit._for_analysis("ac")  # noqa: SLF001 -- inspect the public analysis registration
                                _, cv = assemble_gc_real(solution, ac_circuit.groups)
                                rows = np.concatenate(
                                    [np.asarray(g.jac_rows).reshape(-1) for _, g in sorted(circuit.groups.items())]
                                )
                                cols = np.concatenate(
                                    [np.asarray(g.jac_cols).reshape(-1) for _, g in sorted(circuit.groups.items())]
                                )
                                cg = np.zeros((circuit.sys_size, circuit.sys_size))
                                cc = np.zeros_like(cg)
                                np.add.at(cg, (rows, cols), np.asarray(gv))
                                np.add.at(cc, (rows, cols), np.asarray(cv))

                                def forcing(amplitude: jax.Array) -> jax.Array:
                                    groups = circuit._with_param_values({"device1.V": amplitude})  # noqa: SLF001, B023 -- consumed immediately inside this comparison
                                    return assemble_residual_only_real(solution, groups, 0.0, 0.0)[0]  # noqa: B023 -- immediate jacfwd

                                rhs = np.asarray(-jax.jacfwd(forcing)(jnp.asarray(0.0))).copy()
                                rhs[0] = 0
                                source_index = group.var_indices[group.index_map["device0"], -1]
                                responses = []
                                for frequency in frequencies:
                                    matrix = cg + 2j * np.pi * frequency * cc
                                    matrix[0, :] = 0
                                    matrix[0, 0] = 1
                                    responses.append(np.linalg.solve(matrix, rhs)[source_index])
                                report["results"][name]["circulax_ac"] = compare(
                                    responses, reference_ac.vectors["vd:flow(br)"], atol=1e-12
                                )

                        except Exception as error:  # noqa: BLE001 -- report and continue independent comparisons
                            report["results"][name]["circulax_error"] = str(error)
                    except Exception as error:  # noqa: BLE001 -- report and continue independent comparisons
                        report["results"][name] = {"error": str(error)}
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(name, report["results"][name], flush=True)  # noqa: T201 -- CLI progress


if __name__ == "__main__":
    main()
