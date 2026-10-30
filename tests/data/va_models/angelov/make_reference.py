"""Generate independent Angelov OSDI fixtures from the retained VA source."""

# ruff: noqa: ANN201, CPY001, FBT003, S603
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from bosdi.circulax import osdi_component
from osdi_jax import osdi_eval

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
RAW_KEEP = list(range(14)) + list(range(16, 21))
DYNAMIC = {
    "Selft": 1,
    "Idsmod": 3,
    "Cgs0": 2e-12,
    "Cgd0": 1e-12,
    "Cds": 0.2e-12,
    "Cgdpe": 0.1e-12,
    "Ld": 1e-9,
    "Ls": 2e-9,
    "Lg": 0.5e-9,
    "Tau": 1e-10,
    "Crf": 1e-12,
    "Crfin": 1e-12,
    "Rth": 100.0,
}
SETTINGS = [
    {
        "Idsmod": i,
        "Capmod": c,
        "Igmod": i % 2,
        "Cgspi": 0.2e-12,
        "Cgdpi": 0.1e-12,
        "Cgs0": 2e-12,
        "Cgd0": 1e-12,
        "P2": 0.03,
        "P3": 0.005,
        "P111": 0.05,
        "Rd2": 0.02,
        "Lvg": 0.01,
    }
    for i in range(4)
    for c in range(3)
] + [{}, DYNAMIC, {"Temp": 85.0, "Tnom": 25.0, "Ne": 1.2}, {"Temp": 85.0, "Tnom": 25.0, "Ne": 1.2, "Pg": 18.0}]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "angelov.osdi"
        subprocess.run([shutil.which("openvaf-r"), str(HERE / "angelov.va"), "-o", str(binary)], check=True)
        d = osdi_component(str(binary), ("d", "g", "s"), temperature=300.15)
        result = []
        for settings in SETTINGS:
            # Raw OSDI order: d,g,s,di,gi,si,sii,gdi,gsi,bi,rf,t,xt1,xt2,ia,ib, five branch currents.
            bias = jnp.array(
                [
                    [
                        2.0,
                        -0.3,
                        0.0,
                        1.99,
                        -0.31,
                        0.01,
                        0.005,
                        -0.315,
                        -0.32,
                        0.015,
                        0.02,
                        10.0,
                        0.04,
                        0.04,
                        0.0,
                        0.0,
                        0.04,
                        -1e-8,
                        0.04,
                        0.04,
                        -0.04,
                    ],
                    [
                        0.3,
                        -1.0,
                        0.0,
                        0.29,
                        -1.01,
                        0.01,
                        0.005,
                        -1.015,
                        -1.02,
                        0.015,
                        0.02,
                        5.0,
                        0.002,
                        0.002,
                        0.0,
                        0.0,
                        0.002,
                        -1e-8,
                        0.002,
                        0.002,
                        -0.002,
                    ],
                    [
                        -0.4,
                        0.15,
                        0.0,
                        -0.39,
                        0.14,
                        0.01,
                        0.005,
                        0.135,
                        0.13,
                        0.015,
                        0.02,
                        2.0,
                        -0.01,
                        -0.01,
                        0.0,
                        0.0,
                        -0.01,
                        -1e-8,
                        -0.01,
                        -0.01,
                        0.01,
                    ],
                ]
            )
            p = jnp.array([[{"Noise": 0.0, **settings}.get(k, float("nan")) for k in d.param_names]] * len(bias))
            f, g, q, c, _ = osdi_eval(d.model.id, bias, p, jnp.zeros((len(bias), d.model.num_states)))
            result.append(
                {
                    "settings": settings,
                    "bias": bias[:, RAW_KEEP].tolist(),
                    "raw_bias": bias.tolist(),
                    "f": f[:, RAW_KEEP].tolist(),
                    "q": q[:, RAW_KEEP].tolist(),
                    "g": g.reshape((-1, 21, 21))[:, RAW_KEEP, :][:, :, RAW_KEEP].tolist(),
                    "c": c.reshape((-1, 21, 21))[:, RAW_KEEP, :][:, :, RAW_KEEP].tolist(),
                }
            )
        (HERE / "reference.json").write_text(
            json.dumps(
                {
                    "upstream": "6099b4e12e6f66ff530f59b8d01fb909094be85a",
                    "compiler": "OpenVAF-reloaded20260616-4-g5ed9e63a",
                    "cases": result,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
