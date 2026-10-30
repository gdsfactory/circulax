"""Compile the pinned VA source and write independent F/Q/G/C fixtures."""

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
SETTINGS = [
    {},
    {"RSCOM": 2.0, "TT": 1e-8},
    {"RSCOM": 1.0, "SWJUNEXP": 1.0},
    {"RSCOM": 1.0, "CORECOVERY": 1.0, "DEPNQS": 1e-8},
    {"RSCOM": 1.0, "CORECOVERY": 1.0, "DEPNQS": 1e-8, "AB": 1e-8, "LS": 1e-4, "LG": 1e-5, "DTA": 60.0, "NJH": 2.0, "INJ2": 12.0},
    {"CORECOVERY": 1.0, "NQS": 0.0, "DEPNQS": 0.0, "TT": 1e-8},
    {"TYPE": -1.0, "RSCOM": 1.0, "CORECOVERY": 1.0},
]


def main():
    with tempfile.TemporaryDirectory() as temp:
        binary = Path(temp) / "diode.osdi"
        subprocess.run([shutil.which("openvaf-r"), str(HERE / "diode_cmc.va"), "-o", str(binary)], check=True)
        descriptor = osdi_component(str(binary), ("A", "K"), temperature=300.15)
        result = []
        for settings in SETTINGS:
            bias = jnp.array(
                [[-1.0, 0.0, 0.0, 1e-10, 2e-10, 1e-6], [0.3, 0.0, 0.01, 2e-10, 3e-10, 2e-6], [0.9, 0.0, 0.1, 5e-10, 6e-10, 3e-6]]
            )
            if settings.get("TYPE", 1.0) == -1.0:
                bias = bias.at[:, :3].multiply(-1.0)
            series = settings.get("RSCOM", 0.0) > 0
            carrier = settings.get("CORECOVERY", 0.0) > 0 and settings.get("NQS", 5e-9) > 0
            depletion = settings.get("CORECOVERY", 0.0) > 0 and settings.get("DEPNQS", 0.0) > 0
            active = [0, 1] + ([2] if series else []) + ([3, 4] if carrier else []) + ([5] if depletion else [])
            projection = jnp.zeros((6, len(active)))
            for index, raw in enumerate(active):
                projection = projection.at[raw, index].set(1.0)
            if not series:
                projection = projection.at[2, 1].set(1.0)
            bias = bias[:, active] @ projection.T
            params = jnp.array([[settings.get(name, float("nan")) for name in descriptor.param_names]] * len(bias))
            f, g, q, c, _ = osdi_eval(descriptor.model.id, bias, params, jnp.zeros((len(bias), descriptor.model.num_states)))
            result.append(
                {
                    "settings": settings,
                    "bias": bias.tolist(),
                    "projection": projection.tolist(),
                    "f": f.tolist(),
                    "q": q.tolist(),
                    "g": g.reshape((-1, 6, 6)).tolist(),
                    "c": c.reshape((-1, 6, 6)).tolist(),
                }
            )
        (HERE / "reference.json").write_text(
            json.dumps({"upstream": "6099b4e12e6f66ff530f59b8d01fb909094be85a", "cases": result}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
