# Copyright (c) 2026 Circulax contributors
# ruff: noqa: E402, S603
"""Refresh deterministic F/Q/G/C oracles using the pinned VA and OpenVAF."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

_DATA = Path(__file__).resolve().parent
_scratch = tempfile.TemporaryDirectory()
_target = Path(_scratch.name) / "r3.osdi"
subprocess.run([shutil.which("openvaf-r"), str(_DATA / "r3_cmc.va"), "-o", str(_target)], check=True, capture_output=True)
import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003
import jax.numpy as jnp
from bosdi.circulax.osdi_component import osdi_component
from osdi_jax import osdi_eval

from circulax.components.r3 import R3_CMC

cases = [
    ("default", {}, 300.15),
    ("contacts", {"rc": 10.0, "c1": 2, "c2": 2, "xl": 0.2}, 300.15),
    (
        "junction_thermal",
        {
            "ca": 1e-15,
            "cja": 2e-15,
            "a1": 1e-12,
            "a2": 2e-12,
            "cth0": 1e-6,
            "gth0": 0.01,
            "tc1": 0.002,
            "isa": 1e-14,
            "isp": 1e-15,
            "p1": 2e-6,
            "p2": 3e-6,
        },
        340.0,
    ),
    ("linear", {"sw_lin": 1, "sw_et": 0}, 300.15),
    (
        "nonlinear_thermal",
        {
            "type": 1,
            "gth0": 0.01,
            "tegth": 0.5,
            "cth0": 1e-6,
            "sw_accpo": 2,
            "sw_vsatt": 1,
            "ats": 0.2,
            "xvsat": 0.001,
            "tc1": 0.002,
        },
        300.15,
    ),
    (
        "mismatch",
        {
            "sw_mman": 1,
            "sw_mmgeo": 1,
            "sw_dfgeo": 0,
            "nsig_rsh": 0.2,
            "sig_rsh": 1.0,
            "nsig_l": 0.1,
            "sig_l": 0.01,
            "nsig_w": 0.2,
            "sig_w": 0.01,
        },
        280.0,
    ),
]
out = []
for name, settings, temp in cases:
    r = osdi_component(str(_target), ports=R3_CMC.ports, temperature=temp)
    inst = r.make_instance(settings)
    p = jnp.array([[inst[k] for k in r.param_names]])
    for bias in [[0.3, -0.2, 0.0, 2.0, 0.28, 0.03, 0.002, -0.002], [-0.2, -0.6, 0.3, 5.0, -0.17, 0.25, -0.003, 0.003]]:
        v = jnp.array([bias])
        f, g, q, c, _ = osdi_eval(r.model.id, v, p, jnp.empty((1, 0)))
        data = {
            "name": name,
            "settings": settings,
            "temperature": temp,
            "bias": bias,
            "f": f[0].tolist(),
            "q": q[0].tolist(),
            "g": g.reshape(8, 8).tolist(),
            "c": c.reshape(8, 8).tolist(),
        }
        if name == "nonlinear_thermal":
            cols = []
            for k in range(8):
                d = jnp.eye(8)[k][None, :] * 1e-5
                fp = osdi_eval(r.model.id, v + d, p, jnp.empty((1, 0)))[0][0]
                fm = osdi_eval(r.model.id, v - d, p, jnp.empty((1, 0)))[0][0]
                cols.append((fp - fm) / (2e-5))
            data["g_fd"] = jnp.stack(cols, axis=1).tolist()
        out.append(data)
(_DATA / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
