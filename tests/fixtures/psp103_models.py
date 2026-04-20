"""PSP103 NMOS/PMOS parameter fixtures for PSP103 OSDI tests.

Merges Verilog-A compile-time defaults (extracted from PSP103_module.include)
with the model-card overrides from VACASK's ring-osc models.inc, producing
complete 783-parameter dicts suitable for passing as ``default_params`` to
:func:`circulax.osdi_component` (canonical mode).

Exposes:
    PSP103_OSDI:       path to the compiled psp103v4_psp103.osdi binary.
    PSP103_MODELS_INC: path to VACASK's ring-osc models.inc (card overrides).
    PSP103_VA_SOURCE:  path to PSP103_module.include (VA-source defaults).
    PSP103N_DEFAULTS:  dict of {canonical_name -> float} for NMOS (793 keys).
    PSP103P_DEFAULTS:  dict of {canonical_name -> float} for PMOS (793 keys).
    make_psp103_descriptors(): builds (psp103n, psp103p) OsdiModelDescriptors.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

PSP103_OSDI = str(
    Path(__file__).resolve().parents[2]
    / "circulax"
    / "components"
    / "osdi"
    / "compiled"
    / "psp103v4_psp103.osdi"
)

PSP103_MODELS_INC = Path(
    "/home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask/models.inc"
)

PSP103_VA_SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "circulax"
    / "components"
    / "osdi"
    / "psp103v4"
)
PSP103_VA_SOURCE = PSP103_VA_SOURCE_DIR / "PSP103_module.include"


_PARAM_DEF_RE = re.compile(r"`(?:MPR|MPI|IPR|IPI)\w*\(\s*(\w+)\s*,\s*([^,\)]+)")
_MODEL_BLOCK_RE = re.compile(
    r"model\s+(\S+)\s+psp103va\s*\((?P<body>[^)]*)\)", re.DOTALL
)
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+\-]?[\d.eE+\-]+)")


def _to_float(raw: str) -> float:
    s = raw.strip().replace("inf", "1e30").replace("-inf", "-1e30")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_va_defaults(path: Path, seen: set[str] | None = None) -> dict[str, float]:
    """Walk PSP103 VA includes and harvest every MPR/MPI/IPR/IPI default.

    Keys are uppercased so the output is directly lookup-able by canonical
    OSDI parameter names (PSP103's canonical names are uppercase).
    """
    if seen is None:
        seen = set()
    result: dict[str, float] = {}
    for line in path.read_text().splitlines():
        include = re.match(r'\s*`include\s+"([^"]+)"', line)
        if include:
            child = path.parent / include.group(1)
            for k, v in _extract_va_defaults(child, seen).items():
                result.setdefault(k, v)
            continue
        match = _PARAM_DEF_RE.search(line)
        if match:
            key = match.group(1).upper()
            if key not in seen:
                seen.add(key)
                result[key] = _to_float(match.group(2))
    return result


def _parse_models_inc() -> tuple[dict[str, float], dict[str, float]]:
    text = PSP103_MODELS_INC.read_text()
    blocks = {
        match.group(1): match.group("body")
        for match in _MODEL_BLOCK_RE.finditer(text)
    }
    n = {k.upper(): _to_float(v) for k, v in _KV_RE.findall(blocks["psp103n"])}
    p = {k.upper(): _to_float(v) for k, v in _KV_RE.findall(blocks["psp103p"])}
    return n, p


def _read_canonical_names() -> list[str]:
    """Canonical OSDI parameter names (indexed order) from the bosdi loader."""
    from osdi_loader import load_osdi_model

    model = load_osdi_model(PSP103_OSDI)
    return list(model.param_names)


def _build_defaults() -> tuple[dict[str, float], dict[str, float]]:
    canonical = _read_canonical_names()
    va = _extract_va_defaults(PSP103_VA_SOURCE)
    n_over, p_over = _parse_models_inc()

    n_out: dict[str, float] = {}
    p_out: dict[str, float] = {}
    for name in canonical:
        if not name:
            continue
        key = name.upper()
        base = 1.0 if name == "$mfactor" else va.get(key, 0.0)
        n_out[name] = n_over.get(key, base)
        p_out[name] = p_over.get(key, base)
    return n_out, p_out


PSP103N_DEFAULTS, PSP103P_DEFAULTS = _build_defaults()


@cache
def make_psp103_descriptors():
    """Build canonical-mode PSP103 NMOS + PMOS descriptors."""
    from circulax.components.osdi import osdi_component

    psp103n = osdi_component(
        osdi_path=PSP103_OSDI,
        ports=("D", "G", "S", "B"),
        default_params=PSP103N_DEFAULTS,
    )
    psp103p = osdi_component(
        osdi_path=PSP103_OSDI,
        ports=("D", "G", "S", "B"),
        default_params=PSP103P_DEFAULTS,
    )
    return psp103n, psp103p


def geom_settings(
    w: float, length: float, ld: float = 0.5e-6, ls: float = 0.5e-6
) -> dict[str, float]:
    """Per-instance geometry settings matching VACASK's nmos/pmos subckt."""
    return {
        "W": w,
        "L": length,
        "AD": w * ld,
        "AS": w * ls,
        "PD": 2.0 * (w + ld),
        "PS": 2.0 * (w + ls),
    }
