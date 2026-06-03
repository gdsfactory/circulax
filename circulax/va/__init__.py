"""Backward-compatibility shim — re-exports the VA stack from ``bosdi.va``.

The Verilog-A lowering / SCCP / emitter machinery lives in the ``bosdi``
package (installed via ``pip install circulax[verilog-a]``).  Existing
imports keep working through this shim:

  - ``from circulax.va import lower``                      ✓
  - ``from circulax.va.lowering import _BINOP_FOLDS``      ✓ (private symbols too)

For new code, import from ``bosdi.va`` directly.

Raises:
    ImportError: If ``bosdi`` is not installed. Install via
        ``pip install circulax[verilog-a]``.

"""

import sys

try:
    import bosdi.va
except ImportError as _err:
    raise ImportError(
        "circulax.va requires the 'bosdi' package. "
        "Install it with: pip install circulax[verilog-a]"
    ) from _err

# Re-export the public API at the top level (``from circulax.va import lower``).
# Submodule aliases so ``from circulax.va.lowering import _BINOP_FOLDS`` and
# similar deep imports keep working.
import bosdi.va.dump_parser
import bosdi.va.emitter
import bosdi.va.ir_client
import bosdi.va.lowering
import bosdi.va.mir
import bosdi.va.sccp
import bosdi.va.uniform_params
import bosdi.va.va_defaults
from bosdi.va import *  # noqa: F403
from bosdi.va import (  # noqa: F401  — explicit names so linters / IDEs see them
    AbstimeInput,
    Block,
    CachedValues,
    CallDecl,
    CompiledModule,
    Constant,
    CurrentKind,
    DaeInfo,
    DaeMatrixEntry,
    DaeResidual,
    DumpFile,
    DumpParseError,
    Function,
    HiddenStateInput,
    HirInterner,
    InputKind,
    Inst,
    LoweredDevice,
    LoweringError,
    ParamGivenRef,
    ParamRef,
    ParamSysFunInput,
    PhiEdge,
    PortConnectedInput,
    TemperatureInput,
    Value,
    Voltage,
    compile_va,
    compile_va_unopt,
    compile_va_unopt_with_split,
    emit_source,
    lower,
    parse_dump,
    uniform_static_params,
    write_source,
)

for _sub in ("dump_parser", "emitter", "ir_client", "lowering",
             "mir", "sccp", "uniform_params", "va_defaults"):
    sys.modules[f"circulax.va.{_sub}"] = sys.modules[f"bosdi.va.{_sub}"]
del _sub
