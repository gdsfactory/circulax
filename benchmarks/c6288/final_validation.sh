#!/usr/bin/env bash
# Run after repeat_integration.py exits, so validation and diagnostics do not
# compete with the measured simulators. Invoke from the repository root.
set -euo pipefail
out=benchmarks/c6288
# Pin the optional provisioning tests to the same verified ABI 0.4 resistor.
.venv/bin/python - <<'PYFIXTURE'
import json
from pathlib import Path
baseline=json.loads(Path('benchmarks/c6288/reduced-1.json').read_text())
resistor=Path(baseline['modules'][1]['path'])
target=Path('/tmp/circulax-abi04-modules/spice/resistor.osdi')
target.parent.mkdir(parents=True,exist_ok=True)
if not target.exists(): target.symlink_to(resistor)
assert target.resolve()==resistor.resolve()
PYFIXTURE
export VACASK_MODULE_PATH=/tmp/circulax-abi04-modules
export CIRCULAX_TEST_OSDI_RESISTOR=/tmp/circulax-abi04-modules/spice/resistor.osdi
.venv/bin/python -m pytest \
  tests/test_compiler.py tests/test_netlist.py tests/test_kfnetlist.py \
  tests/netlist_io tests/test_prepare_analysis.py tests/test_osdi_sp_hb.py \
  tests/test_transient.py tests/test_sensitivity.py tests/test_adjoint.py \
  tests/test_c6288_benchmark.py tests/test_osdi_topology.py \
  tests/test_sp_wavelength.py tests/test_ac_response.py tests/test_ac_sweep.py \
  tests/test_dc_checked.py -q -rs > "$out/integration-tests-complete.log" 2>&1
PYTHONPATH=/tmp/circulax-bosdi-released .venv/bin/python -m pytest \
  tests/netlist_io/test_osdi_analysis.py tests/test_osdi_sp_hb.py -q -rs \
  > "$out/released-bosdi-tests.log" 2>&1
PYTHONPATH=/tmp/circulax-bosdi-released .venv/bin/python benchmarks/c6288/run.py \
  --vacask-root /home/cdaunt/code/vacask/VACASK --prepare-only \
  --output "$out/released-bosdi-prepare.json" > "$out/released-bosdi-prepare-driver.log" 2>&1
for representation in full reduced; do
  JAX_LOG_COMPILES=1 .venv/bin/python -m benchmarks.c6288.profile_stamps \
    --baseline "$out/reduced-1.json" \
    --vacask-root /home/cdaunt/code/vacask/VACASK \
    --representation "$representation" --output "$out/profile-$representation.json" \
    > "$out/profile-$representation.log" 2>&1
done
.venv/bin/python benchmarks/c6288/summarize_integration.py > "$out/summary-driver.log"
MPLCONFIGDIR=/tmp/circulax-integration-mpl .venv/bin/python benchmarks/c6288/plot_integration.py
