#!/usr/bin/env bash
set -euo pipefail
: "${IHP_ROOT:?Set IHP_ROOT to the gdsfactory/IHP checkout}"
: "${OSDI_DIR:?Set OSDI_DIR to the six Circulax-compatible OSDI modules}"
exec "${PYTHON:-python}" -m benchmarks.ihp_parity.run \
  --pdk "$IHP_ROOT" \
  --module-path "$OSDI_DIR" \
  --osdi-module "$OSDI_DIR/psp103.osdi" \
  --osdi-module "$OSDI_DIR/psp103_nqs.osdi" \
  --osdi-module "$OSDI_DIR/r3_cmc.osdi" \
  --osdi-module "$OSDI_DIR/sp_resistor.osdi" \
  --osdi-module "$OSDI_DIR/sp_capacitor.osdi" \
  --osdi-module "$OSDI_DIR/sp_inductor.osdi" \
  --ngspice-osdi-module "$OSDI_DIR/psp103.osdi" \
  --ngspice-osdi-module "$OSDI_DIR/psp103_nqs.osdi" \
  --ngspice-osdi-module "$OSDI_DIR/r3_cmc.osdi" "$@"
