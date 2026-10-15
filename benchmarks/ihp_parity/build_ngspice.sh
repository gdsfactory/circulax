#!/usr/bin/env bash
# The Conda ngspice package lacks OSDI. Build the same pinned release used locally.
set -euo pipefail
revision=724dc77b9153dc75eaec474b1f7a44f1fcf5f362
root="${NGSPICE_BUILD_DIR:-.pixi/ihp-parity/toolchain/ngspice}"
mkdir -p "$root"
root="$(cd "$root" && pwd)"
if [[ -x "$root/install/bin/ngspice" && -f "$root/revision" && "$(cat "$root/revision")" == "$revision" ]]; then
  exit 0
fi
if [[ ! -d "$root/source/.git" ]]; then
  git clone --depth=1 --branch ngspice-45.2 https://github.com/imr/ngspice.git "$root/source"
fi
if [[ "$(git -C "$root/source" rev-parse HEAD)" != "$revision" ]]; then
  printf 'Unexpected ngspice source revision; refusing to build\n' >&2
  exit 1
fi
cd "$root/source"
./autogen.sh > "$root/autogen.log" 2>&1
mkdir -p "$root/build"
cd "$root/build"
"$root/source/configure" --prefix="$root/install" --disable-debug \
  --with-x=no --with-readline=no --enable-osdi CFLAGS=-O2 > "$root/configure.log" 2>&1
make -j "${NGSPICE_BUILD_JOBS:-2}" > "$root/build.log" 2>&1
make install > "$root/install.log" 2>&1
printf '%s\n' "$revision" > "$root/revision"
"$root/install/bin/ngspice" --version
