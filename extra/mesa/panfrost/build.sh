#!/usr/bin/env bash
set -euo pipefail

MESA_DIR="${1:-$HOME/mesa-tinymesa}"
PANFROST_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$MESA_DIR/.git" ]; then
  git clone --depth 1 --branch mesa-25.2.7 https://gitlab.freedesktop.org/mesa/mesa.git "$MESA_DIR"
fi

git -C "$MESA_DIR" apply --check "$PANFROST_DIR/mesa.patch" 2>/dev/null && \
  git -C "$MESA_DIR" apply "$PANFROST_DIR/mesa.patch" || true
mkdir -p "$MESA_DIR/src/tinymesa"
cp "$PANFROST_DIR/meson.build" "$MESA_DIR/src/tinymesa/"

if [ -d "$MESA_DIR/build" ]; then
  meson setup --reconfigure "$MESA_DIR/build" -Dc_args=-UNDEBUG
else
  meson setup "$MESA_DIR/build" \
    -Dgallium-drivers=panfrost -Dvulkan-drivers= -Dplatforms= \
    -Dglx=disabled -Degl=disabled -Dgbm=disabled \
    -Dopengl=false -Dgles1=disabled -Dgles2=disabled \
    -Dllvm=enabled -Dvalgrind=disabled -Dlibunwind=disabled \
    -Dlmsensors=disabled -Dbuildtype=release -Dtools= -Dc_args=-UNDEBUG
fi

ninja -C "$MESA_DIR/build" -j"${JOBS:-4}" src/tinymesa/libtinymesa.so

cat <<EOF
export MESA_PATH="$MESA_DIR/build/src/tinymesa/libtinymesa.so"
EOF
