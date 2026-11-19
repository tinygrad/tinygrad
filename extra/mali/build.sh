#!/usr/bin/env bash
set -euo pipefail

MESA_DIR="${1:-$HOME/mesa-tinymali}"
MALI_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$MESA_DIR/.git" ]; then
  git clone --depth 1 --branch mesa-25.2.7 https://gitlab.freedesktop.org/mesa/mesa.git "$MESA_DIR"
fi

git -C "$MESA_DIR" apply --check "$MALI_DIR/mesa.patch" 2>/dev/null && \
  git -C "$MESA_DIR" apply "$MALI_DIR/mesa.patch" || true
mkdir -p "$MESA_DIR/src/tinymali"
cp "$MALI_DIR/tinymali.c" "$MALI_DIR/tinymali.h" "$MALI_DIR/meson.build" "$MESA_DIR/src/tinymali/"

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

ninja -C "$MESA_DIR/build" -j"${JOBS:-4}" src/tinymali/libtinymali.so

cat <<EOF
export MESA_PATH="$MESA_DIR/build/src/tinymali/libtinymali.so"
export TINYMALI_PATH="$MESA_DIR/build/src/tinymali/libtinymali.so"
EOF
