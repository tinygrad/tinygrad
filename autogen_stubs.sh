#!/bin/bash -e

# setup instructions for clang2py
if [[ ! $(clang2py -V) ]]; then
  pushd .
  cd /tmp
  sudo apt-get install -y --no-install-recommends clang
  pip install --upgrade pip setuptools
  pip install clang==14.0.6
  git clone https://github.com/nimlgen/ctypeslib.git
  cd ctypeslib
  pip install .
  clang2py -V
  popd
fi

BASE=tinygrad/runtime/autogen/

fixup() {
  sed -i '1s/^/# mypy: ignore-errors\n/' $1
  sed -i 's/ *$//' $1
  grep FIXME_STUB $1 || true
}

patch_dlopen() {
  path=$1; shift
  name=$1; shift
  cat <<EOF | sed -i "/import ctypes.*/r /dev/stdin" $path
PATHS_TO_TRY = [
$(for p in "$@"; do echo "  $p,"; done)
]
def _try_dlopen_$name():
  library = ctypes.util.find_library("$name")
  if library:
    try: return ctypes.CDLL(library)
    except OSError: pass
  for candidate in PATHS_TO_TRY:
    try: return ctypes.CDLL(candidate)
    except OSError: pass
  return None
EOF
}

generate_sqtt() {
  clang2py -k cdefstum \
    extra/sqtt/sqtt.h \
    -o $BASE/sqtt.py
  fixup $BASE/sqtt.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/sqtt.py
  python3 -c "import tinygrad.runtime.autogen.sqtt"

  ROCPROF_COMMIT_HASH=dd0485100971522cc4cd8ae136bdda431061a04d
  ROCPROF_SRC=/tmp/rocprof-trace-decoder-$ROCPROF_COMMIT_HASH
  if [ ! -d "$ROCPROF_SRC" ]; then
    git clone https://github.com/ROCm/rocprof-trace-decoder $ROCPROF_SRC
    pushd .
    cd $ROCPROF_SRC
    git reset --hard $ROCPROF_COMMIT_HASH
    popd
  fi

  clang2py -k cdefstum \
    $ROCPROF_SRC/include/rocprof_trace_decoder.h \
    $ROCPROF_SRC/include/trace_decoder_instrument.h \
    $ROCPROF_SRC/include/trace_decoder_types.h \
    -o $BASE/rocprof.py
  fixup $BASE/rocprof.py
  sed -i '1s/^/# pylint: skip-file\n/' $BASE/rocprof.py
  sed -i "s/import ctypes/import ctypes, ctypes.util/g" $BASE/rocprof.py
  patch_dlopen $BASE/rocprof.py rocprof-trace-decoder "'/usr/local/lib/librocprof-trace-decoder.so'" "'/usr/local/lib/librocprof-trace-decoder.dylib'"
  sed -i "s/def _try_dlopen_rocprof-trace-decoder():/def _try_dlopen_rocprof_trace_decoder():/g" $BASE/rocprof.py
  sed -i "s|FunctionFactoryStub()|_try_dlopen_rocprof_trace_decoder()|g" $BASE/rocprof.py
}

generate_mesa() {
  MESA_TAG="mesa-25.2.4"
  MESA_SRC=/tmp/mesa-$MESA_TAG
  TINYMESA_TAG=tinymesa-32dc66c
  TINYMESA_DIR=/tmp/tinymesa-$MESA_TAG-$TINYMESA_TAG/
  TINYMESA_SO=$TINYMESA_DIR/libtinymesa_cpu.so
  if [ ! -d "$MESA_SRC" ]; then
    git clone --depth 1 --branch $MESA_TAG https://gitlab.freedesktop.org/mesa/mesa.git $MESA_SRC
    pushd .
    cd $MESA_SRC
    git reset --hard $MESA_COMMIT_HASH
    # clang 14 doesn't support packed enums
    sed -i "s/enum \w\+ \(\w\+\);$/uint8_t \1;/" $MESA_SRC/src/nouveau/headers/nv_device_info.h
    sed -i "s/enum \w\+ \(\w\+\);$/uint8_t \1;/" $MESA_SRC/src/nouveau/compiler/nak.h
    sed -i "s/nir_instr_type \(\w\+\);/uint8_t \1;/" $MESA_SRC/src/compiler/nir/nir.h
    mkdir -p gen/util/format
    python3 src/util/format/u_format_table.py src/util/format/u_format.yaml --enums > gen/util/format/u_format_gen.h
    python3 src/compiler/nir/nir_opcodes_h.py > gen/nir_opcodes.h
    python3 src/compiler/nir/nir_intrinsics_h.py --outdir gen
    python3 src/compiler/nir/nir_intrinsics_indices_h.py --outdir gen
    python3 src/compiler/nir/nir_builder_opcodes_h.py > gen/nir_builder_opcodes.h
    python3 src/compiler/nir/nir_intrinsics_h.py --outdir gen
    python3 src/compiler/builtin_types_h.py gen/builtin_types.h
    popd
  fi

  if [ ! -d "$TINYMESA_DIR" ]; then
    mkdir $TINYMESA_DIR
    curl -L https://github.com/sirhcm/tinymesa/releases/download/$TINYMESA_TAG/libtinymesa_cpu-$MESA_TAG-linux-amd64.so -o $TINYMESA_SO
  fi

  clang2py -k cdefstu \
    $MESA_SRC/src/compiler/nir/nir.h \
    $MESA_SRC/src/compiler/nir/nir_builder.h \
    $MESA_SRC/src/compiler/nir/nir_shader_compiler_options.h \
    $MESA_SRC/src/compiler/nir/nir_serialize.h \
    $MESA_SRC/gen/nir_intrinsics.h \
    $MESA_SRC/src/nouveau/headers/nv_device_info.h \
    $MESA_SRC/src/nouveau/compiler/nak.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_passmgr.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_misc.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_type.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_init.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_nir.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_struct.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_jit_types.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_flow.h \
    $MESA_SRC/src/gallium/auxiliary/gallivm/lp_bld_const.h \
    $MESA_SRC/src/compiler/glsl_types.h \
    $MESA_SRC/src/util/blob.h \
    $MESA_SRC/src/util/ralloc.h \
    --clang-args="-DHAVE_ENDIAN_H -DHAVE_STRUCT_TIMESPEC -DHAVE_PTHREAD -I$MESA_SRC/src -I$MESA_SRC/include -I$MESA_SRC/gen -I$MESA_SRC/src/compiler/nir -I$MESA_SRC/src/gallium/auxiliary -I$MESA_SRC/src/gallium/include -I$(llvm-config-20 --includedir)" \
    -l $TINYMESA_SO \
    -o $BASE/mesa.py

  LVP_NIR_OPTIONS=$(./extra/mesa/lvp_nir_options.sh $MESA_SRC)

  fixup $BASE/mesa.py
  patch_dlopen $BASE/mesa.py tinymesa_cpu "(BASE:=os.getenv('MESA_PATH', f\"/usr{'/local/' if helpers.OSX else '/'}lib\"))+'/libtinymesa_cpu'+(EXT:='.dylib' if helpers.OSX else '.so')" "f'{BASE}/libtinymesa{EXT}'" "'/opt/homebrew/lib/libtinymesa_cpu.dylib'" "'/opt/homebrew/lib/libtinymesa.dylib'"
  echo "lvp_nir_options = gzip.decompress(base64.b64decode('$LVP_NIR_OPTIONS'))" >> $BASE/mesa.py
  sed -i "/in_dll/s/.*/try: &\nexcept (AttributeError, ValueError): pass/" $BASE/mesa.py
  sed -i "s/import ctypes/import ctypes, ctypes.util, os, gzip, base64, subprocess, tinygrad.helpers as helpers/" $BASE/mesa.py
  sed -i "s/ctypes.CDLL('.\+')/(dll := _try_dlopen_tinymesa_cpu())/" $BASE/mesa.py
  echo "def __getattr__(nm): raise AttributeError('LLVMpipe requires tinymesa_cpu' if 'tinymesa_cpu' not in dll._name else f'attribute {nm} not found') if dll else FileNotFoundError(f'libtinymesa not found (MESA_PATH={BASE}). See https://github.com/sirhcm/tinymesa ($TINYMESA_TAG, $MESA_TAG)')" >> $BASE/mesa.py
  sed -i "s/ctypes.glsl_base_type/glsl_base_type/" $BASE/mesa.py
  # bitfield bug in clang2py
  sed -i "s/('fp_fast_math', ctypes.c_bool, 9)/('fp_fast_math', ctypes.c_uint32, 9)/" $BASE/mesa.py
  sed -i "s/('\(\w\+\)', pipe_shader_type, 8)/('\1', ctypes.c_ubyte)/" $BASE/mesa.py
  sed -i "s/\([0-9]\+\)()/\1/" $BASE/mesa.py
  sed -i '/struct_nir_builder._pack_ = 1 # source:False/d' "$BASE/mesa.py"
  python3 -c "import tinygrad.runtime.autogen.mesa"
}

if [ "$1" == "sqtt" ]; then generate_sqtt
elif [ "$1" == "mesa" ]; then generate_mesa
elif [ "$1" == "all" ]; then generate_mesa
else echo "usage: $0 <type>"
fi
