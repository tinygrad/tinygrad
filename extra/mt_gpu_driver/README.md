# Moore Threads header sources

The headers in this directory are vendored inputs for tinygrad's autogen jobs. They are not hand-written replacements for the hardware ABI. The generated Python files live in `tinygrad/runtime/autogen/` and are checked by `.github/workflows/autogen.yml`.

## Moore Threads MUSA 3.3.0 driver headers

The reference package is `musa_3.3.0-server_amd64.deb`, from `MT_Linux_Driver_3.3.0` in the MUSA SDK 4.3.0 CC2.1 distribution. Extract it without running the installer:

```bash
dpkg-deb -x musa_3.3.0-server_amd64.deb /tmp/mt-driver
src=/tmp/mt-driver/usr/src/mtgpu-3.0.0/inc
```

Vendored inputs:

| Destination | Source under `$src` |
| --- | --- |
| `fwif/mtfw_fwif*.h` | `shared_include/fwif/mtfw_fwif*.h` |
| `mtgpu_defs.h` | `mtgpu/mtgpu_defs.h` |
| `mtgpu_segment.h` | `mtgpu/mtgpu_segment.h` |
| `vgpu_heap.h` | `mtgpu/vgpu/vgpu_heap.h` |
| `pvr/hwdefs/km/rgx_cr_defs_km.h` | `pvr/hwdefs/km/rgx_cr_defs_km.h` |
| `pvr/include/img_defs.h` | `pvr/include/img_defs.h` |
| `pvr/include/img_types.h` | `pvr/include/img_types.h` |
| `pvr/include/linux-types.h` | `pvr/include/linux-types.h` |
| `pvr/include/rgx_heap_firmware.h` | `pvr/include/rgx_heap_firmware.h` |

`fwif/` is copied as a complete directory so FWIF include dependencies remain intact. `RGX_FW_HEAP_SHIFT=25` is supplied when regenerating `mtheap.py`, selecting the documented 32 MiB default firmware heap.

The firmware image itself is not a header and is not vendored. `mtfw-gen1.bin` is installed separately from this driver package.

## Rogue MMU definitions

`pvr/include/pvr_rogue_mmu_defs.h` comes from the Linux Imagination driver:

```text
drivers/gpu/drm/imagination/pvr_rogue_mmu_defs.h
```

The reference source used for this checkout was Linux 7.4-rc3. It is an upstream, machine-generated hardware definition header, not a tinygrad-authored layout.

## Mesa 25.2.7 definitions

`pds/pvr_rogue_pds_defs.h` is copied directly from:

```text
src/imagination/vulkan/pds/pvr_rogue_pds_defs.h
```

`csbgen/pvr_packet_helpers.h` is copied directly from:

```text
src/imagination/csbgen/pvr_packet_helpers.h
```

`csbgen/rogue_cdm.h` is a **generated** Mesa header. Build it from `rogue_cdm.xml` with Mesa's own generator rather than editing it:

```bash
python3 src/imagination/csbgen/gen_pack_header.py \
  src/imagination/csbgen/rogue_cdm.xml > rogue_cdm.h
```

The equivalent Meson target is `src/imagination/csbgen/rogue_cdm.h`, generated when the `imagination-experimental` Vulkan driver is enabled. The vendored file is the exact output of that target.

## Arch210 compatibility headers

`cuda_bf16.h` and `cuda_fp16.h` are tinygrad-authored compatibility shims for the MTT S80's arch210 target. They provide float-backed math and conversions where MUSA exposes the corresponding CUDA operations only for arch220 or later. They are not vendor headers and are not used for other architectures.

## Regenerating Python bindings

From the tinygrad repository root, regenerate every MT binding and require a clean diff:

```bash
REGEN=1 python3 -c "from tinygrad.runtime.autogen import mtfw, mtregs, mtpds, mtcdm, mtheap, mtdev, mtmmu"
git diff --exit-code -- tinygrad/runtime/autogen/
```

The CI autogen job deletes generated files, runs the same imports, and fails if the results differ. Changes under `extra/mt_gpu_driver/` trigger that job.

## Values not present in public headers

Some generation-1 launch/context values are only available from MUSA allocator and launch code or binary disassembly; they do not have public struct definitions. Those values remain explicitly identified as `MT_GEN1_*` protocol data in `ops_mt.py`. They are not represented as generated header constants. The public FWIF, register, heap, MMU, CDM, and PDS definitions used by the driver come from the files above.
