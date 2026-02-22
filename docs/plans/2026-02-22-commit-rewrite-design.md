# Commit Rewrite Plan: CDNA4 Emulator PR

## Overview

Rewrite 12 messy debugging commits into 16 clean, bug-by-bug commits with tests.
Each commit is self-contained, testable, and tells one story.

## PR Scope

**Files in PR** (11 files, ~2400 lines added):

| File | Changes |
|------|---------|
| `test/mockgpu/amd/emu.py` | Core emulator: MUBUF, MFMA 32x32, OOB clamping, M0 fix, branch overflow, etc. |
| `test/mockgpu/amd/pcode.py` | _extract_bits upcast, int64 vmem indices |
| `test/mockgpu/amd/amdgpu.py` | LDS 160KB, scratch 1024B, comments |
| `test/mockgpu/helpers.py` | valid_mem_ranges passthrough |
| `tinygrad/runtime/autogen/amd/cdna/ins.py` | SMEM_F61 subclass (3 lines) |
| `tinygrad/renderer/amd/__init__.py` | SMEM_F61 in decoder format list |
| `test/amd/test_emu_cdna_bugs.py` | 31 regression tests (new file) |
| `.github/workflows/test.yml` | Uncomment cdna4 arch + timeout 40min |

**Files NOT in PR**: CLAUDE.md, scripts/test_cdna4_fa.sh, TODOS.md, test_gemm_dispatch.py

**Strip from code**: All debug instrumentation (WG timing prints, MFMA C source dumps, lds_size logging).

## Commit Sequence

### Commit 1: `cdna4: add CDNA4 emulator support from geohot's #14815 PR`

The PR #14815 base. Wave64, ACCVGPR, MFMA 16x16, CDNA pcode, DSL changes.

**Files**: emu.py (+594), pcode.py (+146), amdgpu.py (+10), dsl.py (+14), generate.py (+2), ins.py x3 (+2 each)

This is the existing commit `5c5f4183b` content, unchanged.

---

### Commit 2: `cdna4: fix scratch alignment to 1024B for CDNA (was 256B RDNA)`

CDNA uses 1024-byte scratch alignment (vs RDNA 256B). Emulator had 4x too small scratch, causing OOB writes corrupting instruction bytes.

**Files**: amdgpu.py (1 line — already in commit 1's amdgpu.py changes, need to separate)

**Note**: This fix is already part of commit 1's amdgpu.py changes. During rebase, extract this as a separate commit.

---

### Commit 3: `cdna4: fix LDS size to 160KB matching MI350X specification`

MI350X has 160KB LDS per CU (64 banks x 2.5KB). PR #14815 set 128KB. GEMM kernel needs 130KB+.

**Files**: amdgpu.py (1 line: `lds_size_in_kb 128` -> `160`)

---

### Commit 4: `cdna4: add GFX950 SMEM format 61 decoder support`

GFX950 comgr emits SMEM at format 61 (0b111101) instead of ISA spec's format 48 (0b110000). 3-line subclass.

**Files**: cdna/ins.py (+4), renderer/amd/__init__.py (+3/-2)
**Tests**: `test_decode_0xf4080500`, `test_s_load_dwordx2_format61`

---

### Commit 5: `cdna4: add MUBUF handler to emulator dispatch table`

GEMM uses buffer_load/store_dwordx4 (MUBUF format). Dispatch table had no entry. The existing `_compile_mem_op` already handles MUBUF address math — just needed wiring.

**Files**: emu.py (add `irc.MUBUF: _compile_mem_op` to dispatch + MUBUF field extraction in `_compile_mem_op` + MUBUF make_addr + MUBUF make_srcs + MUBUF make_stores + buffer_load_lds handler)
**Tests**: `test_buffer_load_dwordx4_basic`, `test_buffer_load_dwordx4_per_lane_offsets`, `test_buffer_store_dwordx4_basic`, `test_buffer_store_dwordx4_advancing_srd`, `test_buffer_load_store_copy`, `test_buffer_load_store_advancing_srd_loop`, `test_buffer_load_store_64_lanes_advancing_srd`, `test_buffer_load_dwordx4_lds_basic`, `test_buffer_load_dwordx4_lds_per_lane`

---

### Commit 6: `cdna4: fix _extract_bits upcast before shift/mask operations`

Pcode `_extract_bits` applied shift/mask to original dtype which could be narrower than the extraction range. Cast to uint32/uint64 first.

**Files**: pcode.py (+2/-1)

---

### Commit 7: `cdna4: use int64 vmem indices for 48-bit GPU address support`

MUBUF addresses are 48-bit. The emulator used `dtypes.int` (32-bit) for vmem index casts, truncating high bits. Changed to `dtypes.int64`.

**Files**: emu.py (~6 lines: `.cast(dtypes.int)` -> `.cast(dtypes.int64)` in vmem paths), pcode.py (+4/-3)

---

### Commit 8: `cdna4: fix ACC bit handling on CDNA memory operations`

CDNA scratch/global/flat/MUBUF memory instructions have an `acc` bit. When `acc=1`, vdata targets ACCVGPRs instead of VGPRs. Needed for ACCVGPR spill/reload in GEMM.

**Files**: emu.py (~10 lines: `use_acc = getattr(inst, 'acc', 0)`, conditional raccvgpr_dyn/waccvgpr_dyn)

---

### Commit 9: `cdna4: fix M0/NULL register encoding swap between CDNA and RDNA`

CDNA: M0=124, NULL=125. RDNA: M0=125, NULL=124. `wsgpr_dyn` was discarding writes to 124 (thinking it was NULL on CDNA). `buffer_load_lds` was reading M0 from 125 on CDNA.

**Files**: emu.py (3 lines: arch-aware null_idx in wsgpr_dyn, M0 index in buffer_load_lds)
**Tests**: `test_buffer_load_lds_m0_offset`

---

### Commit 10: `cdna4: add direct DS_WRITE/READ_B128/B96 handler`

CDNA pcode for DS_WRITE_B128/B96 uses `DATA[127:96]` on 32-bit DATA register — undefined behavior. Generate stores directly instead of relying on pcode.

**Files**: emu.py (~40 lines: two new `if isinstance(inst, irc.DS)` blocks)
**Tests**: `test_ds_write_b128_high_vgpr_gemm_regs`, `test_ds_write_b128_high_vgpr_4_waves_barrier`, `test_ds_write_b128_high_vgpr_4_waves_wave_offset`, `test_ds_write_b128_garbage_addr_inactive_lanes`

---

### Commit 11: `cdna4: fix chained VOPC assignment in CDNA comparison pcode`

CDNA CMPX pcode uses `EXEC[laneId] = D0[laneId] = expr` chained assignment. Strip `D0[laneId] = ` prefix since EXEC/D0 writes are handled separately.

**Files**: emu.py (~4 lines: regex strip in `_compile_vopc`)

---

### Commit 12: `cdna4: add MFMA 32x32 and I8 datatype support`

Extend MFMA from 16x16-only to 32x32. Add I8 integer datatype support. Different register layouts:
- 16x16: lane->col, VGPR->row_block
- 32x32: lane->row, VGPR->col_block

**Files**: emu.py (~80 lines: generalized _compile_mfma)

---

### Commit 13: `cdna4: fix branch offset overflow for kernels >32KB`

SOPP branch pcode uses `SIMM16.i16 * 16'4` which overflows int16 for branch offsets > 32KB (GEMM kernel is 81KB). Compute branch targets directly in int64.

**Files**: emu.py (~37 lines: new branch handling in `_compile_sopp`)
**Tests**: `test_s_branch_over_32kb`, `test_s_cbranch_scc1_over_32kb`

---

### Commit 14: `cdna4: add OOB address clamping with trash page for MUBUF stores`

Pure-Python OOB protection: mmap anonymous RW trash page, redirect OOB/inactive-lane vmem addresses there. Three layers: make_addr, buffer_load_lds, wmem. Canonical 47-bit mask for x86_64.

**Files**: emu.py (~80 lines: trash page, canonical mask, clamping in make_addr/buffer_load_lds/wmem), helpers.py (+2: valid_mem_ranges passthrough)

---

### Commit 15: `cdna4: add remaining regression tests`

Any regression tests not already included with their bug fix commits. Collects `test_s_add_u32_*` (s_add carry chain tests) and any other standalone tests.

**Files**: test/amd/test_emu_cdna_bugs.py

---

### Commit 16: `cdna4: enable cdna4 in CI arch matrix`

Uncomment cdna4 in the MOCKGPU test matrix. Increase timeout to 40 minutes (emulator is slower than hardware).

**Files**: .github/workflows/test.yml (2 lines)

---

## Dependency Graph

```
1 (base) -> 2 (scratch) -> 3 (LDS) -> 4 (SMEM F61)
                                         |
                                         v
5 (MUBUF handler) -> 6 (_extract_bits) -> 7 (int64 vmem)
                                            |
                                            v
8 (ACC bit) -> 9 (M0/NULL) -> 10 (DS_B128) -> 11 (VOPC)
                                                  |
                                                  v
                           12 (MFMA 32x32) -> 13 (branch overflow) -> 14 (OOB clamping)
                                                                        |
                                                                        v
                                                              15 (tests) -> 16 (CI)
```

## Execution Strategy

This will be done via `git rebase -i` on a fresh branch. For each commit:
1. Cherry-pick the relevant hunks from the current working tree
2. Include the associated test(s)
3. Verify tests pass at that commit
4. Move to next

The key challenge is that our current commits are organized by *debugging session* (what we found when), not by *logical change*. The rebase rewrites history to tell the story as if we fixed bugs one at a time, in dependency order.
