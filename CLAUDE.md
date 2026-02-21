# CDNA4 Emulator Work

## What This Is

Adding CDNA4 (gfx950 / MI350X) support to tinygrad's MOCKGPU Python emulator.
The emulator interprets GPU assembly instructions in Python -- no actual AMD GPU needed.

## Current Status (2026-02-21)

### Our Work

**Core emulator: DONE.** 648 tests passing, 0 failures. 26 regression tests in `test/amd/test_emu_cdna_bugs.py`.

**Branch `cdna4-fa`** in `./` (this repo): 10 commits on top of `origin/master` (0255a64a2, Feb 21). Rebased clean.

**GEMM status**: All emulator-side bugs fixed. `test_tiny` (M=N=256, K=64) runs without crashing on our 8-core VPS but times out due to slow `c.realize()` scheduling step. On qazalin's machine (cdna_emu_2 branch), same test completes in ~12s. Root cause: our rebased-on-master code has a scheduling regression — the `realize()` step hangs for 300+ seconds before the GEMM kernel even starts executing. The emulator execution itself is fast (<1s per kernel).

### Upstream

**PR #14815** ("CDNA emulator, try 2") by geohot:
- State: **DRAFT**, all 49 CI checks passing, no human review comments
- Diff: +656/-121 across 10 files (emu.py, pcode.py, amdgpu.py, dsl.py, generate.py, 3x ins.py, test.yml, test_cfg_viz.py)
- The `cdna_emu_2` branch is 109 commits behind master (stale, needs rebase)
- geohot wants someone to take it over: "test it on the flash attention and CDNA gemm"

**PR #14914** ("amd/cdna4: cleanup") by hurrrsh (Harsh N.):
- State: **OPEN**, picks up DSL cleanup from PR #14815
- Adds unit tests for dsl.py, cleans up dsl.py code
- Says he'll start GEMM/FA tests next
- hurrrsh is a tinygrad contributor (10 PRs since 2023, smaller fixes/refactors), NOT a PyTorch contributor

**Master is moving fast**: scheduler rework (#14909), buffer preallocation (#14823), mockam driver (#14889), SQTT CDNA4 prep (#14900-#14904), isel renderer (#14730). Any branch needs frequent rebasing.

### Coordination Plan for PR Submission

When we publish our PR (FA/GEMM emulator tests passing):
- **Message to hurrrsh on PR #14914**: We were working on FA/GEMM testing in parallel. He did the DSL cleanup, we did the FA/GEMM emulator work. Propose splitting: he takes the cleanup PR, we extend with FA/GEMM test results. He can take the bounty credit for cleanup portion. Frame it as collaborative: "we wanted to get our feet wet on tinygrad and go deeper."
- **Our PR scope**: FA + GEMM tests passing under `MOCKGPU=1 MOCKGPU_ARCH=cdna4`, building on PR #14815's emulator changes.
- **Tone**: Polite, collaborative. Acknowledge his cleanup work. Offer to coordinate so PRs don't conflict.
- **Key point**: We're not competing for the bounty — we're learning the codebase and contributing complementary work.

## Next Phase: GEMM + Flash Attention + DSL Cleanup

### Bounty

geohot (2026-02-18, #general):
> "i'll pay out the CDNA bounty if someone adds GEMM/flash attention tests and cleans up this code"
> "main thing to clean up is dsl.py"

### Who's Working on What

**mtthw13 (Michael)**: Claimed the broader bounty ("all tests passing in CI with MOCKGPU_ARCH=cdna4") in #general on Feb 19. kuba (team) told him the emulator CI passing part is done, but the bounty isn't green because he still needs to finish a PR. As of Feb 19: 183 failed, 3839 passed. Filed PR #14890 (MFMA/GEMM/FA tests) targeting `cdna_emu_2` -- closed same day (fell behind master). Also filed PR #14889 (mockam, merged). WIP, no clean PR yet.

**kzhu**: Working on emulator size reduction (Feb 16): "emu has ballooned in size, i want to cut the code size first before writing for cdna4"

**No one** has responded to geohot's "someone take this over" on PR #14815.

### Work Items

1. **Flash attention under emulator** -- `test/testextra/test_tk.py` with `MOCKGPU=1 MOCKGPU_ARCH=cdna4`
2. **GEMM under emulator** -- `test/backend/test_asm_gemm.py` same way
3. **DSL cleanup** -- `tinygrad/renderer/amd/dsl.py`

Items 1-2 are CPU-only (no GPU needed). Item 3 is code cleanup.

### Flash Attention Status

- Compile-only (`DEV=NULL EMULATE=AMD_CDNA4`) **already works** (PRs #14742, #14770 merged)
- FA was broken by #14763, fixed by wozeparrot (#14767)
- Full emulator execution (`MOCKGPU=1 MOCKGPU_ARCH=cdna4`) untested -- this is the work
- Assembly-level FA (#14725, qazalin) was closed with MMU fault -- not merged
- FA gives wrong answers on HIP vs AMD driver -- qazalin (Feb 12)
- FA NaN with JIT enabled -- reported in mlperf-llama-405b channel

### GEMM Status

- `DEBUG=3 ... test_asm_gemm.py TestGemmLarge.test_simple` doesn't work with EMULATE yet -- qazalin
- GEMM assembly: `extra/gemm/asm/cdna/asm.py` (11,501 lines), recently parameterized (#14813) and late-compiled (#14783)
- CI expects `SCALE=128` by default. geohot: "don't do `SCALE = 128 if CI else 1` ... always the CI version"
- GEMM tests crash MI350X machines (hardware stability, not emulator)

## DSL Cleanup

File: `tinygrad/renderer/amd/dsl.py` (453 lines, only 2 commits on master)

### geohot's Requirements (chronological, from Discord)

- **Jan 11**: "dsl needs to be a lot cleaner" before moving into core tinygrad
- **Jan 16**: Target ~2000 lines total for decode/dsl/pdf/autogen combined. Deleted IMAGE/BUFFER instruction support ("no reason for compute"). Needs literal/dpp encoding fixes.
- **Jan 27**: Kernel class "not ready yet" for dsl.py. `waitcnt` must be in amdxml, not dsl. Must be generic across RDNA4/CDNA.
- **Feb 6**: "anyone opposed to moving it to the main repo? it's 4043 lines"
- **Feb 14**: "the sqtt and dsl stuff is decent quality" (can go into tinygrad proper). "emu and pcode don't meet the bar for tinygrad code" (moved to test/mockgpu/amd).
- **Feb 18**: "main thing to clean up is dsl.py"

### qazalin's Context

- "the assembly dsl is still pretty unusable for the cdna gemm kernel" (Jan 2026)
- Refactored fast CDNA GEMM to DSL, assembles bytes directly
- Hopes DSL in tinygrad proper enables "inlining significantly better assembly"

### Current DSL Structure

- `Reg` class -- 0-511 src encoding with neg/abs/hi modifiers
- `BitField` + variants (`FixedBitField`, `EnumBitField`) -- instruction bit manipulation
- `SrcField`, `VGPRField`, `SGPRField`, `SSrcField` -- typed register fields
- `AlignedSGPRField`, `SBaseField`, `SRsrcField` -- aligned SGPR fields
- `Inst` base class -- auto-upgrade to variants (_LIT, _DPP16, _SDWA), operand validation

### Cleanup Targets

- Simplify `op_bits` special cases (WAVE32 adjustments, addr sizing, MFMA formats)
- Clean up variant auto-upgrade logic in `Inst.__new__`
- Reduce field subclass proliferation
- Support all MFMA sizes (4x4, 16x16, 32x32) -- currently hardcoded `assert M == 16 and N == 16`
- Make `waitcnt` generic across RDNA4/CDNA
- PR #14815 only changes +8 lines in dsl.py (minor tweaks, not a rewrite)
- Related: PR #14730 (geohot, OPEN) adds new isel renderer -- may affect DSL direction

### Code Quality Bar

- "the goal is never ever low line count, it's low complexity and most importantly readability"
- "hacks like that are never ever worth it ... costs hours and hours of time later"
- "if the code isn't tasteful or looks like AI slop, i will close without review"
- Self-contained backends required, no reinventing UOps internally
- "AI cannot do them, anything with codex or claude code will be promptly closed"
- PRs must be minified, every line understood by submitter
- First PR to tinygrad cannot be a bounty claim

## Architecture

### Key Files

| File                                    | Lines  | Purpose                                      |
|-----------------------------------------|--------|----------------------------------------------|
| `test/mockgpu/amd/emu.py`              | 1,518  | Emulator: pcode -> UOps -> CPU execution     |
| `test/mockgpu/amd/pcode.py`            |        | Pcode parser: instruction semantics          |
| `test/mockgpu/amd/amdgpu.py`           |        | GPU plumbing: wave dispatch, LDS, registers  |
| `tinygrad/renderer/amd/dsl.py`         | 453    | Register abstraction, instruction encoding   |
| `tinygrad/runtime/autogen/amd/cdna/`   | 7,355  | Autogen ISA instruction definitions          |
| `extra/gemm/asm/cdna/asm.py`           | 11,501 | CDNA GEMM assembly kernel builder            |
| `extra/gemm/asm/cdna/gemm.py`          | 101    | GEMM high-level interface                    |
| `test/testextra/test_tk.py`            | 973    | TinyKittens: matmul + flash attention tests  |
| `test/backend/test_asm_gemm.py`        | 142    | Assembly GEMM correctness tests              |
| `.github/workflows/test.yml`           |        | CI: `arch: [rdna3, rdna4, cdna4]`           |

### CDNA4 vs RDNA3/RDNA4

| Feature        | RDNA3       | CDNA4 (gfx950)               |
|----------------|-------------|-------------------------------|
| Wave size      | 32          | **64**                        |
| LDS size       |             | **160KB** (1280-byte blocks)  |
| LDS banks      | 32          | **64**                        |
| Tensor core    | WMMA        | **MFMA** (4x4, 16x16, 32x32) |
| Accum regs     | VGPRs       | **ACCVGPRs** (separate file)  |
| FP8            | --          | **SDWA + MFMA**               |
| BF16 cast      | emulated    | **native** (`v_cvt_pk_bf16_f32`, CDNA4-only) |
| Scratch align  | 256 bytes   | **1024 bytes**                |

### What the Tests Exercise

**test_tk.py** (TinyKittens flash attention + matmul):
- `test_simple_matmul` -- 8192x8192 BF16 matmul using MFMA + LDS
- `test_simple_matmul_transposed` -- same with transposed B
- `test_fast_fa_bwd` -- flash attention backward pass
- Forward + backward, causal + non-causal variants
- Exercises: MFMA, LDS load/store, barriers, ACCVGPR management

**test_asm_gemm.py** (assembly GEMM):
- Standard shapes, K-sharded, validation, shape variety, llama3 shapes
- `TestMagicGu` -- magic number computation
- Tile constants: TILE_M=256, TILE_N=256, TILE_K=64, NUM_WG=256

### Upstream PRs (merged, relevant to CDNA4)

| PR      | Title                                                         | Merged     |
|---------|---------------------------------------------------------------|------------|
| #14938  | update test_jit_init_empty                                    | Feb 21     |
| #14925  | gemm/asm: smallest cdna4 asm gemm test (test_tiny)            | Feb 21     |
| #14909  | start ripping out old scheduler -- no maps                    | Feb 20     |
| #14904  | viz/sqtt: rdna4 wmma, cleanup inst rows                       | Feb 20     |
| #14900  | viz/sqtt: decoder fixes pre rdna4/cdna4 work                  | Feb 20     |
| #14889  | init mockam (PCI-based AM driver mock, orthogonal to emu)     | Feb 19     |
| #14857  | fa: explicitly pass shapes                                    | Feb 20     |
| #14823  | preallocate all realized buffers                              | Feb 20     |
| #14813  | parameterize the CDNA asm gemm                                | Feb 17     |
| #14791  | add sqtt support to the emulator                              | Feb 16     |
| #14786  | amd asm emulator fixes + run it in CI                         | Feb 16     |
| #14783  | late compile the cdna gemm                                    | Feb 16     |
| #14773  | Rdna4 emulator test_ops, dtypes pass                          | Feb 15     |
| #14770  | tinykittens flash attention dtype fix, add CI                  | Feb 15     |
| #14742  | make flash attention tests run on DEV=NULL EMULATE=AMD_CDNA4  | Feb 14     |
| #14721  | renderer/amd: add cdna emulator                               | Feb 13     |

### Open CDNA PRs/Issues

| #       | Title                                                          | State  |
|---------|----------------------------------------------------------------|--------|
| #14815  | CDNA emulator, try 2                                           | DRAFT  |
| #14730  | AMD isel renderer                                              | OPEN   |
| #14308  | CDNA SQTT timing attempt                                       | OPEN   |
| #13692  | Flash Attention fwd+bwd on MI300X + MI350X (memory)            | OPEN   |
| #13697  | Flash Attention MI350X speed on par with HipKittens            | OPEN   |
| #13696  | MI350X assembly output                                         | OPEN   |

## Test Commands

```bash
# Flash attention (compile-only, fast):
NULL_ALLOW_COPYOUT=1 PYTHONPATH=. DEV=NULL EMULATE=AMD_CDNA4 python test/testextra/test_tk.py

# Flash attention (full emulator):
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 PYTHONPATH=. python test/testextra/test_tk.py

# GEMM compile-only:
DEBUG=3 PYTHONPATH=. DEV=NULL EMULATE=AMD_CDNA4 python3 test/backend/test_asm_gemm.py TestGemmLarge.test_simple

# GEMM full emulator (slow):
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 PYTHONPATH=. python -m pytest test/backend/test_asm_gemm.py -x --tb=short

# Core test suite:
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=0 \
  python -m pytest -n=auto test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py --tb=line -q

# Full CI suite:
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=0 \
  python -m pytest -n=auto \
  test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py \
  test/backend/test_linearizer.py test/backend/test_randomness.py test/backend/test_jit.py \
  test/backend/test_graph.py test/backend/test_multitensor.py \
  test/device/test_hcq.py test/testextra/test_cfg_viz.py test/external/external_test_am.py
```

## Our 10 Commits (on `cdna4-fa`, rebased on master 0255a64a2)

1. `5c5f4183b` feat: apply PR #14815 CDNA4 emulator changes
2. `9819865f5` fix: set CDNA4 mock GPU LDS size to 160KB (matching real MI350X)
3. `c42ad9f47` docs: add CLAUDE.md and test script with CDNA4 findings
4. `3a00c2d2a` docs: document GEMM emulator bugs (unknown instruction, missing MUBUF handler)
5. `eb8b097cf` fix: upcast val to dt in _extract_bits before shift/mask ops
6. `865d56f62` feat: add CDNA4 MFMA 32x32, MUBUF, and VOPC chained-assignment support
7. `2e7b5f878` fix: use int64 vmem indices for 48-bit GPU address support
8. `920565a7a` feat: add CDNA4 emulator bug fixes and 23 regression tests
9. `9421eb845` fix: prevent SEGV from OOB MUBUF stores in persistent GEMM kernels
10. `0103326be` fix: protect JIT code pages from MUBUF store corruption + add SMEM_F61

## PR #14815 vs Our Implementation

| Aspect                                     | Our approach                                | PR #14815                                   |
|--------------------------------------------|---------------------------------------------|---------------------------------------------|
| ACCVGPR buffer                             | Single VGPR buf, ACCVGPR at [256:511]       | Separate ACCVGPR buffer (PARAM arg=5)       |
| ACCVGPR access                             | `rvgpr_dyn(reg + ACCVGPR_OFFSET)`           | `raccvgpr_dyn(reg)` / `waccvgpr_dyn(reg)`  |
| c_bufs                                     | 5 params (sgpr,vgpr,vmem,lds,scratch)       | 6 params (+accvgpr)                         |
| MFMA aliasing (vdst==src)                  | Not needed (disjoint in same buffer)        | 2-phase temp array                          |
| GFX9 16-bit lo-half zeroing                | No (preserves upper 16 -- RDNA behavior)    | Yes                                         |
| VGPR bit-slice pcode assignment            | No                                          | Yes (read-modify-write)                     |
| Conditional pcode side effects             | No                                          | Yes                                         |
| Scope                                      | 3 files (emu.py, pcode.py, amdgpu.py)      | 10 files (+ dsl.py, generate.py, 3x ins.py)|

Edge cases where our implementation may diverge: f16 VOP lo-half writes, sub-32-bit VGPR pcode, conditional stores, MFMA with vdst==src overlap.

## Learnings

### Bug #1 (FIXED): Scratch Size 4x Too Small for CDNA

COMPUTE_TMPRING_SIZE.WAVESIZE uses different alignment: GFX11 (RDNA) = 256 bytes, GFX9 (CDNA) = 1024 bytes. Emulator had `scratch_size = wavesize * 4` (RDNA), should be `wavesize * 16` for CDNA. Caused OOB writes corrupting instruction bytes. Fix: `scratch_size = wavesize * (16 if self.gpu.arch == "cdna" else 4)`.

Debug path: segfault -> SIGSEGV handler -> mprotect -> GDB -> traced garbage address to scratch_load_dword offset exceeding buffer -> verified against `ops_amd.py:1043` mem_alignment_size.

### Bug #2 (FIXED): Missing `acc` Bit on Memory Ops

CDNA scratch/global/flat memory instructions have an `acc` bit. When `acc=1`, vdata targets ACCVGPRs instead of VGPRs. Our `_compile_mem_op` ignored this, corrupting accumulator spill/reload. Only affected kernels with ACCVGPR scratch spills (e.g., `test_padded_conv2d_bs1` backward grad_w). Fix: 4 lines offsetting vdata_reg by ACCVGPR_OFFSET when `acc=1`.

### Emulator Performance Analysis

The emulator (`emu.py`) is slow by design — it interprets every GPU instruction in Python. Architecture (lines 1852-1930):

1. **Per-instruction compilation**: Each instruction is decoded, compiled to a tinygrad CPU kernel via UOps, and cached (`_get_runner`, line 1721). First execution compiles; subsequent hits use cache.
2. **Per-instruction dispatch**: Inner loop (line 1908) calls `fxn(*[c_bufs[g] for g in globals_list])` for every instruction. Overhead: Python function call + ctypes argument marshaling + kernel launch.
3. **Sequential waves**: Waves within a workgroup execute sequentially (line 1905). With wave64 and 256 threads, 4 waves run one after another.
4. **Sequential workgroups**: Triple loop (lines 1886-1888) over `gx * gy * gz`.

**Low-hanging fruit (NOT implementing — risk of bugs):**
- **Multiprocess workgroups**: Workgroups are independent (share global memory, but writes don't conflict in practice). Could use `multiprocessing.Pool` for `gx * gy * gz > 1`.
- **Batch instruction sequences**: Compile multiple non-barrier instructions into a single larger UOp kernel instead of one kernel per instruction.
- **Wave parallelism**: Waves between barriers could run in parallel threads (they share LDS, but loads/stores are sequential within a wave anyway).

**Why test_fa is slow**: Flash attention kernel has ~1000+ instructions per wave, 4+ waves per workgroup, many workgroups. At ~1ms per instruction (Python overhead), a 1000-instruction kernel takes ~1 second per wave × 4 waves × N workgroups. The 20+ minute runtime is expected.

**xdist parallelism**: Works for independent tests (test_ops/test_dtype: 648 tests in 85s with 16 cores). Does NOT help for single monolithic tests like test_fa (one big kernel execution).

### Key Technical Details

- Kernel descriptor at `prog_addr - kernel_code_entry_byte_offset` (NOT always `prog_addr - 256`; our kernels use kce_offset 4160-4352)
- `v_pk_add_f32` OPSEL latent bug: `opsel & 2` works by accident (Python/C truthiness of `2`), should be `(opsel >> 1) & 1`
- CDNA3/CDNA4 instruction compatibility: opcodes are empty spaces in CDNA3 -- merged ISA possible

### Environment Setup

- `AMD=1 MOCKGPU=1` needs `comgr` (ROCm) + `clang` to compile kernels
- No GPU hardware needed -- everything runs on CPU
- `pytest -n=auto` scales linearly with cores; single kernel is single-threaded
- CDNA4 gets 40min CI timeout vs 15min for RDNA (emulator is slow)

### Test Results (2026-02-21, cc50e/cc100e 8-core Ubuntu, master 0255a64a2 + 10 commits)

| Test Suite                                    | Result                                           | Time  |
|-----------------------------------------------|--------------------------------------------------|-------|
| test_ops + test_dtype + test_dtype_alu (core) | **648 passed, 0 failed**, 99 skipped, 1 xfailed | 35s   |
| Regression tests (test_emu_cdna_bugs.py)      | **26 passed, 0 failed**                          | 3s    |
| TK compile-only (DEV=NULL EMULATE=AMD_CDNA4)  | **20 passed, 0 failed**                          | 27s   |
| GEMM test_tiny (M=N=256,K=64) emulator exec  | **No crash** (exit 124 timeout, not 139 SEGV)   | >300s |
| GEMM test_tiny on cdna_emu_2 (qazalin)       | **PASS** (10 kernels, all complete)              | 12s   |

**GEMM kernel execution is fast** (<1s per kernel on our 8-core VPS). The bottleneck is `c.realize()` — tinygrad's scheduling/compilation phase before the emulator runs. On `cdna_emu_2` branch this takes ~315ms ("scheduled 5 kernels in 315.28 ms"). On our rebased-on-master code, it hangs for 300+ seconds.

**CI scope**: GitHub Actions runs test_ops/dtype/linearizer/jit/graph/multitensor/hcq/cfg_viz. **NOT test_tk.py or test_asm_gemm.py**.

### Bug #3 (FIXED): MOCKGPU LDS Size Too Small for GEMM

PR #14815 set `lds_size_in_kb = 128` in `_gpu_props_cdna`. Real MI350X has 160KB LDS per CU. The GEMM asm kernel needs 133,120 bytes (130KB) of LDS, exceeding 128KB. Fix: `lds_size_in_kb = 160`.

### Bug #4 (FIXED): GEMM ASM Kernel Uses Unknown GFX950 SMEM Encoding

The CDNA4 GEMM asm kernel compiles via `comgr` to machine code containing instruction word `0xf4080500`. bits[31:26] = `0b111101` = 61. **This is NOT an unknown buffer load** — it's a GFX950-specific SMEM encoding. Standard SMEM uses format 48 (`0b110000`), but `comgr` for gfx950 emits format 61 (`0b111101`). Same fields, same opcodes, different prefix.

Fix: 3-line `SMEM_F61(SMEM)` subclass in `cdna/ins.py` overriding only the encoding field. Added `SMEM_F61` to decoder format list (before `SMEM` so it matches first) and to emulator dispatch table.

### Bug #5 (FIXED): MUBUF Handler Missing from Emulator Dispatch

The emulator dispatch table had no MUBUF handler. GEMM kernel uses `buffer_load/store_dwordx4` (MUBUF format). Fix: Added `irc.MUBUF: _compile_mem_op` to `_INST_HANDLERS`. The existing `_compile_mem_op` already handled MUBUF address computation (SRD base + vaddr + soffset + offset).

### Bug #6 (FIXED): MUBUF OOB Stores Corrupt Host Memory

**Root cause**: The emulator maps the entire host address space as GPU vmem (`external_ptr=0`). MUBUF stores with `num_records=0x80000000` (2GB) compute addresses that always pass bounds checks. When these addresses land on non-writable pages, the CPU backend's `*ptr = active ? val : *ptr` pattern triggers SEGV_ACCERR (always writes, even for inactive lanes).

**Two-layer fix**:
1. **Trash page + address clamping**: Allocate anonymous RW page. In MUBUF make_addr, redirect OOB addresses (below 0x10000 or beyond num_records) and inactive-lane addresses to trash page.
2. **vmem_guard.so**: Custom SIGSEGV handler for Linux. On SEGV_ACCERR, modifies RAX in ucontext to trash page address, retrying the write harmlessly. Catches the remaining edge cases where JIT-compiled `*ptr = active ? val : *ptr` touches non-writable pages.

### Bug #7 (FIXED): JIT Code Pages Silently Corrupted by MUBUF Stores

**Symptom**: GEMM test crashes with SIGSEGV at NULL function pointer (DEBUG=0/2) but works with DEBUG=1/3. Different memory layout from debug prints prevents the overlap.

**Root cause**: CPU JIT backend maps code pages as RWX. MUBUF stores with 2GB num_records compute addresses landing on these pages. Writes succeed silently (page is writable), corrupting the JIT code. Next function call jumps to garbage.

**Fix**: `_protect_jit_code_page()` — after each instruction JIT-compiles via `_get_runner()`, call `mprotect(page, PROT_READ|PROT_EXEC)` to remove write permission. Now writes trigger SEGV_ACCERR, caught by vmem_guard.so and redirected to trash page. Each CPUProgram gets its own mmap region, so protecting one page doesn't affect later compilations.

### Test Scalability Issue

test_fa has 32,768 workgroups × 512 KV iterations. At ~0.1-1ms per GPU instruction through the Python emulator, a single test takes 1-9 hours. test_simple_matmul (8192x8192) similarly has 16,384 workgroups. These tests are not designed for the emulator — they're for real GPU hardware.

For emulator validation of MFMA/ACCVGPR/LDS correctness, the smaller TK tests (test_add, test_load_store, test_max) are sufficient and pass in seconds.

## Local Repos

All under `~/dev/tinygrad/`:

| Repo                     | Branch              | Purpose                                                    |
|--------------------------|---------------------|------------------------------------------------------------|
| `repo/`                  | many                | Main working repo, many branches. Remotes: origin, fork, private-cdna4, tg-pub |
| `tinygrad-cdna4/`        | `cdna4-fa`          | CDNA4 emulator work (PR #14815 + our 16 commits + GEMM/FA). 902/902 CI tests pass |
| `tinygrad-cdna4-smem61/` | `cdna4-smem61`      | **SMEM F61 fix only.** Fresh master clone + 3-line decoder fix + 2 regression tests |
| `tinygrad-cdna4-agent2/` | `cdna4-fa`          | Another agent's copy, behind on fixes                      |
| `tinygrad-cdna4-pr-draft/`| `public-clean`     | PR draft repo, has `draft` remote                          |

### SMEM F61 Fix (`tinygrad-cdna4-smem61/`)

Minimal fix for `ValueError: unknown cdna format word=0xf4080500`. GFX950 `comgr` encodes SMEM at format 61 (0b111101) instead of ISA XML's format 48 (0b110000).

**Files changed** (7 lines, 2 files):
- `tinygrad/runtime/autogen/amd/cdna/ins.py` — 3-line `SMEM_F61(SMEM)` subclass (inherits all fields, overrides encoding)
- `tinygrad/renderer/amd/__init__.py` — import + format list entry (before `C_SMEM` so it matches first)

**No `emu.py` changes needed** — MRO fallback in dispatch table (lines 1340-1344) finds `SMEM` -> `_compile_smem` automatically.

**Tests** (`test/amd/test_emu_cdna_bugs.py`, new file):
- `test_decode_0xf4080500` — decoder test with actual crashing bytes
- `test_s_load_dwordx2_format61` — full emulator pipeline test

## Infrastructure

### Vultr CI Instance

- **Instance**: `fba7aca8-d9e5-43aa-bd5d-69e74e66f2bd` / `64.237.48.234`
- **Plan**: vhp-8c-16gb-amd ($0.143/hr)
- **OS**: Ubuntu 24.04, Python 3.12.3
- **Setup**: ROCm comgr + clang + shallow tinygrad clone + our patches copied over

### Pricing Alternatives

| Provider | Plan               | Cost/hr | Notes                               |
|----------|--------------------|---------|-------------------------------------|
| Vultr    | vhp-8c-16gb-amd    | $0.143  | Proven setup, documented            |
| Vultr    | vhp-12c-24gb-amd   | $0.214  | Max VHP, only 50% more cores        |
| Hetzner  | cx53 16vCPU         | $0.027  | 5x cheaper, no API key set up       |

## Next Steps (for next agent session)

### 1. Fix test_tiny scheduling hang (HIGH PRIORITY)

**Problem**: `verify_asm_gemm(1,256,256,64)` hangs in `c.realize()` for 300+ seconds before the emulator even runs. On qazalin's `cdna_emu_2` branch, the same call schedules in 315ms.

**Root cause hypothesis**: The scheduler rework (PR #14909 "start ripping out old scheduler -- no maps") changed how `Ops.PROGRAM` / `Ops.LINEAR` / `Ops.INS` nodes are processed. `cdna_emu_2` is 109 commits behind master and uses the OLD scheduler. Our code is rebased on new master.

**Investigation steps**:
1. Profile the `realize()` call: `cProfile.run("c.realize()")` to see what function takes 300+ seconds
2. Compare scheduler behavior: check if `Ops.PROGRAM` (used by custom_kernel/asm_gemm) is handled correctly by the new scheduler
3. Try checking out `cdna_emu_2` branch temporarily and running `test_tiny` there to confirm it works
4. Look at PR #14909 diff for changes affecting `Ops.PROGRAM` / `Ops.INS` scheduling
5. Check if `build_kernel` returns 11,365 instruction UOps that are each being individually scheduled (instead of treated as opaque bytes)

**Quick test**: `AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 DEBUG=2 PYTHONPATH=. timeout 60 python3 -m pytest test/backend/test_asm_gemm.py::TestGemmLarge::test_tiny -v --tb=short 2>&1`

### 2. Get test_tiny actually passing

Once the scheduling hang is fixed, the GEMM kernel execution itself works:
- No crashes (JIT page protection + vmem_guard + trash page all working)
- 256 workgroups complete in <1s each (measured with instrumentation)
- Need to verify numerical correctness (allclose with reference)

### 3. Flash attention under emulator

After GEMM passes:
- `AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 PYTHONPATH=. python test/testextra/test_tk.py`
- Will be slow (thousands of workgroups) but should work with all the MUBUF/MFMA/ACCVGPR fixes
- May find new emulator bugs (FA uses different instruction patterns than GEMM)

### 4. Remove vmem_guard.so dependency

Currently requires `/tmp/vmem_guard.so` on Linux for crash-free GEMM execution. This is a compiled C library with a custom SIGSEGV handler. For upstream PR, we need a pure-Python alternative:
- Option A: Pre-map all possible vmem target pages as RW before kernel execution
- Option B: Use Python signal handler (slower but no C dependency)
- Option C: Improve address clamping in MUBUF make_addr to eliminate all OOB addresses

### 5. Commit and prepare PR

Current uncommitted state: clean (all changes committed after rebase). Branch has 10 commits on master.

Files changed from master:
- `test/mockgpu/amd/emu.py` — MUBUF handler, trash page, vmem_guard, JIT page protection, SMEM_F61 dispatch
- `test/mockgpu/amd/amdgpu.py` — LDS 160KB
- `test/mockgpu/helpers.py` — PythonRemu valid_mem_ranges
- `tinygrad/runtime/autogen/amd/cdna/ins.py` — SMEM_F61 subclass
- `tinygrad/renderer/amd/__init__.py` — SMEM_F61 in decoder
- `test/amd/test_emu_cdna_bugs.py` — 26 regression tests

## Key Reference: PR #14925 Screenshot (qazalin's working run)

From qazalin's screenshot on `cdna_emu_2` branch with `DEBUG=2 MOCKGPU=1 AMD=1 MOCKGPU_ARCH="cdna4"`:
- **Scheduled**: 5 kernels in 315ms, CACHE MISS, 7382 uops in cache
- **Kernel 12** `gemm_1_256_256_64`: compile 2649ms, total 2868ms (the actual GEMM)
- **Kernel 13** `r_2_16_32_4_4_64_4`: compile 91ms (epilog/conversion)
- **Second batch** (backward): cache hit, GEMM 1791ms
- **Total**: 11.882s, 10 kernels (5 per scheduling batch)
- **Key**: "~2 seconds" in PR description refers to GEMM kernel emulator execution time (2.6s first, 1.8s cached), NOT total test time

## vmem_guard.so

Custom SIGSEGV handler. Source at `/tmp/vmem_guard.c` on both cc50e and cc100e. Compiled with `gcc -shared -fPIC -o /tmp/vmem_guard.so /tmp/vmem_guard.c`.

```c
// Catches SEGV_ACCERR with RDX==0 (vmem base pattern from CPU backend).
// Modifies RAX to point to trash page, retries the faulting instruction.
// install(trash_addr), remove(), redirects() -> count
```

Build: `ssh cc50e 'gcc -shared -fPIC -o /tmp/vmem_guard.so /tmp/vmem_guard.c'`

## Sync Commands

```bash
# Local -> staging -> both nodes
cd "/Users/npinto/dev/tinygrad/tinygrad-cdna4" && \
  rsync -az --delete --exclude='.git' --exclude='__pycache__' \
  "/Users/npinto/Nico Dropbox Dropbox/Nicolas Pinto/dev/tinygrad/tinygrad-cdna4/" . && \
  rsync -az --delete --exclude='.git' --exclude='__pycache__' . cc50e:~/dev/tinygrad/tinygrad-cdna4/ && \
  rsync -az --delete --exclude='.git' --exclude='__pycache__' . cc100e:~/dev/tinygrad/tinygrad-cdna4/

# Regression tests (cc50e, 3s):
ssh cc50e 'cd ~/dev/tinygrad/tinygrad-cdna4 && PYTHONPATH=. python3 -m pytest test/amd/test_emu_cdna_bugs.py -v --tb=short'

# Core tests (cc100e, 35s):
ssh cc100e 'cd ~/dev/tinygrad/tinygrad-cdna4 && AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=0 PYTHONPATH=. python3 -m pytest -n=auto test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py --tb=line -q'

# GEMM test_tiny (currently hangs in scheduling):
ssh cc50e 'cd ~/dev/tinygrad/tinygrad-cdna4 && AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 DEBUG=2 PYTHONPATH=. timeout 60 python3 -m pytest test/backend/test_asm_gemm.py::TestGemmLarge::test_tiny -v --tb=short 2>&1'
```

## Dataset (refreshed 2026-02-20)

- `../dataset/tinygrad_prs.jsonl` -- 13,714 PRs
- `../dataset/tinygrad_issues.jsonl` -- 1,079 issues
- Refresh: `cd ../dataset && python download_prs.py tinygrad/tinygrad . --refresh && python download_issues.py tinygrad/tinygrad . --refresh`
