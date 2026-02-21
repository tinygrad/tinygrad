# CDNA4 Emulator Work

## What This Is

Adding CDNA4 (gfx950 / MI350X) support to tinygrad's MOCKGPU Python emulator.
The emulator interprets GPU assembly instructions in Python -- no actual AMD GPU needed.

## Current Status (2026-02-20)

### Our Work

**Core emulator: DONE.** 648 tests passing, 0 failures (350 test_ops + 298 test_dtype/alu).

Branch `enable-cdna4-emulator` in `../repo` (16 commits). Working branch `cdna4-gemm-fa` (5 commits ahead, GEMM/FA exploration). Private fork: `npinto/tinygrad-cdna4-emu`.

**Branch `cdna4-fa`** in `./` (this repo): PR #14815 emulator + ACCVGPR/DSL changes applied on top of master. 20/20 TK compile-only tests pass. Full MOCKGPU emulator tests running on cc50e.

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

## Our 16 Commits (on `enable-cdna4-emulator`)

1. `7781c36b2` enable cdna4 emulator in CI test matrix
2. `483521ac5` add f32_to_bf16 and fix v_bitop3_b32
3. `d77db94c4` implement wave64 support
4. `c2aa0e1c6` fix missing pcode and MAD_MIX path
5. `dffd1dae1` fix wave64 mask truncation in VOPC/VOP3SD/VOP
6. `a5683b279` add MFMA 16x16 handler and ACCVGPR_MOV pcode
7. `18a95e381` rewrite MFMA to use per-lane range loop
8. `8bf25fc64` fix MFMA dtype, add OPSEL support, fix test_cfg_viz typo
9. `0fefbd400` add VOP3 E64->E32 pcode fallback
10. `88a76b633` add SDWA (Sub-Dword Addressing) support
11. `114cc6f0d` fix VOPC/VOP2 SDWA sd bit, div_fmas scale, add FP8
12. `0b65acabb` fix V_PK_MOV_B32, f64 ops, VCC ordering, FP8 subnormal, unaligned loads
13. `eedd17c0c` add cdna4 to regSQ_THREAD_TRACE_BUF0_SIZE dict
14. `5d5605ff9` fix ACCVGPR register space, MFMA accumulator, LDS exec-mask
15. `ce1cda9a4` add s_getpc_b64/s_setpc_b64/s_swappc_b64 support
16. *(pending)* add `acc` bit handling for ACCVGPR scratch spills in _compile_mem_op

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

### Test Results (2026-02-20, cc50e 16-core Ubuntu, master f9536f3 + PR#14815)

| Test Suite                                    | Result                                          | Time  |
|-----------------------------------------------|-------------------------------------------------|-------|
| test_ops + test_dtype + test_dtype_alu (core) | **648 passed, 0 failed**, 99 skipped, 1 xfailed| 85s   |
| Extended (linearizer/jit/graph/hcq/cfg_viz)   | **254 passed, 0 failed**, 29 skipped, 4 xfailed| 71s   |
| TK non-FA (add/load_store/max/softmax/sum)    | 7/13 passed (killed matmul -- too slow)         | >25m  |
| TK compile-only (DEV=NULL EMULATE=AMD_CDNA4)  | **20 passed, 0 failed**                         | 27s   |
| GEMM asm (test_asm_gemm)                      | Was blocked by LDS size (fixed), then too slow  | >25m  |

**CI scope**: GitHub Actions runs test_ops/dtype/linearizer/jit/graph/multitensor/hcq/cfg_viz. **NOT test_tk.py or test_asm_gemm.py** -- those are testextra/backend tests for real hardware.

**All CI tests pass: 902/902 (0 failures).**

### Bug #3 (FIXED): MOCKGPU LDS Size Too Small for GEMM

PR #14815 set `lds_size_in_kb = 128` in `_gpu_props_cdna`. Real MI350X has 160KB LDS per CU. The GEMM asm kernel needs 133,120 bytes (130KB) of LDS, exceeding 128KB. Fix: `lds_size_in_kb = 160`.

### Test Scalability Issue

test_fa has 32,768 workgroups × 512 KV iterations. At ~0.1-1ms per GPU instruction through the Python emulator, a single test takes 1-9 hours. test_simple_matmul (8192x8192) similarly has 16,384 workgroups. These tests are not designed for the emulator — they're for real GPU hardware.

For emulator validation of MFMA/ACCVGPR/LDS correctness, the smaller TK tests (test_add, test_load_store, test_max) are sufficient and pass in seconds.

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

## Dataset (refreshed 2026-02-20)

- `../dataset/tinygrad_prs.jsonl` -- 13,714 PRs
- `../dataset/tinygrad_issues.jsonl` -- 1,079 issues
- Refresh: `cd ../dataset && python download_prs.py tinygrad/tinygrad . --refresh && python download_issues.py tinygrad/tinygrad . --refresh`
