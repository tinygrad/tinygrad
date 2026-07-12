# Adreno 630 (Snapdragon 845) FP16 GEMM Optimization

## Device Access

```bash
ssh tc3
cd /data/openpilot/tinygrad_repo
pkill -9 python3   # recover from GPU hangs (no reboot needed)
```

## Running the benchmarks

```bash
# Patched compiled kernel (~190 GFLOPS)
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_gemm.py
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_gemm.py --m 512 --n 512 --k 512

# Hand-assembled kernel tests (pure ALU, pure load, patched GEMM)
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_asm_gemm.py

# Subgroup/quad broadcast probes
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_shfl_probe.py
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_shfl_probe.py --bench throughput --ops-per-iter 16
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_shfl_probe.py --op quad --bench throughput --ops-per-iter 16

# Direct texture/isam bandwidth sweep
PYTHONPATH=. DEV=QCOM python3 extra/gemm/qcom_texture_bw.py --threads 128 --loads 32
```

## Current Findings: THREAD128 Runtime

The QCOM runtime used to hardcode `mesa.THREAD64` in compute dispatch state. Adding
`THREAD128=1` to `tinygrad/runtime/ops_qcom.py` selects `mesa.THREAD128` for:

- `A6XX_SP_CS_WGE_CNTL`
- `A6XX_SP_CS_CNTL_0`
- the NIR `A6XX_SP_CS_WGE_CNTL` path

This matches OpenCL's FP16 MAD peak on A630:

| Command | Result |
|---------|--------|
| `PYTHONPATH=. DEV=QCOM python3 extra/mmapeak/qcom_fp16_mad_peak.py` | `345.64 GFLOPS` |
| `PYTHONPATH=. DEV=QCOM THREAD128=1 python3 extra/mmapeak/qcom_fp16_mad_peak.py` | `690.35 GFLOPS` |
| `PYTHONPATH=. DEV=CL python3 extra/mmapeak/qcom_fp16_mad_peak.py` | `690.76 GFLOPS` |

For hand GEMM kernels, use `THREAD128=1` for all new measurements.

### ALU-Only GEMM-Shape Measurements

Measured on `tc3` with `THREAD128=1`, scalar `8x8` GEMM shape:

| Kernel/profile | Registers | Result | Notes |
|----------------|-----------|--------|-------|
| Compiler vector16 `mmapeak` | compiler | `~690 GFLOPS` | Not GEMM-shaped; vector-vector MAD stream |
| Hand compiler-pattern ALU stream | `f9 h8` | `714-718 GFLOPS` | Mirrors OpenCL vec16 lowering; `x=mad(x,y,y)`, `y=mad(x,y,x)` |
| True GEMM ALU body, `4x12`, distinct B, `row_col_kk` | `f8 h28` | `676.1 GFLOPS` | `acc=A_scalar*B_half4+acc`, one-shot unrolled body |
| True GEMM ALU body, `4x8`, distinct B, `row_col_kk` | `f8 h24` | `662.3 GFLOPS` | `acc=A_scalar*B_half4+acc`, four wave-pairs |
| True GEMM ALU body, `4x16`, reused B, `row_col_kk` | `f8 h32` | `679.3 GFLOPS` | Valid FMA form, but B columns are reused for ALU stress |
| Generic hand ALU stream, bad source pattern | `f8 h48` | `~357 GFLOPS` | Repeatedly reads same `hr0.x/hr4.x` |
| Generic hand ALU stream with source1 relative `(r)` | `f8 h32` | `~519 GFLOPS` | Best at 3-4 wave-pair occupancy |
| Correct high-reg `8x8 --profile alu` | `f28 h32` | `454.6 GFLOPS` | GEMM scalar-broadcast schedule |
| Low-reg `8x8 --experimental-twopass --profile alu` | `f15 h32` | `452.4 GFLOPS` | Donor/two-pass profile remains occupancy-limited |
| Low-reg `8x8 serial --profile alu` | `f8 h32` | `681.7 GFLOPS` | Four wave-pair ALU profile; not a correct full GEMM path yet |
| Serial `8x16 --profile alu` | `f8 h48` | `467.5 GFLOPS` | More accumulators, but lower occupancy |

Takeaways:

- Raw hand ALU can exceed `600 GFLOPS` when it uses the compiler vec16 source pattern and a low register footprint: `qcom_alu_peak.py --compiler-pattern --pairs 8 --loops 64` measured `714.0 GFLOPS`.
- The >600 pattern is not the GEMM accumulation form. It writes `dst=src1` and uses the other vector as addend, while GEMM needs `dst += A*B` (`dst=src3`).
- A true scalar-broadcast GEMM FMA body can also exceed `600 GFLOPS` if scheduled as `row_col_kk` and measured as a one-shot unrolled body: `qcom_alu_peak.py --gemm-pattern --rows 4 --ncols 3 --bmode percol --order row_col_kk --unroll 16 --loops 1` measured `676.1 GFLOPS`.
- The `row_col_kk` ordering is the key ALU finding: consume all four K components for one output vector accumulator before moving to the next accumulator.
- Repeating the synthetic GEMM ALU body in a loop is not a valid source-preserving benchmark unless A/B sources are reloaded or loop-control registers are kept out of their half-register aliases; use `--loops 1` for `--gemm-pattern`.
- Arithmetic intensity is not the current ALU issue limit.
- The old high-reg/donor-style `8x8` GEMM ALU profiles are capped around `452-455 GFLOPS`, but the low-freg serial profile reaches `681.7 GFLOPS`; the `8x8` ALU body is not inherently capped.
- MAD instruction order and source1-relative encoding did not materially improve the donor-style `8x8` profiles.
- Occupancy/register footprint, texture-sync placement, and a correct low-reg store path matter more than the specific legal MAD order.

### Current Correct GEMM Results

All entries below are full-output all-ones checked unless noted otherwise.

| Kernel | THREAD128 | Result | Notes |
|--------|-----------|--------|-------|
| Correct scalar `8x4` donor-store | yes | `255.9 GFLOPS` | `f12 h24`, texture-roof limited by AI 2.67 |
| Correct high-reg scalar `8x8` donor-store | yes | `196.8 GFLOPS` | `f28 h32`, store/loop not improved by THREAD128 |
| Low-reg scalar `8x8` two-pass store | yes | `188.8 GFLOPS` | Now correctness-stable under THREAD128 but slower |
| Low-reg scalar `8x8` serial + donor8 store | yes | `189.1-191.1 GFLOPS` | Correct; `f12 h32`, proves low-reg serial compute is valid when store is fixed |
| Low-reg scalar `8x8` split-A + add256 donor store | yes | `360.2-378.6 GFLOPS` | Correct; `f10 h28`, four wave-pairs, pre-unroll baseline |
| Low-reg scalar `8x8` split-A + K-unroll 4 + add256 donor store | yes | `425.8-436.0 GFLOPS` | Correct; `f10 h28`, four wave-pairs, previous best 8x8 path |
| Low-reg scalar `8x8` split-A + K-unroll 8 + next-B prefetch + tight add256 store | yes | `467.9-468.8 GFLOPS` | Correct; `f8/f9 h28`, four wave-pairs, first verified >460 path |
| Low-reg scalar `8x8` pipelined A/B | yes | `287.2 GFLOPS` | Correct; double-buffered inputs, `f15 h48` |
| Low-reg scalar `8x8` pipelined A/B, no next-buffer sync | yes | `288.4 GFLOPS` | Correct; `--b-coord-delay -1 --no-next-sy`, current-buffer sync still required |
| Low-reg scalar `8x8` pipeline4 | yes | `287.9 GFLOPS` | Correct; 4x K4 unroll needs larger donor envelope, does not improve throughput |
| Low-reg scalar `8x8` batch2 | yes | `222.0 GFLOPS` | Correct but slower; loading two K steps then computing loses overlap |
| Pipelined scalar `8x4` | yes | `200.7 GFLOPS` | Correct with `--a-coord-delay 0`; lower AI plus extra buffering is slower than baseline `8x4` |
| Direct `4x8` low-reg donor-store | yes | `271.5 GFLOPS` | Correct; repeated `4x4` compiler donor store, `--coord-delay 0` or `-1` |
| Direct `4x16` native-store | yes | `184.2 GFLOPS` | Correct; native `4x16` compiler store fixes coverage but needs high full-register footprint |
| Direct `4x16` low-reg donor-store | yes | `331.4 GFLOPS` | Correct; stride dependency waits fixed full coverage, `f8 h32`, `--coord-delay 4` |
| Direct `4x16` compact-acc hand ASM store | yes | `~334-336 GFLOPS` | Correct; accumulators start at `hr12`, `f8 h28`, `--k-unroll 4`, no runtime donor-store slicing |
| Direct `4x16` compact-acc hand ASM store, reduced K-sync | yes | `~382-388 GFLOPS` | Correct; `--stable-bx --k-unroll 4 --first-sync-only`, `f8 h28`, `sy=2` |
| Direct `4x16` compact-acc hand ASM store, persistent coords | yes | `400.5-402.1 GFLOPS` | Correct; `--stable-bx --stable-ay --inc-coords --persistent-coords --first-sync-only`, `f10 h28`, loop `421 -> 417` |
| Direct `4x16` persistent coords, B-first schedule | yes | `421.2-434.8 GFLOPS` | Correct; same `f10 h28` and loop size, but loads first B pair before A to hide B texture latency |
| Direct `4x16` B-first with low A coords | yes | `424.4-429.6 GFLOPS` | Correct; lowers metadata to `f8 h28`, but speed is flat vs `f10 h28` |

The split-A `8x8` K-unroll-8 path with next-B prefetch and tight add256 stores is the fastest correct hand path so far and is the first verified path above 460 GFLOPS. The compact-acc direct low-register `4x16` kernel with reduced per-unroll sync, persistent coordinates, and B-first scheduling remains the fastest correct 4x16 hand path.

#### FP32 Accumulate From FP16 Images

The standalone hand FP32 path in `qcom_8x4_gemm.py` is correctness-stable but not competitive with the compiler-shaped assembly patch. The original scalar `8x4` route reads FP16 images with `isam.f16`, converts with `cov.f16f32`, accumulates with `(rpt3)mad.f32`, and writes a float C buffer with `stg.f32`.

```bash
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_8x4_gemm.py --fp32-accum --variant serial \
  --ncols 1 --threads 128 --b-coord-delay 5 --check
```

Current checked result:

```text
serial:fp32 ncols=1 scalar_tile=8x4 threads=128 fregs=28 hregs=1 reg_count=29 wave_pairs=3 intensity=2.67 flop/B mad_density=1.03 shader_instrs=273 loop_instrs=124 bytes=2184 envelope_bytes=2832
mad.f16=0 mad.f32=32 rpt3=32 isam=12 sy=14 serial_syncs=all
CHECK PASS all 1048576 float outputs are 1024.0
```

Latest direct-load probes added `emit_isam_f32_vec`, `--direct-f32-loads`, `--sampler-per-texture`, and `--fp32-accum --ncols 2`. Correct checked timings were still low: ncols1 conversion path `43.7 GFLOPS`, ncols1 direct `110.5 GFLOPS`, ncols2 direct `146.6 GFLOPS`, and ncols2 direct no-store `145.1 GFLOPS`. Direct `isam.f32` from `imageh` is therefore valid with the sampler-per-texture path, but this full hand-assembled route is too slow for the 250 GFLOPS target.

The lower-register `4x4` FP32 prototype in `qcom_intensity_gemm.py` is now verified with full-output float checks. It must use the direct FP32 donor prologue; the older half donor prologue made B loads miss 1-2 K contributions in row/column-dependent regions even though post-constant stores passed.

```bash
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --fp32-accum --ncols 1 \
  --threads 128 --coord-delay 4 --direct-f32-loads \
  --sampler-per-texture --check
```

Current checked/timed result:

```text
ncols=1 covered_N=1024 fregs=20 hregs=1 waves=96 intensity=2.00 flop/B mad_density=1.36 shader_instrs=161 loop_instrs=47 bytes=1288 envelope_bytes=2792
mad.f16=0 mad.f32=16 rpt3=16 isam=8 qbc=0 sy=2
CHECK PASS all 1048576 float outputs are 1024.0
best observed timing: 154.8 GFLOPS (13.872 ms)
```

Direct `isam.f32` from `imageh` is correct in this direct-prologue `4x4` path. With sampler 0 for both textures it reached `137.7 GFLOPS`; using sampler index equal to texture index reached `151.6-154.8 GFLOPS`. Probe timings for the faster direct-load shape: no-store `150.8 GFLOPS`, skip A loads `171.4 GFLOPS`, skip B loads `233.9 GFLOPS`, skip A+B loads `292.8 GFLOPS`. The scalar-MAD variant (`64` scalar `mad.f32`, no `rpt3`) is correct but slower at `104.6 GFLOPS`.

THREAD128 compact-register `4x4` FP32 probes in `qcom_intensity_gemm.py` are correct but not a 300 route. `--compact-fp32` streams one A vector at a time and lowers metadata to `f12`; it passes full-output float checks with the donor float-store epilogue but only measured `120.5 GFLOPS` full and `114.0 GFLOPS` no-store. `--compact-fp32-preload` preloads A/B into `r0-r7` and keeps state in `r12`; a short wait is required before the donor store when copying state back to `r7`, and the checked full kernel measured `217.2 GFLOPS` at `f13`. `--compact-fp32-hybrid` keeps row/col/K state in `r7`, places A3 in `r12`, and is the cleanest low-register variant: full-output checks pass, `--coord-delay 3` is valid and measured `209.9 GFLOPS`, while delays `1` and `2` are invalid (`1020.0` outputs). At `--coord-delay 4`, the hybrid path measured `205.6 GFLOPS` full, `205.4 GFLOPS` no-store, `233.8 GFLOPS` no-store skip-A, `277.3 GFLOPS` no-store skip-B, and `312.9 GFLOPS` no-store skip-A+B. The generic hand `STG_F32` store path produced mostly zero output; the compiler-donor float epilogue is still required for reliable stores. Lowering the full-register footprint alone is therefore insufficient: real A/B texture scheduling remains the limiter.

Low-register `4x8` FP32 A-reuse now works correctly in `qcom_intensity_gemm.py`, and the fastest checked version uses default dispatch rather than `THREAD128=1`. The useful version is `--low-4x8-fp32 --preload-b`, which keeps both B column blocks live, uses `r12-r19` for accumulators, and keeps state in `r20` (`f21`). Reusing the 4-row donor float epilogue twice was invalid because the donor slice carried an `end` and needed store-spacing; the working full-store path uses the compiler's `ncols=2` float donor epilogue with a low-copy repack through dead input registers, avoiding the old `r24-r31` temp copy and preserving `f21`. Best checked command so far:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 HCQ2=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --fp32-accum --low-4x8-fp32 \
    --preload-b --batch-coords --ncols 2 --threads 128 --sampler-per-texture \
    --coord-delay -1 --alu-order kk_col_row --check
```

It passes all `1048576` float outputs. A 120-iteration full benchmark measured `232.3 GFLOPS` (`f21`, loop `73`, `12` direct `isam.f32`, `32` `(rpt3)mad.f32`, `sy=2`). The same shape without `HCQ2=1` measured `230.4 GFLOPS`; with `THREAD128=1 HCQ2=1` it only measured `199.5 GFLOPS`, so this hand FP32 path should currently use default dispatch. Correct no-store with the best order is `230.3 GFLOPS`, skip-A is `247.5 GFLOPS`, skip-B is `286.9 GFLOPS`, and skip-A+B is `309.2 GFLOPS`, showing B texture latency is still the primary limiter and the FP32 MAD body itself is only slightly above 300 in this schedule.

Negative `4x8` FP32 follow-ups: the original `f17` non-preload path is correct only as a diagnostic and remains slow (`~120 GFLOPS` no-store under default dispatch, `117.1` under `THREAD128=1`). Half-image `isam.f16` plus explicit `cov.f16f32` collapses to `56.5 GFLOPS` no-store, so direct `isam.f32` is still the right input path. `--stream-b` without a sync reaches `241.0 GFLOPS` no-store under default dispatch but fails full checks; adding the required sync makes it correct but only `202.2 GFLOPS`. Fixed NOP waits before consuming streamed B1 do not fix correctness. Double-buffered software pipeline variants are slower (`f30` B-only pipeline `~151 GFLOPS`, `f34` A+B pipeline `~162 GFLOPS` no-store), so the extra live registers cost more than the overlap buys. Underdeclaring the working `f21` kernel as `f20` hangs, so the metadata cannot be lowered. The explicit hand `STG_F32` path remains mostly zero/sparse output. The remaining limiter is real B texture scheduling, not store correctness.

Wider hand FP32 attempts in `qcom_intensity_gemm.py` are still not promising. The `--fp32-accum --ncols 2` path now has a correct compiler-donor `ncols=2` float-store epilogue and passes full-output checks, but the real conversion-load path is only `30.3 GFLOPS` under `THREAD128=1` (`fregs=32`, `loop_instrs=124`). No-store is still only `28-29 GFLOPS`; skip-A, skip-B, and skip-A+B no-store probes measured `37.8`, `112.0`, and `235.5 GFLOPS`, respectively. Direct `isam.f32` loads raise the ncols2 no-store probe to about `111 GFLOPS`, but full-output checks remain unstable/incorrect for ncols2, so those timings are diagnostics only. The full hand 4x4 direct path did pass a coordinate-delay sweep, with the best historical run around `153.1 GFLOPS` at `--coord-delay 1`, but that remains far below the compiler-shaped assembly patch.

The compiler-generated `simple_matmul.py` path with `DEV=QCOM:IR3 DEBUG=2 IMAGE=1 FLOAT16=1 N=1024 HALF=1` reaches about `196-199 GFLOPS` in the main `r_32_16_8_16_4_4_256_4` kernel. Disassembly shows a `4x4` FP32 accumulator tile with `max_reg=12`, `64` scalar `mad.f32`, `8` direct `isam.f32`, `1` `(sy)`, and typed image-float stores. That is the current practical compiler baseline for FP32 accumulate from FP16 images.

The best verified compiler-side FP32 patch is now `qcom_ir3_matmul_patch.py --n 704 --patch rpt3_l25_postinc_unroll22`. It keeps tinygrad's normal packed image layout, rewrites the compiler's `l25` loop into `(rpt3)mad.f32` accumulator groups, increments the K loop counter after the texture loads, and compares only once per 22-way unrolled group. The first l25 rewrite missed the original `end` instruction and hung; the fixed epilogue includes `instrs[119:134]`.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. DEV=QCOM:IR3 IMAGE=1 FLOAT16=1 HCQ2=1 \
  python3 extra/gemm/qcom_ir3_matmul_patch.py --n 704 --dtype half \
  --acc-dtype none --patch rpt3_l25_postinc_unroll22 --check --bench --iters 40
```

Verified result on `tc3`, `HCQ2=1` with default THREAD64 dispatch:

```text
main=r_22_11_8_16_4_4_176_4 image_bytes=7208 instrs=901 fregs=16 hregs=0
mad.f32=352 rpt_mad=352 isam=176 stores=4
CHECK PASS all 495616 outputs are 704.0
BENCH main 269.9 GFLOPS (2.585 ms)
```

The previous long-run l25 best was `rpt3_l25_unroll16_nosnop` at `258.8 GFLOPS`; `rpt3_l25_unroll16_nosnop_lastcmp0` reached `262.7 GFLOPS` by comparing only in the last unrolled body. The post-increment rewrite removes the explicit `mov r2.y, r10.x` loop-counter copy, drops obsolete loop nops, and moves the increment under the MAD body. Long checked results for the post-increment form: default dispatch `269.4 GFLOPS`, `HCQ2=1` `269.9 GFLOPS`, and `THREAD128=1` `256.2 GFLOPS`.

For the same post-increment l25 shape, `THREAD128=1` is still required for a possible 300+ path even though the full kernel is currently slower. Under `THREAD128=1`, no-store is only `253.8 GFLOPS`, but no-store with skipped A loads reaches `294.4 GFLOPS`, skipped B loads reaches `266.2 GFLOPS`, and skipped A+B loads reaches `316.7 GFLOPS`. This shows the THREAD128 control/ALU ceiling can cross 300, but the current A/B texture schedule cannot. A is the larger limiter on this shape.

Nearby checked post-increment probes did not beat N=704: N=736/unroll23 reached `262.8 GFLOPS`, N=800/unroll25 reached `263.9 GFLOPS` on a long run, N=608/unroll19 reached `266.1 GFLOPS` on a short run, and N=832/unroll26 fell to `203.0 GFLOPS`. For N=704, unroll22 is best so far; unroll16 was `268.9 GFLOPS`, unroll11 was `268.7 GFLOPS`, and unroll44 fell to `206.4 GFLOPS`. MAD accumulator reorderings were flat (`acc3210` long `269.6 GFLOPS` with `HCQ2=1`), and reverse `k3210` remained slower (`258.6 GFLOPS`).

THREAD128-specific l25 probes were negative: unroll4/8/11/16/22 measured about `247.7/250.9/253.2/250.1/249.8 GFLOPS`, while unroll44 fell to `181.3 GFLOPS`; `THREAD128=1 HCQ2=1` was also flat at `253.6 GFLOPS`. Correct load-order variants (`a0early`, `bfirst`) remained around `250-252 GFLOPS`, single-coordinate hoisting was either slower or invalid, and an A `isam.f16` plus `cov.f16f32` path was correct but collapsed to `94.2 GFLOPS`. A0 prefetch into `r6.w` after the current A0 MADs was invalid even with waits, so source-overwrite hazards are stricter than the logical liveness suggests. Follow-up prefetch diagnostics confirmed the constraint: moving A0 to `r6.x` corrupts accumulator registers, moving it to `r15.w..r16.z` is correct but drops to `182.4 GFLOPS` from `fregs=17`, fregs16 coordinate-pair rewrites for A0 still fail checks, and A2/A3 prefetch fail even when delayed until after all current MADs. B0-low remaps are not THREAD128-safe: waits around the late B0 reload and an extra `(sy)` after it still fail checks; the symmetric B0-first low-register schedule also fails.

Additional 300 push checks: `QCOM_PRIORITY=15` did not improve the current best (`THREAD128=1` remained `250.9-253.5 GFLOPS`, `HCQ2=1` default stayed `269.9 GFLOPS`). A short THREAD128 shape sweep around N704 left N704 as the only useful l25 candidate: N608/unroll19 passed but was only `203.3 GFLOPS`, N736/unroll23 passed but was `196.4 GFLOPS`, and N576/N640/N672/N768 did not match the l25 patch shape. Hand FP32 8x8 remains structurally register-heavy (`fregs` in the high 30s for ncols=2), so it is not a near-term 300 route without a major register-layout rewrite.

More N704 THREAD128/300-route probes were also negative. Reversing local-axis priority produced `r_11_22_16_8_4_4_176_4`, but noop was only `193.1 GFLOPS` and the l25 postinc patch was `250.7 GFLOPS`; skip-A/skip-B/skip-A+B no-store ceilings were `286.7`, `268.2`, and `316.7 GFLOPS`, so the load balance did not improve. Applying locals unsorted changed the prologue but kept A driven by `r48.x`, and noop fell to `183.1 GFLOPS`. Image upcast 8 collapsed to `79.0 GFLOPS`, image upcast 2 collapsed to `11.9 GFLOPS`, and nearby N640/N896 l23 patches stayed around `219-222 GFLOPS`. Corrected quad-A with `r48.x&3`, quad-A with an explicit texture wait, and quad-B one-load-per-quad with an explicit post-broadcast wait all failed checks with zero output. Low-register B0 remaps into `r1.y`, `r0.z`, and aligned `r1.x` failed (`352`, `4`, and `352` at idx0), so the low coordinate registers are not a usable f15 escape hatch for this l25 schedule. A bounded `BEAM=2` run again hit `OSError: [Errno 35] Resource deadlock avoided`; avoid longer BEAM on this device for this route.

Follow-up THREAD128 l25 scheduling checks also did not find a 300 route. Splitting the texture wait by delaying A0/A1 loads until after the first A2 MAD was only correct if the `r2.y` loop-counter increment stayed after the delayed A loads; the corrected variants passed but dropped to `166.3 GFLOPS` and `173.8 GFLOPS`, while moving the increment immediately after B0 failed (`idx=16 got=700.0`). Runtime local-size overrides were invalid for this compiled shape: `16,8,1` does not divide the total launch, while `4,32,1` and `8,8,1` failed checks with zero-output regions. Additional checked K/accumulator orders were flat or slower under THREAD128: `k2301` `234.4`, `k2310` `213.4`, `k1023` `242.7`, `k0132` `250.7`, `k0213` `251.2` on a longer run, `k3210` `209.5`, `acc3210` `253.2`, `acc1230` `253.3`, and accumulator-major `239.6 GFLOPS`; `a1mid` load order failed (`idx=32 got=700.0`).

THREAD128 runtime-state probes were also negative and the env hooks were removed. Mesa-like `QCOM_TSIZE=2` was flat on a sequential long run (`251.3 GFLOPS`), `QCOM_TSIZE=1` failed with zero output, `QCOM_TSIZE=4` and `QCOM_USIZE=1` were flat, `QCOM_WGE_SCALAR=1` was flat, `QCOM_SINGLE_SP=1` dropped to `130.2 GFLOPS`, `QCOM_CONSTLEN=128/192` only produced short-run noise and long `CONSTLEN=128` was `252.0 GFLOPS`, `QCOM_THREADMODE=1` dropped to `50.5 GFLOPS`, `QCOM_MERGEDREGS=1` failed with zero output, `QCOM_ISAMMODE_CL=1` was flat, and TPL1 destination datatype override dropped to `240.5 GFLOPS`. Underdeclaring the normal l25 kernel as `f15` failed at idx0, so THREAD128 needs a real lower-register schedule rather than metadata-only occupancy tricks. New f15 B0-streaming attempts into old B1/B2 slots failed checks (`700.0` outputs), and the old `b0low` f15 schedule still fails THREAD128 even at shorter unrolls and stronger waits.

Additional THREAD128-focused follow-up remained negative. Rebaselining current code gave `rpt3_l25_postinc_unroll22` at `253.7 GFLOPS` on a short checked run and `253.7 GFLOPS` on a 30-iter run, while default dispatch stayed around `269.8 GFLOPS`. Setting `SP_PS_WAVE_CNTL.THREADSIZE` through a temporary `QCOM_PS_WAVE_THREADSIZE=1` runtime hook was flat/slower (`251.3 GFLOPS`), so the hook was removed. Moving A1 earlier is not safe: `a1copyearly`, `a1copyearly_wait`, and the f17 coordinate-copy version all failed full-output checks at `idx=32 got=700.0`, even when B2/B3 coordinates were copied away from `r8.*`. Combining `a0early` with K orders where late A1 is consumed last was correctness-safe but not a stable speedup: best short run was `a0early_k0231` at `254.9 GFLOPS`, but a 30-iter comparison fell to `252.5 GFLOPS`. A full 24-permutation `a0early_k####` sweep did not produce a clear winner. An in-unroll coordinate-increment rewrite, intended to avoid recomputing A/B coordinates after the first body of `unroll22`, failed checks (`idx=0 got=440.0`, then `606.0/611.0` after safer recomputation attempts) because A0/A1 texture destinations clobber the apparent persistent coordinate registers. Current conclusion is unchanged: THREAD128 is blocked by texture scheduling/register-liveness constraints in this l25 shape, not by stores or dispatch bits.

The lower-register `b0low` post-increment schedule removed all `r15.*` B-vector use and passed at `fregs=15` under default dispatch, but it was slower (`~255.6 GFLOPS`) and failed correctness under `THREAD128=1`. Lower metadata alone is therefore not enough; the MAD/load order must also be THREAD128-safe.

Important diagnostics: for the earlier `rpt3_l25_unroll16_nosnop` loop, no-store measured only about `256.8 GFLOPS`; no-store with skipped A loads reached `277.1 GFLOPS`, skipped B loads `264.1 GFLOPS`, and skipped A+B loads `284.4 GFLOPS`. That ceiling is still below 300, so load/store deletion alone is not enough; the remaining FP32 gap is dominated by full-register pressure/control scheduling rather than the typed image stores.

The best verified N=1024 compiler-side patch remains `rpt3_accum_f32_unroll8`. It keeps tinygrad's normal packed image layout and rewrites only the default main IR3 kernel. The compact `rpt3_accum_f32_default` patch moves the A0 vector to `r13`, raises the declared full-register footprint to `f14`, replaces the compiler's `64` scalar `mad.f32` ops with `16` `(rpt3)mad.f32` groups, and removes the now-dead `r5.z/r5.w` saves from the loop prefix. Full-output all-ones checks pass.

Same-session patch-harness comparison on `tc3` with `THREAD128=1`, `IMAGE=1`, `FLOAT16=1`, `dtype=half`, and `acc_dtype=float`:

| Patch | Main GFLOPS | Notes |
|-------|-------------|-------|
| `noop` | `159.7` | Compiler default: `f13`, `64` scalar `mad.f32` |
| `reorder_rpt_f32_compact` | `196.3` | `f13`, `36` scalar `mad.f32`, `16` rpt groups |
| `rpt3_accum_f32_default` | `204.5` | `f14`, `16` `(rpt3)mad.f32`, dead saves removed |
| `rpt3_accum_f32_unroll8` | `208.0` | `f14`, unrolled checked best for N=1024 |

The same tightened `rpt3_accum_f32_default` patch measured `155.1 GFLOPS` in the harness full-flow timer and `198 GFLOPS` for the main kernel in a single `DEBUG=2 --stats-run` run where noop measured `155 GFLOPS`; the device was in a throttled/low-clock state for that comparison. Negative but correct probes: `rpt3_accum_f32_accmajor` (`199.2 GFLOPS`) and `rpt3_accum_f32_nosnop` (`199.5 GFLOPS`) were slower than the default ordering with the compiler `(ss)nop` retained.

For N=512, the best checked path so far is `rpt3_n512_b0low_k3210_unroll16_nosnop`, which moves the B0 texture vector below `r15`, declares `f15` instead of `f16`, uses `rpt3` accumulator groups, unrolls the K loop by 16, and drops the `(ss)nop`. Fresh checked runs after killing stale remote Python measured `234.7-234.8 GFLOPS` on default THREAD64. The same patch with `THREAD128=1` measured about `231.0 GFLOPS`; `HCQ2=1` measured `234.6 GFLOPS`. Earlier `rpt3_n512_b0low_k3210_unroll16` measured `229.4-230.4 GFLOPS`, and `rpt3_n512_unroll16` measured `225.5 GFLOPS`.

Important negative probes: deleting or NOPing the apparent N=512 dead coordinate copies corrupts output, so those packed sampler-coordinate writes are semantically required. `rpt3_n512_b0low_unroll32` is not reliable (`idx=33216 got=508.0`), `rpt3_n512_b0low_k3210_unroll32_nosnop` is also wrong (`idx=65664 got=508.0`), and f14 N=512 repacks fail even when declared as f15, so the shifted-load schedule is wrong rather than merely underdeclared. The N=1024 `f13pack` attempt also fails all-ones checks (`got=1020.0`), so the current N=1024 verified ceiling remains around `208 GFLOPS`. `BEAM=2/4` hit QCOM deadlocks during beam-search timing and should be avoided for this route.

`BEAM=1` found an alternate compiler schedule (`r_2_32_16_4_4_4_2_4_2_256_4`), but it is not a valid improvement candidate: with real filled inputs it measured only `~92-94 GFLOPS` despite passing all-ones correctness. Earlier higher BEAM timings came from an uninitialized/zero-like input state and should not be counted.

Follow-up performance probes showed the current `8x8` FP32 shape is not the route to 400 GFLOPS:

| Probe | Result | Finding |
|-------|--------|---------|
| Compiler imageh input, FP32 output, `ncols=2` | `82.9 GFLOPS` | Correct but far below target |
| Hand FP32 `ncols=2`, donor `ncols=2` store, post-constant | correct | Reusing the compiler 16-vector `stg.f32` epilogue can cover the full output |
| Hand FP32 `ncols=2`, real B loads | invalid | B texture path produces sparse/row-group-dependent output; not countable |
| Hand FP32 `ncols=2`, skip A/B loads, no store | `199.0 GFLOPS` | Upper bound for this 8x8 register footprint/schedule is about stock FP32 speed |
| Raw hand FP32 MAD microbench | `~355 GFLOPS` | Device can issue more FP32 ALU than the GEMM-shaped loop, but still below the nominal 468 note here |

Implication: pushing FP32 GEMM above 400 needs a different tile/schedule, not incremental fixes to this `8x8` path. The likely next candidate is a lower-register `4x16` FP32-accumulate shape that keeps more wave-pairs resident while amortizing B loads; the current `8x8` FP32 footprint (`f37`) is boxed in around 200 even before real texture loads.

#### Latest 420+ GFLOPS Run

Measured on `tc3` with `THREAD128=1`, `IMAGE=1`, `FLOAT16=1`:

```bash
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --threads 128 --ncols 4 \
  --direct --compact-acc --stable-bx --stable-ay --inc-coords \
  --persistent-coords --alu-order row_col_kk --coord-delay -1 \
  --k-unroll 4 --first-sync-only --b-first --check
```

Final check:

```text
ncols=4 covered_N=1024 fregs=10 hregs=28 waves=3 intensity=3.20 flop/B mad_density=2.46 shader_instrs=677 loop_instrs=417 bytes=5416 envelope_bytes=15744
mad.f16=256 rpt3=256 isam=80 qbc=0 sy=2
CHECK PASS all 1048576 outputs are 1024.0
```

Benchmark command:

```bash
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --threads 128 --ncols 4 \
  --direct --compact-acc --stable-bx --stable-ay --inc-coords \
  --persistent-coords --alu-order row_col_kk --coord-delay -1 \
  --k-unroll 4 --first-sync-only --b-first --iters 220
```

Final checked benchmark runs:

| Run | GFLOPS | Time |
|-----|--------|------|
| 1 | `430.8` | `4.984 ms` |
| 2 | `430.0` | `4.994 ms` |
| 3 | `434.8` | `4.939 ms` |
| 4 | `425.6` | `5.046 ms` |
| 5 | `421.2` | `5.099 ms` |

#### How 420 Was Reached

Starting point was the previous fastest verified kernel:

```bash
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --threads 128 --ncols 4 \
  --direct --compact-acc --stable-bx --stable-ay --inc-coords \
  --persistent-coords --alu-order row_col_kk --coord-delay -1 \
  --k-unroll 4 --first-sync-only
```

Rebaseline before tuning was noisy but centered around 390-401 GFLOPS:

| Run | GFLOPS | Time |
|-----|--------|------|
| 1 | `399.4` | `5.377 ms` |
| 2 | `387.3` | `5.544 ms` |
| 3 | `385.7` | `5.567 ms` |
| 4 | `401.4` | `5.349 ms` |
| 5 | `394.1` | `5.449 ms` |

The bottleneck probes showed stores were not limiting:

| Probe | Result | Finding |
|-------|--------|---------|
| Same kernel, `--no-store` | `382.9-403.9 GFLOPS` | Removing stores did not materially improve throughput |
| Same kernel, `--post-constant` | `390.7-394.4 GFLOPS` | Store path plus loop remained in the same range |
| Same kernel, `--store-constant` | `~0.089 ms` | Store-only epilogue is tiny vs `~5.0 ms` full GEMM |

The ALU/load probes showed the checked kernel was not ALU-issue limited:

| Probe | Result | Finding |
|-------|--------|---------|
| Same kernel, `--no-store --alu-reps 2` | `536.4-545.2 GFLOPS` | More ALU per same loads immediately beats 420 |
| Same kernel, `--no-store --alu-reps 3` | `594.1-595.3 GFLOPS` | Load/setup overhead is being amortized |
| Same kernel, `--no-store --alu-reps 4` | `538.2-549.1 GFLOPS` | Too much body/envelope pressure; not useful as a real path |
| Same kernel, `--no-store --skip-a-loads` | `448.5-452.7 GFLOPS` | A loads have cost but are not dominant |
| Same kernel, `--no-store --skip-b-loads` | `587.6-595.7 GFLOPS` | B texture loads/setup dominate the gap |
| Same kernel, `--no-store --skip-a-loads --skip-b-loads` | `673.3-673.4 GFLOPS` | ALU/control ceiling for this loop shape |

The successful change was `--b-first`: load the first B pair before issuing A loads. This keeps the same `f10 h28`, same `loop_instrs=417`, same `mad.f16=256`, same `isam=80`, and same `sy=2`, but lets the A texture loads hide part of first-pair B texture latency. That moved the full-output checked kernel from `~400 GFLOPS` to `421.2-434.8 GFLOPS`.

Robustness checks around `--b-first`:

| Variant | Result | Finding |
|---------|--------|---------|
| `--coord-delay -1` | correct, `421.2-434.8 GFLOPS` | Best path |
| `--coord-delay 0/1/2/4` | correct, slower | Extra NOPs reduce MAD density from `2.46` to `2.06` |
| `--store-shlg-offsets` | correct, `426.5-428.7 GFLOPS` | Store variant is flat; default hand store is fine |
| `--store-scalar-offsets` | correct, no speedup | Store math is not bottleneck |
| `--donor-store` | correct, no speedup | Donor-store slicing is not needed |
| `--threads 128` | correct, best | Best balance for this schedule |
| `--threads 256` | correct, `421.1-423.0 GFLOPS` | Works but slightly slower |
| `--threads 64` | invalid | Sparse wrong outputs; do not use with `--b-first` |
| `--low-a-coords` | correct, `424.4-429.6 GFLOPS` | Reduces metadata to `f8 h28`; not faster, so full-register metadata is not limiting |
| `--low-a-coords --threads 64` | correct, `257.9-261.6 GFLOPS` | Lower fregs fixes 64-thread correctness but remains slow |
| `--low-a-coords --threads 256` | correct, `426.3-428.1 GFLOPS` | Flat vs 128-thread path |
| `--k-unroll 2 --first-sync-only` | correct, `369.6-375.3 GFLOPS` | Too little latency hiding |
| `--k-unroll 4` without `--first-sync-only` | correct, `326.5-341.1 GFLOPS` | Extra MAD syncs dominate |
| `--k-unroll 8 --b-first --first-sync-only` | correct, `411.1-413.1 GFLOPS` | B-first fixes the old sparse-output failure but the larger body is slower |
| `--stream-b --stream-b-no-sync` variants | correct, `349-375 GFLOPS` | Hides some latency but adds too many instructions |

#### 460 GFLOPS Attempt

The current 4x16 tile appears boxed in below 460 GFLOPS without reducing B ingress or changing tile shape.

Hard upper-bound probes on the current B-first path:

| Probe | Result | Finding |
|-------|--------|---------|
| `--b-first --no-store --skip-a-loads` | `453.3-453.5 GFLOPS` | Even deleting all A loads stays below 460 |
| `--b-first --low-a-coords --no-store --skip-a-loads` | `458.0-459.4 GFLOPS` | Best A-free upper bound; still below target |
| `--b-first --no-store --skip-b-loads` | `590.5-590.8 GFLOPS` | B ingress remains the dominant limiter |
| `--b-first --no-store --skip-a-loads --skip-b-loads` | `673.4 GFLOPS` | ALU/control body has enough headroom |

Additional 460-path probes:

| Probe | Result | Finding |
|-------|--------|---------|
| Raise KGSL `devfreq/min_freq` to `710000000` | permission denied | Cannot lock max clock from this user |
| `--b-first` MAD order sweep | `row_col_kk` still best | Other legal orders were `~385-397 GFLOPS`; `kk_col_row` was invalid |
| Col2 prefetch into `hr28..hr31` | correct only with targeted waits, `~240 GFLOPS` | Extra high half regs / waits destroy throughput; probe removed from script |
| Tail column split schedule | correct, `426.0-427.8 GFLOPS` | Same loop size, no improvement; probe removed from script |
| Partial `ncols=5` B-first probe | `~250 GFLOPS` with `f10 h32`, `~388-393 GFLOPS` with `f8 h32` | Wider 4-row tile is not promising; probe removed from parser |
| Low-freg `8x8 serial --profile alu` | `681.7 GFLOPS` | Strong ALU headroom, but full serial path still lacks a correct low-reg store/prologue combination |
| Correct `8x8 --experimental-twopass` with `--fregs-override 8` | hung | High full-register use cannot be hidden by lowering metadata |
| `8x8 serial --donor8-store` | correct, `189.1-191.1 GFLOPS` | Known-good 8-row donor store fixes correctness at `f12 h32`, but remains slow |
| `8x8 --split-a --donor8-add256-store --no-next-sy` | correct, `360.2-378.6 GFLOPS` | Pre-unroll split-A baseline; `f10 h28`, four wave-pairs |
| `8x8 --split-a --split-k-unroll 2 --donor8-add256-store` | correct, `404.2 GFLOPS` | K-unroll starts to hide texture/setup cost |
| `8x8 --split-a --split-k-unroll 4 --b-coord-delay 3 --donor8-add256-store` | correct, `425.8-436.0 GFLOPS` | Previous best 8x8 path; `f10 h28`, `loop_instrs=110`, `isam=64`, `sy=2` |
| `8x8 --split-a --split-k-unroll 8 --b-coord-delay 3 --donor8-add256-store` | correct, `415.8 GFLOPS` | Same register footprint but larger shader; instruction-cache/body size likely hurts |
| `8x8 split-A K-unroll-4 --split-prefetch-next-b --split-fast-coords --fregs-override 8` | correct, `446.2 GFLOPS` | Refills dead B registers for next K step; first real improvement after K-unroll-4 |
| `8x8 split-A K-unroll-8 --split-prefetch-next-b --split-fast-coords --fregs-override 8` | correct, `449.5-453.3 GFLOPS` | Next-B prefetch makes unroll-8 viable; best before store tightening |
| Same K-unroll-8 prefetch path, `--no-store` | `466.7 GFLOPS` | Shows store epilogue became the final blocker for 460 |
| Same K-unroll-8 prefetch path, `--add256-store-mode pairs` | correct, `451.9 GFLOPS` | Generated store slice with fewer nops; correct but not enough |
| Same K-unroll-8 prefetch path, `--add256-store-mode tight` | correct, `467.9-468.8 GFLOPS` | First verified >460 path; generated SAD + back-to-back stores |
| Same tight path, `--b-coord-delay 0` | correct, `468.5 GFLOPS` | Flat vs delay 1; delay `-1` is still invalid |
| Same tight path, `--split-hoist-b0-coord --fregs-override 9` | correct, `468.8 GFLOPS` long run, `469.3 GFLOPS` short run | Hoisting first next-B0 coord into `r8.x/r8.y` is correct but essentially flat |
| Same tight path, no-store/skip probes | `466.7 / 529.1 / 535.5 / 562.4 GFLOPS` | no-store / skip-A / skip-B / skip-both; remaining 500 gap is A+B texture ingress, not ALU |
| Same tight path, `--threads 64` | correct, `277.0 GFLOPS` | Lower thread count is much slower |
| Same tight path, `--threads 256` | correct, `464.7-468.6 GFLOPS` | Fixed 8-row prologue row-log for 256 threads; no speedup vs 128 |
| Same tight path, `--fregs-override 7` | invalid | Full-register metadata below 8 corrupts output |
| Same tight path, `--fregs-override 6` | hung | Recover with `pkill -9 python3`; do not use |
| Same tight path, `--add256-gap <16` | invalid | Tight store still needs the old inter-column gap |
| Same tight path, `--add256-direct-sources` | invalid | Direct stores from accumulator hregs still violate the low-reg store-source convention |
| Same tight path, `--split-buffer-a` | invalid | Both `hr28..hr31` A buffering and low `hr12..hr15` A buffering with accumulators at `hr16` corrupt output |
| Same tight path, `--split-prefetch-next-a` | correct, `465.2 GFLOPS`; swapped before B1 `445.8 GFLOPS` | Moving A0-next earlier hurts texture issue balance |
| Same tight path, `--split-interleave-next-b` | correct, `460.4 GFLOPS` | Splitting B0-next refill around col1 MADs is slower |
| Same tight path, `--split-hoist-b0-coord` with `fregs=8` | invalid | Hoisted coord in `r4.y/r4.z` is clobbered before ISAM |
| Same tight path, `--split-inline-b-wait --split-inline-b-nop 1..7` | invalid | Inline `add.s(nop)` cannot replace the explicit coordinate wait NOP |
| Same tight path, `--split-add-a-rows` | invalid | A row coordinate formation must stay `or.b` for this schedule |
| Same tight path, `--split-prefetch-loop-b` | correct, `442.6 GFLOPS` | Predicate-skipped final prefetch fixes correctness, but loop-boundary B prefetch is much slower |
| Same tight path, `--split-quad-a` | hung/invalid | Row-per-quad A sharing with full-register quad broadcasts is not a valid path yet; early high/default layouts hung |
| Same tight path, `--split-high-a` | correct, `468.6 GFLOPS` | Moves A to `hr24..hr27` and accumulators to `hr8..hr23`; register layout is flat |
| Same tight path, `--split-high-a --split-hoist-b0-coord --fregs-override 9` | correct, `469.0 GFLOPS` | Flat vs non-high-A B0 hoist |
| Same tight path, `--split-low-a` | correct, `459.2-461.9 GFLOPS` | Moves A to `hr0..hr3` and B to `hr4..hr11`; needs declared `fregs=10`, while `fregs=8` corrupts output |
| Same tight path, `--split-low-a --split-quad-a` | invalid | Single-component quad broadcasts avoid the earlier hang but rows sourced through qbc are mixed/NaN; `shader_instrs=1127`, `loop_instrs=122`, `isam=80`, `sy=18` |
| Same tight path, `--split-high-a --split-quad-a --fregs-override 14` | invalid | Same row pattern as low-A qbc: directly loaded rows are ok, broadcast-derived rows are mixed; register placement/freg declaration is not the fix |
| Same tight path, branch-gated low-A quad load | invalid, not kept | Lane-0-only A loading plus qbc kept `isam=128`, grew to `shader_instrs=1231`, and corrupted every row; divergent hand branch form is not usable here |
| Same tight path, `--split-pair-b-coords` | initially correct but slower, then invalid when tightened | Pairing two B coordinates per wait did not improve texture issue; tightened `1006`-instruction version corrupts output |
| Same tight path, `--split-base-b-y` | invalid | Keeping B y as a base multiple of 4 and forming kk offsets with `or.b` corrupts first outputs |
| Same tight path, `--split-stream-next-b0` | correct, `458.2 GFLOPS` | Per-K component streaming of next B0 frees texture issue earlier but hurts MAD order enough to lose speed |
| Same tight path, `--split-stream-next-b1` | correct, `456.1 GFLOPS` | Same result for B1 streaming; earlier B issue does not offset disrupted row/col/kk order |
| Same tight path, `--swap-grid` | invalid at `--b-coord-delay 0`, correct but `441.1 GFLOPS` at delay 3 | Swapping row/column group IDs can make the store map correct, but fast B delay loses contributions and safe delay is slower |
| Same tight path, `--hregs-override 27/24/22/20/18` | hung before check | Underdeclaring half-register metadata is unsafe; recover with `pkill -9 python3` |
| Same tight path after FP16 peak warmup | `455.4 GFLOPS` | Governor/preheat did not help; long warmup can be slower |
| `8x16 split-A K-unroll-4` | correct, `307.3 GFLOPS` | Higher arithmetic intensity is overwhelmed by `hregs=48` / 3-wave occupancy; threads 64 is slower and threads 256 is invalid |
| `8x8 split-A K-unroll-16` with next-B prefetch | correct, `391.3 GFLOPS` | Fits larger envelope but instruction-cache/body size dominates |
| `8x8 split-A --no-store` | `380.0 GFLOPS` | Store overhead is modest; same `f10 h28` metadata |
| `8x8 split-A --no-store --skip-a-loads` | `425.0 GFLOPS` | A texture path costs about 45 GFLOPS from no-store baseline |
| `8x8 split-A --no-store --skip-b-loads` | `480.8 GFLOPS` | B texture path is the main limiter and has enough headroom for 4x16 parity if reduced |
| `8x8 split-A --no-store --skip-a-loads --skip-b-loads` | `583.2 GFLOPS` | ALU/control ceiling for this split-A loop; not ALU-limited |
| `8x8 split-A K-unroll-4 --strip-mad-sy` | invalid | First outputs become `-inf` / `NaN`; keep the current two MAD syncs |
| `8x8 split-A K-unroll-4 --grouped-b --b-coord-delay -1` | correct, `405.5 GFLOPS` | Fewer B coord waits but worse texture issue pattern |
| `8x8 split-A K-unroll-4 --grouped-b-cols --b-coord-delay -1` | correct, `409.9 GFLOPS` | Also slower than the scalar B setup path |
| `8x8 split-A K-unroll-8 --grouped-b --b-coord-delay -1` | correct, `379.9 GFLOPS` | Larger body plus grouped B is a dead end |
| `8x8 split-A K-unroll-4 --no-store` | `429.8 GFLOPS` | Store is not the main remaining limiter in the unrolled path |
| `8x8 split-A K-unroll-4 --no-store --skip-a-loads` | `508.2 GFLOPS` | A texture path still costs significant throughput |
| `8x8 split-A K-unroll-4 --no-store --skip-b-loads` | `524.6 GFLOPS` | B texture path is still the larger limiter |
| `8x8 split-A K-unroll-4 --no-store --skip-a-loads --skip-b-loads` | `567.0 GFLOPS` | ALU/control ceiling for the unrolled split-A shape |
| `8x8 split-A K-unroll-4 --b-coord-delay 2/1/0/-1` | invalid | `--b-coord-delay 3` is still required |
| `8x8 split-A K-unroll-4 --threads 64` | correct, `259.8 GFLOPS` | Lower occupancy/parallelism is much slower |
| `8x8 split-A K-unroll-4 --threads 256` | correct, `429.1 GFLOPS` | Fixed by 8-row prologue row-log update; still flat vs 128 |
| `8x8 split-A --add256-gap <16` | invalid | Gap 16 is required; smaller gaps corrupt row 7 / first column |
| `8x8 split-A --stream-b1` | invalid | Tried to load second B group during first-column MADs; still misses contributions even with waits and syncs; parser flag removed |

Conclusion: 460 was reached by combining real B-ingress overlap with an epilogue reduction. The next-B prefetch schedule moves B loads for the next unrolled K step into dead B registers after current group-4 col0/col1 use, and tight generated add256 stores remove the donor store nops that became visible once the loop reached the mid-450s. The first 500 push did not find a valid faster schedule; the best verified long run remains `468.8 GFLOPS`, with skip-A and skip-B probes showing that another real A/B texture-ingress reduction is needed.

The key fix was adding dependency waits while widening the donor prologue's
column base from `gid.x*32+tid` to `gid.x*128+tid`; without waits, the repeated
adds did not chain and the kernel overlapped columns instead of covering the
tail.
The later compact-acc improvement moves the accumulator base from `hr16` to
`hr12`, reducing metadata from `hregs=32` to `hregs=28`. This is store-safe
because the 4-row donor pack only overwrites `hr12` after `row0,col0` has already
been copied into the store scratch registers.
The current default direct store is now explicit hand ASM rather than a runtime
slice from a donor binary. It hard-codes the compiler-style four-row address
schedule and `stg.f16` sequence, then packs output rows into `hr0..hr3` before
stores. A naive single-address `stg` path was invalid because it used dependent
scalar address math too aggressively and did not follow the compiler's low-register
store-source convention.

Useful commands:

```bash
# FP16 MAD peak parity with OpenCL
PYTHONPATH=. DEV=QCOM THREAD128=1 python3 extra/mmapeak/qcom_fp16_mad_peak.py

# GEMM-shaped ALU-only profile
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_8x4_gemm.py --ncols 2 --threads 128 --profile alu

# Fastest correct 4x16 full GEMM path
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --threads 128 --ncols 4 \
  --direct --compact-acc --stable-bx --stable-ay --inc-coords \
  --persistent-coords --alu-order row_col_kk --coord-delay -1 \
  --k-unroll 4 --first-sync-only --b-first --check
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --threads 128 --ncols 4 \
  --direct --compact-acc --stable-bx --stable-ay --inc-coords \
  --persistent-coords --alu-order row_col_kk --coord-delay -1 \
  --k-unroll 4 --first-sync-only --b-first --iters 220

# Fastest correct 8x8 path so far
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_8x4_gemm.py --variant serial --ncols 2 \
  --threads 128 --split-a --split-k-unroll 8 --b-coord-delay 0 \
  --donor8-add256-store --split-prefetch-next-b --split-fast-coords \
  --fregs-override 8 --add256-store-mode tight --check
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_8x4_gemm.py --variant serial --ncols 2 \
  --threads 128 --split-a --split-k-unroll 8 --b-coord-delay 0 \
  --donor8-add256-store --split-prefetch-next-b --split-fast-coords \
  --fregs-override 8 --add256-store-mode tight --warmup 10 --iters 500

# Previous fastest pipelined 8x8 path
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_8x4_gemm.py --ncols 2 --threads 128 --pipeline --a-coord-delay 4 --b-coord-delay -1 --no-next-sy --check
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_8x4_gemm.py --ncols 2 --threads 128 --pipeline --a-coord-delay 4 --b-coord-delay -1 --no-next-sy --warmup 5 --iters 30
```

Recent results and negative checks:

- Grouped A/B coordinate scheduling can pass some all-ones runs but is flaky under full scan/check; do not count its timings.
- Removing the current-buffer pipeline `(sy)` is incorrect; removing only the next-buffer `(sy)` is correct and gives a small speedup.
- `4x16` direct constant-store diagnostics still fail with the hand store path, proving that path is store/address incorrect before GEMM math is considered.
- Reusing a sliced donor `4x4` store epilogue is correct for direct `4x8` and direct `4x16` only after adding waits between every dependent widened-column stride add, including the final wait before B-coordinate setup.
- The earlier `4x16` tail-zero pattern was not primarily a store-epilogue limit: the donor prologue stride adds were reading the old `r7.y`, effectively using a `4x8` column stride and overlapping workgroups.
- The native direct `4x16` compiler store epilogue fixes full-output coverage, but it uses high full registers and drops the verified full kernel to about `184 GFLOPS`.
- Hybrid hand-tail stores and scalar/shlg-offset donor-store diagnostics did not beat the fixed low-reg donor-store path.
- A pipelined direct `4x16` no-store experiment was slower (`~220 GFLOPS`) because the extra double-buffer registers reduced occupancy (`hregs=40`).
- `ncols=3`/`4x12` probes now report `covered_N=768`; after correcting for partial coverage the donor-store path is only `295.0 GFLOPS`, so a split `4x12 + tail` plan is not promising.
- `threads=256` is correctness-clean for direct `4x8`, but slower (`~252.6 GFLOPS`) than `threads=128`.
- The experimental `8x16` donor-store path in `qcom_8x4_gemm.py` also needed stride-add waits; this fixes tail coverage but it still has sparse row failures and remains invalid.
- Semantic K-unroll for direct `4x16` is correct for `--k-unroll 2` and `4`, but mostly flat (`~330-332 GFLOPS` without compact accumulators). `--k-unroll 8` produced sparse zero output chunks and is invalid.
- Direct `4x16 --preload-b` is full-output correct but slow (`176.6 GFLOPS`) because `hregs=36` drops occupancy.
- Direct `4x16 --stream-b` and `--stream-b --stream-b-no-sync` are full-output correct, but did not beat the baseline (`~310 GFLOPS` with sync, `~329 GFLOPS` without the extra pair sync).
- Direct `4x16 --compact-acc` is correct and is the best small improvement so far (`~334-336 GFLOPS` with hand ASM stores and `--k-unroll 4`).
- Direct `4x16 --compact-acc --first-sync-only` is the main current improvement. Full-output checks pass with only the first MAD sync in each unrolled K loop (`sy=2` total), and `--stable-bx --k-unroll 4` measures `~382-388 GFLOPS`.
- Direct `4x16 --compact-acc --stable-bx --stable-ay --inc-coords --persistent-coords --first-sync-only` is the previous best verified 4x16 path. It keeps A row coords in `r8/r9`, increments A/B coords across unrolled K steps, preserves them across loop iterations, and has measured `400.5-402.1 GFLOPS` with full-output checks.
- Direct `4x16 --compact-acc --stable-bx --stable-ay --inc-coords --persistent-coords --first-sync-only --b-first` is the current best verified 4x16 path. It preserves the same loop instruction count/register footprint as the persistent-coordinate path but loads the first B pair before A, hiding part of the B texture latency under the A loads. Final checked runs measured `421.2-434.8 GFLOPS`.
- Direct `4x16 --b-first --low-a-coords` is correct and reduces metadata to `f8 h28`, but it remains flat at `424.4-429.6 GFLOPS`; metadata pressure is not the current limiter.
- Current 4x16 upper-bound probes put the practical scheduling ceiling below 460: `--b-first --no-store --skip-a-loads` is only `453.3-453.5 GFLOPS`, and `--b-first --low-a-coords --no-store --skip-a-loads` is only `458.0-459.4 GFLOPS`.
- The 460-specific schedule probes were negative: col2 B prefetch was either invalid or `~240 GFLOPS`, tail column split was flat at `426.0-427.8 GFLOPS`, and temporary partial `ncols=5` was slow and not full-output coverage.
- Low-freg `8x8 serial --profile alu` reaches `681.7 GFLOPS`, so `8x8` has ALU headroom if it stays at `f8 h32`; the full serial path still fails full-output checks because the naive scalar store is unsafe, the 4x16 hand epilogue mismatches the 8-row prologue, and the dynamic 4-row store remains invalid.
- `8x8 --experimental-twopass --fregs-override 8` hung on `tc3`; recover with `pkill -9 python3`. Keep the correct two-pass path at its declared `f15 h32` metadata.
- `8x8 serial --donor8-store` proves the low-reg serial compute loop is correct once stores are fixed, but only reaches `189.1-191.1 GFLOPS` at `f12 h32`.
- `8x8 --split-a --donor8-add256-store --no-next-sy` is the correct pre-unroll split-A baseline. It preloads both B groups, computes two 4-row A groups, and uses a low-freg donor store that forms the second column by adding `+256` bytes to the first column's row addresses. Full-output checks pass at `f10 h28`; benchmark range is `360.2-378.6 GFLOPS`.
- `8x8 --split-a --split-k-unroll 4 --b-coord-delay 3 --donor8-add256-store` was the previous best verified 8x8 path. Full-output checks pass with `f10 h28`, `reg_count=24`, `shader_instrs=602`, `loop_instrs=110`, `isam=64`, `sy=2`; benchmark range is `425.8-436.0 GFLOPS`.
- `8x8 --split-a --split-k-unroll 8 --b-coord-delay 0 --split-prefetch-next-b --split-fast-coords --fregs-override 8 --add256-store-mode tight` is the current best practical path. Full-output checks pass with `f8 h28`, `reg_count=22`, `shader_instrs=1022`, `loop_instrs=109`, `isam=128`, `sy=2`; benchmark range is `467.9-468.6 GFLOPS` on long runs, with prior short runs at `468.1-468.6 GFLOPS`.
- The best 500-push variant, `--b-coord-delay 0 --split-hoist-b0-coord --fregs-override 9`, also full-output checks and measured `468.8 GFLOPS` on a long run (`469.3 GFLOPS` short run). It needs `f9` for `r8.x/r8.y` hoisted B0 coords and is effectively tied with the `f8` path.
- The winning path depends on both parts. K-unroll-8 plus next-B prefetch but donor store mode topped out at `449.5-453.3 GFLOPS`; `--no-store` reached `466.7 GFLOPS`, exposing the epilogue as the last blocker. `--add256-store-mode tight` replaces the donor store slice with generated SAD plus back-to-back stores and raises the verified full kernel above 460.
- The post-460 bottleneck is A+B texture ingress. On the tight K-unroll-8 path, no-store is `466.7 GFLOPS`, skip-A is `529.1 GFLOPS`, skip-B is `535.5 GFLOPS`, and skip-both is `562.4 GFLOPS`.
- The 500-specific low-register schedule probes were negative: direct accumulator store sources are invalid, smaller add256 gaps are invalid, next-A prefetch is correct but slower, buffered-A variants are invalid, high-A and low-A layouts are flat/slower, paired/base B-coordinate forms are invalid or slower, interleaved next-B refill is slower, per-component B0/B1 streaming is slower, swapped-grid B-cache reuse is invalid or slow, predicated loop-boundary B prefetch is correct but slower, inline B wait encoding is invalid, row-per-quad A sharing is invalid/hung, and fregs/hregs below the known-safe footprint corrupt or hang.
- K-unroll-16 with the same next-B prefetch is correct but slow (`391.3 GFLOPS`) despite fitting a larger envelope; do not continue in that direction unless instruction-cache behavior changes.
- `8x8 --split-a --split-k-unroll 8 --b-coord-delay 3 --donor8-add256-store` is correct and still fits the enlarged donor envelope (`8336 / 13064` bytes), but it is slower at `415.8 GFLOPS`; doubling the body does not pay for reduced loop control.
- Split-A K-unroll robustness is narrow. The original K-unroll-4 path requires `--b-coord-delay 3`; lower B coordinate delays corrupt output. `--threads 64` is correct but slow at `259.8 GFLOPS`; `--threads 256` is now correct after the row-log fix but flat (`429.1 GFLOPS`). The add256 donor store still needs `--add256-gap 16`; smaller gaps corrupt row 7 / first column.
- Split-A K-unroll grouped-B modes are correct with `--b-coord-delay -1`, but slower: K-unroll-4 `--grouped-b` is `405.5 GFLOPS`, K-unroll-4 `--grouped-b-cols` is `409.9 GFLOPS`, and K-unroll-8 `--grouped-b` is `379.9 GFLOPS`. The scalar B setup with explicit delay remains best.
- Removing MAD syncs with `--strip-mad-sy` is invalid on K-unroll-4; first outputs become `-inf` / `NaN`.
- Split-A K-unroll-4 bottleneck probes show the path is still ISAM/texture limited, not ALU limited: no-store is `429.8 GFLOPS`, skipping A loads is `508.2 GFLOPS`, skipping B loads is `524.6 GFLOPS`, and skipping both reaches `567.0 GFLOPS`.
- Experimental split-A `--stream-b1` did not become correct. It still misses one contribution in the streamed column even after adding B-coordinate waits, col1 syncs, and hard gaps; the parser flag was removed.
- Direct `4x16 --b-kk-pipeline` is invalid: it repeatedly missed 1-2 FP16 contributions even with strong MAD sync diagnostics.
- Direct `4x16 --compact-acc --stable-bx --first-sync-only --k-unroll 8` is full-output correct but slower (`~355-358 GFLOPS`); non-stable `k-unroll 8` still has sparse zero chunks and is invalid.
- Experimental `8x16` is full-output correct at `--threads 64` and `--threads 256`, but slow (`144.7` and `~260 GFLOPS` respectively); `--threads 128` still has sparse failures.
- Experimental low-freg `8x8 --donor4-store` is invalid. The two 4-row donor chunks do not match the 8-row prologue/store convention; observed failures include `1020.0` outputs and zero rows.

### Combined ISAM + Real GEMM ALU Probes

These are throughput probes, not valid GEMM results when `--no-store` or
`--alu-reps > 1` is used. They combine real texture `isam` loads with the legal
GEMM FMA form `acc = A_scalar * B_half4 + acc`.

| Probe | Result | Notes |
|-------|--------|-------|
| `4x16 --direct --no-store --row-col-kk --alu-reps 1` | `328.4 GFLOPS` | One real ALU body per loaded A/B tile; no stores |
| `4x16 --direct --no-store --row-col-kk --alu-reps 2` | `471.2 GFLOPS` | First combined ISAM+real-GEMM-ALU probe over 400 |
| `4x16 --direct --no-store --row-col-kk --alu-reps 4` | `577.1 GFLOPS` | Load overhead amortized further |
| `4x16 --direct --no-store --row-col-kk --alu-reps 8` | `639.8 GFLOPS` | Approaches true GEMM ALU body ceiling |
| `4x16 --direct --no-store --row-col-kk --quad-a --alu-reps 2` | `444.0 GFLOPS` | Quad-A path is slower than normal A loads here |
| `4x16 --direct --donor-store --row-col-kk --coord-delay 4` | `331.4 GFLOPS` | Correct full-output GEMM after stride-dependency fix |
| `4x16 --direct --compact-acc --stable-bx --first-sync-only --k-unroll 4` | `~382-388 GFLOPS` | Correct full-output GEMM; previous reduced-sync path |
| `4x16 --direct --compact-acc --stable-bx --stable-ay --inc-coords --persistent-coords --first-sync-only --k-unroll 4` | `400.5-402.1 GFLOPS` | Correct full-output GEMM; previous persistent-coordinate path |
| Same persistent-coordinate path with `--b-first` | `421.2-434.8 GFLOPS` | Correct full-output GEMM; current best verified path |
| Same B-first path with `--low-a-coords` | `424.4-429.6 GFLOPS` | Correct full-output GEMM; fregs drops to 8 but speed is flat |
| Same persistent-coordinate path, `--no-store` | `382.9-403.9 GFLOPS` | Store path is not the bottleneck |
| Same persistent-coordinate path, `--store-constant` | `~0.089 ms` | Store-only lower bound; epilogue is negligible vs `~5.0 ms` GEMM |
| Same persistent-coordinate path, `--no-store --alu-reps 2` | `536.4-545.2 GFLOPS` | Load/setup amortization probe |
| Same persistent-coordinate path, `--no-store --alu-reps 3` | `594.1-595.3 GFLOPS` | Confirms the checked kernel is not ALU-issue limited |
| Same persistent-coordinate path, `--no-store --skip-a-loads` | `448.5-452.7 GFLOPS` | A loads cost measurable time but are not dominant |
| Same persistent-coordinate path, `--no-store --skip-b-loads` | `587.6-595.7 GFLOPS` | B texture loads/setup are the dominant bottleneck |
| Same persistent-coordinate path, `--no-store --skip-a-loads --skip-b-loads` | `673.3-673.4 GFLOPS` | ALU/control ceiling for this loop shape |
| Same B-first path, `--no-store --skip-a-loads` | `453.3-453.5 GFLOPS` | A-free upper bound for current B ingress is still below 460 |
| Same B-first low-A path, `--no-store --skip-a-loads` | `458.0-459.4 GFLOPS` | Best current 4x16 upper-bound probe, still below 460 |
| `4x16 --direct --native-store --row-col-kk` | `184.2 GFLOPS` | Correct full coverage, high full-register store epilogue |

Useful command for the first over-400 combined probe:

```bash
PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 THREAD128=1 \
  python3 extra/gemm/qcom_intensity_gemm.py --threads 128 --ncols 4 \
  --direct --no-store --row-col-kk --alu-reps 2 --iters 40
```

## Hardware: Adreno 630

- **SP**: Shader Processor, Qualcomm's shader core/cluster; roughly analogous to an NVIDIA SM or AMD CU
- **2 SPs**, each with 64 ALUs, 128 total
- **Clock**: ~400 MHz (thermal-dependent)
- **FP16 MAD peak**: 690 GFLOPS (measured via mmapeak with same-register repeated MAD)
- **FP16 MAD sustained**: 590 GFLOPS (realistic with `(rpt3)mad.f16`, 16 groups in a tight loop)
- **FP32 MAD peak**: 468 GFLOPS
- **Texture bandwidth**: 168 GB/s (measured, isam throughput)
- **Register file**: 192 KiB per SP on A630 (`reg_size_vec4=96`, `threadsize_base=64`, `wave_granularity=2`)
- **Wave sizes**: THREAD128 (128 fibers/wave) or THREAD64 (64 fibers/wave)

### Register File Constraints

The `fregs` and `hregs` fields in the shader binary are **vec4 footprints**, not scalar component counts.

- `r0` is one full vec4: `r0.x/r0.y/r0.z/r0.w`, four 32-bit components.
- `hr0` is one half vec4: `hr0.x/hr0.y/hr0.z/hr0.w`, four 16-bit components.
- In split OpenCL mode, one full vec4 costs the same storage as two half vec4s.
- The useful GPR namespace is `r0..r47` for full regs and `hr0..hr47` for half regs. `hr48+` reaches special/non-GPR names and is not usable for hand accumulators on A630.

For split full/half allocation, the full-equivalent per-fiber footprint is:

`reg_count = fregs + ceil(hregs / 2)`

Mesa reports A630 as `reg_size_vec4=96`, so a single 128-fiber wave-pair can hold up to 96 full-equivalent vec4 registers per fiber. The physical storage per SP is:

`96 vec4/fiber * 64 fibers * 2 wave-granularity * 16 bytes/vec4 = 196608 bytes = 192 KiB`

Older notes used `floor(12288 / (hregs * threads))`, which is only a rough half-only shortcut and is wrong once full regs or split full/half accounting matters.

### 8x4 Half Register Budget

`hr0..hr47` is 48 half4 registers = 192 FP16 scalar values per fiber. That is enough only for the serial-B 8x4 schedule:

| Live data | half4 regs | FP16 values |
|-----------|------------|-------------|
| 8x4 accumulators | 32 | 128 |
| 8 A texels | 8 | 32 |
| 4 B texels, one column group | 4 | 16 |
| **Serial-B subtotal** | **44** | **176** |
| Spare before scratch/alias pressure | **4** | **16** |

The preload-B schedule does not fit:

| Live data | half4 regs | FP16 values |
|-----------|------------|-------------|
| 8x4 accumulators | 32 | 128 |
| 8 A texels | 8 | 32 |
| 16 B texels, all column groups | 16 | 64 |
| **Preload-B subtotal** | **56** | **224** |

So 8x4 is not register-impossible, but only the serial-B form fits the addressable half register file. Preloading all B values needs at least 56 half4 registers before any scratch or store epilogue, beyond the usable `hr0..hr47` range.

| hregs | Max waves | Total fibers |
|-------|-----------|-------------|
| 24    | 4         | 512         |
| 31    | 3         | 384         |
| 48    | 2         | 256         |

Full registers and half registers **share the same physical storage**:
`r0.x` = `{hr0.x, hr0.y}`, `r0.y` = `{hr0.z, hr0.w}`, etc.
Writing a full register clobbers the aliased half registers and vice versa.

## Architecture of the GEMM Kernel

### Tiling

- **128 threads/workgroup** = 4 subgroups of 32 threads
- Each thread computes **4 rows x 1 col4** (4 output half4 vectors)
- Grid: `(N/128, M/16, 1)` — 16 rows per WG (4 subgroups x 4 rows)
- A is stored as `image2d_t` shape `(M, K/4)`, each pixel = half4
- B is stored as `image2d_t` shape `(K, N/4)`, each pixel = half4
- Per K iteration: 4 A loads + 4 B loads = 8 `isam.1d` texture fetches

### Loop Body (compiled, before patching)

```
mov r2.y, r6.z      ;; k4 -> A coord x (row0)
(rpt5)nop            ;; wait for mov
isam hr3.x, r2.y, t#0   ;; A[k4, row0] -> hr3
mov r2.w, r6.z
(rpt5)nop
isam hr2.x, r2.w, t#0   ;; A[k4, row1] -> hr2
mov r3.y, r6.z
(rpt5)nop
isam hr1.x, r3.y, t#0   ;; A[k4, row2] -> hr1
mov r3.w, r6.z
(rpt5)nop
isam hr0.x, r3.w, t#0   ;; A[k4, row3] -> hr0

add.s r4.z, r6.y, -3
(rpt5)nop
isam hr4.x, r4.y, t#1   ;; B[col4, k4*4+0] -> hr4
(sy)mad.f16 ...          ;; 16 scalar MADs for B[0] x 4 rows
;; ... repeat for B[1], B[2], B[3] with more isam + (sy) + MADs
```

**Problems**: 5 `(sy)` syncs per iteration (~100 cycles each), 4 `(rpt5)nop` waits
(6 wasted cycles each), scalar MADs instead of packed `(rpt3)`.

### Binary Patching (`patch_kernel` in `qcom_gemm.py`)

1. **Strip redundant `(sy)`**: Keep only the first `(sy)` on a MAD instruction per loop
   iteration. The QCOM compiler inserts `(sy)` before every MAD that follows an isam,
   but only one sync is needed to wait for all pending texture results.

2. **Convert scalar MADs to `(rpt3)mad.f16`**: When 4 consecutive MAD instructions have
   the same `src1`, sequential `dst/src2/src3`, the pattern matches `(rpt3)` repeat
   encoding. Each `(rpt3)` packs 4 MADs into 1 instruction slot.

3. **Merge `(rpt1)+(rpt1)` into `(rpt3)`**: Two adjacent `(rpt1)mad.f16` with compatible
   register sequences combine into a single `(rpt3)`.

Result: **5 `(sy)` → 2**, **41 scalar MADs → 15 `(rpt3)` + 2 `(rpt1)`**.
Speedup: **78 → 190 GFLOPS** (2.4x).

### Hand-Assembled Optimized Loop

Best verified kernel places B texels into 4 separate registers (hr4-hr7 instead of
all-hr4), enabling all 8 isam to be issued back-to-back with a single `(sy)`:

```
;; Coord setup (8 instructions)
mov r2.y, r6.z       ;; A coords
mov r2.w, r6.z
mov r3.y, r6.z
mov r3.w, r6.z
add.s r4.z, r6.y, -3 ;; B coords
add.s r5.x, r6.y, -2
add.s r5.z, r6.y, -1
mov r6.x, r6.y

;; 8 isam back-to-back (no nops between)
isam hr3.x, r2.y, t#0   ;; A row0
isam hr2.x, r2.w, t#0   ;; A row1
isam hr1.x, r3.y, t#0   ;; A row2
isam hr0.x, r3.w, t#0   ;; A row3
isam hr4.x, r4.y, t#1   ;; B k0
isam hr5.x, r4.w, t#1   ;; B k1
isam hr6.x, r5.y, t#1   ;; B k2
isam hr7.x, r5.w, t#1   ;; B k3

;; Single (sy) + 15 (rpt3)mad.f16 + 2 (rpt1)mad.f16 = 64 MADs
(sy)(rpt3)mad.f16 hr20.z, hr3.x, (r)hr4.x, (r)hr20.z  ;; row0 x B0
(rpt3)mad.f16 hr24.z, hr2.x, (r)hr4.x, (r)hr24.z       ;; row1 x B0
... ;; 13 more (rpt3) groups
(rpt1)mad.f16 hr13.z, hr0.w, (r)hr7.x, (r)hr13.z       ;; row3 x B3 (noncontiguous)
(rpt1)mad.f16 hr15.x, hr0.w, (r)hr7.z, (r)hr15.x

;; Loop control
cmps.s.eq p0.x, r6.z, 255
add.s r6.z, r6.z, 1
add.s r6.y, r6.y, 4
(rpt3)nop
br !p0.x, #loop_top
```

Result: **200 GFLOPS** (verified correct), limited by 3-wave occupancy (`hregs=31`).

## ir3 Assembler (`ir3asm.py`)

Hand-assembles Adreno a6xx (ir3 ISA) instructions. Uses a compiled OpenCL kernel as
a "donor" for the binary envelope (headers, buffer descriptors, sampler info, constant
tables) and replaces the shader instructions and register counts.

### Key functions

| Function | Description |
|----------|-------------|
| `get_envelope(dev, src)` | Compile OpenCL, return `(lib, img_off, img_sz, reg_off)` |
| `inject(lib, ..., shader, fregs, hregs)` | Replace shader + reg counts in binary |
| `assemble(instr_list)` | Concatenate instruction bytes |
| `disasm(shader_bytes)` | Disassemble via Mesa `ir3_isa_disasm` |
| `MAD_F16(dst, src1, src2, src3, rpt, sy, r)` | Encode `(sy?)(rptN?)mad.f16` |
| `ISAM_F16(dst, coord, tex)` | Encode `isam.1d (f16)(xyzw)` |
| `STG_F16(addr, data_hreg)` | Encode `stg.f16 g[rADDR], hrDATA, 4` |

### Instruction encoding (64-bit, little-endian)

Each instruction is 8 bytes stored as two 32-bit words `[lo, hi]`:

- **hi[31:24]**: Opcode category (0x00=nop/br, 0x20=mov, 0x42=add.s, 0x40=add.f,
  0x63/0x73=mad.f16, 0xa0=isam, 0xc0=stg)
- **hi[23:16]**: Sub-opcode and flags (e.g., `(sy)` sets bit 28 → 0x73 vs 0x63)
- **hi[15:8]**: Repeat count and register flags (`rpt` in bits [6:0], `r` flag in bit 7)
- **hi[7:0]**: Destination register index
- **lo**: Source registers and immediates (layout varies by category)

## Measured Performance

| Configuration | GFLOPS | Notes |
|---------------|--------|-------|
| Pure ALU ceiling (16 rpt3, T128) | 590 | No texture, just MADs |
| Pure texture ceiling (8 isam/iter) | 168 GB/s ≈ 335 GFLOPS equiv | No MADs |
| Compiled 4-row GEMM (unpatched) | 78 | 5 (sy), scalar MADs |
| Patched 4-row GEMM (sy-strip + rpt3) | 190 | 2 (sy), 15 rpt3 |
| Hand-assembled (separate B, hregs=31) | 200 | 1 (sy), 16 rpt3, 3 waves |
| Hand-assembled (hregs=24, WRONG output) | 240 | Register aliasing, 4 waves |
| Direct 4x16 compact persistent coords | 400.5-402.1 | Correct full-output GEMM, `f10 h28`, `loop_instrs=417` |
| Direct 4x16 compact persistent coords + B-first | 421.2-434.8 | Current fastest checked full GEMM |

## Legacy Bottleneck Analysis (200 GFLOPS Kernel)

This older analysis explains the first hand-assembled 200 GFLOPS kernel. The current 420+ kernel bottleneck analysis is in the `How 420 Was Reached` section above.

At 200 GFLOPS with 3 waves and `hregs=31`:

- **Loop body**: 8 coord setup + 8 isam + 17 MAD instrs + 5 loop ctrl = **38 instructions**
- **Effective**: 64 MADs / 38 total = 1.68 MADs/instruction
- **Texture-limited peak**: 168 GB/s / (64 bytes/iter) × 128 FLOPS/iter = **336 GFLOPS**
- **Achieved/peak**: 200/336 = **60%** — the gap is `(sy)` stall time not hidden by 3 waves

### Why 300+ GFLOPS requires 4 waves

With 4 waves, the GPU can switch to another wave during the `(sy)` stall, keeping ALUs
busy. But 4 waves requires `hregs ≤ 24` (24 × 128 × 4 = 12288 = register file size).

The compiled kernel uses `hregs=31` because its accumulator layout spans hreg indices
54-121 (max index 121, requiring ≥31 vec4 slots). A clean layout using indices 32-95
(max 95, requiring 24 slots) fits in 4 waves but needs a **custom store epilogue**.

The store epilogue is difficult because:
1. Full registers (r0-r3) alias half registers (hr0-hr7) in the same physical file
2. The QCOM runtime uses 64-bit buffer addresses requiring `cmps.u.lt` + `sad.s32`
   for carry propagation, which references constant registers `c20.x/c20.y`
3. The address computation and accumulator reduction must be sequenced to avoid
   clobbering results through register aliasing

## Approaches Tried

| Approach | Result | Why |
|----------|--------|-----|
| Strip `(sy)` + rpt3 patching | 190 GFLOPS | Baseline, 2.4x over compiled |
| Separate B texture registers | 200 GFLOPS | Single `(sy)`, 3 waves |
| Remove coord nops | +5 GFLOPS | Nops not needed between mov and isam |
| Fast B coords (increment vs recompute) | Same | Saves instructions but not cycles |
| 8-row kernel (2 waves) | 53 GFLOPS | Too few waves, 4 `(sy)` after patching |
| Software pipelining (double buffer) | N/A | Requires hregs>31 for double A+B, ≤2 waves |
| Interleaved B (4x sy) | 84 GFLOPS | 4 `(sy)` stalls kill throughput |
| 2x K-unroll | GPU hang | Immediate overflow (256 > 8-bit) in CMPS |
| Clean acc layout + custom epilogue | Close | Full/half reg aliasing in epilogue |
| hregs=24 with compiled epilogue | 240 GFLOPS wrong | Acc indices > 95 alias across fibers |
| Local-memory staging | 99 GFLOPS | Barriers/local-memory path are slower than direct texture fetch here |
| Buffer/global loads | 87 GFLOPS | `ldg.f16` path measured far below texture throughput |
| Compiler 4x2 col tile | 47 GFLOPS | Higher arithmetic intensity, but register allocation destroys `(rpt3)` MAD packing |
| Hand 4x2 col tile, 4 partial accs | 204 GFLOPS wrong | Intended 12 isam + 32 rpt3 loop, custom epilogue still writes partial output |
| Hand 4x2 direct acc, hregs=24 | 247 GFLOPS wrong | Faster occupancy, but repeated accumulator dependencies produce NaNs/infs |
| `shfl.rdown.u32` A broadcast probe | 9.0 G lane-shuffles/s | Too slow to replace texture ingress |
| `quad_shuffle.brcst.u32` probe | 22.7 G lane-broadcasts/s | Fast enough for quad-level A sharing on paper |
| 4x2 direct baseline, T128 | 249.6 GFLOPS wrong | 61-instruction loop, 32 `(rpt3)` MADs |
| 4x2 quad-A, 8 scalar qbc | 210.9 GFLOPS wrong | Branch + 8 broadcasts cost more than saved A ingress |
| 4x2 quad-A, 4 `(xy)` qbc | 225.4 GFLOPS wrong | Wrmask cuts qbc count but still below baseline |
| 4x2 quad-A, 2 `(xyzw)` qbc | 232.9 GFLOPS wrong | Best quad-A result so far, still slower than baseline |

### Direct Texture Bandwidth Sweep

`qcom_texture_bw.py` measures logical half4 `isam.1d` bytes issued by a hand shader.
Each load is 8 bytes. The best stable point measured on tc3 is ~148 GB/s.

| Threads | Loads/K step | hregs | waves | GB/s | Notes |
|---------|--------------|-------|-------|------|-------|
| 128 | 4 | 20 | 4 | 96.9 | Too few independent loads per sync |
| 128 | 8 | 24 | 4 | 127.4 | 4-wave 4x1-like load count |
| 128 | 12 | 28 | 3 | 143.5 | Good balance |
| 128 | 16 | 32 | 3 | 75.1 | Stable slow point; not enough load depth after occupancy drop |
| 128 | 20 | 36 | 2 | 72.1 | Stable slow point |
| 128 | 24 | 40 | 2 | 144.5 | Recovers with deeper load stream |
| 128 | 28 | 44 | 2 | 146.4 | Near roof |
| 128 | 32 | 48 | 2 | 147.8 | Best measured |

If the ALU target is 717 GFLOPS, the texture path requires arithmetic intensity
`717 / 147.8 = 4.85 FLOP/byte`. With the 590 GFLOPS sustained ALU number, the
requirement is `590 / 147.8 = 3.99 FLOP/byte`.

For an `R x C` per-thread tile, where `C` is the number of col4 output vectors:

`AI = 32*R*C / (8*R + 32*C) = 4*R*C / (R + 4*C)`.

This explains why widening only columns helps slowly:

| Tile | AI |
|------|----|
| 4x2 | 2.67 |
| 4x8 | 3.56 |
| 8x2 | 4.00 |
| 8x4 | 5.33 |
| 16x2 | 5.33 |

So 4x8 cannot feed a 717 GFLOPS target from the measured texture path. 8x4 or
16x2 is the first class of tiles with enough texture arithmetic intensity.

## Current 4x2 Intensity Experiment

`qcom_intensity_gemm.py` is an experimental hand-assembled 4-row x 2-col4 tile:

- Per K iteration: 4 A `isam` + 8 B `isam` = 96 bytes/thread
- Work per K iteration: 8 output half4 vectors x 4 K lanes = 128 MADs = 256 FLOPs/thread
- Texture roof: `168 GB/s / 96 bytes * 256 FLOPs` = **448 GFLOPS**
- The loop assembles as 12 `isam`, 32 `(rpt3)mad.f16`, one `(sy)`-bearing MAD, plus loop/control overhead.

Important pitfalls found while building this:

1. `BR(offset)` is relative to the branch instruction, not the next instruction.
   The old `loop_start - loop_end - 1` form jumps back one instruction too far.
2. `(rpt3)mov.f16f16 hrX.x, hrX.x` does **not** broadcast an immediate to `xyzw`.
   Use `mov imm hrX.x` then `mov hrX.y, hrX.x (rpt2)`, or copy from a known scalar into a different destination base.
3. The `SAD_S32` encoding only matched the observed odd component forms initially.
   Using `r6.x` decoded as `(neg)r6.y`; use/check disassembly for every new source register.
4. Patching `shlg` from immediate 5 to 6 is not a safe way to compute `gid.x*64 + lane`.
   Use the raw group id (`r51.w`) and integer adds, then refresh duplicated B coordinate registers.
5. Direct accumulation into one output vector is too dependent: updating the same accumulator four times inside one loop iteration produced NaNs/infs even though it lowers `hregs` to 24.
6. The custom store epilogue is still not correct. With all-one inputs, row 0 starts correctly but most output locations remain zero, so the 4x2 GFLOPS numbers are throughput probes only.

## Subgroup / Quad Broadcast Findings

`extra/gemm/qcom_shfl_probe.py` tests register-to-register data movement across
fibers using the hand assembler.

Measured on tc3:

| Operation | Result | Notes |
|-----------|--------|-------|
| `shfl.rdown.u32` immediate 1 | Works, ~9.0 G lane-ops/s | Other tested immediates/register xor read back as zero in the current probe |
| `quad_shuffle.brcst.u32` | Works, ~22.7 G lane-ops/s | Requires cat5 FULL bit set for u32 sources; supports wrmask `(xy)`/`(xyzw)` |
| `getfiberid.u32` | Hangs in injected envelope | Do not use in GEMM kernels until the required envelope/control setup is understood |
| Simple hand divergent `br` | Not reliable | Uniform loop branch encoding is not enough for divergent control flow |
| Compiler image branch | Emits `br !p0.x` around `isam` plus `(ss)(jp)` join target | Use this pattern before hand-assembling conditional A loads |

Implication: full-subgroup `shfl` is not the right ingress path. Quad broadcast is
fast enough as an instruction by itself, but the first 4x2 GEMM integration is
slower than the direct 4x2 baseline because the branch/join and broadcast
instructions reduce MAD issue density.

Arithmetic intensity if quad-level A sharing works:

| Tile | Texture bytes/thread/K | FLOPs/thread/K | Intensity | Texture roof @168 GB/s |
|------|------------------------|----------------|-----------|------------------------|
| Current 4x1 | 64 | 128 | 2.00 FLOP/B | 336 GFLOPS |
| 4x1 + quad A sharing | 40 | 128 | 3.20 FLOP/B | 538 GFLOPS |
| Current 4x2 | 96 | 256 | 2.67 FLOP/B | 448 GFLOPS |
| 4x2 + quad A sharing | 72 | 256 | 3.56 FLOP/B | 597 GFLOPS |

Quad broadcast itself is not the limiting roof for 4x1: 2 `(xyzw)` quad broadcasts
per K iteration gives roughly `22.7 / 2 * 128 = 1453 GFLOPS` of broadcast capacity.
The limiting issue is the extra loop instructions. In 4x2 direct mode, the loop
grew from 61 to 70 instructions while keeping the same 32 `(rpt3)` MADs, so static
MAD density fell from `128/61 = 2.10` to `128/70 = 1.83` MADs/instruction. Even if
the divergent branch suppresses 3/4 of A texture lanes, this does not compensate
at the 4x2 tile size.

Next implication: do not use quad-A sharing for 4x2. If this path is tried again,
it needs a wider in-register tile where the 2 qbc + branch/join overhead is
amortized across more B columns/MADs, or a way to suppress A loads without a
divergent branch sequence.

## Key ISA Details

### `(sy)` — Texture Sync

Stalls until all pending texture results have arrived. Costs ~80-100 cycles per
occurrence. With 4 waves, other waves execute during the stall. With 3 waves,
the stall is only partially hidden.

### `(rpt3)mad.f16` — Packed 4x MAD

Executes 4 MAD operations in a single instruction slot. Requires consecutive
`dst`, `src2`, `src3` registers. The `(r)` flag enables auto-increment on
`src2` and `src3`. Throughput: 1 `(rpt3)` per cycle → 4 MADs/cycle/ALU.

### `isam.1d (f16)(xyzw)` — Integer-Sampled Texture Fetch

Reads a half4 from an image using integer coordinates packed in a full register pair.
Latency ~100 cycles. Multiple isam can be pipelined (issued back-to-back); `(sy)`
waits for all of them.

### `shlg` / `shrm` — Shift with Merge

Used for packing workgroup/thread IDs into coordinate registers.
`shlg(imm, src1, src2)` ≈ `(src1 << imm) | (src2 & ((1<<imm)-1))`.

### `stg.f16` — Global Store (FP16)

`stg.f16 g[rADDR], hrDATA, 4` stores 4 consecutive half-registers (8 bytes) to the
address in a full register pair. The data hreg index in the encoding is `hreg * 2`
(byte offset within the register file).

### `quad_shuffle.brcst` — Quad Register Broadcast

`quad_shuffle.brcst (u32)(x)rD, rS, rI` broadcasts one source lane inside a 4-lane
quad. For full-width types the cat5 FULL bit must be set; otherwise Mesa disassembles
the sources as half registers. The cat5 wrmask works: `(xy)` and `(xyzw)` forms
disassemble and run, allowing two A half4 rows to be broadcast with one u32 `(xyzw)`
instruction. Measured throughput is ~22.7 G lane-broadcasts/s.

### `shfl` — Subgroup Shuffle

`shfl.rdown.u32` encodes and executes, but measured throughput is only ~9.0 G
lane-shuffles/s on this device. That is below the texture-ingress rate it would need
to replace, so it is not the preferred A broadcast primitive.

## Files

| File | Description |
|------|-------------|
| `extra/gemm/ir3asm.py` | ir3 instruction assembler + binary envelope injection |
| `extra/gemm/qcom_gemm.py` | Compiled GEMM + binary patching benchmark |
| `extra/gemm/qcom_asm_gemm.py` | Hand-assembled GEMM test suite (ALU, load, full) |
| `extra/gemm/qcom_shfl_probe.py` | `shfl`, `quad_shuffle.brcst`, and branch/join probes |
| `extra/gemm/qcom_texture_bw.py` | Direct hand-assembled `isam.1d` texture GB/s benchmark |
