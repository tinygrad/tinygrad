# FA Backward Kernel Debug Notes

## ROOT CAUSE FOUND AND FIXED

**Bug**: VGPR v29 (`neg_inf_v`) corruption by compiler register reallocation.

**Details**: The backward kernel stores -inf (0xff800000) in VGPR v29 at startup via inline asm `v_mov_b32<29>(0xff800000)`. This value is read later by the causal masking code (`mov<neg_inf_v>(P_ij)`) at 12 different points throughout the kernel. However, because both the write and reads use **hardcoded register numbers** in inline asm (not compiler-allocated registers with proper constraints), the LLVM compiler cannot track the data dependency between them. The compiler is free to reuse v29 as a temporary between the initial write and subsequent reads, corrupting the -inf sentinel value.

**When v29 gets corrupted to ~0**:
- `mov<neg_inf_v>(P_ij)` fills P with 0 instead of -inf
- `exp2(0) = 1.0` instead of `exp2(-inf) = 0.0`
- Every masked query contributes 1.0 to dV/dK instead of 0.0
- dV error at key position k ≈ number of masked queries at that position

**Error pattern explained** (N=256):
- pos 0: err=0.004 (no masking needed, no corruption)
- pos 16: err≈16 (16 masked queries × 1.0)
- pos 64: err≈64 (64 masked queries × 1.0)
- pos 128: err≈128 (128 masked queries × 1.0)
- pos 192: err≈192 (192 masked queries × 1.0)
- pos 255: err≈5 (in sub-block [0,3] where scale/sub after mask converts 0 to -L, so exp2(-L) ≈ small)

**Fix**: Re-initialize v29 before every masking point:
```cpp
kittens::macros::v_mov_b32<neg_inf_v>(0xff800000);
if constexpr (causal) { ... }
```

**Results after fix**:
- N=256: dK max_err=0.031, dV max_err=0.031 (PASS)
- N=512: dK max_err=0.031, dV max_err=0.031 (PASS)
- N=1024: dK max_err=0.031, dV max_err=0.031 (PASS)

## Why this only affects the backward kernel, not forward

The forward kernel does NOT use the HipKittens `art` (assigned register tile) system or manually-allocated VGPRs. It uses standard register allocation. The backward kernel is the only kernel that uses hardcoded VGPR numbers via inline asm, making it vulnerable to this compiler interaction issue.

## Ruled Out During Investigation
1. `-ffast-math` / `-fhonor-infinities`: Identical results with/without (same binary even)
2. `__attribute__((amdgpu_num_vgpr(29)))`: Causes GPU hang when added back (too few VGPRs)
3. v126/v127 (L_i/delta_i) clobbers: No change (same binary)
4. Compiler optimization levels (-O1/-O2/-O3): Identical results
5. Grid order (H_KV,B,N//BLOCK vs H_KV,N//BLOCK,B): Identical wrong results
6. dQ layout (B,N,H,D vs B,H,N,D): Doesn't affect dK/dV
7. Struct vs direct gl construction: Identical results
8. Argument order/shapes: Verified all correct
9. Forward output, l_vec, delta_vec: All correct
10. swap_layout_inplace: Register-only, no race conditions
11. Original kernel (commit 9cf133a73): Same bug - never worked in tinygrad

## Test Files
- `test_bwd_reinit_v29.py`: **The fix test** - original vs v29-reinit comparison
- `test_bwd_fix_sizes.py`: Tests fix at N=256, 512, 1024
- `test_bwd_nofastmath.py`: Proved -ffast-math is not the issue
- `test_bwd_vgpr.py` / `test_bwd_vgpr2.py`: Tested vgpr attribute and clobber hypotheses
- `test_bwd_original.py` / `test_bwd_original2.py`: Proved bug existed since initial commit
