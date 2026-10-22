# Held-out SQTT timing: 73.6% exact on kernels the model never saw

On 11 random SRAM kernels captured on a 7900 XTX, the mockgpu emulator's RDNA3 SQTT timing
(`test/mockgpu/amd/sqtt_enc.py`) predicts 815 of 1108 instruction deltas to the cycle. None of these kernels
were used to fit the timing rules. The fitted examples in `test/amd/test_sqtt_timing.py` score higher (gemm 52/59)
because the rules were read off them, so 73.6% is the number to quote.

`randprobe.py` builds the kernel from a seed: VALU, SALU, LDS store and load, `s_waitcnt`, `s_nop` and barriers,
on 128 threads. `captures/` holds one real trace per seed (gfx1100, `profile_standard` clocks, SQTT on WGP 3).
`score.py` replays each kernel in the emulator and compares every delta.

```
$ PYTHONPATH=. python3 extra/sqtt/heldout/score.py
  seed1:  76/94  exact (80.9%), mean abs error 29.54 cycles
  seed2:  61/84  exact (72.6%), mean abs error 30.38 cycles
  seed3:  72/104 exact (69.2%), mean abs error 26.40 cycles
  seed4:  72/108 exact (66.7%), mean abs error 26.23 cycles
  seed5:  77/100 exact (77.0%), mean abs error 12.99 cycles
  seed6:  68/90  exact (75.6%), mean abs error 14.82 cycles
  seed7:  81/108 exact (75.0%), mean abs error 25.44 cycles
  seed8:  72/102 exact (70.6%), mean abs error 27.21 cycles
  seed9:  83/112 exact (74.1%), mean abs error 9.86 cycles
 seed10:  68/90  exact (75.6%), mean abs error 30.23 cycles
 seed11:  85/116 exact (73.3%), mean abs error 36.29 cycles
  total: 815/1108 exact (73.6%)
```

## The large misses are DRAM, the in-scope ones are small

Seeds 1-10, wave 0:

| packet | misses | mean error (cycles) | where |
|---|---|---|---|
| IMMEDIATE (`s_waitcnt`) | 41 | 29 | partly waits on the DRAM load below |
| VALUINST | 39 | 6 | VALU issue |
| WAVEEND | 10 | 529 | after the global store (DRAM) |
| SMEM_RD | 10 | 464 | first `s_load` (DRAM) |
| LDS_WR_2, LDS_RD | 10 | 1-2 | LDS |
| SALU, SGMEM_WR_2 | 7 | 2-3 | scalar |

Open gap: hardware traces `v_mul_lo_u32` as `VALUB_4`, the emulator emits `VALUINST`. The generator leaves it out
until that is fixed.

## The test holds every seed at its score

```
DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_heldout.py test/amd/test_sqtt_timing.py
```

`test/amd/test_sqtt_heldout.py` fails if any seed loses an exact delta, so a rule that improves the fitted
examples at the cost of unseen kernels does not land. Checked with a mutant: `VALU_AFTER_LDS_WAIT` 7 -> 9 drops
seeds 2, 8 and 9 below their floor.

## Capturing a new seed

Needs a 7900 XTX and [wavewait](https://github.com/amarbaro/wavewait)'s `tools/sqtt-capture`, with its
`wgp-probe.patch` applied to this checkout:

```
SQTT_SEED=12 SQTT_WGP_PROBE=3 TINYGRAD=$PWD sqtt-capture $PWD/extra/sqtt/heldout/randprobe.py OUT
PYTHONPATH=. python3 extra/sqtt/heldout/score.py OUT/profile_randprobe.pkl
```
