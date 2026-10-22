# Held-out SRAM probe: a seeded random mix of VALU, SALU, LDS, waitcnt, s_nop and barriers on 128 threads,
# never used to fit sqtt_enc.py. Seed via SQTT_SEED. Capture on a 7900 XTX with wavewait's tools/sqtt-capture
# (github.com/amarbaro/wavewait, needs its wgp-probe.patch):  SQTT_SEED=3 SQTT_WGP_PROBE=3 sqtt-capture $PWD/extra/sqtt/heldout/randprobe.py OUTDIR
import os, random
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

# v_mul_lo_u32 left out: hw traces it as VALUB_4, the emulator as VALUINST (packet-kind gap, finding 1)
SEED = int(os.environ.get("SQTT_SEED", "0"))

def body(rng):
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  out, pending = [], 0
  for _ in range(rng.randint(24, 40)):
    k = rng.choice("vvvvssllwnb")
    a, b, d = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    if k == "v": out.append(rng.choice([v_add_nc_u32_e32(v[d], v[a], v[b]), v_sub_nc_u32_e32(v[d], v[a], v[b]),
                                         v_add_f32_e32(v[d], v[a], v[b]), v_fmac_f32_e32(v[d], v[a], v[b])]))
    elif k == "s": out.append(rng.choice([s_add_u32(s[4+d%4], s[4+a%4], s[4+b%4]), s_mov_b32(s[4+d%4], a)]))
    elif k == "l":  # conflict-free b32: lane address v1 = 4*lane (+ 512 per slot)
      if rng.random() < .5: out.append(ds_store_b32(addr=v[1], data0=v[a]))
      else: out.append(ds_load_b32(vdst=v[d], addr=v[1]))
      pending += 1
    elif k == "w" and pending: out.append(w); pending = 0
    elif k == "n": out.append(s_nop(rng.randint(0, 7)))
    elif k == "b": out += [w, s_barrier()]; pending = 0
  return out + [w]

def randprobe(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(128, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((2048,), dtypes.uint8, 0, AddrSpace.LOCAL)
  init = [s_load_b64(s[0:1], s[0:1], soffset=NULL), s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
          v_lshlrev_b32_e32(v[1], 2, v[0])] + [v_mov_b32_e32(v[i], i) for i in range(2, 10)] + \
         [s_mov_b32(s[i], i) for i in range(4, 8)]
  insts = init + body(random.Random(SEED)) + [global_store_b32(addr=v[1], data=v[2], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo(f"probe_rand{SEED}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=randprobe)[0].realize()
