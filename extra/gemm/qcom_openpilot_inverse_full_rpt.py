#!/usr/bin/env python3
"""Pack all four FP32 accumulator vectors in the OpenPilot inverse projection."""
import argparse, hashlib, pickle, struct

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_openpilot_ir3 import branch as BR, mad_f32 as MAD_F32, mov_f32 as MOV_F32, nop as NOP, plain_name

TARGET="r_32_64_4_4_192_4"
INVERSE_W_TARGETS={"r_512_16_4_4_48_4","r_128_32_4_4_96_4"}
SAFE_DONORS={"d4c281a1","1fe26758","e34e7e58"}


def replace_src2(ins:bytes, src2:int) -> bytes:
  lo,hi=struct.unpack("<II",ins)
  return struct.pack("<II",(lo&0xff00ffff)|(src2<<16),hi)


def replace_low_src(ins:bytes, src:int) -> bytes:
  lo,hi=struct.unpack("<II",ins)
  return struct.pack("<II",(lo&0xffffff00)|src,hi)


def pack_inverse_full(lib:bytes) -> bytes:
  off,size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  instrs=[lib[i:i+8] for i in range(off,off+size,8)]
  if len(instrs)!=175: raise RuntimeError(f"expected 175 inverse instructions, got {len(instrs)}")
  # Move loop control from r13.x into the existing r12.w zero register. This
  # makes r13-r16 four contiguous accumulator vectors without growing the
  # shader's declared register file.
  out=instrs[:11]
  for acc in ("r13.x","r14.x","r15.x","r16.x"): out.append(MOV_F32(acc,"r12.w",rpt=3))
  loop_start=len(out)
  body=list(instrs[16:32])
  body[0]=replace_low_src(body[0],51)   # add r8.z, r12.w, 192
  body[1]=replace_low_src(body[1],51)   # add r9.x, r12.w, 384
  body[2]=replace_src2(body[2],51)      # add r9.z, c28.y, r12.w
  body[3]=replace_low_src(body[3],51)   # mov r10.x, r12.w
  out+=body
  for component,weight in zip("xyzw",("r5.x","r2.x","r3.x","r4.x")):
    out.append(MAD_F32("r13.x",f"r7.{component}",weight,"r13.x",rpt=3,r=True,sy=component=="x"))
  out+=instrs[48:60]
  control=list(instrs[60:65])
  control[0]=replace_low_src(control[0],51)  # increment r12.w
  control[2]=replace_low_src(control[2],51)  # compare r12.w
  control[3]=MOV_F32("r12.w","r0.x")
  out+=control
  out.append(BR(loop_start-len(out)))
  tail=list(instrs[66:131])
  # The first residual moved from r12.w to r13.x; y/z/w were already in r13.
  tail[90-66]=replace_src2(tail[90-66],52)
  out+=tail
  out += [NOP()]*(len(instrs)-len(out))
  if len(out)!=len(instrs): raise RuntimeError(f"packed image has {len(out)} instructions")
  return lib[:off]+b"".join(out)+lib[off+size:]


def with_fregs(lib:bytes, count:int) -> bytes:
  out=bytearray(lib)
  regoff=struct.unpack_from("<I",out,0x34)[0]+0x14
  regs=struct.unpack_from("<I",out,regoff)[0]
  struct.pack_into("<I",out,regoff,(regs&0x80000000)|max(regs&0x7fffffff,count))
  return bytes(out)


def pack_inverse_w_full(lib:bytes) -> bytes:
  off,size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  ins=[lib[i:i+8] for i in range(off,off+size,8)]
  if len(ins) not in (174,178): raise RuntimeError(f"expected 174/178 inverse-W instructions, got {len(ins)}")
  out=ins[:19]+[MOV_F32("r17.x","r12.z",rpt=3)]
  loop_start=len(out)
  out+=ins[19:35]
  for component,weight in zip("xyzw",("r5.x","r2.x","r3.x","r4.x")):
    out.append(MAD_F32("r17.x",f"r7.{component}",weight,"r17.x",rpt=3,r=True,sy=component=="x"))
  out+=ins[51:68]
  out.append(BR(loop_start-len(out)))
  tail=list(ins[69:])
  mapping={50:68,52:69,53:70,54:71}
  first_store=next(i for i,x in enumerate(tail) if struct.unpack_from("<I",x,4)[0]>>24==0xc0)
  for i in range(first_store):
    lo,_=struct.unpack("<II",tail[i])
    src2=(lo>>16)&0xff
    if src2 in mapping: tail[i]=replace_src2(tail[i],mapping[src2])
  out+=tail
  out += [NOP()]*(len(ins)-len(out))
  return with_fregs(lib[:off]+b"".join(out)+lib[off+size:],18)


def patch_model(model,names:set[str]|None=None) -> int:
  outer=model.captured.linear.src[0]
  batch=list(outer.src[0].src[0].src)
  cache,patched={},0
  for index,call in enumerate(batch):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    name=plain_name(call.src[0].arg.name)
    if name not in (names if names is not None else {TARGET}|INVERSE_W_TARGETS): continue
    program=call.src[0]
    old=program.src[3].arg
    # These transforms relocate fixed compiler registers. A different QCOM compiler allocation can
    # have the same instruction count but different live values and must not be patched by index.
    if hashlib.sha1(old).hexdigest()[:8] not in SAFE_DONORS: continue
    if old not in cache:
      cache[old]=pack_inverse_full(old) if name==TARGET else pack_inverse_w_full(old)
    program=program.replace(src=program.src[:3]+(program.src[3].replace(arg=cache[old]),))
    batch[index]=call.replace(src=(program,*call.src[1:]))
    patched+=1
  model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(batch)},walk=True)
  model.captured.__dict__.pop("linear",None)
  return patched


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("input")
  ap.add_argument("output")
  ap.add_argument("--names",help="comma-separated program families")
  args=ap.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_model(model,set(args.names.split(",")) if args.names else None))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__=="__main__":main()
