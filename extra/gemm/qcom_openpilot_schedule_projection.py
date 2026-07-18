#!/usr/bin/env python3
"""Reschedule independent texture addresses in the dominant vision projection."""
import argparse, pickle, struct

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops
from extra.gemm.ir3asm import ADD_S, BR, ISAM_F32, MOV_F32, NOP
from extra.gemm.qcom_ir3_matmul_patch import plain_name

TARGET="r_32_192_4_4_64_4"
FIRST_CONV_TARGET="r_64_32_16_4_4_6_3_3_4"
FORWARD_STYLE={TARGET,"r_32_64_4_4_64_4","r_8_384_4_4_128_4"}
GAP_STYLE={"r_512_16_4_4_16_4","r_512_48_4_4_16_4","r_128_32_4_4_32_4","r_128_96_4_4_32_4"}
INVERSE_W_STYLE={"r_512_16_4_4_48_4","r_128_32_4_4_96_4"}
INVERSE_STYLE={"r_32_64_4_4_192_4"}
TARGETS=FORWARD_STYLE|GAP_STYLE|INVERSE_W_STYLE|INVERSE_STYLE|{FIRST_CONV_TARGET}


def schedule_first_conv(lib:bytes) -> bytes:
  image_offset, image_size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  instrs=[lib[i:i+8] for i in range(image_offset,image_offset+image_size,8)]
  if len(instrs) != 262: raise RuntimeError(f"expected 262 first-conv instructions, got {len(instrs)}")
  # Use registers that the subsequent texture loads overwrite, allowing all
  # eight independent input/weight coordinates to precede the texture reads.
  addresses=[]
  for index,(register,offset) in enumerate(zip(("r0","r2","r3","r4"),(-36,-24,-12,0))):
    addresses.append(MOV_F32(f"{register}.x","r16.w",ss=index > 0) if offset == 0 else
                     ADD_S(f"{register}.x","r16.w",offset,ss=index > 0))
    addresses.append(MOV_F32(f"{register}.y","r16.z"))
  addresses.extend(instrs[i] for i in (48,51,54,57))
  loads=[ISAM_F32(dst,f"{coord}.x",tex=0) for dst,coord in zip(("r7.x","r6.x","r1.x","r0.x"),("r0","r2","r3","r4"))]
  loads.extend(ISAM_F32(dst,coord,tex=1) for dst,coord in zip(("r2.x","r3.x","r4.x","r5.x"),("r8.x","r8.z","r9.x","r9.z")))
  out=instrs[:32]+addresses+loads+instrs[60:86]
  out.append(BR(31-len(out)))
  out.extend(instrs[87:92])
  out.append(BR(26-len(out)))
  out.extend(instrs[93:99])
  out.append(BR(24-len(out)))
  out.extend(instrs[100:])
  out.extend([NOP()]*(len(instrs)-len(out)))
  if len(out) != len(instrs): raise RuntimeError(f"scheduled first conv has {len(out)} instructions")
  return lib[:image_offset]+b"".join(out)+lib[image_offset+image_size:]


def schedule_native_f16(lib:bytes) -> bytes:
  image_offset, image_size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  instrs=[lib[i:i+8] for i in range(image_offset,image_offset+image_size,8)]
  if len(instrs) != 222: raise RuntimeError(f"expected 222 native-FP16 instructions, got {len(instrs)}")
  addresses=(21,24,27,30,33,40,47,54)
  loads=(23,26,29,32,35,42,49,56)
  mads=tuple(range(36,40))+tuple(range(43,47))+tuple(range(50,54))+tuple(range(57,61))
  out=instrs[:21]+[instrs[i] for i in addresses]+[instrs[i] for i in loads]+[instrs[i] for i in mads]+instrs[61:67]
  out.append(BR(21-len(out)))
  out.extend(instrs[68:])
  out.extend([NOP()]*(len(instrs)-len(out)))
  if len(out) != len(instrs): raise RuntimeError(f"scheduled native FP16 has {len(out)} instructions")
  return lib[:image_offset]+b"".join(out)+lib[image_offset+image_size:]


def schedule_loads(lib:bytes, name:str) -> bytes:
  image_offset, image_size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  instrs=[lib[i:i+8] for i in range(image_offset,image_offset+image_size,8)]
  if name == TARGET and len(instrs) == 222: return schedule_native_f16(lib)
  if len(instrs) < 160: raise RuntimeError(f"expected projection shader, got {len(instrs)} instructions")
  # The compiler emits address, rpt5 nop, texture-read eight times. Calculate
  # every independent address first, then issue the reads as one contiguous run.
  if name in FORWARD_STYLE: start,body_end=20,66
  elif name in GAP_STYLE: start,body_end=26,84
  elif name in INVERSE_W_STYLE: start,body_end=19,76
  elif name in INVERSE_STYLE: start,body_end=16,73
  else: raise RuntimeError(f"unsupported projection {name}")
  address_indices=tuple(start+3*i for i in range(8))
  load_indices=tuple(start+3*i+2 for i in range(8))
  out=instrs[:start]+[instrs[i] for i in address_indices]+[instrs[i] for i in load_indices]+instrs[start+24:body_end]
  branch_index=len(out)
  out.append(BR(start-branch_index))
  out.extend(instrs[body_end+1:])
  out.extend([NOP()]*(len(instrs)-len(out)))
  if len(out) != len(instrs): raise RuntimeError(f"scheduled image has {len(out)} instructions")
  return lib[:image_offset]+b"".join(out)+lib[image_offset+image_size:]


def patch_projection(model) -> int:
  outer=model.captured.linear.src[0]
  batch=outer.src[0].src[0].src
  new_batch=[]
  cache={}
  replaced=0
  for call in batch:
    name=plain_name(call.src[0].arg.name) if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM else ""
    if name in TARGETS:
      program=call.src[0]
      old=program.src[3].arg
      if old not in cache: cache[old]=schedule_first_conv(old) if name == FIRST_CONV_TARGET else schedule_loads(old,name)
      program=program.replace(src=program.src[:3]+(program.src[3].replace(arg=cache[old]),))
      call=call.replace(src=(program,*call.src[1:]))
      replaced+=1
    new_batch.append(call)
  model.captured._linear=model.captured.linear.substitute({outer:create_graph_call(new_batch)},walk=True)
  model.captured.__dict__.pop("linear",None)
  return replaced
def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  args=parser.parse_args()
  with open(args.input,"rb") as f:model=pickle.load(f)
  print("patched",patch_projection(model))
  with open(args.output,"wb") as f:pickle.dump(model,f)


if __name__ == "__main__":main()
