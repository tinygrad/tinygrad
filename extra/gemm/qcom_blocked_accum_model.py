#!/usr/bin/env python3
"""Patch openpilot's 4x16 FP32 GEMM with FP16 K4 partials and FP32 totals."""
import argparse, itertools, pickle, struct

from tinygrad.engine.jit import create_graph_call
from tinygrad.uop.ops import Ops, UOp
from extra.gemm.ir3asm import BR, CMPS_S_EQ, COV_F16F32, ISAM_F16, JUMP, MAD_F16, MAD_F32, MOV_F32, MOV_H_IMM, MOV_S32, NOP, inject
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def blocked_image(image:bytes, block:int=1, direct_branch:bool=False, outer_iters:int|None=None, no_back_edge:bool=False) -> bytes:
  instrs = [image[i:i+8] for i in range(0, len(image), 8)]
  if len(instrs) != 349: raise ValueError(f"expected 349 instructions, got {len(instrs)}")
  if block not in (1, 2, 4): raise ValueError(f"block must be 1, 2, or 4, got {block}")
  # The compiler's loop is 47..100. Preserve its coordinate arithmetic and
  # loop control, but sample native half vectors into a disjoint register bank.
  # Each partial vector contains four output columns. Accumulate four scalar K
  # terms per substep. Several substeps can share one partial before promotion.
  body = [MOV_H_IMM(f"hr{34+row}.x", 0, rpt=3) for row in range(4)]
  for substep in range(block):
    # Drain the preceding half MADs before reusing their texture-source
    # registers. Reissuing ISAM into a still-live half register can deadlock.
    if substep: body += [MOV_F32("r0.x", "r0.x", sy=True), NOP(rpt=2)]
    body += instrs[47:55]
    for dst, coord in zip(("hr26.x", "hr27.x", "hr28.x", "hr29.x"), ("r0.x", "r1.x", "r2.x", "r3.x")):
      body.append(ISAM_F16(dst, coord, 1, 1))
    body += instrs[63:71]
    for dst, coord in zip(("hr30.x", "hr31.x", "hr32.x", "hr33.x"), ("r4.x", "r5.x", "r6.x", "r7.x")):
      body.append(ISAM_F16(dst, coord, 0, 0))
    first = True
    for kk in range(4):
      for row in range(4):
        body.append(MAD_F16(f"hr{34+row}.x", 4*(30+row)+kk, f"hr{26+kk}.x", f"hr{34+row}.x",
                              rpt=3, sy=first, r=True))
        first = False
    # Keep the compare even between substeps: besides setting p0 it provides
    # the latency slot needed by add r0.x -> mov r12.w. The final compare below
    # overwrites p0 before loop control.
    if substep != block-1: body += instrs[95:100]
  # r4 is dead after all texture operations and supplies scalar 1.0 to vector
  # MADs, giving FP32 total += promoted_partial without a separate add opcode.
  body.append(MOV_S32("r4.x", 0x3f800000))
  for row in range(4): body.append(COV_F16F32(f"r{row}.x", f"hr{34+row}.x", sy=(row == 0), rpt=3, r=True))
  for row in range(4): body.append(MAD_F32(f"r{8+row}.x", "r4.x", f"r{row}.x", f"r{8+row}.x", rpt=3, r=True))
  loop_limit = 95 if outer_iters is None else outer_iters*block-1
  body += instrs[95:97] + [CMPS_S_EQ("r12.w", loop_limit, nop=1)] + instrs[98:100]

  out = instrs[:47] + body
  if no_back_edge:
    pass
  elif block == 1 or direct_branch:
    out.append(BR(47-len(out), inv=True))
  else:
    # A6xx conditional branches have a much shorter reliable backward range
    # than unconditional jumps. Branch past a long-range jump when complete.
    branch_index = len(out)
    out += [BR(2, inv=False), JUMP(47-(branch_index+1))]
  out += instrs[101:]
  while len(out) > len(instrs) and out[-1] == NOP(): out.pop()
  if len(out) > len(instrs): raise ValueError(f"patched shader grew beyond envelope: {len(out)} > {len(instrs)}")
  out += [NOP()] * (len(instrs)-len(out))
  return b"".join(out)


def patch_lib(lib:bytes, block:int, direct_branch:bool=False, outer_iters:int|None=None, no_back_edge:bool=False) -> bytes:
  image_off = struct.unpack_from("<I", lib, 0xc0)[0]
  image_size = struct.unpack_from("<I", lib, 0x100)[0]
  reg_off = struct.unpack_from("<I", lib, 0x34)[0]
  image = blocked_image(lib[image_off:image_off+image_size], block, direct_branch, outer_iters, no_back_edge)
  return inject(lib, image_off, image_size, reg_off, image, fregs=13, hregs=38)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--global-size", default="12,8,1")
  parser.add_argument("--block", type=int, default=1)
  parser.add_argument("--direct-branch", action="store_true")
  parser.add_argument("--outer-iters", type=int, help="diagnostic loop limit; normal model execution requires 96/block iterations")
  parser.add_argument("--no-back-edge", action="store_true", help="diagnostic: execute one outer body with no loop branch")
  args = parser.parse_args()
  target_global = tuple(int(x) for x in args.global_size.split(","))
  with open(args.input, "rb") as f: jit = pickle.load(f)
  slots = [x.arg.slot for x in jit.captured.linear.toposort()
           if x.op is Ops.BUFFER and hasattr(x.arg, "slot") and x.arg.slot >= 0]
  UOp.unique_num = itertools.count(max(slots, default=-1)+1)
  outer = jit.captured.linear.src[0]
  batch = outer.src[0].src[0].src
  cache, replacements = {}, {}
  for call in batch:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    program = call.src[0]
    if plain_name(program.arg.name) != "gemm_h" or tuple(program.arg.global_size) != target_global: continue
    old_lib = program.src[3].arg
    new_lib = cache.setdefault(old_lib, patch_lib(old_lib, args.block, args.direct_branch, args.outer_iters, args.no_back_edge))
    replacements[call] = call.replace(src=(program.replace(src=program.src[:3]+(program.src[3].replace(arg=new_lib),)), *call.src[1:]))
  if not replacements: raise ValueError(f"no gemm_h calls with global size {target_global}")
  new_outer = create_graph_call([replacements.get(call, call) for call in batch])
  jit.captured._linear = jit.captured.linear.substitute({outer:new_outer}, walk=True)
  jit.captured.__dict__.pop("linear", None)
  with open(args.output, "wb") as f: pickle.dump(jit, f)
  print(f"patched {len(replacements)} calls across {len(cache)} binaries with block={args.block} direct_branch={args.direct_branch}")


if __name__ == "__main__": main()
