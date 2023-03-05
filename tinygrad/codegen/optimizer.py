from tinygrad.helpers import prod
from tinygrad.runtime.ops_gpu import CL
import pyopencl as cl

def optimize_local_workgroup(prg, rawbufs):
  MAX_WORKGROUP = CL.cl_ctx.devices[0].max_work_group_size
  args = [prg.global_size, prg.local_size]
  potential_locals = [None, tuple(args[1])] if args[1] is not None else [None]

  # NOTE: if args[1] is not None, it may use local variables and you shouldn't change this
  if args[1] is None and len(args[0]) == 1:
    for l1 in [args[0][0], 1, 4, 16, MAX_WORKGROUP//4, MAX_WORKGROUP]:
      potential_locals.append((l1,))

  if args[1] is None and len(args[0]) == 2:
    for l2 in [1, 4, 16, MAX_WORKGROUP//4, MAX_WORKGROUP]:
      potential_locals.append((min(MAX_WORKGROUP, args[0][0]), l2))

  if args[1] is None and len(args[0]) == 3:
    for l2 in [16,args[0][1],MAX_WORKGROUP]:
      for l3 in [4,16,args[0][2],MAX_WORKGROUP]:
        for l1 in [max(1, MAX_WORKGROUP//(l2*l3)), args[0][0], 4, 16, MAX_WORKGROUP]:
          if l1 > args[0][0] or l2 > args[0][1] or l3 > args[0][2]: continue
          potential_locals.append((l1, l2, l3))

  best, choice = None, None
  for local_args in potential_locals:
    if local_args is not None and prod(local_args) > MAX_WORKGROUP: continue
    prg.local_size = local_args
    try:
      et = min([prg(rawbufs, wait=True, silent=True) for _ in range(3)])
    except (cl.LogicError, cl.RuntimeError):
      continue
    if best is None or et < best:
      best = et
      choice = local_args
  prg.local_size = choice
  return best

opt_key = {}
def optimize(runtime_type, codegen_type, ast, output_buffer):
  def get():
    k = codegen_type(ast, output_buffer)
    k.process()
    k.hand_coded_optimizations(False)
    return k
  
  def apply(k, axis):
    k.shift_to(axis, min(4, k.full_shape[axis]))
    k.upcast()
    k.simplify_ones()

  k = get()
  if k.key not in opt_key:
    print(f"opt {k.full_shape}")
    axes = []

    for _ in range(3):
      ops = None
      best, new_axis = None, None
      global_size, local_size = None, None
      valid_axes = [None]
      while len(valid_axes):
        i = valid_axes.pop(0)
        k = get()
        for axis in axes: apply(k, axis)
        if i is None:
          valid_axes += [i for i,s in enumerate(k.full_shape) if s < 4 or s%4 == 0 and not k.group_for_reduce and k.first_reduce >= 2]
          print(k.full_shape, valid_axes)
        else:
          apply(k, i)
        colorshape = k.colorshape()
        try:
          prg = k.codegen().build(runtime_type)
          if i == None: ops = prg.op_estimate
          rawbufs = prg.lower(k.bufs)
          et = optimize_local_workgroup(prg, rawbufs)
          #et = min([prg(rawbufs, wait=True, silent=True) for _ in range(3)])
          if best is None or et < best:
            best = et
            new_axis = i
            global_size, local_size = prg.global_size, prg.local_size
        except Exception as e:
          et = float('nan')
          raise e
        print(f"{str(axes)+' '+str(i):8s} : {colorshape} : {et*1e6:7.2f} us : {str(global_size):18s} {str(local_size):12s} ")
      axes += [] if new_axis is None else [new_axis]
      if new_axis is None: break

    opt_key[k.key] = axes
    print(f"best is {axes} in {best*1e6:.2f} us with {ops/best*1e-9:.2f} GFLOPS")
    k = get()

  for axis in opt_key[k.key]: apply(k, axis)
  return k
