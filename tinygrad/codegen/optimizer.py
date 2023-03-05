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

    for _ in range(2):
      ops = None
      best, new_axis = None, None
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
          et = prg(prg.lower(k.bufs), wait=True, silent=True)
          if best is None or et < best:
            best = et
            new_axis = i
        except Exception as e:
          et = float('nan')
          raise e
        print(f"{str(axes)+' '+str(i):8s} : {colorshape} : {et*1e6:7.2f} us")
      axes += [] if new_axis is None else [new_axis]
      if new_axis is None: break

    opt_key[k.key] = axes
    print(f"best is {axes} in {et*1e6:.2f} us with {ops/et*1e-9:.2f} GFLOPS")
    k = get()

  for axis in opt_key[k.key]: apply(k, axis)
  return k
