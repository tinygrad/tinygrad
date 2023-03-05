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
  if k.key in opt_key:
    axes = opt_key[k.key]
  else:
    valid_axes = [i for i,s in enumerate(k.full_shape) if s < 4 or s%4 == 0 and not k.group_for_reduce]
    print(f"opt {k.full_shape} with {valid_axes}")

    ops = None
    best, axis = None, None
    for i in [None]+valid_axes:
      k = get()
      if i != None:
        apply(k, i)
      colorshape = k.colorshape()
      try:
        prg = k.codegen().build(runtime_type)
        if i == None: ops = prg.op_estimate
        et = prg(prg.lower(k.bufs), wait=True, silent=True)
        if best is None or et < best:
          best = et
          axis = i
      except Exception as e:
        et = float('nan')
        raise e
      print(f"{str(i):8s} : {colorshape} : {et*1e6:7.2f} us")
    axes = [] if axis is None else [axis]
    opt_key[k.key] = axes

    print(f"best is {axes} in {et*1e6:.2f} us with {ops/et*1e-9:.2f} GFLOPS")
    k = get()

  for axis in axes: apply(k, axis)
  return k
