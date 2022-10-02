#!/usr/bin/env python3
import os
import sys
import time
import struct
import numpy as np

BASEDIR = os.path.dirname(os.path.abspath(__file__))+"/"
THNEED_KERNELS = "../../selfdrive/modeld/thneed/kernels/"

def load_thneed_model(fn="model.thneed", float32=False, replace=None):
  import pyopencl as cl
  devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
  if len(devices) == 0:  # settle for CPU
    devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
  ctx = cl.Context(devices=devices[0:1])
  q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  mf = cl.mem_flags
  image_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT if float32 else cl.channel_type.HALF_FLOAT)
  image_fmt_32 = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

  import json
  import traceback
  with open(fn if (fn[0] == "/") else (BASEDIR+"../models/"+fn), "rb") as f:
    json_len = struct.unpack("I", f.read(4))[0]
    jdat = json.loads(f.read(json_len).decode('latin_1'))
    weights = f.read()

  # jdat = ['kernels', 'objects', 'programs']
  prgs = {}
  for k,v in jdat['programs'].items():
    print("building", k)
    try:
      prgs[k] = cl.Program(ctx, v).build().__getattr__(k)
    except Exception:
      print("FAILED", k)
      traceback.print_exc()
      exit(0)

  bufs = {'\x00\x00\x00\x00\x00\x00\x00\x00': None}
  bufs_loaded = {}
  ptr = 0
  for o in jdat['objects']:
    #print(o)
    if o['needs_load']:
      nptr = ptr + o['size']
      o['data'] = weights[ptr:nptr]
      ptr = nptr

    if o['arg_type'] == "image2d_t" or o['arg_type'] == "image1d_t":
      tfmt = image_fmt_32 if 'float32' in o and o['float32'] else image_fmt
      if o['arg_type'] == "image2d_t":
        if 'buffer_id' in o and o['height'] == 1 and not bufs_loaded[o['buffer_id']]:
          # hack: use a image1d since we can back that with a buffer
          buf = cl.Image(ctx, mf.READ_WRITE, tfmt, shape=(o['width'],), buffer=bufs[o['buffer_id']])
        else:
          # buffer isn't supported in image2d, copy buffer into image
          if 'buffer_id' in o and bufs_loaded[o['buffer_id']]:
            arr = np.zeros(bufs[o['buffer_id']].size // 2, dtype=np.float16)
            cl.enqueue_copy(q, arr, bufs[o['buffer_id']])
            buf = cl.Image(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, tfmt,
              shape=(o['width'], o['height']), pitches=(o['row_pitch'],), hostbuf=arr)
          elif o['needs_load']:
            buf = cl.Image(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, tfmt,
              shape=(o['width'], o['height']), pitches=(o['row_pitch'],), hostbuf=o['data'])
          else:
            buf = cl.Image(ctx, mf.READ_WRITE, tfmt, shape=(o['width'], o['height']))
      if o['arg_type'] == "image1d_t":
        assert not o['needs_load']
        assert not bufs_loaded[o['buffer_id']]
        buf = cl.Image(ctx, mf.READ_WRITE, tfmt, shape=(o['width'],), buffer=bufs[o['buffer_id']])
    else:
      if 'data' in o:
        buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=o['data'])
      else:
        # zero out buffers
        buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b'\x00'*o['size']*(2 if float32 else 1))
      
    bufs[o['id']] = buf
    bufs_loaded[o['id']] = 'data' in o

  # load binaries
  for o in jdat['binaries']:
    nptr = ptr + o['length']
    prgs[o['name']] = cl.Program(ctx, ctx.devices, [weights[ptr:nptr]]).build().__getattr__(o['name'])
    ptr = nptr

  inputs, vnum, vision, outputs = [], [], [], []

  for k in jdat['inputs'] if 'inputs' in jdat else []:
    print(f"new style input {k['name']} with size {k['size']}")
    inputs.append(bufs[k['buffer_id']])
  for k in jdat['outputs'] if 'outputs' in jdat else []:
    outputs.append(bufs[k['buffer_id']])

  # old style inputs
  for i,k in enumerate(jdat['kernels']):
    if k['name'] == 'zero_pad_image_float':
      inputs.append(bufs[k['args'][1]])

    # vision model
    if k['name'] == 'zero_pad_image_half':
      vision.append(bufs[k['args'][1]])
      vnum.append(i)

    if k['name'] == 'image2d_to_buffer_float':
      outputs.append(bufs[k['args'][2]])

    k['args_name'] = []
    prg = prgs[k['name']]
    for i,arg in enumerate(k['args']):
      try:
        k['args_name'].append(prg.get_arg_info(i, cl.kernel_arg_info.NAME))
      except cl.RuntimeError:
        k['args_name'].append("<UNKNOWN>")

  vision = vision[0:1]
  vnum = vnum[0] if len(vnum) >= 1 else None

  def runner(inp=[], policy_only=False, vision_only=False, debug=False):
    kernels = []
    total_runtime = 0
    # [2048, 8, 32, 1572864]
    real_inputs = inputs[0:3]+vision if policy_only else inputs
    for a,b in zip(real_inputs, inp):
      if debug:
        print(a.size, b.size*b.itemsize)
      #assert a.size == (b.size * b.itemsize) or float32
      cl.enqueue_copy(q, a, np.array(b, dtype=np.float16 if len(vision) > 0 and a == vision[0] else np.float32))

    #jdat['kernels'] = jdat['kernels'][0:8]

    seen_output = set(real_inputs)
    for k in jdat['kernels'][0:3]+jdat['kernels'][vnum:] if policy_only else (jdat['kernels'][:vnum] if vision_only else jdat['kernels']):
      kernel = prgs[k['name']]
      aaa = []
      has_local = False
      for i,(a,sz) in enumerate(zip(k['args'], k['args_size'])):
        arg_name = k['args_name'][i]
        if len(a) == 0:
          aa = cl.LocalMemory(sz)
          has_local = True
        elif len(a) == 4:
          a = a.encode('latin_1')
          aa = np.uint32(struct.unpack("I", a)[0])
        elif len(a) == 2:
          a = a.encode('latin_1')
          aa = np.uint16(struct.unpack("H", a)[0])
        elif len(a) == 8:
          aa = bufs[a]
          if debug:
            #print(f"  {arg_name:20s} : {aa}")
            if arg_name == "output":
              seen_output.add(aa)
            if arg_name == "input":
              if aa not in seen_output:
                print("ERROR", aa, "is not seen in output")
        aaa.append(aa)

      """
      for a in aaa:
        types = {cl.mem_object_type.IMAGE1D_BUFFER: "IMAGE1D_BUFFER", cl.mem_object_type.IMAGE2D: "IMAGE2D", cl.mem_object_type.IMAGE1D: "IMAGE1D"}
        if isinstance(a, cl.Image):
          if a.type == cl.mem_object_type.IMAGE2D:
            print("  ", a, types[a.type], a.shape)
          elif a.type == cl.mem_object_type.IMAGE1D_BUFFER:
            print("  ", a, types[a.type], a.size)
          else:
            print("  ", a, types[a.type])
        elif isinstance(a, cl.Buffer):
          print("  ", a, a.size)
        elif isinstance(a, cl.LocalMemory):
          print("  ", a, a.size)
        else:
          print("  ", a)
      """

      if has_local:
        e = kernel(q, k['global_work_size'], k['local_work_size'], *aaa)
      else:
        e = kernel(q, k['global_work_size'], None, *aaa)

      kernels.append((k,e))

      #if k['name'] == 'zero_pad_image_float':
        #arr = np.zeros((aaa[1].size//4), dtype=np.float32)
        #cl.enqueue_copy(q, arr, aaa[1])
    
      """
      if k['name'] == "convolution_horizontal_reduced_reads":
        print(aaa)
        return dump_image(ctx, q, aaa[0]), dump_image(ctx, q, aaa[6]), dump_image(ctx, q, aaa[10])
      """

      """
      if isinstance(aaa[0], cl.Image):
        dump_image(ctx, q, aaa[0])
        if k['name'] == "convolution_horizontal_reduced_reads":
          dump_image(ctx, q, aaa[6])
      """

      #q.finish()

    q.finish()
    for k,e in kernels:
      print("%-60s" % k['name'], f"{str(k['global_work_size']):20s} {str(k['local_work_size']):20s} {(e.profile.end - e.profile.start)/1e3:9.2f} us")
      total_runtime += e.profile.end - e.profile.start
    print(f"total runtime: {total_runtime/1e6:.2f} ms")

    if len(outputs) == 0: return
    if vision_only:
      output = vision[0]
      ret = np.zeros(output.size//2, dtype=np.float16)
    else:
      output = outputs[0]
      ret = np.zeros(output.size//4, dtype=np.float32)
    cl.enqueue_copy(q, ret, output)
    if float32:
      return ret[:len(ret)//2]
    else:
      return ret

  return runner

if __name__ == "__main__":
  runner = load_thneed_model("/data/openpilot/selfdrive/modeld/models/supercombo.thneed" if len(sys.argv) == 1 else sys.argv[1], float32=bool(int(os.getenv("FLOAT32", "0"))))

  np.random.seed(1338)
  np_inputs = {
    "input_imgs": np.random.randn(*(1, 12, 128, 256))*256,
    "big_input_imgs": np.random.randn(*(1, 12, 128, 256))*256,
    "desire": np.zeros((1, 8)),
    "traffic_convention": np.array([[1., 0.]]),
    "initial_state": np.random.randn(*(1, 512))
  }
  np_inputs = {k:v.astype(np.float32) for k,v in np_inputs.items()}
  inputs = list(np_inputs.values())[::-1]

  ret = runner(inputs, vision_only=False, debug=True)
  print(ret.shape)

  if len(sys.argv) > 2:
    print("comparing to ONNX")
    from test.test_onnx import run_onnx_torch
    from extra.utils import fetch
    import onnx, io
    dat = fetch(sys.argv[2])
    onnx_model = onnx.load(io.BytesIO(dat))
    out = run_onnx_torch(onnx_model, np_inputs).numpy()[0]

    diff = 0
    diffs = []
    for i in range(ret.shape[0]):
      if abs(out[i]-ret[i]) > 0.1 and abs((out[i]-ret[i])/out[i]) > 0.01:
        diff += 1
        diffs.append(out[i] - ret[i])
        if diff == 10:
          print("...")
        elif diff < 10:
          print(i, out[i], ret[i], out[i]-ret[i])
    if len(diffs) > 0:
      print("%d differences min: %f max: %f" % (diff, min(diffs), max(diffs)))
    assert diff == 0

  """
  for i in range(0, len(ret), 0x10):
    p = []
    for j in ret[i:i+0x10]:
      p.append("%6.2f " % j)
    print("%5d" % i + ''.join(p))
  """
  exit(0)

  #test_dat = [open("/home/batman/openpilot/xx/tools/snpe/compile_test_data/dlc_input_%d" % i, "rb").read() for i in range(4)]
  #cl.enqueue_copy(q, inputs[3], test_dat[0])

  for i in range(5):
    st = time.time()
    ret = runner()
    et = time.time()
    print(ret.shape, ret, (et-st)*1000.)
  exit(0)

  print([x.size for x in inputs])
  print("**************", outputs)
  output = outputs[0]

  #print(dir(output))
  #print(output.buffer)

  ret = np.zeros(output.size//4, dtype=np.float32)
  if output.type == cl.mem_object_type.IMAGE2D:
    cl.enqueue_copy(q, ret, output, origin=(0,0), region=output.shape)
  else:
    cl.enqueue_copy(q, ret, output)
  #cl.enqueue_copy(q, ret, output.buffer)
  #for i in range(0, 32, 16):
  #  print(ret[i:i+0x10])
  print(ret.shape, ret)


