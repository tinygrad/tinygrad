import struct
from models.efficientnet import EfficientNet
from tinygrad.tensor import Tensor
from extra.utils import fetch
import ast

def compile_net(run, special_names):
  # functions that run the net
  functions = {}
  bufs = {}
  bufnum = 0
  statements = []
  bufs_to_save = {}
  for fxn,args in run.jit_cache:
    functions[fxn.name] = fxn.prg   # NOTE: this assumes all with the same name are the same
    cargs = []
    for i,arg in enumerate(args):
      key = id(arg)
      if key not in bufs:
        if key in special_names:
          bufs[key] = (special_names[key], len(arg._buf))
        else:
          bufs[key] = (f"buf_{bufnum}", len(arg._buf))
          bufnum += 1
          if i > 0: bufs_to_save[bufs[key][0]] = arg   # if first usage of a buffer is not an output, and it's not a special name
      if bufs[key][0] in bufs_to_save:
        cargs.append(f"&{bufs[key][0]}")
      else:
        cargs.append(f"&mut {bufs[key][0]}")
    statements.append(f"{fxn.name}({', '.join(cargs)});")

  return functions, statements, bufs, bufs_to_save

if __name__ == "__main__":
  model = EfficientNet(0)
  model.load_from_pretrained()

  from tinygrad.jit import TinyJit
  @TinyJit
  def run(x): return model.forward(x).realize()

  # twice to run the JIT
  the_input = Tensor.randn(1,3,224,224)
  the_output = run(the_input)
  the_output = run(the_input)

  # hack to put the inputs back
  assert len(run.input_replace) == 1, f"didn't get one input to replace {run.input_replace}"
  for (j,i),idx in run.input_replace.items():
    run.jit_cache[j][1][i] = the_input.lazydata.realized

  # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  special_names = {id(the_input.lazydata.realized): "input", id(the_output.lazydata.realized): "outputs"}

  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)

  rsprog = []

  # save the weights
  for name,cl in bufs_to_save.items():
    b = bytes(cl._buf)
    num = len(b) // 4
    weights = ",".join([str(f) for f in struct.unpack(str(num)+'f', b)])
    rsprog.append(f"static {name} : [f32; {num}] = [{weights}];")

  # image library!
  # rsprog += ["#define STB_IMAGE_IMPLEMENTATION", fetch("https://raw.githubusercontent.com/nothings/stb/master/stb_image.h").decode('utf-8')]

  # imagenet labels, move to datasets?
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))
  lbls = ['"'+lbls[i]+'"' for i in range(1000)]
  rsprog.append(f"pub static lbls : [&'static str; 1000] = [{','.join(lbls)}];")

  # empty buffers
  # TODO move all, but the input buffer into the net function to help the compiler?
  # if name == "input" or name == "outputs":
  rsprog += [f"pub static mut {name} : &'static mut [f32; {len}] = &mut[0.0; {len}];" for name,len in bufs.values() if name not in bufs_to_save]

  # the functions
  rsprog += list(functions.values())

  # the net
  # TODO reduce unsafe scope to just indexing ops
  rsprog += ["pub unsafe fn net() {"] + statements + ["}"]

  # CLANG=1 python3 examples/compile_efficientnet.py | clang -O2 -lm -x c - -o recognize && DEBUG=1 time ./recognize docs/stable_diffusion_by_tinygrad.jpg
  # category : 281 (tabby, tabby cat) with 9.452788
  with open("./examples/efficientnet_rs/src/net.rs", "w") as f:
    f.write('\n'.join(rsprog))
