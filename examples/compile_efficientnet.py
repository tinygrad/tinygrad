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
      cargs.append(bufs[key][0])
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
  assert len(run.input_replace) == 1
  for (j,i),idx in run.input_replace.items():
    run.jit_cache[j][1][i] = the_input.lazydata.realized.raw()

  # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  special_names = {id(the_input.lazydata.realized.raw()): "input", id(the_output.lazydata.realized.raw()): "outputs"}

  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)

  # c header
  cprog = ["#include <stdio.h>", "#include <math.h>", "#define max(x,y) ((x>y)?x:y)"]

  # save the weights
  for name,cl in bufs_to_save.items():
    weight = ''.join(["\\x%02X"%x for x in bytes(cl._buf)])
    cprog.append(f"unsigned char {name}_data[] = \"{weight}\";")

  # image library!
  cprog += ["#define STB_IMAGE_IMPLEMENTATION", fetch("https://raw.githubusercontent.com/nothings/stb/master/stb_image.h").decode('utf-8')]

  # imagenet labels, move to datasets?
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))
  lbls = ['"'+lbls[i]+'"' for i in range(1000)]
  cprog.append(f"char *lbls[] = {{{','.join(lbls)}}};")

  # buffers (empty + weights)
  cprog += [f"float {name}[{len}];" if name not in bufs_to_save else f"float *{name} = (float *){name}_data;" for name,len in bufs.values()] 

  # the functions
  cprog += list(functions.values())

  # the net
  cprog += ["void net() {"] + statements + ["}"]

  cprog += ["""
int main(int argc, char* argv[]) {
  int DEBUG = getenv("DEBUG") != NULL ? atoi(getenv("DEBUG")) : 0;
  int X=0, Y=0, chan=0;
  stbi_uc *image = (argc > 1) ? stbi_load(argv[1], &X, &Y, &chan, 3) : stbi_load_from_file(stdin, &X, &Y, &chan, 3);
  assert(image != NULL);
  if (DEBUG) printf("loaded image %dx%d channels %d\\n", X, Y, chan);
  assert(chan == 3);
  // resize to input[1,3,224,224] and rescale
  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      // get sample position
      int tx = (x/224.)*X;
      int ty = (y/224.)*Y;
      for (int c = 0; c < 3; c++) {
        input[c*224*224 + y*224 + x] = (image[ty*X*chan + tx*chan + c] / 255.0 - 0.45) / 0.225;
      }
    }
  }
  net();
  float best = -INFINITY;
  int best_idx = -1;
  for (int i = 0; i < 1000; i++) {
    if (outputs[i] > best) {
      best = outputs[i];
      best_idx = i;
    }
  }
  if (DEBUG) printf("category : %d (%s) with %f\\n", best_idx, lbls[best_idx], best);
  else printf("%s\\n", lbls[best_idx]);
}"""]

  # CLANG=1 python3 examples/compile_efficientnet.py | clang -O2 -lm -x c - -o recognize && DEBUG=1 time ./recognize docs/stable_diffusion_by_tinygrad.jpg
  print('\n'.join(cprog))
