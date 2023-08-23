from models.efficientnet import EfficientNet
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.utils import fetch
from extra.export_model import export_model
from tinygrad.helpers import getenv
import ast, os

if __name__ == "__main__":
  model = EfficientNet(0)
  model.load_from_pretrained()
  mode = "clang" if getenv("CLANG", "") != "" else "webgpu" if getenv("WEBGPU", "") != "" else ""
  prg, inp_size, out_size, state = export_model(model, Tensor.randn(1,3,224,224), mode)
  if getenv("CLANG", "") == "":
    safe_save(state, os.path.join(os.path.dirname(__file__), "net.safetensors"))
    ext = "js" if getenv("WEBGPU", "") != "" else "json"
    with open(os.path.join(os.path.dirname(__file__), f"net.{ext}"), "w") as text_file:
      text_file.write(prg)
  else:
    cprog = [prg]
    # image library!
    cprog += ["#define STB_IMAGE_IMPLEMENTATION", fetch("https://raw.githubusercontent.com/nothings/stb/master/stb_image.h").decode('utf-8').replace("half", "_half")]

    # imagenet labels, move to datasets?
    lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
    lbls = ast.literal_eval(lbls.decode('utf-8'))
    lbls = ['"'+lbls[i]+'"' for i in range(1000)]
    cprog.append(f"char *lbls[] = {{{','.join(lbls)}}};")
    cprog.append(f"float input[{inp_size}];")
    cprog.append(f"float outputs[{out_size}];")

    # buffers (empty + weights)
    cprog.append("""
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
    net(input, outputs);
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
  }""")

    # CLANG=1 python3 examples/compile_efficientnet.py | clang -O2 -lm -x c - -o recognize && DEBUG=1 time ./recognize docs/showcase/stable_diffusion_by_tinygrad.jpg
    # category : 281 (tabby, tabby cat) with 9.452788
    print('\n'.join(cprog))
