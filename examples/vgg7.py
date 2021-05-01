from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.optim import SGD
import extra.waifu2x
from extra.kinne import KinneDir
import sys
import os
import random
import json

# amount of context erased by model
CONTEXT = 7

if len(sys.argv) < 2:
  print("python3 -m examples.vgg7 import MODELJSON MODELDIR")
  print(" imports a waifu2x JSON vgg_7 model, i.e. waifu2x/models/vgg_7/art/scale2.0x_model.json")
  print(" into a directory of float binaries along with a meta.txt file containing tensor sizes")
  print(" weight tensors are ordered in tinygrad/ncnn form, as so: (outC,inC,H,W)")
  print(" *this format is used by all other commands in this program*")
  print("python3 -m examples.vgg7 execute MODELDIR IMG_IN IMG_OUT")
  print(" given an already-nearest-neighbour-scaled image, runs vgg7 on it")
  print(" output image has 7 pixels removed on all edges")
  print(" do not run on large images, will have *hilarious* RAM use")
  print("python3 -m examples.vgg7 execute_full MODELDIR IMG_IN IMG_OUT")
  print(" does the 'whole thing' (padding, tiling)")
  print(" safe for large images, etc.")
  print("python3 -m examples.vgg7 new MODELDIR")
  print(" creates a new model (experimental)")
  print("python3 -m examples.vgg7 train MODELDIR SAMPLES_DIR SAMPLES_COUNT ROUNDS")
  print(" trains a model (experimental)")
  print(" (how experimental? well, every time I tried it, it flooded w/ NaNs)")
  print(" expects the same inputs as execute, however it autocrops the output,")
  print("  as SAMPLES_DIR/IDXa.png and SAMPLES_DIR/IDXb.png")
  print(" (i.e. my_samples/0a.png is the first pre-scaled image,")
  print("       my_samples/0b.png is the first output image)")
  print(" won't pad or tile, so keep image sizes sane")
  sys.exit(1)

cmd = sys.argv[1]
vgg7 = extra.waifu2x.Vgg7()

def load_and_save(path, save):
  kn = KinneDir(model, save)
  kn.parameters(vgg7.get_parameters())
  kn.close()

if cmd == "import":
  src = sys.argv[2]
  model = sys.argv[3]

  vgg7.load_waifu2x_json(json.load(open(src, "rb")))

  os.mkdir(model)
  load_and_save(model, True)
elif cmd == "execute":
  model = sys.argv[2]
  in_file = sys.argv[3]
  out_file = sys.argv[4]

  load_and_save(model, False)

  extra.waifu2x.image_save(out_file, vgg7.forward(Tensor(extra.waifu2x.image_load(in_file))).data)
elif cmd == "execute_full":
  model = sys.argv[2]
  in_file = sys.argv[3]
  out_file = sys.argv[4]

  load_and_save(model, False)

  extra.waifu2x.image_save(out_file, vgg7.forward_tiled(extra.waifu2x.image_load(in_file), 156))
elif cmd == "new":
  model = sys.argv[2]

  os.mkdir(model)
  load_and_save(model, True)
elif cmd == "train":
  model = sys.argv[2]
  samples_base = sys.argv[3]
  samples_count = int(sys.argv[4])
  rounds = int(sys.argv[5])

  load_and_save(model, False)

  x_imgs = []
  y_imgs = []
  for idx in range(samples_count):
    # x image is image that has been scaled down by some arbitrary method
    #  (chosen method is dependent on what you want to train for),
    #  and has been scaled up again via nearest-neighbour.
    #  (This scaling was determined through eyeballing quality of
    #    the mainline waifu2x vgg7 models.
    #   Using better scalings actually worsens quality for these models.)
    # Of course, you may train your models as you wish,
    #  but your implementation then has to provide similar input.
    x_img = extra.waifu2x.image_load(samples_base + "/" + str(idx) + "a.png")
    # y image is original image.
    y_img = extra.waifu2x.image_load(samples_base + "/" + str(idx) + "b.png")
    # erase context from y_img
    y_img = y_img[:,:,CONTEXT:y_img.shape[2]-CONTEXT,CONTEXT:y_img.shape[3]-CONTEXT]
    x_imgs += [x_img]
    y_imgs += [y_img]

  optim = SGD(vgg7.get_parameters(), lr=0.01)

  for rnum in range(rounds):
    print("Round " + str(rnum))
    sample_idx = random.randint(0, samples_count - 1)
    sample_x = Tensor(x_imgs[sample_idx])
    sample_y = Tensor(y_imgs[sample_idx])
    # magic code roughly from readme example
    out = vgg7.forward(sample_x)
    loss = out.mul(sample_y).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("Saving")
    load_and_save(model, True)

else:
  print("unknown command")

