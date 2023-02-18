# load weights from
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
import sys
import io
import ast
import time
import cv2
import numpy as np
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from extra.utils import fetch
from models.efficientnet import EfficientNet
np.set_printoptions(suppress=True)

def infer(model, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # if you want to look at the image
  """
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()
  """

  # low level preprocess
  img = np.moveaxis(img, [2,0,1], [0,1,2])
  img = img.astype(np.float32)[:3].reshape(1,3,224,224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

  # run the net
  out = model.forward(Tensor(img)).cpu()

  # if you want to look at the outputs
  """
  import matplotlib.pyplot as plt
  plt.plot(out.numpy()[0])
  plt.show()
  """
  return out, retimg

if __name__ == "__main__":
  # instantiate my net
  model = EfficientNet(getenv("NUM", 0))
  model.load_from_pretrained()

  # category labels
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  # load image and preprocess
  url = sys.argv[1] if len(sys.argv) >= 2 else "https://raw.githubusercontent.com/geohot/tinygrad/master/docs/stable_diffusion_by_tinygrad.jpg"
  if url == 'webcam':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
      _ = cap.grab() # discard one frame to circumvent capture buffering
      ret, frame = cap.read()
      img = Image.fromarray(frame[:, :, [2,1,0]])
      out, retimg = infer(model, img)
      print(np.argmax(out.numpy()), np.max(out.numpy()), lbls[np.argmax(out.numpy())])
      SCALE = 3
      simg = cv2.resize(retimg, (224*SCALE, 224*SCALE))
      retimg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
      cv2.imshow('capture', retimg)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
  else:
    if url.startswith('http'):
      img = Image.open(io.BytesIO(fetch(url)))
    else:
      img = Image.open(url)
    st = time.time()
    out, _ = infer(model, img)
    print(np.argmax(out.numpy()), np.max(out.numpy()), lbls[np.argmax(out.numpy())])
    print(f"did inference in {(time.time()-st):2f}")
