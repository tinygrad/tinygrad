# load weights from
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
import os
import sys
import io
import time
import numpy as np
np.set_printoptions(suppress=True)
from tinygrad.tensor import Tensor
from extra.utils import fetch, get_parameters
from models.efficientnet import EfficientNet

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
  plt.plot(out.data[0])
  plt.show()
  """
  return out, retimg

if __name__ == "__main__":
  # instantiate my net
  model = EfficientNet(int(os.getenv("NUM", "0")))
  model.load_weights_from_torch()

  # category labels
  import ast
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  # load image and preprocess
  from PIL import Image
  url = sys.argv[1]
  if url == 'webcam':
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
      _ = cap.grab() # discard one frame to circumvent capture buffering
      ret, frame = cap.read()
      img = Image.fromarray(frame[:, :, [2,1,0]])
      out, retimg = infer(model, img)
      print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
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
    print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
    print("did inference in %.2f s" % (time.time()-st))
  #print("NOT", np.argmin(out.data), np.min(out.data), lbls[np.argmin(out.data)])
