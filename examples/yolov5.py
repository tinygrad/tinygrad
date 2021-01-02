#!/usr/bin/env python3
import io
import pickle
from extra.utils import fetch, my_unpickle

if __name__ == "__main__":
  dat = fetch('https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt')
  #import torch
  #td = torch.load(io.BytesIO(dat))
  #print(td)

  import zipfile
  fp = zipfile.ZipFile(io.BytesIO(dat))
  #fp.printdir()
  data = fp.read('archive/data.pkl')

  #import pickletools
  #pickletools.dis(io.BytesIO(data))

  ret, out = my_unpickle(io.BytesIO(data))
  print(dir(ret['model']))
  for m in ret['model']._modules['model']:
    print(m)
    print(m._modules.keys())

  """
  weights = fake_torch_load(data)
  for k,v in weights:
    print(k)
  """

