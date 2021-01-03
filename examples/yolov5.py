#!/usr/bin/env python3
import io
import pickle
from extra.utils import fetch, my_unpickle

if __name__ == "__main__":
  dat = fetch('https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt')
  #dat = fetch('https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt')

  import zipfile
  fp = zipfile.ZipFile(io.BytesIO(dat))
  #fp.printdir()
  data = fp.read('archive/data.pkl')

  # yolo specific
  ret, out = my_unpickle(io.BytesIO(data))
  d = ret['model'].yaml
  for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
    tm = ret['model']._modules['model'][i]
    print(i, f, n, m, args, tm._modules.keys())
    # Focus, Conv, BottleneckCSP, SPP, Concat, Detect
    #for k,v in tm._modules.items():
    #  print("   ", k, v)
    if m in "Focus":
      conv = tm._modules['conv']
      print("   ", conv._modules)
    if m in "Conv":
      conv, bn = tm._modules['conv'], tm._modules['bn']
      print("   ", conv)
      #print(bn)



