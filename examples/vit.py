
"""
fn = "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz"
import tensorflow as tf
with tf.io.gfile.GFile(fn, "rb") as f:
  dat = f.read()
  with open("cache/"+ fn.rsplit("/", 1)[1], "wb") as g:
    g.write(dat)
"""

import numpy as np
dat = np.load("cache/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz")
for x in dat.keys():
  print(x, dat[x].shape)


