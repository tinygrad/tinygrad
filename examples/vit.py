
"""
fn = "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz"
import tensorflow as tf
with tf.io.gfile.GFile(fn, "rb") as f:
  dat = f.read()
  with open("cache/"+ fn.rsplit("/", 1)[1], "wb") as g:
    g.write(dat)
"""


from tinygrad.tensor import Tensor
from models.transformer import TransformerBlock
class ViT:
  def __init__(self):
    self.conv_weight = Tensor.uniform(192, 3, 16, 16)
    self.conv_bias = Tensor.zeros(192)
    self.cls = Tensor.ones(1, 1, 192)
    self.tbs = [TransformerBlock(embed_dim=192, num_heads=3, ff_dim=768) for i in range(12)]
    self.pos = Tensor.ones(1, 197, 192)
    self.head = (Tensor.uniform(192, 21843), Tensor.zeros(21843))

  def forward(self, x):
    print(x.shape)
    x = x.conv2d(self.conv_weight, stride=16)
    x = x.add(self.conv_bias.reshape(shape=(1,-1,1,1)))
    print(x.shape)
    x = x.reshape(shape=(x.shape[0], 192, -1)).transpose(order=(0,2,1))
    print(x.shape)
    x = self.cls.cat(x, dim=1)
    print(x.shape)
    for l in self.tbs:
      x = l(x)
    return x[:, 0].affine(self.head)

m = ViT()

import numpy as np
dat = np.load("cache/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz")
for x in dat.keys():
  print(x, dat[x].shape, dat[x].dtype)

m.conv_weight.assign(np.transpose(dat['embedding/kernel'], (3,2,1,0)))
m.conv_bias.assign(dat['embedding/bias'])
m.cls.assign(dat['cls'])
m.pos.assign(dat['Transformer/posembed_input/pos_embedding'])

for i in range(12):
  m.tbs[i].query_dense[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].reshape(192, 192))
  m.tbs[i].query_dense[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].reshape(192))
  m.tbs[i].key_dense[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].reshape(192, 192))
  m.tbs[i].key_dense[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].reshape(192))
  m.tbs[i].value_dense[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].reshape(192, 192))
  m.tbs[i].value_dense[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].reshape(192))
  m.tbs[i].final[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].reshape(192, 192))
  m.tbs[i].final[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'].reshape(192))
  m.tbs[i].ff1[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'])
  m.tbs[i].ff1[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
  m.tbs[i].ff2[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'])
  m.tbs[i].ff2[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])
  m.tbs[i].ln1[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])
  m.tbs[i].ln1[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
  m.tbs[i].ln2[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])
  m.tbs[i].ln2[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
  
test_input = Tensor.ones(1, 3, 224, 224)
out = m.forward(test_input)
print(out.shape)


