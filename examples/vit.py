
"""
fn = "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz"
import tensorflow as tf
with tf.io.gfile.GFile(fn, "rb") as f:
  dat = f.read()
  with open("cache/"+ fn.rsplit("/", 1)[1], "wb") as g:
    g.write(dat)
"""

import io
from extra.utils import fetch

from tinygrad.tensor import Tensor
from models.transformer import TransformerBlock, layernorm
class ViT:
  def __init__(self):
    self.conv_weight = Tensor.uniform(192, 3, 16, 16)
    self.conv_bias = Tensor.zeros(192)
    self.cls_token = Tensor.ones(1, 1, 192)
    self.tbs = [TransformerBlock(embed_dim=192, num_heads=3, ff_dim=768) for i in range(12)]
    self.pos_embed = Tensor.ones(1, 197, 192)
    self.head = (Tensor.uniform(192, 1000), Tensor.zeros(1000))
    self.norm = (Tensor.uniform(192), Tensor.zeros(192))

  def forward(self, x):
    print(x.shape)
    x = x.conv2d(self.conv_weight, stride=16)
    x = x.add(self.conv_bias.reshape(shape=(1,-1,1,1)))
    print(x.shape)
    x = x.reshape(shape=(x.shape[0], 192, -1)).transpose(order=(0,2,1))
    print(x.shape)
    # TODO: expand cls_token for batch
    x = self.cls_token.cat(x, dim=1).add(self.pos_embed)
    print(x.shape)
    #print(x.sum())
    for l in self.tbs:
      x = l(x)
      #print(x.sum())
    print(x.shape)
    x = layernorm(x, 192).affine(self.norm)
    return x[:, 0].affine(self.head)

m = ViT()

import numpy as np
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
dat = np.load(io.BytesIO(fetch("https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz")))
for x in dat.keys():
  print(x, dat[x].shape, dat[x].dtype)

m.conv_weight.assign(np.transpose(dat['embedding/kernel'], (3,2,1,0)))
m.conv_bias.assign(dat['embedding/bias'])

m.norm[0].assign(dat['Transformer/encoder_norm/scale'])
m.norm[1].assign(dat['Transformer/encoder_norm/bias'])

m.head[0].assign(dat['head/kernel'])
m.head[1].assign(dat['head/bias'])
m.cls_token.assign(dat['cls'])
m.pos_embed.assign(dat['Transformer/posembed_input/pos_embedding'])

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
  
#url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Chicken.jpg"
url = "https://repository-images.githubusercontent.com/296744635/39ba6700-082d-11eb-98b8-cb29fb7369c0"

# category labels
import ast
lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
lbls = ast.literal_eval(lbls.decode('utf-8'))

# junk
from PIL import Image
img = Image.open(io.BytesIO(fetch(url)))
aspect_ratio = img.size[0] / img.size[1]
img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
img = np.array(img)
y0,x0=(np.asarray(img.shape)[:2]-224)//2
img = img[y0:y0+224, x0:x0+224]
img = np.moveaxis(img, [2,0,1], [0,1,2])
img = img.astype(np.float32)[:3].reshape(1,3,224,224)
img /= 255.0
img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

Tensor.training = False
out = m.forward(Tensor(img))
outnp = out.cpu().data.ravel()
choice = outnp.argmax()
print(out.shape, choice, outnp[choice])

print(lbls[choice])

#lookup = dict([x.split(" ") for x in open("cache/classids.txt").read().strip().split("\n")])
#cls = open("cache/imagenet21k_wordnet_ids.txt").read().strip().split("\n")
#print(cls[choice], lookup[cls[choice]])



