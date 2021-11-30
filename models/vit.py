import numpy as np
from tinygrad.tensor import Tensor
from models.transformer import TransformerBlock

class ViT:
  def __init__(self, layers=12, embed_dim=192, num_heads=3):
    self.conv = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
    self.cls_token = Tensor.ones(1, 1, embed_dim)
    self.pos_embed = Tensor.ones(1, 197, embed_dim)
    self.tbs = [
      TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
        prenorm=True, act=lambda x: x.gelu())
      for i in range(layers)]
    self.norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
    self.head = (Tensor.uniform(embed_dim, 1000), Tensor.zeros(1000))

  def patch_embed(self, x):
    x = x.conv2d(*self.conv, stride=16)
    x = x.reshape(shape=(x.shape[0], x.shape[1], -1)).transpose(order=(0,2,1))
    return x

  def forward(self, x):
    pe = self.patch_embed(x)
    x = self.cls_token.add(Tensor.zeros(pe.shape[0],1,1)).cat(pe, dim=1) + self.pos_embed
    x = x.sequential(self.tbs)
    x = x.layernorm().linear(*self.norm)
    return x[:, 0].linear(*self.head)

  def load_from_pretrained(m):
    import io
    from extra.utils import fetch

    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    url = "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    dat = np.load(io.BytesIO(fetch(url)))

    #for x in dat.keys():
    #  print(x, dat[x].shape, dat[x].dtype)

    m.conv[0].assign(np.transpose(dat['embedding/kernel'], (3,2,0,1)))
    m.conv[1].assign(dat['embedding/bias'])

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
