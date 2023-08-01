from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, getenv
from tinygrad.state import torch_load
from vits import ResidualCouplingBlock, PosteriorEncoder, Encoder, ResBlock1, ResBlock2, LRELU_SLOPE, sequence_mask, split, download_if_not_present, get_hparams_from_file, load_checkpoint, weight_norm, HParams
import numpy as np
import math
import os
import logging
import sys
from pathlib import Path
from typing import Optional
import time

# original code: https://github.com/svc-develop-team/so-vits-svc

DEBUG = getenv("DEBUG")

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)

# TODO: add speech encoder ContentVec256L9 (most weights trained on 4.0 architecture require this)
# TODO: add Synthesizer
# TODO: add Volume Extractor
class Svc():
  def __init__(self, net_g_config): pass

# original code for contentvec: https://github.com/auspicious3000/contentvec/
class ContentVec:
  # TODO: self.label_embs_concat and self.final_proj dims are hardcoded
  # and depend on fairseq.data.dictionary Dictionary in the checkpoint.
  # This param can't yet be loaded since there is no pickle for it. See with DEBUG=2.
  def __init__(self, cfg: HParams):
    self.feature_grad_mult, self.untie_final_proj = cfg.feature_grad_mult, cfg.untie_final_proj
    self.embed = feature_enc_layers[-1][0]
    final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
    feature_enc_layers = eval(cfg.conv_feature_layers)
    self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, mode=cfg.extractor_mode, conv_bias=cfg.conv_bias)
    self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
    self.encoder = TransformerEncoder(cfg)
    self.layer_norm = nn.LayerNorm(self.embed)
    self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim * 1) if self.untie_final_proj else nn.Linear(cfg.encoder_embed_dim, final_dim)
    self.mask_emb = Tensor.uniform(cfg.encoder_embed_dim, dtype=dtypes.float32)
    self.label_embs_concat = Tensor.uniform(504, final_dim, dtype=dtypes.float32)

  def forward_features(self, source, padding_mask):
    if self.feature_grad_mult > 0:
      features = self.feature_extractor(source, padding_mask)
      if self.feature_grad_mult != 1.0:
        features = GradMultiply.forward(features, self.feature_grad_mult)  # TODO: this isn't required for inference
    else:
      features = self.feature_extractor(source, padding_mask)
    return features

  def forward_padding_mask(self, features, padding_mask):
    # replaces original forward_padding_mask for batch inference
    lengths_org = tilde(padding_mask.cast(dtypes.bool)).cast(dtypes.int64).sum(1)  # ensure its bool for tilde
    lengths = (lengths_org - 400).cast(dtypes.float32).div(320).floor().cast(dtypes.int64) + 1
    padding_mask = lengths_to_padding_mask(lengths)
    return padding_mask

  def extract_features(self, source: Tensor, spk_emb: Tensor, padding_mask=None, mask=False, ret_conv=False, output_layer=None, tap=False):
    features = self.forward_features(source, padding_mask)
    if padding_mask is not None:
      padding_mask = self.forward_padding_mask(features, padding_mask)
    features = features.transpose(1, 2)
    features = self.layer_norm(features)
    if self.post_extract_proj is not None:
      features = self.post_extract_proj(features)
    x, layer_results = self.encoder(features, spk_emb, padding_mask=padding_mask, layer=(None if output_layer is None else output_layer - 1), tap=tap)
    res = features if ret_conv else x
    return res, padding_mask

  @classmethod
  def load_model(cls, checkpoint_path:str, checkpoint_url:str):
    download_if_not_present(checkpoint_path, checkpoint_url)
    cfg = load_config_from_checkpoint(checkpoint_path)
    enc = cls(cfg.model)
    _ = load_checkpoint_enc(checkpoint_path, enc, None)
    logging.debug(f"{cls.__name__}: Loaded model with cfg={cfg}")
    return enc

class TransformerEncoder:
  def __init__(self, cfg: HParams):
    def make_conv() -> nn.Conv1d:
      layer = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=cfg.conv_pos, padding=cfg.conv_pos // 2, groups=cfg.conv_pos_groups)
      std = std = math.sqrt(4 / (cfg.conv_pos * self.embedding_dim))
      layer.weight, layer.bias = (Tensor.normal(*layer.weight.shape, std=std)), (Tensor.zeros(*layer.bias.shape))
      # for training: layer.weights need to be weight_normed
      return layer
    self.dropout, self.embedding_dim, self.layer_norm_first, self.layerdrop, self.num_layers, self.num_layers_1 = cfg.dropout, cfg.encoder_embed_dim, cfg.layer_norm_first, cfg.encoder_layerdrop, cfg.encoder_layers, cfg.encoder_layers_1
    self.pos_conv = Sequential([make_conv(), SamePad(cfg.conv_pos), GELU()])
    self.layers = [
      TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim,
                                      ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
                                      num_attention_heads=cfg.encoder_attention_heads,
                                      dropout=0.0,  # training: dropout=self.dropout,
                                      attention_dropout=0.0,  # training: attention_dropout=cfg.attention_dropout,
                                      activation_dropout=0.0,  # training: activation_dropout=cfg.activation_dropout,
                                      activation_fn=cfg.activation_fn,
                                      layer_norm_first=self.layer_norm_first)
      for _ in range(cfg.encoder_layers)
      ]
    for _ in range(cfg.encoder_layers_1):  # this one uses CondLayerNorm
      self.layers.append(
        TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim,
                                        ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
                                        num_attention_heads=cfg.encoder_attention_heads,
                                        dropout=0.0,  # dropout=self.dropout,
                                        attention_dropout=0.0,  # training: attention_dropout=cfg.attention_dropout,
                                        activation_dropout=0.0,  # training: activation_dropout=cfg.activation_dropout,
                                        activation_fn=cfg.activation_fn,
                                        layer_norm_first=self.layer_norm_first,
                                        cond_layer_norm=True)
      )
    self.layer_norm = nn.LayerNorm(self.embedding_dim)
    self.cond_layer_norm = CondLayerNorm(self.embedding_dim) if cfg.encoder_layers_1 > 0 else None
    # training: apply init_bert_params

  def __call__(self, x, spk_emb, padding_mask=None, layer=None, tap=False):
    x, layer_results = self.extract_features(x, spk_emb, padding_mask, layer, tap)
    if self.layer_norm_first and layer is None:
      x = self.cond_layer_norm(x, spk_emb) if (self.num_layers_1 > 0) else self.layer_norm(x)
    return x, layer_results

  def extract_features(self, x: Tensor, spk_emb: Tensor, padding_mask=None, tgt_layer=None, tap=False):
    if tgt_layer is not None:  # and not self.training
      assert tgt_layer >= 0 and tgt_layer < len(self.layers)
    if padding_mask is not None: x = Tensor.where(tilde(padding_mask.cast(dtypes.bool)), x, 0)  # in torch x = x[padding_mask] := 0  -> all True indices set to 0. exactly the opposite of what where() does.
    x_conv = self.pos_conv(x.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)
    x = x + x_conv
    x = x.transpose(0, 1)  # B x T x C -> T x B x C
    if not self.layer_norm_first: x = self.layer_norm(x)
    x = x.dropout(p=self.dropout)
    layer_results = []
    r = None
    for i, layer in enumerate(self.layers):
      # dropout_probability = np.random.random()
      if i < self.num_layers:  # if (not self.training or (dropout_probability > self.layerdrop)) and (i < self.num_layers):
        x = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
        if tgt_layer is not None or tap:
          layer_results.append(x.transpose(0, 1))
      if i>= self.num_layers:
        x = layer(x, spk_emb, self_attn_padding_mask=padding_mask, need_weights=False)
      if i == tgt_layer:
        r = x
        break
    if r is not None:
      x = r
    x = x.transpose(0, 1)  # T x B x C -> B x T x C
    return x, layer_results

class TransformerSentenceEncoderLayer:
  def __init__(self, embedding_dim=768.0, ffn_embedding_dim=3072.0, num_attention_heads=8.0, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, activation_fn="relu", layer_norm_first=False, cond_layer_norm=False):
    def get_activation_fn(activation):
      if activation == "relu":
        return Tensor.relu
      if activation == "gelu":
        return Tensor.gelu
      else:
        raise RuntimeError(f"activation function={activation} is not forseen")
    self.embedding_dim, self.dropout, self.activation_dropout, self.layer_norm_first, self.num_attention_heads = embedding_dim, dropout, activation_dropout, layer_norm_first, num_attention_heads
    self.activation_fn = get_activation_fn(activation_fn)
    self.self_attn = MultiHeadAttention(self.embedding_dim, self.num_attention_heads)
    self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim) if not cond_layer_norm else CondLayerNorm(self.embedding_dim)
    self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
    self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
    self.final_layer_norm = nn.LayerNorm(self.embedding_dim) if not cond_layer_norm else CondLayerNorm(self.embedding_dim)

  def __call__(self, x:Tensor, self_attn_mask:Tensor=None, self_attn_padding_mask:Tensor=None):  # TODO: refactor this later
    # TODO: might need to do the following to self_attn_padding_mask:
    #self_attn_padding_mask = self_attn_padding_mask.view(x.shape[0], 1, 1, self_attn_padding_mask.shape[1]).expand(-1, self.num_attention_heads, -1, -1).reshape(x.shape[0] * self.num_attention_heads, 1, self_attn_padding_mask.shape[1]) if self_attn_padding_mask is not None else None
    residual = x
    if self.layer_norm_first:
      if self_attn_mask is None and self_attn_padding_mask is not None: self_attn_mask = self_attn_padding_mask
      elif self_attn_padding_mask is None and self_attn_mask is not None: self_attn_padding_mask = self_attn_mask
      mask = self_attn_mask.cast(dtypes.bool) + self_attn_padding_mask.cast(dtypes.bool) if (self_attn_mask is not None) else None  # logical or them here
      x = self.self_attn_layer_norm(x)
      x = self.self_attn(x=x, mask=mask)  # self attention. TODO: need_weights=False for not cond_layer_norm
      x = x.dropout(self.dropout)
      x = residual + x
      x = self.final_layer_norm(x)
      x = self.activation_fn(self.fc1(x))
      x = x.dropout(self.activation_dropout)
      x = self.fc2(x)
      layer_result = x
      x = x.dropout(self.dropout)
      x = residual + x
    else:
      x = self.self_attn(x=x, mask=self_attn_padding_mask)  # self attention. TODO: need_weights=False for not cond_layer_norm
      x = x.dropout(self.dropout)
      x = residual + x
      x = self.self_attn_layer_norm(x)
      residual = x
      x = self.activation_fn(self.fc1(x))
      x = x.dropout(self.activation_dropout)
      x = self.fc2(x)
      layer_result = x
      x = x.dropout(self.dropout)
      x = residual + x
      x = self.final_layer_norm(x)
    return x

# from examples/whisper.py. Renamed attributes for weight loading and allow bias on k_proj.
class MultiHeadAttention:
  def __init__(self, n_state, n_head):
    self.n_head = n_head
    self.q_proj = nn.Linear(n_state, n_state)
    self.k_proj = nn.Linear(n_state, n_state)
    self.v_proj = nn.Linear(n_state, n_state)
    self.out_proj = nn.Linear(n_state, n_state)

  def __call__(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None):
    q = self.q_proj(x)
    k = self.k_proj(xa or x)
    v = self.v_proj(xa or x)
    wv, qk = self.qkv_attention(q, k, v, mask)
    # NOTE: we aren't returning qk
    return self.out_proj(wv)

  def qkv_attention(self, q, k, v, mask=None):
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.reshape(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.reshape(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.reshape(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    qk = q @ k
    if mask is not None: qk = qk + mask[:n_ctx, :n_ctx]
    w = qk.softmax(-1)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ConvFeatureExtractionModel():
  def __init__(self, conv_layers, dropout=.0, mode="default", conv_bias=False):
    assert mode in {"default", "group_norm_masked", "layer_norm"}
    self.mode = mode
    def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
      def make_conv():
        conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
        conv.weight = Tensor.kaiming_normal(*conv.weight.shape)
        return conv
      class SequentialMasked(Sequential):  # https://github.com/auspicious3000/contentvec/blob/main/contentvec/models/wav2vec/wav2vec2_1.py#L56
        def __init__(self, list): super().__init__(list)
        def __call__(self, x, mask):
          x = self.list[0](x)
          x = self.list[1](x)
          x = self.list[2](x, mask)
          x = self.list[3](x)
          return x
      assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"
      if is_layer_norm:
        return Sequential([make_conv(), Dropout(p=dropout), Sequential([TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=True), TransposeLast()]), GELU()])
      if is_group_norm:
        if mode == "default":
          return Sequential([make_conv(), Dropout(p=dropout), Fp32GroupNorm(dim, dim, affine=True), GELU()])
        elif mode == "group_norm_masked":
          return SequentialMasked([make_conv(), Dropout(p=dropout), GroupNormMasked(dim, dim, affine=True), GELU()])
      else:
        return Sequential([make_conv(), Dropout(p=dropout), GELU()])
    in_d = 1
    self.conv_layers = []
    for i, cl in enumerate(conv_layers):
      assert len(cl) == 3, "invalid conv definition: " + str(cl)
      (dim, k, stride) = cl
      if i == 0:
        self.cl = cl
      self.conv_layers.append(block(in_d, dim, k, stride, is_layer_norm=(mode == "layer_norm"), is_group_norm=(mode == "default" or mode == "group_norm_masked") and i == 0, conv_bias=conv_bias))
      in_d = dim
  def forward(self, x, padding_mask):  # TODO: refactor this later
    # BxT -> BxCxT
    x = x.unsqueeze(1)
    for i, conv in enumerate(self.conv_layers) :
      if i == 0:
        if self.mode == "group_norm_masked":
          if padding_mask is not None:
            _, k, stride = self.cl
            lengths_org = tilde(padding_mask.cast(dtypes.bool)).cast(dtypes.int64).sum(1)  # ensure padding_mask is bool for tilde
            lengths = (((lengths_org - k) / stride) + 1).floor().cast(dtypes.int64)
            padding_mask = tilde(lengths_to_padding_mask(lengths)).cast(dtypes.int64)  # lengths_to_padding_mask returns bool tensor
          x = conv(x, padding_mask)  # padding_mask is numeric
        else:
          x = conv(x)  # default
      else:
        x = conv(x)
    return x

def tilde(x: Tensor) -> Tensor:
  if x.dtype == dtypes.bool: return (1 - x).cast(dtypes.bool)
  return (x + 1) * -1  # this seems to be what the ~ operator does in pytorch for non bool
def lengths_to_padding_mask(lens: Tensor) -> Tensor:
  bsz, max_lens = lens.shape[0], lens.max().numpy().item()
  mask = Tensor.arange(max_lens).to(lens.device).reshape(1, max_lens)
  mask = mask.expand(bsz, -1) >= lens.reshape(bsz, 1).expand(-1, max_lens)
  return mask.cast(dtypes.bool)

class GradMultiply:
  def forward(ctx, x, scale):
    ctx.scale = scale
    res = x.to(x.device)  # clone
    return res

class Sequential:  # allows recursive sequential
  def __init__(self, list): self.list=list
  def __call__(self, x: Tensor): return x.sequential(self.list)
  def __getitem__(self, i: int): return self.list[i]  # for loading weights

class Dropout:
  def __init__(self, p=0.5): self.p=p
  def __call__(self, x: Tensor): return x.dropout(self.p)

class GELU:
  def __init__(self): pass
  def __call__(self, x: Tensor): return x.gelu()

class SamePad:
  def __init__(self, kernel_size, causal=False):
    self.remove = (kernel_size - 1) if causal else (1 if kernel_size % 2 == 0 else 0)
  def __call__(self, x):
    if self.remove > 0:
      x = x[:, :, : -self.remove]
    return x

class TransposeLast:
  def __init__(self, deconstruct_idx=None): self.deconstruct_idx = deconstruct_idx
  def __call__(self, x: Tensor):
    if self.deconstruct_idx is not None:
      x = x[self.deconstruct_idx]
    return x.transpose(-2, -1)

class Fp32LayerNorm(nn.LayerNorm):
  def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
  def __call__(self, input: Tensor):
    self.weight = self.weight.cast(dtypes.float32) if self.weight is not None else None
    self.bias = self.bias.cast(dtypes.float32) if self.bias is not None else None
    output = super().__call__(input.cast(dtypes.float32))
    return output.cast(input.dtype)

class Fp32GroupNorm(nn.GroupNorm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  def __call__(self, input: Tensor):
    self.weight = self.weight.cast(dtypes.float32) if self.weight is not None else None
    self.bias = self.bias.cast(dtypes.float32) if self.bias is not None else None
    output = super().__call__(input.cast(dtypes.float32))
    return output.cast(input.dtype)

class CondLayerNorm:  # https://github.com/auspicious3000/contentvec/blob/main/contentvec/modules/cond_layer_norm.py#L10
  # this one is a bit weird since it has slightly different constructor args than nn.LayerNorm
  def __init__(self, dim_last, eps=1e-5, dim_spk=256, elementwise_affine=True):
    self.dim_last, self.eps, self.dim_spk, self.elementwise_affine = dim_last, eps, dim_spk, elementwise_affine
    if self.elementwise_affine:
      self.weight_ln = nn.Linear(self.dim_spk, self.dim_last, bias=False)
      self.bias_ln = nn.Linear(self.dim_spk, self.dim_last, bias=False)
      self.weight_ln.weight, self.bias_ln.weight = (Tensor.ones(*self.weight_ln.weight.shape)), (Tensor.zeros(*self.bias_ln.weight.shape))
  def __call__(self, x: Tensor, spk_emb: Tensor):
    axis = tuple(-1-i for i in range(len(x.shape[1:])))
    x = x.layernorm(axis=axis, eps=self.eps)
    if not self.elementwise_affine: return x
    weights, bias = self.weight_ln(spk_emb), self.bias_ln(spk_emb)
    return weights * x + bias

# TODO: not actually used in our case. maybe implement later
class GroupNormMasked:
  def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
    self.num_groups, self.num_channels, self.eps, self.affine = num_groups, num_channels, eps, affine
    self.weight, self.bias = (Tensor.ones(num_channels)), (Tensor.zeros(num_channels))  if self.affine else (None, None)
  def __call__(self, x, mask):
    raise NotImplementedError

# TODO: depthwise conv (optional) #self.use_depthwise_conv = use_depthwise_conv
# TODO: f0 decoder (optional for infer) #self.use_automatic_f0_prediction = use_automatic_f0_prediction
# TODO: flow_share_parameter
class Synthesizer:
  def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, ssl_dim, n_speakers, sampling_rate=44100,
               vol_embedding=False,
               #use_depthwise_conv=False,
               #use_automatic_f0_prediction=True,
               #flow_share_parameter=False,
               n_flow_layer=4,
               **kwargs):
    self.spec_channels, self.inter_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.segment_size, self.n_speakers, self.gin_channels, self.vol_embedding = spec_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, segment_size, n_speakers, gin_channels, vol_embedding
    self.emb_g = nn.Embedding(n_speakers, gin_channels)
    if vol_embedding:
      self.emb_vol = nn.Linear(1, hidden_channels)
    self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)
    self.enc_p = TextEncoder(inter_channels, hidden_channels, kernel_size, n_layers, filter_channels=filter_channels, n_heads=n_heads, p_dropout=p_dropout)
    self.dec = Generator(sampling_rate, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels)
    self.emb_uv = nn.Embedding(vocab_size=2, embed_size=hidden_channels)
    self.character_mix = False
  def EnableCharacterMix(self, n_speakers_map, device):
    self.speaker_map = Tensor.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
    for i in range(n_speakers_map):
      self.speaker_map[i] = self.emb_g(Tensor([[i]], dtype=dtypes.int64).to(device))
    self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
    self.character_mix = True
  def infer(self, c, f0, uv, g=None, noise_scale=0.35, seed=52468, vol=None):
    Tensor.manual_seed(getenv('SEED', seed))
    c_lengths = (Tensor.ones([c.shape[0]]) * c.shape[-1]).to(c.device)
    if self.character_mix and g.shape[0] > 1:  # [N, S]  *  [S, B, 1, H]
      g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
      g = g * self.speaker_map  # [N, S, B, 1, H]
      g = g.sum(dim=1) # [N, 1, B, 1, H]
      g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
    else:
      if g.dim() == 1:
        g = g.unsqueeze(0)
      g = self.emb_g(g).transpose(1, 2)
    x_mask = sequence_mask(c_lengths, c.shape[2]).unsqueeze(1).cast(c.dtype)
    vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0
    x = self.pre(c) * x_mask + self.emb_uv(uv.cast(dtypes.int64)).transpose(1, 2) + vol
    z_p, _, _, c_mask = self.enc_p.forward(x, x_mask, f0=f0_to_coarse(f0), noise_scale=noise_scale)
    z = self.flow.forward(z_p, c_mask, g=g, reverse=True)
    o = self.dec.forward(z * c_mask, g=g, f0=f0)
    return o,f0
  @classmethod
  def load_model(cls, config_path:str, config_url:str, weights_path:str, weights_url:str):
    download_if_not_present(config_path, config_url)
    hps = get_hparams_from_file(config_path)
    download_if_not_present(weights_path, weights_url)
    net_g = cls(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model)
    _ = load_checkpoint(weights_path, net_g, None, skip_list=["f0_decoder"])
    logging.debug(f"{cls.__name__}:Loaded model with hps: {hps}")
    return net_g

def randn_like(x): return Tensor.randn(*x.shape, dtype=x.dtype).to(device=x.device)

class TextEncoder:
  def __init__(self, out_channels, hidden_channels, kernel_size, n_layers, gin_channels=0, filter_channels=None, n_heads=None, p_dropout=None):
    self.out_channels, self.hidden_channels, self.kernel_size, self.n_layers, self.gin_channels = out_channels, hidden_channels, kernel_size, n_layers, gin_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.f0_emb = nn.Embedding(256, hidden_channels)  # n_vocab = 256
    self.enc_ = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
  def forward(self, x, x_mask, f0=None, noise_scale=1):
    x = x + self.f0_emb(f0).transpose(1, 2)
    x = self.enc_.forward(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = split(stats, self.out_channels, dim=1)
    z = (m + randn_like(m) * logs.exp() * noise_scale) * x_mask
    return z, m, logs, x_mask

# TODO: this is tragic. remove this
import torch
class Upsample:
  def __init__(self, scale_factor):
    self.scale_factor=scale_factor
    self.torch_ups = torch.nn.Upsample(scale_factor=scale_factor)
  def forward(self, x):
    return Tensor(self.torch_ups(torch.from_numpy(x.numpy())).numpy()).to(x.device)

class SineGen():
  def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voice_threshold=0, flag_for_pulse=False):
    self.sine_amp, self.noise_std, self.harmonic_num, self.sampling_rate, self.voiced_threshold, self.flag_for_pulse = sine_amp, noise_std, harmonic_num, samp_rate, voice_threshold, flag_for_pulse
    self.dim = self.harmonic_num + 1
  def _f02uv(self, f0):
    return (f0 > self.voiced_threshold).cast(dtypes.float32)  #generate uv signal
  def _f02sine(self, f0_values):
    def padDiff(x : Tensor): return (x.pad2d((0,0,-1,1)) - x).pad2d((0,0,0,-1))
    def mod(x: Tensor, n: int) -> Tensor: return x - n * x.div(n).floor()  # TODO: this is what the % operator does in pytorch.
    rad_values = mod((f0_values / self.sampling_rate) , 1)  # convert to F0 in rad
    rand_ini = Tensor.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)  # initial phase noise
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    # TODO: optional flag_for_pulse. Not required for now.
    tmp_over_one = mod(rad_values.cumsum(1), 1)
    tmp_over_one_idx = padDiff(tmp_over_one) < 0
    cumsum_shift = Tensor.zeros_like(rad_values)
    cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
    sines = ((rad_values + cumsum_shift).cumsum(1) * 2 * np.pi).sin()
    return sines
  def forward(self, f0, upp=None):
    fn = f0.mul(Tensor([[range(1, self.harmonic_num + 2)]], dtype=dtypes.float32).to(f0.device))
    sine_waves = self._f02sine(fn) * self.sine_amp  #generate sine waveforms
    uv = self._f02uv(f0)  # generate uv signal
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    noise = noise_amp * randn_like(sine_waves)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise

# TODO: verify tanh
class SourceHnNSF:
  def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
    self.sine_amp, self.noise_std = sine_amp, add_noise_std
    self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
    self.l_linear = nn.Linear(harmonic_num + 1, 1)
  def forward(self, x, upp=None):
    sine_waves, uv, _ = self.l_sin_gen.forward(x, upp)
    sine_merge = self.l_linear(sine_waves.cast(self.l_linear.weight.dtype)).tanh()
    noise = randn_like(uv) * self.sine_amp / 3
    return sine_merge, noise, uv

# TODO: most of the hifigan in standard vits is reused here
class Generator:
  def __init__(self, sampling_rate, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels):
    self.sampling_rate, self.inter_channels, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.gin_channels = sampling_rate, inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels
    self.num_kernels, self.num_upsamples = len(resblock_kernel_sizes), len(upsample_rates)
    self.conv_pre = nn.Conv1d(inter_channels, upsample_initial_channel, 7, 1, padding=3)
    self.f0_upsamp = Upsample(scale_factor=np.prod(upsample_rates))
    self.m_source = SourceHnNSF(sampling_rate, harmonic_num=8)
    resblock = ResBlock1 if resblock == '1' else ResBlock2
    self.ups, self.noise_convs = [], []
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
      c_cur = upsample_initial_channel//(2**(i+1))
      self.ups.append(nn.ConvTranspose1d(upsample_initial_channel//(2**i), c_cur, k, u, padding=(k-u)//2))
      if i + 1 < len(upsample_rates):  # TODO: make oneliner after debugging
        stride_f0 = int(np.prod(upsample_rates[i + 1:]))
        self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
      else:
        self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
    self.resblocks = []
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))
    self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
    if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    self.upp = np.prod(upsample_rates)
  def forward(self, x, f0, g=None):
    f0 = self.f0_upsamp.forward(f0[:, None]).transpose(1, 2)  # bs,n,t
    har_source, _, _ = self.m_source.forward(f0, self.upp)
    har_source = har_source.transpose(1, 2)
    x = self.conv_pre(x)
    if g is not None:  x = x + self.cond(g)
    for i in range(self.num_upsamples):
      x, xs = self.ups[i](x.leakyrelu(LRELU_SLOPE)), None
      x_source = self.noise_convs[i](har_source)
      x = x + x_source
      for j in range(self.num_kernels):
        if xs is None: xs = self.resblocks[i * self.num_kernels + j].forward(x)
        else: xs += self.resblocks[i * self.num_kernels + j].forward(x)
      x = xs / self.num_kernels
    return self.conv_post(x.leakyrelu()).tanh()

# TODO: this is ugly
def f0_to_coarse(f0 : Tensor):
  f0_mel = 1127 * (1 + f0 / 700).log()
  a = (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN)
  b = F0_MEL_MIN * a - 1.
  f0_mel = (f0_mel > 0).where(f0_mel * a - b, f0_mel)
  # TODO: use round() instead of ceil()
  f0_coarse = f0_mel.ceil().cast(dtype=dtypes.int64) # TODO: make oneliner after debugging
  f0_coarse = f0_coarse * (f0_coarse > 0)
  f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
  f0_coarse = f0_coarse * (f0_coarse < F0_BIN)
  f0_coarse = f0_coarse + ((f0_coarse >= F0_BIN) * (F0_BIN - 1))
  return f0_coarse

"""
TODO: optional F0 decoder
class FFT():
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.proximal_bias, self.proximal_init = hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, proximal_bias, proximal_init
    #self.drop = nn.Dropout()
    #self.self_attn_layers = nn.
    self.attn_layers, self.norm_layers_0, self.ffn_layers, self.norm_layers_1 = []
    for _ in range(self.n_layers):
      self.attn_layers.append(
          MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init)
        )

class F0Decoder():
  def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels=0):
    self.out_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.spk_channels = out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels
    self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
    self.decoder = None
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
    self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)
"""

def load_contentvec(model) -> ContentVec:
  weights_path = model[0]
  download_if_not_present(weights_path, model[1])
  cfg = load_config_from_checkpoint(model[0])
  enc = ContentVec(cfg.model)
  _ = load_checkpoint_enc(weights_path, enc, None)
  logging.debug(f"Loaded model with cfg={cfg}")
  return enc

def load_config_from_checkpoint(checkpoint_path):
  assert os.path.isfile(checkpoint_path)
  state = torch_load(checkpoint_path)
  cfg = None
  learning_rate = state["cfg"]["lr_scheduler"]["lr"][0]
  epoch = state["extra_state"]["train_iterator"]["epoch"]
  if "cfg" in state and state["cfg"] is not None:
    cfg = state["cfg"]
  elif "args" in state and state["args"] is not None:
    raise NotImplementedError  # TODO: not required for the checkpoint files yet encountered
  else:
    raise RuntimeError(f"Neither args nor cfg exist in state keys = {state.keys()}")
  return HParams(**cfg)

def load_checkpoint_enc(checkpoint_path, model: ContentVec, optimizer=None, skip_list=[]):
  assert os.path.isfile(checkpoint_path)
  start_time = time.time()
  checkpoint_dict = torch_load(checkpoint_path)
  saved_state_dict = checkpoint_dict['model']
  weight_g, weight_v, parent = None, None, None
  for key, v in saved_state_dict.items():
    if any(layer in key for layer in skip_list): continue
    try:
      obj, skip = model, False
      for k in key.split('.'):
        if k.isnumeric(): obj = obj[int(k)]
        elif isinstance(obj, dict): obj = obj[k]
        else:
          if k in ["weight_g", "weight_v"]:
            parent, skip = obj, True
            if k == "weight_g": weight_g = v
            else: weight_v = v
          if not skip: obj = getattr(obj, k)
      if weight_g and weight_v:
        setattr(obj, "weight_g", weight_g.numpy())
        setattr(obj, "weight_v", weight_v.numpy())
        obj, v = getattr(parent, "weight"), weight_norm(weight_v, weight_g, 0)
        weight_g, weight_v, parent, skip = None, None, None, False
      if not skip and obj.shape == v.shape: obj.assign(v.to(obj.device))
      elif not skip: logging.error(f"MISMATCH SHAPE IN {key}, {obj.shape} {v.shape}")
    except Exception as e: raise e
  logging.info(f"Loaded checkpoint '{checkpoint_path}' in {time.time() - start_time:.4f}s")
  return model, optimizer

SO_VITS_SVC_PATH = Path(__file__).parent.parent / "weights/So-VITS-SVC"
VITS_MODELS = { # config_path, weights_path, config_url, weights_url
  "saul_goodman" : (SO_VITS_SVC_PATH / "config_saul_gman.json", SO_VITS_SVC_PATH / "pretrained_saul_gman.pth", "https://huggingface.co/Amo/so-vits-svc-4.0_GA/resolve/main/ModelsFolder/Saul_Goodman_80000/config.json", "https://huggingface.co/Amo/so-vits-svc-4.0_GA/resolve/main/ModelsFolder/Saul_Goodman_80000/G_80000.pth")
}
ENCODER_MODELS = { # weights_path, weights_url
  "contentvec": (SO_VITS_SVC_PATH / "contentvec_checkpoint.pt", "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
}

if __name__=="__main__":
  logging.basicConfig(stream=sys.stdout, level=(logging.INFO if DEBUG < 1 else logging.DEBUG))

  encoder_model = "contentvec"
  vits_model = "saul_goodman"
  encoder_location = ENCODER_MODELS[encoder_model]
  vits_location = VITS_MODELS[vits_model]

  Tensor.no_grad = True
  Tensor.training = False

  # 1. ContentVec
  contentvec = ContentVec.load_model(encoder_location[0], encoder_location[1])

  # 2. Synthesizer
  net_g = Synthesizer.load_model(config_path=vits_location[0], config_url=vits_location[2], weights_path=vits_location[1], weights_url=vits_location[3])
