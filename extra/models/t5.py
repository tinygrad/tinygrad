# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tinygrad T5 model."""

import copy
import math

from tinygrad import nn, Tensor, dtypes


class T5Config:
  def __init__(
      self,
      d_ff=1024,
      d_kv=64,
      d_model=512,
      layer_norm_epsilon=1e-6,
      num_decoder_layers=8,
      num_heads=6,
      num_layers=8,
      relative_attention_num_buckets=32,
      relative_attention_max_distance=128,
      vocab_size=32128,
  ):
    self.d_ff = d_ff
    self.d_kv = d_kv
    self.d_model = d_model
    self.layer_norm_epsilon = layer_norm_epsilon
    self.num_decoder_layers = num_decoder_layers
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.relative_attention_max_distance = relative_attention_max_distance
    self.vocab_size = vocab_size


class NewGELUActivation:
  """
  Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
  the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
  """

  def __call__(self, x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + Tensor.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * Tensor.pow(x, 3.0))))

class T5LayerNorm:
  def __init__(self, hidden_size, eps=1e-6):
    """
    Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
    """
    self.weight = Tensor.ones(hidden_size)
    self.variance_epsilon = eps

  def __call__(self, hidden_states):
    # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
    # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    # half-precision inputs is done in fp32

    variance = hidden_states.cast(dtypes.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * Tensor.rsqrt(variance + self.variance_epsilon)

    # convert into half-precision if necessary
    if self.weight.dtype in [dtypes.float16, dtypes.bfloat16]:
      hidden_states = hidden_states.cast(self.weight.dtype)

    return self.weight * hidden_states


class T5DenseGatedActDense:
  def __init__(self, config: T5Config):
    self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
    self.act = NewGELUActivation()

  def __call__(self, hidden_states):
    hidden_gelu = self.act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.wo(hidden_states)
    return hidden_states


class T5LayerFF:
  def __init__(self, config: T5Config):
    self.DenseReluDense = T5DenseGatedActDense(config)
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  def __call__(self, hidden_states):
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    hidden_states = hidden_states + forwarded_states
    return hidden_states


class T5Attention:
  def __init__(self, config: T5Config, has_relative_attention_bias=False):
    self.has_relative_attention_bias = has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = config.relative_attention_max_distance
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    # Mesh TensorFlow initialization to avoid scaling before softmax
    self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    if self.has_relative_attention_bias:
      self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

  @staticmethod
  def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
      num_buckets //= 2
      relative_buckets += (relative_position > 0).cast(dtypes.long) * num_buckets
      relative_position = Tensor.abs(relative_position)
    else:
      relative_position = -Tensor.min(relative_position, Tensor.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        Tensor.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).cast(dtypes.long)

    relative_position_if_large = Tensor.min(
        Tensor.stack(
            relative_position_if_large, Tensor.full_like(relative_position_if_large, num_buckets - 1)
        ),
        axis=0,
    )
    relative_buckets += Tensor.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets

  def compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
      device = self.relative_attention_bias.weight.device
    context_position = Tensor.arange(query_length, dtype=dtypes.long, device=device)[:, None]
    memory_position = Tensor.arange(key_length, dtype=dtypes.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=True,
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(
        relative_position_bucket
    )  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values

  def __call__(
      self,
      hidden_states,
      key_value_states=None,
      position_bias=None,
  ):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
      """projection"""
      return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
      """reshape"""
      return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer):
      """projects hidden states correctly to key/query states"""
      # self-attn
      # (batch_size, n_heads, seq_length, dim_per_head)
      hidden_states = shape(proj_layer(hidden_states))

      return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(hidden_states, self.k)
    value_states = project(hidden_states, self.v)

    # compute scores
    scores = Tensor.matmul(query_states, key_states.transpose(3, 2))
    # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
      position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

    scores += position_bias
    attn_weights = Tensor.softmax(scores.float(), axis=-1).cast(
        scores.dtype
    )  # (batch_size, n_heads, seq_length, key_length)

    attn_output = unshape(Tensor.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    outputs = (attn_output,) + (None,) + (position_bias,)

    return outputs


class T5LayerSelfAttention:
  def __init__(self, config, has_relative_attention_bias=False):
    self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  def __call__(
      self,
      hidden_states,
      position_bias=None,
  ):
    normed_hidden_states = self.layer_norm(hidden_states)
    attention_output = self.SelfAttention(
        normed_hidden_states,
        position_bias=position_bias,
    )
    hidden_states = hidden_states + attention_output[0]
    outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
    return outputs


class T5Block:
  def __init__(self, config, has_relative_attention_bias=False):
    self.layer = []
    self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
    self.layer.append(T5LayerFF(config))

  def __call__(
      self,
      hidden_states,
      position_bias=None,
  ):
    self_attention_outputs = self.layer[0](
        hidden_states,
        position_bias=position_bias,
    )
    hidden_states = self_attention_outputs[0]
    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)

    outputs = (hidden_states,)

    outputs = outputs + attention_outputs

    return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights) #noqa:E501


class T5Stack:
  def __init__(self, config, embed_tokens=None):
    self.config = config
    self.embed_tokens = embed_tokens

    self.block = [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
    self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, new_embeddings):
    self.embed_tokens = new_embeddings

  def __call__(
      self,
      input_ids=None,
  ):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embed_tokens(input_ids)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.

    # Prepare head mask if needed
    position_bias = None

    hidden_states = inputs_embeds

    for i, layer_module in enumerate(self.block):
      layer_outputs = layer_module(
          hidden_states,
          position_bias=position_bias,
      )

      # layer_outputs is a tuple with:
      # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights) #noqa:E501
      hidden_states, position_bias = layer_outputs[0], layer_outputs[1]

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = hidden_states

    return {"last_hidden_states": hidden_states}


class T5EncoderModel:
  def __init__(self, config: T5Config):
    self.config = config
    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    self.encoder = T5Stack(encoder_config, self.shared)

  ### TODO: typing
  def __call__(
      self,
      input_ids=None,
      attention_mask=None,
  ):
    r"""
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, T5EncoderModel

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    >>> model = T5EncoderModel.from_pretrained("google-t5/t5-small")
    >>> input_ids = tokenizer(
    ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ... ).input_ids  # Batch size 1
    >>> outputs = model(input_ids=input_ids)
    >>> last_hidden_states = outputs.last_hidden_state
    ```"""

    return self.encoder(input_ids=input_ids)