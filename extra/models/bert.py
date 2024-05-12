from tinygrad.tensor import Tensor
from tinygrad import nn, dtypes
from tinygrad.helpers import fetch, get_child
from pathlib import Path

from examples.mlperf.initializers import LinearBert, LayerNormBert

# allow for monkeypatching
Embedding = nn.Embedding
Linear = nn.Linear
LayerNorm = nn.LayerNorm

class BertForQuestionAnswering:
  def __init__(self, hidden_size=1024, intermediate_size=4096, max_position_embeddings=512, num_attention_heads=16, num_hidden_layers=24, type_vocab_size=2, vocab_size=30522, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
    self.bert = Bert(hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob)
    self.qa_outputs = Linear(hidden_size, 2)

  def load_from_pretrained(self):
    fn = Path(__file__).parents[1] / "weights/bert_for_qa.pt"
    fetch("https://zenodo.org/record/3733896/files/model.pytorch?download=1", fn)
    fn_vocab = Path(__file__).parents[1] / "weights/bert_vocab.txt"
    fetch("https://zenodo.org/record/3733896/files/vocab.txt?download=1", fn_vocab)

    import torch
    with open(fn, "rb") as f:
      state_dict = torch.load(f, map_location="cpu")

    for k, v in state_dict.items():
      if "dropout" in k: continue # skip dropout
      if "pooler" in k: continue # skip pooler
      get_child(self, k).assign(v.numpy()).realize()

  def __call__(self, input_ids:Tensor, attention_mask:Tensor, token_type_ids:Tensor):
    sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.chunk(2, dim=-1)
    start_logits = start_logits.reshape(-1, 1)
    end_logits = end_logits.reshape(-1, 1)

    return Tensor.stack([start_logits, end_logits])

class BertForMLPerf:
  def __init__(self, hidden_size:int, intermediate_size:int, max_position_embeddings:int, num_attention_heads:int, num_hidden_layers:int, type_vocab_size:int, vocab_size:int, attention_probs_dropout_prob:float, hidden_dropout_prob:float) -> None:
    self.model = Bert(hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob)
    # for clsf:
    self.clsf_pooler = LinearBert(hidden_size, hidden_size) # [bs, seq, hidden] -> [bs, hidden]
    self.clsf_pooling_activation = Tensor.tanh
    self.clsf_output = LinearBert(hidden_size, 2) # [bs, hidden] -> [bs, 2]

    # for lm:
    self.lm_transform = LinearBert(hidden_size, hidden_size)
    self.lm_transform_activation = gelu
    self.lm_norm = LayerNormBert(hidden_size, eps=1e-12)
    self.lm_output = LinearBert(hidden_size, vocab_size, bias=False) # [bs, seq, hidden] -> [bs, seq, vocab]
    self.lm_output.weight = self.model.embeddings.word_embeddings.weight
    self.lm_output_bias = Tensor.zeros(vocab_size, dtype=dtypes.float32)

  def __call__(self, input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor):
    output = self.model(input_ids, attention_mask, segment_ids)
    clsf_logits = self.clsf_output(self.clsf_pooling_activation(self.clsf_pooler(output[:, 0]))).cast(dtypes.float32)

    # gather only the masked_positions we care about
    counter = Tensor.arange(output.shape[1], requires_grad=False, device=output.device).reshape(1, 1, output.shape[1]).expand(*masked_positions.shape, output.shape[1])
    onehot = counter == masked_positions.unsqueeze(2).expand(*masked_positions.shape, output.shape[1])
    h_masked = onehot @ output

    h_masked = self.lm_norm(self.lm_transform_activation(self.lm_transform(h_masked)))
    lm_logits = self.lm_output(h_masked) + self.lm_output_bias

    return lm_logits, clsf_logits

class Bert:
  def __init__(self, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob):
    self.embeddings = BertEmbeddings(hidden_size, max_position_embeddings, type_vocab_size, vocab_size, hidden_dropout_prob)
    self.encoder = BertEncoder(hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob)

  def __call__(self, input_ids, attention_mask, token_type_ids):
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    embedding_output = self.embeddings(input_ids, token_type_ids)
    encoder_outputs = self.encoder(embedding_output, extended_attention_mask)

    return encoder_outputs

class BertEmbeddings:
  def __init__(self, hidden_size, max_position_embeddings, type_vocab_size, vocab_size,  hidden_dropout_prob):
    self.word_embeddings = Embedding(vocab_size, hidden_size)
    self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
    self.token_type_embeddings = Embedding(type_vocab_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, input_ids, token_type_ids):
    input_shape = input_ids.shape
    seq_length = input_shape[1]

    position_ids = Tensor.arange(seq_length, requires_grad=False, device=input_ids.device).unsqueeze(0).expand(*input_shape)
    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = embeddings.dropout(self.dropout)
    return embeddings

class BertEncoder:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, attention_probs_dropout_prob, hidden_dropout_prob):
    self.layer = [BertLayer(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob) for _ in range(num_hidden_layers)]

  def __call__(self, hidden_states, attention_mask):
    for layer in self.layer:
      hidden_states = layer(hidden_states, attention_mask)
    return hidden_states

class BertLayer:
  def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
    self.intermediate = BertIntermediate(hidden_size, intermediate_size)
    self.output = BertOutput(hidden_size, intermediate_size, hidden_dropout_prob)

  def __call__(self, hidden_states, attention_mask):
    attention_output = self.attention(hidden_states, attention_mask)
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output

class BertOutput:
  def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
    self.dense = Linear(intermediate_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = hidden_states.dropout(self.dropout)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

def gelu(x):
  return x * 0.5 * (1.0 + erf(x / 1.41421))

# approximation of the error function
def erf(x):
  t = (1 + 0.3275911 * x.abs()).reciprocal()
  return x.sign() * (1 - ((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t + 0.254829592) * t * (-(x.square())).exp())

class BertIntermediate:
  def __init__(self, hidden_size, intermediate_size):
    self.dense = Linear(hidden_size, intermediate_size)

  def __call__(self, hidden_states):
    x = self.dense(hidden_states)
    # tinygrad gelu is openai gelu but we need the original bert gelu
    return gelu(x)

class BertAttention:
  def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
    self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

  def __call__(self, hidden_states, attention_mask):
    self_output = self.self(hidden_states, attention_mask)
    attention_output = self.output(self_output, hidden_states)
    return attention_output

class BertSelfAttention:
  def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
    self.num_attention_heads = num_attention_heads
    self.attention_head_size = int(hidden_size / num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = Linear(hidden_size, self.all_head_size)
    self.key = Linear(hidden_size, self.all_head_size)
    self.value = Linear(hidden_size, self.all_head_size)

    self.dropout = attention_probs_dropout_prob

  def __call__(self, hidden_states, attention_mask):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    context_layer = Tensor.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, self.dropout)

    context_layer = context_layer.transpose(1, 2)
    context_layer = context_layer.reshape(context_layer.shape[0], context_layer.shape[1], self.all_head_size)

    return context_layer

  def transpose_for_scores(self, x):
    x = x.reshape(x.shape[0], x.shape[1], self.num_attention_heads, self.attention_head_size)
    return x.transpose(1, 2)

class BertSelfOutput:
  def __init__(self, hidden_size, hidden_dropout_prob):
    self.dense = Linear(hidden_size, hidden_size)
    self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
    self.dropout = hidden_dropout_prob

  def __call__(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = hidden_states.dropout(self.dropout)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states
